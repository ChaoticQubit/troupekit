use crate::condition::Conditions;
use crate::error::GraphError;
use crate::flow::Flow;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use troupekit_agent::budget::UsageLedger;
use troupekit_agent::eval::EvalRegistry;
use troupekit_agent::{AgentError, AgentIo, EventSink, ModelHandle, Persona, PersonaLoader};
use troupekit_core::clock::Clock;
use troupekit_core::event::{Event, EventBody};
use troupekit_core::ids::RunId;
use troupekit_core::store::CheckpointStore;
use troupekit_tools::ToolRegistry;

/// Everything live a graph run needs, supplied by the host on every step:
/// the `AgentIo` its agent and tool nodes use, persona loading by name
/// and the router conditions.
#[async_trait]
pub trait GraphIo: Send + Sync {
    fn agent(&self) -> &dyn AgentIo;
    async fn persona(&self, name: &str) -> Result<Persona, AgentError>;
    fn conditions(&self) -> &Conditions;

    /// Returns the agent io as a shared handle, which agent nodes with
    /// `delegates` need to run member agents as tools. The default `None`
    /// makes such a node fail at `GraphRun::start`.
    fn agent_arc(&self) -> Option<Arc<dyn AgentIo>> {
        None
    }

    /// Returns the store member agent runs checkpoint to, so a delegated
    /// call resumes after a restart. The default `None` runs members
    /// without checkpoints.
    fn checkpoints(&self) -> Option<Arc<dyn CheckpointStore>> {
        None
    }

    /// Decides whether agent node `node` may run `persona` on model
    /// `selector`, a flow's per-node `model` override. A host whose
    /// privacy depends on the handle (a private or on-device persona)
    /// returns `Err` with the reason for any handle that breaks it. The
    /// default allows every override.
    fn allow_model(&self, node: &str, persona: &Persona, selector: &str) -> Result<(), String> {
        let _ = (node, persona, selector);
        Ok(())
    }

    /// Loads flow `name` for a subflow node. The default has no flows and
    /// returns `Setup` "no flow source", so a flow with subflow nodes fails
    /// at `GraphRun::start` over such an io.
    async fn flow(&self, name: &str) -> Result<Flow, GraphError> {
        Err(GraphError::Setup(format!("no flow source for `{name}`")))
    }
}

/// A `GraphIo` built from plain fields: an agent io, a persona loader and
/// conditions. Good for tests, examples and single-user hosts.
/// `new` starts with the built-in conditions.
pub struct SimpleGraphIo {
    pub agent: Arc<dyn AgentIo>,
    pub personas: PersonaLoader,
    pub conditions: Conditions,
}

impl SimpleGraphIo {
    /// Builds an io over `agent` and `personas` with only the built-in
    /// `flag` condition; add more through the `conditions` field.
    /// Returns the io.
    pub fn new(agent: Arc<dyn AgentIo>, personas: PersonaLoader) -> Self {
        Self {
            agent,
            personas,
            conditions: Conditions::new(),
        }
    }
}

#[async_trait]
impl GraphIo for SimpleGraphIo {
    fn agent(&self) -> &dyn AgentIo {
        self.agent.as_ref()
    }

    async fn persona(&self, name: &str) -> Result<Persona, AgentError> {
        self.personas.load(name).await
    }

    fn conditions(&self) -> &Conditions {
        &self.conditions
    }

    fn agent_arc(&self) -> Option<Arc<dyn AgentIo>> {
        Some(self.agent.clone())
    }
}

/// Numbers and sends the events of one graph step under the graph run id.
/// Branches emit concurrently, so the next `seq` is an atomic seeded from
/// the run and read back when the step ends.
pub(crate) struct Emitter<'a> {
    pub run_id: RunId,
    pub seq: &'a AtomicU64,
    pub io: &'a dyn AgentIo,
}

impl Emitter<'_> {
    /// Sends `body` as the run's next event, stamped with the io clock.
    /// Takes the next `seq` atomically, so concurrent branches never
    /// share or skip a number.
    pub fn emit(&self, body: EventBody) {
        self.io.events().emit(Event {
            run_id: self.run_id,
            seq: self.seq.fetch_add(1, Ordering::SeqCst),
            at_ms: self.io.clock().now_ms(),
            body,
        });
    }
}

struct NodeSink<'a> {
    emitter: &'a Emitter<'a>,
    node: &'a str,
    persona: &'a str,
}

impl EventSink for NodeSink<'_> {
    fn emit(&self, event: Event) {
        let body = match event.body {
            EventBody::RunStarted { .. }
            | EventBody::RunFinished { .. }
            | EventBody::RunPaused { .. }
            | EventBody::RunFailed { .. } => return,
            EventBody::StepStarted { step, .. } => EventBody::StepStarted {
                step: format!("{}/{step}", self.node),
                persona: Some(self.persona.to_string()),
            },
            EventBody::StepFinished { step, usage } => EventBody::StepFinished {
                step: format!("{}/{step}", self.node),
                usage,
            },
            other => other,
        };
        self.emitter.emit(body);
    }
}

/// The `AgentIo` an agent node steps with: the graph's agent io, except
/// that events go out under the graph run id and `seq`, step names gain a
/// `node/` prefix, step starts name the node's persona slug and the
/// agent's own run lifecycle events are dropped.
pub(crate) struct NodeIo<'a> {
    inner: &'a dyn AgentIo,
    tools: Option<&'a ToolRegistry>,
    sink: NodeSink<'a>,
}

impl<'a> NodeIo<'a> {
    /// Builds the io for agent node `node` running persona slug `persona`,
    /// emitting through `emitter`. Everything else delegates to the
    /// emitter's agent io. Returns the io.
    pub fn new(emitter: &'a Emitter<'a>, node: &'a str, persona: &'a str) -> Self {
        Self {
            inner: emitter.io,
            tools: None,
            sink: NodeSink {
                emitter,
                node,
                persona,
            },
        }
    }
}

impl<'a> NodeIo<'a> {
    /// Serves `tools` instead of the agent io's registry when set: the
    /// per-node registry of an agent node with delegates. Returns the io.
    pub fn with_tools(mut self, tools: Option<&'a ToolRegistry>) -> Self {
        self.tools = tools;
        self
    }
}

/// The agent io an agent node with delegates starts its run over: the
/// graph's agent io with the node's own tool registry, so the run's setup
/// check sees the delegation tools.
pub(crate) struct ToolsIo<'a> {
    pub inner: &'a dyn AgentIo,
    pub tools: &'a ToolRegistry,
}

#[async_trait]
impl AgentIo for ToolsIo<'_> {
    async fn model(&self, selector: &str) -> Result<ModelHandle, AgentError> {
        self.inner.model(selector).await
    }

    fn tools(&self) -> &ToolRegistry {
        self.tools
    }

    fn evals(&self) -> &EvalRegistry {
        self.inner.evals()
    }

    fn ledger(&self) -> &dyn UsageLedger {
        self.inner.ledger()
    }

    fn events(&self) -> &dyn EventSink {
        self.inner.events()
    }

    fn clock(&self) -> &dyn Clock {
        self.inner.clock()
    }
}

#[async_trait]
impl AgentIo for NodeIo<'_> {
    async fn model(&self, selector: &str) -> Result<ModelHandle, AgentError> {
        self.inner.model(selector).await
    }

    fn tools(&self) -> &ToolRegistry {
        self.tools.unwrap_or_else(|| self.inner.tools())
    }

    fn evals(&self) -> &EvalRegistry {
        self.inner.evals()
    }

    fn ledger(&self) -> &dyn UsageLedger {
        self.inner.ledger()
    }

    fn events(&self) -> &dyn EventSink {
        &self.sink
    }

    fn clock(&self) -> &dyn Clock {
        self.inner.clock()
    }
}
