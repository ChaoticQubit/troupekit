use crate::condition::Conditions;
use async_trait::async_trait;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use troupekit_agent::budget::UsageLedger;
use troupekit_agent::eval::EvalRegistry;
use troupekit_agent::{AgentError, AgentIo, EventSink, ModelHandle, Persona, PersonaLoader};
use troupekit_core::clock::Clock;
use troupekit_core::event::{Event, EventBody};
use troupekit_core::ids::RunId;
use troupekit_tools::ToolRegistry;
use troupekit_trace::TraceSpec;

/// Everything live a graph run needs, supplied by the host on every step:
/// the `AgentIo` its agent and tool nodes use, persona loading by name
/// and the router conditions.
#[async_trait]
pub trait GraphIo: Send + Sync {
    fn agent(&self) -> &dyn AgentIo;
    async fn persona(&self, name: &str) -> Result<Persona, AgentError>;
    fn conditions(&self) -> &Conditions;

    /// Returns the trace settings of persona `name`, applied to the agent
    /// run of every node using it. The default is `None`, no tracing.
    /// Input: the persona name. Output: the spec.
    async fn persona_trace(&self, _name: &str) -> Option<TraceSpec> {
        None
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

    async fn persona_trace(&self, name: &str) -> Option<TraceSpec> {
        self.personas.trace_spec(name).await.ok().flatten()
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
    sink: NodeSink<'a>,
}

impl<'a> NodeIo<'a> {
    /// Builds the io for agent node `node` running persona slug `persona`,
    /// emitting through `emitter`. Everything else delegates to the
    /// emitter's agent io. Returns the io.
    pub fn new(emitter: &'a Emitter<'a>, node: &'a str, persona: &'a str) -> Self {
        Self {
            inner: emitter.io,
            sink: NodeSink {
                emitter,
                node,
                persona,
            },
        }
    }
}

#[async_trait]
impl AgentIo for NodeIo<'_> {
    async fn model(&self, selector: &str) -> Result<ModelHandle, AgentError> {
        self.inner.model(selector).await
    }

    fn tools(&self) -> &ToolRegistry {
        self.inner.tools()
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
