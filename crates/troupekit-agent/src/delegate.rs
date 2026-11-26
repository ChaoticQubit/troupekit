use crate::error::AgentError;
use crate::io::AgentIo;
use crate::persona::Persona;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;
use troupekit_core::store::CheckpointStore;
use troupekit_core::tool::ToolSpec;
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput};

const DEFAULT_MAX_DEPTH: u32 = 3;

/// Bounds and options for agents called as tools: the delegation depth,
/// a cap on each member run's spend, an optional store for member
/// checkpoints and whether `ask_question` is offered. `Default` holds the
/// safe values: depth 3, the member persona budget, no store, `ask_question` on.
#[non_exhaustive]
#[derive(Clone)]
pub struct DelegateSettings {
    max_depth: u32,
    max_run_usd_micros: Option<u64>,
    store: Option<Arc<dyn CheckpointStore>>,
    ask_question: bool,
}

impl Default for DelegateSettings {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            max_run_usd_micros: None,
            store: None,
            ask_question: true,
        }
    }
}

impl std::fmt::Debug for DelegateSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelegateSettings")
            .field("max_depth", &self.max_depth)
            .field("max_run_usd_micros", &self.max_run_usd_micros)
            .field("store", &self.store.is_some())
            .field("ask_question", &self.ask_question)
            .finish()
    }
}

impl DelegateSettings {
    /// Sets how deep agents may call agents; a call above it is a tool error.
    /// Returns the settings.
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Caps each member run at `micros` millionths of a US dollar, below
    /// the member persona's own budget when that is lower.
    /// Returns the settings.
    pub fn with_max_run_usd_micros(mut self, micros: u64) -> Self {
        self.max_run_usd_micros = Some(micros);
        self
    }

    /// Checkpoints member runs in `store`, so a re-run of the same call
    /// resumes the member instead of starting it again.
    /// Returns the settings.
    pub fn with_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Turns the `ask_question` delegation tool on or off.
    /// Returns the settings.
    pub fn with_ask_question(mut self, on: bool) -> Self {
        self.ask_question = on;
        self
    }

    /// Returns the delegation depth cap.
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Returns the member run spend cap, if set.
    pub fn max_run_usd_micros(&self) -> Option<u64> {
        self.max_run_usd_micros
    }

    /// Returns the member checkpoint store, if set.
    pub fn store(&self) -> Option<&Arc<dyn CheckpointStore>> {
        self.store.as_ref()
    }

    /// Returns whether the `ask_question` tool is offered.
    pub fn ask_question(&self) -> bool {
        self.ask_question
    }
}

/// An agent called as a tool: the model sends `{task, context?}`, the
/// persona runs to its end inside the call and its final text comes back
/// fenced as untrusted output with the member run id.
/// Not built yet: `call` returns a tool error.
pub struct AgentTool {
    spec: ToolSpec,
}

impl AgentTool {
    /// Builds tool `name` described by `description` that runs `persona`
    /// over `io` within `settings`.
    /// Returns the tool or an `AgentError` for a bad name or setup.
    pub fn new(
        name: &str,
        description: &str,
        persona: Persona,
        io: Arc<dyn AgentIo>,
        settings: DelegateSettings,
    ) -> Result<AgentTool, AgentError> {
        let _ = (persona, io, settings);
        let spec = ToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            schema: json!({
                "type": "object",
                "properties": {"task": {"type": "string"}, "context": {"type": "string"}},
                "required": ["task"],
                "additionalProperties": false
            }),
            hosted: None,
        };
        Ok(AgentTool { spec })
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let _ = (args, ctx);
        Err(ToolError::Failed("agent tools are not built yet".into()))
    }
}

/// The `delegate_work` and `ask_question` tools of a leader over a named
/// set of coworker agents, each listed as an enum value in the specs.
/// Not built yet: `tools` returns no tool.
pub struct DelegationTools {
    tools: Vec<Arc<dyn Tool>>,
}

impl DelegationTools {
    /// Builds the delegation tools over `coworkers`, each a name and its
    /// agent tool, within `settings`.
    /// Returns the tools or an `AgentError` for an empty or duplicate set.
    pub fn new(
        coworkers: Vec<(String, AgentTool)>,
        settings: DelegateSettings,
    ) -> Result<DelegationTools, AgentError> {
        let _ = (coworkers, settings);
        Ok(DelegationTools { tools: Vec::new() })
    }

    /// Returns the tools to register for the leader.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}
