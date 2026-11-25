use troupekit_agent::{AgentError, PersonaError};
use troupekit_core::error::CoreError;
use troupekit_core::event::PauseReason;
use troupekit_core::ids::RunId;
use troupekit_graph::{GraphError, WorkflowError};
use troupekit_providers::CatalogError;
use troupekit_tools::ToolError;

/// Every failure of the kit: builder settings, catalog and key problems,
/// and the wrapped engine errors, plus run outcomes that are not an answer.
/// Messages name the agent and the fix; they never hold an API key.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KitError {
    #[error("agent `{agent}`: {message}")]
    Config { agent: String, message: String },
    #[error("vendor `{vendor}` needs an API key: set {env} or call .api_key(..)")]
    MissingKey { vendor: String, env: String },
    #[error("agent `{agent}`: {source}; {hint}")]
    Catalog {
        agent: String,
        source: CatalogError,
        hint: String,
    },
    #[error(transparent)]
    Persona(#[from] PersonaError),
    #[error(transparent)]
    Agent(#[from] AgentError),
    #[error(transparent)]
    Tool(#[from] ToolError),
    #[error(transparent)]
    Workflow(#[from] WorkflowError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error(transparent)]
    Store(#[from] CoreError),
    #[error("run {run_id} failed: {reason}")]
    Failed { run_id: RunId, reason: String },
    #[error("run {run_id} paused: {reason:?}")]
    Paused { run_id: RunId, reason: PauseReason },
    #[error("run {run_id}: gave up after {attempts} retries: {last_error}")]
    Retries {
        run_id: RunId,
        attempts: u32,
        last_error: String,
    },
}

impl KitError {
    pub(crate) fn config(agent: &str, message: impl Into<String>) -> Self {
        KitError::Config {
            agent: agent.to_string(),
            message: message.into(),
        }
    }
}
