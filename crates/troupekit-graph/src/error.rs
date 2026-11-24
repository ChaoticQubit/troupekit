use troupekit_agent::AgentError;
use troupekit_core::error::CoreError;

/// A workflow file that failed validation.
/// Names the file, the field (dotted, such as `nodes.plan.persona` or
/// `edges[2].when`) and what is wrong with it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{file}: {field}: {message}")]
pub struct WorkflowError {
    pub file: String,
    pub field: String,
    pub message: String,
}

impl WorkflowError {
    /// Builds the error for `field` of `file` with `message`.
    /// Every workflow check reports through it.
    /// Returns the error.
    pub fn new(file: &str, field: &str, message: impl Into<String>) -> Self {
        Self {
            file: file.to_string(),
            field: field.to_string(),
            message: message.into(),
        }
    }
}

/// Why parsing a workflow, building a graph run, restoring one or applying
/// a human answer or new budget failed. Failures during a step are not
/// errors: they end the run in its `failed` status instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GraphError {
    #[error(transparent)]
    Workflow(#[from] WorkflowError),
    #[error("setup: {0}")]
    Setup(String),
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    #[error("resume: {0}")]
    Resume(String),
    #[error(transparent)]
    Agent(#[from] AgentError),
    #[error(transparent)]
    Store(#[from] CoreError),
}
