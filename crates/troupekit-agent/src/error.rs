use troupekit_core::error::CoreError;

/// A persona file that failed validation.
/// Names the file (its source label and path), the field (dotted, such as
/// `output.cases` or `evals[1]`) and what is wrong with it.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{file}: {field}: {message}")]
pub struct PersonaError {
    pub file: String,
    pub field: String,
    pub message: String,
}

impl PersonaError {
    /// Builds the error for `field` of `file` with `message`.
    /// Every persona check reports through it.
    /// Returns the error.
    pub fn new(file: &str, field: &str, message: impl Into<String>) -> Self {
        Self {
            file: file.to_string(),
            field: field.to_string(),
            message: message.into(),
        }
    }
}

/// Why loading a persona, building a run or resuming one failed.
/// Run failures during a step are not errors: they end the run in its
/// `Failed` state instead.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum AgentError {
    #[error(transparent)]
    Persona(#[from] PersonaError),
    #[error("persona not found: {0}")]
    NotFound(String),
    #[error("persona source: {0}")]
    Source(String),
    #[error("setup: {0}")]
    Setup(String),
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    #[error(transparent)]
    Store(#[from] CoreError),
}
