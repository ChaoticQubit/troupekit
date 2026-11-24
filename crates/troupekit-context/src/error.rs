use troupekit_core::error::CoreError;

/// Errors from memory, ingest and retrieval.
/// `Core` passes a store error through unchanged, `Invalid` is rejected input
/// and `Embed` is a failed or malformed embedding call.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ContextError {
    #[error(transparent)]
    Core(#[from] CoreError),
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("embedding failed: {0}")]
    Embed(String),
}
