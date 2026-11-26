/// Every failure of an on-device model: load, tokenize, decode, context
/// size, grammar, unsupported operations, the model cache and catalog.
/// Messages name the cause and never hold prompt or reply text.
/// `LocalChat` maps these to `ProviderErrorKind::Server` with this text.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum LocalError {
    #[error("local model load failed: {0}")]
    Load(String),
    #[error("local tokenize failed: {0}")]
    Tokenize(String),
    #[error("local decode failed: {0}")]
    Decode(String),
    #[error("local context full: {needed} tokens needed, limit {limit}")]
    ContextFull { needed: usize, limit: usize },
    #[error("local grammar refused: {0}")]
    Grammar(String),
    #[error("local model does not support {0}")]
    Unsupported(&'static str),
    #[error("local model cache: {0}")]
    Cache(String),
    #[error("local model catalog: {0}")]
    Catalog(String),
}
