/// Stable numeric code of every failure the neutral API reports.
/// The numbers are fixed forever: a binding may match on them, store
/// them or send them over the wire. New codes only ever get new numbers.
#[non_exhaustive]
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    Config = 1,
    MissingKey = 2,
    Catalog = 3,
    Persona = 4,
    Agent = 5,
    Tool = 6,
    Workflow = 7,
    Graph = 8,
    Store = 9,
    Context = 10,
    Failed = 11,
    Paused = 12,
    Retries = 13,
    Cancelled = 14,
    HostCallback = 15,
    InvalidArgument = 16,
    Unsupported = 17,
    Panic = 18,
    Internal = 99,
}

/// The one error type of the neutral API: a stable `code`, a `message`
/// that never holds a key or secret, and the `run_id` text when the
/// failure belongs to a run that can be resumed or inspected.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct FfiError {
    pub code: ErrorCode,
    pub message: String,
    pub run_id: Option<String>,
}

impl FfiError {
    /// Builds an error with `code` and `message` and no run id.
    /// Input: the code and a message free of secrets. Output: the error.
    /// The message must not be blank.
    pub fn new(code: ErrorCode, message: impl Into<String>) -> FfiError {
        let message = message.into();
        assert!(!message.trim().is_empty());
        FfiError {
            code,
            message,
            run_id: None,
        }
    }

    /// Builds the `Unsupported` error for `what`, a neutral method name.
    /// The 0.4 skeleton returns it from every body that is not filled in
    /// yet. Output: an error naming `what` and saying it is not built yet.
    pub(crate) fn unsupported(what: &str) -> FfiError {
        assert!(!what.is_empty());
        FfiError::new(ErrorCode::Unsupported, format!("{what} is not built yet"))
    }
}
