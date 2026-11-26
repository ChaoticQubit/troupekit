use std::fmt;
use troupekit_ffi_core as neutral;

/// Stable numeric code of every failure, mirrored from the neutral `ErrorCode`.
/// Swift and Kotlin compare `TroupekitError.code` against these numbers.
/// The numbers are fixed forever and checked by `error_code_survives_conversion`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, uniffi::Enum)]
#[repr(u32)]
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

/// The one error Swift and Kotlin see: the neutral `FfiError` with its code
/// as a number, so a code added later still reaches the app unchanged.
/// Input: a neutral error. Output: `Failure { code, message, run_id }`.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Error)]
pub enum TroupekitError {
    Failure {
        code: u32,
        message: String,
        run_id: Option<String>,
    },
}

impl fmt::Display for TroupekitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let TroupekitError::Failure { code, message, .. } = self;
        write!(f, "troupekit error {code}: {message}")
    }
}

impl std::error::Error for TroupekitError {}

impl From<neutral::FfiError> for TroupekitError {
    fn from(error: neutral::FfiError) -> Self {
        TroupekitError::Failure {
            code: error.code as u32,
            message: error.message,
            run_id: error.run_id,
        }
    }
}

/// Error a Swift or Kotlin host callback returns; the neutral layer turns it
/// into a tool error output, a model failure or a transport failure.
/// Input: the foreign message. Output: the neutral `HostError` with that text.
#[derive(Clone, Debug, PartialEq, Eq, uniffi::Error)]
pub enum HostError {
    Failed { message: String },
}

impl fmt::Display for HostError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let HostError::Failed { message } = self;
        f.write_str(message)
    }
}

impl std::error::Error for HostError {}

impl From<uniffi::UnexpectedUniFFICallbackError> for HostError {
    fn from(error: uniffi::UnexpectedUniFFICallbackError) -> Self {
        HostError::Failed {
            message: error.reason,
        }
    }
}

impl From<HostError> for neutral::HostError {
    fn from(error: HostError) -> Self {
        let HostError::Failed { message } = error;
        neutral::HostError::new(message)
    }
}
