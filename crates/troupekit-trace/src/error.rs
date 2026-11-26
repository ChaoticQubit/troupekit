use std::error::Error;
use std::fmt;

const SHOWN_BYTES_MAX: usize = 64;

/// Errors from parsing trace settings or building a tracer.
/// Input: caller settings. Output: one variant naming what was wrong.
/// Later releases may add variants.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum TraceError {
    UnknownTrace(String),
    TooManyNames(usize),
    TooManySecrets(usize),
    MissingSink,
}

impl TraceError {
    pub(crate) fn unknown(text: &str) -> TraceError {
        TraceError::UnknownTrace(crate::value::cut_to(text, SHOWN_BYTES_MAX).to_owned())
    }
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TraceError::UnknownTrace(name) => write!(f, "unknown trace category `{name}`"),
            TraceError::TooManyNames(n) => write!(f, "{n} trace names, at most 64"),
            TraceError::TooManySecrets(n) => write!(f, "{n} scrubbed secrets, at most 1024"),
            TraceError::MissingSink => f.write_str("a tracer needs a sink"),
        }
    }
}

impl Error for TraceError {}
