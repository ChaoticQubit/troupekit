use serde::{Deserialize, Serialize};

/// Errors from core types and storage backends.
/// Covers bad input, optimistic-lock conflicts, vector dimension mismatches,
/// missing records and backend failures; each variant carries its detail.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CoreError {
    #[error("invalid message: {0}")]
    InvalidMessage(String),
    #[error("conflict: expected version {expected}, found {actual}")]
    Conflict { expected: u64, actual: u64 },
    #[error("vector dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("not found: {0}")]
    NotFound(String),
    #[error("store error: {0}")]
    Store(String),
}

/// Kind of a failed model or provider call.
/// The kind alone decides whether a retry middleware may repeat the call;
/// see `ProviderError::is_retryable`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProviderErrorKind {
    RateLimited,
    Auth,
    BadRequest,
    Server,
    Network,
    Stalled,
    Decode,
}

/// A failed provider call: its kind, a human-readable message and an
/// optional server-sent retry delay in milliseconds.
/// Displays as `Kind: message`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, thiserror::Error)]
#[error("{kind:?}: {message}")]
pub struct ProviderError {
    pub kind: ProviderErrorKind,
    pub message: String,
    pub retry_after_ms: Option<u64>,
}

impl ProviderError {
    /// Builds an error of `kind` with a human-readable `message`.
    /// The retry delay starts empty; add one with `with_retry_after`.
    /// Returns the new error.
    pub fn new(kind: ProviderErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into(), retry_after_ms: None }
    }

    /// Sets the retry delay to `ms` milliseconds, as sent by the provider.
    /// Consumes the error and returns it with the delay set,
    /// so it chains after `new`.
    pub fn with_retry_after(mut self, ms: u64) -> Self {
        self.retry_after_ms = Some(ms);
        self
    }

    /// Reports whether the failure is transient.
    /// True for rate limits, server errors, network errors and stalls;
    /// false for auth, bad request and decode errors.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self.kind,
            ProviderErrorKind::RateLimited | ProviderErrorKind::Server | ProviderErrorKind::Network | ProviderErrorKind::Stalled
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retryable_kinds() {
        for k in [ProviderErrorKind::RateLimited, ProviderErrorKind::Server, ProviderErrorKind::Network, ProviderErrorKind::Stalled] {
            assert!(ProviderError::new(k, "x").is_retryable());
        }
        for k in [ProviderErrorKind::Auth, ProviderErrorKind::BadRequest, ProviderErrorKind::Decode] {
            assert!(!ProviderError::new(k, "x").is_retryable());
        }
    }

    #[test]
    fn carries_retry_after() {
        let e = ProviderError::new(ProviderErrorKind::RateLimited, "slow down").with_retry_after(1500);
        assert_eq!(e.retry_after_ms, Some(1500));
        assert_eq!(e.to_string(), "RateLimited: slow down");
    }
}
