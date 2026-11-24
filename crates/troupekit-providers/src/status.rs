use serde_json::Value;
use troupekit_core::error::{ProviderError, ProviderErrorKind};

const MAX_DETAIL_CHARS: usize = 300;

/// Builds a `BadRequest` error with `message`: the request cannot be sent
/// as given. Used by request validation and every protocol encoder.
/// Returns the error.
pub fn bad_request(message: impl Into<String>) -> ProviderError {
    ProviderError::new(ProviderErrorKind::BadRequest, message)
}

/// Builds a `Decode` error with `message`: a reply did not have the shape
/// its protocol promises. Used by every decoder and the SSE parser.
/// Returns the error.
pub fn decode_error(message: impl Into<String>) -> ProviderError {
    ProviderError::new(ProviderErrorKind::Decode, message)
}

/// Maps an HTTP `status` to the kind of failure it means.
/// 401 and 403 are auth, 408 a stall, 429 a rate limit, other 4xx a bad
/// request, and every other status a server error.
pub fn kind_for_status(status: u16) -> ProviderErrorKind {
    match status {
        401 | 403 => ProviderErrorKind::Auth,
        408 => ProviderErrorKind::Stalled,
        429 => ProviderErrorKind::RateLimited,
        400..=499 => ProviderErrorKind::BadRequest,
        _ => ProviderErrorKind::Server,
    }
}

/// Reads the retry delay a provider sent, in milliseconds.
/// `retry-after-ms` wins over `retry-after`, which counts seconds and may
/// be fractional. Names match case-insensitively; dates give `None`.
pub fn retry_after_ms(headers: &[(String, String)]) -> Option<u64> {
    let find = |name: &str| {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.trim())
    };
    if let Some(ms) = find("retry-after-ms").and_then(|v| v.parse::<u64>().ok()) {
        return Some(ms);
    }
    let secs: f64 = find("retry-after")?.parse().ok()?;
    if !secs.is_finite() || secs < 0.0 {
        return None;
    }
    Some((secs * 1000.0).ceil().min(u64::MAX as f64) as u64)
}

/// Pulls a short readable message out of an error reply `body`.
/// Prefers JSON `error.message`, then `message`, then the raw text, cut
/// to 300 characters. Returns `"empty body"` when there is nothing.
pub fn error_detail(body: &[u8]) -> String {
    if let Ok(v) = serde_json::from_slice::<Value>(body) {
        if let Some(m) = v
            .pointer("/error/message")
            .or_else(|| v.get("message"))
            .and_then(Value::as_str)
        {
            return cut(m);
        }
    }
    let text = String::from_utf8_lossy(body);
    let text = text.trim();
    if text.is_empty() {
        "empty body".into()
    } else {
        cut(text)
    }
}

/// Builds the error for a non-success HTTP reply from its `status`,
/// `headers` and `body`: kind from the status, message from the body,
/// retry delay from the headers.
pub fn status_error(status: u16, headers: &[(String, String)], body: &[u8]) -> ProviderError {
    let err = ProviderError::new(
        kind_for_status(status),
        format!("HTTP {status}: {}", error_detail(body)),
    );
    match retry_after_ms(headers) {
        Some(ms) => err.with_retry_after(ms),
        None => err,
    }
}

fn cut(text: &str) -> String {
    text.chars().take(MAX_DETAIL_CHARS).collect()
}
