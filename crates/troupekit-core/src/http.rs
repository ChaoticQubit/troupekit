use crate::error::{ProviderError, ProviderErrorKind};
use async_trait::async_trait;
use serde::Deserialize;
use std::collections::VecDeque;
use std::fmt;

/// HTTP method of a request.
/// Deserializes from upper-case names such as `GET` and `POST`;
/// `as_str` gives the same upper-case name for the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum Method {
    Get,
    Post,
    Put,
    Patch,
    Delete,
}

impl Method {
    /// Returns the method's upper-case wire name, such as `GET`.
    /// Transports build their client's method from it.
    /// Never fails.
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
            Method::Put => "PUT",
            Method::Patch => "PATCH",
            Method::Delete => "DELETE",
        }
    }
}

/// One outgoing HTTP request for any transport.
/// `accept` lists the media types a 2xx answer may have (empty accepts any;
/// guarded transports enforce it). Debug prints header names only, so a
/// request holding an API key can be logged safely.
#[derive(Clone, PartialEq, Eq)]
pub struct HttpRequest {
    pub method: Method,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Option<Vec<u8>>,
    pub accept: Vec<String>,
}

impl HttpRequest {
    /// Builds a GET request for `url` with no headers, no body and any
    /// accepted media type.
    /// Returns the request for further field edits.
    pub fn get(url: &str) -> Self {
        Self {
            method: Method::Get,
            url: url.to_string(),
            headers: Vec::new(),
            body: None,
            accept: Vec::new(),
        }
    }

    /// Builds a POST request for `url` with `headers` and the `body` bytes.
    /// Any media type is accepted.
    /// Returns the request.
    pub fn post(url: impl Into<String>, headers: Vec<(String, String)>, body: Vec<u8>) -> Self {
        Self {
            method: Method::Post,
            url: url.into(),
            headers,
            body: Some(body),
            accept: Vec::new(),
        }
    }

    /// Looks up request header `name`, ignoring ASCII case.
    /// When a header repeats, the first value wins.
    /// Returns the value, or `None` when absent.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }
}

impl fmt::Debug for HttpRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let names: Vec<&str> = self.headers.iter().map(|(n, _)| n.as_str()).collect();
        f.debug_struct("HttpRequest")
            .field("method", &self.method)
            .field("url", &self.url)
            .field("headers", &names)
            .field("body_bytes", &self.body.as_ref().map(Vec::len))
            .field("accept", &self.accept)
            .finish()
    }
}

/// The body of an HTTP reply, read chunk by chunk so streams stay streams.
/// `chunk` yields the next bytes, then `None` once the body has ended.
/// Transports implement it over their own body type.
#[async_trait]
pub trait Body: Send {
    async fn chunk(&mut self) -> Result<Option<Vec<u8>>, HttpError>;
}

/// A body already held in memory as a list of chunks.
/// Fakes and transports that read eagerly hand it out; chunks come back
/// in order and the body ends after the last one.
#[derive(Debug, Default)]
pub struct ChunkBody(VecDeque<Vec<u8>>);

impl ChunkBody {
    /// Builds a body that yields `chunks` first to last.
    /// An empty list is an empty body.
    /// Returns the body.
    pub fn new(chunks: Vec<Vec<u8>>) -> Self {
        Self(chunks.into())
    }
}

#[async_trait]
impl Body for ChunkBody {
    async fn chunk(&mut self) -> Result<Option<Vec<u8>>, HttpError> {
        Ok(self.0.pop_front())
    }
}

/// An HTTP reply: status, final URL after redirects, headers, a body still
/// to be read, and `truncated` when the transport cut the body at its own
/// size limit. Header names keep the case the transport saw.
pub struct HttpResponse {
    pub status: u16,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Box<dyn Body>,
    pub truncated: bool,
}

impl HttpResponse {
    /// Looks up reply header `name`, ignoring ASCII case.
    /// When a header repeats, the first value wins.
    /// Returns the value, or `None` when absent.
    pub fn header(&self, name: &str) -> Option<&str> {
        find_header(&self.headers, name)
    }

    /// Returns the reply's media type in lower case without parameters,
    /// such as `text/html` for `Text/HTML; charset=utf-8`.
    /// Returns an empty string when there is no content type.
    pub fn content_type(&self) -> String {
        self.header("content-type")
            .map(media_type)
            .unwrap_or_default()
    }
}

/// Sends HTTP requests. HTTP clients, guarded clients, proxies and test
/// fakes implement it, so models and tools never own an HTTP stack.
/// Returns `Ok` for any status the server sent and `Err` when the request
/// was refused, failed in transport or broke a limit.
#[async_trait]
pub trait Transport: Send + Sync {
    async fn send(&self, req: HttpRequest) -> Result<HttpResponse, HttpError>;
}

/// Kind of a failed transport call.
/// `Invalid` is a malformed request, `Blocked` a refused address or scheme,
/// `Rejected` a reply that broke a request rule, `Timeout` a time limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HttpErrorKind {
    Invalid,
    Blocked,
    Network,
    Timeout,
    Rejected,
}

/// A failed transport call: its kind and a message without secrets.
/// Converts into `ProviderError` here and into tool errors in the tools
/// crate, keeping the message. Displays as the message.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("{message}")]
pub struct HttpError {
    pub kind: HttpErrorKind,
    pub message: String,
}

impl HttpError {
    /// Builds an error of `kind` with `message`.
    /// The message must not hold secrets; it is shown and logged.
    /// Returns the error.
    pub fn new(kind: HttpErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl From<HttpError> for ProviderError {
    fn from(e: HttpError) -> Self {
        let kind = match e.kind {
            HttpErrorKind::Timeout => ProviderErrorKind::Stalled,
            HttpErrorKind::Invalid | HttpErrorKind::Blocked | HttpErrorKind::Rejected => {
                ProviderErrorKind::BadRequest
            }
            HttpErrorKind::Network => ProviderErrorKind::Network,
        };
        ProviderError::new(kind, e.message)
    }
}

/// Reads `body` into memory, keeping at most `limit` bytes.
/// Returns the bytes and `true` when more bytes arrived than were kept;
/// reading stops at the first byte past the limit.
pub async fn read_limited(body: &mut dyn Body, limit: usize) -> Result<(Vec<u8>, bool), HttpError> {
    let mut out = Vec::new();
    while let Some(chunk) = body.chunk().await? {
        let room = limit - out.len();
        if chunk.len() > room {
            out.extend_from_slice(&chunk[..room]);
            assert!(out.len() == limit);
            return Ok((out, true));
        }
        out.extend_from_slice(&chunk);
    }
    Ok((out, false))
}

/// Reduces a `Content-Type` header value to its media type: the part
/// before any `;`, trimmed and in lower case, so `Text/HTML; charset=utf-8`
/// becomes `text/html`. Returns an empty string for an empty value.
pub fn media_type(value: &str) -> String {
    value
        .split(';')
        .next()
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
}

fn find_header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name))
        .map(|(_, v)| v.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(headers: Vec<(String, String)>, chunks: Vec<&str>) -> HttpResponse {
        let chunks = chunks.into_iter().map(|c| c.as_bytes().to_vec()).collect();
        HttpResponse {
            status: 200,
            url: "https://x/".into(),
            headers,
            body: Box::new(ChunkBody::new(chunks)),
            truncated: false,
        }
    }

    #[test]
    fn request_debug_hides_header_values() {
        let req = HttpRequest::post(
            "https://x/v1",
            vec![("Authorization".into(), "Bearer k-123".into())],
            b"{}".to_vec(),
        );
        let dbg = format!("{req:?}");
        assert!(
            dbg.contains("Authorization") && !dbg.contains("k-123"),
            "{dbg}"
        );
        assert_eq!(req.header("authorization"), Some("Bearer k-123"));
        assert_eq!(req.method.as_str(), "POST");
    }

    #[test]
    fn content_type_is_lowercase_media_type() {
        let r = reply(
            vec![("Content-Type".into(), "Text/HTML; charset=utf-8".into())],
            vec![],
        );
        assert_eq!(r.content_type(), "text/html");
        assert_eq!(reply(vec![], vec![]).content_type(), "");
    }

    #[tokio::test]
    async fn read_limited_flags_overflow() {
        let mut r = reply(vec![], vec!["abc", "def"]);
        assert_eq!(
            read_limited(r.body.as_mut(), 10).await.unwrap(),
            (b"abcdef".to_vec(), false)
        );
        let mut r = reply(vec![], vec!["abc", "def"]);
        assert_eq!(
            read_limited(r.body.as_mut(), 4).await.unwrap(),
            (b"abcd".to_vec(), true)
        );
        let mut r = reply(vec![], vec!["abc"]);
        assert_eq!(
            read_limited(r.body.as_mut(), 3).await.unwrap(),
            (b"abc".to_vec(), false)
        );
    }

    #[test]
    fn http_error_maps_to_provider_kinds() {
        let cases = [
            (HttpErrorKind::Timeout, ProviderErrorKind::Stalled),
            (HttpErrorKind::Invalid, ProviderErrorKind::BadRequest),
            (HttpErrorKind::Network, ProviderErrorKind::Network),
            (HttpErrorKind::Blocked, ProviderErrorKind::BadRequest),
            (HttpErrorKind::Rejected, ProviderErrorKind::BadRequest),
        ];
        for (from, to) in cases {
            let e: ProviderError = HttpError::new(from, "boom").into();
            assert_eq!((e.kind, e.message.as_str()), (to, "boom"));
        }
    }
}
