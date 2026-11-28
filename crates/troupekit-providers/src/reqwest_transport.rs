use crate::transport::{Body, HttpError, HttpErrorKind, HttpRequest, HttpResponse, Transport};
use async_trait::async_trait;
use std::time::Duration;
use troupekit_core::error::ProviderError;

/// The default HTTP transport for model providers, on reqwest with rustls.
/// A connect timeout and a per-read timeout turn a silent server into
/// `Timeout`; other failures are `Network`, with URLs stripped. Redirects
/// are never followed, so a key header never reaches another host; a 3xx
/// comes back as the reply. It does not guard addresses: give tools a
/// guarded transport instead.
#[derive(Debug, Clone)]
pub struct ReqwestTransport {
    client: reqwest::Client,
}

impl ReqwestTransport {
    /// Builds a transport with a 30 s connect timeout and a 300 s read
    /// timeout, long enough for a slow first token.
    /// Returns `Network` when the HTTP client cannot start.
    pub fn new() -> Result<Self, ProviderError> {
        Self::with_timeouts(Duration::from_secs(30), Duration::from_secs(300))
    }

    /// Builds a transport with the given `connect` timeout and per-read
    /// `read` timeout, with redirect following off.
    /// Returns `Network` when the HTTP client cannot start.
    pub fn with_timeouts(connect: Duration, read: Duration) -> Result<Self, ProviderError> {
        let client = reqwest::Client::builder()
            .connect_timeout(connect)
            .read_timeout(read)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(map_error)?;
        Ok(Self { client })
    }

    /// Wraps an existing `client`, keeping its own settings.
    /// Returns the transport.
    /// Timeouts are whatever the client was built with.
    pub fn from_client(client: reqwest::Client) -> Self {
        Self { client }
    }
}

#[async_trait]
impl Transport for ReqwestTransport {
    async fn send(&self, request: HttpRequest) -> Result<HttpResponse, HttpError> {
        let method = reqwest::Method::from_bytes(request.method.as_str().as_bytes())
            .map_err(|e| HttpError::new(HttpErrorKind::Invalid, e.to_string()))?;
        let mut builder = self.client.request(method, &request.url);
        for (name, value) in &request.headers {
            builder = builder.header(name.as_str(), value.as_str());
        }
        if let Some(body) = request.body {
            builder = builder.body(body);
        }
        let reply = builder.send().await.map_err(map_error)?;
        let status = reply.status().as_u16();
        let url = reply.url().to_string();
        let headers = reply
            .headers()
            .iter()
            .filter_map(|(k, v)| {
                v.to_str()
                    .ok()
                    .map(|v| (k.as_str().to_owned(), v.to_owned()))
            })
            .collect();
        Ok(HttpResponse {
            status,
            url,
            headers,
            body: Box::new(ReqwestBody(reply)),
            truncated: false,
        })
    }
}

struct ReqwestBody(reqwest::Response);

#[async_trait]
impl Body for ReqwestBody {
    async fn chunk(&mut self) -> Result<Option<Vec<u8>>, HttpError> {
        Ok(self
            .0
            .chunk()
            .await
            .map_err(map_error)?
            .map(|bytes| bytes.to_vec()))
    }
}

fn map_error(err: reqwest::Error) -> HttpError {
    let kind = if err.is_timeout() {
        HttpErrorKind::Timeout
    } else {
        HttpErrorKind::Network
    };
    let err = err.without_url();
    let mut message = err.to_string();
    let mut source = std::error::Error::source(&err);
    while let Some(cause) = source {
        message.push_str(": ");
        message.push_str(&cause.to_string());
        source = cause.source();
    }
    HttpError::new(kind, message)
}
