use crate::fold::Fold;
use crate::request::{CompletionRequest, CompletionResponse, Delta};
use crate::sse::{SseEvent, SseParser};
use crate::status::{bad_request, decode_error, status_error};
use crate::transport::{read_body, HttpRequest, Transport};
use crate::wire::{self, anthropic, AuthStyle, Decoder, Protocol, Quirks};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::secret::Secret;

/// Largest reply body or stream, in bytes, a model reads before failing
/// with `Decode`. Bounds memory against a runaway or hostile server.
/// Applies to unary bodies and to the total bytes of a stream.
pub const MAX_REPLY_BYTES: usize = 32 * 1024 * 1024;

/// A chat model behind one object-safe interface. Providers, fakes and
/// middleware all implement it, so callers hold `Box<dyn Model>`.
/// `stream` hands each delta to `on_delta` and returns what `complete` would.
#[async_trait]
pub trait Model: Send + Sync {
    fn id(&self) -> &str;
    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError>;
    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError>;
}

/// Where and how to reach one vendor: protocol, base URL, auth style and
/// quirks. Catalog vendors turn into one; hosts may also build one for
/// a self-hosted server.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Endpoint {
    pub protocol: Protocol,
    pub base_url: String,
    pub auth: AuthStyle,
    pub quirks: Quirks,
}

impl Endpoint {
    /// Builds the HTTP request for one call of wire `model` with `req`:
    /// URL, content and auth headers, encoded body. Pure, no I/O. Returns
    /// `Auth` when a key is needed and `key` is `None`.
    pub fn request(
        &self,
        model: &str,
        key: Option<&Secret>,
        req: &CompletionRequest,
        stream: bool,
    ) -> Result<HttpRequest, ProviderError> {
        let body = wire::encode(self.protocol, &self.quirks, model, req, stream)?;
        let base = self.base_url.trim_end_matches('/');
        let url = match (self.protocol, stream) {
            (Protocol::Anthropic, _) => format!("{base}/v1/messages"),
            (Protocol::OpenAi | Protocol::OpenAiCompatible, _) => {
                format!("{base}/chat/completions")
            }
            (Protocol::Gemini, false) => format!("{base}/models/{model}:generateContent"),
            (Protocol::Gemini, true) => {
                format!("{base}/models/{model}:streamGenerateContent?alt=sse")
            }
        };
        let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        if stream {
            headers.push(("accept".into(), "text/event-stream".into()));
        }
        if self.protocol == Protocol::Anthropic {
            headers.push(("anthropic-version".into(), anthropic::API_VERSION.into()));
        }
        let auth = match (self.auth, key) {
            (AuthStyle::NoKey, _) => None,
            (_, None) => {
                return Err(ProviderError::new(
                    ProviderErrorKind::Auth,
                    "this endpoint needs an API key",
                ))
            }
            (AuthStyle::Bearer, Some(k)) => {
                Some(("authorization", format!("Bearer {}", k.expose())))
            }
            (AuthStyle::XApiKey, Some(k)) => Some(("x-api-key", k.expose().to_owned())),
            (AuthStyle::XGoogApiKey, Some(k)) => Some(("x-goog-api-key", k.expose().to_owned())),
        };
        if let Some((name, value)) = auth {
            headers.push((name.to_owned(), value));
        }
        let body = serde_json::to_vec(&body)
            .map_err(|e| bad_request(format!("request does not serialize: {e}")))?;
        Ok(HttpRequest::post(url, headers, body))
    }
}

/// Reports whether `id` is safe to place in a URL path as a model id:
/// 1 to 200 ASCII letters, digits and `._:/-`, with no `..` segment.
/// Catalog and `HttpModel` both check it.
pub fn valid_model_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 200
        && !id.contains("..")
        && id
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b"._:/-".contains(&b))
}

/// A model reached over HTTP through an injected `Transport`.
/// Unary and streamed calls share one encoder, one decoder per protocol
/// and one `Fold`, so both return the same response for the same reply.
pub struct HttpModel {
    id: String,
    model: String,
    endpoint: Endpoint,
    key: Option<Secret>,
    transport: Arc<dyn Transport>,
}

impl HttpModel {
    /// Builds a model named `id` (usually `vendor/model`) that sends wire
    /// `model` to `endpoint` with an optional `key` through `transport`.
    /// Returns `BadRequest` when `model` fails `valid_model_id`.
    pub fn new(
        id: impl Into<String>,
        model: impl Into<String>,
        endpoint: Endpoint,
        key: Option<Secret>,
        transport: Arc<dyn Transport>,
    ) -> Result<Self, ProviderError> {
        let model = model.into();
        if !valid_model_id(&model) {
            return Err(bad_request(format!("model id {model:?} is not allowed")));
        }
        Ok(Self {
            id: id.into(),
            model,
            endpoint,
            key,
            transport,
        })
    }

    fn prepare(&self, req: &CompletionRequest, stream: bool) -> Result<HttpRequest, ProviderError> {
        req.validate()?;
        if let Some(asked) = req.model.as_deref().filter(|m| *m != self.model) {
            return Err(bad_request(format!(
                "request names model {asked}, this handle serves {}",
                self.model
            )));
        }
        self.endpoint
            .request(&self.model, self.key.as_ref(), req, stream)
    }
}

#[async_trait]
impl Model for HttpModel {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let request = self.prepare(req, false)?;
        let mut reply = self.transport.send(request).await?;
        let body = read_body(reply.body.as_mut(), MAX_REPLY_BYTES).await?;
        if !(200..300).contains(&reply.status) {
            return Err(status_error(reply.status, &reply.headers, &body));
        }
        let value: Value = serde_json::from_slice(&body)
            .map_err(|e| decode_error(format!("reply is not JSON: {e}")))?;
        let mut fold = Fold::new();
        for delta in wire::decode_body(self.endpoint.protocol, &value)? {
            fold.push(&delta)?;
        }
        fold.finish()
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let request = self.prepare(req, true)?;
        let mut reply = self.transport.send(request).await?;
        if !(200..300).contains(&reply.status) {
            let body = read_body(reply.body.as_mut(), MAX_REPLY_BYTES)
                .await
                .unwrap_or_default();
            return Err(status_error(reply.status, &reply.headers, &body));
        }
        let mut parser = SseParser::new();
        let mut decoder = Decoder::new(self.endpoint.protocol);
        let mut fold = Fold::new();
        let mut received = 0usize;
        while let Some(chunk) = reply.body.chunk().await? {
            received += chunk.len();
            if received > MAX_REPLY_BYTES {
                return Err(decode_error("reply stream is too large"));
            }
            apply(parser.push(&chunk)?, &mut decoder, &mut fold, on_delta)?;
        }
        apply(parser.finish()?, &mut decoder, &mut fold, on_delta)?;
        fold.finish()
    }
}

fn apply(
    events: Vec<SseEvent>,
    decoder: &mut Decoder,
    fold: &mut Fold,
    on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
) -> Result<(), ProviderError> {
    for event in &events {
        for delta in decoder.sse(event)? {
            fold.push(&delta)?;
            on_delta(&delta);
        }
    }
    Ok(())
}
