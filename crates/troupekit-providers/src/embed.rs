use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};
use troupekit_context::embed::Embedder;
use troupekit_context::ContextError;
use troupekit_core::secret::Secret;

use crate::model::{Endpoint, MAX_REPLY_BYTES};
use crate::transport::{read_body, HttpRequest, Transport};
use crate::wire::{AuthStyle, Protocol};

const ERROR_SNIPPET_CHARS: usize = 200;
const REDACTED: &str = "[redacted]";
const JSON_BYTES_PER_FLOAT: usize = 32;
const JSON_BYTES_PER_ITEM: usize = 256;
const JSON_BYTES_FIXED: usize = 4_096;

/// An `Embedder` backed by a vendor's embeddings endpoint.
/// OpenAI and OpenAI-compatible vendors use `POST /embeddings`, Gemini
/// `batchEmbedContents`. Build one with `Catalog::embedder_with`.
pub struct HttpEmbedder {
    selector: String,
    model: String,
    endpoint: Endpoint,
    key: Option<Secret>,
    transport: Arc<dyn Transport>,
    dims: usize,
    max_batch: usize,
}

impl fmt::Debug for HttpEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HttpEmbedder").field("selector", &self.selector).field("dims", &self.dims).field("max_batch", &self.max_batch).finish_non_exhaustive()
    }
}

impl HttpEmbedder {
    /// Builds an embedder from a resolved catalog row.
    /// Input: selector, wire model id, endpoint, key, transport, dims and
    /// batch size, all checked by the catalog. Output: the embedder.
    pub(crate) fn new(selector: String, model: String, endpoint: Endpoint, key: Option<Secret>, transport: Arc<dyn Transport>, dims: usize, max_batch: usize) -> Self {
        assert!(endpoint.protocol != Protocol::Anthropic, "anthropic has no embeddings endpoint");
        assert!(dims > 0 && max_batch > 0, "catalog checked dims and max_batch");
        assert!(key.is_some() || endpoint.auth == AuthStyle::NoKey, "catalog checked the key");
        Self { selector, model, endpoint, key, transport, dims, max_batch }
    }

    fn reply_limit(&self, items: usize) -> usize {
        let per_item = self.dims.saturating_mul(JSON_BYTES_PER_FLOAT).saturating_add(JSON_BYTES_PER_ITEM);
        per_item.saturating_mul(items).saturating_add(JSON_BYTES_FIXED).min(MAX_REPLY_BYTES)
    }

    fn redact(&self, text: &str) -> String {
        match self.key.as_ref() {
            Some(k) => text.replace(k.expose(), REDACTED),
            None => text.to_owned(),
        }
    }

    fn fail(&self, why: impl fmt::Display) -> ContextError {
        ContextError::Embed(format!("{}: {why}", self.selector))
    }

    fn request(&self, batch: &[String]) -> HttpRequest {
        let base = self.endpoint.base_url.trim_end_matches('/');
        let (url, body) = match self.endpoint.protocol {
            Protocol::Gemini => {
                let requests: Vec<Value> = batch.iter().map(|t| json!({ "model": format!("models/{}", self.model), "content": { "parts": [{ "text": t }] } })).collect();
                (format!("{base}/models/{}:batchEmbedContents", self.model), json!({ "requests": requests }))
            }
            _ => (format!("{base}/embeddings"), json!({ "model": self.model, "input": batch, "encoding_format": "float" })),
        };
        let mut headers = vec![("content-type".to_owned(), "application/json".to_owned())];
        match (self.endpoint.auth, self.key.as_ref()) {
            (AuthStyle::Bearer, Some(k)) => headers.push(("authorization".into(), format!("Bearer {}", k.expose()))),
            (AuthStyle::XApiKey, Some(k)) => headers.push(("x-api-key".into(), k.expose().to_owned())),
            (AuthStyle::XGoogApiKey, Some(k)) => headers.push(("x-goog-api-key".into(), k.expose().to_owned())),
            _ => {}
        }
        let bytes = serde_json::to_vec(&body).expect("a JSON value always serializes");
        HttpRequest::post(url, headers, bytes)
    }

    fn vector(&self, value: Option<&Value>) -> Result<Vec<f32>, ContextError> {
        let items = value.and_then(Value::as_array).ok_or_else(|| self.fail("reply item has no vector"))?;
        if items.len() != self.dims {
            return Err(self.fail(format!("expected {} dimensions, got {}", self.dims, items.len())));
        }
        let mut out = Vec::with_capacity(self.dims);
        for x in items {
            let f = x.as_f64().ok_or_else(|| self.fail("vector holds a non-number"))? as f32;
            if !f.is_finite() {
                return Err(self.fail("vector holds a non-finite value"));
            }
            out.push(f);
        }
        Ok(out)
    }

    fn decode(&self, value: &Value, count: usize) -> Result<Vec<Vec<f32>>, ContextError> {
        let items = match self.endpoint.protocol {
            Protocol::Gemini => value.get("embeddings"),
            _ => value.get("data"),
        }
        .and_then(Value::as_array)
        .ok_or_else(|| self.fail("reply has no embeddings list"))?;
        if items.len() != count {
            return Err(self.fail(format!("expected {count} vectors, got {}", items.len())));
        }
        let mut slots: Vec<Option<Vec<f32>>> = vec![None; count];
        for (pos, item) in items.iter().enumerate() {
            let (at, v) = match self.endpoint.protocol {
                Protocol::Gemini => (pos, self.vector(item.get("values"))?),
                _ => {
                    let at = match item.get("index") {
                        None => pos,
                        Some(i) => i.as_u64().and_then(|i| usize::try_from(i).ok()).ok_or_else(|| self.fail("reply index is not a number"))?,
                    };
                    (at, self.vector(item.get("embedding"))?)
                }
            };
            let slot = slots.get_mut(at).ok_or_else(|| self.fail("reply index out of range"))?;
            if slot.replace(v).is_some() {
                return Err(self.fail("reply index repeated"));
            }
        }
        let out: Vec<Vec<f32>> = slots.into_iter().flatten().collect();
        assert_eq!(out.len(), count, "count checked and every index filled once");
        Ok(out)
    }

    async fn embed_batch(&self, batch: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        assert!(!batch.is_empty() && batch.len() <= self.max_batch, "batch size out of range");
        let mut reply = self.transport.send(self.request(batch)).await.map_err(|e| self.fail(format!("transport: {:?}", e.kind)))?;
        let body = read_body(reply.body.as_mut(), self.reply_limit(batch.len())).await.map_err(|e| self.fail(e.message))?;
        if !(200..300).contains(&reply.status) {
            let text: String = self.redact(&String::from_utf8_lossy(&body)).chars().take(ERROR_SNIPPET_CHARS).collect();
            return Err(self.fail(format!("HTTP {}: {text}", reply.status)));
        }
        let value: Value = serde_json::from_slice(&body).map_err(|_| self.fail("reply is not JSON"))?;
        self.decode(&value, batch.len())
    }
}

#[async_trait]
impl Embedder for HttpEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        if texts.iter().any(|t| t.trim().is_empty()) {
            return Err(ContextError::Invalid("cannot embed a blank text".into()));
        }
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(self.max_batch) {
            out.extend(self.embed_batch(batch).await?);
        }
        assert_eq!(out.len(), texts.len(), "one vector per input text");
        Ok(out)
    }
}
