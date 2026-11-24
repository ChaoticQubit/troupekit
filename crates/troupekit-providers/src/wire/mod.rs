//! The wire protocols: pure encode and decode functions, no I/O.
//! Each protocol module turns a `CompletionRequest` into JSON and wire
//! JSON back into `Delta`s; `Fold` turns deltas into the response.

pub mod anthropic;
pub mod gemini;
pub mod openai;

use crate::request::{CompletionRequest, Delta};
use crate::sse::SseEvent;
use crate::status::{decode_error, kind_for_status};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use troupekit_core::error::{ProviderError, ProviderErrorKind};

/// The wire protocols Troupekit speaks. OpenAI-compatible vendors use
/// the OpenAI Chat Completions encoding with their own `Quirks`.
/// Serializes as `anthropic`, `openai`, `gemini`, `openai_compatible`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Protocol {
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "gemini")]
    Gemini,
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible,
}

/// How a vendor takes its API key: a bearer token, an `x-api-key` or
/// `x-goog-api-key` header, or no key at all for local servers.
/// Serializes as `bearer`, `x-api-key`, `x-goog-api-key`, `none`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthStyle {
    #[serde(rename = "bearer")]
    Bearer,
    #[serde(rename = "x-api-key")]
    XApiKey,
    #[serde(rename = "x-goog-api-key")]
    XGoogApiKey,
    #[serde(rename = "none")]
    NoKey,
}

/// Which JSON field carries the output token cap on the OpenAI wire.
/// OpenAI's own reasoning models want `max_completion_tokens`.
/// Serializes as `max_tokens` or `max_completion_tokens`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MaxTokensField {
    #[default]
    MaxTokens,
    MaxCompletionTokens,
}

/// How a vendor takes a requested output JSON Schema: as a JSON Schema,
/// as plain JSON mode, or not at all (the schema is then left out and
/// the caller validates the output). Serializes as `json_schema`, `json_object`, `none`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum SchemaMode {
    #[default]
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
    #[serde(rename = "none")]
    Unsupported,
}

/// Per-vendor differences inside one protocol, read from the catalog.
/// Every field has a default, so most vendors set none of them; unknown
/// fields are rejected so a typo cannot pass silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Quirks {
    pub max_tokens_field: MaxTokensField,
    pub stream_usage: bool,
    pub output_schema: SchemaMode,
    pub temperature: bool,
}

impl Default for Quirks {
    fn default() -> Self {
        Self {
            max_tokens_field: MaxTokensField::MaxTokens,
            stream_usage: true,
            output_schema: SchemaMode::JsonSchema,
            temperature: true,
        }
    }
}

/// Encodes `req` as the JSON body `protocol` expects for wire `model`.
/// Pure, so goldens pin the exact shape. Returns `BadRequest` when the
/// request cannot be expressed on this wire.
pub fn encode(
    protocol: Protocol,
    quirks: &Quirks,
    model: &str,
    req: &CompletionRequest,
    stream: bool,
) -> Result<Value, ProviderError> {
    let req = &req.sendable(protocol == Protocol::Anthropic)?;
    match protocol {
        Protocol::Anthropic => anthropic::encode(quirks, model, req, stream),
        Protocol::OpenAi | Protocol::OpenAiCompatible => openai::encode(quirks, model, req, stream),
        Protocol::Gemini => gemini::encode(quirks, model, req, stream),
    }
}

/// Turns a whole unary reply `body` into the deltas a stream of the
/// same reply would yield, so both paths share one `Fold`.
/// Returns `Decode` for a malformed body, or the error the body carries.
pub fn decode_body(protocol: Protocol, body: &Value) -> Result<Vec<Delta>, ProviderError> {
    match protocol {
        Protocol::Anthropic => anthropic::decode_body(body),
        Protocol::OpenAi | Protocol::OpenAiCompatible => openai::decode_body(body),
        Protocol::Gemini => gemini::decode_body(body),
    }
}

/// The streaming decoder for one reply, chosen by protocol.
/// Make a new one per reply; it keeps per-reply state such as which tool
/// calls have started.
#[derive(Debug)]
pub enum Decoder {
    Anthropic(anthropic::StreamDecoder),
    OpenAi(openai::StreamDecoder),
    Gemini(gemini::StreamDecoder),
}

impl Decoder {
    /// Starts a decoder for one streamed reply of `protocol`.
    /// Returns the decoder.
    /// Use one per reply; it holds per-reply state.
    pub fn new(protocol: Protocol) -> Self {
        match protocol {
            Protocol::Anthropic => Decoder::Anthropic(Default::default()),
            Protocol::OpenAi | Protocol::OpenAiCompatible => Decoder::OpenAi(Default::default()),
            Protocol::Gemini => Decoder::Gemini(Default::default()),
        }
    }

    /// Decodes the parsed `data` of one stream event.
    /// Returns its deltas, or the error an error event carries.
    /// Callers that parse SSE themselves use `sse` instead.
    pub fn event(&mut self, data: &Value) -> Result<Vec<Delta>, ProviderError> {
        match self {
            Decoder::Anthropic(d) => d.event(data),
            Decoder::OpenAi(d) => d.event(data),
            Decoder::Gemini(d) => d.event(data),
        }
    }

    /// Decodes one server-sent `event`: skips empty data and the OpenAI
    /// `[DONE]` marker, parses the JSON and hands it to `event`.
    /// Returns `Decode` when the data is not JSON.
    pub fn sse(&mut self, event: &SseEvent) -> Result<Vec<Delta>, ProviderError> {
        let data = event.data.trim();
        if data.is_empty() || data == "[DONE]" {
            return Ok(Vec::new());
        }
        let value: Value = serde_json::from_str(data)
            .map_err(|e| decode_error(format!("event data is not JSON: {e}")))?;
        self.event(&value)
    }
}

pub(crate) fn push_turn(turns: &mut Vec<Value>, role: &str, key: &str, items: Vec<Value>) {
    if items.is_empty() {
        return;
    }
    if let Some(last) = turns.last_mut() {
        if last["role"] == role {
            if let Some(existing) = last[key].as_array_mut() {
                existing.extend(items);
                return;
            }
        }
    }
    let mut turn = Map::new();
    turn.insert("role".into(), Value::from(role));
    turn.insert(key.into(), Value::Array(items));
    turns.push(Value::Object(turn));
}

pub(crate) fn str_at<'a>(value: &'a Value, pointer: &str) -> Option<&'a str> {
    value.pointer(pointer).and_then(Value::as_str)
}

pub(crate) fn u64_at(value: &Value, pointer: &str) -> Option<u64> {
    value.pointer(pointer).and_then(Value::as_u64)
}

pub(crate) fn index_at(value: &Value, pointer: &str) -> Result<u32, ProviderError> {
    u64_at(value, pointer)
        .and_then(|n| u32::try_from(n).ok())
        .ok_or_else(|| decode_error(format!("missing or bad index at {pointer}")))
}

pub(crate) fn position(i: usize) -> Result<u32, ProviderError> {
    u32::try_from(i).map_err(|_| decode_error("too many parts in one reply"))
}

pub(crate) fn text_delta(text: Option<&str>) -> Option<Delta> {
    text.filter(|t| !t.is_empty())
        .map(|t| Delta::TextDelta { text: t.to_owned() })
}

pub(crate) fn error_object(error: &Value) -> ProviderError {
    let kind = u64_at(error, "/code")
        .and_then(|c| u16::try_from(c).ok())
        .map(kind_for_status)
        .unwrap_or(ProviderErrorKind::Server);
    ProviderError::new(
        kind,
        str_at(error, "/message")
            .unwrap_or("provider error")
            .to_owned(),
    )
}
