use super::{error_object, push_turn, str_at, text_delta, u64_at, Quirks, SchemaMode};
use crate::request::{CompletionRequest, Delta, StopReason};
use crate::status::{bad_request, decode_error};
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use troupekit_core::error::ProviderError;
use troupekit_core::message::{ContentPart, MediaSource, Role};

/// Encodes `req` as a Gemini generateContent body; the model id goes in
/// the URL. Returns `BadRequest` for a tool result whose call id matches
/// no earlier tool call, since Gemini needs the function name.
pub fn encode(
    quirks: &Quirks,
    _model: &str,
    req: &CompletionRequest,
    _stream: bool,
) -> Result<Value, ProviderError> {
    let mut names: HashMap<&str, &str> = HashMap::new();
    let mut contents = Vec::new();
    for m in &req.messages {
        let role = match m.role {
            Role::System => continue,
            Role::Assistant => "model",
            Role::User | Role::Tool => "user",
        };
        let mut parts = Vec::new();
        for part in &m.parts {
            parts.push(match part {
                ContentPart::Hosted { .. } => continue,
                ContentPart::Text { text } => json!({"text": text}),
                ContentPart::Image {
                    media_type,
                    source: MediaSource::Url { url },
                } => {
                    json!({"fileData": {"mimeType": media_type, "fileUri": url}})
                }
                ContentPart::Image {
                    media_type,
                    source: MediaSource::Base64 { data },
                } => {
                    json!({"inlineData": {"mimeType": media_type, "data": data}})
                }
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => {
                    names.insert(id.as_str(), name.as_str());
                    json!({"functionCall": {"name": name, "args": arguments}})
                }
                ContentPart::ToolResult {
                    call_id,
                    content,
                    is_error,
                } => {
                    let name = names.get(call_id.as_str()).ok_or_else(|| {
                        bad_request(format!(
                            "tool result {call_id} matches no earlier tool call"
                        ))
                    })?;
                    let mut response = Map::new();
                    response.insert(
                        if *is_error { "error" } else { "output" }.into(),
                        json!(content),
                    );
                    json!({"functionResponse": {"name": name, "response": response}})
                }
            });
        }
        push_turn(&mut contents, role, "parts", parts);
    }
    let mut config = json!({"maxOutputTokens": req.max_tokens});
    if let (true, Some(t)) = (quirks.temperature, req.temperature) {
        config["temperature"] = json!(t);
    }
    if let Some(schema) = &req.output_schema {
        match quirks.output_schema {
            SchemaMode::JsonSchema => {
                config["responseMimeType"] = json!("application/json");
                config["responseJsonSchema"] = schema.clone();
            }
            SchemaMode::JsonObject => config["responseMimeType"] = json!("application/json"),
            SchemaMode::Unsupported => {}
        }
    }
    let mut body = json!({"contents": contents, "generationConfig": config});
    if let Some(system) = req.system_text() {
        body["systemInstruction"] = json!({"parts": [{"text": system}]});
    }
    if !req.tools.is_empty() {
        let declarations: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "parametersJsonSchema": t.schema}))
            .collect();
        body["tools"] = json!([{"functionDeclarations": declarations}]);
    }
    Ok(body)
}

/// Turns a whole generateContent reply into deltas. Streamed chunks have
/// the same shape, so this is one event of a fresh `StreamDecoder`.
/// Returns the error an error body carries.
pub fn decode_body(body: &Value) -> Result<Vec<Delta>, ProviderError> {
    StreamDecoder::default().event(body)
}

/// Decodes one Gemini stream. Thought parts are skipped, each function
/// call arrives whole, and calls without a wire id get `call_<n>`,
/// numbered per reply. Make one per streamed reply.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    next_call: u32,
}

impl StreamDecoder {
    /// Decodes one chunk's parsed `data`, reading the first candidate.
    /// Usage values are running totals; output counts thought tokens.
    /// Returns its deltas, or the error an error chunk carries.
    pub fn event(&mut self, data: &Value) -> Result<Vec<Delta>, ProviderError> {
        if let Some(err) = data.get("error").filter(|e| !e.is_null()) {
            return Err(error_object(err));
        }
        let mut out = Vec::new();
        if let Some(candidate) = data.pointer("/candidates/0") {
            for part in candidate
                .pointer("/content/parts")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                if part.get("thought").and_then(Value::as_bool) == Some(true) {
                    continue;
                }
                out.extend(text_delta(str_at(part, "/text")));
                if let Some(call) = part.get("functionCall") {
                    let index = self.next_call;
                    self.next_call = index
                        .checked_add(1)
                        .ok_or_else(|| decode_error("too many function calls"))?;
                    let id = str_at(call, "/id")
                        .filter(|id| !id.is_empty())
                        .map(str::to_owned)
                        .unwrap_or_else(|| format!("call_{index}"));
                    out.push(Delta::ToolCallStarted {
                        index,
                        id,
                        name: str_at(call, "/name").unwrap_or_default().to_owned(),
                    });
                    let args = call.get("args").cloned().unwrap_or_else(|| json!({}));
                    out.push(Delta::ToolCallArgs {
                        index,
                        json: args.to_string(),
                    });
                }
            }
            if let Some(reason) = str_at(candidate, "/finishReason") {
                out.push(Delta::Stop {
                    reason: finish_reason(reason),
                });
            }
        }
        if str_at(data, "/promptFeedback/blockReason").is_some() {
            out.push(Delta::Stop {
                reason: StopReason::Refusal,
            });
        }
        if let Some(u) = data.get("usageMetadata") {
            let output = [
                u64_at(u, "/candidatesTokenCount"),
                u64_at(u, "/thoughtsTokenCount"),
            ]
            .into_iter()
            .flatten()
            .reduce(u64::saturating_add);
            out.push(Delta::Usage {
                input_tokens: u64_at(u, "/promptTokenCount"),
                output_tokens: output,
            });
        }
        Ok(out)
    }
}

fn finish_reason(reason: &str) -> StopReason {
    match reason {
        "STOP" => StopReason::EndTurn,
        "MAX_TOKENS" => StopReason::MaxTokens,
        "SAFETY" | "RECITATION" | "BLOCKLIST" | "PROHIBITED_CONTENT" | "SPII" | "IMAGE_SAFETY" => {
            StopReason::Refusal
        }
        other => StopReason::Other(other.to_owned()),
    }
}
