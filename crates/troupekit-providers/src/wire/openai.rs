use super::{
    error_object, index_at, position, str_at, text_delta, u64_at, MaxTokensField, Quirks,
    SchemaMode,
};
use crate::request::{CompletionRequest, Delta, StopReason};
use crate::status::{bad_request, decode_error};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use troupekit_core::error::ProviderError;
use troupekit_core::message::{ContentPart, MediaSource, Message, Role};

/// Encodes `req` as an OpenAI Chat Completions body for `model`, shaped
/// by the vendor `quirks`. Returns `BadRequest` for parts this wire
/// cannot carry, such as text in a tool turn or images from the assistant.
pub fn encode(
    quirks: &Quirks,
    model: &str,
    req: &CompletionRequest,
    stream: bool,
) -> Result<Value, ProviderError> {
    let mut messages = Vec::new();
    if let Some(system) = req.system_text() {
        messages.push(json!({"role": "system", "content": system}));
    }
    for m in &req.messages {
        match m.role {
            Role::System => {}
            Role::User => messages.push(json!({"role": "user", "content": user_content(m)?})),
            Role::Assistant => messages.push(assistant(m)?),
            Role::Tool => {
                for part in &m.parts {
                    match part {
                        ContentPart::ToolResult {
                            call_id, content, ..
                        } => messages.push(
                            json!({"role": "tool", "tool_call_id": call_id, "content": content}),
                        ),
                        _ => {
                            return Err(bad_request(
                                "a tool turn may only hold tool results on this wire",
                            ))
                        }
                    }
                }
            }
        }
    }
    let mut body = json!({"model": model, "messages": messages});
    let cap = match quirks.max_tokens_field {
        MaxTokensField::MaxTokens => "max_tokens",
        MaxTokensField::MaxCompletionTokens => "max_completion_tokens",
    };
    body[cap] = json!(req.max_tokens);
    if !req.tools.is_empty() {
        body["tools"] = req
            .tools
            .iter()
            .map(|t| json!({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}))
            .collect();
    }
    if let (true, Some(t)) = (quirks.temperature, req.temperature) {
        body["temperature"] = json!(t);
    }
    if let Some(schema) = &req.output_schema {
        match quirks.output_schema {
            SchemaMode::JsonSchema => {
                body["response_format"] = json!({"type": "json_schema", "json_schema": {"name": "output", "schema": schema}})
            }
            SchemaMode::JsonObject => body["response_format"] = json!({"type": "json_object"}),
            SchemaMode::Unsupported => {}
        }
    }
    if stream {
        body["stream"] = json!(true);
        if quirks.stream_usage {
            body["stream_options"] = json!({"include_usage": true});
        }
    }
    Ok(body)
}

/// Turns a whole Chat Completions reply into the deltas its stream would
/// yield, reading the first choice only.
/// Returns `Decode` without choices, or the error an error body carries.
pub fn decode_body(body: &Value) -> Result<Vec<Delta>, ProviderError> {
    if let Some(err) = body.get("error").filter(|e| !e.is_null()) {
        return Err(error_object(err));
    }
    let choice = body
        .pointer("/choices/0")
        .ok_or_else(|| decode_error("reply has no choices"))?;
    let message = &choice["message"];
    let mut out: Vec<Delta> = text_delta(str_at(message, "/content"))
        .into_iter()
        .collect();
    if let Some(calls) = message["tool_calls"].as_array() {
        for (i, c) in calls.iter().enumerate() {
            let index = position(i)?;
            out.push(Delta::ToolCallStarted {
                index,
                id: str_at(c, "/id").unwrap_or_default().to_owned(),
                name: str_at(c, "/function/name").unwrap_or_default().to_owned(),
            });
            out.push(Delta::ToolCallArgs {
                index,
                json: arguments(&c["function"]["arguments"]),
            });
        }
    }
    out.push(usage(&body["usage"]));
    out.push(Delta::Stop {
        reason: finish_reason(str_at(choice, "/finish_reason").unwrap_or("stop")),
    });
    Ok(out)
}

/// Decodes one Chat Completions stream. Tool calls are keyed by their
/// wire index and start once, even when a server repeats the id in every
/// fragment. Make one per streamed reply.
#[derive(Debug, Default)]
pub struct StreamDecoder {
    started: BTreeSet<u32>,
}

impl StreamDecoder {
    /// Decodes one chunk's parsed `data`, reading the first choice only.
    /// Returns its deltas, or the error an error chunk carries.
    /// Usage arrives in a final chunk when the vendor sends it.
    pub fn event(&mut self, data: &Value) -> Result<Vec<Delta>, ProviderError> {
        if let Some(err) = data.get("error").filter(|e| !e.is_null()) {
            return Err(error_object(err));
        }
        let mut out = Vec::new();
        if let Some(choice) = data.pointer("/choices/0") {
            let delta = &choice["delta"];
            out.extend(text_delta(str_at(delta, "/content")));
            if let Some(calls) = delta["tool_calls"].as_array() {
                for (i, c) in calls.iter().enumerate() {
                    let index = if c.get("index").is_some() {
                        index_at(c, "/index")?
                    } else {
                        position(i)?
                    };
                    if let Some(id) = str_at(c, "/id").filter(|id| !id.is_empty()) {
                        if self.started.insert(index) {
                            out.push(Delta::ToolCallStarted {
                                index,
                                id: id.to_owned(),
                                name: str_at(c, "/function/name").unwrap_or_default().to_owned(),
                            });
                        }
                    }
                    let json = arguments(&c["function"]["arguments"]);
                    if !json.is_empty() {
                        out.push(Delta::ToolCallArgs { index, json });
                    }
                }
            }
            if let Some(reason) = str_at(choice, "/finish_reason") {
                out.push(Delta::Stop {
                    reason: finish_reason(reason),
                });
            }
        }
        if data.get("usage").is_some_and(Value::is_object) {
            out.push(usage(&data["usage"]));
        }
        Ok(out)
    }
}

fn user_content(m: &Message) -> Result<Value, ProviderError> {
    if m.parts
        .iter()
        .all(|p| matches!(p, ContentPart::Text { .. }))
    {
        return Ok(json!(m.text_content()));
    }
    let mut items = Vec::new();
    for part in &m.parts {
        items.push(match part {
            ContentPart::Text { text } => json!({"type": "text", "text": text}),
            ContentPart::Image { media_type, source } => {
                let url = match source {
                    MediaSource::Url { url } => url.clone(),
                    MediaSource::Base64 { data } => format!("data:{media_type};base64,{data}"),
                };
                json!({"type": "image_url", "image_url": {"url": url}})
            }
            _ => {
                return Err(bad_request(
                    "a user turn may only hold text and images on this wire",
                ))
            }
        });
    }
    Ok(Value::Array(items))
}

fn assistant(m: &Message) -> Result<Value, ProviderError> {
    let mut calls = Vec::new();
    for part in &m.parts {
        match part {
            ContentPart::Text { .. } => {}
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => calls.push(json!({
                "id": id,
                "type": "function",
                "function": {"name": name, "arguments": arguments.to_string()}
            })),
            _ => {
                return Err(bad_request(
                    "an assistant turn may only hold text and tool calls on this wire",
                ))
            }
        }
    }
    let text = m.text_content();
    let mut message = json!({"role": "assistant", "content": if text.is_empty() { Value::Null } else { json!(text) }});
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    Ok(message)
}

fn arguments(raw: &Value) -> String {
    match raw {
        Value::String(s) => s.clone(),
        Value::Null => String::new(),
        other => other.to_string(),
    }
}

fn usage(u: &Value) -> Delta {
    Delta::Usage {
        input_tokens: u64_at(u, "/prompt_tokens"),
        output_tokens: u64_at(u, "/completion_tokens"),
    }
}

fn finish_reason(reason: &str) -> StopReason {
    match reason {
        "stop" => StopReason::EndTurn,
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::Refusal,
        other => StopReason::Other(other.to_owned()),
    }
}
