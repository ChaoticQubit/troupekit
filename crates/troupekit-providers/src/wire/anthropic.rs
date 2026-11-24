use super::{index_at, position, push_turn, str_at, text_delta, u64_at, Quirks, SchemaMode};
use crate::request::{CompletionRequest, Delta, StopReason};
use crate::status::decode_error;
use serde_json::{json, Value};
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::message::{ContentPart, MediaSource, Role};
use troupekit_core::tool::{HostedTool, ToolSpec};

/// The `anthropic-version` header this encoder targets.
/// On it Claude rejects sampling parameters, so the catalog sets the
/// `temperature` quirk to false for Anthropic.
pub const API_VERSION: &str = "2026-10-21";

/// Server tool types for hosted web search and fetch. The basic versions
/// run on every current Claude model without a beta header.
/// Hosted tools are sent with these and their fixed names.
pub const WEB_SEARCH_TYPE: &str = "web_search_20250305";

/// The hosted web fetch tool type; see `WEB_SEARCH_TYPE`.
/// It fetches only URLs already in the conversation.
/// Its page size is capped by the tool's `max_content_tokens`.
pub const WEB_FETCH_TYPE: &str = "web_fetch_20250910";

/// Encodes `req` as an Anthropic Messages body for `model`.
/// Tool results ride in user turns and same-role turns are merged.
/// Hosted tools become server tools and drop `output_config`, since
/// cited search results cannot be combined with a JSON output format.
pub fn encode(
    quirks: &Quirks,
    model: &str,
    req: &CompletionRequest,
    stream: bool,
) -> Result<Value, ProviderError> {
    let mut messages = Vec::new();
    for m in &req.messages {
        let role = match m.role {
            Role::System => continue,
            Role::Assistant => "assistant",
            Role::User | Role::Tool => "user",
        };
        push_turn(
            &mut messages,
            role,
            "content",
            m.parts.iter().map(block).collect(),
        );
    }
    let mut body = json!({"model": model, "max_tokens": req.max_tokens, "messages": messages});
    if let Some(system) = req.system_text() {
        body["system"] = json!(system);
    }
    if !req.tools.is_empty() {
        body["tools"] = req.tools.iter().map(tool).collect();
    }
    if let (true, Some(t)) = (quirks.temperature, req.temperature) {
        body["temperature"] = json!(t);
    }
    let hosted = req.tools.iter().any(|t| t.hosted.is_some());
    if let (Some(schema), false, false) = (
        &req.output_schema,
        quirks.output_schema == SchemaMode::Unsupported,
        hosted,
    ) {
        body["output_config"] = json!({"format": {"type": "json_schema", "schema": schema}});
    }
    if stream {
        body["stream"] = json!(true);
    }
    Ok(body)
}

/// Turns a whole Messages reply into the deltas its stream would yield.
/// Thinking is skipped; server tool blocks become `Hosted`. Returns `Decode` without a
/// content array, or the mapped error for an error body.
pub fn decode_body(body: &Value) -> Result<Vec<Delta>, ProviderError> {
    if body["type"] == "error" {
        return Err(error_event(body));
    }
    let blocks = body["content"]
        .as_array()
        .ok_or_else(|| decode_error("reply has no content"))?;
    let mut out = vec![usage(&body["usage"])];
    for (i, b) in blocks.iter().enumerate() {
        let index = position(i)?;
        match str_at(b, "/type") {
            Some("text") => out.extend(text_delta(str_at(b, "/text"))),
            Some("tool_use") => {
                out.push(started(index, b));
                out.push(Delta::ToolCallArgs {
                    index,
                    json: b["input"].to_string(),
                });
            }
            Some("thinking") | Some("redacted_thinking") => {}
            Some(_) => out.push(hosted(index, b)),
            None => {}
        }
    }
    out.push(Delta::Stop {
        reason: stop_reason(str_at(body, "/stop_reason").unwrap_or("end_turn")),
    });
    Ok(out)
}

/// Decodes one Anthropic Messages stream. Wire block indexes key tool
/// calls; thinking, signature, ping and stop events yield nothing.
/// Make one per streamed reply.
#[derive(Debug, Default)]
pub struct StreamDecoder;

impl StreamDecoder {
    /// Decodes one event's parsed `data`.
    /// Returns its deltas, or the mapped error for an `error` event.
    /// Usage arrives in message_start and message_delta.
    pub fn event(&mut self, data: &Value) -> Result<Vec<Delta>, ProviderError> {
        Ok(match str_at(data, "/type").unwrap_or("") {
            "message_start" => vec![usage(&data["message"]["usage"])],
            "content_block_start" => {
                let b = &data["content_block"];
                match str_at(b, "/type") {
                    Some("tool_use") => vec![started(index_at(data, "/index")?, b)],
                    Some("text") => text_delta(str_at(b, "/text")).into_iter().collect(),
                    Some("thinking") | Some("redacted_thinking") | None => Vec::new(),
                    Some(_) => vec![hosted(index_at(data, "/index")?, b)],
                }
            }
            "content_block_delta" => match str_at(data, "/delta/type") {
                Some("text_delta") => text_delta(str_at(data, "/delta/text"))
                    .into_iter()
                    .collect(),
                Some("input_json_delta") => vec![Delta::ToolCallArgs {
                    index: index_at(data, "/index")?,
                    json: str_at(data, "/delta/partial_json")
                        .unwrap_or_default()
                        .to_owned(),
                }],
                _ => Vec::new(),
            },
            "message_delta" => {
                let mut out = vec![usage(&data["usage"])];
                if let Some(reason) = str_at(data, "/delta/stop_reason") {
                    out.push(Delta::Stop {
                        reason: stop_reason(reason),
                    });
                }
                out
            }
            "error" => return Err(error_event(data)),
            _ => Vec::new(),
        })
    }
}

fn tool(t: &ToolSpec) -> Value {
    match t.hosted {
        None => json!({"name": t.name, "description": t.description, "input_schema": t.schema}),
        Some(HostedTool::WebSearch { max_uses }) => {
            json!({"type": WEB_SEARCH_TYPE, "name": "web_search", "max_uses": max_uses})
        }
        Some(HostedTool::WebFetch {
            max_uses,
            max_content_tokens,
        }) => json!({"type": WEB_FETCH_TYPE, "name": "web_fetch", "max_uses": max_uses,
            "max_content_tokens": max_content_tokens}),
    }
}

fn block(part: &ContentPart) -> Value {
    match part {
        ContentPart::Hosted { block, .. } => block.clone(),
        ContentPart::Text { text } => json!({"type": "text", "text": text}),
        ContentPart::Image {
            source: MediaSource::Url { url },
            ..
        } => {
            json!({"type": "image", "source": {"type": "url", "url": url}})
        }
        ContentPart::Image {
            media_type,
            source: MediaSource::Base64 { data },
        } => {
            json!({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}})
        }
        ContentPart::ToolCall {
            id,
            name,
            arguments,
        } => json!({"type": "tool_use", "id": id, "name": name, "input": arguments}),
        ContentPart::ToolResult {
            call_id,
            content,
            is_error,
        } => {
            json!({"type": "tool_result", "tool_use_id": call_id, "content": content, "is_error": is_error})
        }
    }
}

fn hosted(index: u32, block: &Value) -> Delta {
    let tool = (str_at(block, "/type") == Some("server_tool_use"))
        .then(|| str_at(block, "/name").map(str::to_owned))
        .flatten();
    Delta::Hosted {
        index,
        tool,
        block: block.clone(),
    }
}

fn started(index: u32, block: &Value) -> Delta {
    Delta::ToolCallStarted {
        index,
        id: str_at(block, "/id").unwrap_or_default().to_owned(),
        name: str_at(block, "/name").unwrap_or_default().to_owned(),
    }
}

fn usage(u: &Value) -> Delta {
    let input = [
        "/input_tokens",
        "/cache_creation_input_tokens",
        "/cache_read_input_tokens",
    ]
    .iter()
    .filter_map(|p| u64_at(u, p))
    .reduce(u64::saturating_add);
    Delta::Usage {
        input_tokens: input,
        output_tokens: u64_at(u, "/output_tokens"),
    }
}

fn stop_reason(reason: &str) -> StopReason {
    match reason {
        "end_turn" | "stop_sequence" => StopReason::EndTurn,
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "refusal" => StopReason::Refusal,
        "pause_turn" => StopReason::PauseTurn,
        other => StopReason::Other(other.to_owned()),
    }
}

fn error_event(data: &Value) -> ProviderError {
    let kind = match str_at(data, "/error/type").unwrap_or("") {
        "rate_limit_error" => ProviderErrorKind::RateLimited,
        "authentication_error" | "permission_error" => ProviderErrorKind::Auth,
        "invalid_request_error" | "not_found_error" | "request_too_large" => {
            ProviderErrorKind::BadRequest
        }
        "timeout_error" => ProviderErrorKind::Stalled,
        _ => ProviderErrorKind::Server,
    };
    ProviderError::new(
        kind,
        str_at(data, "/error/message")
            .unwrap_or("stream error")
            .to_owned(),
    )
}
