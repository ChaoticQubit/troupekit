use crate::status::bad_request;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use troupekit_core::error::ProviderError;
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::tool::ToolSpec;

/// What a blank tool result is sent as, since vendors reject blank text.
/// The model still sees that its call ran and returned nothing.
/// Used by `CompletionRequest::sendable`.
pub const NO_OUTPUT: &str = "(no output)";

/// One call to a chat model: the conversation, the tools it may call and
/// the output limits. `model` is the wire model id the caller expects,
/// or `None` for the model the handle was connected to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub max_tokens: u32,
    pub temperature: Option<f64>,
    pub output_schema: Option<Value>,
}

impl CompletionRequest {
    /// Builds a request holding one user message with `text`, capped at
    /// `max_tokens` output tokens. Every other field starts empty.
    /// Returns the new request.
    pub fn user(text: impl Into<String>, max_tokens: u32) -> Self {
        Self {
            model: None,
            system: None,
            messages: vec![Message::text(Role::User, text)],
            tools: Vec::new(),
            max_tokens,
            temperature: None,
            output_schema: None,
        }
    }

    /// Checks the request before any protocol encodes it.
    /// Returns `BadRequest` for no messages, a zero token cap, a temperature
    /// outside 0 to 2, or a blank or repeated tool name.
    pub fn validate(&self) -> Result<(), ProviderError> {
        if self.messages.is_empty() {
            return Err(bad_request("request has no messages"));
        }
        if self.max_tokens == 0 {
            return Err(bad_request("max_tokens must be above zero"));
        }
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(bad_request("temperature must be between 0 and 2"));
            }
        }
        let mut names = HashSet::new();
        for tool in &self.tools {
            if tool.name.trim().is_empty() {
                return Err(bad_request("tool name is blank"));
            }
            if !names.insert(tool.name.as_str()) {
                return Err(bad_request(format!("tool name {} is repeated", tool.name)));
            }
        }
        Ok(())
    }

    /// Returns the copy a wire sends: blank text parts are dropped, hosted
    /// blocks too unless `keep_hosted`, a message left with no part is
    /// dropped, and a blank tool result reads `(no output)` so it still
    /// answers its call. Returns `BadRequest` when no message is left.
    pub fn sendable(&self, keep_hosted: bool) -> Result<CompletionRequest, ProviderError> {
        let mut out = self.clone();
        out.messages = self
            .messages
            .iter()
            .filter_map(|m| {
                let parts: Vec<ContentPart> = m
                    .parts
                    .iter()
                    .filter(|p| !matches!(p, ContentPart::Text { text } if text.trim().is_empty()))
                    .filter(|p| keep_hosted || !matches!(p, ContentPart::Hosted { .. }))
                    .cloned()
                    .map(|p| match p {
                        ContentPart::ToolResult {
                            call_id,
                            content,
                            is_error,
                        } if content.trim().is_empty() => ContentPart::ToolResult {
                            call_id,
                            content: NO_OUTPUT.into(),
                            is_error,
                        },
                        other => other,
                    })
                    .collect();
                (!parts.is_empty()).then(|| Message {
                    role: m.role,
                    parts,
                })
            })
            .collect();
        if out.messages.iter().all(|m| m.role == Role::System) {
            return Err(bad_request("request has no message with content"));
        }
        assert!(
            out.messages.iter().all(|m| !m.parts.is_empty()),
            "kept messages have parts"
        );
        Ok(out)
    }

    /// Joins `system` and the text of every system-role message, in order,
    /// with blank lines between them. Every protocol sends this one text.
    /// Returns `None` when there is no system text.
    pub fn system_text(&self) -> Option<String> {
        let parts: Vec<String> = self
            .system
            .iter()
            .cloned()
            .chain(
                self.messages
                    .iter()
                    .filter(|m| m.role == Role::System)
                    .map(Message::text_content),
            )
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }
}

/// Why a model stopped producing output.
/// Vendor reasons are normalized; unknown ones are kept as `Other`.
/// `PauseTurn` means a server-side tool loop paused: send the reply back
/// unchanged to continue. Serializes in snake case.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    MaxTokens,
    Refusal,
    PauseTurn,
    Other(String),
}

/// The finished reply to one `CompletionRequest`: the assistant message,
/// token usage (cost filled in by the cost meter) and the stop reason.
/// Unary and streamed calls return the same value for the same reply.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub message: Message,
    pub usage: Usage,
    pub stop: StopReason,
}

/// One piece of a reply as it arrives. Tool calls and hosted tool blocks
/// are keyed by `index`, and their JSON arguments arrive as fragments;
/// usage values are running totals. Serializes with a `type` tag.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Delta {
    TextDelta {
        text: String,
    },
    ToolCallStarted {
        index: u32,
        id: String,
        name: String,
    },
    ToolCallArgs {
        index: u32,
        json: String,
    },
    Hosted {
        index: u32,
        tool: Option<String>,
        block: Value,
    },
    Usage {
        input_tokens: Option<u64>,
        output_tokens: Option<u64>,
    },
    Stop {
        reason: StopReason,
    },
}
