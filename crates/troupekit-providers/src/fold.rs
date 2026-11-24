use crate::request::{CompletionResponse, Delta, StopReason};
use crate::status::decode_error;
use serde_json::Value;
use std::collections::BTreeMap;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};

/// Builds one `CompletionResponse` from the deltas of one reply.
/// Unary and streamed replies both go through it, so they cannot drift.
/// Feed deltas in order with `push`, then call `finish` once.
#[derive(Debug, Default)]
pub struct Fold {
    parts: Vec<ContentPart>,
    calls: BTreeMap<u32, (usize, String)>,
    hosted: BTreeMap<u32, (usize, String)>,
    input_tokens: u64,
    output_tokens: u64,
    stop: Option<StopReason>,
}

impl Fold {
    /// Starts an empty fold for one reply.
    /// Returns the fold.
    /// Push every delta of the reply, then call `finish`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds one `delta`: text joins the previous text part, argument
    /// fragments join their call, usage overwrites the totals so far.
    /// Returns `Decode` for a call started twice or arguments for no call.
    pub fn push(&mut self, delta: &Delta) -> Result<(), ProviderError> {
        match delta {
            Delta::TextDelta { text } => {
                if text.is_empty() {
                    return Ok(());
                }
                match self.parts.last_mut() {
                    Some(ContentPart::Text { text: last }) => last.push_str(text),
                    _ => self.parts.push(ContentPart::Text { text: text.clone() }),
                }
            }
            Delta::ToolCallStarted { index, id, name } => {
                if id.is_empty() || name.is_empty() {
                    return Err(decode_error("tool call without an id or a name"));
                }
                if self.calls.contains_key(index) {
                    return Err(decode_error(format!("tool call {index} started twice")));
                }
                self.calls.insert(*index, (self.parts.len(), String::new()));
                self.parts.push(ContentPart::ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: Value::Null,
                });
            }
            Delta::Hosted { index, tool, block } => {
                if self.calls.contains_key(index) || self.hosted.contains_key(index) {
                    return Err(decode_error(format!("block {index} started twice")));
                }
                self.hosted
                    .insert(*index, (self.parts.len(), String::new()));
                self.parts.push(ContentPart::Hosted {
                    tool: tool.clone(),
                    block: block.clone(),
                });
            }
            Delta::ToolCallArgs { index, json } => match self
                .calls
                .get_mut(index)
                .or_else(|| self.hosted.get_mut(index))
            {
                Some((_, raw)) => raw.push_str(json),
                None => {
                    return Err(decode_error(format!(
                        "arguments for unknown tool call {index}"
                    )))
                }
            },
            Delta::Usage {
                input_tokens,
                output_tokens,
            } => {
                if let Some(n) = input_tokens {
                    self.input_tokens = *n;
                }
                if let Some(n) = output_tokens {
                    self.output_tokens = *n;
                }
            }
            Delta::Stop { reason } => self.stop = Some(reason.clone()),
        }
        Ok(())
    }

    /// Ends the reply and builds the response. Tool arguments are parsed
    /// as JSON (empty means `{}`), and an end-turn stop with tool calls
    /// becomes `ToolUse`. Returns `Network` when no stop reason arrived.
    pub fn finish(self) -> Result<CompletionResponse, ProviderError> {
        let Fold {
            mut parts,
            calls,
            hosted,
            input_tokens,
            output_tokens,
            stop,
        } = self;
        let Some(mut stop) = stop else {
            return Err(ProviderError::new(
                ProviderErrorKind::Network,
                "reply ended before a stop reason",
            ));
        };
        for (pos, raw) in hosted.values().filter(|(_, raw)| !raw.trim().is_empty()) {
            let parsed: Value = serde_json::from_str(raw)
                .map_err(|e| decode_error(format!("hosted tool input is not JSON: {e}")))?;
            match &mut parts[*pos] {
                ContentPart::Hosted { block, .. } => block["input"] = parsed,
                _ => unreachable!("a hosted slot always points at its hosted part"),
            }
        }
        for (pos, raw) in calls.values() {
            let parsed = if raw.trim().is_empty() {
                Value::Object(Default::default())
            } else {
                serde_json::from_str(raw)
                    .map_err(|e| decode_error(format!("tool arguments are not JSON: {e}")))?
            };
            match &mut parts[*pos] {
                ContentPart::ToolCall { arguments, .. } => *arguments = parsed,
                _ => unreachable!("a call slot always points at its tool call part"),
            }
        }
        if !calls.is_empty() && stop == StopReason::EndTurn {
            stop = StopReason::ToolUse;
        }
        if parts.is_empty() {
            parts.push(ContentPart::Text {
                text: String::new(),
            });
        }
        let message =
            Message::new(Role::Assistant, parts).map_err(|e| decode_error(e.to_string()))?;
        Ok(CompletionResponse {
            message,
            usage: Usage {
                input_tokens,
                output_tokens,
                cost_usd_micros: 0,
            },
            stop,
        })
    }
}
