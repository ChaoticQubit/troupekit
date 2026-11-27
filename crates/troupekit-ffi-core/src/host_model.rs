use crate::error::{ErrorCode, FfiError};
use crate::host::{HostModel, TokenSink, MAX_QUEUED_DELTAS};
use crate::records::{HostMessage, HostModelCaps, HostReply, HostRequest};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_providers::model::MAX_REPLY_BYTES;
use troupekit_providers::{CompletionRequest, CompletionResponse, Delta, Model, StopReason};

/// Starts the message of every host model failure, so the run's
/// `Failed` reason maps to `ErrorCode::HostCallback`.
pub(crate) const HOST_CALLBACK: &str = "host model callback failed: ";

const REPAIR: &str = "Your last reply was not valid. Reply again with only one JSON object: \
{\"tool\": \"<name>\", \"arguments\": {...}} to call a tool, or {\"answer\": ...} to answer.";

/// Wraps host model `host` as a kit model, reading its id once.
/// Output: the model and its id, or `InvalidArgument` for an empty or
/// over-long id. Each request reads `capabilities` again.
pub(crate) fn wrap_model(host: Arc<dyn HostModel>) -> Result<(Arc<dyn Model>, String), FfiError> {
    let id = host.id();
    if id.trim().is_empty() || id.len() > 256 {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "host model id must be 1 to 256 bytes",
        ));
    }
    assert!(!id.is_empty());
    Ok((
        Arc::new(HostModelAdapter {
            host,
            id: id.clone(),
        }),
        id,
    ))
}

struct HostModelAdapter {
    host: Arc<dyn HostModel>,
    id: String,
}

#[async_trait]
impl Model for HostModelAdapter {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        self.stream(req, &mut |_| {}).await
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        req.validate()?;
        let req = req.sendable(false)?;
        let caps = self.capabilities().await?;
        let emulate = !caps.tools && !req.tools.is_empty();
        let mut request = render(&req, &caps)?;
        let first = self.generate(request.clone(), !emulate, on_delta).await?;
        let mut usage = usage_of(&first);
        let parts = match parse(&req, &caps, &first) {
            Some(parts) => parts,
            None if emulate => {
                request
                    .messages
                    .push(host_message("assistant", &first.text));
                request.messages.push(host_message("user", REPAIR));
                let second = self.generate(request, false, on_delta).await?;
                usage.input_tokens += second.input_tokens;
                usage.output_tokens += second.output_tokens;
                parse(&req, &caps, &second).ok_or_else(|| {
                    decode("host model reply is not the tool JSON after one repair")
                })?
            }
            None => return Err(decode("host model tool_calls_json is not a list of calls")),
        };
        let called = parts
            .iter()
            .any(|p| matches!(p, ContentPart::ToolCall { .. }));
        if emulate {
            for part in &parts {
                if let ContentPart::Text { text } = part {
                    on_delta(&Delta::TextDelta { text: text.clone() });
                }
            }
        }
        assert!(!parts.is_empty());
        Ok(CompletionResponse {
            message: Message {
                role: Role::Assistant,
                parts,
            },
            usage,
            stop: if called {
                StopReason::ToolUse
            } else {
                StopReason::EndTurn
            },
        })
    }
}

impl HostModelAdapter {
    async fn capabilities(&self) -> Result<HostModelCaps, ProviderError> {
        let host = self.host.clone();
        let read = tokio::task::spawn_blocking(move || {
            catch_unwind(AssertUnwindSafe(|| host.capabilities()))
        })
        .await;
        match read {
            Ok(Ok(caps)) => Ok(caps),
            _ => Err(failed(format!(
                "host model `{}` panicked in capabilities",
                self.id
            ))),
        }
    }

    async fn generate(
        &self,
        request: HostRequest,
        forward: bool,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<HostReply, ProviderError> {
        let (sink, mut deltas) = TokenSink::channel(forward, MAX_REPLY_BYTES, MAX_QUEUED_DELTAS);
        let host = self.host.clone();
        let mut job = tokio::task::spawn_blocking(move || {
            catch_unwind(AssertUnwindSafe(|| host.generate(request, sink)))
        });
        let joined = loop {
            tokio::select! {
                Some(text) = deltas.recv() => on_delta(&Delta::TextDelta { text }),
                joined = &mut job => break joined,
            }
        };
        while let Ok(text) = deltas.try_recv() {
            on_delta(&Delta::TextDelta { text });
        }
        let reply = match joined {
            Ok(Ok(Ok(reply))) => reply,
            Ok(Ok(Err(error))) => return Err(failed(error.message)),
            Ok(Err(_)) | Err(_) => return Err(failed(format!("`{}` panicked", self.id))),
        };
        let size = reply.text.len() + reply.tool_calls_json.as_ref().map_or(0, String::len);
        if size > MAX_REPLY_BYTES {
            return Err(decode("host model reply is over the reply cap"));
        }
        Ok(reply)
    }
}

fn failed(message: String) -> ProviderError {
    ProviderError::new(
        ProviderErrorKind::BadRequest,
        format!("{HOST_CALLBACK}{message}"),
    )
}

fn decode(message: &str) -> ProviderError {
    ProviderError::new(ProviderErrorKind::Decode, message)
}

fn usage_of(reply: &HostReply) -> Usage {
    Usage {
        input_tokens: reply.input_tokens,
        output_tokens: reply.output_tokens,
        cost_usd_micros: 0,
    }
}

fn host_message(role: &str, text: &str) -> HostMessage {
    let mut message = HostMessage::default();
    message.role = role.into();
    message.text = text.into();
    message
}

/// Renders kit request `req` for a host model with `caps`: native tools
/// as `tools_json`, else the tool list and the JSON reply shape in the
/// system text, with that shape as `schema_json` when supported.
/// Output: the host request, or `BadRequest` for a non-text part.
pub(crate) fn render(
    req: &CompletionRequest,
    caps: &HostModelCaps,
) -> Result<HostRequest, ProviderError> {
    let emulate = !caps.tools && !req.tools.is_empty();
    let mut messages = Vec::with_capacity(req.messages.len());
    for message in req.messages.iter().filter(|m| m.role != Role::System) {
        render_message(message, caps.tools, emulate, &mut messages)?;
    }
    let mut system: Vec<String> = req.system_text().into_iter().collect();
    let mut request = HostRequest::default();
    let shape = if emulate {
        system.push(tool_prompt(req));
        Some(envelope(req))
    } else {
        req.output_schema.clone()
    };
    if caps.tools && !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({"name": t.name, "description": t.description, "schema": t.schema}))
            .collect();
        request.tools_json = Some(Value::Array(tools).to_string());
    }
    match shape {
        Some(shape) if caps.json_schema => request.schema_json = Some(shape.to_string()),
        Some(shape) => system.push(format!(
            "Reply with only JSON that matches this schema: {shape}"
        )),
        None => {}
    }
    request.system = (!system.is_empty()).then(|| system.join("\n\n"));
    request.messages = messages;
    request.max_tokens = req.max_tokens;
    assert!(request.tools_json.is_none() || !emulate);
    Ok(request)
}

fn render_message(
    message: &Message,
    native: bool,
    emulate: bool,
    out: &mut Vec<HostMessage>,
) -> Result<(), ProviderError> {
    let mut text = Vec::new();
    let mut calls = Vec::new();
    for part in &message.parts {
        match part {
            ContentPart::Text { text: t } => text.push(t.clone()),
            ContentPart::Image { .. } => {
                return Err(ProviderError::new(
                    ProviderErrorKind::BadRequest,
                    "host models take text only",
                ))
            }
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => calls.push(json!({"id": id, "name": name, "arguments": arguments})),
            ContentPart::ToolResult {
                call_id,
                content,
                is_error,
            } => {
                let content = if *is_error {
                    format!("Error: {content}")
                } else {
                    content.clone()
                };
                let mut result = if native {
                    host_message("tool", &content)
                } else {
                    host_message(
                        "user",
                        &format!("Result of tool call {call_id}:\n{content}"),
                    )
                };
                if native {
                    result.tool_call_id = Some(call_id.clone());
                }
                out.push(result);
            }
            ContentPart::Hosted { .. } => {}
        }
    }
    let role = match message.role {
        Role::User => "user",
        Role::Assistant => "assistant",
        Role::Tool | Role::System => return Ok(()),
    };
    let mut rendered = host_message(role, &text.join("\n"));
    if native && !calls.is_empty() {
        rendered.tool_calls_json = Some(Value::Array(calls).to_string());
    } else if !calls.is_empty() {
        let envelopes: Vec<String> = calls
            .iter()
            .map(|c| json!({"tool": c["name"], "arguments": c["arguments"]}).to_string())
            .collect();
        rendered.text = envelopes.join("\n");
    } else if role == "assistant" && emulate {
        rendered.text = json!({"answer": rendered.text}).to_string();
    }
    assert!(rendered.tool_calls_json.is_none() || native);
    out.push(rendered);
    Ok(())
}

fn tool_prompt(req: &CompletionRequest) -> String {
    let mut prompt = String::from(
        "You can call tools. To call one, reply with only the JSON \
{\"tool\": \"<name>\", \"arguments\": {...}}. To answer, reply with only the JSON \
{\"answer\": <answer>}. Tools:",
    );
    for tool in &req.tools {
        prompt.push_str(&format!(
            "\n- {}: {}\n  arguments schema: {}",
            tool.name, tool.description, tool.schema
        ));
    }
    prompt
}

const ANSWER_POINTER: &str = "#/oneOf/0/properties/answer";
const DATA_KEYWORDS: [&str; 5] = ["$ref", "enum", "const", "default", "examples"];
const NAMED_SCHEMAS: [&str; 5] = [
    "properties",
    "patternProperties",
    "$defs",
    "definitions",
    "dependentSchemas",
];

fn envelope(req: &CompletionRequest) -> Value {
    let mut answer = req
        .output_schema
        .clone()
        .unwrap_or_else(|| json!({"type": "string"}));
    rebase_refs(&mut answer);
    let names: Vec<&str> = req.tools.iter().map(|t| t.name.as_str()).collect();
    json!({"oneOf": [
        {"type": "object", "properties": {"answer": answer}, "required": ["answer"], "additionalProperties": false},
        {"type": "object", "properties": {"tool": {"type": "string", "enum": names}, "arguments": {"type": "object"}},
         "required": ["tool", "arguments"], "additionalProperties": false}
    ]})
}

/// Rewrites every local pointer `$ref` (`#` or `#/...`) in `schema` to
/// resolve from the envelope root, where the answer schema now sits at
/// `ANSWER_POINTER`. A subschema with its own `$id` is its own resource,
/// so it and everything under it are left as they are. Anchors and
/// instance data (`enum`, `const`, `default`, `examples`) stay; maps
/// of named subschemas such as `properties` are walked by value.
fn rebase_refs(schema: &mut Value) {
    match schema {
        Value::Object(map) => {
            if map.contains_key("$id") {
                return;
            }
            if let Some(Value::String(target)) = map.get_mut("$ref") {
                if target == "#" || target.starts_with("#/") {
                    let rest = &target[1..];
                    *target = format!("{ANSWER_POINTER}{rest}");
                    assert!(target.starts_with(ANSWER_POINTER));
                }
            }
            for (key, value) in map.iter_mut() {
                let key = key.as_str();
                if DATA_KEYWORDS.contains(&key) {
                    continue;
                }
                match value {
                    Value::Object(named) if NAMED_SCHEMAS.contains(&key) => {
                        named.values_mut().for_each(rebase_refs);
                    }
                    _ => rebase_refs(value),
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                rebase_refs(item);
            }
        }
        _ => {}
    }
}

/// Turns host `reply` into message parts: the emulated JSON envelope
/// when tools are emulated, else the text and any native tool calls.
/// Call ids count prior calls in `req`, so a replayed step gets the
/// same ids. Output: the parts, or `None` when the reply does not parse.
pub(crate) fn parse(
    req: &CompletionRequest,
    caps: &HostModelCaps,
    reply: &HostReply,
) -> Option<Vec<ContentPart>> {
    let prior = req
        .messages
        .iter()
        .flat_map(|m| &m.parts)
        .filter(|p| matches!(p, ContentPart::ToolCall { .. }))
        .count();
    if !caps.tools && !req.tools.is_empty() {
        return parse_envelope(req, &reply.text, prior).map(|part| vec![part]);
    }
    let mut parts = Vec::new();
    if !reply.text.is_empty() {
        parts.push(ContentPart::Text {
            text: reply.text.clone(),
        });
    }
    if let Some(calls) = &reply.tool_calls_json {
        let Ok(Value::Array(calls)) = serde_json::from_str::<Value>(calls) else {
            return None;
        };
        for (index, call) in calls.iter().enumerate() {
            let name = call.get("name")?.as_str()?.to_string();
            let arguments = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
            let id = match call.get("id").and_then(Value::as_str) {
                Some(id) if !id.is_empty() => id.to_string(),
                _ => format!("call_{}", prior + index),
            };
            parts.push(ContentPart::ToolCall {
                id,
                name,
                arguments,
            });
        }
    }
    if parts.is_empty() {
        parts.push(ContentPart::Text {
            text: String::new(),
        });
    }
    Some(parts)
}

fn parse_envelope(req: &CompletionRequest, text: &str, prior: usize) -> Option<ContentPart> {
    let body = text.trim();
    let body = body
        .strip_prefix("```json")
        .or_else(|| body.strip_prefix("```"))
        .and_then(|b| b.strip_suffix("```"))
        .unwrap_or(body)
        .trim();
    let Value::Object(object) = serde_json::from_str::<Value>(body).ok()? else {
        return None;
    };
    if let Some(name) = object.get("tool").and_then(Value::as_str) {
        if !req.tools.iter().any(|t| t.name == name) {
            return None;
        }
        let arguments = object
            .get("arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        if !arguments.is_object() {
            return None;
        }
        return Some(ContentPart::ToolCall {
            id: format!("call_{prior}"),
            name: name.to_string(),
            arguments,
        });
    }
    match object.get("answer")? {
        Value::String(answer) => Some(ContentPart::Text {
            text: answer.clone(),
        }),
        other => Some(ContentPart::Text {
            text: other.to_string(),
        }),
    }
}
