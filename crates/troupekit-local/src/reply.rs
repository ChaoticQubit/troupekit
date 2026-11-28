use crate::error::LocalError;
use serde_json::Value;
use troupekit_core::message::ContentPart;
use troupekit_core::tool::ToolSpec;

/// The largest local reply `parse_reply` reads, in bytes (1 MiB).
/// Longer replies are refused before any JSON parsing.
/// Bounds memory on a runaway generation.
pub const REPLY_MAX_BYTES: usize = 1024 * 1024;

/// Turns a local model's reply into message parts. With tools, a reply
/// that is exactly `{"tool":"<name>","arguments":{..}}` becomes one
/// `ToolCall` with id `local-1`; anything else is one `Text` part.
/// Errors: over 1 MiB, an unknown tool, or a malformed call object.
pub fn parse_reply(text: &str, tools: &[ToolSpec]) -> Result<Vec<ContentPart>, LocalError> {
    if text.len() > REPLY_MAX_BYTES {
        return Err(LocalError::Decode(format!(
            "local reply is {} bytes, limit {REPLY_MAX_BYTES}",
            text.len()
        )));
    }
    let text_part = || {
        vec![ContentPart::Text {
            text: text.to_string(),
        }]
    };
    if tools.is_empty() || !text.trim_start().starts_with('{') {
        return Ok(text_part());
    }
    let Ok(Value::Object(map)) = serde_json::from_str::<Value>(text.trim()) else {
        return Ok(text_part());
    };
    let Some(tool) = map.get("tool") else {
        return Ok(text_part());
    };
    let Value::String(name) = tool else {
        return Err(LocalError::Decode(
            "local tool call has a non-string tool name".into(),
        ));
    };
    if !tools.iter().any(|t| &t.name == name) {
        return Err(LocalError::Decode(format!(
            "local reply calls unknown tool {name:?}"
        )));
    }
    if map.len() != 2 {
        return Err(LocalError::Decode(
            "local tool call must hold only tool and arguments".into(),
        ));
    }
    let Some(arguments @ Value::Object(_)) = map.get("arguments") else {
        return Err(LocalError::Decode(format!(
            "local call to {name:?} has no arguments object"
        )));
    };
    Ok(vec![ContentPart::ToolCall {
        id: "local-1".to_string(),
        name: name.clone(),
        arguments: arguments.clone(),
    }])
}
