use troupekit_core::tool::ToolSpec;

/// Whether a local model may answer in text when tools are offered.
/// `Auto` allows text or one tool call; `Required` allows only a call.
/// Default `Auto`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolMode {
    #[default]
    Auto,
    Required,
}

/// Builds the system prompt block that lists the tools and the reply
/// format `parse_reply` reads. Tool names and schemas are written as
/// JSON, so a name cannot break out of its quotes. Empty for no tools.
pub fn tool_prompt(tools: &[ToolSpec], mode: ToolMode) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::from("You can call these tools:\n");
    for tool in tools {
        let name = serde_json::Value::String(tool.name.clone()).to_string();
        let description = serde_json::Value::String(tool.description.clone()).to_string();
        out.push_str(&format!(
            "- name: {name}\n  description: {description}\n  arguments schema: {}\n",
            tool.schema
        ));
    }
    out.push_str("To call a tool, reply with only this JSON object and nothing else:\n");
    out.push_str("{\"tool\":\"<name>\",\"arguments\":{...}}\n");
    match mode {
        ToolMode::Auto => out.push_str("If no tool is needed, reply in plain text.\n"),
        ToolMode::Required => out.push_str("You must call exactly one tool.\n"),
    }
    out
}
