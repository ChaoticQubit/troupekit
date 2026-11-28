use crate::gbnf::{literal, schema_to_gbnf, GbnfBuilder};
use crate::{Grammar, LocalError};
use serde_json::Value;
use troupekit_core::tool::ToolSpec;

const MAX_TOOLS: usize = 64;
const TEXT_RULE: &str = r"[^{ \t\r\n\x00] [^\x00]*";

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
    turn_prompt(tools, mode, None)
}

/// Builds the tool prompt for a `LocalChat` turn. With an output schema
/// in `Auto`, the no-tool reply is JSON of that schema instead of plain
/// text, matching `request_grammar`. Empty for no tools.
pub(crate) fn turn_prompt(tools: &[ToolSpec], mode: ToolMode, output: Option<&Value>) -> String {
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
        ToolMode::Auto => match output {
            None => out.push_str("If no tool is needed, reply in plain text.\n"),
            Some(schema) => out.push_str(&format!(
                "If no tool is needed, reply with only JSON that matches this schema: {schema}\n"
            )),
        },
        ToolMode::Required => out.push_str("You must call exactly one tool.\n"),
    }
    out
}

/// Builds the grammar for a turn with tools: one root branch per tool,
/// `{"tool":"<name>","arguments":<schema>}`, plus a plain-text branch in
/// `ToolMode::Auto`. Errors: no tools, more than 64, a schema whose root
/// is not `type: "object"`, or one the converter refuses.
pub fn tool_grammar(tools: &[ToolSpec], mode: ToolMode) -> Result<Grammar, LocalError> {
    turn_grammar(tools, mode, None)
}

/// Builds the grammar for `output_schema`: generation must produce one
/// JSON value of the schema. Input: the schema. Output: the grammar, or
/// `LocalError::Grammar` for a schema the converter refuses.
pub fn output_grammar(schema: &Value) -> Result<Grammar, LocalError> {
    schema_to_gbnf(schema)
}

/// Builds the grammar a `LocalChat` turn uses: tool branches and, in
/// `Auto`, the output schema (when given) or plain text as the last
/// branch. Output: `None` when there is neither a tool nor a schema.
pub(crate) fn request_grammar(
    tools: &[ToolSpec],
    mode: ToolMode,
    output: Option<&Value>,
) -> Result<Option<Grammar>, LocalError> {
    match (tools.is_empty(), output) {
        (true, None) => Ok(None),
        (true, Some(schema)) => output_grammar(schema).map(Some),
        (false, _) => turn_grammar(tools, mode, output).map(Some),
    }
}

fn turn_grammar(
    tools: &[ToolSpec],
    mode: ToolMode,
    output: Option<&Value>,
) -> Result<Grammar, LocalError> {
    if tools.is_empty() {
        return Err(LocalError::Grammar(
            "a tool grammar needs at least one tool".into(),
        ));
    }
    if tools.len() > MAX_TOOLS {
        return Err(LocalError::Grammar(format!(
            "{} tools offered, a local tool grammar takes at most {MAX_TOOLS}",
            tools.len()
        )));
    }
    let mut builder = GbnfBuilder::new();
    let mut branches = Vec::with_capacity(tools.len() + 1);
    for tool in tools {
        let fail = |e: LocalError| LocalError::Grammar(format!("tool {:?}: {e}", tool.name));
        if tool.schema.get("type").and_then(Value::as_str) != Some("object") {
            return Err(fail(LocalError::Grammar(
                "its schema must have `type: \"object\"` at the root".into(),
            )));
        }
        let arguments = builder.schema(&tool.schema).map_err(fail)?;
        let name = serde_json::to_string(&tool.name).expect("a string serializes");
        let body = format!(
            r#""{{" ws "\"tool\"" ws ":" ws {} ws "," ws "\"arguments\"" ws ":" ws {arguments} ws "}}""#,
            literal(&name)
        );
        branches.push(builder.rule(body)?);
    }
    if mode == ToolMode::Auto {
        let last = match output {
            None => TEXT_RULE.to_string(),
            Some(schema) => {
                if may_hold_tool_key(schema) {
                    return Err(LocalError::Grammar(
                        "output_schema may produce a top-level `tool` key (root `properties`, `const`, `enum`, `$ref`, `anyOf` or `oneOf`), which a local reply would read as a tool call".into(),
                    ));
                }
                builder.schema(schema)?
            }
        };
        branches.push(builder.rule(last)?);
    }
    assert_eq!(
        branches.len(),
        tools.len() + usize::from(mode == ToolMode::Auto),
        "one branch per tool, plus the Auto branch"
    );
    builder.finish(&branches.join(" | "))
}

/// Tells whether a value of `schema` may be an object with a top-level
/// `tool` key. Root `$ref`, `anyOf` and `oneOf` count as may, since
/// their targets are not walked. Input: the output schema.
fn may_hold_tool_key(schema: &Value) -> bool {
    let holds = |value: &Value| value.get("tool").is_some();
    ["$ref", "anyOf", "oneOf"]
        .iter()
        .any(|k| schema.get(k).is_some())
        || schema.get("properties").is_some_and(holds)
        || schema.get("const").is_some_and(holds)
        || schema
            .get("enum")
            .and_then(Value::as_array)
            .is_some_and(|values| values.iter().any(holds))
}
