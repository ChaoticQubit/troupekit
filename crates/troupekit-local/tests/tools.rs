use serde_json::json;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::tool::ToolSpec;
use troupekit_local::{parse_reply, tool_prompt, LocalError, ToolMode, REPLY_MAX_BYTES};

fn tools() -> Vec<ToolSpec> {
    vec![
        ToolSpec {
            name: "get_weather".into(),
            description: "Weather for a city".into(),
            schema: json!({"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}),
            hosted: None,
        },
        ToolSpec {
            name: "say \"hi\"".into(),
            description: "A name that needs JSON escaping".into(),
            schema: json!({"type":"object"}),
            hosted: None,
        },
    ]
}

#[test]
fn parse_reply_builds_tool_call_parts() {
    let parts = parse_reply(
        r#" {"tool":"get_weather","arguments":{"city":"Lisbon"}} "#,
        &tools(),
    )
    .expect("parses");
    assert_eq!(
        parts,
        vec![ContentPart::ToolCall {
            id: "local-1".into(),
            name: "get_weather".into(),
            arguments: json!({"city":"Lisbon"}),
        }]
    );
    assert!(Message::new(Role::Assistant, parts).is_ok());
}

#[test]
fn unknown_tool_in_local_reply_is_refused() {
    let err = parse_reply(r#"{"tool":"rm_rf","arguments":{}}"#, &tools()).expect_err("unknown");
    assert!(matches!(err, LocalError::Decode(_)), "{err:?}");
    let err =
        parse_reply(r#"{"tool":"get_weather","arguments":[1]}"#, &tools()).expect_err("array args");
    assert!(matches!(err, LocalError::Decode(_)), "{err:?}");
    let err = parse_reply(
        r#"{"tool":"get_weather","arguments":{},"extra":1}"#,
        &tools(),
    )
    .expect_err("extra key");
    assert!(matches!(err, LocalError::Decode(_)), "{err:?}");
}

#[test]
fn oversized_reply_is_refused() {
    let text = "a".repeat(REPLY_MAX_BYTES + 1);
    let err = parse_reply(&text, &tools()).expect_err("too big");
    assert!(matches!(err, LocalError::Decode(_)), "{err:?}");
    assert!(parse_reply(&"a".repeat(REPLY_MAX_BYTES), &tools()).is_ok());
}

#[test]
fn plain_text_and_json_without_tools_stay_text() {
    let parts = parse_reply("The weather is fine.", &tools()).expect("text");
    assert_eq!(
        parts,
        vec![ContentPart::Text {
            text: "The weather is fine.".into()
        }]
    );
    let json_text = r#"{"tool":"x","score":3}"#;
    let parts = parse_reply(json_text, &[]).expect("structured output");
    assert_eq!(
        parts,
        vec![ContentPart::Text {
            text: json_text.into()
        }]
    );
    let parts = parse_reply(r#"{"answer":"yes"}"#, &tools()).expect("json text");
    assert!(matches!(&parts[0], ContentPart::Text { .. }));
    assert!(Message::new(Role::Assistant, parts).is_ok());
}

#[test]
fn tool_prompt_lists_names_as_json_strings() {
    let auto = tool_prompt(&tools(), ToolMode::Auto);
    assert!(auto.contains(r#""get_weather""#), "{auto}");
    assert!(auto.contains(r#""say \"hi\"""#), "{auto}");
    assert!(auto.contains(r#"{"tool":"#), "{auto}");
    let required = tool_prompt(&tools(), ToolMode::Required);
    assert_ne!(auto, required);
    assert!(tool_prompt(&[], ToolMode::Auto).is_empty());
    assert_eq!(ToolMode::default(), ToolMode::Auto);
}
