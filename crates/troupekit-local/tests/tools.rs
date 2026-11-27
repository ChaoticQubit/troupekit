use serde_json::json;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::tool::ToolSpec;
use troupekit_local::{
    output_grammar, parse_reply, schema_to_gbnf, tool_grammar, tool_prompt, LocalError, ToolMode,
    REPLY_MAX_BYTES,
};

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

fn root_branches(gbnf: &str) -> Vec<String> {
    let first = gbnf.lines().next().expect("a root line");
    let body = first
        .strip_prefix("root ::= ")
        .expect("the root rule comes first");
    body.split(" | ").map(str::to_string).collect()
}

fn rule_body<'a>(gbnf: &'a str, name: &str) -> &'a str {
    let prefix = format!("{name} ::= ");
    gbnf.lines()
        .find_map(|l| l.strip_prefix(prefix.as_str()))
        .unwrap_or_else(|| panic!("no rule {name} in {gbnf}"))
}

#[test]
fn tool_grammar_has_one_branch_per_tool() {
    let grammar = tool_grammar(&tools(), ToolMode::Required).expect("grammar");
    assert_eq!(grammar.root(), "root");
    let gbnf = grammar.gbnf();
    let branches = root_branches(gbnf);
    assert_eq!(branches.len(), tools().len(), "{gbnf}");
    let names = [r#""\"get_weather\"""#, r#""\"say \\\"hi\\\"\"""#];
    for (branch, name) in branches.iter().zip(names) {
        let body = rule_body(gbnf, branch);
        assert!(body.contains(r#""\"tool\"""#), "{body}");
        assert!(body.contains(r#""\"arguments\"""#), "{body}");
        assert!(body.contains(name), "{name} in {body}");
    }
}

#[test]
fn required_mode_has_no_text_branch() {
    let auto = tool_grammar(&tools(), ToolMode::Auto).expect("auto");
    let required = tool_grammar(&tools(), ToolMode::Required).expect("required");
    let auto_branches = root_branches(auto.gbnf());
    let required_branches = root_branches(required.gbnf());
    assert_eq!(auto_branches.len(), tools().len() + 1, "{}", auto.gbnf());
    assert_eq!(required_branches.len(), tools().len());
    let text = rule_body(auto.gbnf(), auto_branches.last().expect("text branch"));
    assert!(text.starts_with(r"[^{ \t\r\n\x00]"), "{text}");
    for branch in &required_branches {
        assert!(!rule_body(required.gbnf(), branch).starts_with("[^{"));
    }
}

#[test]
fn tool_grammar_refuses_tool_sets_it_cannot_express() {
    let err = tool_grammar(&[], ToolMode::Required).expect_err("no tools");
    assert!(matches!(err, LocalError::Grammar(_)), "{err:?}");
    let mut list = tools();
    list[0].schema = json!({"type":"string"});
    let err = tool_grammar(&list, ToolMode::Auto).expect_err("string arguments");
    assert!(
        matches!(&err, LocalError::Grammar(m) if m.contains("get_weather")),
        "{err:?}"
    );
    let mut list = tools();
    list[0].schema = json!({"type":"object","properties":{"q":{"type":"string","pattern":"x"}}});
    let err = tool_grammar(&list, ToolMode::Auto).expect_err("unsupported keyword");
    assert!(
        matches!(&err, LocalError::Grammar(m) if m.contains("get_weather") && m.contains("pattern")),
        "{err:?}"
    );
    let many: Vec<ToolSpec> = (0..65)
        .map(|i| ToolSpec {
            name: format!("t{i}"),
            description: String::new(),
            schema: json!({"type":"object"}),
            hosted: None,
        })
        .collect();
    let err = tool_grammar(&many, ToolMode::Auto).expect_err("too many tools");
    assert!(matches!(err, LocalError::Grammar(_)), "{err:?}");
    assert!(tool_grammar(&many[..64], ToolMode::Auto).is_ok());
}

#[test]
fn output_grammar_is_the_schema_grammar() {
    let schema =
        json!({"type":"object","properties":{"score":{"type":"integer"}},"required":["score"]});
    assert_eq!(
        output_grammar(&schema).expect("output"),
        schema_to_gbnf(&schema).expect("schema")
    );
    assert!(output_grammar(&json!({"type":"object","pattern":"x"})).is_err());
}

#[test]
fn admitted_replies_parse_into_valid_messages() {
    let samples = [
        r#"{"tool":"get_weather","arguments":{"city":"Lisbon"}}"#,
        "{ \"tool\" : \"say \\\"hi\\\"\" ,\n  \"arguments\" : { } }",
        "Sunny, 21 C. {not a call}",
    ];
    for sample in samples {
        let parts = parse_reply(sample, &tools()).expect(sample);
        assert!(Message::new(Role::Assistant, parts).is_ok(), "{sample}");
    }
}
