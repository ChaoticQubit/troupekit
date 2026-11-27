use crate::host::TokenSink;
use crate::host_model::{parse, render};
use crate::records::{HostModelCaps, HostReply};
use serde_json::json;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::tool::ToolSpec;
use troupekit_providers::CompletionRequest;

fn with_tool() -> CompletionRequest {
    let mut req = CompletionRequest::user("1 + 2", 100);
    req.tools.push(ToolSpec {
        name: "add".into(),
        description: "Adds.".into(),
        schema: json!({"type": "object"}),
        hosted: None,
    });
    req
}

fn text_reply(text: &str) -> HostReply {
    let mut reply = HostReply::default();
    reply.text = text.into();
    reply
}

#[test]
fn delta_bytes_capped() {
    let (sink, mut deltas) = TokenSink::channel(true, 10, 8);
    sink.delta("12345".into());
    sink.delta("123456".into());
    sink.delta("12345".into());
    sink.delta("1".into());
    let mut got = Vec::new();
    while let Ok(text) = deltas.try_recv() {
        got.push(text);
    }
    assert_eq!(got, vec!["12345".to_string(), "12345".to_string()]);
    let (off, mut none) = TokenSink::channel(false, 10, 8);
    off.delta("1".into());
    assert!(none.try_recv().is_err());
}

#[test]
fn empty_deltas_ignored_and_queue_bounded() {
    let (sink, mut deltas) = TokenSink::channel(true, 100, 2);
    sink.delta(String::new());
    sink.delta("a".into());
    sink.delta(String::new());
    sink.delta("b".into());
    sink.delta("c".into());
    let mut got = Vec::new();
    while let Ok(text) = deltas.try_recv() {
        got.push(text);
    }
    assert_eq!(got, vec!["a".to_string(), "b".to_string()]);
    sink.delta("d".into());
    assert_eq!(deltas.try_recv().unwrap(), "d");
}

#[test]
fn output_schema_without_tools_goes_to_schema_or_prompt() {
    let mut req = CompletionRequest::user("hi", 100);
    req.output_schema = Some(json!({"type": "object", "properties": {"n": {"type": "integer"}}}));
    let mut caps = HostModelCaps::default();
    let prompted = render(&req, &caps).unwrap();
    assert!(prompted.schema_json.is_none());
    assert!(prompted.system.unwrap().contains("\"integer\""));
    caps.json_schema = true;
    let schema = render(&req, &caps).unwrap();
    let shape: serde_json::Value = serde_json::from_str(&schema.schema_json.unwrap()).unwrap();
    assert_eq!(shape["properties"]["n"]["type"], "integer");
}

#[test]
fn output_schema_with_emulated_tools_types_the_answer() {
    let mut req = with_tool();
    req.output_schema = Some(json!({"type": "object", "properties": {"n": {"type": "integer"}}}));
    let mut caps = HostModelCaps::default();
    caps.json_schema = true;
    let rendered = render(&req, &caps).unwrap();
    let shape: serde_json::Value = serde_json::from_str(&rendered.schema_json.unwrap()).unwrap();
    assert_eq!(
        shape["oneOf"][0]["properties"]["answer"]["properties"]["n"]["type"],
        "integer"
    );
    let parts = parse(&req, &caps, &text_reply(r#"{"answer": {"n": 3}}"#)).unwrap();
    assert_eq!(
        parts,
        vec![ContentPart::Text {
            text: r#"{"n":3}"#.into()
        }]
    );
}

#[test]
fn envelope_keeps_local_refs_of_the_answer_schema() {
    let mut req = with_tool();
    req.output_schema = Some(json!({
        "$defs": {"N": {"type": "integer"}},
        "type": "object",
        "properties": {
            "n": {"$ref": "#/$defs/N"},
            "me": {"$ref": "#"},
            "tag": {"$ref": "#t"},
            "default": {"$ref": "#/$defs/N"},
        },
    }));
    let mut caps = HostModelCaps::default();
    caps.json_schema = true;
    let rendered = render(&req, &caps).unwrap();
    let shape: serde_json::Value = serde_json::from_str(&rendered.schema_json.unwrap()).unwrap();
    let answer = &shape["oneOf"][0]["properties"]["answer"];
    let target = answer["properties"]["n"]["$ref"].as_str().unwrap();
    let resolved = shape.pointer(target.strip_prefix('#').unwrap()).unwrap();
    assert_eq!(resolved["type"], "integer");
    let me = answer["properties"]["me"]["$ref"].as_str().unwrap();
    assert_eq!(
        shape.pointer(me.strip_prefix('#').unwrap()).unwrap(),
        answer
    );
    assert_eq!(answer["properties"]["tag"]["$ref"], "#t");
    assert_eq!(answer["properties"]["default"]["$ref"], target);
}

#[test]
fn envelope_leaves_refs_of_an_own_id_schema() {
    let mut req = with_tool();
    req.output_schema = Some(json!({
        "$id": "https://example.test/out",
        "$defs": {"N": {"type": "integer"}},
        "properties": {"n": {"$ref": "#/$defs/N"}},
    }));
    let mut caps = HostModelCaps::default();
    caps.json_schema = true;
    let rendered = render(&req, &caps).unwrap();
    let shape: serde_json::Value = serde_json::from_str(&rendered.schema_json.unwrap()).unwrap();
    let answer = &shape["oneOf"][0]["properties"]["answer"];
    assert_eq!(answer["properties"]["n"]["$ref"], "#/$defs/N");
}

#[test]
fn output_schema_with_native_tools_sends_both() {
    let mut req = with_tool();
    req.output_schema = Some(json!({"type": "object"}));
    let mut caps = HostModelCaps::default();
    caps.tools = true;
    caps.json_schema = true;
    let rendered = render(&req, &caps).unwrap();
    assert!(rendered.tools_json.is_some());
    assert!(rendered.schema_json.is_some());
}

#[test]
fn emulated_call_ids_count_prior_calls() {
    let mut req = with_tool();
    let call = ContentPart::ToolCall {
        id: "call_0".into(),
        name: "add".into(),
        arguments: json!({}),
    };
    req.messages
        .push(Message::new(Role::Assistant, vec![call]).unwrap());
    let caps = HostModelCaps::default();
    let reply = text_reply(r#"{"tool": "add", "arguments": {}}"#);
    let first = parse(&req, &caps, &reply).unwrap();
    let again = parse(&req, &caps, &reply).unwrap();
    assert_eq!(first, again);
    assert!(matches!(&first[0], ContentPart::ToolCall { id, .. } if id == "call_1"));
}

#[test]
fn unknown_tool_or_bad_shape_does_not_parse() {
    let req = with_tool();
    let caps = HostModelCaps::default();
    for text in [
        "",
        "3",
        "[1]",
        r#"{"tool": "sub", "arguments": {}}"#,
        r#"{"tool": "add", "arguments": 1}"#,
        r#"{"other": 1}"#,
    ] {
        assert!(parse(&req, &caps, &text_reply(text)).is_none(), "{text}");
    }
}

#[test]
fn image_part_is_refused() {
    let mut req = CompletionRequest::user("look", 100);
    req.messages[0].parts.push(ContentPart::Image {
        media_type: "image/png".into(),
        source: troupekit_core::message::MediaSource::Base64 {
            data: "AA==".into(),
        },
    });
    let err = render(&req, &HostModelCaps::default()).unwrap_err();
    assert!(err.message.contains("text only"), "{err}");
}

#[test]
fn history_without_tools_in_request_stays_text_for_a_plain_model() {
    let mut req = CompletionRequest::user("1 + 2", 100);
    let call = ContentPart::ToolCall {
        id: "call_0".into(),
        name: "add".into(),
        arguments: json!({"a": 1}),
    };
    req.messages
        .push(Message::new(Role::Assistant, vec![call]).unwrap());
    let result = ContentPart::ToolResult {
        call_id: "call_0".into(),
        content: "3".into(),
        is_error: false,
    };
    req.messages
        .push(Message::new(Role::Tool, vec![result]).unwrap());
    let rendered = render(&req, &HostModelCaps::default()).unwrap();
    assert!(rendered
        .messages
        .iter()
        .all(|m| m.tool_calls_json.is_none()));
    assert!(rendered.messages.iter().all(|m| m.tool_call_id.is_none()));
    assert!(rendered.messages.iter().all(|m| m.role != "tool"));
    assert!(rendered.messages[1].text.contains(r#""tool":"add""#));
    assert!(rendered.messages[2].text.contains('3'));
}
