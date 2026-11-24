use serde_json::json;
use troupekit_core::error::ProviderErrorKind;
use troupekit_core::message::{Message, Role};
use troupekit_core::tool::ToolSpec;
use troupekit_providers::request::{CompletionRequest, Delta, StopReason};

fn tool(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: "d".into(),
        schema: json!({"type":"object"}),
        hosted: None,
    }
}

#[test]
fn validates_requests() {
    assert!(CompletionRequest::user("hi", 10).validate().is_ok());
    let cases: [fn(&mut CompletionRequest); 6] = [
        |r| r.messages.clear(),
        |r| r.max_tokens = 0,
        |r| r.temperature = Some(2.5),
        |r| r.temperature = Some(f64::NAN),
        |r| r.tools = vec![tool(" ")],
        |r| r.tools = vec![tool("a"), tool("a")],
    ];
    for (i, change) in cases.iter().enumerate() {
        let mut r = CompletionRequest::user("hi", 10);
        change(&mut r);
        assert_eq!(
            r.validate().unwrap_err().kind,
            ProviderErrorKind::BadRequest,
            "case {i}"
        );
    }
}

#[test]
fn system_text_joins_field_and_system_messages() {
    let mut r = CompletionRequest::user("hi", 10);
    assert_eq!(r.system_text(), None);
    r.system = Some("Be brief.".into());
    r.messages
        .insert(0, Message::text(Role::System, "Use metric units."));
    assert_eq!(
        r.system_text().as_deref(),
        Some("Be brief.\n\nUse metric units.")
    );
}

#[test]
fn deltas_and_stops_serialize_tagged() {
    assert_eq!(
        serde_json::to_value(Delta::TextDelta { text: "a".into() }).unwrap(),
        json!({"type":"text_delta","text":"a"})
    );
    assert_eq!(
        serde_json::to_value(StopReason::ToolUse).unwrap(),
        json!("tool_use")
    );
    assert_eq!(
        serde_json::to_value(StopReason::Other("pause_turn".into())).unwrap(),
        json!({"other":"pause_turn"})
    );
}
