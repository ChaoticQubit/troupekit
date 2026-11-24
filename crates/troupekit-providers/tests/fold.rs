use serde_json::json;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, Role};
use troupekit_providers::fold::Fold;
use troupekit_providers::request::{CompletionResponse, Delta, StopReason};

fn fold(deltas: &[Delta]) -> Result<CompletionResponse, ProviderError> {
    let mut f = Fold::new();
    for d in deltas {
        f.push(d)?;
    }
    f.finish()
}

fn text(t: &str) -> Delta {
    Delta::TextDelta { text: t.into() }
}

fn stop(reason: StopReason) -> Delta {
    Delta::Stop { reason }
}

fn started(index: u32, id: &str, name: &str) -> Delta {
    Delta::ToolCallStarted {
        index,
        id: id.into(),
        name: name.into(),
    }
}

fn args(index: u32, json: &str) -> Delta {
    Delta::ToolCallArgs {
        index,
        json: json.into(),
    }
}

#[test]
fn joins_text_and_keeps_part_order() {
    let r = fold(&[
        text("Mon: "),
        text("run"),
        started(4, "c1", "calendar"),
        args(4, "{\"days\":"),
        args(4, "3}"),
        started(7, "c2", "now"),
        Delta::Usage {
            input_tokens: Some(5),
            output_tokens: Some(1),
        },
        Delta::Usage {
            input_tokens: None,
            output_tokens: Some(9),
        },
        stop(StopReason::EndTurn),
    ])
    .unwrap();
    assert_eq!(r.message.role, Role::Assistant);
    assert_eq!(
        r.message.parts,
        vec![
            ContentPart::Text {
                text: "Mon: run".into()
            },
            ContentPart::ToolCall {
                id: "c1".into(),
                name: "calendar".into(),
                arguments: json!({"days":3})
            },
            ContentPart::ToolCall {
                id: "c2".into(),
                name: "now".into(),
                arguments: json!({})
            },
        ]
    );
    assert_eq!(
        r.usage,
        Usage {
            input_tokens: 5,
            output_tokens: 9,
            cost_usd_micros: 0
        }
    );
    assert_eq!(r.stop, StopReason::ToolUse);
}

#[test]
fn text_after_a_tool_call_starts_a_new_part() {
    let r = fold(&[
        text("a"),
        started(0, "c", "n"),
        text("b"),
        stop(StopReason::ToolUse),
    ])
    .unwrap();
    assert_eq!(r.message.parts.len(), 3);
    assert_eq!(r.message.text_content(), "a\nb");
}

#[test]
fn empty_reply_is_one_empty_text_part() {
    let r = fold(&[stop(StopReason::MaxTokens)]).unwrap();
    assert_eq!(
        r.message.parts,
        vec![ContentPart::Text {
            text: String::new()
        }]
    );
    assert_eq!(r.stop, StopReason::MaxTokens);
}

#[test]
fn rejects_broken_delta_sequences() {
    let kind = |d: &[Delta]| fold(d).unwrap_err().kind;
    assert_eq!(kind(&[text("x")]), ProviderErrorKind::Network);
    assert_eq!(
        kind(&[args(1, "{}"), stop(StopReason::EndTurn)]),
        ProviderErrorKind::Decode
    );
    assert_eq!(
        kind(&[
            started(0, "c", "n"),
            started(0, "c", "n"),
            stop(StopReason::EndTurn)
        ]),
        ProviderErrorKind::Decode
    );
    assert_eq!(
        kind(&[
            started(0, "c", "n"),
            args(0, "{nope"),
            stop(StopReason::EndTurn)
        ]),
        ProviderErrorKind::Decode
    );
    assert_eq!(
        kind(&[started(0, "", "n"), stop(StopReason::EndTurn)]),
        ProviderErrorKind::Decode
    );
}
