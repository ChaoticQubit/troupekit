mod common;

use common::*;
use serde_json::json;
use troupekit_core::error::ProviderErrorKind;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_providers::request::{CompletionRequest, Delta, StopReason};
use troupekit_providers::wire::{decode_body, encode, Decoder, Protocol, Quirks};

#[test]
fn encodes_the_golden_request() {
    let body = encode(
        Protocol::Gemini,
        &Quirks::default(),
        "gemini-3.8-flash",
        &rich_request(),
        false,
    )
    .unwrap();
    assert_eq!(body, golden("gemini_request.json"));
}

#[test]
fn tool_results_need_an_earlier_call_and_take_its_latest_name() {
    let mut req = CompletionRequest::user("go", 10);
    req.messages
        .push(Message::new(Role::Tool, vec![result("call_0")]).unwrap());
    assert_eq!(
        encode(Protocol::Gemini, &Quirks::default(), "m", &req, false)
            .unwrap_err()
            .kind,
        ProviderErrorKind::BadRequest
    );

    let mut req = CompletionRequest::user("go", 10);
    req.messages
        .push(Message::new(Role::Assistant, vec![call("call_0", "calendar")]).unwrap());
    req.messages
        .push(Message::new(Role::Tool, vec![result("call_0")]).unwrap());
    req.messages
        .push(Message::new(Role::Assistant, vec![call("call_0", "weather")]).unwrap());
    let failed = ContentPart::ToolResult {
        call_id: "call_0".into(),
        content: "offline".into(),
        is_error: true,
    };
    req.messages
        .push(Message::new(Role::Tool, vec![failed]).unwrap());
    let body = encode(Protocol::Gemini, &Quirks::default(), "m", &req, false).unwrap();
    assert_eq!(
        body["contents"][2]["parts"][0]["functionResponse"]["name"],
        "calendar"
    );
    assert_eq!(
        body["contents"][4]["parts"][0]["functionResponse"],
        json!({"name": "weather", "response": {"error": "offline"}})
    );
}

#[test]
fn unary_and_stream_fold_to_the_same_reply() {
    let unary = fold_all(&decode_body(Protocol::Gemini, &golden("gemini_reply.json")).unwrap());
    let (deltas, streamed) = stream_deltas(Protocol::Gemini, &golden_text("gemini_stream.sse"));
    assert_eq!(unary, expected_reply("call_0"));
    assert_eq!(streamed, unary);
    assert!(deltas.contains(&Delta::TextDelta {
        text: "Mon: run ".into()
    }));
}

#[test]
fn keeps_wire_call_ids_and_numbers_the_rest() {
    let body = json!({"candidates":[{"content":{"parts":[
        {"functionCall":{"id":"fc-9","name":"a","args":{}}},
        {"functionCall":{"name":"b"}}
    ]},"finishReason":"STOP"}]});
    let reply = fold_all(&decode_body(Protocol::Gemini, &body).unwrap());
    let ids: Vec<&str> = reply.message.tool_calls().map(|(id, _, _)| id).collect();
    assert_eq!(ids, ["fc-9", "call_1"]);
    assert_eq!(reply.stop, StopReason::ToolUse);
}

#[test]
fn maps_errors_blocks_and_finish_reasons() {
    let mut decoder = Decoder::new(Protocol::Gemini);
    let err = decoder
        .event(&json!({"error":{"code":429,"message":"quota","status":"RESOURCE_EXHAUSTED"}}))
        .unwrap_err();
    assert_eq!(
        (err.kind, err.message.as_str()),
        (ProviderErrorKind::RateLimited, "quota")
    );
    let blocked =
        json!({"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":4}});
    assert_eq!(
        fold_all(&decode_body(Protocol::Gemini, &blocked).unwrap()).stop,
        StopReason::Refusal
    );
    let table = [
        ("MAX_TOKENS", StopReason::MaxTokens),
        ("SAFETY", StopReason::Refusal),
        ("OTHER", StopReason::Other("OTHER".into())),
    ];
    for (wire, want) in table {
        let body = json!({"candidates":[{"content":{"parts":[{"text":"x"}]},"finishReason":wire}]});
        assert_eq!(
            fold_all(&decode_body(Protocol::Gemini, &body).unwrap()).stop,
            want,
            "{wire}"
        );
    }
}
