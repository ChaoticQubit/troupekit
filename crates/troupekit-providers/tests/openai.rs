mod common;

use common::*;
use serde_json::json;
use troupekit_core::error::ProviderErrorKind;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_providers::request::{CompletionRequest, Delta, StopReason};
use troupekit_providers::wire::{
    decode_body, encode, Decoder, MaxTokensField, Protocol, Quirks, SchemaMode,
};

fn openai_quirks() -> Quirks {
    Quirks {
        max_tokens_field: MaxTokensField::MaxCompletionTokens,
        temperature: false,
        ..Quirks::default()
    }
}

#[test]
fn encodes_the_golden_request() {
    let body = encode(
        Protocol::OpenAi,
        &openai_quirks(),
        "gpt-6.2-sol",
        &rich_request(),
        false,
    )
    .unwrap();
    assert_eq!(body, golden("openai_request.json"));
}

#[test]
fn compatible_vendors_follow_their_quirks() {
    let req = rich_request();
    let plain = Quirks {
        output_schema: SchemaMode::Unsupported,
        stream_usage: false,
        ..Quirks::default()
    };
    let body = encode(Protocol::OpenAiCompatible, &plain, "m", &req, true).unwrap();
    assert_eq!(body["max_tokens"], json!(1024));
    assert!(body.get("max_completion_tokens").is_none());
    assert!(body.get("response_format").is_none());
    assert_eq!(body["temperature"], json!(0.5));
    assert_eq!(body["stream"], json!(true));
    assert!(body.get("stream_options").is_none());
    let json_mode = Quirks {
        output_schema: SchemaMode::JsonObject,
        ..Quirks::default()
    };
    let body = encode(Protocol::OpenAiCompatible, &json_mode, "m", &req, true).unwrap();
    assert_eq!(body["response_format"], json!({"type": "json_object"}));
    assert_eq!(body["stream_options"], json!({"include_usage": true}));
}

#[test]
fn text_only_user_turns_are_plain_strings() {
    let body = encode(
        Protocol::OpenAiCompatible,
        &Quirks::default(),
        "m",
        &CompletionRequest::user("hi", 5),
        false,
    )
    .unwrap();
    assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
}

#[test]
fn rejects_parts_the_wire_cannot_carry() {
    let mut req = CompletionRequest::user("go", 10);
    req.messages.push(
        Message::new(
            Role::Tool,
            vec![
                result("a"),
                ContentPart::Text {
                    text: "note".into(),
                },
            ],
        )
        .unwrap(),
    );
    let err = encode(Protocol::OpenAi, &Quirks::default(), "m", &req, false).unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::BadRequest);
}

#[test]
fn unary_and_stream_fold_to_the_same_reply() {
    let unary = fold_all(&decode_body(Protocol::OpenAi, &golden("openai_reply.json")).unwrap());
    let (deltas, streamed) = stream_deltas(Protocol::OpenAi, &golden_text("openai_stream.sse"));
    assert_eq!(unary, expected_reply("toolu_1"));
    assert_eq!(streamed, unary);
    assert!(deltas.contains(&Delta::TextDelta { text: "5k.".into() }));
}

#[test]
fn a_tool_call_id_repeated_in_every_fragment_starts_once() {
    let mut decoder = Decoder::new(Protocol::OpenAiCompatible);
    let chunk = |args: &str| json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"calendar","arguments":args}}]}}]});
    let mut deltas = decoder.event(&chunk("{\"days\":")).unwrap();
    deltas.extend(decoder.event(&chunk("3}")).unwrap());
    deltas.extend(
        decoder
            .event(&json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}))
            .unwrap(),
    );
    let reply = fold_all(&deltas);
    assert_eq!(reply.message.tool_calls().count(), 1);
    assert_eq!(reply.stop, StopReason::ToolUse);
}

#[test]
fn maps_errors_and_finish_reasons() {
    let mut decoder = Decoder::new(Protocol::OpenAi);
    let err = decoder
        .event(&json!({"error":{"message":"quota","code":429}}))
        .unwrap_err();
    assert_eq!(
        (err.kind, err.message.as_str()),
        (ProviderErrorKind::RateLimited, "quota")
    );
    assert_eq!(
        decode_body(Protocol::OpenAi, &json!({"choices":[]}))
            .unwrap_err()
            .kind,
        ProviderErrorKind::Decode
    );
    let table = [
        ("stop", StopReason::EndTurn),
        ("length", StopReason::MaxTokens),
        ("content_filter", StopReason::Refusal),
        ("weird", StopReason::Other("weird".into())),
    ];
    for (wire, want) in table {
        let body = json!({"choices":[{"message":{"content":"x"},"finish_reason":wire}]});
        assert_eq!(
            fold_all(&decode_body(Protocol::OpenAi, &body).unwrap()).stop,
            want,
            "{wire}"
        );
    }
}
