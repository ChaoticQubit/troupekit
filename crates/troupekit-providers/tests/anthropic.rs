use crate::common;

use common::*;
use serde_json::json;
use troupekit_core::error::ProviderErrorKind;
use troupekit_core::message::{Message, Role};
use troupekit_providers::request::{CompletionRequest, Delta, StopReason};
use troupekit_providers::wire::{decode_body, encode, Decoder, Protocol, Quirks, SchemaMode};

fn anthropic_quirks() -> Quirks {
    Quirks {
        temperature: false,
        ..Quirks::default()
    }
}

#[test]
fn encodes_the_golden_request() {
    let body = encode(
        Protocol::Anthropic,
        &anthropic_quirks(),
        "claude-opus-5-5",
        &rich_request(),
        false,
    )
    .unwrap();
    assert_eq!(body, golden("anthropic_request.json"));
}

#[test]
fn stream_flag_temperature_and_schema_follow_mode_and_quirks() {
    let req = rich_request();
    let body = encode(Protocol::Anthropic, &Quirks::default(), "m", &req, true).unwrap();
    assert_eq!(body["stream"], json!(true));
    assert_eq!(body["temperature"], json!(0.5));
    let plain = Quirks {
        output_schema: SchemaMode::Unsupported,
        ..Quirks::default()
    };
    assert!(encode(Protocol::Anthropic, &plain, "m", &req, false)
        .unwrap()
        .get("output_config")
        .is_none());
}

#[test]
fn merges_consecutive_tool_results_into_one_user_turn() {
    let mut req = CompletionRequest::user("go", 10);
    req.messages
        .push(Message::new(Role::Assistant, vec![call("a", "t"), call("b", "t")]).unwrap());
    req.messages
        .push(Message::new(Role::Tool, vec![result("a")]).unwrap());
    req.messages
        .push(Message::new(Role::Tool, vec![result("b")]).unwrap());
    let body = encode(Protocol::Anthropic, &anthropic_quirks(), "m", &req, false).unwrap();
    let turns = body["messages"].as_array().unwrap();
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[2]["role"], "user");
    assert_eq!(turns[2]["content"].as_array().unwrap().len(), 2);
}

#[test]
fn unary_and_stream_fold_to_the_same_reply() {
    let unary =
        fold_all(&decode_body(Protocol::Anthropic, &golden("anthropic_reply.json")).unwrap());
    let (deltas, streamed) =
        stream_deltas(Protocol::Anthropic, &golden_text("anthropic_stream.sse"));
    assert_eq!(unary, expected_reply("toolu_1"));
    assert_eq!(streamed, unary);
    assert!(deltas.contains(&Delta::TextDelta {
        text: "Mon: run ".into()
    }));
}

#[test]
fn maps_stream_error_events() {
    let mut decoder = Decoder::new(Protocol::Anthropic);
    let err = decoder
        .event(&json!({"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}))
        .unwrap_err();
    assert_eq!(
        (err.kind, err.message.as_str()),
        (ProviderErrorKind::Server, "Overloaded")
    );
    let err = decoder
        .event(&json!({"type":"error","error":{"type":"rate_limit_error","message":"slow"}}))
        .unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::RateLimited);
    let err = decode_body(
        Protocol::Anthropic,
        &json!({"type":"error","error":{"type":"invalid_request_error","message":"bad"}}),
    );
    assert_eq!(err.unwrap_err().kind, ProviderErrorKind::BadRequest);
}

#[test]
fn maps_stop_reasons() {
    let table = [
        ("end_turn", StopReason::EndTurn),
        ("stop_sequence", StopReason::EndTurn),
        ("max_tokens", StopReason::MaxTokens),
        ("refusal", StopReason::Refusal),
        ("pause_turn", StopReason::PauseTurn),
    ];
    for (wire, want) in table {
        let body = json!({"type":"message","content":[],"stop_reason":wire,"usage":{"input_tokens":1,"output_tokens":1}});
        assert_eq!(
            fold_all(&decode_body(Protocol::Anthropic, &body).unwrap()).stop,
            want,
            "{wire}"
        );
    }
}

fn blank_blocks(body: &serde_json::Value) -> Vec<String> {
    let mut out = Vec::new();
    for (t, turn) in body["messages"].as_array().unwrap().iter().enumerate() {
        for block in turn["content"].as_array().unwrap() {
            let text = match block["type"].as_str() {
                Some("text") => block["text"].as_str(),
                Some("tool_result") => block["content"].as_str(),
                _ => None,
            };
            if text.is_some_and(|s| s.trim().is_empty()) {
                out.push(format!("turn {t}: {block}"));
            }
        }
    }
    out
}

#[test]
fn never_sends_a_blank_text_block() {
    use troupekit_core::message::ContentPart;
    let mut req = CompletionRequest::user("plan this", 10);
    req.messages.push(Message::text(Role::Assistant, ""));
    req.messages
        .push(Message::text(Role::User, "Your answer failed these checks"));
    req.messages.push(
        Message::new(
            Role::Assistant,
            vec![ContentPart::Text { text: "  ".into() }, call("a", "t")],
        )
        .unwrap(),
    );
    req.messages.push(
        Message::new(
            Role::Tool,
            vec![ContentPart::ToolResult {
                call_id: "a".into(),
                content: String::new(),
                is_error: false,
            }],
        )
        .unwrap(),
    );
    req.messages.push(Message::text(Role::User, "\n"));
    let body = encode(Protocol::Anthropic, &anthropic_quirks(), "m", &req, false).unwrap();
    assert_eq!(blank_blocks(&body), Vec::<String>::new());
    let turns = body["messages"].as_array().unwrap();
    assert_eq!(turns[0]["role"], "user");
    assert_eq!(turns[0]["content"].as_array().unwrap().len(), 2);
    assert!(turns.windows(2).all(|w| w[0]["role"] != w[1]["role"]));
    let only_blank = CompletionRequest::user(" ", 10);
    let err = encode(
        Protocol::Anthropic,
        &anthropic_quirks(),
        "m",
        &only_blank,
        false,
    )
    .unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::BadRequest);
}

fn hosted_request() -> CompletionRequest {
    use troupekit_core::tool::{HostedTool, ToolSpec};
    let mut req = CompletionRequest::user("research", 100);
    req.output_schema = Some(json!({"type": "object"}));
    let spec = |hosted: HostedTool| ToolSpec {
        name: hosted.name().into(),
        description: "d".into(),
        schema: json!({"type": "object"}),
        hosted: Some(hosted),
    };
    req.tools = vec![
        spec(HostedTool::WebSearch { max_uses: 5 }),
        spec(HostedTool::WebFetch {
            max_uses: 3,
            max_content_tokens: 8000,
        }),
    ];
    req
}

#[test]
fn hosted_tools_encode_as_server_tools_without_output_format() {
    let body = encode(
        Protocol::Anthropic,
        &anthropic_quirks(),
        "m",
        &hosted_request(),
        true,
    )
    .unwrap();
    assert_eq!(
        body["tools"],
        json!([
            {"type": "web_search_20250305", "name": "web_search", "max_uses": 5},
            {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 3, "max_content_tokens": 8000}
        ])
    );
    assert!(body.get("output_config").is_none());
    let openai = encode(
        Protocol::OpenAi,
        &Quirks::default(),
        "m",
        &hosted_request(),
        false,
    )
    .unwrap();
    assert_eq!(openai["tools"][0]["function"]["name"], "web_search");
}

#[test]
fn search_stream_keeps_hosted_blocks_and_sends_them_back() {
    use troupekit_core::message::ContentPart;
    let (_, reply) = stream_deltas(
        Protocol::Anthropic,
        &golden_text("anthropic_search_stream.sse"),
    );
    assert_eq!(reply.stop, StopReason::EndTurn);
    assert_eq!(reply.message.hosted_calls("web_search"), 1);
    assert_eq!(
        reply.message.final_text(),
        "{\"summary\":\"Run 3 times a week.\",\"sources\":[]}"
    );
    let ContentPart::Hosted { block, .. } = &reply.message.parts[1] else {
        panic!("{:?}", reply.message.parts)
    };
    assert_eq!(block["input"], json!({"query": "couch to 5k plan"}));
    let mut body = json!({"type": "message", "content": [], "stop_reason": "end_turn",
        "usage": {"input_tokens": 40, "output_tokens": 30}});
    for p in &reply.message.parts {
        body["content"].as_array_mut().unwrap().push(match p {
            ContentPart::Text { text } => json!({"type": "text", "text": text}),
            ContentPart::Hosted { block, .. } => block.clone(),
            other => panic!("{other:?}"),
        });
    }
    assert_eq!(
        fold_all(&decode_body(Protocol::Anthropic, &body).unwrap()),
        reply
    );
    let mut req = hosted_request();
    req.messages.push(reply.message.clone());
    let sent = encode(Protocol::Anthropic, &anthropic_quirks(), "m", &req, false).unwrap();
    assert_eq!(sent["messages"][1]["content"][1]["type"], "server_tool_use");
    assert_eq!(
        sent["messages"][1]["content"][2]["content"][0]["encrypted_content"],
        "EqAB"
    );
    let openai = encode(Protocol::OpenAi, &Quirks::default(), "m", &req, false).unwrap();
    assert!(!openai.to_string().contains("server_tool_use"));
}

#[test]
fn pause_turn_keeps_the_hosted_call() {
    let body = json!({"type": "message",
        "content": [{"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "q"}}],
        "stop_reason": "pause_turn", "usage": {"input_tokens": 1, "output_tokens": 1}});
    let reply = fold_all(&decode_body(Protocol::Anthropic, &body).unwrap());
    assert_eq!(reply.stop, StopReason::PauseTurn);
    assert_eq!(reply.message.hosted_calls("web_search"), 1);
}
