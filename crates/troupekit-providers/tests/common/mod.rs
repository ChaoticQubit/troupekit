#![allow(dead_code)]

use serde_json::{json, Value};
use std::sync::Arc;
use troupekit_core::event::Usage;
use troupekit_core::message::{ContentPart, MediaSource, Message, Role};
use troupekit_core::secret::Secret;
use troupekit_core::tool::ToolSpec;
use troupekit_providers::fold::Fold;
use troupekit_providers::model::{Endpoint, HttpModel};
use troupekit_providers::request::{CompletionRequest, CompletionResponse, Delta, StopReason};
use troupekit_providers::sse::SseParser;
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::wire::{AuthStyle, Decoder, Protocol, Quirks};

pub fn call(id: &str, name: &str) -> ContentPart {
    ContentPart::ToolCall {
        id: id.into(),
        name: name.into(),
        arguments: json!({"days": 7}),
    }
}

pub fn result(id: &str) -> ContentPart {
    ContentPart::ToolResult {
        call_id: id.into(),
        content: "free: mon, wed".into(),
        is_error: false,
    }
}

pub fn rich_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        system: Some("You plan workouts.".into()),
        messages: vec![
            Message::new(
                Role::User,
                vec![
                    ContentPart::Text {
                        text: "Plan my week".into(),
                    },
                    ContentPart::Image {
                        media_type: "image/png".into(),
                        source: MediaSource::Base64 {
                            data: "aGk=".into(),
                        },
                    },
                ],
            )
            .unwrap(),
            Message::new(
                Role::Assistant,
                vec![
                    ContentPart::Text {
                        text: "Checking the calendar.".into(),
                    },
                    call("call_1", "calendar"),
                ],
            )
            .unwrap(),
            Message::new(Role::Tool, vec![result("call_1")]).unwrap(),
        ],
        tools: vec![ToolSpec {
            name: "calendar".into(),
            description: "Reads free slots".into(),
            schema: json!({"type":"object","properties":{"days":{"type":"integer"}},"required":["days"]}),
            hosted: None,
        }],
        max_tokens: 1024,
        temperature: Some(0.5),
        output_schema: Some(json!({"type":"object","properties":{"plan":{"type":"string"}}})),
    }
}

pub fn golden_text(name: &str) -> String {
    std::fs::read_to_string(format!(
        "{}/tests/golden/{name}",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
}

pub fn golden(name: &str) -> Value {
    serde_json::from_str(&golden_text(name)).unwrap()
}

pub fn fold_all(deltas: &[Delta]) -> CompletionResponse {
    let mut fold = Fold::new();
    for d in deltas {
        fold.push(d).unwrap();
    }
    fold.finish().unwrap()
}

pub fn stream_deltas(protocol: Protocol, text: &str) -> (Vec<Delta>, CompletionResponse) {
    let mut parser = SseParser::new();
    let mut events = Vec::new();
    for chunk in text.as_bytes().chunks(7) {
        events.extend(parser.push(chunk).unwrap());
    }
    events.extend(parser.finish().unwrap());
    let mut decoder = Decoder::new(protocol);
    let mut deltas = Vec::new();
    for event in &events {
        deltas.extend(decoder.sse(event).unwrap());
    }
    let response = fold_all(&deltas);
    (deltas, response)
}

pub fn expected_reply(call_id: &str) -> CompletionResponse {
    CompletionResponse {
        message: Message::new(
            Role::Assistant,
            vec![
                ContentPart::Text {
                    text: "Mon: run 5k.".into(),
                },
                ContentPart::ToolCall {
                    id: call_id.into(),
                    name: "calendar".into(),
                    arguments: json!({"days": 3}),
                },
            ],
        )
        .unwrap(),
        usage: Usage {
            input_tokens: 12,
            output_tokens: 30,
            cost_usd_micros: 0,
        },
        stop: StopReason::ToolUse,
    }
}

pub fn test_key() -> Option<Secret> {
    Some(Secret::new("test-key-123".into()).unwrap())
}

pub fn scripted_model(
    protocol: Protocol,
    replies: Vec<Scripted>,
) -> (HttpModel, Arc<ScriptedTransport>) {
    let transport = Arc::new(ScriptedTransport::new(replies));
    let endpoint = Endpoint {
        protocol,
        base_url: "https://h.test".into(),
        auth: AuthStyle::Bearer,
        quirks: Quirks::default(),
    };
    let model =
        HttpModel::new("vendor/m-1", "m-1", endpoint, test_key(), transport.clone()).unwrap();
    (model, transport)
}
