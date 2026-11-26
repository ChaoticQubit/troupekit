use crate::common;

use common::*;
use serde_json::Value;
use std::sync::Arc;
use troupekit_core::error::ProviderErrorKind;
use troupekit_providers::model::{valid_model_id, Endpoint, HttpModel, Model};
use troupekit_providers::request::{CompletionRequest, Delta};
use troupekit_providers::transport::{HttpError, HttpErrorKind, Scripted, ScriptedTransport};
use troupekit_providers::wire::{anthropic, AuthStyle, Protocol, Quirks};

fn endpoint(protocol: Protocol, auth: AuthStyle) -> Endpoint {
    Endpoint {
        protocol,
        base_url: "https://h.test/".into(),
        auth,
        quirks: Quirks::default(),
    }
}

#[test]
fn builds_urls_and_auth_headers_per_protocol() {
    let req = CompletionRequest::user("hi", 10);
    let cases = [
        (
            Protocol::Anthropic,
            AuthStyle::XApiKey,
            false,
            "https://h.test/v1/messages",
            "x-api-key",
            "test-key-123",
        ),
        (
            Protocol::OpenAi,
            AuthStyle::Bearer,
            true,
            "https://h.test/chat/completions",
            "authorization",
            "Bearer test-key-123",
        ),
        (
            Protocol::OpenAiCompatible,
            AuthStyle::Bearer,
            false,
            "https://h.test/chat/completions",
            "authorization",
            "Bearer test-key-123",
        ),
        (
            Protocol::Gemini,
            AuthStyle::XGoogApiKey,
            false,
            "https://h.test/models/m-1:generateContent",
            "x-goog-api-key",
            "test-key-123",
        ),
        (
            Protocol::Gemini,
            AuthStyle::XGoogApiKey,
            true,
            "https://h.test/models/m-1:streamGenerateContent?alt=sse",
            "x-goog-api-key",
            "test-key-123",
        ),
    ];
    for (protocol, auth, stream, url, header, value) in cases {
        let r = endpoint(protocol, auth)
            .request("m-1", test_key().as_ref(), &req, stream)
            .unwrap();
        assert_eq!(r.url, url);
        assert_eq!(r.header(header), Some(value));
        assert_eq!(r.header("content-type"), Some("application/json"));
        assert_eq!(r.header("accept").is_some(), stream);
        assert!(!r.url.contains("test-key-123"));
        assert!(!format!("{r:?}").contains("test-key-123"));
    }
    let r = endpoint(Protocol::Anthropic, AuthStyle::XApiKey)
        .request("m", test_key().as_ref(), &req, false)
        .unwrap();
    assert_eq!(r.header("anthropic-version"), Some(anthropic::API_VERSION));
}

#[test]
fn keyless_endpoints_send_no_auth_and_keyed_ones_need_a_key() {
    let req = CompletionRequest::user("hi", 10);
    let r = endpoint(Protocol::OpenAiCompatible, AuthStyle::NoKey)
        .request("m", test_key().as_ref(), &req, false)
        .unwrap();
    assert!(r.header("authorization").is_none());
    let err = endpoint(Protocol::OpenAi, AuthStyle::Bearer)
        .request("m", None, &req, false)
        .unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::Auth);
}

#[test]
fn rejects_unsafe_model_ids() {
    for bad in ["", "a b", "x?key=1", "../admin", "m#frag", "caf\u{e9}"] {
        assert!(!valid_model_id(bad), "{bad}");
    }
    for good in [
        "claude-opus-5-5",
        "anthropic/claude-opus-5.5",
        "qwen3:8b",
        "gpt-6.2-sol",
    ] {
        assert!(valid_model_id(good), "{good}");
    }
    let transport = Arc::new(ScriptedTransport::default());
    assert!(HttpModel::new(
        "v/x",
        "a b",
        endpoint(Protocol::OpenAi, AuthStyle::Bearer),
        None,
        transport
    )
    .is_err());
}

#[tokio::test]
async fn complete_sends_one_unary_request_and_folds_the_body() {
    let (model, transport) = scripted_model(
        Protocol::OpenAi,
        vec![Scripted::body(200, golden_text("openai_reply.json"))],
    );
    assert_eq!(model.id(), "vendor/m-1");
    assert_eq!(
        model.complete(&rich_request()).await.unwrap(),
        expected_reply("toolu_1")
    );
    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    let body: Value = serde_json::from_slice(sent[0].body.as_deref().unwrap()).unwrap();
    assert_eq!(body["model"], "m-1");
    assert!(body.get("stream").is_none());
}

#[tokio::test]
async fn stream_hands_out_deltas_and_returns_the_unary_reply() {
    let cases = [
        (Protocol::Anthropic, "anthropic_stream.sse", "toolu_1"),
        (Protocol::OpenAi, "openai_stream.sse", "toolu_1"),
        (Protocol::Gemini, "gemini_stream.sse", "call_0"),
    ];
    for (protocol, fixture, call_id) in cases {
        let (model, transport) =
            scripted_model(protocol, vec![Scripted::chunked(&golden_text(fixture), 5)]);
        let mut seen = Vec::new();
        let reply = model
            .stream(&rich_request(), &mut |d: &Delta| seen.push(d.clone()))
            .await
            .unwrap();
        assert_eq!(reply, expected_reply(call_id), "{fixture}");
        assert_eq!(fold_all(&seen), reply, "{fixture}");
        assert_eq!(
            transport.sent()[0].header("accept"),
            Some("text/event-stream")
        );
    }
}

#[tokio::test]
async fn maps_error_statuses_bodies_and_transport_failures() {
    let (model, _) = scripted_model(
        Protocol::OpenAi,
        vec![
            Scripted::body(429, r#"{"error":{"message":"slow down"}}"#)
                .with_header("Retry-After", "2"),
            Scripted::body(503, "upstream down"),
            Scripted::body(200, "not json"),
            Scripted::Fail(HttpError::new(HttpErrorKind::Timeout, "read timed out")),
        ],
    );
    let req = CompletionRequest::user("hi", 10);
    let err = model.complete(&req).await.unwrap_err();
    assert_eq!(
        (err.kind, err.retry_after_ms, err.message.as_str()),
        (
            ProviderErrorKind::RateLimited,
            Some(2000),
            "HTTP 429: slow down"
        )
    );
    assert_eq!(
        model.complete(&req).await.unwrap_err().kind,
        ProviderErrorKind::Server
    );
    assert_eq!(
        model.complete(&req).await.unwrap_err().kind,
        ProviderErrorKind::Decode
    );
    assert_eq!(
        model.complete(&req).await.unwrap_err().kind,
        ProviderErrorKind::Stalled
    );
}

#[tokio::test]
async fn stream_maps_statuses_and_truncation() {
    let (model, _) = scripted_model(Protocol::Anthropic, vec![
        Scripted::body(401, r#"{"type":"error","error":{"type":"authentication_error","message":"invalid x-api-key"}}"#),
        Scripted::chunked("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":1}}}\n\n", 4),
    ]);
    let req = CompletionRequest::user("hi", 10);
    let mut sink = |_: &Delta| {};
    let err = model.stream(&req, &mut sink).await.unwrap_err();
    assert_eq!(
        (err.kind, err.message.as_str()),
        (ProviderErrorKind::Auth, "HTTP 401: invalid x-api-key")
    );
    assert_eq!(
        model.stream(&req, &mut sink).await.unwrap_err().kind,
        ProviderErrorKind::Network
    );
}

#[tokio::test]
async fn rejects_invalid_or_mismatched_requests_before_sending() {
    let (model, transport) = scripted_model(Protocol::OpenAi, vec![]);
    let mut req = CompletionRequest::user("hi", 0);
    assert_eq!(
        model.complete(&req).await.unwrap_err().kind,
        ProviderErrorKind::BadRequest
    );
    req.max_tokens = 10;
    req.model = Some("other".into());
    assert_eq!(
        model.complete(&req).await.unwrap_err().kind,
        ProviderErrorKind::BadRequest
    );
    assert!(transport.sent().is_empty());
    req.model = Some("m-1".into());
    assert_eq!(
        model.complete(&req).await.unwrap_err().kind,
        ProviderErrorKind::Network
    );
    assert_eq!(transport.sent().len(), 1);
}

#[tokio::test]
async fn http_model_span_has_no_headers_or_query() {
    let query_canary = "sk-QUERY-CANARY-0123456789";
    let header_canary = "sk-HEADER-CANARY-0123456789";
    for stream in [false, true] {
        let fixture = if stream {
            "openai_stream.sse"
        } else {
            "openai_reply.json"
        };
        let transport = Arc::new(ScriptedTransport::new(vec![Scripted::body(
            200,
            golden_text(fixture),
        )]));
        let endpoint = Endpoint {
            protocol: Protocol::OpenAi,
            base_url: format!("https://h.test/v1?key={query_canary}"),
            auth: AuthStyle::Bearer,
            quirks: Quirks::default(),
        };
        let key = troupekit_core::secret::Secret::new(header_canary.into()).unwrap();
        let model = HttpModel::new("vendor/m-1", "m-1", endpoint, Some(key), transport).unwrap();
        let sink = Arc::new(troupekit_trace::MemorySink::default());
        let tracer = troupekit_trace::Tracer::builder()
            .content(true)
            .sink(sink.clone())
            .build()
            .unwrap();
        let req = CompletionRequest::user("hi", 10);
        let reply = if stream {
            tracer
                .in_scope(model.stream(&req, &mut |_: &Delta| {}))
                .await
        } else {
            tracer.in_scope(model.complete(&req)).await
        };
        reply.unwrap();
        drop(tracer);
        let lines = sink.lines();
        let records: Vec<Value> = lines
            .iter()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let start = records
            .iter()
            .find(|r| r["t"] == "start" && r["name"] == "model.http")
            .expect("a model.http span");
        assert_eq!(start["cat"], "models");
        assert_eq!(start["fields"]["model"], "vendor/m-1");
        assert_eq!(start["fields"]["stream"], stream);
        let url = start["fields"]["url"].as_str().unwrap();
        assert_eq!(url, "https://h.test/v1");
        let span = start["span"].clone();
        let end = records
            .iter()
            .find(|r| r["t"] == "end" && r["span"] == span)
            .expect("the span ends");
        assert_eq!(end["ok"], true);
        assert_eq!(end["fields"]["status"], 200);
        assert!(end["fields"]["reply_bytes"].as_u64().unwrap() > 0);
        for line in &lines {
            assert!(!line.contains(query_canary), "query key leaked: {line}");
            assert!(!line.contains(header_canary), "header key leaked: {line}");
            assert!(
                !line.to_ascii_lowercase().contains("authorization"),
                "header leaked: {line}"
            );
        }
    }
}
