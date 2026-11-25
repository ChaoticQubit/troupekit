use serde_json::{json, Value};
use std::sync::Arc;
use troupekit_context::embed::Embedder;
use troupekit_context::ContextError;
use troupekit_core::secret::Secret;
use troupekit_providers::catalog::{Catalog, CatalogError};
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::HttpEmbedder;

const CATALOG: &str = include_str!("../../../catalog/providers.toml");
const SMALL: &str = "[[vendor]]\nid = \"v\"\nname = \"V\"\nprotocol = \"openai_compatible\"\nbase_url = \"https://v.test/v1\"\nauth = \"bearer\"\n\n[[embedding]]\nvendor = \"v\"\nid = \"e\"\ndims = 2\nmax_batch = 2\ninput_usd_micros_per_mtok = 0\nprice_verified = false\n";

fn key(value: &str) -> Option<Secret> {
    Some(Secret::new(value.into()).unwrap())
}

fn texts(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("text {i}")).collect()
}

fn openai_reply(vectors: &[(usize, Vec<f32>)]) -> Scripted {
    let data: Vec<Value> = vectors.iter().map(|(i, v)| json!({ "object": "embedding", "index": i, "embedding": v })).collect();
    Scripted::body(200, json!({ "object": "list", "data": data }).to_string())
}

fn small(replies: Vec<Scripted>) -> (HttpEmbedder, Arc<ScriptedTransport>) {
    let transport = Arc::new(ScriptedTransport::new(replies));
    let e = Catalog::load(SMALL).unwrap().embedder_with("v/e", key("test-key"), transport.clone()).unwrap();
    (e, transport)
}

fn body(transport: &ScriptedTransport, n: usize) -> Value {
    serde_json::from_slice(transport.sent()[n].body.as_ref().unwrap()).unwrap()
}

fn header(transport: &ScriptedTransport, n: usize, name: &str) -> Option<String> {
    transport.sent()[n].headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
}

#[tokio::test]
async fn openai_embeddings_request_shape() {
    let v = vec![0.5f32; 1536];
    let transport = Arc::new(ScriptedTransport::new(vec![openai_reply(&[(0, v.clone()), (1, v.clone())])]));
    let e = Catalog::load(CATALOG).unwrap().embedder_with("openai/text-embedding-3-small", key("test-key"), transport.clone()).unwrap();
    assert_eq!(e.dims(), 1536);
    let out = e.embed(&texts(2)).await.unwrap();
    assert_eq!(out, vec![v.clone(), v]);
    let sent = transport.sent();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].url, "https://api.openai.com/v1/embeddings");
    assert_eq!(header(&transport, 0, "authorization").as_deref(), Some("Bearer test-key"));
    assert_eq!(body(&transport, 0), json!({ "model": "text-embedding-3-small", "input": ["text 0", "text 1"], "encoding_format": "float" }));
}

#[tokio::test]
async fn gemini_batch_embed_request_shape() {
    let v = vec![0.25f32; 3072];
    let reply = json!({ "embeddings": [{ "values": v }, { "values": v }] }).to_string();
    let transport = Arc::new(ScriptedTransport::new(vec![Scripted::body(200, reply)]));
    let e = Catalog::load(CATALOG).unwrap().embedder_with("google/gemini-embedding-001", key("g-key"), transport.clone()).unwrap();
    let out = e.embed(&texts(2)).await.unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[1].len(), 3072);
    let sent = transport.sent();
    assert_eq!(sent[0].url, "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents");
    assert_eq!(header(&transport, 0, "x-goog-api-key").as_deref(), Some("g-key"));
    let b = body(&transport, 0);
    assert_eq!(b["requests"][0], json!({ "model": "models/gemini-embedding-001", "content": { "parts": [{ "text": "text 0" }] } }));
    assert_eq!(b["requests"][1]["content"]["parts"][0]["text"], "text 1");
}

#[tokio::test]
async fn batches_split_and_rejoin_in_order() {
    let (e, transport) = small(vec![
        openai_reply(&[(1, vec![1.0, 1.0]), (0, vec![0.0, 0.0])]),
        openai_reply(&[(1, vec![3.0, 3.0]), (0, vec![2.0, 2.0])]),
        openai_reply(&[(0, vec![4.0, 4.0])]),
    ]);
    let out = e.embed(&texts(5)).await.unwrap();
    let firsts: Vec<f32> = out.iter().map(|v| v[0]).collect();
    assert_eq!(firsts, vec![0.0, 1.0, 2.0, 3.0, 4.0]);
    assert_eq!(transport.sent().len(), 3);
    assert_eq!(body(&transport, 2)["input"], json!(["text 4"]));
    let (e, transport) = small(vec![]);
    assert!(e.embed(&[]).await.unwrap().is_empty());
    assert!(transport.sent().is_empty());
    assert!(matches!(e.embed(&["  ".to_string()]).await, Err(ContextError::Invalid(_))));
}

#[tokio::test]
async fn wrong_count_dims_or_nan_is_an_embed_error() {
    let bad = [
        openai_reply(&[(0, vec![1.0, 1.0])]),
        openai_reply(&[(0, vec![1.0, 1.0]), (1, vec![1.0])]),
        openai_reply(&[(0, vec![1.0, 1.0]), (0, vec![1.0, 1.0])]),
        openai_reply(&[(0, vec![1.0, 1.0]), (5, vec![1.0, 1.0])]),
        Scripted::body(200, r#"{"data":[{"index":0,"embedding":[1e39,0]},{"index":1,"embedding":[0,0]}]}"#),
        Scripted::body(200, r#"{"data":[{"index":0,"embedding":["NaN",0]},{"index":1,"embedding":[0,0]}]}"#),
        Scripted::body(200, "not json"),
        Scripted::body(500, "boom"),
    ];
    for reply in bad {
        let (e, _) = small(vec![reply]);
        let r = e.embed(&texts(2)).await;
        assert!(matches!(r, Err(ContextError::Embed(_))), "{r:?}");
    }
}

#[tokio::test]
async fn key_never_appears_in_errors_or_debug() {
    let secret = "sk-very-secret-123";
    let transport = Arc::new(ScriptedTransport::new(vec![Scripted::body(401, "invalid key"), Scripted::body(200, "{}")]));
    let e = Catalog::load(SMALL).unwrap().embedder_with("v/e", key(secret), transport.clone()).unwrap();
    assert!(!format!("{e:?}").contains(secret));
    for _ in 0..3 {
        let err = e.embed(&texts(1)).await.unwrap_err();
        assert!(!format!("{err} {err:?}").contains(secret));
    }
    assert_eq!(header(&transport, 0, "authorization").as_deref(), Some("Bearer sk-very-secret-123"));
    let missing = Catalog::load(SMALL).unwrap().embedder_with("v/e", None, transport);
    assert_eq!(missing.unwrap_err(), CatalogError::MissingKey("v".into()));
}

#[tokio::test]
async fn anthropic_has_no_embeddings() {
    let c = Catalog::load(CATALOG).unwrap();
    let t = Arc::new(ScriptedTransport::new(vec![]));
    match c.embedder_with("anthropic/claude-opus-5-5", key("k"), t.clone()) {
        Err(CatalogError::BadSelector(m)) => assert!(m.contains("no embeddings endpoint"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert!(matches!(c.embedder_with("openai/gpt-nope", key("k"), t.clone()), Err(CatalogError::UnknownModel(_))));
    assert!(matches!(c.embedder_with("nobody/e", key("k"), t.clone()), Err(CatalogError::UnknownVendor(_))));
    assert!(matches!(c.embedder_with("no-slash", key("k"), t), Err(CatalogError::BadSelector(_))));
}

#[tokio::test]
async fn oversized_body_is_refused() {
    let padding = " ".repeat(10_000);
    let reply = format!(r#"{{"data":[{{"index":0,"embedding":[1,0]}}]{padding}}}"#);
    let (e, _) = small(vec![Scripted::body(200, reply)]);
    let r = e.embed(&texts(1)).await;
    assert!(matches!(r, Err(ContextError::Embed(ref m)) if m.contains("over")), "{r:?}");
}

#[tokio::test]
async fn echoed_key_is_redacted_from_error_text() {
    let secret = "sk-echoed-secret-456";
    let echo = format!("{{\"error\":\"Incorrect API key provided: {secret}. Check it.\"}}");
    let long = format!("{}{secret}", "x".repeat(190));
    let transport = Arc::new(ScriptedTransport::new(vec![Scripted::body(401, echo), Scripted::body(403, long)]));
    let e = Catalog::load(SMALL).unwrap().embedder_with("v/e", key(secret), transport).unwrap();
    for _ in 0..2 {
        let err = e.embed(&texts(1)).await.unwrap_err();
        let text = format!("{err} {err:?}");
        assert!(!text.contains(secret), "{text}");
        assert!(!text.contains("sk-echoed"), "{text}");
        assert!(text.contains("HTTP 40"), "{text}");
    }
}
