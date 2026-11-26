#[path = "../examples/support/agent.rs"]
mod agent;

use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;
use troupekit::context::embed::{Embedder, HashingEmbedder};
use troupekit::core::secret::Secret;
use troupekit_providers::model::{Endpoint, HttpModel};
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::wire::{AuthStyle, Protocol, Quirks};
use troupekit_providers::Price;

const SELECTOR: &str = "fake/support";
const USER: &str = "user-1";
const SESSION: &str = "session-1";
const DIMS: usize = 256;
const ANSWER: &str = "You have 30 days from delivery to return an unused item [returns-window.md].";
const SUMMARY: &str = "Customer asked about the return window for order A1234.";

fn docs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/support/docs")
}

fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect::<String>() + "data: [DONE]\n\n"
}

fn chunk(delta: Value, finish: Option<&str>) -> Value {
    json!({ "id": "c", "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }], "usage": null })
}

fn call(index: usize, id: &str, name: &str, args: Value) -> Value {
    chunk(json!({ "tool_calls": [{ "index": index, "id": id, "type": "function", "function": { "name": name, "arguments": args.to_string() } }] }), None)
}

fn usage() -> Value {
    json!({ "id": "c", "choices": [], "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 } })
}

fn script() -> Vec<Scripted> {
    let first = sse(&[
        chunk(json!({ "role": "assistant", "content": "" }), None),
        call(0, "call_search", "search_knowledge", json!({ "query": "return window days unused tags original packaging" })),
        call(1, "call_fact", "remember_fact", json!({ "key": "order_number", "value": "A1234" })),
        call(2, "call_summary", "update_session_summary", json!({ "text": SUMMARY })),
        chunk(json!({}), Some("tool_calls")),
        usage(),
    ]);
    let second = sse(&[chunk(json!({ "role": "assistant", "content": ANSWER }), None), chunk(json!({}), Some("stop")), usage()]);
    vec![Scripted::chunked(&first, 64), Scripted::chunked(&second, 64)]
}

fn tool_result(body: &[u8], id: &str) -> String {
    let body: Value = serde_json::from_slice(body).unwrap();
    let message = body["messages"].as_array().unwrap().iter().find(|m| m["role"] == "tool" && m["tool_call_id"] == id).unwrap_or_else(|| panic!("no tool result {id} in {body}"));
    match &message["content"] {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

#[tokio::test]
async fn support_agent_answers_with_citations_and_remembers() {
    let dir = tempfile::tempdir().unwrap();
    let db = dir.path().join("support.db");
    let embedder: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(DIMS));
    let transport = Arc::new(ScriptedTransport::new(script()));
    let endpoint = Endpoint { protocol: Protocol::OpenAi, base_url: "https://h.test/".into(), auth: AuthStyle::Bearer, quirks: Quirks::default() };
    let key = Some(Secret::new("sk-test-not-real".into()).unwrap());
    let model = Arc::new(HttpModel::new(SELECTOR, "support", endpoint, key, transport.clone()).unwrap());
    {
        let (store, vectors, report) = agent::open(&db, &docs(), embedder.as_ref()).await.unwrap();
        assert_eq!((report.added, report.skipped), (30, 0));
        assert!(report.failed.is_empty(), "{:?}", report.failed);
        let kit = agent::kit(&store, &vectors, embedder.clone(), USER, SESSION).unwrap();
        let answer = agent::answer(&store, &kit, SELECTOR, model, Price::FREE, "How long do I have to send back a jacket? My order is A1234.").await.unwrap();
        assert_eq!(answer, ANSWER);
    }
    let sent = transport.sent();
    assert_eq!(sent.len(), 2);
    let first: Value = serde_json::from_slice(sent[0].body.as_deref().unwrap()).unwrap();
    let tools: Vec<&str> = first["tools"].as_array().unwrap().iter().map(|t| t["function"]["name"].as_str().unwrap()).collect();
    assert!(tools.contains(&"search_knowledge") && tools.contains(&"remember_fact"), "{tools:?}");
    let found = tool_result(sent[1].body.as_deref().unwrap(), "call_search");
    assert!(found.contains("[1] returns-window.md"), "{found}");
    assert!(found.contains("<untrusted-content>") && found.contains("Sources:"), "{found}");
    assert!(tool_result(sent[1].body.as_deref().unwrap(), "call_fact").contains("Remembered"));
    let (store, vectors, report) = agent::open(&db, &docs(), embedder.as_ref()).await.unwrap();
    assert_eq!((report.added, report.replaced, report.skipped), (0, 0, 30));
    let kit = agent::kit(&store, &vectors, embedder, USER, SESSION).unwrap();
    let preamble = kit.preamble().await.unwrap();
    assert!(preamble.contains("order_number: A1234"), "{preamble}");
    assert!(preamble.contains(SUMMARY), "{preamble}");
    let other = agent::kit(&store, &vectors, Arc::new(HashingEmbedder::new(DIMS)), "user-2", "session-2").unwrap();
    assert_eq!(other.preamble().await.unwrap(), "");
}
