#[allow(dead_code)]
#[path = "../examples/support/agent.rs"]
mod agent;

use serde_json::{json, Value};
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use troupekit::agent::Limits;
use troupekit::context::embed::{Embedder, HashingEmbedder};
use troupekit::core::secret::Secret;
use troupekit::trace::view::{read_jsonl, render_html, render_terminal, TermOptions, ViewLimits};
use troupekit::trace::{CallbackSink, MemorySink, Trace, TraceSink};
use troupekit_kit::{tool, Agent, Graph, Storage};
use troupekit_providers::route::RoutePolicy;
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::Price;

const CANARY: &str = "sk-TROUPEKIT-CANARY-0123456789";
const DIMS: usize = 256;
const ANSWER: &str = "Returns are open for 30 days; 2 + 3 = 5.";

/// Adds two integers a and b.
#[tool]
async fn add(a: i64, b: i64) -> i64 {
    a + b
}

fn docs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/support/docs")
}

fn sse(events: &[Value]) -> String {
    events
        .iter()
        .map(|e| format!("data: {e}\n\n"))
        .collect::<String>()
        + "data: [DONE]\n\n"
}

fn chunk(delta: Value, finish: Option<&str>) -> Value {
    json!({"id": "c", "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], "usage": null})
}

fn call(index: usize, id: &str, name: &str, args: Value) -> Value {
    chunk(
        json!({"tool_calls": [{"index": index, "id": id, "type": "function",
            "function": {"name": name, "arguments": args.to_string()}}]}),
        None,
    )
}

fn usage() -> Value {
    json!({"id": "c", "choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
}

fn script() -> Vec<Scripted> {
    let first = sse(&[
        chunk(json!({"role": "assistant", "content": ""}), None),
        call(0, "call_add", "add", json!({"a": 2, "b": 3})),
        call(
            1,
            "call_search",
            "search_knowledge",
            json!({"query": "return window days"}),
        ),
        call(
            2,
            "call_fact",
            "remember_fact",
            json!({"key": "order", "value": "A1234"}),
        ),
        chunk(json!({}), Some("tool_calls")),
        usage(),
    ]);
    let second = sse(&[
        chunk(json!({"role": "assistant", "content": ANSWER}), None),
        chunk(json!({}), Some("stop")),
        usage(),
    ]);
    vec![
        Scripted::chunked(&first, 64),
        Scripted::chunked(&second, 64),
    ]
}

/// Where fixture A sends its trace records: a sink, or a new JSONL file.
enum Out {
    Sink(Arc<dyn TraceSink>),
    File(PathBuf),
}

/// Runs fixture A once: a kit graph with one agent node on a routed
/// cloud model (scripted transport, canary key), a user tool, the
/// knowledge tools wired by hand, an eval, a budget, a law and SQLite
/// checkpoints, traced with every category. Output: the answer.
async fn fixture_a(dir: &Path, out: Out, content: bool) -> String {
    std::fs::create_dir_all(dir).unwrap();
    let db = dir.join("office.db");
    let embedder: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(DIMS));
    let (store, vectors, report) = agent::open(&db, &docs(), embedder.as_ref()).await.unwrap();
    assert!(report.failed.is_empty(), "{:?}", report.failed);
    let kit = agent::kit(&store, &vectors, embedder, "user-1", "session-1").unwrap();
    let policy = RoutePolicy::parse(
        "office.toml",
        "name = \"office\"\n[[target]]\nid = \"cloud\"\n",
    )
    .unwrap();
    let node = Agent::new("helper")
        .instructions("Answer with the tools.")
        .model("openai/gpt-6-luna")
        .api_key(Secret::new(CANARY.into()).unwrap())
        .transport(Arc::new(ScriptedTransport::new(script())))
        .price(Price::FREE)
        .route(policy)
        .tool(add)
        .toolkit(kit.toolkit())
        .eval_fn("answered", Arc::new(|_out: &Value, _args: &[Value]| Ok(())))
        .eval("answered()");
    let graph = Graph::new("office")
        .agent("helper", node)
        .limits(Limits {
            run: Some(1_000_000),
            ..Limits::default()
        })
        .law("forall x: x == x")
        .checkpoints(Storage::custom(Arc::new(store.clone())))
        .tracing(true)
        .trace_content(content);
    let graph = match out {
        Out::Sink(sink) => graph.trace_to(sink),
        Out::File(path) => graph.trace_file(path),
    };
    let answer = graph
        .run(format!(
            "When can I return order A1234? My key is {CANARY}."
        ))
        .await
        .unwrap();
    answer
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| answer.to_string())
}

fn records(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[tokio::test]
async fn every_trace_category_emits_a_span() {
    let dir = tempfile::tempdir().unwrap();
    let sink = Arc::new(MemorySink::default());
    let answer = fixture_a(dir.path(), Out::Sink(sink.clone()), false).await;
    assert!(answer.contains(ANSWER), "{answer}");
    let records = records(&sink.lines());
    let named: Vec<&Value> = records
        .iter()
        .filter(|r| r["t"] == "start" || r["t"] == "event")
        .collect();
    for category in Trace::ALL {
        assert!(
            named.iter().any(|r| r["cat"] == category.as_str()),
            "no span or event in category {}",
            category.as_str()
        );
    }
}

#[tokio::test]
async fn api_key_never_appears_in_any_trace_output() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Arc::new(MemorySink::default());
    fixture_a(&dir.path().join("m"), Out::Sink(memory.clone()), true).await;
    assert!(!memory.lines().is_empty());
    assert!(
        memory.lines().iter().all(|l| !l.contains(CANARY)),
        "memory sink"
    );
    assert!(memory.lines().iter().any(|l| l.contains("[redacted]")));

    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let into = seen.clone();
    let callback = Arc::new(CallbackSink::new(move |line| {
        into.lock().unwrap().push(line.to_owned());
    }));
    fixture_a(&dir.path().join("c"), Out::Sink(callback), true).await;
    let lines = seen.lock().unwrap().clone();
    assert!(!lines.is_empty());
    assert!(lines.iter().all(|l| !l.contains(CANARY)), "callback sink");

    let path = dir.path().join("f").join("office.jsonl");
    fixture_a(&dir.path().join("f"), Out::File(path.clone()), true).await;
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(!text.contains(CANARY), "JSONL file");
    let tree = read_jsonl(BufReader::new(text.as_bytes()), ViewLimits::default()).unwrap();
    assert!(!render_html(&tree).contains(CANARY), "HTML view");
    assert!(
        !render_terminal(&tree, &TermOptions::default()).contains(CANARY),
        "terminal view"
    );
}

#[tokio::test]
async fn trace_file_views_in_terminal_and_html() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("trace").join("office.jsonl");
    fixture_a(dir.path(), Out::File(path.clone()), false).await;
    let file = std::fs::File::open(&path).unwrap();
    let tree = read_jsonl(BufReader::new(file), ViewLimits::default()).unwrap();
    let terminal = render_terminal(&tree, &TermOptions::default());
    for name in ["graph.run", "model.call", "route.decide", "tool.call"] {
        assert!(
            terminal.contains(name),
            "terminal view lacks {name}:\n{terminal}"
        );
    }
    let html = render_html(&tree);
    assert!(html.contains("<html"), "an HTML page");
    assert!(html.contains("graph.run") && html.contains("route.decide"));
}

#[cfg(feature = "llama")]
#[tokio::test]
async fn private_request_never_reaches_the_network() {
    use troupekit_kit::Knowledge;
    use troupekit_providers::route::NetworkState;
    let gguf = std::env::var_os("TROUPEKIT_TEST_GGUF")
        .map(PathBuf::from)
        .expect("TROUPEKIT_TEST_GGUF is set: run sh scripts/fetch-test-model.sh");
    let dir = tempfile::tempdir().unwrap();
    let transport = Arc::new(ScriptedTransport::new(script()));
    let network = NetworkState::new();
    network.set(Some(true));
    let policy = RoutePolicy::parse(
        "office.toml",
        "name = \"office\"\ndefault = \"cloud\"\n[[target]]\nid = \"local\"\n[[target]]\nid = \"cloud\"\n",
    )
    .unwrap();
    let knowledge = Knowledge::dir(docs())
        .store(dir.path().join("k.db"))
        .local_embedder(gguf.clone());
    let answer = Agent::new("private")
        .instructions("Answer in one sentence.")
        .model("openai/gpt-6-luna")
        .api_key(Secret::new(CANARY.into()).unwrap())
        .transport(transport.clone())
        .local(gguf)
        .route(policy)
        .network(network)
        .private(true)
        .knowledge(knowledge)
        .max_tokens(32)
        .run("How long is the return window?")
        .await
        .unwrap();
    assert!(!answer.is_empty());
    assert!(transport.sent().is_empty(), "no request left the device");
}

#[tokio::test]
async fn fixture_runs_in_a_directory_it_creates() {
    let dir = tempfile::tempdir().unwrap();
    let nested = dir.path().join("a").join("b");
    assert!(!nested.exists());
    let answer = fixture_a(&nested, Out::Sink(Arc::new(MemorySink::default())), false).await;
    assert!(answer.contains(ANSWER), "{answer}");
    assert!(nested.join("office.db").is_file());
}
