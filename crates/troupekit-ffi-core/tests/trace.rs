use serde_json::Value;
use std::sync::{Arc, Mutex};
use troupekit_ffi_core::{
    AgentBuilder, ErrorCode, GraphBuilder, HostError, HostModel, HostModelCaps, HostReply,
    HostRequest, TokenSink, TraceListener,
};

const KEY: &str = "sk-trace-test-7f3a9c1e5b2d4f60";

struct Says(String);

impl HostModel for Says {
    fn id(&self) -> String {
        "host/says".into()
    }

    fn capabilities(&self) -> HostModelCaps {
        HostModelCaps::default()
    }

    fn generate(
        &self,
        _request: HostRequest,
        _sink: Arc<TokenSink>,
    ) -> Result<HostReply, HostError> {
        let mut reply = HostReply::default();
        reply.text = self.0.clone();
        reply.input_tokens = 5;
        reply.output_tokens = 1;
        Ok(reply)
    }
}

#[derive(Default)]
struct Lines(Mutex<Vec<String>>);

impl TraceListener for Lines {
    fn on_record(&self, line: String) {
        self.0.lock().unwrap().push(line);
    }
}

impl Lines {
    fn records(&self) -> Vec<Value> {
        self.0
            .lock()
            .unwrap()
            .iter()
            .map(|l| serde_json::from_str(l).expect("each record is one JSON line"))
            .collect()
    }
}

fn says(name: &str, text: &str) -> Arc<AgentBuilder> {
    AgentBuilder::new(name.into())
        .instructions("Reply.".into())
        .host_model(Arc::new(Says(text.into())))
}

fn categories(records: &[Value]) -> Vec<String> {
    let mut cats: Vec<String> = records
        .iter()
        .filter_map(|r| r["cat"].as_str().map(str::to_string))
        .collect();
    cats.sort();
    cats.dedup();
    cats
}

#[tokio::test]
async fn tracing_on_reaches_listener() {
    let lines = Arc::new(Lines::default());
    let agent = says("echo", "hi")
        .tracing(true)
        .trace_listener(lines.clone())
        .build()
        .await
        .expect("builds");
    assert_eq!(agent.run("hello".into()).await.expect("runs"), "hi");
    let records = lines.records();
    assert!(!records.is_empty(), "a traced run sends records");
    let cats = categories(&records);
    assert!(cats.contains(&"agents".to_string()), "{cats:?}");
    assert!(cats.contains(&"models".to_string()), "{cats:?}");
}

#[tokio::test]
async fn trace_categories_limit_the_records() {
    let lines = Arc::new(Lines::default());
    let agent = says("echo", "hi")
        .trace(vec!["models".into()])
        .trace_listener(lines.clone())
        .build()
        .await
        .expect("builds");
    agent.run("hello".into()).await.expect("runs");
    assert_eq!(categories(&lines.records()), vec!["models".to_string()]);
}

#[tokio::test]
async fn tracing_off_sends_nothing() {
    let lines = Arc::new(Lines::default());
    let agent = says("echo", "hi")
        .trace_listener(lines.clone())
        .build()
        .await
        .expect("builds");
    agent.run("hello".into()).await.expect("runs");
    assert!(lines.records().is_empty());
}

#[tokio::test]
async fn unknown_trace_category_refused() {
    let err = says("echo", "hi")
        .trace(vec!["tools".into(), "bogus".into()])
        .build()
        .await
        .expect_err("refused");
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(err.message.contains("`bogus`"), "{}", err.message);
    assert!(err.message.contains("routing"), "{}", err.message);
}

#[tokio::test]
async fn trace_never_holds_key() {
    let lines = Arc::new(Lines::default());
    let agent = says("echo", &format!("the key is {KEY}"))
        .api_key(KEY.into())
        .tracing(true)
        .trace_content(true)
        .trace_listener(lines.clone())
        .build()
        .await
        .expect("builds");
    agent.run(format!("repeat {KEY}")).await.expect("runs");
    let all = lines.0.lock().unwrap().clone();
    assert!(!all.is_empty());
    for line in &all {
        assert!(!line.contains(KEY), "key in a span: {line}");
    }
}

#[tokio::test]
async fn trace_jsonl_writes_the_file() {
    let dir = std::env::temp_dir().join(format!("troupekit-ffi-trace-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("run.jsonl");
    let _ = std::fs::remove_file(&path);
    let agent = says("echo", "hi")
        .tracing(true)
        .trace_jsonl(path.to_string_lossy().into_owned())
        .build()
        .await
        .expect("builds");
    agent.run("hello".into()).await.expect("runs");
    drop(agent);
    let text = std::fs::read_to_string(&path).expect("trace file written");
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(text.lines().count() > 0);
    for line in text.lines() {
        serde_json::from_str::<Value>(line).expect("one JSON record per line");
    }
}

#[tokio::test]
async fn graph_tracing_reaches_listener() {
    let lines = Arc::new(Lines::default());
    let graph = GraphBuilder::new("pair".into())
        .agent("a".into(), says("a", "hi"))
        .tracing(true)
        .trace_listener(lines.clone())
        .build()
        .await
        .expect("builds");
    graph.run("hello".into()).await.expect("runs");
    let cats = categories(&lines.records());
    assert!(cats.contains(&"graphs".to_string()), "{cats:?}");
}

#[tokio::test]
async fn graph_unknown_trace_category_refused() {
    let err = GraphBuilder::new("pair".into())
        .agent("a".into(), says("a", "hi"))
        .trace(vec!["bogus".into()])
        .build()
        .await
        .expect_err("refused");
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    assert!(err.message.contains("`bogus`"), "{}", err.message);
}
