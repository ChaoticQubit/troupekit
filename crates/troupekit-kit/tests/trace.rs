use crate::common::*;
use serde_json::Value;
use std::sync::Arc;
use troupekit::prelude::Trace;
use troupekit::{Agent, Graph};
use troupekit_trace::MemorySink;

fn echo(name: &str) -> Agent {
    Agent::new(name)
        .instructions("Echo the user.")
        .model("fake/m")
        .model_client(echo_model())
}

fn records(lines: &[String]) -> Vec<Value> {
    lines
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn file_lines(path: &std::path::Path) -> Vec<String> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .map(str::to_owned)
        .collect()
}

fn names(records: &[Value], t: &str) -> Vec<String> {
    records
        .iter()
        .filter(|r| r["t"] == t)
        .map(|r| r["name"].as_str().unwrap().to_owned())
        .collect()
}

#[tokio::test]
async fn tracing_true_writes_jsonl_with_all_categories_enabled() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("echo.jsonl");
    let outcome = echo("echo")
        .tracing(true)
        .trace_file(&path)
        .run_with("hi")
        .await
        .unwrap();
    assert_eq!(outcome.trace_path(), Some(path.as_path()));
    let records = records(&file_lines(&path));
    let header = &records[0];
    assert_eq!(header["t"], "header");
    assert_eq!(header["name"], "echo");
    assert_eq!(header["content"], false);
    let categories: Vec<&str> = header["categories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap())
        .collect();
    assert_eq!(categories.len(), 11);
    for t in Trace::ALL {
        assert!(categories.contains(&t.as_str()), "missing {}", t.as_str());
    }
    let run = records
        .iter()
        .find(|r| r["t"] == "start" && r["name"] == "agent.run")
        .expect("an agent.run root span");
    assert_eq!(run["parent"], 0);
    assert_eq!(run["cat"], "agents");
    let end = records
        .iter()
        .find(|r| r["t"] == "end" && r["span"] == run["span"])
        .expect("the run span ends");
    assert!(end["fields"]["run_id"].as_str().unwrap().len() > 8);
    assert!(names(&records, "start").contains(&"model.call".to_owned()));
}

#[tokio::test]
async fn build_spans_are_inside_the_trace() {
    let sink = Arc::new(MemorySink::default());
    let built = echo("echo")
        .law("forall x: x == x")
        .tracing(true)
        .trace_to(sink.clone())
        .build()
        .await
        .unwrap();
    built.run("hi").await.unwrap();
    built.run("again").await.unwrap();
    drop(built);
    let records = records(&sink.lines());
    let trace = records[0]["trace"].clone();
    assert!(records.iter().all(|r| r["trace"] == trace), "one trace");
    let load = records
        .iter()
        .position(|r| r["t"] == "event" && r["name"] == "law.load")
        .expect("a law.load event at build");
    assert_eq!(records[load]["fields"]["source"], "inline");
    let first_run = records
        .iter()
        .position(|r| r["t"] == "start" && r["name"] == "agent.run")
        .unwrap();
    assert!(load < first_run, "build records come before the first run");
    let runs = names(&records, "start")
        .iter()
        .filter(|n| *n == "agent.run")
        .count();
    assert_eq!(runs, 2, "one root span per run");
}

#[tokio::test]
async fn trace_list_limits_categories() {
    let sink = Arc::new(MemorySink::default());
    echo("echo")
        .trace([Trace::Agents])
        .trace_to(sink.clone())
        .run("hi")
        .await
        .unwrap();
    let records = records(&sink.lines());
    assert_eq!(records[0]["categories"], serde_json::json!(["agents"]));
    let named = records.iter().filter(|r| r["t"] == "start" || r["t"] == "event");
    assert!(named.clone().count() > 0);
    assert!(named.into_iter().all(|r| r["cat"] == "agents"));
    assert!(names(&records, "start").contains(&"agent.run".to_owned()));

    let none = Arc::new(MemorySink::default());
    echo("echo")
        .trace([])
        .trace_to(none.clone())
        .run("hi")
        .await
        .unwrap();
    assert!(none.lines().is_empty(), "an empty list is tracing off");
}

#[tokio::test]
async fn trace_content_off_by_default() {
    let canary = "content-canary-4417";
    let off = Arc::new(MemorySink::default());
    echo("echo")
        .tracing(true)
        .trace_to(off.clone())
        .run(canary)
        .await
        .unwrap();
    assert!(!off.lines().is_empty());
    assert!(off.lines().iter().all(|l| !l.contains(canary)));

    let on = Arc::new(MemorySink::default());
    echo("echo")
        .tracing(true)
        .trace_content(true)
        .trace_to(on.clone())
        .run(canary)
        .await
        .unwrap();
    assert!(on.lines().iter().any(|l| l.contains(canary)));
}

#[tokio::test]
async fn api_key_is_scrubbed_from_spans() {
    let key = "sk-kit-trace-canary-5521";
    let sink = Arc::new(MemorySink::default());
    echo("echo")
        .api_key(secret(key))
        .tracing(true)
        .trace_content(true)
        .trace_to(sink.clone())
        .run(format!("my key is {key}"))
        .await
        .unwrap();
    let lines = sink.lines();
    assert!(lines.iter().all(|l| !l.contains(key)), "key leaked");
    assert!(lines.iter().any(|l| l.contains("[redacted]")));
}

#[tokio::test]
async fn graph_tracing_covers_agent_nodes() {
    let sink = Arc::new(MemorySink::default());
    Graph::new("pipe")
        .agent("first", echo("first"))
        .agent("second", echo("second"))
        .edge("first", "second")
        .tracing(true)
        .trace_to(sink.clone())
        .run("hi")
        .await
        .unwrap();
    let records = records(&sink.lines());
    assert_eq!(records[0]["name"], "pipe");
    let starts = names(&records, "start");
    assert_eq!(starts.iter().filter(|n| *n == "graph.run").count(), 1);
    assert!(starts.contains(&"graph.step".to_owned()));
    assert!(starts.contains(&"model.call".to_owned()));
    let root = records
        .iter()
        .find(|r| r["t"] == "start" && r["name"] == "graph.run")
        .unwrap();
    assert_eq!(root["parent"], 0);

    let err = Graph::new("pipe")
        .agent("first", echo("first").tracing(true))
        .build()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("tracing"), "{err}");
}

#[tokio::test]
async fn trace_path_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nested").join("t.jsonl");
    let built = echo("echo")
        .tracing(true)
        .trace_file(&path)
        .build()
        .await
        .unwrap();
    assert_eq!(built.trace_path(), Some(path.as_path()));
    assert!(path.exists());

    let to_sink = echo("echo")
        .tracing(true)
        .trace_to(Arc::new(MemorySink::default()))
        .build()
        .await
        .unwrap();
    assert_eq!(to_sink.trace_path(), None);

    let off_path = dir.path().join("off.jsonl");
    let off = echo("echo").trace_file(&off_path).build().await.unwrap();
    assert_eq!(off.trace_path(), None);
    assert!(!off_path.exists(), "no file without tracing");
    let outcome = off.run_with("hi").await.unwrap();
    assert_eq!(outcome.trace_path(), None);

    let dir_graph = dir.path().join("g.jsonl");
    let graph = Graph::new("pipe")
        .agent("first", echo("first"))
        .tracing(true)
        .trace_file(&dir_graph)
        .build()
        .await
        .unwrap();
    assert_eq!(graph.trace_path(), Some(dir_graph.as_path()));
    let report = graph.run_with("hi").await.unwrap();
    assert_eq!(report.trace_path(), Some(dir_graph.as_path()));
}
