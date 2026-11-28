#[cfg(feature = "sqlite")]
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit_ffi_core::{
    AgentBuilder, ErrorCode, EventListener, FfiEvent, FfiEventKind, GraphBuilder, HostError,
    HostHeader, HostHttpRequest, HostModel, HostModelCaps, HostReply, HostRequest, HostTool,
    HostTransport, HttpSink, TokenSink,
};

const CASES: [&str; 26] = [
    "version_matches_core",
    "missing_model_is_config_error",
    "missing_key_names_env_not_value",
    "host_model_run_returns_text",
    "host_tool_called_with_json_args",
    "host_tool_error_reaches_model",
    "host_tool_throw_does_not_crash",
    "stream_yields_deltas_then_finished",
    "listener_sees_same_events_as_pull",
    "start_returns_run_id_before_model_call",
    "resume_after_handles_dropped",
    "resume_for_zero_budget_is_pending",
    "cancel_keeps_checkpoint_and_resumes",
    "toml_agent_runs",
    "toml_over_cap_refused",
    "graph_agent_then_step_runs",
    "knowledge_with_host_embedder_finds_doc",
    "memory_facts_persist_between_runs",
    "error_codes_are_stable",
    "laws_path_with_dotdot_refused",
    "host_transport_refuses_other_origin",
    "host_transport_streams_chunks",
    "full_buffer_drops_deltas_not_lifecycle",
    "callback_from_worker_thread_no_deadlock",
    "concurrent_runs_on_one_handle",
    "host_request_debug_hides_key",
];

#[test]
fn every_case_is_implemented() {
    let listed: Vec<&str> = include_str!("../contract/cases.txt")
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(listed, CASES.to_vec());
    let runner = include_str!("contract.rs");
    for case in CASES {
        assert!(
            runner.contains(&format!("async fn {case}()"))
                || runner.contains(&format!("fn {case}()")),
            "{case} has no runner test"
        );
    }
}

type Reply = (Vec<String>, String);

struct Scripted {
    caps: HostModelCaps,
    replies: Mutex<Vec<Reply>>,
    fallback: String,
    seen: Mutex<Vec<HostRequest>>,
}

impl HostModel for Scripted {
    fn id(&self) -> String {
        "host/scripted".into()
    }

    fn capabilities(&self) -> HostModelCaps {
        self.caps.clone()
    }

    fn generate(&self, request: HostRequest, sink: Arc<TokenSink>) -> Result<HostReply, HostError> {
        self.seen.lock().unwrap().push(request);
        let next = {
            let mut replies = self.replies.lock().unwrap();
            if replies.is_empty() {
                (Vec::new(), self.fallback.clone())
            } else {
                replies.remove(0)
            }
        };
        for delta in next.0 {
            sink.delta(delta);
        }
        let mut reply = HostReply::default();
        reply.text = next.1;
        reply.input_tokens = 4;
        reply.output_tokens = 1;
        Ok(reply)
    }
}

fn model(replies: Vec<Reply>, fallback: &str) -> Arc<Scripted> {
    let mut caps = HostModelCaps::default();
    caps.streaming = true;
    Arc::new(Scripted {
        caps,
        replies: Mutex::new(replies),
        fallback: fallback.into(),
        seen: Mutex::new(Vec::new()),
    })
}

fn says(text: &str) -> Arc<Scripted> {
    model(Vec::new(), text)
}

fn texts_of(model: &Scripted) -> String {
    let seen = model.seen.lock().unwrap();
    seen.iter()
        .flat_map(|r| {
            r.system
                .clone()
                .into_iter()
                .chain(r.messages.iter().map(|m| m.text.clone()))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn agent(model: Arc<Scripted>) -> Arc<AgentBuilder> {
    AgentBuilder::new("helper".into())
        .instructions("Help.".into())
        .host_model(model)
}

struct Add {
    seen: Mutex<Vec<String>>,
}

impl HostTool for Add {
    fn name(&self) -> String {
        "add".into()
    }

    fn description(&self) -> String {
        "Adds a and b.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object","properties":{"a":{"type":"number"},"b":{"type":"number"}},"required":["a","b"]}"#.into()
    }

    fn call(&self, args_json: String) -> Result<String, HostError> {
        self.seen.lock().unwrap().push(args_json.clone());
        let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
        Ok(format!(
            "{}",
            args["a"].as_f64().unwrap() + args["b"].as_f64().unwrap()
        ))
    }
}

struct Failing {
    panics: bool,
}

impl HostTool for Failing {
    fn name(&self) -> String {
        "lookup".into()
    }

    fn description(&self) -> String {
        "Looks an order up.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object"}"#.into()
    }

    fn call(&self, _args_json: String) -> Result<String, HostError> {
        if self.panics {
            panic!("host adapter bug");
        }
        Err(HostError::new("order not found"))
    }
}

fn call(tool: &str, arguments: &str) -> Reply {
    (
        Vec::new(),
        format!(r#"{{"tool": "{tool}", "arguments": {arguments}}}"#),
    )
}

fn answer(text: &str) -> Reply {
    (Vec::new(), format!(r#"{{"answer": "{text}"}}"#))
}

#[cfg(any(feature = "sqlite", feature = "knowledge"))]
fn sqlite_path(dir: &tempfile::TempDir, name: &str) -> String {
    dir.path().join(name).to_string_lossy().into_owned()
}

async fn pull(stream: &troupekit_ffi_core::RunStream) -> Vec<FfiEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await.unwrap() {
        events.push(event);
        assert!(events.len() < 100_000);
    }
    events
}

fn kinds(events: &[FfiEvent]) -> Vec<FfiEventKind> {
    events.iter().map(|e| e.kind.clone()).collect()
}

fn deltas(events: &[FfiEvent]) -> Vec<String> {
    events
        .iter()
        .filter_map(|e| match &e.kind {
            FfiEventKind::TextDelta { text } => Some(text.clone()),
            _ => None,
        })
        .collect()
}

#[test]
fn version_matches_core() {
    assert_eq!(troupekit_ffi_core::version(), env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn missing_model_is_config_error() {
    let err = AgentBuilder::new("helper".into())
        .instructions("Help.".into())
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::Config);
}

#[tokio::test]
async fn missing_key_names_env_not_value() {
    let err = AgentBuilder::new("helper".into())
        .instructions("Help.".into())
        .model("openai/gpt-6-luna".into())
        .api_key_env("TROUPEKIT_CONTRACT_NEVER_SET".into())
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::MissingKey);
    assert!(
        err.message.contains("TROUPEKIT_CONTRACT_NEVER_SET"),
        "{err}"
    );
}

#[tokio::test]
async fn host_model_run_returns_text() {
    let handle = agent(says("hello")).build().await.unwrap();
    assert_eq!(handle.run("hi".into()).await.unwrap(), "hello");
}

#[tokio::test]
async fn host_tool_called_with_json_args() {
    let tool = Arc::new(Add {
        seen: Mutex::new(Vec::new()),
    });
    let scripted = model(vec![call("add", r#"{"a": 1, "b": 2}"#), answer("3")], "");
    let handle = agent(scripted).tool(tool.clone()).build().await.unwrap();
    assert_eq!(handle.run("1 + 2".into()).await.unwrap(), "3");
    let seen = tool.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let args: serde_json::Value = serde_json::from_str(&seen[0]).unwrap();
    assert_eq!(args, serde_json::json!({"a": 1, "b": 2}));
}

#[tokio::test]
async fn host_tool_error_reaches_model() {
    let scripted = model(vec![call("lookup", "{}"), answer("sorry")], "");
    let handle = agent(scripted.clone())
        .tool(Arc::new(Failing { panics: false }))
        .build()
        .await
        .unwrap();
    assert_eq!(handle.run("order 7".into()).await.unwrap(), "sorry");
    assert!(texts_of(&scripted).contains("Error: order not found"));
}

#[tokio::test]
async fn host_tool_throw_does_not_crash() {
    let scripted = model(vec![call("lookup", "{}"), answer("sorry")], "");
    let handle = agent(scripted.clone())
        .tool(Arc::new(Failing { panics: true }))
        .build()
        .await
        .unwrap();
    assert_eq!(handle.run("order 7".into()).await.unwrap(), "sorry");
    assert!(texts_of(&scripted).contains("panicked"));
}

const SSE: &str = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n\
data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
data: {\"choices\":[],\"usage\":{\"prompt_tokens\":3,\"completion_tokens\":2}}\n\n\
data: [DONE]\n\n";

const UNARY: &str = r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"hello"},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2}}"#;

fn stub_catalog(base_url: &str) -> String {
    format!(
        "[[vendor]]\nid = \"stub\"\nname = \"Stub\"\nprotocol = \"openai_compatible\"\nbase_url = \"{base_url}\"\nauth = \"none\"\nopen_models = true\n"
    )
}

#[cfg(feature = "reqwest")]
async fn openai_stub() -> String {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        for _ in 0..16 {
            let Ok((mut socket, _)) = listener.accept().await else {
                return;
            };
            let mut read = Vec::new();
            let mut buf = [0u8; 4096];
            let body_start = loop {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                read.extend_from_slice(&buf[..n]);
                if let Some(at) = read.windows(4).position(|w| w == b"\r\n\r\n") {
                    break at + 4;
                }
            };
            let head = String::from_utf8_lossy(&read[..body_start]).to_ascii_lowercase();
            let length: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .map_or(0, |v| v.trim().parse().unwrap());
            while read.len() < body_start + length {
                let n = socket.read(&mut buf).await.unwrap();
                assert!(n > 0);
                read.extend_from_slice(&buf[..n]);
            }
            let body = String::from_utf8_lossy(&read[body_start..]).replace(' ', "");
            let (kind, reply) = if body.contains("\"stream\":true") {
                ("text/event-stream", SSE)
            } else {
                ("application/json", UNARY)
            };
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: {kind}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                reply.len()
            );
            socket.write_all(response.as_bytes()).await.unwrap();
            socket.shutdown().await.ok();
        }
    });
    format!("http://127.0.0.1:{port}/v1")
}

#[cfg(feature = "reqwest")]
#[tokio::test]
async fn stream_yields_deltas_then_finished() {
    let base = openai_stub().await;
    let handle = AgentBuilder::new("cloud".into())
        .instructions("Say hello.".into())
        .catalog_toml(stub_catalog(&base))
        .model("stub/m".into())
        .build()
        .await
        .unwrap();
    let stream = handle.stream("hi".into()).await.unwrap();
    let events = pull(&stream).await;
    assert_eq!(deltas(&events).concat(), "hello");
    assert!(matches!(
        events.first().unwrap().kind,
        FfiEventKind::RunStarted { .. }
    ));
    assert!(matches!(
        events.last().unwrap().kind,
        FfiEventKind::RunFinished { .. }
    ));
    let last_delta = events
        .iter()
        .rposition(|e| matches!(e.kind, FfiEventKind::TextDelta { .. }))
        .unwrap();
    assert!(last_delta < events.len() - 1);
    assert_eq!(stream.result().await.unwrap().text, "hello");
}

#[cfg(not(feature = "reqwest"))]
#[tokio::test]
async fn stream_yields_deltas_then_finished() {
    let scripted = model(vec![(vec!["hel".into(), "lo".into()], "hello".into())], "");
    let stream = agent(scripted)
        .build()
        .await
        .unwrap()
        .stream("hi".into())
        .await
        .unwrap();
    let events = pull(&stream).await;
    assert_eq!(deltas(&events).concat(), "hello");
    assert!(matches!(
        events.last().unwrap().kind,
        FfiEventKind::RunFinished { .. }
    ));
}

#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<FfiEvent>>,
}

impl EventListener for Recorder {
    fn on_event(&self, event: FfiEvent) {
        self.events.lock().unwrap().push(event);
    }
}

async fn heard_finish(recorder: &Recorder) -> Vec<FfiEvent> {
    for _ in 0..1000 {
        let events = recorder.events.lock().unwrap().clone();
        if events
            .iter()
            .any(|e| matches!(e.kind, FfiEventKind::RunFinished { .. }))
        {
            return events;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    panic!("listener never heard RunFinished");
}

#[tokio::test]
async fn listener_sees_same_events_as_pull() {
    let reply = || vec![(vec!["a".to_string(), "b".to_string()], "ab".to_string())];
    let pulled = agent(model(reply(), "")).build().await.unwrap();
    let pulled = pull(&*pulled.stream("hi".into()).await.unwrap()).await;
    let handle = agent(model(reply(), "")).build().await.unwrap();
    let stream = handle.stream("hi".into()).await.unwrap();
    let recorder = Arc::new(Recorder::default());
    stream.set_listener(recorder.clone()).unwrap();
    assert_eq!(stream.result().await.unwrap().text, "ab");
    let heard = heard_finish(&recorder).await;
    assert_eq!(kinds(&heard), kinds(&pulled));
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn start_returns_run_id_before_model_call() {
    let dir = tempfile::tempdir().unwrap();
    let scripted = says("hello");
    let handle = agent(scripted.clone())
        .sqlite(sqlite_path(&dir, "runs.db"))
        .build()
        .await
        .unwrap();
    let run_id = handle.start("hi".into()).await.unwrap();
    assert_eq!(run_id.len(), 36);
    assert!(scripted.seen.lock().unwrap().is_empty());
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn resume_after_handles_dropped() {
    let dir = tempfile::tempdir().unwrap();
    let path = sqlite_path(&dir, "runs.db");
    let first = agent(says("hello"))
        .sqlite(path.clone())
        .build()
        .await
        .unwrap();
    let run_id = first.start("hi".into()).await.unwrap();
    drop(first);
    let second = agent(says("hello")).sqlite(path).build().await.unwrap();
    assert_eq!(second.resume(run_id).await.unwrap(), "hello");
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn resume_for_zero_budget_is_pending() {
    let dir = tempfile::tempdir().unwrap();
    let scripted = says("hello");
    let handle = agent(scripted.clone())
        .sqlite(sqlite_path(&dir, "runs.db"))
        .build()
        .await
        .unwrap();
    let run_id = handle.start("hi".into()).await.unwrap();
    let progress = handle.resume_for(run_id.clone(), 0).await.unwrap();
    assert_eq!(progress, troupekit_ffi_core::Progress::Pending { run_id });
    assert!(scripted.seen.lock().unwrap().is_empty());
}

#[cfg(feature = "sqlite")]
struct Gated {
    gate: Mutex<mpsc::Receiver<()>>,
    calls: Mutex<u32>,
}

#[cfg(feature = "sqlite")]
impl HostModel for Gated {
    fn id(&self) -> String {
        "host/gated".into()
    }

    fn capabilities(&self) -> HostModelCaps {
        HostModelCaps::default()
    }

    fn generate(
        &self,
        _request: HostRequest,
        _sink: Arc<TokenSink>,
    ) -> Result<HostReply, HostError> {
        let first = {
            let mut calls = self.calls.lock().unwrap();
            *calls += 1;
            *calls == 1
        };
        if first {
            let _ = self
                .gate
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(10));
        }
        let mut reply = HostReply::default();
        reply.text = if first { "late" } else { "back" }.into();
        Ok(reply)
    }
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn cancel_keeps_checkpoint_and_resumes() {
    let dir = tempfile::tempdir().unwrap();
    let (open, gate) = mpsc::channel();
    let gated = Arc::new(Gated {
        gate: Mutex::new(gate),
        calls: Mutex::new(0),
    });
    let handle = AgentBuilder::new("helper".into())
        .instructions("Help.".into())
        .host_model(gated.clone())
        .sqlite(sqlite_path(&dir, "runs.db"))
        .build()
        .await
        .unwrap();
    let stream = handle.stream("hi".into()).await.unwrap();
    let run_id = stream.run_id();
    while let Some(event) = stream.next().await.unwrap() {
        if matches!(event.kind, FfiEventKind::StepStarted { .. }) {
            break;
        }
    }
    for _ in 0..500 {
        if *gated.calls.lock().unwrap() == 1 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    stream.cancel();
    let err = stream.result().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::Cancelled);
    assert_eq!(err.run_id.as_deref(), Some(run_id.as_str()));
    open.send(()).unwrap();
    assert_eq!(handle.resume(run_id).await.unwrap(), "back");
}

const PERSONA: &str = r#"
name = "greeter"
display_name = "Greeter"
role = "Greets."
goal = "Say hello."
instructions = "Say hello to the user."
"#;

#[tokio::test]
async fn toml_agent_runs() {
    let handle = AgentBuilder::from_toml(PERSONA.into())
        .host_model(says("hello"))
        .build()
        .await
        .unwrap();
    assert_eq!(handle.run("hi".into()).await.unwrap(), "hello");
}

#[tokio::test]
async fn toml_over_cap_refused() {
    let cap = troupekit_ffi_core::RuntimeSettings::default().max_input_bytes;
    let err = AgentBuilder::from_toml("x".repeat(cap + 1))
        .host_model(says("hello"))
        .build()
        .await
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

struct Shout;

impl HostTool for Shout {
    fn name(&self) -> String {
        "shout".into()
    }

    fn description(&self) -> String {
        "Upper-cases the writer's text.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object","properties":{"writer":{"type":"string"}},"required":["writer"]}"#
            .into()
    }

    fn call(&self, args_json: String) -> Result<String, HostError> {
        let args: serde_json::Value = serde_json::from_str(&args_json).unwrap();
        Ok(args["writer"].as_str().unwrap().to_uppercase())
    }
}

#[tokio::test]
async fn graph_agent_then_step_runs() {
    let writer = AgentBuilder::new("writer".into())
        .instructions("Write.".into())
        .host_model(says("draft"));
    let handle = GraphBuilder::new("loud".into())
        .agent("writer".into(), writer)
        .step("shout".into(), Arc::new(Shout))
        .edge("writer".into(), "shout".into())
        .build()
        .await
        .unwrap();
    let output = handle.run("go".into()).await.unwrap();
    assert!(output.contains("DRAFT"), "{output}");
}

#[cfg(feature = "knowledge")]
struct Lengths;

#[cfg(feature = "knowledge")]
impl troupekit_ffi_core::HostEmbedder for Lengths {
    fn dims(&self) -> u32 {
        4
    }

    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, HostError> {
        Ok(texts
            .iter()
            .map(|t| vec![t.len() as f32, 1.0, 0.5, 0.25])
            .collect())
    }
}

#[cfg(feature = "knowledge")]
#[tokio::test]
async fn knowledge_with_host_embedder_finds_doc() {
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    std::fs::write(
        docs.join("returns.md"),
        "# Returns\n\nReturns take 30 days.\n",
    )
    .unwrap();
    let mut config = troupekit_ffi_core::KnowledgeConfig::new(docs.to_string_lossy());
    config.store = Some(sqlite_path(&dir, "index.db"));
    let scripted = model(
        vec![
            call("search_knowledge", r#"{"query": "returns"}"#),
            answer("30 days"),
        ],
        "",
    );
    let handle = agent(scripted.clone())
        .knowledge(config, Some(Arc::new(Lengths)))
        .build()
        .await
        .unwrap();
    assert_eq!(
        handle
            .run("How long do returns take?".into())
            .await
            .unwrap(),
        "30 days"
    );
    let seen = scripted.seen.lock().unwrap().clone();
    assert_eq!(seen.len(), 2);
    let result = &seen[1].messages.last().unwrap().text;
    assert!(result.contains("30 days"), "{result}");
}

#[cfg(feature = "knowledge")]
#[tokio::test]
async fn memory_facts_persist_between_runs() {
    let dir = tempfile::tempdir().unwrap();
    let path = sqlite_path(&dir, "memory.db");
    let memory = || {
        let mut config = troupekit_ffi_core::MemoryConfig::new("ada", "s1");
        config.sqlite = Some(path.clone());
        config
    };
    let first = model(
        vec![
            call("remember_fact", r#"{"key": "color", "value": "teal"}"#),
            answer("noted"),
        ],
        "",
    );
    let handle = agent(first).memory(memory()).build().await.unwrap();
    assert_eq!(handle.run("I like teal.".into()).await.unwrap(), "noted");
    drop(handle);
    let second = model(vec![answer("teal")], "");
    let handle = agent(second.clone())
        .memory(memory())
        .build()
        .await
        .unwrap();
    handle.run("What color do I like?".into()).await.unwrap();
    assert!(texts_of(&second).contains("teal"));
}

#[test]
fn error_codes_are_stable() {
    let codes = [
        (ErrorCode::Config, 1),
        (ErrorCode::MissingKey, 2),
        (ErrorCode::Catalog, 3),
        (ErrorCode::Persona, 4),
        (ErrorCode::Agent, 5),
        (ErrorCode::Tool, 6),
        (ErrorCode::Workflow, 7),
        (ErrorCode::Graph, 8),
        (ErrorCode::Store, 9),
        (ErrorCode::Context, 10),
        (ErrorCode::Failed, 11),
        (ErrorCode::Paused, 12),
        (ErrorCode::Retries, 13),
        (ErrorCode::Cancelled, 14),
        (ErrorCode::HostCallback, 15),
        (ErrorCode::InvalidArgument, 16),
        (ErrorCode::Unsupported, 17),
        (ErrorCode::Panic, 18),
        (ErrorCode::Internal, 99),
    ];
    for (code, number) in codes {
        assert_eq!(code as u32, number, "{code:?}");
    }
}

#[tokio::test]
async fn laws_path_with_dotdot_refused() {
    let err = agent(says("hello"))
        .laws("../laws/LAWS.bend".into())
        .build()
        .await
        .unwrap_err();
    assert!(err.message.contains(".."), "{err}");
}

struct Replies {
    final_url: Option<String>,
    chunk: usize,
    seen: Mutex<Vec<HostHttpRequest>>,
}

impl HostTransport for Replies {
    fn send(&self, request: HostHttpRequest, sink: Arc<HttpSink>) -> Result<(), HostError> {
        self.seen.lock().unwrap().push(request.clone());
        let streamed = String::from_utf8_lossy(&request.body)
            .replace(' ', "")
            .contains("\"stream\":true");
        let (kind, body) = if streamed {
            ("text/event-stream", SSE)
        } else {
            ("application/json", UNARY)
        };
        let url = self.final_url.clone().unwrap_or(request.url);
        sink.head(200, url, vec![HostHeader::new("content-type", kind)]);
        for part in body.as_bytes().chunks(self.chunk) {
            sink.chunk(part.to_vec());
        }
        Ok(())
    }
}

fn via_host(transport: Arc<Replies>) -> Arc<AgentBuilder> {
    AgentBuilder::new("cloud".into())
        .instructions("Say hello.".into())
        .catalog_toml(stub_catalog("https://api.stub.test/v1"))
        .model("stub/m".into())
        .retries(0)
        .host_transport(transport)
}

#[tokio::test]
async fn host_transport_refuses_other_origin() {
    let transport = Arc::new(Replies {
        final_url: Some("https://evil.test/v1/chat/completions".into()),
        chunk: 64,
        seen: Mutex::new(Vec::new()),
    });
    let handle = via_host(transport.clone()).build().await.unwrap();
    let err = handle.run("hi".into()).await.unwrap_err();
    assert!(err.message.contains("origin"), "{err}");
    let seen = transport.seen.lock().unwrap().clone();
    assert!(seen
        .iter()
        .all(|r| r.url.starts_with("https://api.stub.test/v1/")));
}

#[tokio::test]
async fn host_transport_streams_chunks() {
    let transport = Arc::new(Replies {
        final_url: None,
        chunk: 7,
        seen: Mutex::new(Vec::new()),
    });
    let handle = via_host(transport).build().await.unwrap();
    let stream = handle.stream("hi".into()).await.unwrap();
    let events = pull(&stream).await;
    assert_eq!(deltas(&events).concat(), "hello");
    assert_eq!(stream.result().await.unwrap().text, "hello");
}

#[tokio::test]
async fn full_buffer_drops_deltas_not_lifecycle() {
    let many: Vec<String> = (0..10_000).map(|_| "a".to_string()).collect();
    let scripted = model(vec![(many, "a".repeat(10_000))], "");
    let stream = agent(scripted)
        .build()
        .await
        .unwrap()
        .stream("hi".into())
        .await
        .unwrap();
    let result = stream.result().await.unwrap();
    assert_eq!(result.text.len(), 10_000);
    let events = pull(&stream).await;
    assert!(stream.dropped_deltas() > 0);
    assert!(deltas(&events).len() < 10_000);
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, FfiEventKind::RunStarted { .. })));
    assert!(events
        .iter()
        .any(|e| matches!(e.kind, FfiEventKind::RunFinished { .. })));
}

struct CallsBack {
    inner: Arc<troupekit_ffi_core::AgentHandle>,
}

impl HostTool for CallsBack {
    fn name(&self) -> String {
        "ask_helper".into()
    }

    fn description(&self) -> String {
        "Asks another agent.".into()
    }

    fn schema_json(&self) -> String {
        r#"{"type":"object"}"#.into()
    }

    fn call(&self, _args_json: String) -> Result<String, HostError> {
        let local = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        local
            .block_on(self.inner.run("hi".into()))
            .map_err(|e| HostError::new(e.message))
    }
}

#[tokio::test]
async fn callback_from_worker_thread_no_deadlock() {
    let inner = agent(says("inner answer")).build().await.unwrap();
    let outer = agent(model(vec![call("ask_helper", "{}"), answer("done")], ""))
        .tool(Arc::new(CallsBack { inner }))
        .build()
        .await
        .unwrap();
    let run = tokio::time::timeout(Duration::from_secs(30), outer.run("go".into()));
    assert_eq!(run.await.expect("no deadlock").unwrap(), "done");
}

#[tokio::test]
async fn concurrent_runs_on_one_handle() {
    let handle = agent(says("hello")).build().await.unwrap();
    let mut runs = Vec::new();
    for i in 0..8 {
        let handle = handle.clone();
        runs.push(tokio::spawn(
            async move { handle.run(format!("hi {i}")).await },
        ));
    }
    for run in runs {
        assert_eq!(run.await.unwrap().unwrap(), "hello");
    }
}

#[test]
fn host_request_debug_hides_key() {
    let mut request = HostHttpRequest::default();
    request.url = "https://api.stub.test/v1/chat/completions".into();
    request.headers = vec![
        HostHeader::new("authorization", "Bearer sk-contract-secret"),
        HostHeader::new("content-type", "application/json"),
    ];
    let text = format!("{request:?}");
    assert!(!text.contains("sk-contract-secret"), "{text}");
    assert!(text.contains("application/json"), "{text}");
}
