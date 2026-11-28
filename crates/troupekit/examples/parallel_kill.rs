use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit::agent::{Limits, PersonaLoader, SimpleIo};
use troupekit::core::clock::SystemClock;
use troupekit::core::ids::RunId;
use troupekit::core::store::{BlobStore, CheckpointStore, JobState};
use troupekit::graph::{GraphRun, SimpleGraphIo, Workflow};
use troupekit::host::TickOptions;
use troupekit::{SqliteStore, Troupe};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSpec};

const WORKFLOW: &str = r#"
name = "parallel"
start = "seed"
[state]
goal = "string"
a = "string"
b = "string"
[[nodes]]
id = "seed"
kind = "tool"
tool = "seed"
[[nodes]]
id = "left"
kind = "tool"
tool = "tool_a"
writes = "a"
[[nodes]]
id = "right"
kind = "tool"
tool = "tool_b"
writes = "b"
[[nodes]]
id = "join"
kind = "join"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "seed"
to = "left"
[[edges]]
from = "seed"
to = "right"
[[edges]]
from = "left"
to = "join"
[[edges]]
from = "right"
to = "join"
[[edges]]
from = "join"
to = "done"
"#;

const LOG_A: &str = "example/calls/a";
const LOG_B: &str = "example/calls/b";
const DEADLINE: Duration = Duration::from_secs(10);

/// A tool that appends its idempotency key to the call log `log` in the
/// blob table and, with `gate` set, waits until that file exists.
/// Answers `answer`. The parent reads the logs to count each branch's runs.
struct Logged {
    spec: ToolSpec,
    log: &'static str,
    answer: &'static str,
    blobs: Arc<dyn BlobStore>,
    gate: Option<PathBuf>,
}

#[async_trait]
impl Tool for Logged {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, _args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let key = ctx.idempotency_key.clone().unwrap_or_default();
        let mut lines = self.blobs.get(self.log).await.unwrap_or_default();
        lines.extend_from_slice(format!("{key}\n").as_bytes());
        self.blobs
            .put(self.log, lines)
            .await
            .map_err(|e| ToolError::Failed(e.to_string()))?;
        if let Some(gate) = &self.gate {
            while !gate.exists() {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        Ok(ToolOutput::text(
            serde_json::to_string(self.answer).expect("a string serializes"),
        ))
    }
}

fn io(troupe: &Troupe, gate: &Path) -> Arc<SimpleGraphIo> {
    let mut tools = ToolRegistry::new();
    let rows = [
        ("seed", "example/calls/seed", "go", None),
        ("tool_a", LOG_A, "A", None),
        ("tool_b", LOG_B, "B", Some(gate.to_path_buf())),
    ];
    for (name, log, answer, gate) in rows {
        let spec = ToolSpec {
            name: name.into(),
            description: format!("the {name} node of the parallel kill example"),
            schema: json!({"type": "object"}),
            hosted: None,
        };
        let blobs = troupe.blobs();
        let tool = Logged {
            spec,
            log,
            answer,
            blobs,
            gate,
        };
        tools.register(Arc::new(tool)).expect("valid tool");
    }
    let mut agent = SimpleIo::new(tools);
    agent.ledger = troupe.ledger();
    agent.clock = Arc::new(SystemClock);
    Arc::new(SimpleGraphIo::new(
        Arc::new(agent),
        PersonaLoader::new(Vec::new()),
    ))
}

async fn open(path: &Path) -> Troupe {
    Troupe::builder().sqlite(path).build().await.expect("open")
}

async fn start(path: &Path, gate: &Path) -> RunId {
    let troupe = open(path).await;
    let wf = Workflow::parse("parallel.toml", WORKFLOW).expect("valid workflow");
    let input: Map<String, Value> = [("goal".to_string(), json!("kill b"))]
        .into_iter()
        .collect();
    let io = io(&troupe, gate);
    let run = GraphRun::new(wf, input, Limits::default(), io.as_ref())
        .await
        .expect("run");
    troupe.runner(io).enqueue(&run).await.expect("enqueue");
    run.run_id()
}

async fn work(path: &Path, id: RunId, gate: &Path) {
    let troupe = open(path).await;
    let runner = troupe.runner(io(&troupe, gate)).with_lease_ms(200);
    let store = SqliteStore::open(path).await.expect("open");
    let start = Instant::now();
    while store.job(id).await.expect("job") != Some(JobState::Done) {
        assert!(start.elapsed() < DEADLINE, "worker ran past its deadline");
        runner.tick(TickOptions::default()).await.expect("tick");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn spawn(path: &Path, id: RunId, gate: &Path) -> Child {
    Command::new(std::env::current_exe().expect("own path"))
        .env("PARALLEL_KILL_DB", path)
        .env("PARALLEL_KILL_RUN", id.to_string())
        .env("PARALLEL_KILL_GATE", gate)
        .spawn()
        .expect("spawn worker")
}

async fn calls(store: &SqliteStore, log: &str) -> Vec<String> {
    let bytes = store.get(log).await.unwrap_or_default();
    String::from_utf8(bytes)
        .expect("keys are text")
        .lines()
        .map(str::to_string)
        .collect()
}

/// Returns true once branch A's write of the step in flight is kept and
/// tool B has logged its call: the moment the parent kills the worker.
/// Reads only; store errors panic, as everywhere in this example.
async fn at_kill_point(store: &SqliteStore, id: RunId) -> bool {
    let step = store
        .latest(id)
        .await
        .expect("read")
        .map_or(0, |cp| cp.step + 1);
    let writes = store.branch_writes(id, step).await.expect("read");
    writes.iter().any(|w| w.node() == "left") && !calls(store, LOG_B).await.is_empty()
}

async fn parent() -> ExitCode {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("parallel.db");
    let gate = dir.path().join("gate");
    let id = start(&path, &gate).await;
    println!("1. run {id} created and enqueued in {}", path.display());
    let mut first = spawn(&path, id, &gate);
    let watcher = SqliteStore::open(&path).await.expect("open");
    let begin = Instant::now();
    while !at_kill_point(&watcher, id).await {
        if begin.elapsed() > DEADLINE {
            let _ = first.kill();
            let _ = first.wait();
            println!("branch A was never written while tool B ran");
            return ExitCode::FAILURE;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    first.kill().expect("kill");
    let _ = first.wait();
    println!("2. worker killed with SIGKILL after branch A was kept, inside tool B");
    std::fs::write(&gate, b"open").expect("open the gate");
    let status = spawn(&path, id, &gate).wait().expect("wait");
    println!("3. second worker finished: {status}");
    let cp = watcher.latest(id).await.expect("read").expect("checkpoint");
    let done = GraphRun::from_checkpoint(&cp).expect("restore");
    let (a, b) = (calls(&watcher, LOG_A).await, calls(&watcher, LOG_B).await);
    println!(
        "4. tool A ran {} time(s), tool B ran {} time(s)",
        a.len(),
        b.len()
    );
    let state = done.state();
    let joined = state.get("a") == Some(&json!("A")) && state.get("b") == Some(&json!("B"));
    let once = a.len() == 1 && b.len() == 2 && b[0] == b[1];
    println!("5. joined state {}", Value::Object(state.clone()));
    if status.success() && joined && once {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Starts a two-branch run in a SQLite file, kills the worker process with
/// SIGKILL after branch A is kept and while branch B runs, and resumes it
/// in a new process. Exits 0 only when A ran once, B ran twice with one
/// idempotency key, and the joined state holds both answers.
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    match (
        std::env::var("PARALLEL_KILL_DB"),
        std::env::var("PARALLEL_KILL_RUN"),
        std::env::var("PARALLEL_KILL_GATE"),
    ) {
        (Ok(path), Ok(run), Ok(gate)) => {
            let id: RunId = serde_json::from_value(json!(run)).expect("run id");
            work(Path::new(&path), id, Path::new(&gate)).await;
            ExitCode::SUCCESS
        }
        _ => parent().await,
    }
}
