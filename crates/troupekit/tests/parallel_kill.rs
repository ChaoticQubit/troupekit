#![cfg(unix)]

use crate::common;

use async_trait::async_trait;
use common::graph_io;
use serde_json::{json, Map, Value};
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit::agent::Limits;
use troupekit::core::clock::SystemClock;
use troupekit::core::error::CoreError;
use troupekit::core::ids::RunId;
use troupekit::core::store::{BlobStore, Checkpoint, CheckpointStore, JobState};
use troupekit::graph::{GraphRun, GraphStatus, Workflow};
use troupekit::host::{Runner, TickOptions};
use troupekit::{SqliteStore, Troupe};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSpec};

const ROLE: &str = "TROUPEKIT_E2E_ROLE";
const DB: &str = "TROUPEKIT_E2E_DB";
const RUN: &str = "TROUPEKIT_E2E_RUN";
const GATE: &str = "TROUPEKIT_E2E_GATE";
const STORE: &str = "TROUPEKIT_E2E_STORE";
const LOG_A: &str = "e2e/calls/a";
const LOG_B: &str = "e2e/calls/b";
const DEADLINE: Duration = Duration::from_secs(10);
const LEASE_MS: i64 = 200;

const PARALLEL: &str = r#"
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

fn runs() -> usize {
    std::env::var("TROUPEKIT_E2E_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3)
}

/// Which checkpoint store the worker steps against: the SQLite store
/// itself, which keeps branch writes, or a wrapper that keeps whole steps
/// only, so the control test can show branch A running twice.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Durable,
    WholeSteps,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Durable => "durable",
            Kind::WholeSteps => "whole",
        }
    }
}

/// A tool-only node: it appends its idempotency key to a call log in the
/// blob table, then, when `gate` is set, waits until that file exists.
/// Answers its own fixed text.
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
        self.blobs.put(self.log, lines).await.unwrap();
        if let Some(gate) = &self.gate {
            let start = Instant::now();
            while !gate.exists() {
                assert!(start.elapsed() < DEADLINE, "the gate never opened");
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        }
        Ok(ToolOutput::text(self.answer))
    }
}

fn spec(name: &str) -> ToolSpec {
    ToolSpec {
        name: name.into(),
        description: format!("the {name} node of the parallel kill test"),
        schema: json!({"type": "object"}),
        hosted: None,
    }
}

/// The three tools of `PARALLEL`: `seed`, `tool_a` that answers at once
/// and `tool_b` that waits for the `gate` file. Both branch tools log
/// their idempotency keys in the troupe's blob table.
fn parallel_tools(troupe: &Troupe, gate: &Path) -> ToolRegistry {
    let logged = |name: &str, log, answer, gate: Option<PathBuf>| -> Arc<dyn Tool> {
        Arc::new(Logged { spec: spec(name), log, answer, blobs: troupe.blobs(), gate })
    };
    let mut tools = ToolRegistry::new();
    tools.register(logged("seed", "e2e/calls/seed", "go", None)).unwrap();
    tools.register(logged("tool_a", LOG_A, "A", None)).unwrap();
    tools.register(logged("tool_b", LOG_B, "B", Some(gate.to_path_buf()))).unwrap();
    tools
}

/// Wraps the troupe's checkpoint store and keeps whole steps only:
/// `save_branch` keeps its default and reports a branch write as not
/// durable. Every other call goes to the wrapped store.
struct WholeSteps(Arc<dyn CheckpointStore>);

#[async_trait]
impl CheckpointStore for WholeSteps {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        self.0.save(cp).await
    }
    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        self.0.latest(run_id).await
    }
    async fn history(&self, run_id: RunId) -> Result<Vec<Checkpoint>, CoreError> {
        self.0.history(run_id).await
    }
}

async fn troupe_on(path: &Path) -> Troupe {
    Troupe::builder().sqlite(path).build().await.unwrap()
}

fn runner(troupe: &Troupe, kind: Kind, gate: &Path) -> Runner {
    let io = graph_io(troupe, parallel_tools(troupe, gate), Arc::new(SystemClock));
    let checkpoints: Arc<dyn CheckpointStore> = match kind {
        Kind::Durable => troupe.checkpoints(),
        Kind::WholeSteps => Arc::new(WholeSteps(troupe.checkpoints())),
    };
    Runner::new(troupe.queue(), checkpoints, io).with_lease_ms(LEASE_MS)
}

async fn new_run(path: &Path, gate: &Path) -> RunId {
    let troupe = troupe_on(path).await;
    let io = graph_io(&troupe, parallel_tools(&troupe, gate), Arc::new(SystemClock));
    let wf = Workflow::parse("parallel.toml", PARALLEL).unwrap();
    let input: Map<String, Value> = [("goal".to_string(), json!("kill b"))].into_iter().collect();
    let run = GraphRun::new(wf, input, Limits::default(), io.as_ref()).await.unwrap();
    troupe.runner(io).enqueue(&run).await.unwrap();
    run.run_id()
}

/// Kills and reaps the worker when dropped, so a failed assertion never
/// leaves a child process alive.
struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

impl Worker {
    fn spawn(path: &Path, id: RunId, kind: Kind, gate: &Path) -> Worker {
        let child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--ignored",
                "--exact",
                "parallel_kill::parallel_kill_worker",
                "--test-threads=1",
                "--quiet",
            ])
            .env(ROLE, "worker")
            .env(DB, path)
            .env(RUN, id.to_string())
            .env(GATE, gate)
            .env(STORE, kind.name())
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap();
        Worker(child)
    }

    fn kill(mut self) {
        self.0.kill().unwrap();
        let status = self.0.wait().unwrap();
        assert!(!status.success(), "a killed worker does not exit cleanly");
    }

    async fn finish(mut self) {
        let start = Instant::now();
        loop {
            if let Some(status) = self.0.try_wait().unwrap() {
                assert!(status.success(), "worker failed: {status}");
                return;
            }
            assert!(start.elapsed() < DEADLINE, "worker did not finish within {DEADLINE:?}");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

async fn call_log(store: &SqliteStore, log: &str) -> Vec<String> {
    let bytes = store.get(log).await.unwrap_or_default();
    String::from_utf8(bytes).unwrap().lines().map(str::to_string).collect()
}

/// Waits until branch B's call is logged and, for a durable store, branch
/// A's write of the step in flight is visible; for the whole-step control,
/// until branch A's call is logged. Fails after `DEADLINE`.
async fn wait_for_kill_point(store: &SqliteStore, id: RunId, kind: Kind) {
    let start = Instant::now();
    loop {
        let b_called = !call_log(store, LOG_B).await.is_empty();
        let a_done = match kind {
            Kind::Durable => {
                let step = store.latest(id).await.unwrap().map_or(0, |cp| cp.step + 1);
                let writes = store.branch_writes(id, step).await.unwrap();
                writes.iter().any(|w| w.node() == "left")
            }
            Kind::WholeSteps => !call_log(store, LOG_A).await.is_empty(),
        };
        if a_done && b_called {
            return;
        }
        assert!(start.elapsed() < DEADLINE, "branch A was never written while B ran");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
}

/// Runs one kill round on a fresh file: worker one stalls in tool B and
/// is killed at the kill point, worker two resumes with the gate open.
/// Returns the call logs of tool A and tool B.
async fn kill_round(dir: &Path, round: usize, kind: Kind) -> (Vec<String>, Vec<String>) {
    let path = dir.join(format!("parallel-{round}-{}.db", kind.name()));
    let gate = dir.join(format!("gate-{round}-{}", kind.name()));
    let id = new_run(&path, &gate).await;
    let watcher = SqliteStore::open(&path).await.unwrap();
    let first = Worker::spawn(&path, id, kind, &gate);
    wait_for_kill_point(&watcher, id, kind).await;
    first.kill();
    std::fs::write(&gate, b"open").unwrap();
    Worker::spawn(&path, id, kind, &gate).finish().await;
    assert_eq!(watcher.job(id).await.unwrap(), Some(JobState::Done));
    let troupe = troupe_on(&path).await;
    let cp = troupe.checkpoints().latest(id).await.unwrap().unwrap();
    let done = GraphRun::from_checkpoint(&cp).unwrap();
    assert_eq!(done.status(), &GraphStatus::Done);
    assert_eq!(done.state()["a"], json!("A"));
    assert_eq!(done.state()["b"], json!("B"));
    (call_log(&watcher, LOG_A).await, call_log(&watcher, LOG_B).await)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kill_after_branch_a_resume_reruns_only_b() {
    let dir = tempfile::tempdir().unwrap();
    for round in 0..runs() {
        let (a, b) = kill_round(dir.path(), round, Kind::Durable).await;
        assert_eq!(a.len(), 1, "round {round}: tool A ran again: {a:?}");
        assert_eq!(b.len(), 2, "round {round}: {b:?}");
        assert_eq!(b[0], b[1], "round {round}: tool B kept its idempotency key");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn kill_with_whole_step_store_reruns_both() {
    let dir = tempfile::tempdir().unwrap();
    for round in 0..runs() {
        let (a, b) = kill_round(dir.path(), round, Kind::WholeSteps).await;
        assert_eq!(a.len(), 2, "round {round}: a whole-step store reruns A: {a:?}");
        assert_eq!(a[0], a[1], "round {round}");
        assert_eq!(b.len(), 2, "round {round}: {b:?}");
        assert_eq!(b[0], b[1], "round {round}");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "worker process of the parallel kill tests, started by them"]
async fn parallel_kill_worker() {
    if std::env::var(ROLE).as_deref() != Ok("worker") {
        return;
    }
    let path = PathBuf::from(std::env::var(DB).unwrap());
    let gate = PathBuf::from(std::env::var(GATE).unwrap());
    let kind = match std::env::var(STORE).as_deref() {
        Ok("whole") => Kind::WholeSteps,
        _ => Kind::Durable,
    };
    let troupe = troupe_on(&path).await;
    let runner = runner(&troupe, kind, &gate);
    let id: RunId = serde_json::from_value(json!(std::env::var(RUN).unwrap())).unwrap();
    let store = SqliteStore::open(&path).await.unwrap();
    let start = Instant::now();
    loop {
        runner.tick(TickOptions::default()).await.unwrap();
        if store.job(id).await.unwrap() == Some(JobState::Done) {
            return;
        }
        assert!(start.elapsed() < DEADLINE, "worker ran past its deadline");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
