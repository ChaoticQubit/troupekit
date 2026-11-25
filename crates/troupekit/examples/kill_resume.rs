use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::path::Path;
use std::process::{Child, Command, ExitCode};
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit::agent::{Limits, PersonaLoader, SimpleIo};
use troupekit::core::clock::SystemClock;
use troupekit::core::ids::RunId;
use troupekit::core::store::{BlobStore, JobState};
use troupekit::graph::{GraphRun, SimpleGraphIo, Workflow};
use troupekit::host::TickOptions;
use troupekit::{SqliteStore, Troupe};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSpec};

const WORKFLOW: &str = r#"
name = "three"
start = "first"
[state]
goal = "string"
a = "string"
b = "string"
c = "string"
[[nodes]]
id = "first"
kind = "tool"
tool = "first"
reads = ["goal"]
writes = "a"
[[nodes]]
id = "second"
kind = "tool"
tool = "second"
reads = ["a"]
writes = "b"
[[nodes]]
id = "third"
kind = "tool"
tool = "third"
reads = ["b"]
writes = "c"
[[nodes]]
id = "done"
kind = "end"
[[edges]]
from = "first"
to = "second"
[[edges]]
from = "second"
to = "third"
[[edges]]
from = "third"
to = "done"
"#;

const MARKER: &str = "example/marker";
const DEADLINE: Duration = Duration::from_secs(10);

/// A pure tool that joins its one string argument with its name. With
/// `gate` set it writes a marker blob and then never answers, so the
/// parent knows the worker is inside the step and kills it there.
struct Join {
    spec: ToolSpec,
    gate: Option<Arc<dyn BlobStore>>,
}

#[async_trait]
impl Tool for Join {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, _ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let input = args
            .as_object()
            .and_then(|m| m.values().next())
            .and_then(Value::as_str)
            .ok_or_else(|| ToolError::InvalidArgs("one string argument".into()))?
            .to_string();
        if let Some(blobs) = &self.gate {
            blobs
                .put(MARKER, vec![1])
                .await
                .map_err(|e| ToolError::Failed(e.to_string()))?;
            std::future::pending::<()>().await;
        }
        let out = format!("{input}>{}", self.spec.name);
        Ok(ToolOutput::text(
            serde_json::to_string(&out).expect("a string serializes"),
        ))
    }
}

fn io(troupe: &Troupe, gate: bool) -> Arc<SimpleGraphIo> {
    let mut tools = ToolRegistry::new();
    for name in ["first", "second", "third"] {
        let gate = (gate && name == "second").then(|| troupe.blobs());
        let spec = ToolSpec {
            name: name.into(),
            description: format!("joins its input with {name}"),
            schema: json!({"type": "object"}),
            hosted: None,
        };
        tools
            .register(Arc::new(Join { spec, gate }))
            .expect("valid tool");
    }
    let mut agent = SimpleIo::new(tools);
    agent.ledger = troupe.ledger();
    agent.clock = Arc::new(SystemClock);
    Arc::new(SimpleGraphIo::new(
        Arc::new(agent),
        PersonaLoader::new(Vec::new()),
    ))
}

async fn start(troupe: &Troupe) -> RunId {
    let wf = Workflow::parse("three.toml", WORKFLOW).expect("valid workflow");
    let input: Map<String, Value> = [("goal".to_string(), json!("survive"))]
        .into_iter()
        .collect();
    let io = io(troupe, false);
    let run = GraphRun::new(wf, input, Limits::default(), io.as_ref())
        .await
        .expect("run");
    troupe.runner(io).enqueue(&run).await.expect("enqueue");
    run.run_id()
}

async fn work(path: &Path, id: RunId, gate: bool) {
    let troupe = Troupe::builder().sqlite(path).build().await.expect("open");
    let runner = troupe.runner(io(&troupe, gate)).with_lease_ms(200);
    let store = SqliteStore::open(path).await.expect("open");
    let start = Instant::now();
    while store.job(id).await.expect("job") != Some(JobState::Done) {
        assert!(start.elapsed() < DEADLINE, "worker ran past its deadline");
        runner.tick(TickOptions::default()).await.expect("tick");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

fn spawn(path: &Path, id: RunId, gate: bool) -> Child {
    let mut cmd = Command::new(std::env::current_exe().expect("own path"));
    cmd.env("KILL_RESUME_DB", path)
        .env("KILL_RESUME_RUN", id.to_string());
    if gate {
        cmd.env("KILL_RESUME_GATE", "1");
    }
    cmd.spawn().expect("spawn worker")
}

async fn final_run(path: &Path, id: RunId) -> GraphRun {
    let troupe = Troupe::builder().sqlite(path).build().await.expect("open");
    let cp = troupe
        .checkpoints()
        .latest(id)
        .await
        .expect("read")
        .expect("checkpoint");
    GraphRun::from_checkpoint(&cp).expect("restore")
}

async fn parent() -> ExitCode {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("kill.db");
    let id = start(&Troupe::builder().sqlite(&path).build().await.expect("open")).await;
    println!("1. run {id} created and enqueued in {}", path.display());
    let mut first = spawn(&path, id, true);
    let watcher = SqliteStore::open(&path).await.expect("open");
    let begin = Instant::now();
    while watcher.get(MARKER).await.is_err() {
        if begin.elapsed() > DEADLINE {
            let _ = first.kill();
            println!("worker never reached the second tool");
            return ExitCode::FAILURE;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    first.kill().expect("kill");
    let _ = first.wait();
    println!("2. worker killed with SIGKILL inside the second tool");
    let mut second = spawn(&path, id, false);
    let status = second.wait().expect("wait");
    println!("3. second worker finished: {status}");
    let resumed = final_run(&path, id).await;
    let reference_path = dir.path().join("reference.db");
    let reference_id = start(
        &Troupe::builder()
            .sqlite(&reference_path)
            .build()
            .await
            .expect("open"),
    )
    .await;
    work(&reference_path, reference_id, false).await;
    let reference = final_run(&reference_path, reference_id).await;
    let same = resumed.state() == reference.state()
        && resumed.steps() == reference.steps()
        && resumed.usage() == reference.usage();
    println!(
        "4. resumed state {}",
        Value::Object(resumed.state().clone())
    );
    println!("5. same as an uninterrupted run: {same}");
    if status.success() && same {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Starts a three-tool run in a SQLite file, kills the worker process with
/// SIGKILL in the middle of the second tool, resumes it in a new process
/// and compares the result with an uninterrupted run. Exits 0 only when
/// they match.
#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> ExitCode {
    match (
        std::env::var("KILL_RESUME_DB"),
        std::env::var("KILL_RESUME_RUN"),
    ) {
        (Ok(path), Ok(run)) => {
            let id: RunId = serde_json::from_value(json!(run)).expect("run id");
            work(
                Path::new(&path),
                id,
                std::env::var("KILL_RESUME_GATE").is_ok(),
            )
            .await;
            ExitCode::SUCCESS
        }
        _ => parent().await,
    }
}
