#![allow(dead_code)]

use serde_json::{Map, Value};
use std::error::Error;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use troupekit::agent::{DirSource, Limits, PersonaLoader, SimpleIo};
use troupekit::core::clock::SystemClock;
use troupekit::core::ids::RunId;
use troupekit::graph::{patterns, GraphRun, GraphStatus, SimpleGraphIo};
use troupekit::host::TickOptions;
use troupekit::Troupe;
use troupekit_providers::{Model, Price};
use troupekit_tools::ToolRegistry;

/// Result type of the team examples; every error is boxed with its text.
pub type Outcome<T> = Result<T, Box<dyn Error + Send + Sync>>;

/// The support desk: a route team whose lead picks billing or tech, with a
/// human approval before billing answers.
pub const TRIAGE: &str = include_str!("triage.toml");
/// The report crew: research, an async fact check, then the writer.
pub const REPORT: &str = include_str!("report.toml");
/// The model selector every persona under `personas/` names.
pub const PERSONA_MODEL: &str = "anthropic/claude-haiku-5-5";

const TICKS_MAX: usize = 1_000;
const IDLE_WAIT: Duration = Duration::from_millis(10);

/// Builds the graph io of the examples: the personas under `personas/`,
/// every one served by `model` at `price` under `PERSONA_MODEL`, with the
/// troupe's ledger and the system clock.
/// Input: troupe, model and price. Output: the io the runner steps with.
pub fn io(troupe: &Troupe, model: Arc<dyn Model>, price: Price) -> Arc<SimpleGraphIo> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/teams/personas");
    assert!(dir.is_dir(), "the personas folder ships with the crate");
    let mut agent = SimpleIo::new(ToolRegistry::new()).with_model(PERSONA_MODEL, model, price);
    agent.ledger = troupe.ledger();
    agent.clock = Arc::new(SystemClock);
    assert!(agent.models.contains_key(PERSONA_MODEL));
    let loader = PersonaLoader::new(vec![Arc::new(DirSource::new(dir))]);
    Arc::new(SimpleGraphIo::new(Arc::new(agent), loader))
}

/// Compiles pattern `text` from `file`, starts a run on the request
/// `input`, saves it and enqueues it on the troupe's queue.
/// Input: troupe, io, file name, pattern TOML and request.
/// Output: the run id, or the first pattern, flow or store error.
pub async fn start(
    troupe: &Troupe,
    io: Arc<SimpleGraphIo>,
    file: &str,
    text: &str,
    input: &str,
) -> Outcome<RunId> {
    let flow = patterns::load(file, text)?;
    let mut state = Map::new();
    state.insert("input".into(), Value::String(input.to_string()));
    let run = GraphRun::start(flow, state, Limits::default(), io.as_ref()).await?;
    let id = run.run_id();
    troupe.runner(io).enqueue(&run).await?;
    assert_eq!(run.run_id(), id);
    Ok(id)
}

/// Ticks the troupe's runner until run `id` is done, failed or waits for a
/// person, at most 1 000 ticks. Input: troupe, io and run id.
/// Output: the run as its latest checkpoint holds it.
pub async fn work(troupe: &Troupe, io: Arc<SimpleGraphIo>, id: RunId) -> Outcome<GraphRun> {
    let runner = troupe.runner(io);
    for _ in 0..TICKS_MAX {
        let report = runner.tick(TickOptions::default()).await?;
        if let Some(error) = report.errors.first() {
            return Err(format!("run {id}: {error}").into());
        }
        let run = latest(troupe, id).await?;
        if settled(run.status()) {
            return Ok(run);
        }
        if report.processed == 0 {
            tokio::time::sleep(IDLE_WAIT).await;
        }
    }
    Err(format!("run {id} did not settle in {TICKS_MAX} ticks").into())
}

/// Answers the human approval node `node` of run `id` with `yes` and
/// enqueues the run again. Input: troupe, io, run id, node id and answer.
/// Output: nothing, or the runner's error for a run that does not wait.
pub async fn approve(
    troupe: &Troupe,
    io: Arc<SimpleGraphIo>,
    id: RunId,
    node: &str,
    yes: bool,
) -> Outcome<()> {
    assert!(!node.is_empty(), "an approval node has an id");
    troupe.runner(io).resume(id, node, Value::Bool(yes)).await?;
    Ok(())
}

/// Returns the answer of a finished run: its `output` text, or an error
/// naming the status of a run that did not finish.
/// Input: the run. Output: the answer.
pub fn answer(run: &GraphRun) -> Outcome<String> {
    if !matches!(run.status(), GraphStatus::Done) {
        return Err(format!("run {} ended as {:?}", run.run_id(), run.status()).into());
    }
    match run.state().get("output") {
        Some(Value::String(text)) => Ok(text.clone()),
        other => Err(format!("run {} has no text output: {other:?}", run.run_id()).into()),
    }
}

async fn latest(troupe: &Troupe, id: RunId) -> Outcome<GraphRun> {
    let cp = troupe
        .checkpoints()
        .latest(id)
        .await?
        .ok_or_else(|| format!("run {id} has no checkpoint"))?;
    let run = GraphRun::from_checkpoint(&cp)?;
    assert_eq!(run.run_id(), id);
    Ok(run)
}

fn settled(status: &GraphStatus) -> bool {
    matches!(
        status,
        GraphStatus::Done
            | GraphStatus::Failed { .. }
            | GraphStatus::WaitingHuman { .. }
            | GraphStatus::WaitingBudget { .. }
    )
}
