use async_trait::async_trait;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit_agent::{Limits, MemorySource, PersonaLoader, SimpleIo, VecSink};
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::{BranchWrite, Checkpoint, CheckpointStore};
use troupekit_graph::testing::{text_reply, FnModel, GatedTool};
use troupekit_graph::{GraphError, GraphOutcome, GraphRun, SimpleGraphIo, Stop, Workflow};
use troupekit_providers::Price;
use troupekit_tools::{ToolOutput, ToolRegistry};

const PRICE: Price = Price {
    input_usd_micros_per_mtok: 1_000_000,
    output_usd_micros_per_mtok: 2_000_000,
    verified: true,
    long_context: None,
    search_usd_micros: 0,
};

#[derive(Default)]
pub(crate) struct Store {
    cps: Mutex<Vec<Checkpoint>>,
    pub(crate) writes: Mutex<BTreeMap<(RunId, u64, u32), BranchWrite>>,
    whole_steps_only: bool,
    fail_writes: AtomicBool,
}

impl Store {
    fn whole_steps_only() -> Store {
        Store {
            whole_steps_only: true,
            ..Store::default()
        }
    }

    pub(crate) fn has(&self, run: RunId, step: u64, branch: u32) -> bool {
        self.writes
            .lock()
            .unwrap()
            .contains_key(&(run, step, branch))
    }

    fn put(&self, run: RunId, step: u64, write: BranchWrite) {
        self.writes
            .lock()
            .unwrap()
            .insert((run, step, write.branch()), write);
    }

    fn take(&self, run: RunId, step: u64, branch: u32) -> BranchWrite {
        self.writes
            .lock()
            .unwrap()
            .remove(&(run, step, branch))
            .unwrap()
    }
}

#[async_trait]
impl CheckpointStore for Store {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        let mut cps = self.cps.lock().unwrap();
        let last = cps
            .iter()
            .filter(|c| c.run_id == cp.run_id)
            .map(|c| c.step)
            .max();
        let expected = last.map_or(0, |s| s + 1);
        if cp.step != expected {
            return Err(CoreError::Conflict {
                expected,
                actual: cp.step,
            });
        }
        cps.push(cp);
        Ok(())
    }

    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        let cps = self.cps.lock().unwrap();
        Ok(cps
            .iter()
            .filter(|c| c.run_id == run_id)
            .max_by_key(|c| c.step)
            .cloned())
    }

    async fn save_branch(
        &self,
        run_id: RunId,
        step: u64,
        write: BranchWrite,
    ) -> Result<bool, CoreError> {
        if self.whole_steps_only {
            return Ok(false);
        }
        if self.fail_writes.load(Ordering::SeqCst) {
            return Err(CoreError::Store("disk full".into()));
        }
        let mut writes = self.writes.lock().unwrap();
        match writes.get(&(run_id, step, write.branch())) {
            Some(kept) if kept == &write => Ok(true),
            Some(_) => Err(CoreError::Conflict {
                expected: step,
                actual: step,
            }),
            None => {
                writes.insert((run_id, step, write.branch()), write);
                Ok(true)
            }
        }
    }

    async fn branch_writes(&self, run_id: RunId, step: u64) -> Result<Vec<BranchWrite>, CoreError> {
        let writes = self.writes.lock().unwrap();
        Ok(writes
            .iter()
            .filter(|((r, s, _), _)| *r == run_id && *s == step)
            .map(|(_, w)| w.clone())
            .collect())
    }
}

const FAN: &str = "name = 'fan'\nstart = 's'\n[state]\nseed = 'string'\nx = 'string'\ny = 'string'\n[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 's'\nwrites = 'seed'\n[[nodes]]\nid = 'a'\nkind = 'tool'\ntool = 'a'\nwrites = 'x'\n[[nodes]]\nid = 'b'\nkind = 'tool'\ntool = 'b'\nwrites = 'y'\n[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 's'\nto = 'a'\n[[edges]]\nfrom = 's'\nto = 'b'\n[[edges]]\nfrom = 'a'\nto = 'j'\n[[edges]]\nfrom = 'b'\nto = 'j'\n[[edges]]\nfrom = 'j'\nto = 'e'\n";

const AGENT_FAN: &str = "name = 'fan'\nstart = 's'\n[state]\nseed = 'string'\nx = 'string'\ny = 'string'\n[[nodes]]\nid = 's'\nkind = 'tool'\ntool = 's'\nwrites = 'seed'\n[[nodes]]\nid = 'a'\nkind = 'agent'\npersona = 'writer'\nreads = ['seed']\nwrites = 'x'\n[[nodes]]\nid = 'b'\nkind = 'tool'\ntool = 'b'\nwrites = 'y'\n[[nodes]]\nid = 'j'\nkind = 'join'\n[[nodes]]\nid = 'e'\nkind = 'end'\n[[edges]]\nfrom = 's'\nto = 'a'\n[[edges]]\nfrom = 's'\nto = 'b'\n[[edges]]\nfrom = 'a'\nto = 'j'\n[[edges]]\nfrom = 'b'\nto = 'j'\n[[edges]]\nfrom = 'j'\nto = 'e'\n";

/// Awaits `f` for at most 10 seconds. On timeout panics with `what` and
/// the state `report` describes, so a lost wake-up fails the test with a
/// reason instead of hanging it. Returns the output of `f`.
pub(crate) async fn bounded<F: Future>(
    what: &str,
    report: impl FnOnce() -> String,
    f: F,
) -> F::Output {
    match tokio::time::timeout(Duration::from_secs(10), f).await {
        Ok(out) => out,
        Err(_) => panic!("{what} did not finish in 10s: {}", report()),
    }
}

fn io(gates: &GatedTool, model: Arc<FnModel>) -> SimpleGraphIo {
    io_events(gates, model).0
}

fn io_events(gates: &GatedTool, model: Arc<FnModel>) -> (SimpleGraphIo, Arc<VecSink>) {
    let mut tools = ToolRegistry::new();
    tools
        .register(gates.tool("s", |_| ToolOutput::text("seeded")))
        .unwrap();
    let a_calls = Arc::new(AtomicUsize::new(0));
    tools
        .register(gates.tool("a", move |_| {
            let n = a_calls.fetch_add(1, Ordering::SeqCst) + 1;
            ToolOutput::text(format!("a{n}"))
        }))
        .unwrap();
    tools
        .register(gates.tool("b", |_| ToolOutput::text("b")))
        .unwrap();
    let sink = Arc::new(VecSink::default());
    let mut agent = SimpleIo::new(tools).with_model("fake/m", model, PRICE);
    agent.events = sink.clone();
    let personas = PersonaLoader::new(vec![Arc::new(MemorySource::new(
        "test",
        [(
            "writer.toml".to_string(),
            "name = 'writer'\ndisplay_name = 'W'\nrole = 'Writes.'\ngoal = 'Done.'\ninstructions = 'Do it.'\nmodel = 'fake/m'\n".to_string(),
        )],
    ))]);
    (SimpleGraphIo::new(Arc::new(agent), personas), sink)
}

fn model() -> Arc<FnModel> {
    Arc::new(FnModel::new("fake/m", |_| text_reply("written", 10)))
}

async fn started(text: &str, io: &SimpleGraphIo, store: &Store, gates: &GatedTool) -> GraphRun {
    let workflow = Workflow::parse("test/fan.toml", text).unwrap();
    let mut run = GraphRun::new(workflow, Map::new(), Limits::default(), io)
        .await
        .unwrap();
    store.save(run.checkpoint()).await.unwrap();
    gates.open("s");
    assert_eq!(
        run.step_durable(io, store).await.unwrap(),
        GraphOutcome::Continue
    );
    store.save(run.checkpoint()).await.unwrap();
    assert_eq!(run.steps(), 1);
    run
}

async fn crash_after_branch_a(
    text: &str,
    io: &SimpleGraphIo,
    store: &Store,
    gates: &GatedTool,
) -> RunId {
    let mut run = started(text, io, store, gates).await;
    let id = run.run_id();
    gates.open("a");
    let a_kept = async {
        while !store.has(id, 2, 0) {
            tokio::task::yield_now().await;
        }
    };
    let report = || {
        let marks: Vec<_> = store.writes.lock().unwrap().keys().copied().collect();
        format!("a finished: {:?}, marks: {marks:?}", gates.finished())
    };
    bounded("the first durable mark", report, async {
        tokio::select! {
            biased;
            out = run.step_durable(io, store) => panic!("the step must not finish while b waits: {out:?}"),
            () = a_kept => {}
        }
    })
    .await;
    assert!(!store.has(id, 2, 1), "b never finished");
    id
}

async fn drive(run: &mut GraphRun, io: &SimpleGraphIo, store: &Store) -> GraphOutcome {
    for _ in 0..30 {
        let out = run.step_durable(io, store).await.unwrap();
        store.save(run.checkpoint()).await.unwrap();
        if out != GraphOutcome::Continue {
            return out;
        }
    }
    panic!("run did not settle in 30 steps");
}

fn done(out: GraphOutcome) -> Map<String, Value> {
    let GraphOutcome::Done(state) = out else {
        panic!("run finishes: {out:?}");
    };
    state
}

#[tokio::test]
async fn finished_branch_is_written_before_siblings_finish() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let mut run = started(FAN, &io, &store, &gates).await;
    let id = run.run_id();
    let opener = async {
        gates.open("a");
        while !store.has(id, 2, 0) {
            tokio::task::yield_now().await;
        }
        assert_eq!(
            gates.in_flight("b"),
            1,
            "b still runs when a's write is kept"
        );
        assert!(!store.has(id, 2, 1));
        gates.open("b");
    };
    let report = || format!("finished: {:?}", gates.finished());
    let (out, ()) = bounded("the early-save run", report, async {
        tokio::join!(drive(&mut run, &io, &store), opener)
    })
    .await;
    let state = done(out);
    assert_eq!((&state["x"], &state["y"]), (&json!("a1"), &json!("b")));
    assert!(store.has(id, 2, 1));
}

#[tokio::test]
async fn restore_applies_writes_and_reruns_only_unfinished_branches() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    assert_eq!(run.steps(), 1);
    gates.open("b");
    let state = done(drive(&mut run, &io, &store).await);
    assert_eq!(gates.calls("a").len(), 1, "a ran once");
    assert_eq!(gates.calls("b").len(), 2, "b ran again");
    assert_eq!((&state["x"], &state["y"]), (&json!("a1"), &json!("b")));
}

#[tokio::test]
async fn rerun_keeps_idempotency_keys_of_unfinished_branches() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    done(drive(&mut run, &io, &store).await);
    let keys = gates.calls("b");
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0], Some(format!("{id}:2:1")));
    assert_eq!(keys[0], keys[1]);
}

#[tokio::test]
async fn agent_branch_model_call_is_not_billed_twice() {
    let straight_model = model();
    let (store, gates) = (Store::default(), GatedTool::new());
    let io_straight = io(&gates, straight_model.clone());
    for name in ["a", "b"] {
        gates.open(name);
    }
    let mut run = started(AGENT_FAN, &io_straight, &store, &gates).await;
    let straight = done(drive(&mut run, &io_straight, &store).await);

    let crashed_model = model();
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, crashed_model.clone());
    let id = crash_after_branch_a(AGENT_FAN, &io, &store, &gates).await;
    let calls_at_crash = crashed_model.calls();
    assert!(
        calls_at_crash >= 1,
        "the agent called its model before the crash"
    );
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    let state = done(drive(&mut run, &io, &store).await);
    assert_eq!(
        crashed_model.calls(),
        straight_model.calls(),
        "no model call repeats"
    );
    assert_eq!(state["x"], straight["x"]);
    assert_eq!(run.usage().input_tokens, 10 * straight_model.calls() as u64);
}

#[tokio::test]
async fn write_for_another_node_is_refused() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    store.put(
        id,
        2,
        BranchWrite::new(1, "zz", json!({"kind": "tool", "output": "forged"})),
    );
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    let err = run.step_durable(&io, &store).await.unwrap_err();
    assert!(
        matches!(&err, GraphError::Checkpoint(m) if m.contains("node `zz`")),
        "{err}"
    );
    assert_eq!(
        gates.calls("a").len(),
        1,
        "no tool runs for a refused write"
    );
    assert_eq!(
        gates.calls("b").len(),
        1,
        "no tool runs for a refused write"
    );
}

#[tokio::test]
async fn writes_of_an_older_step_are_ignored() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let kept = store.take(id, 2, 0);
    store.put(id, 1, kept);
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    let state = done(drive(&mut run, &io, &store).await);
    assert_eq!(
        gates.calls("a").len(),
        2,
        "a write of step 1 never applies to step 2"
    );
    assert_eq!(state["x"], json!("a2"));
}

#[tokio::test]
async fn store_without_branch_writes_falls_back_to_whole_steps() {
    let (store, gates) = (Store::whole_steps_only(), GatedTool::new());
    for name in ["s", "a", "b"] {
        gates.open(name);
    }
    let (plain_io, durable_io) = (io(&gates, model()), io(&gates, model()));
    let workflow = Workflow::parse("test/fan.toml", FAN).unwrap();
    let mut plain = GraphRun::new(workflow, Map::new(), Limits::default(), &plain_io)
        .await
        .unwrap();
    let mut durable = plain.clone();
    for _ in 0..30 {
        let a = plain.step(&plain_io).await;
        let b = durable.step_durable(&durable_io, &store).await.unwrap();
        assert_eq!(a, b);
        assert_eq!(plain.checkpoint(), durable.checkpoint());
        if a != GraphOutcome::Continue {
            break;
        }
    }
    assert!(store.writes.lock().unwrap().is_empty());
}

#[tokio::test]
async fn save_branch_error_finishes_the_step_and_returns_store_error() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let mut run = started(FAN, &io, &store, &gates).await;
    store.fail_writes.store(true, Ordering::SeqCst);
    gates.open("a");
    gates.open("b");
    let err = run.step_durable(&io, &store).await.unwrap_err();
    assert!(
        matches!(err, GraphError::Store(CoreError::Store(_))),
        "{err}"
    );
    let mut finished = gates.finished();
    finished.sort();
    assert_eq!(
        finished,
        vec!["a", "b", "s"],
        "every branch of the step finished"
    );
}

#[tokio::test]
async fn second_write_for_the_same_branch_conflicts() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let latest = store.latest(id).await.unwrap().unwrap();
    let mut wrong = GraphRun::from_checkpoint(&latest).unwrap();
    gates.open("b");
    let err = wrong.step_durable(&io, &store).await.unwrap_err();
    assert!(
        matches!(err, GraphError::Store(CoreError::Conflict { .. })),
        "{err}"
    );
    assert_eq!(
        gates.calls("a").len(),
        2,
        "reloading from the checkpoint re-ran a"
    );
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    let state = done(drive(&mut run, &io, &store).await);
    assert_eq!(gates.calls("a").len(), 2, "restore applies a's kept write");
    assert_eq!(state["x"], json!("a1"), "the first durable result wins");
}

#[tokio::test]
async fn kept_writes_of_a_moved_step_are_dropped() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    run.stop(Stop::Cancelled).unwrap();
    store.save(run.checkpoint()).await.unwrap();
    run.retry(None).unwrap();
    store.save(run.checkpoint()).await.unwrap();
    gates.open("b");
    let state = done(drive(&mut run, &io, &store).await);
    assert_eq!(
        gates.calls("a").len(),
        2,
        "writes of step 2 never apply to step 4"
    );
    assert_eq!(state["x"], json!("a2"));
}

#[tokio::test]
async fn plain_step_drops_kept_writes() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    loop {
        let out = run.step(&io).await;
        if out != GraphOutcome::Continue {
            done(out);
            break;
        }
    }
    assert_eq!(
        gates.calls("a").len(),
        2,
        "plain step keeps the 0.1 behaviour"
    );
}

#[tokio::test]
async fn restored_step_never_reuses_an_emitted_seq() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let (io, sink) = io_events(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let seen = sink.events();
    let seen_max = seen
        .iter()
        .map(|e| e.seq)
        .max()
        .expect("the crashed step emitted events");
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    done(drive(&mut run, &io, &store).await);
    let after: Vec<u64> = sink.events()[seen.len()..].iter().map(|e| e.seq).collect();
    assert!(!after.is_empty());
    assert!(
        after.iter().all(|seq| *seq > seen_max),
        "events after restore {after:?} reuse a seq up to {seen_max}"
    );
}

#[tokio::test]
async fn second_restore_never_reuses_a_seq_of_the_first_restored_attempt() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let (io, sink) = io_events(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    let b_restarted = async {
        while gates.calls("b").len() < 2 {
            tokio::task::yield_now().await;
        }
    };
    let report = || format!("b calls: {}", gates.calls("b").len());
    bounded("the restored attempt", report, async {
        tokio::select! {
            biased;
            out = run.step_durable(&io, &store) => panic!("b is still closed: {out:?}"),
            () = b_restarted => {}
        }
    })
    .await;
    drop(run);
    let seen = sink.events();
    let seen_max = seen.iter().map(|e| e.seq).max().unwrap();
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    gates.open("b");
    done(drive(&mut run, &io, &store).await);
    let after: Vec<u64> = sink.events()[seen.len()..].iter().map(|e| e.seq).collect();
    assert!(!after.is_empty());
    assert!(
        after.iter().all(|seq| *seq > seen_max),
        "events after the second restore {after:?} reuse a seq up to {seen_max}"
    );
    assert_eq!(gates.calls("a").len(), 1, "a's kept write still applies");
}

#[tokio::test]
async fn seq_reservation_that_cannot_be_kept_runs_nothing() {
    let (store, gates) = (Store::default(), GatedTool::new());
    let io = io(&gates, model());
    let id = crash_after_branch_a(FAN, &io, &store, &gates).await;
    let mut run = GraphRun::restore(&store, id).await.unwrap();
    store.fail_writes.store(true, Ordering::SeqCst);
    gates.open("b");
    let err = run.step_durable(&io, &store).await.unwrap_err();
    assert!(
        matches!(err, GraphError::Store(CoreError::Store(_))),
        "{err}"
    );
    assert_eq!(
        gates.calls("b").len(),
        1,
        "no branch runs before the seq range is kept"
    );
}

#[tokio::test]
async fn whole_steps_store_keeps_the_0_1_seq_after_restore() {
    let (store, gates) = (Store::whole_steps_only(), GatedTool::new());
    for name in ["s", "a", "b"] {
        gates.open(name);
    }
    let (io, sink) = io_events(&gates, model());
    let mut run = started(FAN, &io, &store, &gates).await;
    let id = run.run_id();
    drop(run);
    let before = sink.events().len();
    let next = sink.events().iter().map(|e| e.seq).max().unwrap() + 1;
    run = GraphRun::restore(&store, id).await.unwrap();
    let out = run.step_durable(&io, &store).await.unwrap();
    assert!(matches!(out, GraphOutcome::Done(_)), "{out:?}");
    let after: Vec<u64> = sink.events()[before..].iter().map(|e| e.seq).collect();
    assert_eq!(
        after.first(),
        Some(&next),
        "an unkept reservation is not used: {after:?}"
    );
    assert_eq!(after, (next..next + after.len() as u64).collect::<Vec<_>>());
}
