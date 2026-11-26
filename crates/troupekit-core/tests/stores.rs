use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::{InMemBlobs, InMemCheckpoints, InMemMemory, InMemVectors};
use troupekit_core::message::{Message, Role};
use troupekit_core::store::*;

#[tokio::test]
async fn checkpoints_must_advance_by_one() {
    let s = InMemCheckpoints::default();
    let run = RunId::new();
    s.save(Checkpoint { run_id: run, step: 0, state: json!({"a":1}) }).await.unwrap();
    s.save(Checkpoint { run_id: run, step: 1, state: json!({"a":2}) }).await.unwrap();
    let err = s.save(Checkpoint { run_id: run, step: 1, state: json!({}) }).await.unwrap_err();
    assert_eq!(err, CoreError::Conflict { expected: 2, actual: 1 });
    assert_eq!(s.latest(run).await.unwrap().unwrap().state, json!({"a":2}));
    assert!(s.latest(RunId::new()).await.unwrap().is_none());
}

#[tokio::test]
async fn second_concurrent_writer_conflicts() {
    let m = Arc::new(InMemMemory::default());
    assert_eq!(m.append("g1", 0, vec![Message::text(Role::User, "a")]).await.unwrap(), 1);
    let (a, b) = tokio::join!(
        m.append("g1", 1, vec![Message::text(Role::User, "b")]),
        m.append("g1", 1, vec![Message::text(Role::User, "c")]),
    );
    assert!(a.is_ok() != b.is_ok());
    assert_eq!(m.read("g1").await.unwrap().len(), 2);
}

#[tokio::test]
async fn vector_search_ranks_and_filters() {
    let v = InMemVectors::new(2);
    let meta = |g: &str| BTreeMap::from([("goal".to_string(), g.to_string())]);
    v.upsert(vec![
        VectorRecord { id: "a".into(), vector: vec![1.0, 0.0], text: "protein".into(), meta: meta("g1") },
        VectorRecord { id: "b".into(), vector: vec![0.7, 0.7], text: "sleep".into(), meta: meta("g1") },
        VectorRecord { id: "c".into(), vector: vec![1.0, 0.1], text: "piano".into(), meta: meta("g2") },
    ]).await.unwrap();
    let hits = v.search(&[1.0, 0.0], 2, &Filter(meta("g1"))).await.unwrap();
    assert_eq!(hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
    assert!(hits[0].score > hits[1].score);
    let err = v.upsert(vec![VectorRecord { id: "d".into(), vector: vec![1.0], text: String::new(), meta: BTreeMap::new() }]).await.unwrap_err();
    assert_eq!(err, CoreError::DimensionMismatch { expected: 2, actual: 1 });
}

#[tokio::test]
async fn blobs_round_trip() {
    let b = InMemBlobs::default();
    b.put("goals/1/plan.pdf", vec![1, 2, 3]).await.unwrap();
    assert_eq!(b.get("goals/1/plan.pdf").await.unwrap(), vec![1, 2, 3]);
    assert!(matches!(b.get("nope").await, Err(CoreError::NotFound(_))));
}

#[test]
fn cosine_basics() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
}

#[test]
fn queue_types_live_in_core() {
    let run = RunId::new();
    let lease = troupekit_core::store::Lease { run_id: run, token: 1, attempts: 0 };
    let pending = troupekit_core::store::JobState::Pending { due_ms: 5 };
    assert_eq!(lease.run_id, run);
    assert_eq!(pending, JobState::Pending { due_ms: 5 });
}

struct OnlySaveAndLatest(InMemCheckpoints);

#[async_trait::async_trait]
impl CheckpointStore for OnlySaveAndLatest {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        self.0.save(cp).await
    }
    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        self.0.latest(run_id).await
    }
}

#[tokio::test]
async fn impl_with_only_save_and_latest_still_compiles() {
    let s: Arc<dyn CheckpointStore> = Arc::new(OnlySaveAndLatest(InMemCheckpoints::default()));
    let run = RunId::new();
    s.save(Checkpoint { run_id: run, step: 0, state: json!(1) }).await.unwrap();
    assert_eq!(s.history(run).await.unwrap().len(), 1);
    assert_eq!(s.prune(run, 1).await.unwrap(), 0);
}

#[tokio::test]
async fn default_history_is_the_latest_checkpoint() {
    let s = InMemCheckpoints::default();
    let run = RunId::new();
    s.save(Checkpoint { run_id: run, step: 0, state: json!({"a":1}) }).await.unwrap();
    s.save(Checkpoint { run_id: run, step: 1, state: json!({"a":2}) }).await.unwrap();
    let history = s.history(run).await.unwrap();
    assert_eq!(history, vec![s.latest(run).await.unwrap().unwrap()]);
}

#[tokio::test]
async fn default_history_of_unknown_run_is_empty() {
    let s = InMemCheckpoints::default();
    assert!(s.history(RunId::new()).await.unwrap().is_empty());
}

#[tokio::test]
async fn default_prune_deletes_nothing() {
    let s = InMemCheckpoints::default();
    let run = RunId::new();
    s.save(Checkpoint { run_id: run, step: 0, state: json!(0) }).await.unwrap();
    assert_eq!(s.prune(run, 1).await.unwrap(), 0);
    assert_eq!(s.latest(run).await.unwrap().unwrap().step, 0);
}

#[tokio::test]
async fn prune_rejects_keep_last_zero() {
    let s = InMemCheckpoints::default();
    let err = s.prune(RunId::new(), 0).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}

struct OneHit;

#[async_trait::async_trait]
impl KeywordStore for OneHit {
    async fn search_text(&self, query: &str, k: usize, filter: &Filter) -> Result<Vec<Scored>, CoreError> {
        let hit = Scored { id: query.into(), score: 1.0, text: query.into(), meta: filter.0.clone() };
        Ok(std::iter::once(hit).take(k).collect())
    }
}

#[tokio::test]
async fn keyword_store_is_object_safe() {
    let s: Arc<dyn KeywordStore> = Arc::new(OneHit);
    let f = Filter(BTreeMap::from([("scope".to_string(), "docs".to_string())]));
    let hits = s.search_text("refund", 1, &f).await.unwrap();
    assert_eq!(hits.len(), 1);
    assert_eq!(hits[0].meta, f.0);
    assert!(s.search_text("refund", 0, &f).await.unwrap().is_empty());
}

async fn save_steps(s: &InMemCheckpoints, run: RunId, steps: u64) {
    for step in 0..steps {
        s.save(Checkpoint { run_id: run, step, state: json!({"step": step}) }).await.unwrap();
    }
}

#[tokio::test]
async fn default_inmem_keeps_latest_only() {
    let s = InMemCheckpoints::default();
    let run = RunId::new();
    save_steps(&s, run, 3).await;
    let history = s.history(run).await.unwrap();
    assert_eq!(history.len(), 1);
    assert_eq!(history[0].step, 2);
    let write = BranchWrite::new(0, "a", json!({"at": "leave"}));
    assert!(!s.save_branch(run, 3, write).await.unwrap());
    assert!(s.branch_writes(run, 3).await.unwrap().is_empty());
    let child = RunId::new();
    assert!(!s.link_fork(ForkLink::new(child, run, 1)).await.unwrap());
    assert!(s.forks(run).await.unwrap().is_empty());
    assert_eq!(s.prune(run, 1).await.unwrap(), 0);
}

#[tokio::test]
async fn with_history_lists_every_step() {
    let s = InMemCheckpoints::with_history();
    let run = RunId::new();
    save_steps(&s, run, 3).await;
    let history = s.history(run).await.unwrap();
    let steps: Vec<u64> = history.iter().map(|cp| cp.step).collect();
    assert_eq!(steps, vec![0, 1, 2]);
    assert_eq!(history[1].state, json!({"step": 1}));
    assert_eq!(s.latest(run).await.unwrap().unwrap().step, 2);
    let err = s.save(Checkpoint { run_id: run, step: 5, state: json!(0) }).await.unwrap_err();
    assert_eq!(err, CoreError::Conflict { expected: 3, actual: 5 });
    assert!(s.history(RunId::new()).await.unwrap().is_empty());
}

#[tokio::test]
async fn branch_writes_are_dropped_by_the_next_save() {
    let s = InMemCheckpoints::with_history();
    let run = RunId::new();
    save_steps(&s, run, 1).await;
    assert!(s.save_branch(run, 1, BranchWrite::new(1, "b", json!("b1"))).await.unwrap());
    assert!(s.save_branch(run, 1, BranchWrite::new(0, "a", json!("a1"))).await.unwrap());
    assert!(s.save_branch(run, 2, BranchWrite::new(0, "a", json!("a2"))).await.unwrap());
    let writes = s.branch_writes(run, 1).await.unwrap();
    let order: Vec<(u32, &str)> = writes.iter().map(|w| (w.branch(), w.node())).collect();
    assert_eq!(order, vec![(0, "a"), (1, "b")]);
    s.save(Checkpoint { run_id: run, step: 1, state: json!(1) }).await.unwrap();
    assert!(s.branch_writes(run, 1).await.unwrap().is_empty());
    assert_eq!(s.branch_writes(run, 2).await.unwrap(), vec![BranchWrite::new(0, "a", json!("a2"))]);
    let err = s.save_branch(run, 1, BranchWrite::new(0, "a", json!("a1"))).await.unwrap_err();
    assert_eq!(err, CoreError::Conflict { expected: 2, actual: 1 });
    assert!(s.branch_writes(RunId::new(), 0).await.unwrap().is_empty());
}

#[tokio::test]
async fn second_branch_write_is_a_retry_only_with_the_same_write() {
    let s = InMemCheckpoints::with_history();
    let run = RunId::new();
    save_steps(&s, run, 1).await;
    let write = BranchWrite::new(0, "a", json!({"out": 1}));
    assert!(s.save_branch(run, 1, write.clone()).await.unwrap());
    assert!(s.save_branch(run, 1, write.clone()).await.unwrap());
    let other = BranchWrite::new(0, "a", json!({"out": 2}));
    let err = s.save_branch(run, 1, other).await.unwrap_err();
    assert!(matches!(err, CoreError::Conflict { .. }), "{err:?}");
    assert_eq!(s.branch_writes(run, 1).await.unwrap(), vec![write]);
}

#[tokio::test]
async fn fork_links_list_children() {
    let s = InMemCheckpoints::with_history();
    let (parent, other) = (RunId::new(), RunId::new());
    let (first, second, third) = (RunId::new(), RunId::new(), RunId::new());
    assert!(s.link_fork(ForkLink::new(first, parent, 4)).await.unwrap());
    assert!(s.link_fork(ForkLink::new(third, other, 0)).await.unwrap());
    assert!(s.link_fork(ForkLink::new(second, parent, 1)).await.unwrap());
    assert!(s.link_fork(ForkLink::new(first, parent, 4)).await.unwrap());
    let links = s.forks(parent).await.unwrap();
    assert_eq!(links, vec![ForkLink::new(first, parent, 4), ForkLink::new(second, parent, 1)]);
    assert_eq!(s.forks(other).await.unwrap(), vec![ForkLink::new(third, other, 0)]);
    assert!(s.forks(first).await.unwrap().is_empty());
    let err = s.link_fork(ForkLink::new(first, other, 4)).await.unwrap_err();
    assert!(matches!(err, CoreError::Conflict { .. }), "{err:?}");
    let err = s.link_fork(ForkLink::new(parent, parent, 0)).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}

#[tokio::test]
async fn prune_keeps_latest_with_history() {
    let s = InMemCheckpoints::with_history();
    let run = RunId::new();
    save_steps(&s, run, 5).await;
    assert_eq!(s.prune(run, 2).await.unwrap(), 3);
    let steps: Vec<u64> = s.history(run).await.unwrap().iter().map(|cp| cp.step).collect();
    assert_eq!(steps, vec![3, 4]);
    assert_eq!(s.prune(run, 10).await.unwrap(), 0);
    s.save(Checkpoint { run_id: run, step: 5, state: json!(5) }).await.unwrap();
    assert_eq!(s.latest(run).await.unwrap().unwrap().step, 5);
    assert_eq!(s.prune(RunId::new(), 1).await.unwrap(), 0);
    let err = s.prune(run, 0).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}
