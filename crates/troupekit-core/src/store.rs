use crate::error::CoreError;
use crate::event::Usage;
use crate::ids::RunId;
use crate::message::Message;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

/// Saved state of a run after one step.
/// Steps start at 0 and rise by one per saved checkpoint.
/// `state` is the run's serialized state machine.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Checkpoint {
    pub run_id: RunId,
    pub step: u64,
    pub state: Value,
}

/// Durable run snapshots.
/// `save` accepts only the next step and returns `CoreError::Conflict` otherwise,
/// so two workers can never both advance the same run.
#[async_trait]
pub trait CheckpointStore: Send + Sync {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError>;
    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError>;

    /// Returns the kept checkpoints of `run_id`, lowest step first.
    /// The default lists `latest` as zero or one checkpoint, which is
    /// exact for a store that keeps only the latest step.
    async fn history(&self, run_id: RunId) -> Result<Vec<Checkpoint>, CoreError> {
        Ok(self.latest(run_id).await?.into_iter().collect())
    }

    /// Deletes all but the `keep_last` highest steps of `run_id` and returns
    /// how many it deleted. The latest step is never deleted, so `save` keeps
    /// its next-step rule. `keep_last == 0` returns `CoreError::Store`.
    /// The default deletes nothing.
    async fn prune(&self, run_id: RunId, keep_last: u64) -> Result<u64, CoreError> {
        let _ = run_id;
        if keep_last == 0 {
            return Err(CoreError::Store("prune keep_last must be at least 1".into()));
        }
        Ok(0)
    }

    /// Durably keeps the write of one finished branch of step `step` of
    /// `run_id`. Returns true when the write is durable. The default keeps
    /// whole steps only and returns false, so the engine knows the branch
    /// will run again after a crash.
    async fn save_branch(&self, run_id: RunId, step: u64, write: BranchWrite) -> Result<bool, CoreError> {
        let _ = (run_id, step, write);
        Ok(false)
    }

    /// Returns the kept branch writes of step `step` of `run_id`, lowest
    /// branch index first. The default keeps none and returns an empty
    /// list, so a resumed run runs every branch of the step again.
    async fn branch_writes(&self, run_id: RunId, step: u64) -> Result<Vec<BranchWrite>, CoreError> {
        let _ = (run_id, step);
        Ok(Vec::new())
    }

    /// Deletes the kept branch writes of `run_id` at steps below `below` and
    /// returns how many it deleted. `save` and `prune` never delete a write,
    /// since a checkpoint can still carry an earlier step's writes; the caller
    /// passes the lowest step its saved run still reads. The default keeps none.
    async fn drop_branch_writes(&self, run_id: RunId, below: u64) -> Result<u64, CoreError> {
        let _ = (run_id, below);
        Ok(0)
    }

    /// Records that `link.child` was forked from `link.parent`.
    /// Returns true when the link is kept. The default keeps no lineage
    /// and returns false.
    async fn link_fork(&self, link: ForkLink) -> Result<bool, CoreError> {
        let _ = link;
        Ok(false)
    }

    /// Returns the kept fork links whose parent is `parent`.
    /// Children of children are not listed; walk them one level at a time.
    /// The default keeps none and returns an empty list.
    async fn forks(&self, parent: RunId) -> Result<Vec<ForkLink>, CoreError> {
        let _ = parent;
        Ok(Vec::new())
    }
}

/// The result of one finished parallel branch within one step: the branch
/// index, the node it worked on and its serialized post-work state.
/// A store that keeps branch writes lets a resumed run apply finished
/// branches instead of running them again.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BranchWrite {
    branch: u32,
    node: String,
    write: Value,
}

impl BranchWrite {
    /// Builds the write of branch index `branch` that worked on `node`
    /// with post-work state `write`.
    /// Returns the write.
    pub fn new(branch: u32, node: impl Into<String>, write: Value) -> Self {
        Self {
            branch,
            node: node.into(),
            write,
        }
    }

    /// Returns the branch index within its step.
    /// Indices follow the run's branch order at the start of the step.
    /// A resumed run applies the write to the branch at this index.
    pub fn branch(&self) -> u32 {
        self.branch
    }

    /// Returns the id of the node the branch worked on.
    /// A resumed run refuses the write when the branch at its index sits
    /// at another node.
    pub fn node(&self) -> &str {
        &self.node
    }

    /// Returns the branch's serialized post-work state.
    /// Its shape belongs to the engine that wrote it.
    /// The store keeps it as opaque JSON.
    pub fn write(&self) -> &Value {
        &self.write
    }
}

/// A lineage link: run `child` was forked from step `parent_step` of run
/// `parent`. The child holds a copy of that checkpoint, so pruning the
/// parent never breaks the child.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ForkLink {
    child: RunId,
    parent: RunId,
    parent_step: u64,
}

impl ForkLink {
    /// Builds the link from run `child` to step `parent_step` of `parent`.
    /// Returns the link.
    pub fn new(child: RunId, parent: RunId, parent_step: u64) -> Self {
        Self {
            child,
            parent,
            parent_step,
        }
    }

    /// Returns the forked run.
    /// It starts at step 0 with a copy of the parent checkpoint.
    /// Its id differs from the parent's.
    pub fn child(&self) -> RunId {
        self.child
    }

    /// Returns the run the fork was taken from.
    /// That run's checkpoints are never written by the fork.
    /// Lineage walks from child to parent.
    pub fn parent(&self) -> RunId {
        self.parent
    }

    /// Returns the parent step the fork copied.
    /// It names one checkpoint of the parent run.
    /// The fork's own steps restart at 0.
    pub fn parent_step(&self) -> u64 {
        self.parent_step
    }
}

/// Append-only message logs keyed by name.
/// `append` succeeds only when the log still has `expected_len` entries and
/// returns the new length; otherwise it returns `CoreError::Conflict`.
#[async_trait]
pub trait MemoryLog: Send + Sync {
    async fn append(&self, log: &str, expected_len: u64, entries: Vec<Message>) -> Result<u64, CoreError>;
    async fn read(&self, log: &str) -> Result<Vec<Message>, CoreError>;
}

/// One embedded chunk stored in a `VectorStore`.
/// Holds the id, the embedding, the source text and string metadata
/// used by `Filter`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: String,
    pub vector: Vec<f32>,
    pub text: String,
    pub meta: BTreeMap<String, String>,
}

/// A search hit from a `VectorStore`.
/// Carries the record's id, text and metadata plus its cosine similarity
/// to the query; higher scores are closer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scored {
    pub id: String,
    pub score: f32,
    pub text: String,
    pub meta: BTreeMap<String, String>,
}

/// Equality filter on record metadata.
/// Every key and value pair in the filter must match the record;
/// an empty filter matches every record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Filter(pub BTreeMap<String, String>);

impl Filter {
    /// Checks the filter against a record's `meta`.
    /// Returns true when every filter pair is present in `meta`
    /// with the same value.
    pub fn matches(&self, meta: &BTreeMap<String, String>) -> bool {
        self.0.iter().all(|(k, v)| meta.get(k) == Some(v))
    }
}

/// Object-safe vector index of a fixed dimension, so a store can be chosen
/// at runtime. `upsert` and `search` return `CoreError::DimensionMismatch`
/// for vectors whose length is not `dims`.
#[async_trait]
pub trait VectorStore: Send + Sync {
    fn dims(&self) -> usize;
    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<(), CoreError>;
    async fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<Scored>, CoreError>;
}

/// Object-safe keyword search over stored text, so a store can be chosen
/// at runtime. `search_text` ranks records by BM25, higher scores first,
/// keeps only records whose meta matches `filter` and returns at most `k`
/// hits. Query text is plain words, never a query language.
#[async_trait]
pub trait KeywordStore: Send + Sync {
    async fn search_text(&self, query: &str, k: usize, filter: &Filter) -> Result<Vec<Scored>, CoreError>;
}

/// Byte storage by path, for attached files and media.
/// `put` replaces any existing bytes at the path.
/// `get` of an unknown path returns `CoreError::NotFound`.
#[async_trait]
pub trait BlobStore: Send + Sync {
    async fn put(&self, path: &str, bytes: Vec<u8>) -> Result<(), CoreError>;
    async fn get(&self, path: &str) -> Result<Vec<u8>, CoreError>;
}

/// Cosine similarity of vectors `a` and `b`, which must have equal length.
/// Returns a value in -1.0..=1.0, or 0.0 when either vector is all zeros.
/// Panics on unequal lengths; callers check dimensions first.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "cosine of unequal lengths");
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

/// A claimed job: the run to step, the lease token that proves the claim
/// and how many times the job was claimed since it last completed or was
/// rescheduled. A token from an expired, re-claimed lease is stale.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Lease {
    pub run_id: RunId,
    pub token: u64,
    pub attempts: u32,
}

/// Where one run's job stands. `Pending` waits for `due_ms`; `Leased` is
/// held until `until_ms`, with `next_due_ms` from an enqueue during the
/// lease; `Done` and `Failed` end it until the next enqueue.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobState {
    Pending {
        due_ms: i64,
    },
    Leased {
        token: u64,
        until_ms: i64,
        next_due_ms: Option<i64>,
    },
    Done,
    Failed {
        error: String,
    },
}

/// Jobs keyed by run, at most one live job per run. `claim_due` leases due
/// jobs and expired leases, earliest first; `complete`, `fail` and
/// `reschedule` end a lease and return `CoreError::Conflict` for a stale one.
#[async_trait]
pub trait JobQueue: Send + Sync {
    async fn enqueue(&self, run_id: RunId, due_ms: i64) -> Result<(), CoreError>;
    async fn claim_due(&self, limit: usize, lease_ms: i64) -> Result<Vec<Lease>, CoreError>;
    async fn complete(&self, lease: &Lease) -> Result<(), CoreError>;
    async fn fail(&self, lease: &Lease, error: &str) -> Result<(), CoreError>;
    async fn reschedule(&self, lease: &Lease, due_ms: i64) -> Result<(), CoreError>;
}

/// Spend totals per key for the current period, kept by the host (for
/// example one row per goal and month). `period_total` returns millionths
/// of a US dollar spent so far; `record` adds one call's usage.
#[async_trait]
pub trait UsageLedger: Send + Sync {
    async fn period_total(&self, key: &str) -> Result<u64, CoreError>;
    async fn record(&self, key: &str, usage: Usage) -> Result<(), CoreError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct LatestOnly(Mutex<Option<Checkpoint>>);

    #[async_trait]
    impl CheckpointStore for LatestOnly {
        async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
            *self.0.lock().unwrap() = Some(cp);
            Ok(())
        }

        async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
            Ok(self.0.lock().unwrap().clone().filter(|cp| cp.run_id == run_id))
        }
    }

    #[tokio::test]
    async fn store_without_branch_writes_reports_not_durable() {
        let store = LatestOnly::default();
        let run = RunId::new();
        let write = BranchWrite::new(1, "left", serde_json::json!({"at": "leave"}));
        assert_eq!(write.branch(), 1);
        assert_eq!(write.node(), "left");
        assert!(!store.save_branch(run, 3, write).await.unwrap());
        assert!(store.branch_writes(run, 3).await.unwrap().is_empty());
        let parent = RunId::new();
        let link = ForkLink::new(run, parent, 7);
        assert_eq!((link.child(), link.parent(), link.parent_step()), (run, parent, 7));
        assert!(!store.link_fork(link).await.unwrap());
        assert!(store.forks(parent).await.unwrap().is_empty());
    }
}
