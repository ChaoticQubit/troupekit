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
