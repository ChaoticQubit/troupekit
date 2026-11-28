use crate::error::CoreError;
use crate::ids::RunId;
use crate::message::Message;
use crate::store::*;
use async_trait::async_trait;
use std::collections::HashMap;
use tokio::sync::Mutex;
use troupekit_trace::{trace_span, Trace};

/// In-memory `CheckpointStore` for tests and examples.
/// Keeps only the latest checkpoint per run and enforces the same
/// next-step rule as durable stores.
#[derive(Default)]
pub struct InMemCheckpoints(Mutex<HashMap<RunId, Checkpoint>>);

#[async_trait]
impl CheckpointStore for InMemCheckpoints {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        let mut span = trace_span!(Trace::Store, "checkpoint.save", store = "memory", run = cp.run_id.to_string(), step = cp.step);
        let mut map = self.0.lock().await;
        let expected = map.get(&cp.run_id).map_or(0, |c| c.step + 1);
        if cp.step != expected {
            span.fail("step conflict");
            return Err(CoreError::Conflict { expected, actual: cp.step });
        }
        map.insert(cp.run_id, cp);
        Ok(())
    }

    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        let _span = trace_span!(Trace::Store, "checkpoint.latest", store = "memory", run = run_id.to_string());
        Ok(self.0.lock().await.get(&run_id).cloned())
    }
}

/// In-memory `MemoryLog` for tests and examples.
/// Applies the same conditional-append rule as durable stores;
/// an unknown log reads as empty.
#[derive(Default)]
pub struct InMemMemory(Mutex<HashMap<String, Vec<Message>>>);

#[async_trait]
impl MemoryLog for InMemMemory {
    async fn append(&self, log: &str, expected_len: u64, entries: Vec<Message>) -> Result<u64, CoreError> {
        let mut map = self.0.lock().await;
        let list = map.entry(log.to_string()).or_default();
        let actual = list.len() as u64;
        if actual != expected_len {
            return Err(CoreError::Conflict { expected: expected_len, actual });
        }
        list.extend(entries);
        Ok(list.len() as u64)
    }

    async fn read(&self, log: &str) -> Result<Vec<Message>, CoreError> {
        Ok(self.0.lock().await.get(log).cloned().unwrap_or_default())
    }
}

/// In-memory `VectorStore` doing a full cosine scan.
/// Fine for tests and small corpora; records with the same id are replaced.
/// Ties in score are broken by id for a stable order.
pub struct InMemVectors {
    dims: usize,
    records: Mutex<HashMap<String, VectorRecord>>,
}

impl InMemVectors {
    /// Creates an empty index for vectors of `dims` dimensions.
    /// Panics when `dims` is zero, which no embedding model produces.
    /// Returns the new index.
    pub fn new(dims: usize) -> Self {
        assert!(dims > 0, "vector dimension must be positive");
        Self { dims, records: Mutex::new(HashMap::new()) }
    }

    /// Checks a vector length `len` against the index dimension.
    /// Returns `CoreError::DimensionMismatch` when they differ,
    /// otherwise `Ok(())`.
    fn check(&self, len: usize) -> Result<(), CoreError> {
        if len == self.dims {
            Ok(())
        } else {
            Err(CoreError::DimensionMismatch { expected: self.dims, actual: len })
        }
    }
}

#[async_trait]
impl VectorStore for InMemVectors {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<(), CoreError> {
        for r in &records {
            self.check(r.vector.len())?;
        }
        let mut map = self.records.lock().await;
        for r in records {
            map.insert(r.id.clone(), r);
        }
        Ok(())
    }

    async fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<Scored>, CoreError> {
        self.check(query.len())?;
        let map = self.records.lock().await;
        let mut hits: Vec<Scored> = map
            .values()
            .filter(|r| filter.matches(&r.meta))
            .map(|r| Scored { id: r.id.clone(), score: cosine(query, &r.vector), text: r.text.clone(), meta: r.meta.clone() })
            .collect();
        hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
        hits.truncate(k);
        Ok(hits)
    }
}

/// In-memory `BlobStore` for tests and examples.
/// Holds every blob's bytes in a map keyed by path;
/// `get` of an unknown path returns `CoreError::NotFound`.
#[derive(Default)]
pub struct InMemBlobs(Mutex<HashMap<String, Vec<u8>>>);

#[async_trait]
impl BlobStore for InMemBlobs {
    async fn put(&self, path: &str, bytes: Vec<u8>) -> Result<(), CoreError> {
        self.0.lock().await.insert(path.to_string(), bytes);
        Ok(())
    }

    async fn get(&self, path: &str) -> Result<Vec<u8>, CoreError> {
        self.0.lock().await.get(path).cloned().ok_or_else(|| CoreError::NotFound(path.to_string()))
    }
}
