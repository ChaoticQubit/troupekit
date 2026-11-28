use crate::error::CoreError;
use crate::ids::RunId;
use crate::message::Message;
use crate::store::*;
use async_trait::async_trait;
use std::collections::{BTreeMap, HashMap};
use tokio::sync::Mutex;

/// In-memory `CheckpointStore` for tests and examples.
/// `default()` keeps only the latest checkpoint per run and no branch writes
/// or fork links; `with_history()` keeps every step, branch writes and fork
/// links. Both enforce the same next-step rule as durable stores.
#[derive(Default)]
pub struct InMemCheckpoints {
    keep_history: bool,
    inner: Mutex<CheckpointMaps>,
}

/// Everything one `InMemCheckpoints` holds, behind one lock so a save and
/// the next-step check on branch writes see one state. Branch writes are keyed by
/// step then branch index; `forks` lists links by child id, as SQLite does.
#[derive(Default)]
struct CheckpointMaps {
    runs: HashMap<RunId, Vec<Checkpoint>>,
    writes: HashMap<RunId, BTreeMap<(u64, u32), BranchWrite>>,
    links: Vec<ForkLink>,
}

impl InMemCheckpoints {
    /// Builds a store that keeps every saved step of every run, branch
    /// writes until `drop_branch_writes` and fork links, like a durable store.
    /// Memory grows with the steps saved; `prune` or the run's step cap bound it.
    /// Returns the empty store.
    pub fn with_history() -> Self {
        Self { keep_history: true, inner: Mutex::default() }
    }
}

/// Returns the step a run with checkpoints `kept` accepts next:
/// 0 for a run with none, else one above the latest kept step.
/// The caller holds the store lock, so the answer stays true while it acts.
fn next_step(kept: Option<&Vec<Checkpoint>>) -> u64 {
    kept.and_then(|list| list.last()).map_or(0, |cp| cp.step + 1)
}

#[async_trait]
impl CheckpointStore for InMemCheckpoints {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        let mut maps = self.inner.lock().await;
        let expected = next_step(maps.runs.get(&cp.run_id));
        if cp.step != expected {
            return Err(CoreError::Conflict { expected, actual: cp.step });
        }
        let kept = maps.runs.entry(cp.run_id).or_default();
        if !self.keep_history {
            kept.clear();
        }
        kept.push(cp);
        assert!(kept.windows(2).all(|w| w[0].step + 1 == w[1].step));
        Ok(())
    }

    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        Ok(self.inner.lock().await.runs.get(&run_id).and_then(|list| list.last()).cloned())
    }

    async fn history(&self, run_id: RunId) -> Result<Vec<Checkpoint>, CoreError> {
        Ok(self.inner.lock().await.runs.get(&run_id).cloned().unwrap_or_default())
    }

    async fn prune(&self, run_id: RunId, keep_last: u64) -> Result<u64, CoreError> {
        if keep_last == 0 {
            return Err(CoreError::Store("prune keep_last must be at least 1".into()));
        }
        let mut maps = self.inner.lock().await;
        let Some(kept) = maps.runs.get_mut(&run_id) else {
            return Ok(0);
        };
        let len = kept.len() as u64;
        let deleted = len.saturating_sub(keep_last);
        kept.drain(..deleted as usize);
        assert!(!kept.is_empty(), "prune never deletes the latest step");
        Ok(deleted)
    }

    async fn save_branch(&self, run_id: RunId, step: u64, write: BranchWrite) -> Result<bool, CoreError> {
        if !self.keep_history {
            return Ok(false);
        }
        let mut maps = self.inner.lock().await;
        let first_open = next_step(maps.runs.get(&run_id));
        if step < first_open {
            return Err(CoreError::Conflict { expected: first_open, actual: step });
        }
        let writes = maps.writes.entry(run_id).or_default();
        match writes.get(&(step, write.branch())) {
            Some(kept) if *kept == write => Ok(true),
            Some(_) => Err(CoreError::Conflict { expected: step, actual: step }),
            None => {
                writes.insert((step, write.branch()), write);
                Ok(true)
            }
        }
    }

    async fn branch_writes(&self, run_id: RunId, step: u64) -> Result<Vec<BranchWrite>, CoreError> {
        let maps = self.inner.lock().await;
        let Some(writes) = maps.writes.get(&run_id) else {
            return Ok(Vec::new());
        };
        let out: Vec<BranchWrite> = writes.range((step, 0)..=(step, u32::MAX)).map(|(_, w)| w.clone()).collect();
        assert!(out.windows(2).all(|w| w[0].branch() < w[1].branch()));
        Ok(out)
    }

    async fn drop_branch_writes(&self, run_id: RunId, below: u64) -> Result<u64, CoreError> {
        let mut maps = self.inner.lock().await;
        let Some(writes) = maps.writes.get_mut(&run_id) else {
            return Ok(0);
        };
        let before = writes.len();
        writes.retain(|&(write_step, _), _| write_step >= below);
        assert!(writes.keys().all(|&(write_step, _)| write_step >= below));
        Ok((before - writes.len()) as u64)
    }

    async fn link_fork(&self, link: ForkLink) -> Result<bool, CoreError> {
        if !self.keep_history {
            return Ok(false);
        }
        if link.child() == link.parent() {
            return Err(CoreError::Store("a run cannot be forked from itself".into()));
        }
        let mut maps = self.inner.lock().await;
        match maps.links.iter().find(|kept| kept.child() == link.child()) {
            Some(kept) if *kept == link => Ok(true),
            Some(kept) => Err(CoreError::Conflict { expected: kept.parent_step(), actual: link.parent_step() }),
            None => {
                maps.links.push(link);
                Ok(true)
            }
        }
    }

    async fn forks(&self, parent: RunId) -> Result<Vec<ForkLink>, CoreError> {
        let maps = self.inner.lock().await;
        let mut out: Vec<ForkLink> = maps.links.iter().filter(|link| link.parent() == parent).cloned().collect();
        out.sort_by_key(|link| link.child());
        Ok(out)
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
