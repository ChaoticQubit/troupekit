use crate::chunk::ChunkConfig;
use crate::embed::Embedder;
use crate::error::ContextError;
use crate::facts::FactBook;
use crate::ingest::{ingest, Document};
use crate::retrieve::{retrieve, Hit, RetrievalConfig};
use std::sync::Arc;
use troupekit_core::clock::Clock;
use troupekit_core::ids::RunId;
use troupekit_core::message::Message;
use troupekit_core::store::{Filter, MemoryLog, VectorStore};

const RUN_LOG_PREFIX: &str = "run/";

/// One API over the core stores for every memory scope: run history and
/// profile facts in a `MemoryLog`, scoped knowledge in a `VectorStore`,
/// and retrieval across linked scopes with the given embedder and clock.
pub struct Memory {
    log: Arc<dyn MemoryLog>,
    vectors: Arc<dyn VectorStore>,
    embedder: Arc<dyn Embedder>,
    clock: Arc<dyn Clock>,
    facts: FactBook,
}

impl Memory {
    /// Bundles the stores, the embedder and the clock into one memory.
    /// Returns `ContextError::Invalid` when the embedder's dimension differs
    /// from the vector store's, otherwise the new memory.
    pub fn new(log: Arc<dyn MemoryLog>, vectors: Arc<dyn VectorStore>, embedder: Arc<dyn Embedder>, clock: Arc<dyn Clock>) -> Result<Self, ContextError> {
        if embedder.dims() != vectors.dims() {
            return Err(ContextError::Invalid(format!("embedder has {} dimensions, store has {}", embedder.dims(), vectors.dims())));
        }
        let facts = FactBook::new(log.clone());
        Ok(Self { log, vectors, embedder, clock, facts })
    }

    /// Reads the message history of `run`, oldest first.
    /// An unknown run reads as empty.
    /// Store errors pass through as `ContextError::Core`.
    pub async fn history(&self, run: RunId) -> Result<Vec<Message>, ContextError> {
        Ok(self.log.read(&run_log(run)).await?)
    }

    /// Appends `entries` to the history of `run` if it still has `expected_len`
    /// messages. Returns the new length, or `CoreError::Conflict` inside
    /// `ContextError::Core` when another writer got there first.
    pub async fn append_history(&self, run: RunId, expected_len: u64, entries: Vec<Message>) -> Result<u64, ContextError> {
        Ok(self.log.append(&run_log(run), expected_len, entries).await?)
    }

    /// Returns the profile facts book, which shares this memory's log.
    /// Facts are stored per subject and listed newest per key.
    /// See `FactBook` for the rules.
    pub fn facts(&self) -> &FactBook {
        &self.facts
    }

    /// Chunks, embeds and stores `doc` in its scope with `cfg` limits.
    /// Returns the chunk count; see `ingest::ingest` for validation.
    /// Re-ingesting a source replaces its chunks by id.
    pub async fn ingest(&self, doc: &Document, cfg: &ChunkConfig) -> Result<usize, ContextError> {
        ingest(self.vectors.as_ref(), self.embedder.as_ref(), doc, cfg).await
    }

    /// Retrieves the best chunks for `query` across the linked `scopes`,
    /// restricted by `base`, such as a tenant filter.
    /// Returns ranked hits; see `retrieve::retrieve` for scoring and validation.
    pub async fn search(&self, query: &str, scopes: &[&str], base: &Filter, cfg: &RetrievalConfig) -> Result<Vec<Hit>, ContextError> {
        retrieve(self.vectors.as_ref(), self.embedder.as_ref(), self.clock.as_ref(), query, scopes, base, cfg).await
    }
}

/// Builds the log name holding the history of `run`.
/// Run logs and fact logs use different prefixes in the shared log.
/// Returns `run/<run_id>`.
fn run_log(run: RunId) -> String {
    format!("{RUN_LOG_PREFIX}{run}")
}
