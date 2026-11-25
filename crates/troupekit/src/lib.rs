//! Troupekit entry crate. `Troupe::builder()` wires the five stores, in
//! memory by default or in one SQLite file with `sqlite(path)`, and hands
//! out a `Runner` over them. The engines are re-exported as modules.

pub use troupekit_agent as agent;
pub use troupekit_core as core;
pub use troupekit_graph as graph;
pub use troupekit_host as host;

#[cfg(feature = "sqlite")]
pub use troupekit_store_sqlite::{Durability, SqliteStore, SqliteStoreBuilder};
#[cfg(feature = "knowledge")]
pub use troupekit_context as context;
#[cfg(feature = "knowledge")]
pub use troupekit_store_sqlite::{HnswParams, IndexKind, OpenReport, Quantization, SqliteVectors, VectorSettings};

#[cfg(feature = "knowledge")]
pub mod knowledge;

use std::sync::Arc;
use troupekit_agent::InMemoryLedger;
use troupekit_core::clock::{Clock, SystemClock};
use troupekit_core::error::CoreError;
use troupekit_core::inmem::{InMemBlobs, InMemCheckpoints, InMemMemory};
use troupekit_core::store::{BlobStore, CheckpointStore, JobQueue, MemoryLog, UsageLedger};
use troupekit_graph::GraphIo;
use troupekit_host::{InMemQueue, Runner};

enum Store {
    Memory,
    #[cfg(feature = "sqlite")]
    Sqlite(Box<SqliteStoreBuilder>),
}

/// Settings for a `Troupe`. With no store chosen the five stores live in
/// memory and nothing survives a restart. `sqlite` and `sqlite_with` put
/// all five in one file. `clock` sets the job queue's time source.
pub struct TroupeBuilder {
    store: Store,
    clock: Option<Arc<dyn Clock>>,
}

impl TroupeBuilder {
    /// Keeps every store in one SQLite file at `path` with every store
    /// setting at its default. Input: the file path. Output: the builder.
    #[cfg(feature = "sqlite")]
    pub fn sqlite(self, path: impl AsRef<std::path::Path>) -> Self {
        self.sqlite_with(SqliteStore::builder(path))
    }

    /// Keeps every store in one SQLite file opened from `store`, so any
    /// store setting (durability, caps, busy timeout, key) applies.
    /// Input: the store builder. Output: the builder.
    #[cfg(feature = "sqlite")]
    pub fn sqlite_with(mut self, store: SqliteStoreBuilder) -> Self {
        self.store = Store::Sqlite(Box::new(store));
        self
    }

    /// Sets the job queue's clock, in memory or SQLite. It replaces the
    /// clock of a builder given to `sqlite_with`. Use the clock of the
    /// runner's io. Input: the clock. Output: the builder.
    pub fn clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = Some(clock);
        self
    }

    /// Opens the chosen stores. Needs a tokio runtime for SQLite.
    /// Output: the troupe, or `CoreError::Store` when the file cannot be
    /// created, is foreign or newer, or a setting is out of range.
    pub async fn build(self) -> Result<Troupe, CoreError> {
        match self.store {
            Store::Memory => {
                let clock = self.clock.unwrap_or_else(|| Arc::new(SystemClock));
                Ok(Troupe {
                    checkpoints: Arc::new(InMemCheckpoints::default()),
                    memory: Arc::new(InMemMemory::default()),
                    blobs: Arc::new(InMemBlobs::default()),
                    queue: Arc::new(InMemQueue::new(clock)),
                    ledger: Arc::new(InMemoryLedger::default()),
                })
            }
            #[cfg(feature = "sqlite")]
            Store::Sqlite(builder) => {
                let builder = match self.clock {
                    Some(clock) => builder.clock(clock),
                    None => *builder,
                };
                let store = Arc::new(builder.open().await?);
                Ok(Troupe {
                    checkpoints: store.clone(),
                    memory: store.clone(),
                    blobs: store.clone(),
                    queue: store.clone(),
                    ledger: store,
                })
            }
        }
    }
}

/// The five stores of one Troupekit host, shared by every runner built
/// from it. Cheap to clone. Build it with `Troupe::builder()`.
#[derive(Clone)]
pub struct Troupe {
    checkpoints: Arc<dyn CheckpointStore>,
    memory: Arc<dyn MemoryLog>,
    blobs: Arc<dyn BlobStore>,
    queue: Arc<dyn JobQueue>,
    ledger: Arc<dyn UsageLedger>,
}

impl Troupe {
    /// Starts a builder with in-memory stores and the system clock.
    /// Output: the builder; nothing opens until `build`.
    pub fn builder() -> TroupeBuilder {
        TroupeBuilder {
            store: Store::Memory,
            clock: None,
        }
    }

    /// Returns the checkpoint store runs are saved to.
    pub fn checkpoints(&self) -> Arc<dyn CheckpointStore> {
        self.checkpoints.clone()
    }

    /// Returns the message log store.
    pub fn memory(&self) -> Arc<dyn MemoryLog> {
        self.memory.clone()
    }

    /// Returns the blob store.
    pub fn blobs(&self) -> Arc<dyn BlobStore> {
        self.blobs.clone()
    }

    /// Returns the job queue runners claim work from.
    pub fn queue(&self) -> Arc<dyn JobQueue> {
        self.queue.clone()
    }

    /// Returns the usage ledger. The builder does not plug it in: set
    /// `io.ledger = troupe.ledger()` on the `SimpleIo` given to the runner,
    /// or spend totals stay in memory and reset on restart.
    pub fn ledger(&self) -> Arc<dyn UsageLedger> {
        self.ledger.clone()
    }

    /// Builds a runner over this troupe's queue and checkpoints with `io`.
    /// Call it again for another worker on the same stores.
    /// Output: the runner with the default lease.
    pub fn runner(&self, io: Arc<dyn GraphIo>) -> Runner {
        Runner::new(self.queue.clone(), self.checkpoints.clone(), io)
    }
}
