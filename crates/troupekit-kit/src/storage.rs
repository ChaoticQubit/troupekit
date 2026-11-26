use crate::error::KitError;
use std::fmt;
#[cfg(feature = "sqlite")]
use std::path::PathBuf;
use std::sync::Arc;
use troupekit_core::store::CheckpointStore;

/// Where an agent or graph keeps run checkpoints.
/// `memory()` (the default) uses a fresh in-memory store per run, so
/// nothing grows; `sqlite(path)` and `custom(store)` keep runs for `resume`.
#[derive(Clone)]
pub struct Storage(Kind);

#[derive(Clone)]
enum Kind {
    Memory,
    #[cfg(feature = "sqlite")]
    Sqlite(PathBuf),
    Custom(Arc<dyn CheckpointStore>),
}

impl Storage {
    #[cfg(feature = "knowledge")]
    pub(crate) fn sqlite_path(&self) -> Option<&std::path::Path> {
        match &self.0 {
            Kind::Sqlite(path) => Some(path),
            _ => None,
        }
    }

    /// Keeps each run's checkpoints in memory, dropped when the run ends.
    /// Runs cannot be resumed. This is the default.
    /// Costs nothing at `build`.
    pub fn memory() -> Self {
        Storage(Kind::Memory)
    }

    /// Keeps checkpoints in the SQLite file at `path`, opened at `build`.
    /// Enables `resume(run_id)`, also from a later process.
    /// Needs the `sqlite` feature, on by default.
    #[cfg(feature = "sqlite")]
    pub fn sqlite(path: impl Into<PathBuf>) -> Self {
        Storage(Kind::Sqlite(path.into()))
    }

    /// Keeps checkpoints in `store`, shared by every run.
    /// Enables `resume(run_id)`.
    /// Use it for a store the host already opened.
    pub fn custom(store: Arc<dyn CheckpointStore>) -> Self {
        Storage(Kind::Custom(store))
    }

    pub(crate) async fn open(&self) -> Result<Option<Arc<dyn CheckpointStore>>, KitError> {
        match &self.0 {
            Kind::Memory => Ok(None),
            #[cfg(feature = "sqlite")]
            Kind::Sqlite(path) => {
                let store = troupekit_store_sqlite::SqliteStore::open(path).await?;
                Ok(Some(Arc::new(store)))
            }
            Kind::Custom(store) => Ok(Some(store.clone())),
        }
    }
}

impl Default for Storage {
    fn default() -> Self {
        Storage::memory()
    }
}

impl fmt::Debug for Storage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.0 {
            Kind::Memory => f.write_str("Storage::memory()"),
            #[cfg(feature = "sqlite")]
            Kind::Sqlite(path) => write!(f, "Storage::sqlite({path:?})"),
            Kind::Custom(_) => f.write_str("Storage::custom(..)"),
        }
    }
}
