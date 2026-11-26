//! SQLite storage for Troupekit. One `SqliteStore` over one file implements
//! `CheckpointStore`, `MemoryLog`, `BlobStore`, `JobQueue` and `UsageLedger`,
//! so a run survives a process kill and resumes from its last saved step.
//! `SqliteStore::vectors` opens named vector indexes in the same file.

mod queue;
mod schema;
mod stores;
mod vectors;

use rusqlite::config::DbConfig;
use rusqlite::{Connection, OpenFlags};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit_core::clock::{Clock, SystemClock};
use troupekit_core::error::CoreError;
#[cfg(any(feature = "sqlcipher", feature = "sqlcipher-system"))]
use troupekit_core::secret::Secret;
pub use troupekit_index::{HnswParams, IndexKind, Quantization};
pub use vectors::{OpenReport, SqliteVectors, VectorSettings};

/// The schema version this troupekit writes and reads up to.
/// It equals the number of migration steps. A file at a higher version is
/// refused, a file at a lower one is migrated on open.
pub const SCHEMA_VERSION: u32 = schema::STEPS.len() as u32;

const _: () = assert!(SCHEMA_VERSION > 0);

const BYTES_LIMIT_SQLITE: u64 = 1_000_000_000;
const STATE_BYTES_MAX_DEFAULT: u64 = 64 * 1024 * 1024;
const BLOB_BYTES_MAX_DEFAULT: u64 = 64 * 1024 * 1024;
const HISTORY_STEPS_MAX_DEFAULT: u64 = 100_000;
const BUSY_TIMEOUT_MS_DEFAULT: u32 = 5_000;

/// How hard a commit is pushed to disk, per connection. `Full` survives
/// power loss: every commit uses `fullfsync`, which Apple platforms need.
/// `Normal` is faster and can lose the last commits on power loss, never on
/// a process kill; WAL checkpoints still use `fullfsync`, so no corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum Durability {
    #[default]
    Full,
    Normal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Limits {
    pub(crate) state_bytes_max: u64,
    pub(crate) blob_bytes_max: u64,
    pub(crate) history_steps_max: u64,
    pub(crate) branch_write_bytes_max: Option<u64>,
}

impl Limits {
    /// Returns the largest branch write `save_branch` accepts, in bytes:
    /// the explicit setting, else `state_bytes_max`, so a branch write is
    /// never allowed to be larger than the checkpoint it becomes part of.
    pub(crate) fn branch_write_bytes(&self) -> u64 {
        self.branch_write_bytes_max.unwrap_or(self.state_bytes_max)
    }
}

/// Settings for `SqliteStore`, each with a safe default: full durability,
/// the system clock, 64 MiB caps on one state and one blob, a branch write
/// cap equal to the state cap, 100 000 history steps, a 5 000 ms busy
/// timeout and secure delete on. `open` validates them and opens the file.
#[derive(Clone)]
pub struct SqliteStoreBuilder {
    path: PathBuf,
    durability: Durability,
    clock: Arc<dyn Clock>,
    limits: Limits,
    busy_timeout_ms: u32,
    secure_delete: bool,
    #[cfg(any(feature = "sqlcipher", feature = "sqlcipher-system"))]
    key: Option<Secret>,
}

impl std::fmt::Debug for SqliteStoreBuilder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStoreBuilder")
            .field("path", &self.path)
            .field("durability", &self.durability)
            .field("limits", &self.limits)
            .field("busy_timeout_ms", &self.busy_timeout_ms)
            .field("secure_delete", &self.secure_delete)
            .finish_non_exhaustive()
    }
}

impl SqliteStoreBuilder {
    /// Sets how hard commits are pushed to disk.
    /// Input: the level. Output: the builder. Default `Durability::Full`.
    pub fn durability(mut self, durability: Durability) -> Self {
        self.durability = durability;
        self
    }

    /// Sets the time source of the job queue. Share it with the runner's io
    /// so leases and due times agree. Input: the clock. Output: the builder.
    /// Default `SystemClock`.
    pub fn clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Sets the largest serialized checkpoint state `save` accepts, in bytes.
    /// Input: 1 to 1 000 000 000. Output: the builder. Default 64 MiB.
    pub fn max_state_bytes(mut self, bytes: u64) -> Self {
        self.limits.state_bytes_max = bytes;
        self
    }

    /// Sets the largest blob `put` accepts, in bytes.
    /// Input: 1 to 1 000 000 000. Output: the builder. Default 64 MiB.
    pub fn max_blob_bytes(mut self, bytes: u64) -> Self {
        self.limits.blob_bytes_max = bytes;
        self
    }

    /// Sets the most rows `history`, `branch_writes` and `forks` return
    /// before they error. Input: at least 1. Output: the builder.
    /// Default 100 000.
    pub fn max_history_steps(mut self, steps: u64) -> Self {
        self.limits.history_steps_max = steps;
        self
    }

    /// Sets the largest serialized branch write `save_branch` accepts, in
    /// bytes, counting its node id and its JSON. Input: 1 to 1 000 000 000.
    /// Output: the builder. Default: the `max_state_bytes` value.
    pub fn max_branch_write_bytes(mut self, bytes: u64) -> Self {
        self.limits.branch_write_bytes_max = Some(bytes);
        self
    }

    /// Sets how long a writer waits for another handle's lock, in ms.
    /// Input: any value, 0 fails at once. Output: the builder. Default 5 000.
    pub fn busy_timeout_ms(mut self, ms: u32) -> Self {
        self.busy_timeout_ms = ms;
        self
    }

    /// Sets whether deleted rows are overwritten: SQLite `secure_delete` on
    /// the connection and FTS5 `secure-delete` on the keyword index, so a
    /// removed chunk's text and vector leave the file. Cost: every delete
    /// writes zeros over freed space, more disk writes and slower deletes.
    /// Input: on or off. Output: the builder. Default on.
    pub fn secure_delete(mut self, on: bool) -> Self {
        self.secure_delete = on;
        self
    }

    /// Opens the file encrypted with SQLCipher under `key`. The key is sent
    /// first on the connection and never appears in `Debug` or an error.
    /// Input: the key. Output: the builder. Default: no key, plain file.
    #[cfg(any(feature = "sqlcipher", feature = "sqlcipher-system"))]
    pub fn key(mut self, key: Secret) -> Self {
        self.key = Some(key);
        self
    }

    /// Validates the settings, creates the file when missing, refuses a
    /// foreign or newer file and migrates an older one. Needs a tokio
    /// runtime. Output: the store, or `CoreError::Store`.
    pub async fn open(self) -> Result<SqliteStore, CoreError> {
        validate(&self.limits)?;
        let settings = self.clone();
        let conn = tokio::task::spawn_blocking(move || open_connection(&settings))
            .await
            .unwrap_or_else(|e| join_failed(e))?;
        Ok(SqliteStore {
            inner: Arc::new(Inner {
                conn: Mutex::new(conn),
                clock: self.clock,
                limits: self.limits,
                path: self.path,
                secure_delete: self.secure_delete,
                vectors: Mutex::new(HashMap::new()),
            }),
        })
    }
}

fn validate(limits: &Limits) -> Result<(), CoreError> {
    for (name, value) in [
        ("max_state_bytes", limits.state_bytes_max),
        ("max_blob_bytes", limits.blob_bytes_max),
        ("max_branch_write_bytes", limits.branch_write_bytes()),
    ] {
        if value == 0 || value > BYTES_LIMIT_SQLITE {
            return Err(CoreError::Store(format!(
                "{name} must be 1 to {BYTES_LIMIT_SQLITE}, got {value}"
            )));
        }
    }
    if limits.history_steps_max == 0 {
        return Err(CoreError::Store(
            "max_history_steps must be at least 1".into(),
        ));
    }
    Ok(())
}

/// Creates the store file when it does not exist yet, so the read-write
/// open that follows finds it on every platform. On Unix the new file is
/// 0600. Input: the path. Output: `Ok` when it created the file or found
/// an entry there, `CoreError::Store` when it cannot be created.
fn create_private(path: &Path) -> Result<(), CoreError> {
    if path.exists() {
        return Ok(());
    }
    let mut options = std::fs::OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
    options.open(path).map(drop).or_else(|e| match e.kind() {
        std::io::ErrorKind::AlreadyExists => Ok(()),
        _ => Err(CoreError::Store(format!("cannot create store: {e}"))),
    })
}

fn open_connection(settings: &SqliteStoreBuilder) -> Result<Connection, CoreError> {
    create_private(&settings.path)?;
    let flags = OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let mut conn = Connection::open_with_flags(&settings.path, flags)
        .map_err(|e| CoreError::Store(format!("cannot open store: {e}")))?;
    #[cfg(any(feature = "sqlcipher", feature = "sqlcipher-system"))]
    if let Some(key) = &settings.key {
        conn.pragma_update(None, "key", key.expose())
            .map_err(|_| wrong_key())?;
    }
    conn.query_row("SELECT COUNT(*) FROM sqlite_schema", [], |row| {
        row.get::<_, i64>(0)
    })
    .map_err(|e| match e.sqlite_error_code() {
        Some(rusqlite::ErrorCode::NotADatabase) => wrong_key(),
        _ => db(e),
    })?;
    conn.busy_timeout(Duration::from_millis(u64::from(settings.busy_timeout_ms)))
        .map_err(db)?;
    schema::check_header(schema::read_header(&conn)?, SCHEMA_VERSION)?;
    let mode: String = conn
        .pragma_update_and_check(None, "journal_mode", "WAL", |row| row.get(0))
        .map_err(db)?;
    if !mode.eq_ignore_ascii_case("wal") {
        return Err(CoreError::Store(format!(
            "store cannot use WAL, got {mode}"
        )));
    }
    let (synchronous, fullfsync) = match settings.durability {
        Durability::Full => ("FULL", true),
        Durability::Normal => ("NORMAL", false),
    };
    let checkpoint_fullfsync = true;
    conn.pragma_update(None, "synchronous", synchronous)
        .map_err(db)?;
    conn.pragma_update(None, "fullfsync", fullfsync)
        .map_err(db)?;
    conn.pragma_update(None, "checkpoint_fullfsync", checkpoint_fullfsync)
        .map_err(db)?;
    conn.pragma_update(None, "trusted_schema", "OFF")
        .map_err(db)?;
    conn.pragma_update(None, "secure_delete", settings.secure_delete)
        .map_err(db)?;
    conn.set_db_config(DbConfig::SQLITE_DBCONFIG_DEFENSIVE, true)
        .map_err(db)?;
    let version = schema::migrate(&mut conn, schema::STEPS)?;
    assert_eq!(version, SCHEMA_VERSION);
    assert_eq!(schema::read_header(&conn)?.user_version, SCHEMA_VERSION);
    Ok(conn)
}

fn wrong_key() -> CoreError {
    CoreError::Store("cannot open store: wrong key or not a database".into())
}

/// Maps a rusqlite error to `CoreError::Store`. The text names what failed
/// and never holds a bound value, so state, blob bytes and keys stay out.
/// Input: the error. Output: the store error.
pub(crate) fn db(e: rusqlite::Error) -> CoreError {
    CoreError::Store(format!("sqlite: {e}"))
}

fn join_failed<T>(e: tokio::task::JoinError) -> Result<T, CoreError> {
    if e.is_panic() {
        std::panic::resume_unwind(e.into_panic());
    }
    Err(CoreError::Store(format!("store task did not finish: {e}")))
}

struct Inner {
    conn: Mutex<Connection>,
    clock: Arc<dyn Clock>,
    limits: Limits,
    path: PathBuf,
    secure_delete: bool,
    vectors: Mutex<HashMap<String, Arc<vectors::IndexState>>>,
}

/// Durable Troupekit store over one SQLite file in WAL mode. Cheap to
/// clone; clones share one connection. Every call runs on tokio's blocking
/// pool, so it needs a tokio runtime. A caller that stops waiting does not
/// stop the call: it runs to its commit or rollback.
#[derive(Clone)]
pub struct SqliteStore {
    inner: Arc<Inner>,
}

impl std::fmt::Debug for SqliteStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteStore")
            .field("path", &self.inner.path)
            .field("limits", &self.inner.limits)
            .finish_non_exhaustive()
    }
}

impl SqliteStore {
    /// Starts a builder for the file at `path` with every setting at its
    /// default. Input: the file path. Output: the builder; nothing is
    /// opened until `open`.
    pub fn builder(path: impl AsRef<Path>) -> SqliteStoreBuilder {
        SqliteStoreBuilder {
            path: path.as_ref().to_path_buf(),
            durability: Durability::Full,
            clock: Arc::new(SystemClock),
            limits: Limits {
                state_bytes_max: STATE_BYTES_MAX_DEFAULT,
                blob_bytes_max: BLOB_BYTES_MAX_DEFAULT,
                history_steps_max: HISTORY_STEPS_MAX_DEFAULT,
                branch_write_bytes_max: None,
            },
            busy_timeout_ms: BUSY_TIMEOUT_MS_DEFAULT,
            secure_delete: true,
            #[cfg(any(feature = "sqlcipher", feature = "sqlcipher-system"))]
            key: None,
        }
    }

    /// Opens the file at `path` with every setting at its default.
    /// Same as `SqliteStore::builder(path).open()`.
    /// Output: the store, or `CoreError::Store`.
    pub async fn open(path: impl AsRef<Path>) -> Result<SqliteStore, CoreError> {
        Self::builder(path).open().await
    }

    pub(crate) fn limits(&self) -> Limits {
        self.inner.limits
    }

    /// Runs `f` with the locked connection on the blocking pool. A panic in
    /// `f` reaches the caller; a transaction guard open in `f` rolls back as
    /// it unwinds. A poisoned lock is taken and any open transaction is
    /// rolled back first. Output: what `f` returns.
    pub(crate) async fn call<T, F>(&self, f: F) -> Result<T, CoreError>
    where
        T: Send + 'static,
        F: FnOnce(&mut Connection) -> Result<T, CoreError> + Send + 'static,
    {
        let inner = Arc::clone(&self.inner);
        tokio::task::spawn_blocking(move || {
            let mut conn = match inner.conn.lock() {
                Ok(conn) => conn,
                Err(poisoned) => {
                    let conn = poisoned.into_inner();
                    if !conn.is_autocommit() {
                        conn.execute_batch("ROLLBACK").map_err(db)?;
                    }
                    inner.conn.clear_poison();
                    conn
                }
            };
            assert!(
                conn.is_autocommit(),
                "a store call starts outside a transaction"
            );
            let out = f(&mut conn);
            assert!(conn.is_autocommit(), "a store call left a transaction open");
            out
        })
        .await
        .unwrap_or_else(|e| join_failed(e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::TransactionBehavior;

    async fn temp_store(
        builder: impl FnOnce(SqliteStoreBuilder) -> SqliteStoreBuilder,
    ) -> (tempfile::TempDir, SqliteStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = builder(SqliteStore::builder(dir.path().join("t.db")))
            .open()
            .await
            .unwrap();
        (dir, store)
    }

    async fn pragma(store: &SqliteStore, name: &'static str) -> i64 {
        store
            .call(move |conn| {
                conn.pragma_query_value(None, name, |row| row.get(0))
                    .map_err(db)
            })
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn defaults_are_the_safe_values() {
        let builder = SqliteStore::builder("unused.db");
        assert_eq!(builder.durability, Durability::Full);
        assert_eq!(builder.limits.state_bytes_max, 64 * 1024 * 1024);
        assert_eq!(builder.limits.blob_bytes_max, 64 * 1024 * 1024);
        assert_eq!(builder.limits.history_steps_max, 100_000);
        assert_eq!(builder.limits.branch_write_bytes_max, None);
        assert_eq!(builder.limits.branch_write_bytes(), 64 * 1024 * 1024);
        assert_eq!(builder.clone().max_state_bytes(100).limits.branch_write_bytes(), 100);
        assert_eq!(builder.busy_timeout_ms, 5_000);
        assert!(builder.secure_delete);
        let before = SystemClock.now_ms();
        assert!(builder.clock.now_ms() >= before);
        let (_dir, store) = temp_store(|b| b).await;
        assert_eq!(pragma(&store, "synchronous").await, 2);
        assert_eq!(pragma(&store, "fullfsync").await, 1);
        assert_eq!(pragma(&store, "checkpoint_fullfsync").await, 1);
        assert_eq!(pragma(&store, "busy_timeout").await, 5_000);
        assert_eq!(pragma(&store, "trusted_schema").await, 0);
        assert_eq!(pragma(&store, "secure_delete").await, 1);
    }

    #[tokio::test]
    async fn secure_delete_off_reaches_the_connection() {
        let (_dir, store) = temp_store(|b| b.secure_delete(false)).await;
        assert_eq!(pragma(&store, "secure_delete").await, 0);
    }

    #[tokio::test]
    async fn durability_normal_sets_synchronous_normal() {
        let (_dir, store) = temp_store(|b| b.durability(Durability::Normal)).await;
        assert_eq!(pragma(&store, "synchronous").await, 1);
        assert_eq!(pragma(&store, "fullfsync").await, 0);
        assert_eq!(pragma(&store, "checkpoint_fullfsync").await, 1);
    }

    #[tokio::test]
    async fn busy_timeout_setting_reaches_the_connection() {
        let (_dir, store) = temp_store(|b| b.busy_timeout_ms(1234)).await;
        assert_eq!(pragma(&store, "busy_timeout").await, 1234);
    }

    #[tokio::test]
    async fn zero_cap_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        for builder in [
            SqliteStore::builder(&path).max_state_bytes(0),
            SqliteStore::builder(&path).max_blob_bytes(0),
            SqliteStore::builder(&path).max_history_steps(0),
        ] {
            let err = builder.open().await.unwrap_err();
            assert!(matches!(err, CoreError::Store(_)), "{err:?}");
        }
        assert!(!path.exists());
    }

    #[tokio::test]
    async fn cap_above_sqlite_limit_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.db");
        for builder in [
            SqliteStore::builder(&path).max_state_bytes(1_000_000_001),
            SqliteStore::builder(&path).max_blob_bytes(1_000_000_001),
        ] {
            let err = builder.open().await.unwrap_err();
            assert!(
                matches!(&err, CoreError::Store(m) if m.contains("1000000000")),
                "{err:?}"
            );
        }
        assert!(SqliteStore::builder(&path)
            .max_blob_bytes(1_000_000_000)
            .open()
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn panic_in_a_store_call_reaches_the_caller() {
        let (_dir, store) = temp_store(|b| b).await;
        let task = tokio::spawn(async move {
            store
                .call(|_conn| -> Result<(), CoreError> { panic!("boom in call") })
                .await
        });
        let err = task.await.unwrap_err();
        assert!(err.is_panic());
        let payload = err.into_panic();
        assert_eq!(payload.downcast_ref::<&str>(), Some(&"boom in call"));
    }

    #[tokio::test]
    async fn panic_inside_a_transaction_rolls_it_back() {
        let (_dir, store) = temp_store(|b| b).await;
        let clone = store.clone();
        let task = tokio::spawn(async move {
            clone
                .call(|conn| -> Result<(), CoreError> {
                    let tx = conn
                        .transaction_with_behavior(TransactionBehavior::Immediate)
                        .unwrap();
                    tx.execute(
                        "INSERT INTO blob_entry (path, bytes) VALUES ('p', x'01')",
                        [],
                    )
                    .unwrap();
                    panic!("boom in transaction")
                })
                .await
        });
        assert!(task.await.unwrap_err().is_panic());
        let rows: i64 = store
            .call(|conn| {
                conn.query_row("SELECT COUNT(*) FROM blob_entry", [], |r| r.get(0))
                    .map_err(db)
            })
            .await
            .unwrap();
        assert_eq!(rows, 0);
    }

    #[tokio::test]
    async fn store_stays_usable_after_a_panicked_call() {
        let (_dir, store) = temp_store(|b| b).await;
        let clone = store.clone();
        let task = tokio::spawn(async move {
            clone
                .call(|_conn| -> Result<(), CoreError> { panic!("boom") })
                .await
        });
        assert!(task.await.unwrap_err().is_panic());
        assert!(store.inner.conn.is_poisoned());
        let written = store
            .call(|conn| {
                conn.execute(
                    "INSERT INTO blob_entry (path, bytes) VALUES ('q', x'02')",
                    [],
                )
                .map_err(db)
            })
            .await
            .unwrap();
        assert_eq!(written, 1);
        assert_eq!(pragma(&store, "user_version").await, i64::from(SCHEMA_VERSION));
    }
}
