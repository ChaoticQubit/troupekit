use crate::{db, SqliteStore};
use async_trait::async_trait;
use rusqlite::{params, params_from_iter, Connection, OptionalExtension, TransactionBehavior};
use std::collections::{BTreeMap, HashSet};
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};
use troupekit_core::error::CoreError;
use troupekit_core::store::{Filter, KeywordStore, Scored, VectorRecord, VectorStore};
use troupekit_index::{new_index, HnswParams, IndexError, IndexKind, PageChange, PageKey, Quantization, VectorIndex, MAX_DIMS, MAX_K};

const NAME_BYTES_MAX: usize = 64;
const ID_BYTES_MAX: usize = 256;
const ERROR_BYTES_MAX: usize = 1024;
const SNAPSHOT_TRIES: usize = 4;
const HASH_BYTES_MAX: usize = 256;
const MAX_TEXT_BYTES_DEFAULT: usize = 1024 * 1024;
const MAX_META_PAIRS_DEFAULT: usize = 64;
const MAX_META_KEY_BYTES_DEFAULT: usize = 256;
const MAX_META_VALUE_BYTES_DEFAULT: usize = 4096;
const MAX_BATCH_DEFAULT: usize = 10_000;
const MAX_QUERY_TERMS_DEFAULT: usize = 32;
const MAX_QUERY_CHARS_DEFAULT: usize = 2_000;
const KIND_BRUTE: &str = "brute_force";
const KIND_HNSW: &str = "hnsw";
const SCOPE_KEY: &str = "scope";
const SOURCE_KEY: &str = "source";

/// Settings of one named vector index. `dims` is required; the rest has
/// safe defaults: brute force, f32 in memory, 1 MiB of text per chunk,
/// 64 meta pairs of at most 256 and 4 096 bytes, 10 000 records per call,
/// keyword queries of 32 terms and 2 000 chars. Built with `new(dims)`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct VectorSettings {
    dims: usize,
    kind: IndexKind,
    quantization: Quantization,
    max_text_bytes: usize,
    max_meta_pairs: usize,
    max_meta_key_bytes: usize,
    max_meta_value_bytes: usize,
    max_batch: usize,
    max_query_terms: usize,
    max_query_chars: usize,
}

impl VectorSettings {
    /// Starts settings for vectors of `dims` dimensions.
    /// Input: `dims`, checked in `1..=65 536` when the index opens.
    /// Output: settings with every other value at its default.
    pub fn new(dims: usize) -> Self {
        Self {
            dims,
            kind: IndexKind::default(),
            quantization: Quantization::default(),
            max_text_bytes: MAX_TEXT_BYTES_DEFAULT,
            max_meta_pairs: MAX_META_PAIRS_DEFAULT,
            max_meta_key_bytes: MAX_META_KEY_BYTES_DEFAULT,
            max_meta_value_bytes: MAX_META_VALUE_BYTES_DEFAULT,
            max_batch: MAX_BATCH_DEFAULT,
            max_query_terms: MAX_QUERY_TERMS_DEFAULT,
            max_query_chars: MAX_QUERY_CHARS_DEFAULT,
        }
    }

    /// Sets the index kind. A kind or HNSW params other than the stored
    /// ones rebuild the index and rewrite its pages when it opens.
    /// Default `IndexKind::BruteForce`.
    pub fn with_kind(mut self, kind: IndexKind) -> Self {
        self.kind = kind;
        self
    }

    /// Sets the in-memory vector format. The file always keeps f32, so a
    /// change only reloads memory. The last `vectors` call for a name sets
    /// it for every handle of that name. Default `Quantization::F32`.
    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }

    /// Sets the largest chunk text in bytes.
    /// A larger record fails the whole call with `CoreError::Store`.
    /// Default 1 MiB.
    pub fn with_max_text_bytes(mut self, bytes: usize) -> Self {
        self.max_text_bytes = bytes;
        self
    }

    /// Sets the most meta pairs on one record and in one search filter.
    /// More fails the call with `CoreError::Store`.
    /// Default 64.
    pub fn with_max_meta_pairs(mut self, pairs: usize) -> Self {
        self.max_meta_pairs = pairs;
        self
    }

    /// Sets the longest meta key in bytes.
    /// A longer key fails the call with `CoreError::Store`.
    /// Default 256.
    pub fn with_max_meta_key_bytes(mut self, bytes: usize) -> Self {
        self.max_meta_key_bytes = bytes;
        self
    }

    /// Sets the longest meta value in bytes.
    /// A longer value fails the call with `CoreError::Store`.
    /// Default 4 096.
    pub fn with_max_meta_value_bytes(mut self, bytes: usize) -> Self {
        self.max_meta_value_bytes = bytes;
        self
    }

    /// Sets the most records one `upsert` or `replace_source` writes,
    /// which bounds one transaction.
    /// Default 10 000.
    pub fn with_max_batch(mut self, records: usize) -> Self {
        self.max_batch = records;
        self
    }

    /// Sets the most terms in one keyword query, which bounds the FTS5
    /// work per call. More terms fail with `CoreError::Store`.
    /// Default 32.
    pub fn with_max_query_terms(mut self, terms: usize) -> Self {
        self.max_query_terms = terms;
        self
    }

    /// Sets the longest keyword query in chars.
    /// A longer query fails with `CoreError::Store`.
    /// Default 2 000.
    pub fn with_max_query_chars(mut self, chars: usize) -> Self {
        self.max_query_chars = chars;
        self
    }

    /// Returns the dimension of every vector and query.
    /// It is fixed per index name once the index exists.
    /// Required, no default.
    pub fn dims(&self) -> usize {
        self.dims
    }

    /// Returns the index kind.
    /// Brute force is exact; HNSW is approximate and faster.
    /// Default `IndexKind::BruteForce`.
    pub fn kind(&self) -> &IndexKind {
        &self.kind
    }

    /// Returns the in-memory vector format.
    /// The file always keeps f32.
    /// Default `Quantization::F32`.
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Returns the largest chunk text in bytes.
    /// Checked on every record before anything is written.
    /// Default 1 MiB.
    pub fn max_text_bytes(&self) -> usize {
        self.max_text_bytes
    }

    /// Returns the most meta pairs per record and per filter.
    /// Checked before anything is written or searched.
    /// Default 64.
    pub fn max_meta_pairs(&self) -> usize {
        self.max_meta_pairs
    }

    /// Returns the longest meta key in bytes.
    /// Checked on every record before anything is written.
    /// Default 256.
    pub fn max_meta_key_bytes(&self) -> usize {
        self.max_meta_key_bytes
    }

    /// Returns the longest meta value in bytes.
    /// Checked on every record before anything is written.
    /// Default 4 096.
    pub fn max_meta_value_bytes(&self) -> usize {
        self.max_meta_value_bytes
    }

    /// Returns the most records per write call.
    /// It bounds one transaction.
    /// Default 10 000.
    pub fn max_batch(&self) -> usize {
        self.max_batch
    }

    /// Returns the most terms in one keyword query.
    /// Checked before the query reaches FTS5.
    /// Default 32.
    pub fn max_query_terms(&self) -> usize {
        self.max_query_terms
    }

    /// Returns the longest keyword query in chars.
    /// Checked before the query reaches FTS5.
    /// Default 2 000.
    pub fn max_query_chars(&self) -> usize {
        self.max_query_chars
    }
}

/// How often the index behind a handle was loaded from the file and how
/// often its pages were refused and rebuilt from vectors, counted since
/// the store opened. Shared by every handle of the same name.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpenReport {
    loads: usize,
    rebuilds: usize,
}

impl OpenReport {
    /// Returns how many times the index was loaded from the file.
    /// The first open counts as one; each reload after another
    /// connection's write or a failed call adds one.
    pub fn loads(&self) -> usize {
        self.loads
    }

    /// Returns how many times stored pages were refused as corrupt and
    /// the index was rebuilt from the stored vectors.
    /// Zero on a healthy file.
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }
}

/// The in-memory index of one name and what the file looked like when
/// it was loaded. `stale` forces a reload before the next call, so memory
/// never runs ahead of the file after a failed write.
pub(crate) struct Cached {
    index: Box<dyn VectorIndex>,
    settings: VectorSettings,
    generation: i64,
    stale: bool,
    report: OpenReport,
}

/// One named index shared by every handle of a store: its name, its
/// fixed dimension and the cached index behind a lock. Lock order is
/// always connection, then this lock.
pub(crate) struct IndexState {
    name: String,
    dims: usize,
    cached: RwLock<Cached>,
}

/// A named vector index in the store file, with text, metadata and a
/// keyword index kept in the same transactions. Cheap to clone; clones
/// and other handles of the same name share one in-memory index. Every
/// call checks the dimension and the bounds of its `VectorSettings`.
#[derive(Clone)]
pub struct SqliteVectors {
    store: SqliteStore,
    state: Arc<IndexState>,
}

impl std::fmt::Debug for SqliteVectors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteVectors")
            .field("name", &self.state.name)
            .field("dims", &self.state.dims)
            .finish_non_exhaustive()
    }
}

impl SqliteStore {
    /// Opens the vector index `name`, creating it on first use.
    /// Input: a name of 1 to 64 chars in `[a-z0-9_]` and its settings.
    /// Output: a handle, `CoreError::DimensionMismatch` when the stored
    /// index has other dims, or `CoreError::Store` for bad settings.
    pub async fn vectors(&self, name: &str, settings: VectorSettings) -> Result<SqliteVectors, CoreError> {
        check_name(name)?;
        check_settings(&settings)?;
        let wanted_kind = kind_name(settings.kind())?;
        let wanted_params = params_json(settings.kind())?;
        let name = name.to_string();
        let store = self.clone();
        let now_ms = self.inner.clock.now_ms();
        let state = self
            .call(move |conn| {
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db)?;
                let row: Option<(i64, String, String)> = tx
                    .query_row(
                        "SELECT dims, kind, params_json FROM vector_index WHERE name = ?1",
                        [&name],
                        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                    )
                    .optional()
                    .map_err(db)?;
                let kind_changed = match row {
                    None => {
                        tx.execute(
                            "INSERT INTO vector_index (name, dims, kind, params_json, generation, created_at_ms) \
                             VALUES (?1, ?2, ?3, ?4, 0, ?5)",
                            params![name, settings.dims as i64, wanted_kind, wanted_params, now_ms],
                        )
                        .map_err(db)?;
                        false
                    }
                    Some((dims, kind, params)) => {
                        if usize::try_from(dims).ok() != Some(settings.dims) {
                            return Err(CoreError::DimensionMismatch {
                                expected: usize::try_from(dims).unwrap_or(0),
                                actual: settings.dims,
                            });
                        }
                        kind != wanted_kind || params != wanted_params
                    }
                };
                let state = {
                    let mut map = match store.inner.vectors.lock() {
                        Ok(map) => map,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                    match map.get(&name) {
                        Some(state) => Arc::clone(state),
                        None => {
                            let placeholder = new_index(&IndexKind::BruteForce, settings.dims, Quantization::F32).map_err(index_err)?;
                            let state = Arc::new(IndexState {
                                name: name.clone(),
                                dims: settings.dims,
                                cached: RwLock::new(Cached {
                                    index: placeholder,
                                    settings: settings.clone(),
                                    generation: -1,
                                    stale: true,
                                    report: OpenReport::default(),
                                }),
                            });
                            map.insert(name.clone(), Arc::clone(&state));
                            state
                        }
                    }
                };
                assert_eq!(state.dims, settings.dims, "cached dims differ from the stored row");
                let mut cached = write_cached(&state);
                if cached.settings != settings {
                    cached.settings = settings;
                    cached.stale = true;
                }
                let rebuilt_at = if kind_changed {
                    cached.stale = true;
                    Some(rebuild_kind(&tx, &state, &mut cached)?)
                } else {
                    reload_if_needed(&tx, &state, &mut cached)?;
                    None
                };
                if let Err(e) = tx.commit() {
                    cached.stale = true;
                    return Err(db(e));
                }
                if let Some(generation) = rebuilt_at {
                    cached.generation = generation;
                    cached.stale = false;
                }
                assert!(!cached.stale, "an opened index is fresh");
                drop(cached);
                Ok(state)
            })
            .await?;
        Ok(SqliteVectors { store: self.clone(), state })
    }
}

impl SqliteVectors {
    /// Returns how often this index was loaded and rebuilt since the
    /// store opened. Input: none. Output: the counts, shared by every
    /// handle of the same name.
    pub fn open_report(&self) -> OpenReport {
        read_cached(&self.state).report
    }

    /// Returns the live ids in the index after a freshness check, so a
    /// write from another connection is counted. Input: none. Output:
    /// the count, or `CoreError::Store`.
    pub async fn len(&self) -> Result<usize, CoreError> {
        let state = Arc::clone(&self.state);
        self.store
            .call(move |conn| {
                refresh(conn, &state)?;
                Ok(read_cached(&state).index.len())
            })
            .await
    }

    /// Replaces every chunk of one source in one transaction: deletes
    /// the old rows, keyword rows and index entries, writes `records` and
    /// marks the source `done` with `content_hash`. Every record's meta
    /// must carry `scope` and `source` equal to the arguments. Output: the
    /// number of records written, or an error that writes nothing.
    pub async fn replace_source(&self, scope: &str, source: &str, records: Vec<VectorRecord>, content_hash: &str) -> Result<usize, CoreError> {
        let settings = read_cached(&self.state).settings.clone();
        check_source(scope, source, &settings)?;
        if content_hash.is_empty() || content_hash.len() > HASH_BYTES_MAX {
            return Err(CoreError::Store(format!("content_hash must be 1 to {HASH_BYTES_MAX} bytes")));
        }
        check_records(&records, &settings, self.state.dims)?;
        for (position, record) in records.iter().enumerate() {
            let scope_ok = record.meta.get(SCOPE_KEY).map(String::as_str) == Some(scope);
            let source_ok = record.meta.get(SOURCE_KEY).map(String::as_str) == Some(source);
            if !scope_ok || !source_ok {
                return Err(CoreError::Store(format!(
                    "record {position} must carry meta {SCOPE_KEY} and {SOURCE_KEY} equal to the replaced source"
                )));
            }
        }
        let state = Arc::clone(&self.state);
        let scope = scope.to_string();
        let source = source.to_string();
        let content_hash = content_hash.to_string();
        let now_ms = self.store.inner.clock.now_ms();
        self.store
            .call(move |conn| {
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db)?;
                let mut cached = write_cached(&state);
                reload_if_needed(&tx, &state, &mut cached)?;
                cached.stale = true;
                delete_source_rows(&tx, &state, &mut cached, &scope, &source)?;
                {
                    let mut owned = tx.prepare_cached("SELECT 1 FROM chunk WHERE index_name = ?1 AND id = ?2").map_err(db)?;
                    for record in &records {
                        if owned.exists(params![state.name, record.id]).map_err(db)? {
                            return Err(CoreError::Store(format!("record id {} belongs to another source", cut_utf8(&record.id, 64))));
                        }
                    }
                }
                let keys = write_rows(&tx, &state.name, &records)?;
                for (key, record) in keys.iter().zip(&records) {
                    cached.index.insert(*key, &record.vector).map_err(index_err)?;
                }
                let changes = cached.index.take_changes();
                write_pages(&tx, &state.name, changes)?;
                tx.execute(
                    "INSERT INTO ingest_source (index_name, scope, source, content_hash, chunks, status, error, at_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, 'done', NULL, ?6) \
                     ON CONFLICT (index_name, scope, source) DO UPDATE SET \
                     content_hash = excluded.content_hash, chunks = excluded.chunks, status = 'done', error = NULL, at_ms = excluded.at_ms",
                    params![state.name, scope, source, content_hash, records.len() as i64, now_ms],
                )
                .map_err(db)?;
                let generation = bump_generation(&tx, &state.name)?;
                tx.commit().map_err(db)?;
                cached.generation = generation;
                cached.stale = false;
                Ok(records.len())
            })
            .await
    }

    /// Lists every source of `scope` that has an ingest status row, done
    /// or failed, sorted. Input: the scope. Output: the sources, so a
    /// folder ingest can find the ones whose file is gone.
    pub async fn sources(&self, scope: &str) -> Result<Vec<String>, CoreError> {
        let state = Arc::clone(&self.state);
        let scope = scope.to_string();
        self.store
            .call(move |conn| {
                let mut stmt = conn
                    .prepare("SELECT source FROM ingest_source WHERE index_name = ?1 AND scope = ?2 ORDER BY source")
                    .map_err(db)?;
                let rows = stmt.query_map(params![state.name, scope], |row| row.get(0)).map_err(db)?;
                rows.collect::<Result<Vec<String>, _>>().map_err(db)
            })
            .await
    }

    /// Removes one source in one transaction: its chunk, meta and keyword
    /// rows, its index entries and pages, and its ingest status row.
    /// Input: the scope and source. Output: the chunks removed, 0 when
    /// the source had none, or an error that removes nothing.
    pub async fn remove_source(&self, scope: &str, source: &str) -> Result<usize, CoreError> {
        let settings = read_cached(&self.state).settings.clone();
        check_source(scope, source, &settings)?;
        let state = Arc::clone(&self.state);
        let scope = scope.to_string();
        let source = source.to_string();
        self.store
            .call(move |conn| {
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db)?;
                let mut cached = write_cached(&state);
                reload_if_needed(&tx, &state, &mut cached)?;
                cached.stale = true;
                let removed = delete_source_rows(&tx, &state, &mut cached, &scope, &source)?;
                let changes = cached.index.take_changes();
                write_pages(&tx, &state.name, changes)?;
                tx.execute(
                    "DELETE FROM ingest_source WHERE index_name = ?1 AND scope = ?2 AND source = ?3",
                    params![state.name, scope, source],
                )
                .map_err(db)?;
                let generation = bump_generation(&tx, &state.name)?;
                tx.commit().map_err(db)?;
                cached.generation = generation;
                cached.stale = false;
                Ok(removed)
            })
            .await
    }

    /// Returns the content hash of a source last replaced with success.
    /// Input: the scope and source. Output: the hash, or `None` when the
    /// source was never ingested or its last attempt failed.
    pub async fn source_hash(&self, scope: &str, source: &str) -> Result<Option<String>, CoreError> {
        let name = self.state.name.clone();
        let scope = scope.to_string();
        let source = source.to_string();
        self.store
            .call(move |conn| {
                conn.query_row(
                    "SELECT content_hash FROM ingest_source \
                     WHERE index_name = ?1 AND scope = ?2 AND source = ?3 AND status = 'done'",
                    params![name, scope, source],
                    |row| row.get(0),
                )
                .optional()
                .map_err(db)
            })
            .await
    }

    /// Records that ingesting a source failed, with the error cut to
    /// 1 024 bytes. The source's chunks stay as they were, and
    /// `source_hash` returns `None` so the next run retries it.
    /// Output: `Ok`, or `CoreError::Store`.
    pub async fn record_failed_source(&self, scope: &str, source: &str, error: &str) -> Result<(), CoreError> {
        let settings = read_cached(&self.state).settings.clone();
        check_source(scope, source, &settings)?;
        let error = cut_utf8(error, ERROR_BYTES_MAX);
        let error = if error.is_empty() { String::from("unknown error") } else { error.to_string() };
        assert!(error.len() <= ERROR_BYTES_MAX, "error text over its cap");
        let name = self.state.name.clone();
        let scope = scope.to_string();
        let source = source.to_string();
        let now_ms = self.store.inner.clock.now_ms();
        self.store
            .call(move |conn| {
                let chunks: i64 = conn
                    .query_row(
                        "SELECT chunks FROM ingest_source WHERE index_name = ?1 AND scope = ?2 AND source = ?3",
                        params![name, scope, source],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(db)?
                    .unwrap_or(0);
                let written = conn
                    .execute(
                        "INSERT INTO ingest_source (index_name, scope, source, content_hash, chunks, status, error, at_ms) \
                         VALUES (?1, ?2, ?3, '', ?4, 'failed', ?5, ?6) \
                         ON CONFLICT (index_name, scope, source) DO UPDATE SET \
                         status = 'failed', error = excluded.error, at_ms = excluded.at_ms",
                        params![name, scope, source, chunks, error, now_ms],
                    )
                    .map_err(db)?;
                assert_eq!(written, 1);
                Ok(())
            })
            .await
    }
}

#[async_trait]
impl VectorStore for SqliteVectors {
    fn dims(&self) -> usize {
        self.state.dims
    }

    async fn upsert(&self, records: Vec<VectorRecord>) -> Result<(), CoreError> {
        let settings = read_cached(&self.state).settings.clone();
        check_records(&records, &settings, self.state.dims)?;
        if records.is_empty() {
            return Ok(());
        }
        let state = Arc::clone(&self.state);
        self.store
            .call(move |conn| {
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db)?;
                let mut cached = write_cached(&state);
                reload_if_needed(&tx, &state, &mut cached)?;
                cached.stale = true;
                let keys = write_rows(&tx, &state.name, &records)?;
                for (key, record) in keys.iter().zip(&records) {
                    cached.index.insert(*key, &record.vector).map_err(index_err)?;
                }
                let changes = cached.index.take_changes();
                write_pages(&tx, &state.name, changes)?;
                let generation = bump_generation(&tx, &state.name)?;
                tx.commit().map_err(db)?;
                cached.generation = generation;
                cached.stale = false;
                Ok(())
            })
            .await
    }

    async fn search(&self, query: &[f32], k: usize, filter: &Filter) -> Result<Vec<Scored>, CoreError> {
        if query.len() != self.state.dims {
            return Err(CoreError::DimensionMismatch { expected: self.state.dims, actual: query.len() });
        }
        let settings = read_cached(&self.state).settings.clone();
        if filter.0.len() > settings.max_meta_pairs {
            return Err(CoreError::Store(format!(
                "filter has {} pairs, above max_meta_pairs {}",
                filter.0.len(),
                settings.max_meta_pairs
            )));
        }
        let state = Arc::clone(&self.state);
        let query = query.to_vec();
        let pairs: Vec<(String, String)> = filter.0.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.store
            .call(move |conn| {
                for _ in 0..SNAPSHOT_TRIES {
                    let tx = conn.transaction().map_err(db)?;
                    let generation = read_generation(&tx, &state.name)?;
                    let cached = read_cached(&state);
                    if state.cached.is_poisoned() || cached.stale || cached.generation != generation {
                        drop(cached);
                        drop(tx);
                        refresh(conn, &state)?;
                        continue;
                    }
                    return search_snapshot(&tx, &state, &cached, &query, k, &pairs);
                }
                Err(CoreError::Store(format!("index changed under {SNAPSHOT_TRIES} searches in a row; try again")))
            })
            .await
    }
}

/// Deletes every chunk of one source, with its meta and keyword rows,
/// and removes the chunks from the cached index, inside the caller's
/// write transaction. Input: the transaction, state, cache, scope and
/// source. Output: the number of chunks deleted.
fn delete_source_rows(tx: &Connection, state: &IndexState, cached: &mut Cached, scope: &str, source: &str) -> Result<usize, CoreError> {
    assert!(cached.stale, "source rows change only under a stale cache");
    let old: Vec<(i64, String)> = {
        let mut stmt = tx
            .prepare(
                "SELECT c.key, c.text FROM chunk c \
                 JOIN chunk_meta s ON s.chunk_key = c.key AND s.key = ?2 AND s.value = ?3 \
                 JOIN chunk_meta o ON o.chunk_key = c.key AND o.key = ?4 AND o.value = ?5 \
                 WHERE c.index_name = ?1 ORDER BY c.key",
            )
            .map_err(db)?;
        let rows = stmt
            .query_map(params![state.name, SCOPE_KEY, scope, SOURCE_KEY, source], |row| Ok((row.get(0)?, row.get(1)?)))
            .map_err(db)?;
        rows.collect::<Result<_, _>>().map_err(db)?
    };
    for (key, text) in &old {
        delete_row(tx, *key, text)?;
        cached.index.remove(to_id(*key)?);
    }
    Ok(old.len())
}

/// Searches the cached index and reads the hits inside one read
/// transaction whose generation matches the cache, so a hit's score and
/// its text come from the same committed version. Input: the open
/// transaction, the state, its matching cache, query, k and filter pairs.
fn search_snapshot(tx: &rusqlite::Transaction<'_>, state: &IndexState, cached: &Cached, query: &[f32], k: usize, pairs: &[(String, String)]) -> Result<Vec<Scored>, CoreError> {
    let allowed = if pairs.is_empty() { None } else { Some(allowed_keys(tx, &state.name, pairs)?) };
    if allowed.as_ref().is_some_and(HashSet::is_empty) {
        return Ok(Vec::new());
    }
    let neighbors = {
        let allow_fn = |id: u64| allowed.as_ref().is_none_or(|set| set.contains(&id));
        let allow: Option<&dyn Fn(u64) -> bool> = if allowed.is_some() { Some(&allow_fn) } else { None };
        cached.index.search(query, k, allow).map_err(index_err)?
    };
    let mut stmt = tx
        .prepare_cached("SELECT id, text, meta_json FROM chunk WHERE key = ?1 AND index_name = ?2")
        .map_err(db)?;
    let mut hits = Vec::with_capacity(neighbors.len());
    for neighbor in &neighbors {
        let key = i64::try_from(neighbor.id).map_err(|_| CoreError::Store("index id out of range".into()))?;
        let row: Option<(String, String, String)> = stmt
            .query_row(params![key, state.name], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .optional()
            .map_err(db)?;
        let Some((id, text, meta_json)) = row else {
            continue;
        };
        let meta: BTreeMap<String, String> =
            serde_json::from_str(&meta_json).map_err(|e| CoreError::Store(format!("corrupt store: chunk meta: {e}")))?;
        hits.push(Scored { id, score: neighbor.score, text, meta });
    }
    assert!(hits.len() <= k, "search returned more than k hits");
    Ok(hits)
}

#[async_trait]
impl KeywordStore for SqliteVectors {
    async fn search_text(&self, query: &str, k: usize, filter: &Filter) -> Result<Vec<Scored>, CoreError> {
        let settings = read_cached(&self.state).settings.clone();
        let chars = query.chars().count();
        if chars > settings.max_query_chars {
            return Err(CoreError::Store(format!("query is {chars} chars, above max_query_chars {}", settings.max_query_chars)));
        }
        if k > MAX_K {
            return Err(CoreError::Store(format!("k must be at most {MAX_K}, got {k}")));
        }
        if filter.0.len() > settings.max_meta_pairs {
            return Err(CoreError::Store(format!(
                "filter has {} pairs, above max_meta_pairs {}",
                filter.0.len(),
                settings.max_meta_pairs
            )));
        }
        let (terms, fts) = fts_query(query);
        if terms.len() > settings.max_query_terms {
            return Err(CoreError::Store(format!(
                "query has {} terms, above max_query_terms {}",
                terms.len(),
                settings.max_query_terms
            )));
        }
        if terms.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        let name = self.state.name.clone();
        let pairs: Vec<(String, String)> = filter.0.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        self.store
            .call(move |conn| {
                let limit = 3 + 2 * pairs.len();
                let sql = format!(
                    "SELECT c.id, c.text, c.meta_json, -bm25(chunk_fts) AS score \
                     FROM chunk_fts JOIN chunk c ON c.key = chunk_fts.rowid \
                     WHERE chunk_fts MATCH ?1 AND c.index_name = ?2{} \
                     ORDER BY score DESC, c.id ASC LIMIT ?{limit}",
                    meta_clauses("c.key", pairs.len(), 3)
                );
                let mut values: Vec<rusqlite::types::Value> = vec![fts.into(), name.into()];
                for (key, value) in pairs {
                    values.push(key.into());
                    values.push(value.into());
                }
                values.push((k as i64).into());
                let mut stmt = conn.prepare(&sql).map_err(db)?;
                let rows = stmt
                    .query_map(params_from_iter(values), |row| {
                        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, f64>(3)?))
                    })
                    .map_err(db)?;
                let mut hits = Vec::new();
                for row in rows {
                    let (id, text, meta_json, score) = row.map_err(db)?;
                    let meta: BTreeMap<String, String> =
                        serde_json::from_str(&meta_json).map_err(|e| CoreError::Store(format!("corrupt store: chunk meta: {e}")))?;
                    hits.push(Scored { id, score: score as f32, text, meta });
                }
                assert!(hits.len() <= k, "keyword search returned more than k hits");
                assert!(hits.windows(2).all(|w| w[0].score >= w[1].score), "keyword hits sorted by score");
                Ok(hits)
            })
            .await
    }
}

/// Takes the write lock of a cached index. A lock poisoned by a panic
/// mid-update is taken anyway and marked stale, so the next step reloads
/// it from the file. Output: the guard.
fn write_cached(state: &IndexState) -> RwLockWriteGuard<'_, Cached> {
    match state.cached.write() {
        Ok(guard) => guard,
        Err(poisoned) => {
            let mut guard = poisoned.into_inner();
            guard.stale = true;
            state.cached.clear_poison();
            guard
        }
    }
}

/// Takes the read lock of a cached index, poisoned or not. Callers
/// that search first run `refresh`, which reloads a poisoned index.
/// Output: the guard.
fn read_cached(state: &IndexState) -> RwLockReadGuard<'_, Cached> {
    match state.cached.read() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Reloads the index when another connection wrote since it was loaded,
/// or when it is stale or poisoned. Input: the connection, outside any
/// transaction. Output: `Ok` once memory matches the file.
fn refresh(conn: &mut Connection, state: &IndexState) -> Result<(), CoreError> {
    let generation = read_generation(conn, &state.name)?;
    if !state.cached.is_poisoned() {
        let cached = read_cached(state);
        if cached.generation == generation && !cached.stale {
            return Ok(());
        }
    }
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate).map_err(db)?;
    let mut cached = write_cached(state);
    reload_if_needed(&tx, state, &mut cached)?;
    tx.commit().map_err(db)?;
    Ok(())
}

/// Loads the index from the file when its generation differs from the
/// cached one or the cache is stale. Pages refused as corrupt are
/// rebuilt from vectors and rewritten. Input: a connection inside a
/// write transaction. Output: `Ok` with `cached` fresh.
fn reload_if_needed(conn: &Connection, state: &IndexState, cached: &mut Cached) -> Result<(), CoreError> {
    let generation = read_generation(conn, &state.name)?;
    if cached.generation == generation && !cached.stale {
        return Ok(());
    }
    let kind = stored_kind(conn, &state.name)?;
    let mut index = new_index(&kind, state.dims, cached.settings.quantization).map_err(index_err)?;
    let vectors = load_vectors(conn, &state.name, state.dims)?;
    let pages = load_pages(conn, &state.name)?;
    let rebuilt = match index.restore(&mut vectors.into_iter(), &mut pages.into_iter()) {
        Ok(()) => {
            index.take_changes();
            false
        }
        Err(IndexError::CorruptPage(_)) => {
            index = build_from_file(conn, &kind, state.dims, cached.settings.quantization, &state.name)?;
            true
        }
        Err(e) => return Err(index_err(e)),
    };
    if rebuilt {
        conn.execute("DELETE FROM index_page WHERE index_name = ?1", [&state.name]).map_err(db)?;
        let changes = index.take_changes();
        write_pages(conn, &state.name, changes)?;
        cached.report.rebuilds += 1;
    }
    cached.index = index;
    cached.generation = generation;
    cached.stale = false;
    cached.report.loads += 1;
    Ok(())
}

/// Rebuilds the index under the kind in `cached.settings`, rewrites all
/// pages and stores the new kind, all in the caller's transaction. The
/// cache stays stale: the caller publishes the generation after commit.
/// Input: a connection inside a write transaction. Output: the new generation.
fn rebuild_kind(conn: &Connection, state: &IndexState, cached: &mut Cached) -> Result<i64, CoreError> {
    let kind = cached.settings.kind.clone();
    let index = build_from_file(conn, &kind, state.dims, cached.settings.quantization, &state.name)?;
    conn.execute(
        "UPDATE vector_index SET kind = ?2, params_json = ?3 WHERE name = ?1",
        params![state.name, kind_name(&kind)?, params_json(&kind)?],
    )
    .map_err(db)?;
    cached.index = index;
    conn.execute("DELETE FROM index_page WHERE index_name = ?1", [&state.name]).map_err(db)?;
    let changes = cached.index.take_changes();
    write_pages(conn, &state.name, changes)?;
    let generation = bump_generation(conn, &state.name)?;
    assert!(cached.stale, "a rebuilt index stays stale until its commit");
    cached.report.loads += 1;
    Ok(generation)
}

/// Builds a fresh index by inserting every stored vector of `name`.
/// Input: the connection, kind, dims, format and name. Output: the index
/// with its page changes not yet taken.
fn build_from_file(conn: &Connection, kind: &IndexKind, dims: usize, quantization: Quantization, name: &str) -> Result<Box<dyn VectorIndex>, CoreError> {
    let mut index = new_index(kind, dims, quantization).map_err(index_err)?;
    for (id, vector) in load_vectors(conn, name, dims)? {
        index.insert(id, &vector).map_err(index_err)?;
    }
    Ok(index)
}

/// Reads every stored vector of `name` in key order, checking each blob
/// is `4 * dims` bytes. Input: the connection, name and dims. Output:
/// pairs of index id and vector, or `CoreError::Store` on a bad blob.
fn load_vectors(conn: &Connection, name: &str, dims: usize) -> Result<Vec<(u64, Vec<f32>)>, CoreError> {
    let mut stmt = conn.prepare("SELECT key, vector FROM chunk WHERE index_name = ?1 ORDER BY key").map_err(db)?;
    let rows = stmt.query_map([name], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))).map_err(db)?;
    let mut out = Vec::new();
    for row in rows {
        let (key, bytes) = row.map_err(db)?;
        out.push((to_id(key)?, decode(&bytes, dims)?));
    }
    Ok(out)
}

/// Reads every stored page of `name` in page order. Page 0 is the
/// header and page `n >= 1` is node `n`. Input: the connection and the
/// name. Output: the pages, or `CoreError::Store`.
fn load_pages(conn: &Connection, name: &str) -> Result<Vec<(PageKey, Vec<u8>)>, CoreError> {
    let mut stmt = conn.prepare("SELECT page, bytes FROM index_page WHERE index_name = ?1 ORDER BY page").map_err(db)?;
    let rows = stmt.query_map([name], |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))).map_err(db)?;
    let mut out = Vec::new();
    for row in rows {
        let (page, bytes) = row.map_err(db)?;
        let key = if page == 0 { PageKey::Header } else { PageKey::Node(to_id(page)?) };
        out.push((key, bytes));
    }
    Ok(out)
}

/// Applies page changes to `index_page`. Input: a connection inside a
/// write transaction, the name and the changes. Output: `Ok`, or
/// `CoreError::Store` for a page key the store cannot map.
fn write_pages(conn: &Connection, name: &str, changes: Vec<PageChange>) -> Result<(), CoreError> {
    for change in changes {
        match change {
            PageChange::Put(key, bytes) => {
                conn.prepare_cached(
                    "INSERT INTO index_page (index_name, page, bytes) VALUES (?1, ?2, ?3) \
                     ON CONFLICT (index_name, page) DO UPDATE SET bytes = excluded.bytes",
                )
                .map_err(db)?
                .execute(params![name, page_number(key)?, bytes])
                .map_err(db)?;
            }
            PageChange::Delete(key) => {
                conn.prepare_cached("DELETE FROM index_page WHERE index_name = ?1 AND page = ?2")
                    .map_err(db)?
                    .execute(params![name, page_number(key)?])
                    .map_err(db)?;
            }
            _ => return Err(CoreError::Store("unknown index page change".into())),
        }
    }
    Ok(())
}

/// Maps a page key to its `index_page.page` number: 0 for the header,
/// `n` for node `n`. Input: the key. Output: the number, or
/// `CoreError::Store` for node 0, a number past `i64` or an unknown key.
fn page_number(key: PageKey) -> Result<i64, CoreError> {
    match key {
        PageKey::Header => Ok(0),
        PageKey::Node(n) if n >= 1 => i64::try_from(n).map_err(|_| CoreError::Store("page number out of range".into())),
        _ => Err(CoreError::Store("unknown index page key".into())),
    }
}

/// Writes or updates the rows of `records`: the chunk, its meta pairs
/// and its keyword row. An existing id keeps its key, so the index entry
/// is replaced in place. Input: a connection inside a write transaction.
/// Output: the index id of each record, in order.
fn write_rows(conn: &Connection, name: &str, records: &[VectorRecord]) -> Result<Vec<u64>, CoreError> {
    let mut keys = Vec::with_capacity(records.len());
    for record in records {
        let meta_json = serde_json::to_string(&record.meta).map_err(|e| CoreError::Store(format!("chunk meta: {e}")))?;
        let vector = encode(&record.vector);
        let old: Option<(i64, String)> = conn
            .prepare_cached("SELECT key, text FROM chunk WHERE index_name = ?1 AND id = ?2")
            .map_err(db)?
            .query_row(params![name, record.id], |row| Ok((row.get(0)?, row.get(1)?)))
            .optional()
            .map_err(db)?;
        let key = match old {
            Some((key, old_text)) => {
                conn.prepare_cached("INSERT INTO chunk_fts (chunk_fts, rowid, text) VALUES ('delete', ?1, ?2)")
                    .map_err(db)?
                    .execute(params![key, old_text])
                    .map_err(db)?;
                conn.prepare_cached("UPDATE chunk SET text = ?2, meta_json = ?3, vector = ?4 WHERE key = ?1")
                    .map_err(db)?
                    .execute(params![key, record.text, meta_json, vector])
                    .map_err(db)?;
                conn.prepare_cached("DELETE FROM chunk_meta WHERE chunk_key = ?1")
                    .map_err(db)?
                    .execute([key])
                    .map_err(db)?;
                key
            }
            None => {
                conn.prepare_cached("INSERT INTO chunk (index_name, id, text, meta_json, vector) VALUES (?1, ?2, ?3, ?4, ?5)")
                    .map_err(db)?
                    .execute(params![name, record.id, record.text, meta_json, vector])
                    .map_err(db)?;
                conn.last_insert_rowid()
            }
        };
        assert!(key >= 1, "chunk keys start at 1");
        for (meta_key, value) in &record.meta {
            conn.prepare_cached("INSERT INTO chunk_meta (chunk_key, key, value) VALUES (?1, ?2, ?3)")
                .map_err(db)?
                .execute(params![key, meta_key, value])
                .map_err(db)?;
        }
        conn.prepare_cached("INSERT INTO chunk_fts (rowid, text) VALUES (?1, ?2)")
            .map_err(db)?
            .execute(params![key, record.text])
            .map_err(db)?;
        keys.push(to_id(key)?);
    }
    assert_eq!(keys.len(), records.len());
    Ok(keys)
}

/// Deletes one chunk: its keyword row, meta pairs and row. `text` must
/// be the stored text, which the FTS5 `delete` command needs. Input: a
/// connection inside a write transaction. Output: `Ok`.
fn delete_row(conn: &Connection, key: i64, text: &str) -> Result<(), CoreError> {
    conn.prepare_cached("INSERT INTO chunk_fts (chunk_fts, rowid, text) VALUES ('delete', ?1, ?2)")
        .map_err(db)?
        .execute(params![key, text])
        .map_err(db)?;
    conn.prepare_cached("DELETE FROM chunk_meta WHERE chunk_key = ?1").map_err(db)?.execute([key]).map_err(db)?;
    let deleted = conn.prepare_cached("DELETE FROM chunk WHERE key = ?1").map_err(db)?.execute([key]).map_err(db)?;
    assert_eq!(deleted, 1, "deleted chunk exists");
    Ok(())
}

/// Returns the keys of chunks of `name` whose meta has every pair.
/// Values are bound as parameters; only the pair count shapes the SQL.
/// Input: a connection, the name and at least one pair. Output: the keys.
fn allowed_keys(conn: &Connection, name: &str, pairs: &[(String, String)]) -> Result<HashSet<u64>, CoreError> {
    assert!(!pairs.is_empty(), "an empty filter allows every key");
    let sql = format!("SELECT key FROM chunk WHERE index_name = ?1{}", meta_clauses("key", pairs.len(), 2));
    let mut values: Vec<&str> = vec![name];
    for (key, value) in pairs {
        values.push(key);
        values.push(value);
    }
    let mut stmt = conn.prepare(&sql).map_err(db)?;
    let rows = stmt.query_map(params_from_iter(values), |row| row.get::<_, i64>(0)).map_err(db)?;
    let mut out = HashSet::new();
    for row in rows {
        out.insert(to_id(row.map_err(db)?)?);
    }
    Ok(out)
}

/// Builds one `AND <column> IN (...)` clause per filter pair, with the
/// key and value of pair `i` bound at `?first + 2i` and the next number.
/// Input: the key column, the pair count and the first parameter number.
/// Output: SQL text shaped by the count alone, never by a value.
fn meta_clauses(column: &str, pairs: usize, first: usize) -> String {
    let mut sql = String::new();
    for position in 0..pairs {
        let k = first + 2 * position;
        sql.push_str(&format!(" AND {column} IN (SELECT chunk_key FROM chunk_meta WHERE key = ?{k} AND value = ?{})", k + 1));
    }
    sql
}

/// Turns caller text into an FTS5 query of plain terms: split on every
/// non-alphanumeric char, lowercased, each term quoted, joined with `OR`.
/// No operator, column filter, prefix or `NEAR` survives. Input: the
/// text. Output: the terms and the query, empty when no term is left.
fn fts_query(text: &str) -> (Vec<String>, String) {
    let terms: Vec<String> = text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(str::to_lowercase).collect();
    let quoted: Vec<String> = terms.iter().map(|t| format!("\"{}\"", t.replace('"', "\"\""))).collect();
    (terms, quoted.join(" OR "))
}

/// Reads the generation of index `name`. Input: a connection and the
/// name. Output: the generation, or `CoreError::Store` when the row is
/// gone.
fn read_generation(conn: &Connection, name: &str) -> Result<i64, CoreError> {
    conn.query_row("SELECT generation FROM vector_index WHERE name = ?1", [name], |row| row.get(0))
        .optional()
        .map_err(db)?
        .ok_or_else(|| CoreError::Store(format!("vector index {name} is missing")))
}

/// Raises the generation of index `name` by one, so other connections
/// see the write. Input: a connection inside a write transaction.
/// Output: the new generation.
fn bump_generation(conn: &Connection, name: &str) -> Result<i64, CoreError> {
    let before = read_generation(conn, name)?;
    let updated = conn.execute("UPDATE vector_index SET generation = generation + 1 WHERE name = ?1", [name]).map_err(db)?;
    assert_eq!(updated, 1, "vector index row exists");
    let after = read_generation(conn, name)?;
    assert_eq!(after, before + 1, "generation rises by one");
    Ok(after)
}

/// Reads the stored kind and params of index `name`. Input: a
/// connection and the name. Output: the kind, or `CoreError::Store` for
/// an unknown kind or params that do not parse.
fn stored_kind(conn: &Connection, name: &str) -> Result<IndexKind, CoreError> {
    let (kind, params): (String, String) = conn
        .query_row("SELECT kind, params_json FROM vector_index WHERE name = ?1", [name], |row| Ok((row.get(0)?, row.get(1)?)))
        .map_err(db)?;
    match kind.as_str() {
        KIND_BRUTE => Ok(IndexKind::BruteForce),
        KIND_HNSW => {
            let corrupt = || CoreError::Store(format!("corrupt store: hnsw params of {name}"));
            let value: serde_json::Value = serde_json::from_str(&params).map_err(|_| corrupt())?;
            let uint = |field: &str| value.get(field).and_then(serde_json::Value::as_u64).ok_or_else(corrupt);
            let size = |field: &str| uint(field).and_then(|n| usize::try_from(n).map_err(|_| corrupt()));
            let fraction = value.get("max_deleted_fraction").and_then(serde_json::Value::as_f64).ok_or_else(corrupt)?;
            let params = HnswParams::default()
                .with_m(size("m")?)
                .with_ef_construction(size("ef_construction")?)
                .with_ef_search(size("ef_search")?)
                .with_seed(uint("seed")?)
                .with_max_deleted_fraction(fraction as f32);
            Ok(IndexKind::Hnsw(params))
        }
        _ => Err(CoreError::Store(format!("corrupt store: unknown index kind of {name}"))),
    }
}

/// Returns the `vector_index.kind` text of a kind.
/// Input: the kind. Output: `brute_force` or `hnsw`, or
/// `CoreError::Store` for a kind this store does not know.
fn kind_name(kind: &IndexKind) -> Result<&'static str, CoreError> {
    match kind {
        IndexKind::BruteForce => Ok(KIND_BRUTE),
        IndexKind::Hnsw(_) => Ok(KIND_HNSW),
        _ => Err(CoreError::Store("unsupported index kind".into())),
    }
}

/// Returns the `vector_index.params_json` text of a kind: `{}` for
/// brute force, the five HNSW params otherwise. The same kind always
/// gives the same text, so stored and wanted params compare as text.
fn params_json(kind: &IndexKind) -> Result<String, CoreError> {
    match kind {
        IndexKind::BruteForce => Ok(String::from("{}")),
        IndexKind::Hnsw(p) => Ok(serde_json::json!({
            "m": p.m(),
            "ef_construction": p.ef_construction(),
            "ef_search": p.ef_search(),
            "seed": p.seed(),
            "max_deleted_fraction": p.max_deleted_fraction(),
        })
        .to_string()),
        _ => Err(CoreError::Store("unsupported index kind".into())),
    }
}

/// Maps an index error to a store error. A dimension mismatch keeps its
/// own `CoreError` variant; every other failure is `CoreError::Store`.
/// Input: the index error. Output: the core error.
fn index_err(e: IndexError) -> CoreError {
    match e {
        IndexError::DimensionMismatch { expected, actual } => CoreError::DimensionMismatch { expected, actual },
        other => CoreError::Store(format!("vector index: {other}")),
    }
}

/// Checks an index name: 1 to 64 chars, each a lowercase ASCII letter,
/// a digit or `_`. Input: the name. Output: `Ok`, or `CoreError::Store`.
fn check_name(name: &str) -> Result<(), CoreError> {
    let ok = (1..=NAME_BYTES_MAX).contains(&name.len()) && name.bytes().all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_');
    if ok {
        Ok(())
    } else {
        Err(CoreError::Store(format!("vector index name must match [a-z0-9_]{{1,{NAME_BYTES_MAX}}}")))
    }
}

/// Checks settings: `dims` in `1..=MAX_DIMS` and every bound above 0.
/// Input: the settings. Output: `Ok`, or `CoreError::Store` naming the
/// first bad value.
fn check_settings(settings: &VectorSettings) -> Result<(), CoreError> {
    if settings.dims == 0 || settings.dims > MAX_DIMS {
        return Err(CoreError::Store(format!("dims must be 1 to {MAX_DIMS}, got {}", settings.dims)));
    }
    for (field, value) in [
        ("max_text_bytes", settings.max_text_bytes),
        ("max_meta_pairs", settings.max_meta_pairs),
        ("max_meta_key_bytes", settings.max_meta_key_bytes),
        ("max_meta_value_bytes", settings.max_meta_value_bytes),
        ("max_batch", settings.max_batch),
        ("max_query_terms", settings.max_query_terms),
        ("max_query_chars", settings.max_query_chars),
    ] {
        if value == 0 {
            return Err(CoreError::Store(format!("{field} must be at least 1")));
        }
    }
    Ok(())
}

/// Checks a scope and source used as meta values: not empty and within
/// `max_meta_value_bytes`. Input: both and the settings. Output: `Ok`,
/// or `CoreError::Store`.
fn check_source(scope: &str, source: &str, settings: &VectorSettings) -> Result<(), CoreError> {
    for (field, value) in [(SCOPE_KEY, scope), (SOURCE_KEY, source)] {
        if value.is_empty() || value.len() > settings.max_meta_value_bytes {
            return Err(CoreError::Store(format!("{field} must be 1 to {} bytes", settings.max_meta_value_bytes)));
        }
    }
    Ok(())
}

/// Checks a batch before anything is written: its size, and per record
/// the id length, dimension, finite components, text bytes and meta
/// bounds. Input: the records, settings and dims. Output: `Ok`, or the
/// first failure, naming the record position and never its text.
fn check_records(records: &[VectorRecord], settings: &VectorSettings, dims: usize) -> Result<(), CoreError> {
    if records.len() > settings.max_batch {
        return Err(CoreError::Store(format!("batch of {} records, above max_batch {}", records.len(), settings.max_batch)));
    }
    for (position, record) in records.iter().enumerate() {
        if record.id.is_empty() || record.id.len() > ID_BYTES_MAX {
            return Err(CoreError::Store(format!("record {position}: id must be 1 to {ID_BYTES_MAX} bytes")));
        }
        if record.vector.len() != dims {
            return Err(CoreError::DimensionMismatch { expected: dims, actual: record.vector.len() });
        }
        if record.vector.iter().any(|x| !x.is_finite()) {
            return Err(CoreError::Store(format!("record {position}: vector has a non-finite component")));
        }
        if record.text.len() > settings.max_text_bytes {
            return Err(CoreError::Store(format!(
                "record {position}: text is {} bytes, above max_text_bytes {}",
                record.text.len(),
                settings.max_text_bytes
            )));
        }
        if record.meta.len() > settings.max_meta_pairs {
            return Err(CoreError::Store(format!(
                "record {position}: {} meta pairs, above max_meta_pairs {}",
                record.meta.len(),
                settings.max_meta_pairs
            )));
        }
        for (key, value) in &record.meta {
            if key.is_empty() || key.len() > settings.max_meta_key_bytes {
                return Err(CoreError::Store(format!("record {position}: meta key must be 1 to {} bytes", settings.max_meta_key_bytes)));
            }
            if value.len() > settings.max_meta_value_bytes {
                return Err(CoreError::Store(format!("record {position}: meta value above {} bytes", settings.max_meta_value_bytes)));
            }
        }
    }
    Ok(())
}

/// Encodes a vector as little-endian f32 bytes, `4 * len` long.
/// Input: the vector. Output: the bytes stored in `chunk.vector`.
/// The inverse of `decode`.
fn encode(vector: &[f32]) -> Vec<u8> {
    let bytes: Vec<u8> = vector.iter().flat_map(|x| x.to_le_bytes()).collect();
    assert_eq!(bytes.len(), vector.len() * 4);
    bytes
}

/// Decodes a stored vector blob. Input: the bytes and the index dims.
/// Output: the vector, or `CoreError::Store` when the blob is not
/// `4 * dims` bytes. The inverse of `encode`.
fn decode(bytes: &[u8], dims: usize) -> Result<Vec<f32>, CoreError> {
    if bytes.len() != dims * 4 {
        return Err(CoreError::Store(format!("corrupt store: vector of {} bytes, expected {}", bytes.len(), dims * 4)));
    }
    Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

/// Converts a stored chunk key or page number to an index id.
/// Input: the stored value. Output: the id, or `CoreError::Store` for a
/// value below 1, which the schema never writes.
fn to_id(key: i64) -> Result<u64, CoreError> {
    match u64::try_from(key) {
        Ok(id) if id >= 1 => Ok(id),
        _ => Err(CoreError::Store(format!("corrupt store: key {key}"))),
    }
}

/// Cuts `text` to at most `max` bytes on a char boundary.
/// Input: the text and the cap. Output: the longest prefix that fits.
/// Never splits a UTF-8 char.
fn cut_utf8(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_bytes_round_trip() {
        let v = vec![0.0, -1.5, f32::MIN_POSITIVE, 3.25];
        assert_eq!(decode(&encode(&v), 4).unwrap(), v);
        assert!(decode(&encode(&v), 3).is_err());
    }

    #[test]
    fn names_follow_the_pattern() {
        assert!(check_name("docs_2").is_ok());
        for bad in ["", "Docs", "a-b", "a b", "é", &"a".repeat(65)] {
            assert!(check_name(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn hnsw_params_round_trip_through_json() {
        let kind = IndexKind::Hnsw(HnswParams::default().with_m(8).with_seed(u64::MAX).with_max_deleted_fraction(0.3));
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch("CREATE TABLE vector_index (name TEXT, kind TEXT, params_json TEXT)").unwrap();
        conn.execute("INSERT INTO vector_index VALUES ('x', ?1, ?2)", params![kind_name(&kind).unwrap(), params_json(&kind).unwrap()])
            .unwrap();
        assert_eq!(stored_kind(&conn, "x").unwrap(), kind);
    }

    #[test]
    fn fts_query_quotes_plain_terms() {
        assert_eq!(fts_query("NEAR(a b)").1, "\"near\" OR \"a\" OR \"b\"");
        assert_eq!(fts_query("text:secret a*").1, "\"text\" OR \"secret\" OR \"a\"");
        assert_eq!(fts_query("\" ^ * ()").0.len(), 0);
        assert_eq!(fts_query("Café-au-lait").0, ["café", "au", "lait"]);
    }

    #[test]
    fn cut_never_splits_a_char() {
        assert_eq!(cut_utf8("abc", 5), "abc");
        assert_eq!(cut_utf8("aé", 2), "a");
        assert_eq!(cut_utf8("aé", 3), "aé");
    }
}
