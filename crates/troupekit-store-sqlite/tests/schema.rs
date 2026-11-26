use rusqlite::Connection;
use std::path::Path;
use troupekit_core::error::CoreError;
use troupekit_store_sqlite::{SqliteStore, SCHEMA_VERSION};

fn pragma_i64(path: &Path, name: &str) -> i64 {
    let conn = Connection::open(path).unwrap();
    conn.pragma_query_value(None, name, |row| row.get(0))
        .unwrap()
}

fn pragma_text(path: &Path, name: &str) -> String {
    let conn = Connection::open(path).unwrap();
    conn.pragma_query_value(None, name, |row| row.get(0))
        .unwrap()
}

fn table_names(path: &Path) -> Vec<String> {
    let conn = Connection::open(path).unwrap();
    let mut stmt = conn
        .prepare("SELECT name FROM sqlite_schema WHERE type = 'table' ORDER BY name")
        .unwrap();
    let rows = stmt.query_map([], |row| row.get(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

/// The DDL of a 0.1 store file, frozen as it shipped.
/// Builds a version 1 file without the current migration list,
/// so the step 1 to 2 upgrade is tested on real 0.1 bytes.
/// Never edited: a 0.1 file on disk never changes either.
const V1_DDL: &str = "
CREATE TABLE checkpoint (
  run_id TEXT NOT NULL,
  step INTEGER NOT NULL CHECK (step >= 0),
  state_json BLOB NOT NULL,
  PRIMARY KEY (run_id, step)
) STRICT;

CREATE TABLE memory_entry (
  log TEXT NOT NULL,
  idx INTEGER NOT NULL CHECK (idx >= 0),
  message_json BLOB NOT NULL,
  PRIMARY KEY (log, idx)
) STRICT;

CREATE TABLE blob_entry (
  path TEXT NOT NULL PRIMARY KEY,
  bytes BLOB NOT NULL
) STRICT;

CREATE TABLE job (
  run_id TEXT NOT NULL PRIMARY KEY,
  state TEXT NOT NULL CHECK (state IN ('pending', 'leased', 'done', 'failed')),
  claim_at_ms INTEGER,
  token INTEGER CHECK (token IS NULL OR token > 0),
  next_due_ms INTEGER,
  error TEXT,
  attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
  CHECK ((state IN ('pending', 'leased')) = (claim_at_ms IS NOT NULL)),
  CHECK ((state = 'leased') = (token IS NOT NULL)),
  CHECK (next_due_ms IS NULL OR state = 'leased'),
  CHECK ((state = 'failed') = (error IS NOT NULL))
) STRICT;

CREATE INDEX job_claim ON job (claim_at_ms, run_id) WHERE claim_at_ms IS NOT NULL;

CREATE TABLE lease_token (
  id INTEGER NOT NULL PRIMARY KEY CHECK (id = 1),
  last INTEGER NOT NULL CHECK (last >= 0)
) STRICT;

INSERT INTO lease_token (id, last) VALUES (1, 0);

CREATE TABLE usage_total (
  ledger_key TEXT NOT NULL PRIMARY KEY,
  input_tokens INTEGER NOT NULL CHECK (input_tokens >= 0),
  output_tokens INTEGER NOT NULL CHECK (output_tokens >= 0),
  cost_usd_micros INTEGER NOT NULL CHECK (cost_usd_micros >= 0)
) STRICT;

PRAGMA application_id = 1414679380;
";

/// Writes a version 1 store at `path` with one row in each 0.1 table.
/// Input: a path that does not exist yet. Output: none; the file holds
/// one checkpoint, memory entry, blob, job and usage row.
fn write_v1_file(path: &Path) {
    let conn = Connection::open(path).unwrap();
    conn.execute_batch(V1_DDL).unwrap();
    conn.execute_batch(
        "INSERT INTO checkpoint (run_id, step, state_json) VALUES ('r1', 0, X'7B7D');
         INSERT INTO memory_entry (log, idx, message_json) VALUES ('facts/u', 0, X'5B5D');
         INSERT INTO blob_entry (path, bytes) VALUES ('summary/s', X'0102');
         INSERT INTO job (run_id, state, claim_at_ms, attempts) VALUES ('r1', 'pending', 10, 0);
         INSERT INTO usage_total (ledger_key, input_tokens, output_tokens, cost_usd_micros)
           VALUES ('k', 1, 2, 3);",
    )
    .unwrap();
    conn.pragma_update(None, "user_version", 1).unwrap();
}

/// Dumps every 0.1 table of the file at `path` as text rows, sorted.
/// Input: a store path. Output: one string per row, prefixed with the
/// table name, so two dumps compare row for row.
fn v1_rows(path: &Path) -> Vec<String> {
    let conn = Connection::open(path).unwrap();
    let queries = [
        "SELECT 'checkpoint', run_id, step, hex(state_json) FROM checkpoint",
        "SELECT 'memory_entry', log, idx, hex(message_json) FROM memory_entry",
        "SELECT 'blob_entry', path, 0, hex(bytes) FROM blob_entry",
        "SELECT 'job', run_id, claim_at_ms, state FROM job",
        "SELECT 'lease_token', id, last, '' FROM lease_token",
        "SELECT 'usage_total', ledger_key, input_tokens, output_tokens || '/' || cost_usd_micros FROM usage_total",
    ];
    let mut out = Vec::new();
    for query in queries {
        let mut stmt = conn.prepare(query).unwrap();
        let rows = stmt
            .query_map([], |row| {
                let a: String = row.get(0)?;
                let b: rusqlite::types::Value = row.get(1)?;
                let c: rusqlite::types::Value = row.get(2)?;
                let d: String = row.get(3)?;
                Ok(format!("{a}|{b:?}|{c:?}|{d}"))
            })
            .unwrap();
        out.extend(rows.map(Result::unwrap));
    }
    out.sort();
    out
}

#[tokio::test]
async fn v1_file_migrates_to_v2_and_keeps_rows() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1.db");
    write_v1_file(&path);
    let before = v1_rows(&path);
    assert_eq!(before.len(), 6);
    drop(SqliteStore::open(&path).await.unwrap());
    assert_eq!(pragma_i64(&path, "user_version"), 2);
    assert_eq!(v1_rows(&path), before);
    let tables = table_names(&path);
    for table in ["vector_index", "chunk", "chunk_meta", "chunk_fts", "index_page", "ingest_source"] {
        assert!(tables.iter().any(|t| t == table), "{table} missing: {tables:?}");
    }
}

#[tokio::test]
async fn failing_step_2_leaves_version_1() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("v1.db");
    write_v1_file(&path);
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE ingest_source (x INTEGER)")
            .unwrap();
    }
    let before = v1_rows(&path);
    let err = SqliteStore::open(&path).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("migration step 2 failed")),
        "{err:?}"
    );
    assert_eq!(pragma_i64(&path, "user_version"), 1);
    assert_eq!(v1_rows(&path), before);
    let tables = table_names(&path);
    for table in ["vector_index", "chunk", "chunk_meta", "chunk_fts", "index_page"] {
        assert!(!tables.iter().any(|t| t == table), "{table} left behind: {tables:?}");
    }
}

#[tokio::test]
async fn fts5_is_available() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    drop(SqliteStore::open(&path).await.unwrap());
    let conn = Connection::open(&path).unwrap();
    conn.execute_batch(
        "INSERT INTO chunk (index_name, id, text, meta_json, vector)
           VALUES ('docs', 'a', 'Résumé of the refund policy', '{}', X'0000803F');
         INSERT INTO chunk_fts (rowid, text) SELECT key, text FROM chunk;",
    )
    .unwrap();
    let hits: Vec<String> = conn
        .prepare("SELECT c.id FROM chunk_fts JOIN chunk c ON c.key = chunk_fts.rowid WHERE chunk_fts MATCH 'resume' ORDER BY bm25(chunk_fts)")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(hits, ["a"]);
}

#[tokio::test]
async fn file_reopens_at_the_same_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    drop(SqliteStore::open(&path).await.unwrap());
    let tables = table_names(&path);
    drop(SqliteStore::open(&path).await.unwrap());
    assert_eq!(pragma_i64(&path, "user_version"), i64::from(SCHEMA_VERSION));
    assert_eq!(SCHEMA_VERSION, 2);
    assert_eq!(table_names(&path), tables);
    assert_eq!(
        tables,
        [
            "blob_entry",
            "checkpoint",
            "chunk",
            "chunk_fts",
            "chunk_fts_config",
            "chunk_fts_data",
            "chunk_fts_docsize",
            "chunk_fts_idx",
            "chunk_meta",
            "index_page",
            "ingest_source",
            "job",
            "lease_token",
            "memory_entry",
            "sqlite_sequence",
            "usage_total",
            "vector_index"
        ]
    );
    assert_eq!(pragma_i64(&path, "application_id"), 1_414_679_380);
}

#[tokio::test]
async fn file_uses_wal_and_full_sync_by_default() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    let store = SqliteStore::open(&path).await.unwrap();
    assert_eq!(pragma_text(&path, "journal_mode"), "wal");
    drop(store);
}

#[cfg(unix)]
#[tokio::test]
async fn new_file_is_private_to_its_owner() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    let store = SqliteStore::open(&path).await.unwrap();
    let wal = dir.path().join("store.db-wal");
    for file in [&path, &wal] {
        let mode = std::fs::metadata(file).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "{}", file.display());
    }
    drop(store);
}

#[tokio::test]
async fn foreign_file_is_refused_and_left_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("other.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "application_id", 7).unwrap();
        conn.execute_batch("CREATE TABLE t (x INTEGER)").unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let err = SqliteStore::open(&path).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("not a troupekit store")),
        "{err:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(pragma_text(&path, "journal_mode"), "delete");
}

#[tokio::test]
async fn newer_file_is_refused_and_left_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("newer.db");
    drop(SqliteStore::open(&path).await.unwrap());
    {
        let conn = Connection::open(&path).unwrap();
        conn.pragma_update(None, "journal_mode", "DELETE").unwrap();
        conn.pragma_update(None, "user_version", 9).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let err = SqliteStore::open(&path).await.unwrap_err();
    let text = err.to_string();
    assert!(
        text.contains('9') && text.contains(&SCHEMA_VERSION.to_string()),
        "{text}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(pragma_text(&path, "journal_mode"), "delete");
}

#[tokio::test]
async fn versioned_file_without_the_id_is_refused_and_left_unchanged() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("versioned.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE notes (x INTEGER)")
            .unwrap();
        conn.pragma_update(None, "user_version", 1).unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let err = SqliteStore::open(&path).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("not a troupekit store")),
        "{err:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(pragma_text(&path, "journal_mode"), "delete");
    assert_eq!(pragma_i64(&path, "user_version"), 1);
    assert_eq!(table_names(&path), ["notes"]);
}

#[tokio::test]
async fn table_named_like_an_internal_one_is_still_foreign() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lookalike.db");
    {
        let conn = Connection::open(&path).unwrap();
        conn.execute_batch("CREATE TABLE sqliteXnotes (x INTEGER)")
            .unwrap();
    }
    let before = std::fs::read(&path).unwrap();
    let err = SqliteStore::open(&path).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("not a troupekit store")),
        "{err:?}"
    );
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(table_names(&path), ["sqliteXnotes"]);
}

#[cfg(unix)]
#[tokio::test]
async fn dangling_symlink_path_is_a_store_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("link.db");
    std::os::unix::fs::symlink(dir.path().join("missing.db"), &path).unwrap();
    let err = SqliteStore::open(&path).await.unwrap_err();
    assert!(matches!(&err, CoreError::Store(_)), "{err:?}");
}
