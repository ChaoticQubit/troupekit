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

#[tokio::test]
async fn file_reopens_at_the_same_version() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    drop(SqliteStore::open(&path).await.unwrap());
    let tables = table_names(&path);
    drop(SqliteStore::open(&path).await.unwrap());
    assert_eq!(pragma_i64(&path, "user_version"), i64::from(SCHEMA_VERSION));
    assert_eq!(SCHEMA_VERSION, 1);
    assert_eq!(table_names(&path), tables);
    assert_eq!(
        tables,
        [
            "blob_entry",
            "checkpoint",
            "job",
            "lease_token",
            "memory_entry",
            "usage_total"
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
