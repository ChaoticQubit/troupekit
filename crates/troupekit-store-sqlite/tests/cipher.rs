#![cfg(any(feature = "sqlcipher", feature = "sqlcipher-system"))]

use serde_json::json;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::secret::Secret;
use troupekit_core::store::{BlobStore, Checkpoint, CheckpointStore};
use troupekit_store_sqlite::SqliteStore;

const KEY: &str = "correct-horse-battery-staple-7f3a";

fn key(text: &str) -> Secret {
    Secret::new(text.to_string()).unwrap()
}

async fn encrypted(path: &std::path::Path) -> SqliteStore {
    SqliteStore::builder(path)
        .key(key(KEY))
        .open()
        .await
        .unwrap()
}

#[tokio::test]
async fn encrypted_file_has_no_sqlite_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("secret.db");
    let store = encrypted(&path).await;
    store.put("p", b"plain words".to_vec()).await.unwrap();
    drop(store);
    let bytes = std::fs::read(&path).unwrap();
    assert!(bytes.len() >= 16);
    assert_ne!(&bytes[..16], b"SQLite format 3\0");
    assert!(!bytes.windows(11).any(|w| w == b"plain words"));
}

#[tokio::test]
async fn wrong_key_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("secret.db");
    drop(encrypted(&path).await);
    let err = SqliteStore::builder(&path)
        .key(key("wrong-key"))
        .open()
        .await
        .unwrap_err();
    assert_eq!(
        err,
        CoreError::Store("cannot open store: wrong key or not a database".into())
    );
}

#[tokio::test]
async fn open_without_key_of_encrypted_file_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("secret.db");
    drop(encrypted(&path).await);
    let err = SqliteStore::open(&path).await.unwrap_err();
    assert_eq!(
        err,
        CoreError::Store("cannot open store: wrong key or not a database".into())
    );
}

#[tokio::test]
async fn right_key_reads_back_data() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("secret.db");
    let run = RunId::new();
    let store = encrypted(&path).await;
    store
        .save(Checkpoint {
            run_id: run,
            step: 0,
            state: json!({"msg": "hi"}),
        })
        .await
        .unwrap();
    drop(store);
    let store = encrypted(&path).await;
    assert_eq!(
        store.latest(run).await.unwrap().unwrap().state,
        json!({"msg": "hi"})
    );
}

#[tokio::test]
async fn key_never_appears_in_errors_or_debug() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("secret.db");
    let builder = SqliteStore::builder(&path).key(key(KEY));
    assert!(!format!("{builder:?}").contains(KEY));
    let store = builder.open().await.unwrap();
    assert!(!format!("{store:?}").contains(KEY));
    drop(store);
    let wrong = "a-different-key-0000";
    let err = SqliteStore::builder(&path)
        .key(key(wrong))
        .open()
        .await
        .unwrap_err();
    let text = format!("{err} {err:?}");
    assert!(!text.contains(wrong) && !text.contains(KEY), "{text}");
    let quoted = "it's'; PRAGMA key = 'x";
    let err = SqliteStore::builder(&path)
        .key(key(quoted))
        .open()
        .await
        .unwrap_err();
    let text = format!("{err} {err:?}");
    assert!(!text.contains(quoted), "{text}");
}
