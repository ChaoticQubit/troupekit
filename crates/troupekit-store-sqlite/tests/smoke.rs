use serde_json::json;
use troupekit_core::ids::RunId;
use troupekit_core::store::{BlobStore, Checkpoint, CheckpointStore};
use troupekit_store_sqlite::SqliteStore;

#[tokio::test]
async fn store_opens_migrates_writes_and_reads() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("smoke.db");
    let store = SqliteStore::open(&path).await.unwrap();
    store.put("a", b"bytes".to_vec()).await.unwrap();
    let run = RunId::new();
    store
        .save(Checkpoint {
            run_id: run,
            step: 0,
            state: json!({"k": 1}),
        })
        .await
        .unwrap();
    drop(store);
    let store = SqliteStore::open(&path).await.unwrap();
    assert_eq!(store.get("a").await.unwrap(), b"bytes".to_vec());
    assert_eq!(
        store.latest(run).await.unwrap().unwrap().state,
        json!({"k": 1})
    );
}
