use crate::common::*;
use troupekit::Storage;

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_checkpoints_allow_resume() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("runs.db");
    let built = greeter()
        .checkpoints(Storage::sqlite(&path))
        .build()
        .await
        .unwrap();
    let first = built.run_with("I am Ada").await.unwrap();
    let again = built.resume(first.run_id()).await.unwrap();
    assert_eq!(again, "Hello! You said: I am Ada");
    let reopened = greeter()
        .checkpoints(Storage::sqlite(&path))
        .build()
        .await
        .unwrap();
    assert_eq!(
        reopened.resume(first.run_id()).await.unwrap(),
        "Hello! You said: I am Ada"
    );
    let unknown = reopened.resume(troupekit_core::ids::RunId::new()).await;
    assert!(matches!(unknown, Err(troupekit::KitError::Config { .. })));
}

#[tokio::test]
async fn memory_checkpoints_do_not_grow_across_runs() {
    let built = greeter()
        .checkpoints(Storage::memory())
        .build()
        .await
        .unwrap();
    let first = built.run_with("one").await.unwrap();
    built.run("two").await.unwrap();
    let err = built.resume(first.run_id()).await.unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent `greeter`: resume needs Storage::sqlite(..) or Storage::custom(..) checkpoints"
    );
}

#[tokio::test]
async fn custom_store_keeps_runs_for_resume() {
    let store = std::sync::Arc::new(troupekit_core::inmem::InMemCheckpoints::default());
    let built = greeter()
        .checkpoints(Storage::custom(store))
        .build()
        .await
        .unwrap();
    let first = built.run_with("Ada").await.unwrap();
    assert_eq!(
        built.resume(first.run_id()).await.unwrap(),
        "Hello! You said: Ada"
    );
}
