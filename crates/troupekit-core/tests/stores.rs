use serde_json::json;
use std::collections::BTreeMap;
use std::sync::Arc;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::{InMemBlobs, InMemCheckpoints, InMemMemory, InMemVectors};
use troupekit_core::message::{Message, Role};
use troupekit_core::store::*;

#[tokio::test]
async fn checkpoints_must_advance_by_one() {
    let s = InMemCheckpoints::default();
    let run = RunId::new();
    s.save(Checkpoint { run_id: run, step: 0, state: json!({"a":1}) }).await.unwrap();
    s.save(Checkpoint { run_id: run, step: 1, state: json!({"a":2}) }).await.unwrap();
    let err = s.save(Checkpoint { run_id: run, step: 1, state: json!({}) }).await.unwrap_err();
    assert_eq!(err, CoreError::Conflict { expected: 2, actual: 1 });
    assert_eq!(s.latest(run).await.unwrap().unwrap().state, json!({"a":2}));
    assert!(s.latest(RunId::new()).await.unwrap().is_none());
}

#[tokio::test]
async fn second_concurrent_writer_conflicts() {
    let m = Arc::new(InMemMemory::default());
    assert_eq!(m.append("g1", 0, vec![Message::text(Role::User, "a")]).await.unwrap(), 1);
    let (a, b) = tokio::join!(
        m.append("g1", 1, vec![Message::text(Role::User, "b")]),
        m.append("g1", 1, vec![Message::text(Role::User, "c")]),
    );
    assert!(a.is_ok() != b.is_ok());
    assert_eq!(m.read("g1").await.unwrap().len(), 2);
}

#[tokio::test]
async fn vector_search_ranks_and_filters() {
    let v = InMemVectors::new(2);
    let meta = |g: &str| BTreeMap::from([("goal".to_string(), g.to_string())]);
    v.upsert(vec![
        VectorRecord { id: "a".into(), vector: vec![1.0, 0.0], text: "protein".into(), meta: meta("g1") },
        VectorRecord { id: "b".into(), vector: vec![0.7, 0.7], text: "sleep".into(), meta: meta("g1") },
        VectorRecord { id: "c".into(), vector: vec![1.0, 0.1], text: "piano".into(), meta: meta("g2") },
    ]).await.unwrap();
    let hits = v.search(&[1.0, 0.0], 2, &Filter(meta("g1"))).await.unwrap();
    assert_eq!(hits.iter().map(|h| h.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
    assert!(hits[0].score > hits[1].score);
    let err = v.upsert(vec![VectorRecord { id: "d".into(), vector: vec![1.0], text: String::new(), meta: BTreeMap::new() }]).await.unwrap_err();
    assert_eq!(err, CoreError::DimensionMismatch { expected: 2, actual: 1 });
}

#[tokio::test]
async fn blobs_round_trip() {
    let b = InMemBlobs::default();
    b.put("goals/1/plan.pdf", vec![1, 2, 3]).await.unwrap();
    assert_eq!(b.get("goals/1/plan.pdf").await.unwrap(), vec![1, 2, 3]);
    assert!(matches!(b.get("nope").await, Err(CoreError::NotFound(_))));
}

#[test]
fn cosine_basics() {
    assert!((cosine(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
    assert!(cosine(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    assert_eq!(cosine(&[0.0, 0.0], &[1.0, 0.0]), 0.0);
}
