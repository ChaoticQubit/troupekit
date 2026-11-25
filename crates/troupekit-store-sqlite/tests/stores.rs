use serde_json::json;
use std::path::PathBuf;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::*;
use troupekit_store_sqlite::{SqliteStore, SqliteStoreBuilder};

struct Temp {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

fn temp() -> Temp {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    Temp { _dir: dir, path }
}

impl Temp {
    async fn open(&self) -> SqliteStore {
        SqliteStore::open(&self.path).await.unwrap()
    }

    async fn with(&self, f: impl FnOnce(SqliteStoreBuilder) -> SqliteStoreBuilder) -> SqliteStore {
        f(SqliteStore::builder(&self.path)).open().await.unwrap()
    }
}

fn cp(run_id: RunId, step: u64) -> Checkpoint {
    Checkpoint {
        run_id,
        step,
        state: json!({ "step": step }),
    }
}

async fn save_steps(s: &SqliteStore, run: RunId, steps: std::ops::RangeInclusive<u64>) {
    for step in steps {
        s.save(cp(run, step)).await.unwrap();
    }
}

#[tokio::test]
async fn checkpoints_must_advance_by_one() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    s.save(Checkpoint {
        run_id: run,
        step: 0,
        state: json!({"a":1}),
    })
    .await
    .unwrap();
    s.save(Checkpoint {
        run_id: run,
        step: 1,
        state: json!({"a":2}),
    })
    .await
    .unwrap();
    let err = s
        .save(Checkpoint {
            run_id: run,
            step: 1,
            state: json!({}),
        })
        .await
        .unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 2,
            actual: 1
        }
    );
    assert_eq!(s.latest(run).await.unwrap().unwrap().state, json!({"a":2}));
    assert!(s.latest(RunId::new()).await.unwrap().is_none());
    let err = s.save(cp(RunId::new(), 3)).await.unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 0,
            actual: 3
        }
    );
}

#[tokio::test]
async fn checkpoint_survives_reopen() {
    let t = temp();
    let run = RunId::new();
    save_steps(&t.open().await, run, 0..=2).await;
    assert_eq!(t.open().await.latest(run).await.unwrap(), Some(cp(run, 2)));
}

#[tokio::test]
async fn latest_of_unknown_run_is_none() {
    let t = temp();
    assert_eq!(t.open().await.latest(RunId::new()).await.unwrap(), None);
}

#[tokio::test]
async fn history_lists_every_step_in_order() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    let other = RunId::new();
    save_steps(&s, run, 0..=4).await;
    save_steps(&s, other, 0..=1).await;
    let expected: Vec<Checkpoint> = (0..=4).map(|n| cp(run, n)).collect();
    assert_eq!(s.history(run).await.unwrap(), expected);
    assert!(s.history(RunId::new()).await.unwrap().is_empty());
}

#[tokio::test]
async fn prune_keeps_the_last_n_steps() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    save_steps(&s, run, 0..=5).await;
    assert_eq!(s.prune(run, 2).await.unwrap(), 4);
    assert_eq!(s.history(run).await.unwrap(), vec![cp(run, 4), cp(run, 5)]);
    assert_eq!(s.prune(run, 10).await.unwrap(), 0);
    assert_eq!(s.prune(RunId::new(), 1).await.unwrap(), 0);
}

#[tokio::test]
async fn save_after_prune_takes_the_next_step() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    save_steps(&s, run, 0..=5).await;
    s.prune(run, 1).await.unwrap();
    s.save(cp(run, 6)).await.unwrap();
    let err = s.save(cp(run, 0)).await.unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 7,
            actual: 0
        }
    );
}

#[tokio::test]
async fn prune_rejects_keep_last_zero() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    save_steps(&s, run, 0..=2).await;
    assert!(matches!(s.prune(run, 0).await, Err(CoreError::Store(_))));
    assert_eq!(s.history(run).await.unwrap().len(), 3);
}

#[tokio::test]
async fn step_beyond_i64_is_rejected() {
    let t = temp();
    let s = t.open().await;
    let err = s.save(cp(RunId::new(), u64::MAX)).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}

#[tokio::test]
async fn state_above_the_setting_is_rejected() {
    let t = temp();
    let s = t.with(|b| b.max_state_bytes(16)).await;
    let run = RunId::new();
    let big = Checkpoint {
        run_id: run,
        step: 0,
        state: json!("this text is longer than 16"),
    };
    let err = s.save(big).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("max_state_bytes")),
        "{err:?}"
    );
    assert!(s.latest(run).await.unwrap().is_none());
    s.save(Checkpoint {
        run_id: run,
        step: 0,
        state: json!("short"),
    })
    .await
    .unwrap();
}

#[tokio::test]
async fn history_above_the_setting_is_rejected() {
    let t = temp();
    let s = t.with(|b| b.max_history_steps(2)).await;
    let run = RunId::new();
    save_steps(&s, run, 0..=2).await;
    let err = s.history(run).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("max_history_steps")),
        "{err:?}"
    );
    s.prune(run, 2).await.unwrap();
    assert_eq!(s.history(run).await.unwrap().len(), 2);
}

use std::sync::Arc;
use troupekit_core::message::{ContentPart, MediaSource, Message, Role};

#[tokio::test]
async fn second_concurrent_writer_conflicts() {
    let t = temp();
    let m = Arc::new(t.open().await);
    assert_eq!(
        m.append("g1", 0, vec![Message::text(Role::User, "a")])
            .await
            .unwrap(),
        1
    );
    let (a, b) = tokio::join!(
        m.append("g1", 1, vec![Message::text(Role::User, "b")]),
        m.append("g1", 1, vec![Message::text(Role::User, "c")]),
    );
    assert!(a.is_ok() != b.is_ok());
    let err = a.err().or(b.err()).unwrap();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 1,
            actual: 2
        }
    );
    assert_eq!(m.read("g1").await.unwrap().len(), 2);
}

#[tokio::test]
async fn conflicting_append_writes_nothing() {
    let t = temp();
    let m = t.open().await;
    m.append("log", 0, vec![Message::text(Role::User, "a")])
        .await
        .unwrap();
    let many = vec![
        Message::text(Role::User, "x"),
        Message::text(Role::User, "y"),
    ];
    let err = m.append("log", 0, many).await.unwrap_err();
    assert_eq!(
        err,
        CoreError::Conflict {
            expected: 0,
            actual: 1
        }
    );
    assert_eq!(
        m.read("log").await.unwrap(),
        vec![Message::text(Role::User, "a")]
    );
}

#[tokio::test]
async fn log_survives_reopen_in_order() {
    let t = temp();
    let texts = ["one", "two", "three", "four"];
    {
        let m = t.open().await;
        m.append("log", 0, vec![Message::text(Role::User, texts[0])])
            .await
            .unwrap();
        let rest = texts[1..]
            .iter()
            .map(|s| Message::text(Role::Assistant, *s))
            .collect();
        assert_eq!(m.append("log", 1, rest).await.unwrap(), 4);
    }
    let read = t.open().await.read("log").await.unwrap();
    assert_eq!(read.len(), 4);
    assert_eq!(read[0], Message::text(Role::User, "one"));
    assert_eq!(read[3], Message::text(Role::Assistant, "four"));
}

#[tokio::test]
async fn unknown_log_reads_empty() {
    let t = temp();
    let m = t.open().await;
    assert!(m.read("nobody").await.unwrap().is_empty());
    assert_eq!(m.append("fresh", 0, Vec::new()).await.unwrap(), 0);
}

#[tokio::test]
async fn every_content_part_round_trips() {
    let t = temp();
    let messages = vec![
        Message {
            role: Role::User,
            parts: vec![
                ContentPart::Text {
                    text: "look".into(),
                },
                ContentPart::Image {
                    media_type: "image/png".into(),
                    source: MediaSource::Base64 {
                        data: "aGk=".into(),
                    },
                },
                ContentPart::Image {
                    media_type: "image/jpeg".into(),
                    source: MediaSource::Url {
                        url: "https://example.com/a.jpg".into(),
                    },
                },
            ],
        },
        Message {
            role: Role::Assistant,
            parts: vec![
                ContentPart::ToolCall {
                    id: "c1".into(),
                    name: "search".into(),
                    arguments: json!({"q": "rust", "n": 3}),
                },
                ContentPart::Hosted {
                    tool: Some("web".into()),
                    block: json!({"raw": [1, 2]}),
                },
            ],
        },
        Message {
            role: Role::Tool,
            parts: vec![ContentPart::ToolResult {
                call_id: "c1".into(),
                content: "found".into(),
                is_error: false,
            }],
        },
    ];
    t.open()
        .await
        .append("parts", 0, messages.clone())
        .await
        .unwrap();
    assert_eq!(t.open().await.read("parts").await.unwrap(), messages);
}
