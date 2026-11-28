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

#[tokio::test]
async fn blobs_round_trip() {
    let t = temp();
    let b = t.open().await;
    b.put("goals/1/plan.pdf", vec![1, 2, 3]).await.unwrap();
    assert_eq!(b.get("goals/1/plan.pdf").await.unwrap(), vec![1, 2, 3]);
    assert!(matches!(b.get("nope").await, Err(CoreError::NotFound(_))));
}

#[tokio::test]
async fn put_replaces_existing_bytes() {
    let t = temp();
    let b = t.open().await;
    b.put("p", vec![1, 2, 3]).await.unwrap();
    b.put("p", vec![9]).await.unwrap();
    assert_eq!(b.get("p").await.unwrap(), vec![9]);
}

#[tokio::test]
async fn missing_blob_is_not_found() {
    let t = temp();
    let err = t.open().await.get("missing/path").await.unwrap_err();
    assert_eq!(err, CoreError::NotFound("missing/path".into()));
}

#[tokio::test]
async fn empty_blob_round_trips() {
    let t = temp();
    let b = t.open().await;
    b.put("empty", Vec::new()).await.unwrap();
    assert_eq!(b.get("empty").await.unwrap(), Vec::<u8>::new());
}

#[tokio::test]
async fn blob_survives_reopen() {
    let t = temp();
    t.open().await.put("keep", vec![7; 1000]).await.unwrap();
    assert_eq!(t.open().await.get("keep").await.unwrap(), vec![7; 1000]);
}

#[tokio::test]
async fn blob_above_the_setting_is_rejected() {
    let t = temp();
    let b = t.with(|b| b.max_blob_bytes(8)).await;
    b.put("old", vec![1]).await.unwrap();
    let err = b.put("old", vec![0; 9]).await.unwrap_err();
    assert!(
        matches!(&err, CoreError::Store(m) if m.contains("max_blob_bytes")),
        "{err:?}"
    );
    assert_eq!(b.get("old").await.unwrap(), vec![1]);
    b.put("fits", vec![0; 8]).await.unwrap();
}

use troupekit_core::event::Usage;

fn usage_row(path: &std::path::Path, key: &str) -> Option<(i64, i64, i64)> {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.query_row(
        "SELECT input_tokens, output_tokens, cost_usd_micros FROM usage_total WHERE ledger_key = ?1",
        [key],
        |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
    )
    .ok()
}

#[tokio::test]
async fn ledger_sums_per_key() {
    let t = temp();
    let ledger = t.open().await;
    let u = Usage {
        input_tokens: 3,
        output_tokens: 4,
        cost_usd_micros: 7,
    };
    for _ in 0..2 {
        ledger.record("goal:1", u).await.unwrap();
        ledger.record("user:1", u).await.unwrap();
    }
    assert_eq!(usage_row(&t.path, "goal:1"), Some((6, 8, 14)));
    assert_eq!(ledger.period_total("user:1").await.unwrap(), 14);
    assert_eq!(usage_row(&t.path, "other"), None);
}

#[tokio::test]
async fn ledger_survives_reopen() {
    let t = temp();
    let u = Usage {
        input_tokens: 1,
        output_tokens: 1,
        cost_usd_micros: 250,
    };
    t.open().await.record("k", u).await.unwrap();
    let ledger = t.open().await;
    ledger.record("k", u).await.unwrap();
    assert_eq!(ledger.period_total("k").await.unwrap(), 500);
}

#[tokio::test]
async fn unknown_key_totals_zero() {
    let t = temp();
    assert_eq!(t.open().await.period_total("nobody").await.unwrap(), 0);
}

#[tokio::test]
async fn usage_overflow_is_an_error() {
    let t = temp();
    let ledger = t.open().await;
    let big = Usage {
        input_tokens: 1,
        output_tokens: 1,
        cost_usd_micros: i64::MAX as u64,
    };
    ledger.record("k", big).await.unwrap();
    let one = Usage {
        input_tokens: 0,
        output_tokens: 0,
        cost_usd_micros: 1,
    };
    let err = ledger.record("k", one).await.unwrap_err();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
    assert_eq!(ledger.period_total("k").await.unwrap(), i64::MAX as u64);
    let huge = Usage {
        input_tokens: u64::MAX,
        output_tokens: 0,
        cost_usd_micros: 0,
    };
    assert!(matches!(
        ledger.record("other", huge).await,
        Err(CoreError::Store(_))
    ));
    assert_eq!(usage_row(&t.path, "other"), None);
}

#[tokio::test]
async fn hostile_names_are_stored_as_plain_data() {
    let t = temp();
    let s = t.open().await;
    let names = [
        "x'); DROP TABLE checkpoint; --",
        "\"; DELETE FROM blob_entry; SELECT \"",
        "a;b;c' OR '1'='1",
    ];
    let u = Usage {
        input_tokens: 1,
        output_tokens: 2,
        cost_usd_micros: 3,
    };
    for name in names {
        s.append(name, 0, vec![Message::text(Role::User, name)])
            .await
            .unwrap();
        s.put(name, name.as_bytes().to_vec()).await.unwrap();
        s.record(name, u).await.unwrap();
    }
    for name in names {
        assert_eq!(
            s.read(name).await.unwrap(),
            vec![Message::text(Role::User, name)]
        );
        assert_eq!(s.get(name).await.unwrap(), name.as_bytes().to_vec());
        assert_eq!(s.period_total(name).await.unwrap(), 3);
    }
    let conn = rusqlite::Connection::open(&t.path).unwrap();
    let tables: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sqlite_schema WHERE type = 'table'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(tables, 17);
}

fn nested(depth: usize) -> serde_json::Value {
    let mut value = json!(0);
    for level in 0..depth {
        value = if level % 2 == 0 {
            json!([value])
        } else {
            json!({ "k": value })
        };
    }
    value
}

#[tokio::test]
async fn state_too_deep_to_read_back_is_rejected() {
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    s.save(cp(run, 0)).await.unwrap();
    for depth in [128, 129] {
        let deep = Checkpoint {
            run_id: run,
            step: 1,
            state: nested(depth),
        };
        let err = s.save(deep).await.unwrap_err();
        assert!(
            matches!(&err, CoreError::Store(m) if m.contains("nests deeper")),
            "{err:?}"
        );
        assert_eq!(s.latest(run).await.unwrap(), Some(cp(run, 0)));
    }
    let deepest = Checkpoint {
        run_id: run,
        step: 1,
        state: nested(127),
    };
    s.save(deepest.clone()).await.unwrap();
    assert_eq!(s.latest(run).await.unwrap(), Some(deepest));
}

fn call_with_arguments(depth: usize) -> Message {
    Message {
        role: Role::Assistant,
        parts: vec![ContentPart::ToolCall {
            id: "c1".into(),
            name: "deep".into(),
            arguments: nested(depth),
        }],
    }
}

#[tokio::test]
async fn message_too_deep_to_read_back_is_rejected() {
    let t = temp();
    let m = t.open().await;
    let first = Message::text(Role::User, "one");
    m.append("log", 0, vec![first.clone()]).await.unwrap();
    for depth in [125, 126] {
        let err = m
            .append("log", 1, vec![call_with_arguments(depth)])
            .await
            .unwrap_err();
        assert!(
            matches!(&err, CoreError::Store(m) if m.contains("nests deeper")),
            "{err:?}"
        );
        assert_eq!(m.read("log").await.unwrap(), vec![first.clone()]);
    }
    let deepest = call_with_arguments(124);
    assert_eq!(m.append("log", 1, vec![deepest.clone()]).await.unwrap(), 2);
    assert_eq!(m.read("log").await.unwrap(), vec![first, deepest]);
}

fn trace_records(sink: &troupekit_trace::MemorySink) -> Vec<serde_json::Value> {
    sink.lines().iter().map(|l| serde_json::from_str(l).unwrap()).collect()
}

fn named<'a>(records: &'a [serde_json::Value], name: &str) -> &'a serde_json::Value {
    let found: Vec<&serde_json::Value> = records.iter().filter(|r| r["t"] != "end" && r["name"] == name).collect();
    assert_eq!(found.len(), 1, "{name}: {records:?}");
    found[0]
}

#[tokio::test]
async fn checkpoint_save_emits_store_span() {
    use std::sync::Arc;
    use troupekit_trace::{MemorySink, Tracer};
    let t = temp();
    let s = t.open().await;
    let run = RunId::new();
    let sink = Arc::new(MemorySink::default());
    let tracer = Tracer::builder().sink(sink.clone()).build().unwrap();
    tracer
        .in_scope(async {
            s.save(cp(run, 0)).await.unwrap();
            assert!(s.save(cp(run, 3)).await.is_err());
            assert_eq!(s.latest(run).await.unwrap().unwrap().step, 0);
        })
        .await;
    s.save(cp(run, 1)).await.unwrap();
    drop(tracer);
    let records = trace_records(&sink);
    let saves: Vec<&serde_json::Value> = records.iter().filter(|r| r["t"] == "start" && r["name"] == "checkpoint.save").collect();
    assert_eq!(saves.len(), 2, "the untraced save records nothing");
    assert_eq!(saves[0]["cat"], "store");
    assert_eq!(saves[0]["fields"]["run"], run.to_string().as_str());
    assert_eq!(saves[0]["fields"]["step"], 0);
    assert!(saves[0]["fields"]["bytes"].as_u64().unwrap() > 0);
    let latest = named(&records, "checkpoint.latest");
    let end = records.iter().find(|r| r["t"] == "end" && r["span"] == latest["span"]).unwrap();
    assert_eq!(end["fields"]["found_step"], 0);
    let failed = records.iter().filter(|r| r["t"] == "end" && r["ok"] == false).count();
    assert_eq!(failed, 1);
}

#[tokio::test]
async fn blocking_store_call_keeps_parent_span() {
    use std::sync::Arc;
    use troupekit_core::store::{Filter, VectorRecord, VectorStore};
    use troupekit_store_sqlite::VectorSettings;
    use troupekit_trace::{trace_span, MemorySink, Trace, Tracer};
    let t = temp();
    let s = t.open().await;
    let vectors = s.vectors("docs", VectorSettings::new(4)).await.unwrap();
    let sink = Arc::new(MemorySink::default());
    let tracer = Tracer::builder().sink(sink.clone()).build().unwrap();
    let outer_id = tracer
        .in_scope(async {
            let outer = trace_span!(Trace::Knowledge, "test.outer");
            let record = VectorRecord { id: "a".into(), vector: vec![1.0, 0.0, 0.0, 0.0], text: "a".into(), meta: Default::default() };
            outer.instrument(vectors.upsert(vec![record])).await.unwrap();
            let hits = outer.instrument(vectors.search(&[1.0, 0.0, 0.0, 0.0], 3, &Filter::default())).await.unwrap();
            assert_eq!(hits.len(), 1);
            outer.id()
        })
        .await;
    drop(tracer);
    let records = trace_records(&sink);
    let upsert = named(&records, "vectors.upsert");
    assert_eq!(upsert["parent"], outer_id);
    assert_eq!(upsert["fields"]["records"], 1);
    let search = named(&records, "index.search");
    assert_eq!(search["parent"], outer_id, "the span from inside spawn_blocking keeps the caller as parent");
    assert_eq!(search["cat"], "knowledge");
}
