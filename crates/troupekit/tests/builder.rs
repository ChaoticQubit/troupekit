#[cfg(feature = "sqlite")]
use crate::common::{finish, graph_io, start, tools};
use std::sync::Arc;
#[cfg(feature = "sqlite")]
use troupekit::core::clock::SystemClock;
use troupekit::core::clock::{Clock, FixedClock};
#[cfg(feature = "sqlite")]
use troupekit::core::error::CoreError;
use troupekit::core::event::Usage;
use troupekit::core::ids::RunId;
#[cfg(feature = "sqlite")]
use troupekit::core::message::{Message, Role};
use troupekit::core::store::Checkpoint;
#[cfg(feature = "sqlite")]
use troupekit::core::store::JobState;
#[cfg(feature = "sqlite")]
use troupekit::SqliteStore;
use troupekit::Troupe;

#[cfg(feature = "sqlite")]
fn system() -> Arc<dyn Clock> {
    Arc::new(SystemClock)
}

#[tokio::test]
async fn builder_without_sqlite_uses_memory_stores() {
    let troupe = Troupe::builder().build().await.unwrap();
    let run = RunId::new();
    let cp = Checkpoint {
        run_id: run,
        step: 0,
        state: serde_json::json!(1),
    };
    troupe.checkpoints().save(cp.clone()).await.unwrap();
    assert_eq!(troupe.checkpoints().latest(run).await.unwrap(), Some(cp));
    troupe.blobs().put("p", vec![1]).await.unwrap();
    assert_eq!(troupe.memory().read("none").await.unwrap(), Vec::new());
    troupe
        .ledger()
        .record(
            "k",
            Usage {
                input_tokens: 0,
                output_tokens: 0,
                cost_usd_micros: 5,
            },
        )
        .await
        .unwrap();
    assert_eq!(troupe.ledger().period_total("k").await.unwrap(), 5);
    let other = Troupe::builder().build().await.unwrap();
    assert!(other.checkpoints().latest(run).await.unwrap().is_none());
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sqlite_data_survives_a_rebuild_on_the_same_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("troupe.db");
    let run = RunId::new();
    {
        let troupe = Troupe::builder().sqlite(&path).build().await.unwrap();
        troupe
            .checkpoints()
            .save(Checkpoint {
                run_id: run,
                step: 0,
                state: serde_json::json!("s"),
            })
            .await
            .unwrap();
        troupe
            .memory()
            .append("log", 0, vec![Message::text(Role::User, "hi")])
            .await
            .unwrap();
        troupe.blobs().put("b", vec![4, 2]).await.unwrap();
        troupe.queue().enqueue(run, 0).await.unwrap();
    }
    let troupe = Troupe::builder().sqlite(&path).build().await.unwrap();
    assert_eq!(
        troupe
            .checkpoints()
            .latest(run)
            .await
            .unwrap()
            .unwrap()
            .step,
        0
    );
    assert_eq!(troupe.memory().read("log").await.unwrap().len(), 1);
    assert_eq!(troupe.blobs().get("b").await.unwrap(), vec![4, 2]);
    assert_eq!(troupe.queue().claim_due(10, 1_000).await.unwrap().len(), 1);
    let direct = SqliteStore::open(&path).await.unwrap();
    assert!(matches!(
        direct.job(run).await.unwrap(),
        Some(JobState::Leased { .. })
    ));
}

#[cfg(feature = "sqlite")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn runner_on_sqlite_matches_the_in_memory_run() {
    let dir = tempfile::tempdir().unwrap();
    let memory = Troupe::builder().build().await.unwrap();
    let io = graph_io(&memory, tools(None), system());
    let id = start(&memory, io.clone(), "goal").await;
    let reference = finish(&memory, io, id).await;
    let sqlite = Troupe::builder()
        .sqlite(dir.path().join("run.db"))
        .build()
        .await
        .unwrap();
    let io = graph_io(&sqlite, tools(None), system());
    let id = start(&sqlite, io.clone(), "goal").await;
    let done = finish(&sqlite, io, id).await;
    assert_eq!(done.state(), reference.state());
    assert_eq!(done.steps(), reference.steps());
    assert_eq!(done.usage(), reference.usage());
    assert_eq!(
        done.state()["c"],
        serde_json::json!("goal>first>second>third")
    );
    let history = sqlite.checkpoints().history(id).await.unwrap();
    assert_eq!(
        history.iter().map(|c| c.step).collect::<Vec<_>>(),
        (0..=done.steps()).collect::<Vec<_>>()
    );
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn unwritable_path_is_a_store_error() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("missing").join("dir").join("troupe.db");
    let err = Troupe::builder().sqlite(&path).build().await.err().unwrap();
    assert!(matches!(err, CoreError::Store(_)), "{err:?}");
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn troupe_ledger_plugged_into_the_io_is_durable() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("ledger.db");
    let usage = Usage {
        input_tokens: 10,
        output_tokens: 20,
        cost_usd_micros: 30,
    };
    {
        let troupe = Troupe::builder().sqlite(&path).build().await.unwrap();
        let mut io = troupekit::agent::SimpleIo::new(tools(None));
        io.ledger = troupe.ledger();
        io.ledger.record("user:1", usage).await.unwrap();
    }
    let troupe = Troupe::builder().sqlite(&path).build().await.unwrap();
    assert_eq!(troupe.ledger().period_total("user:1").await.unwrap(), 30);
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn store_settings_pass_through_sqlite_with() {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::builder(dir.path().join("small.db")).max_blob_bytes(8);
    let troupe = Troupe::builder().sqlite_with(store).build().await.unwrap();
    assert!(matches!(
        troupe.blobs().put("big", vec![0; 9]).await,
        Err(CoreError::Store(_))
    ));
    troupe.blobs().put("fits", vec![0; 8]).await.unwrap();
}

async fn clock_drives_queue(troupe: Troupe, clock: Arc<FixedClock>) {
    let run = RunId::new();
    troupe
        .queue()
        .enqueue(run, clock.now_ms() + 1_000)
        .await
        .unwrap();
    assert!(troupe.queue().claim_due(10, 500).await.unwrap().is_empty());
    clock.advance(999);
    assert!(troupe.queue().claim_due(10, 500).await.unwrap().is_empty());
    clock.advance(1);
    let leases = troupe.queue().claim_due(10, 500).await.unwrap();
    assert_eq!(leases.len(), 1);
    assert_eq!(leases[0].run_id, run);
}

#[tokio::test]
async fn clock_setting_drives_the_memory_queue() {
    let clock = Arc::new(FixedClock::new(5_000));
    let troupe = Troupe::builder()
        .clock(clock.clone())
        .build()
        .await
        .unwrap();
    clock_drives_queue(troupe, clock).await;
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn clock_setting_drives_the_sqlite_queue() {
    let dir = tempfile::tempdir().unwrap();
    let clock = Arc::new(FixedClock::new(5_000));
    let store = SqliteStore::builder(dir.path().join("clock.db"));
    let troupe = Troupe::builder()
        .sqlite_with(store)
        .clock(clock.clone())
        .build()
        .await
        .unwrap();
    clock_drives_queue(troupe, clock).await;
}
