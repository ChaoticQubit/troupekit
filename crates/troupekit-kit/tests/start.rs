use crate::common::*;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit::{Agent, KitError, Progress, Storage};
use troupekit_core::error::{CoreError, ProviderError, ProviderErrorKind};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::{Checkpoint, CheckpointStore};
use troupekit_graph::testing::{text_reply, FnModel};

const LONG: Duration = Duration::from_secs(60);

#[derive(Default)]
struct Recording {
    inner: InMemCheckpoints,
    steps: Mutex<Vec<u64>>,
}

#[async_trait]
impl CheckpointStore for Recording {
    async fn save(&self, cp: Checkpoint) -> Result<(), CoreError> {
        let step = cp.step;
        self.inner.save(cp).await?;
        self.steps.lock().unwrap().push(step);
        Ok(())
    }

    async fn latest(&self, run_id: RunId) -> Result<Option<Checkpoint>, CoreError> {
        self.inner.latest(run_id).await
    }
}

fn counting_model() -> Arc<FnModel> {
    Arc::new(FnModel::new("fake/m", |_req| text_reply("counted", 10)))
}

#[tokio::test]
async fn start_returns_id_before_any_model_call() {
    let model = counting_model();
    let store = Arc::new(InMemCheckpoints::default());
    let built = Agent::new("counter")
        .instructions("Answer.")
        .model("fake/m")
        .model_client(model.clone())
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    let run_id = built.start("hello").await.unwrap();
    assert_eq!(model.calls(), 0);
    assert_eq!(store.latest(run_id).await.unwrap().unwrap().step, 0);
    assert_eq!(built.resume(run_id).await.unwrap(), "counted");
    assert_eq!(model.calls(), 1);
}

#[tokio::test]
async fn start_needs_durable_storage() {
    let built = greeter().build().await.unwrap();
    let err = built.start("hello").await.unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
    assert_eq!(
        err.to_string(),
        "agent `greeter`: start needs Storage::sqlite(..) or Storage::custom(..) checkpoints"
    );
    let err = greeter().start("hello").await.unwrap_err();
    assert!(matches!(err, KitError::Config { .. }), "{err:?}");
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn started_run_resumes_in_new_process_shape() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("runs.db");
    let expected = greeter().run("I am Ada").await.unwrap();
    let (first, run_id) = greeter()
        .checkpoints(Storage::sqlite(&path))
        .start("I am Ada")
        .await
        .unwrap();
    drop(first);
    let second = greeter()
        .checkpoints(Storage::sqlite(&path))
        .build()
        .await
        .unwrap();
    assert_eq!(second.resume(run_id).await.unwrap(), expected);
}

#[tokio::test]
async fn resume_for_zero_budget_is_pending() {
    let model = counting_model();
    let store = Arc::new(Recording::default());
    let built = Agent::new("counter")
        .instructions("Answer.")
        .model("fake/m")
        .model_client(model.clone())
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    let run_id = built.start("hello").await.unwrap();
    let progress = built.resume_for(run_id, Duration::ZERO).await.unwrap();
    assert_eq!(progress, Progress::Pending { run_id });
    assert_eq!(*store.steps.lock().unwrap(), vec![0]);
    assert_eq!(model.calls(), 0);
}

#[tokio::test]
async fn resume_for_large_budget_is_done() {
    let built = greeter()
        .checkpoints(Storage::custom(Arc::new(InMemCheckpoints::default())))
        .build()
        .await
        .unwrap();
    let run_id = built.start("I am Ada").await.unwrap();
    let progress = built.resume_for(run_id, LONG).await.unwrap();
    assert_eq!(progress, Progress::Done("Hello! You said: I am Ada".into()));
    let again = built.resume_for(run_id, Duration::ZERO).await.unwrap();
    assert_eq!(again, Progress::Done("Hello! You said: I am Ada".into()));
}

#[tokio::test]
async fn resume_for_saves_after_every_step() {
    let store = Arc::new(Recording::default());
    let built = Agent::new("calc")
        .instructions("Use the add tool.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(add_tool())
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    let run_id = built.start("What is 2 + 40?").await.unwrap();
    let progress = built.resume_for(run_id, LONG).await.unwrap();
    assert_eq!(progress, Progress::Done("The sum is 42".into()));
    let steps = store.steps.lock().unwrap().clone();
    assert!(steps.len() >= 3, "{steps:?}");
    let expected: Vec<u64> = (0..steps.len() as u64).collect();
    assert_eq!(steps, expected);
}

#[test]
fn defaults_are_the_safe_values() {
    assert_eq!(troupekit::MAX_STEPS_PER_WINDOW, 10_000);
    assert_eq!(troupekit::MAX_PERSONA_BYTES, 1024 * 1024);
}

#[tokio::test]
async fn retry_cap_holds_across_resume_for_windows() {
    let model = Arc::new(FnModel::new("fake/m", |_req| {
        Err(
            ProviderError::new(ProviderErrorKind::RateLimited, "slow down")
                .with_retry_after(20_000),
        )
    }));
    let built = Agent::new("limited")
        .instructions("Answer.")
        .model("fake/m")
        .model_client(model.clone())
        .retries(2)
        .checkpoints(Storage::custom(Arc::new(InMemCheckpoints::default())))
        .build()
        .await
        .unwrap();
    let run_id = built.start("hello").await.unwrap();
    let window = Duration::from_secs(1);
    for _ in 0..2 {
        let progress = built.resume_for(run_id, window).await.unwrap();
        assert_eq!(progress, Progress::Pending { run_id });
    }
    let err = built.resume_for(run_id, window).await.unwrap_err();
    assert!(
        matches!(err, KitError::Retries { attempts: 2, .. }),
        "{err:?}"
    );
    assert_eq!(model.calls(), 3);
}

#[tokio::test]
async fn resume_keeps_retries_counted_by_resume_for() {
    let model = Arc::new(FnModel::new("fake/m", |_req| {
        Err(
            ProviderError::new(ProviderErrorKind::RateLimited, "slow down")
                .with_retry_after(20_000),
        )
    }));
    let built = Agent::new("limited")
        .instructions("Answer.")
        .model("fake/m")
        .model_client(model.clone())
        .retries(1)
        .checkpoints(Storage::custom(Arc::new(InMemCheckpoints::default())))
        .build()
        .await
        .unwrap();
    let run_id = built.start("hello").await.unwrap();
    let progress = built
        .resume_for(run_id, Duration::from_secs(1))
        .await
        .unwrap();
    assert_eq!(progress, Progress::Pending { run_id });
    let err = built.resume(run_id).await.unwrap_err();
    assert!(
        matches!(err, KitError::Retries { attempts: 1, .. }),
        "{err:?}"
    );
    assert_eq!(model.calls(), 2);
}
