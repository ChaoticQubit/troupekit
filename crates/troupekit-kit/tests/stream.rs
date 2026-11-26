use crate::common::*;
use std::sync::Arc;
use std::time::Duration;
use troupekit::{tool, Agent, Graph, Progress, Storage};
use troupekit_agent::VecSink;
use troupekit_core::event::EventBody;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::CheckpointStore;

/// Upper-cases the draft.
#[tool]
fn upper_draft(draft: String) -> String {
    draft.to_uppercase()
}

fn drafter() -> Agent {
    Agent::new("drafter")
        .instructions("Draft.")
        .model("fake/m")
        .model_client(echo_model())
}

#[tokio::test]
async fn graph_events_reach_sink() {
    let sink = Arc::new(VecSink::default());
    Graph::new("pipeline")
        .agent("draft", drafter())
        .step("upper", upper_draft)
        .events(sink.clone())
        .run("hi")
        .await
        .unwrap();
    let events = sink.events();
    let started = events
        .iter()
        .any(|e| matches!(&e.body, EventBody::RunStarted { workflow } if workflow == "pipeline"));
    assert!(started, "{events:?}");
    let agent_step = events.iter().any(
        |e| matches!(&e.body, EventBody::StepStarted { persona: Some(p), .. } if p == "drafter"),
    );
    assert!(agent_step, "{events:?}");
    let deltas = events
        .iter()
        .filter(|e| matches!(e.body, EventBody::TextDelta { .. }))
        .count();
    assert!(deltas >= 1, "{events:?}");
    assert!(events
        .iter()
        .any(|e| matches!(e.body, EventBody::RunFinished { .. })));
}

#[tokio::test]
async fn graph_start_then_resume_for() {
    let store = Arc::new(InMemCheckpoints::default());
    let built = Graph::new("pipeline")
        .agent("draft", drafter())
        .step("upper", upper_draft)
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    let run_id = built.start("hi").await.unwrap();
    assert_eq!(store.latest(run_id).await.unwrap().unwrap().step, 0);
    let pending = built.resume_for(run_id, Duration::ZERO).await.unwrap();
    assert_eq!(pending, Progress::Pending { run_id });
    assert_eq!(store.latest(run_id).await.unwrap().unwrap().step, 0);
    let done = built
        .resume_for(run_id, Duration::from_secs(60))
        .await
        .unwrap();
    let Progress::Done(text) = done else {
        panic!("expected Done, got {done:?}");
    };
    assert!(text.starts_with("HELLO! YOU SAID:"), "{text}");
    let in_memory = Graph::new("pipeline")
        .agent("draft", drafter())
        .step("upper", upper_draft)
        .build()
        .await
        .unwrap();
    assert!(in_memory.start("hi").await.is_err());
}

struct Chunky {
    deltas: usize,
    pause: Duration,
}

#[async_trait::async_trait]
impl troupekit_providers::Model for Chunky {
    fn id(&self) -> &str {
        "fake/m"
    }

    async fn complete(
        &self,
        _req: &troupekit_providers::CompletionRequest,
    ) -> Result<troupekit_providers::CompletionResponse, troupekit_core::error::ProviderError> {
        tokio::time::sleep(self.pause).await;
        troupekit_graph::testing::text_reply(&"a".repeat(self.deltas), 10)
    }

    async fn stream(
        &self,
        req: &troupekit_providers::CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d troupekit_providers::Delta) + Send),
    ) -> Result<troupekit_providers::CompletionResponse, troupekit_core::error::ProviderError> {
        let reply = self.complete(req).await?;
        for _ in 0..self.deltas {
            on_delta(&troupekit_providers::Delta::TextDelta { text: "a".into() });
        }
        Ok(reply)
    }
}

fn chunky(deltas: usize, pause: Duration) -> Agent {
    Agent::new("chunky")
        .instructions("Say a.")
        .model("fake/m")
        .model_client(Arc::new(Chunky { deltas, pause }))
}

async fn drain<T>(stream: &mut troupekit::RunStream<T>) -> Vec<troupekit_core::event::Event> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await {
        events.push(event);
        assert!(events.len() < 10_000, "stream did not end");
    }
    events
}

#[tokio::test]
async fn stream_yields_deltas_then_finished() {
    let built = chunky(3, Duration::ZERO).build().await.unwrap();
    let mut stream = built.stream("hi").await.unwrap();
    let run_id = stream.run_id();
    let events = drain(&mut stream).await;
    assert!(
        matches!(events.first().unwrap().body, EventBody::RunStarted { .. }),
        "{events:?}"
    );
    assert!(
        matches!(events.last().unwrap().body, EventBody::RunFinished { .. }),
        "{events:?}"
    );
    let deltas = events
        .iter()
        .filter(|e| matches!(e.body, EventBody::TextDelta { .. }))
        .count();
    assert_eq!(deltas, 3);
    assert!(events.windows(2).all(|w| w[0].seq < w[1].seq));
    assert!(events.iter().all(|e| e.run_id == run_id));
    assert_eq!(stream.dropped_deltas(), 0);
}

#[tokio::test]
async fn stream_finish_matches_run() {
    let built = chunky(3, Duration::ZERO).build().await.unwrap();
    let ran = built.run("hi").await.unwrap();
    let stream = built.stream("hi").await.unwrap();
    let run_id = stream.run_id();
    let outcome = stream.finish().await.unwrap();
    assert_eq!(outcome.text(), ran);
    assert_eq!(outcome.run_id(), run_id);
}

#[tokio::test]
async fn full_buffer_drops_deltas_not_lifecycle() {
    let sink = Arc::new(VecSink::default());
    let built = chunky(100, Duration::ZERO)
        .events(sink.clone())
        .build()
        .await
        .unwrap();
    let mut settings = troupekit::StreamSettings::default();
    settings.capacity = 16;
    let mut stream = built.stream_with("hi", settings).await.unwrap();
    for _ in 0..500 {
        let done = sink
            .events()
            .iter()
            .any(|e| matches!(e.body, EventBody::RunFinished { .. }));
        if done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let events = drain(&mut stream).await;
    let deltas = events
        .iter()
        .filter(|e| matches!(e.body, EventBody::TextDelta { .. }))
        .count() as u64;
    assert!(deltas < 100, "{deltas}");
    assert_eq!(deltas + stream.dropped_deltas(), 100);
    assert!(matches!(
        events.last().unwrap().body,
        EventBody::RunFinished { .. }
    ));
    assert!(events
        .iter()
        .any(|e| matches!(e.body, EventBody::RunStarted { .. })));
}

#[tokio::test]
async fn cancel_keeps_last_checkpoint() {
    let store = Arc::new(InMemCheckpoints::default());
    let slow = chunky(1, Duration::from_secs(600))
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    let mut stream = slow.stream("hi").await.unwrap();
    let run_id = stream.run_id();
    while let Some(event) = stream.next().await {
        if matches!(event.body, EventBody::StepStarted { .. }) {
            break;
        }
    }
    stream.cancel();
    assert!(drain(&mut stream).await.len() < 10);
    let err = stream.finish().await.unwrap_err();
    assert!(
        matches!(err, troupekit::KitError::Cancelled { run_id: id } if id == run_id),
        "{err:?}"
    );
    assert!(store.latest(run_id).await.unwrap().is_some());
    let fast = chunky(1, Duration::ZERO)
        .checkpoints(Storage::custom(store.clone()))
        .build()
        .await
        .unwrap();
    assert_eq!(fast.resume(run_id).await.unwrap(), "a");
}

#[tokio::test]
async fn user_sink_still_gets_every_event() {
    let sink = Arc::new(VecSink::default());
    let built = chunky(5, Duration::ZERO)
        .events(sink.clone())
        .build()
        .await
        .unwrap();
    let mut stream = built.stream("hi").await.unwrap();
    let streamed = drain(&mut stream).await;
    let seen: Vec<_> = sink
        .events()
        .into_iter()
        .filter(|e| e.run_id == stream.run_id())
        .collect();
    assert_eq!(seen, streamed);
}

#[tokio::test]
async fn stream_settings_out_of_range_refused() {
    let built = chunky(1, Duration::ZERO).build().await.unwrap();
    for capacity in [15, 1_048_577] {
        let mut settings = troupekit::StreamSettings::default();
        settings.capacity = capacity;
        let err = built.stream_with("hi", settings).await.unwrap_err();
        assert!(matches!(err, troupekit::KitError::Config { .. }), "{err:?}");
        assert!(err.to_string().contains("capacity"), "{err}");
    }
    let defaults = troupekit::StreamSettings::default();
    assert_eq!(defaults.capacity, 4096);
    assert_eq!(troupekit::StreamSettings::MIN_CAPACITY, 16);
    assert_eq!(troupekit::StreamSettings::MAX_CAPACITY, 1_048_576);
}

#[tokio::test]
async fn agent_stream_builds_then_streams() {
    let mut stream = chunky(2, Duration::ZERO).stream("hi").await.unwrap();
    let events = drain(&mut stream).await;
    assert!(matches!(
        events.last().unwrap().body,
        EventBody::RunFinished { .. }
    ));
}

fn pipeline() -> Graph {
    Graph::new("pipeline")
        .agent("draft", drafter())
        .step("upper", upper_draft)
}

#[tokio::test]
async fn graph_stream_yields_node_events_then_finished() {
    let built = pipeline().build().await.unwrap();
    let ran = built.run("hi").await.unwrap();
    let mut stream = built.stream("hi").await.unwrap();
    let run_id = stream.run_id();
    let events = drain(&mut stream).await;
    let first = events.first().unwrap();
    assert_eq!(first.run_id, run_id);
    assert!(
        matches!(&first.body, EventBody::RunStarted { workflow } if workflow == "pipeline"),
        "{events:?}"
    );
    assert!(events
        .iter()
        .any(|e| matches!(e.body, EventBody::TextDelta { .. })));
    let last = events.last().unwrap();
    assert_eq!(last.run_id, run_id);
    assert!(
        matches!(last.body, EventBody::RunFinished { .. }),
        "{events:?}"
    );
    let report = stream.finish().await.unwrap();
    assert_eq!(report.output(), &ran);
    assert_eq!(report.run_id(), run_id);
}

#[tokio::test]
async fn graph_stream_builds_then_streams() {
    let mut stream = pipeline().stream("hi").await.unwrap();
    let events = drain(&mut stream).await;
    assert!(matches!(
        events.last().unwrap().body,
        EventBody::RunFinished { .. }
    ));
}

#[tokio::test]
async fn cancel_wakes_a_pending_next_on_another_task() {
    let built = chunky(1, Duration::from_secs(600)).build().await.unwrap();
    let stream = Arc::new(built.stream("hi").await.unwrap());
    let reader = stream.clone();
    let pending = tokio::spawn(async move {
        let mut seen = 0usize;
        while reader.next().await.is_some() {
            seen += 1;
            assert!(seen < 100);
        }
        seen
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    stream.cancel();
    let seen = tokio::time::timeout(Duration::from_secs(10), pending)
        .await
        .expect("next ended after cancel")
        .unwrap();
    assert!(seen < 10);
    let err = stream.finish().await.unwrap_err();
    assert!(
        matches!(err, troupekit::KitError::Cancelled { .. }),
        "{err:?}"
    );
}

#[tokio::test]
async fn finish_twice_is_refused() {
    let built = chunky(1, Duration::ZERO).build().await.unwrap();
    let stream = built.stream("hi").await.unwrap();
    stream.finish().await.unwrap();
    let err = stream.finish().await.unwrap_err();
    assert!(matches!(err, troupekit::KitError::Config { .. }), "{err:?}");
    assert!(err.to_string().contains("finish"), "{err}");
}

#[tokio::test]
async fn graph_stream_with_checks_capacity() {
    let mut settings = troupekit::StreamSettings::default();
    settings.capacity = 16;
    let built = pipeline().build().await.unwrap();
    let stream = built.stream_with("hi", settings).await.unwrap();
    assert!(stream.finish().await.is_ok());
    let mut bad = troupekit::StreamSettings::default();
    bad.capacity = 15;
    let err = built.stream_with("hi", bad).await.unwrap_err();
    assert!(err.to_string().contains("capacity"), "{err}");
}
