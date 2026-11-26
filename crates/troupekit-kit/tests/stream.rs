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
    let started = events.iter().any(|e| {
        matches!(&e.body, EventBody::RunStarted { workflow } if workflow == "pipeline")
    });
    assert!(started, "{events:?}");
    let agent_step = events.iter().any(|e| {
        matches!(&e.body, EventBody::StepStarted { persona: Some(p), .. } if p == "drafter")
    });
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
