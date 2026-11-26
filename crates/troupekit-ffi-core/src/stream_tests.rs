use crate::host::EventListener;
use crate::records::{FfiEvent, FfiEventKind, FfiPause, FfiUsage};
use crate::stream::{mirror, RunStream};
use crate::ErrorCode;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit_core::error::ProviderError;
use troupekit_core::event::{Event, EventBody, PauseReason, Usage};
use troupekit_core::ids::RunId;
use troupekit_graph::testing::text_reply;
use troupekit_kit::Agent;
use troupekit_providers::{CompletionRequest, CompletionResponse, Delta, Model};

struct Chunky {
    deltas: usize,
    pause: Duration,
}

#[async_trait]
impl Model for Chunky {
    fn id(&self) -> &str {
        "fake/m"
    }

    async fn complete(
        &self,
        _req: &CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        tokio::time::sleep(self.pause).await;
        text_reply(&"a".repeat(self.deltas), 10)
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let reply = self.complete(req).await?;
        for _ in 0..self.deltas {
            on_delta(&Delta::TextDelta { text: "a".into() });
        }
        Ok(reply)
    }
}

async fn stream_of(deltas: usize, pause: Duration) -> Arc<RunStream> {
    let built = Agent::new("chunky")
        .instructions("Say a.")
        .model("fake/m")
        .model_client(Arc::new(Chunky { deltas, pause }))
        .build()
        .await
        .unwrap();
    let kit = built.stream("hi").await.unwrap();
    RunStream::agent(kit)
}

#[derive(Default)]
struct Recorder {
    events: Mutex<Vec<FfiEvent>>,
    threads: Mutex<Vec<String>>,
}

impl EventListener for Recorder {
    fn on_event(&self, event: FfiEvent) {
        let name = std::thread::current().name().unwrap_or("").to_string();
        self.threads.lock().unwrap().push(name);
        self.events.lock().unwrap().push(event);
    }
}

fn kinds(events: &[FfiEvent]) -> Vec<FfiEventKind> {
    events.iter().map(|e| e.kind.clone()).collect()
}

async fn pull_all(stream: &RunStream) -> Vec<FfiEvent> {
    let mut events = Vec::new();
    while let Some(event) = stream.next().await.unwrap() {
        events.push(event);
        assert!(events.len() < 10_000);
    }
    events
}

#[tokio::test]
async fn pull_and_listener_see_same_events() {
    let pulled = pull_all(&*stream_of(4, Duration::ZERO).await).await;
    let recorder = Arc::new(Recorder::default());
    let stream = stream_of(4, Duration::ZERO).await;
    stream.set_listener(recorder.clone()).unwrap();
    let result = stream.result().await.unwrap();
    assert_eq!(result.text, "aaaa");
    for _ in 0..500 {
        let done = recorder
            .events
            .lock()
            .unwrap()
            .iter()
            .any(|e| matches!(e.kind, FfiEventKind::RunFinished { .. }));
        if done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let heard = recorder.events.lock().unwrap().clone();
    assert_eq!(kinds(&heard), kinds(&pulled));
    assert!(heard.iter().all(|e| e.run_id == stream.run_id()));
}

#[tokio::test]
async fn listener_called_off_runtime_threads() {
    let recorder = Arc::new(Recorder::default());
    let stream = stream_of(2, Duration::ZERO).await;
    stream.set_listener(recorder.clone()).unwrap();
    stream.result().await.unwrap();
    for _ in 0..500 {
        if !recorder.threads.lock().unwrap().is_empty() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    let threads = recorder.threads.lock().unwrap().clone();
    assert!(!threads.is_empty());
    assert!(
        threads.iter().all(|t| t == "troupekit-listener"),
        "{threads:?}"
    );
}

#[tokio::test]
async fn pull_and_listener_are_exclusive() {
    let stream = stream_of(1, Duration::ZERO).await;
    stream.next().await.unwrap();
    let err = stream
        .set_listener(Arc::new(Recorder::default()))
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    let other = stream_of(1, Duration::ZERO).await;
    other.set_listener(Arc::new(Recorder::default())).unwrap();
    let err = other.next().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
    let err = other
        .set_listener(Arc::new(Recorder::default()))
        .unwrap_err();
    assert_eq!(err.code, ErrorCode::InvalidArgument);
}

#[tokio::test]
async fn cancel_ends_stream_with_cancelled() {
    let stream = stream_of(1, Duration::from_secs(600)).await;
    let run_id = stream.run_id();
    stream.cancel();
    let rest = pull_all(&stream).await;
    assert!(rest.len() < 10);
    let err = stream.result().await.unwrap_err();
    assert_eq!(err.code, ErrorCode::Cancelled);
    assert_eq!(err.run_id, Some(run_id));
}

#[test]
fn mirror_covers_every_event_body() {
    let usage = Usage {
        input_tokens: 3,
        output_tokens: 4,
        cost_usd_micros: 5,
    };
    let mirrored = FfiUsage {
        input_tokens: 3,
        output_tokens: 4,
        cost_usd_micros: 5,
    };
    let cases = vec![
        (
            EventBody::RunStarted {
                workflow: "w".into(),
            },
            FfiEventKind::RunStarted {
                workflow: "w".into(),
            },
        ),
        (
            EventBody::StepStarted {
                step: "s".into(),
                persona: Some("p".into()),
            },
            FfiEventKind::StepStarted {
                step: "s".into(),
                persona: Some("p".into()),
            },
        ),
        (
            EventBody::TextDelta { text: "t".into() },
            FfiEventKind::TextDelta { text: "t".into() },
        ),
        (
            EventBody::ToolCallStarted {
                call_id: "c".into(),
                name: "n".into(),
            },
            FfiEventKind::ToolCallStarted {
                call_id: "c".into(),
                name: "n".into(),
            },
        ),
        (
            EventBody::ToolCallFinished {
                call_id: "c".into(),
                is_error: true,
            },
            FfiEventKind::ToolCallFinished {
                call_id: "c".into(),
                is_error: true,
            },
        ),
        (
            EventBody::StepFinished {
                step: "s".into(),
                usage,
            },
            FfiEventKind::StepFinished {
                step: "s".into(),
                usage: mirrored,
            },
        ),
        (
            EventBody::RunPaused {
                reason: PauseReason::Approval { node: "a".into() },
            },
            FfiEventKind::RunPaused {
                reason: FfiPause::Approval { node: "a".into() },
            },
        ),
        (
            EventBody::RunPaused {
                reason: PauseReason::Question { node: "q".into() },
            },
            FfiEventKind::RunPaused {
                reason: FfiPause::Question { node: "q".into() },
            },
        ),
        (
            EventBody::RunPaused {
                reason: PauseReason::Budget {
                    limit: "run".into(),
                },
            },
            FfiEventKind::RunPaused {
                reason: FfiPause::Budget {
                    limit: "run".into(),
                },
            },
        ),
        (
            EventBody::RunFinished { usage },
            FfiEventKind::RunFinished { usage: mirrored },
        ),
        (
            EventBody::RunFailed { error: "e".into() },
            FfiEventKind::RunFailed { error: "e".into() },
        ),
    ];
    let run_id = RunId::new();
    for (body, kind) in cases {
        let event = mirror(Event {
            run_id,
            seq: 7,
            at_ms: 9,
            body,
        });
        assert_eq!(event.kind, kind);
        assert_eq!(event.run_id, run_id.to_string());
        assert_eq!((event.seq, event.at_ms), (7, 9));
    }
}

#[derive(Default)]
struct PanicsOnce {
    calls: Mutex<u32>,
}

impl EventListener for PanicsOnce {
    fn on_event(&self, _event: FfiEvent) {
        *self.calls.lock().unwrap() += 1;
        panic!("host listener bug");
    }
}

#[tokio::test]
async fn listener_panic_stops_delivery_not_the_run() {
    let listener = Arc::new(PanicsOnce::default());
    let stream = stream_of(3, Duration::ZERO).await;
    stream.set_listener(listener.clone()).unwrap();
    let result = stream.result().await.unwrap();
    assert_eq!(result.text, "aaa");
    assert_eq!(*listener.calls.lock().unwrap(), 1);
}
