use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{Message, Role};
use troupekit_providers::catalog::Price;
use troupekit_providers::middleware::{
    FallbackModel, MeteredModel, RetryModel, RetryPolicy, Sleep,
};
use troupekit_providers::model::Model;
use troupekit_providers::request::{CompletionRequest, CompletionResponse, Delta, StopReason};

enum Step {
    Reply,
    Fail(ProviderError),
    DeltaThenFail(ProviderError),
}

struct Fake {
    id: String,
    script: Mutex<VecDeque<Step>>,
    calls: Arc<AtomicUsize>,
}

fn fake(id: &str, steps: Vec<Step>) -> (Box<dyn Model>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = Fake {
        id: id.into(),
        script: Mutex::new(steps.into()),
        calls: calls.clone(),
    };
    (Box::new(model), calls)
}

fn reply() -> CompletionResponse {
    CompletionResponse {
        message: Message::text(Role::Assistant, "ok"),
        usage: Usage {
            input_tokens: 1000,
            output_tokens: 2000,
            cost_usd_micros: 0,
        },
        stop: StopReason::EndTurn,
    }
}

impl Fake {
    fn next(&self) -> Step {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .expect("script ran out")
    }
}

#[async_trait]
impl Model for Fake {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, _: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        match self.next() {
            Step::Reply => Ok(reply()),
            Step::Fail(e) | Step::DeltaThenFail(e) => Err(e),
        }
    }

    async fn stream(
        &self,
        _: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let text = Delta::TextDelta { text: "ok".into() };
        match self.next() {
            Step::Reply => {
                on_delta(&text);
                Ok(reply())
            }
            Step::Fail(e) => Err(e),
            Step::DeltaThenFail(e) => {
                on_delta(&text);
                Err(e)
            }
        }
    }
}

#[derive(Default)]
struct Recorder(Mutex<Vec<u64>>);

#[async_trait]
impl Sleep for Recorder {
    async fn sleep(&self, ms: u64) {
        self.0.lock().unwrap().push(ms);
    }
}

fn err(kind: ProviderErrorKind) -> ProviderError {
    ProviderError::new(kind, "x")
}

fn policy() -> RetryPolicy {
    RetryPolicy {
        max_attempts: 3,
        base_delay_ms: 100,
        max_delay_ms: 5000,
    }
}

fn retrying(inner: Box<dyn Model>, policy: RetryPolicy) -> (RetryModel, Arc<Recorder>) {
    let sleeps = Arc::new(Recorder::default());
    (
        RetryModel::new(inner, policy)
            .unwrap()
            .with_sleep(sleeps.clone()),
        sleeps,
    )
}

fn req() -> CompletionRequest {
    CompletionRequest::user("hi", 10)
}

#[tokio::test]
async fn retries_retryable_errors_with_backoff_and_server_delays() {
    let limited = err(ProviderErrorKind::RateLimited).with_retry_after(1200);
    let (inner, calls) = fake(
        "a",
        vec![
            Step::Fail(err(ProviderErrorKind::Server)),
            Step::Fail(limited),
            Step::Reply,
        ],
    );
    let (model, sleeps) = retrying(inner, policy());
    assert_eq!(model.complete(&req()).await.unwrap(), reply());
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(*sleeps.0.lock().unwrap(), vec![100, 1200]);
    assert_eq!(model.id(), "a");
}

#[tokio::test]
async fn stops_at_max_attempts_and_on_permanent_errors() {
    let server = || Step::Fail(err(ProviderErrorKind::Server));
    let (inner, calls) = fake("a", vec![server(), server(), server(), Step::Reply]);
    let (model, sleeps) = retrying(inner, policy());
    assert_eq!(
        model.complete(&req()).await.unwrap_err().kind,
        ProviderErrorKind::Server
    );
    assert_eq!(calls.load(Ordering::SeqCst), 3);
    assert_eq!(*sleeps.0.lock().unwrap(), vec![100, 200]);

    let (inner, calls) = fake(
        "a",
        vec![Step::Fail(err(ProviderErrorKind::Auth)), Step::Reply],
    );
    let (model, sleeps) = retrying(inner, policy());
    assert_eq!(
        model.complete(&req()).await.unwrap_err().kind,
        ProviderErrorKind::Auth
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(sleeps.0.lock().unwrap().is_empty());

    let too_long = err(ProviderErrorKind::RateLimited).with_retry_after(60_000);
    let (inner, calls) = fake("a", vec![Step::Fail(too_long), Step::Reply]);
    let (model, _) = retrying(inner, policy());
    assert_eq!(
        model.complete(&req()).await.unwrap_err().retry_after_ms,
        Some(60_000)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn stream_retries_only_before_the_first_delta() {
    let (inner, calls) = fake(
        "a",
        vec![Step::Fail(err(ProviderErrorKind::Network)), Step::Reply],
    );
    let (model, _) = retrying(inner, policy());
    let mut seen = 0;
    assert!(model
        .stream(&req(), &mut |_: &Delta| seen += 1)
        .await
        .is_ok());
    assert_eq!((calls.load(Ordering::SeqCst), seen), (2, 1));

    let (inner, calls) = fake(
        "a",
        vec![
            Step::DeltaThenFail(err(ProviderErrorKind::Network)),
            Step::Reply,
        ],
    );
    let (model, _) = retrying(inner, policy());
    let mut seen = 0;
    assert_eq!(
        model
            .stream(&req(), &mut |_: &Delta| seen += 1)
            .await
            .unwrap_err()
            .kind,
        ProviderErrorKind::Network
    );
    assert_eq!((calls.load(Ordering::SeqCst), seen), (1, 1));
}

#[test]
fn checks_policies_and_caps_backoff() {
    for bad in [
        RetryPolicy {
            max_attempts: 0,
            ..policy()
        },
        RetryPolicy {
            max_attempts: 11,
            ..policy()
        },
        RetryPolicy {
            base_delay_ms: 6000,
            ..policy()
        },
    ] {
        let (inner, _) = fake("a", vec![]);
        assert_eq!(
            RetryModel::new(inner, bad).err().unwrap().kind,
            ProviderErrorKind::BadRequest
        );
    }
    let wide = RetryPolicy {
        max_attempts: 100,
        ..policy()
    };
    assert_eq!(
        wide.next_delay_ms(7, &err(ProviderErrorKind::Server)),
        Some(5000)
    );
    assert_eq!(
        wide.next_delay_ms(90, &err(ProviderErrorKind::Server)),
        Some(5000)
    );
    assert_eq!(wide.next_delay_ms(1, &err(ProviderErrorKind::Decode)), None);
}

#[test]
#[should_panic(expected = "before any failure")]
fn a_delay_before_any_failure_is_a_bug() {
    let _ = policy().next_delay_ms(0, &err(ProviderErrorKind::Server));
}

#[tokio::test]
async fn fallback_tries_models_in_order() {
    let (a, a_calls) = fake(
        "a",
        vec![
            Step::Fail(err(ProviderErrorKind::Auth)),
            Step::Fail(err(ProviderErrorKind::Server)),
        ],
    );
    let (b, b_calls) = fake(
        "b",
        vec![Step::Reply, Step::Fail(err(ProviderErrorKind::RateLimited))],
    );
    let model = FallbackModel::new(a, vec![b]);
    assert_eq!(model.id(), "a,b");
    assert_eq!(model.complete(&req()).await.unwrap(), reply());
    assert_eq!(
        model.complete(&req()).await.unwrap_err().kind,
        ProviderErrorKind::RateLimited
    );
    assert_eq!(
        (
            a_calls.load(Ordering::SeqCst),
            b_calls.load(Ordering::SeqCst)
        ),
        (2, 2)
    );
}

#[tokio::test]
async fn fallback_stream_stops_after_a_delta() {
    let (a, _) = fake(
        "a",
        vec![Step::DeltaThenFail(err(ProviderErrorKind::Server))],
    );
    let (b, b_calls) = fake("b", vec![Step::Reply]);
    let model = FallbackModel::new(a, vec![b]);
    let mut sink = |_: &Delta| {};
    assert_eq!(
        model.stream(&req(), &mut sink).await.unwrap_err().kind,
        ProviderErrorKind::Server
    );
    assert_eq!(b_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn meter_fills_in_the_cost() {
    let price = Price {
        input_usd_micros_per_mtok: 3_000_000,
        output_usd_micros_per_mtok: 15_000_000,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    };
    let (inner, _) = fake("a", vec![Step::Reply, Step::Reply]);
    let model = MeteredModel::new(inner, price);
    assert_eq!(model.price(), price);
    assert_eq!(
        model.complete(&req()).await.unwrap().usage.cost_usd_micros,
        33_000
    );
    let mut sink = |_: &Delta| {};
    assert_eq!(
        model
            .stream(&req(), &mut sink)
            .await
            .unwrap()
            .usage
            .cost_usd_micros,
        33_000
    );
}
