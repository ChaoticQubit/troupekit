use crate::error::{ErrorCode, FfiError};
use crate::host::EventListener;
use crate::records::{FfiEvent, FfiEventKind, FfiPause, FfiUsage, RunResult};
use crate::runtime;
use async_trait::async_trait;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex, PoisonError};
use troupekit_core::event::{Event, EventBody, PauseReason, Usage};
use troupekit_kit::RunOutcome;

/// A run in progress that yields its events: pull them with `next`, or
/// hand them to an `EventListener`, not both. The buffer drops text
/// deltas when full and never lifecycle events.
pub struct RunStream {
    run_id: String,
    inner: Arc<dyn KitStream>,
    mode: Mutex<Mode>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Idle,
    Pulling,
    Listening,
}

impl std::fmt::Debug for RunStream {
    /// Writes the run id and the delivery mode.
    /// The queued events are not shown.
    /// Output: the debug text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStream")
            .field("run_id", &self.run_id)
            .field(
                "mode",
                &*self.mode.lock().unwrap_or_else(PoisonError::into_inner),
            )
            .finish_non_exhaustive()
    }
}

impl RunStream {
    pub(crate) fn agent(stream: troupekit_kit::RunStream<RunOutcome>) -> Arc<RunStream> {
        RunStream::wrap(stream.run_id().to_string(), Arc::new(stream))
    }

    fn wrap(run_id: String, inner: Arc<dyn KitStream>) -> Arc<RunStream> {
        assert!(!run_id.is_empty());
        Arc::new(RunStream {
            run_id,
            inner,
            mode: Mutex::new(Mode::Idle),
        })
    }

    fn enter(&self, wanted: Mode) -> Result<(), FfiError> {
        let mut mode = self.mode.lock().unwrap_or_else(PoisonError::into_inner);
        match (*mode, wanted) {
            (Mode::Idle, _) | (Mode::Pulling, Mode::Pulling) => {
                *mode = wanted;
                Ok(())
            }
            (Mode::Listening, Mode::Listening) => Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "a listener is already set on this stream",
            )),
            (Mode::Listening, _) => Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "this stream delivers to its listener; next is not available",
            )),
            (Mode::Pulling, _) => Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "next was already called on this stream; set_listener is not available",
            )),
        }
    }

    /// Waits for the next event of the run.
    /// Output: the event, `None` once the run has finished, failed,
    /// paused or been cancelled, or `InvalidArgument` with a listener set.
    pub async fn next(&self) -> Result<Option<FfiEvent>, FfiError> {
        self.enter(Mode::Pulling)?;
        Ok(self.inner.next().await.map(mirror))
    }

    /// Sends every remaining event to `listener` on its own thread,
    /// `troupekit-listener`, instead of `next`. The run goes on if the host
    /// drops the stream. Output: `Ok`, or `InvalidArgument` after `next`.
    pub fn set_listener(&self, listener: Arc<dyn EventListener>) -> Result<(), FfiError> {
        self.enter(Mode::Listening)?;
        let handle = runtime::handle()?;
        let inner = self.inner.clone();
        let started = std::thread::Builder::new()
            .name("troupekit-listener".into())
            .spawn(move || {
                while let Some(event) = handle.block_on(inner.next()) {
                    let event = mirror(event);
                    if catch_unwind(AssertUnwindSafe(|| listener.on_event(event))).is_err() {
                        break;
                    }
                }
            });
        match started {
            Ok(_) => Ok(()),
            Err(e) => Err(FfiError::new(
                ErrorCode::Internal,
                format!("the listener thread did not start: {e}"),
            )),
        }
    }

    /// Stops the run at its next await point, which may be inside a model
    /// call. The last saved checkpoint stays, so the run can be resumed by
    /// id. Calling it twice, or after the end, does nothing.
    pub fn cancel(&self) {
        self.inner.cancel();
    }

    /// Returns the run id text, known before the first event.
    /// Store it to resume the run after the app restarts.
    /// Output: the id.
    pub fn run_id(&self) -> String {
        self.run_id.clone()
    }

    /// Returns how many text deltas were dropped because the buffer
    /// was full. The final text in `result` is complete either way.
    /// Output: the count.
    pub fn dropped_deltas(&self) -> u64 {
        self.inner.dropped()
    }

    /// Waits for the run to end and returns its result, once.
    /// Output: the result, the run's error with its code, `Cancelled`
    /// after `cancel`, or `Config` on a second call.
    pub async fn result(&self) -> Result<RunResult, FfiError> {
        let inner = self.inner.clone();
        runtime::spawn(async move { inner.result().await }).await
    }
}

#[async_trait]
trait KitStream: Send + Sync + 'static {
    async fn next(&self) -> Option<Event>;
    fn cancel(&self);
    fn dropped(&self) -> u64;
    async fn result(&self) -> Result<RunResult, FfiError>;
}

#[async_trait]
impl KitStream for troupekit_kit::RunStream<RunOutcome> {
    async fn next(&self) -> Option<Event> {
        troupekit_kit::RunStream::next(self).await
    }

    fn cancel(&self) {
        troupekit_kit::RunStream::cancel(self);
    }

    fn dropped(&self) -> u64 {
        self.dropped_deltas()
    }

    async fn result(&self) -> Result<RunResult, FfiError> {
        let outcome = self.finish().await?;
        Ok(RunResult::from_outcome(&outcome))
    }
}

/// Copies one core event into its neutral mirror. The match on
/// `EventBody` and `PauseReason` has no fallback arm, so a new core
/// variant fails to compile here until it has a neutral form.
pub(crate) fn mirror(event: Event) -> FfiEvent {
    let kind = match event.body {
        EventBody::RunStarted { workflow } => FfiEventKind::RunStarted { workflow },
        EventBody::StepStarted { step, persona } => FfiEventKind::StepStarted { step, persona },
        EventBody::TextDelta { text } => FfiEventKind::TextDelta { text },
        EventBody::ToolCallStarted { call_id, name } => {
            FfiEventKind::ToolCallStarted { call_id, name }
        }
        EventBody::ToolCallFinished { call_id, is_error } => {
            FfiEventKind::ToolCallFinished { call_id, is_error }
        }
        EventBody::StepFinished { step, usage } => FfiEventKind::StepFinished {
            step,
            usage: usage_of(usage),
        },
        EventBody::RunPaused { reason } => FfiEventKind::RunPaused {
            reason: match reason {
                PauseReason::Approval { node } => FfiPause::Approval { node },
                PauseReason::Question { node } => FfiPause::Question { node },
                PauseReason::Budget { limit } => FfiPause::Budget { limit },
            },
        },
        EventBody::RunFinished { usage } => FfiEventKind::RunFinished {
            usage: usage_of(usage),
        },
        EventBody::RunFailed { error } => FfiEventKind::RunFailed { error },
    };
    FfiEvent {
        run_id: event.run_id.to_string(),
        seq: event.seq,
        at_ms: event.at_ms,
        kind,
    }
}

pub(crate) fn usage_of(usage: Usage) -> FfiUsage {
    FfiUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        cost_usd_micros: usage.cost_usd_micros,
    }
}
