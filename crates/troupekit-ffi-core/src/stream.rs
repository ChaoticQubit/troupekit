use crate::error::FfiError;
use crate::host::EventListener;
use crate::records::{FfiEvent, RunResult};
use std::sync::Arc;

/// A run in progress that yields its events: pull them with `next`, or
/// hand them to an `EventListener`, not both. The buffer drops text
/// deltas when full and never lifecycle events.
#[derive(Debug)]
pub struct RunStream {
    run_id: String,
}

impl RunStream {
    pub(crate) fn unsupported() -> Arc<RunStream> {
        Arc::new(RunStream {
            run_id: String::new(),
        })
    }

    /// Waits for the next event of the run.
    /// Output: the event, `None` once the run has finished, failed,
    /// paused or been cancelled, or `InvalidArgument` with a listener set.
    pub async fn next(&self) -> Result<Option<FfiEvent>, FfiError> {
        Err(FfiError::unsupported("RunStream::next"))
    }

    /// Sends every remaining event to `listener` instead of `next`.
    /// Output: `Ok`, or `InvalidArgument` when `next` was already
    /// called or a listener is already set.
    pub fn set_listener(&self, _listener: Arc<dyn EventListener>) -> Result<(), FfiError> {
        Err(FfiError::unsupported("RunStream::set_listener"))
    }

    /// Stops the run after its current step. The last saved checkpoint
    /// stays, so the run can be resumed by id. Calling it twice, or
    /// after the end, does nothing.
    pub fn cancel(&self) {}

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
        0
    }

    /// Waits for the run to end and returns its result.
    /// Output: the result, or the run's error with its code; `Cancelled`
    /// after `cancel`.
    pub async fn result(&self) -> Result<RunResult, FfiError> {
        Err(FfiError::unsupported("RunStream::result"))
    }
}
