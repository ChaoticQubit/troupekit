use crate::error::{ErrorCode, FfiError};
use std::any::Any;
use std::future::Future;
use std::ops::RangeInclusive;
use std::sync::OnceLock;
use tokio::runtime::Runtime;

const WORKER_THREADS: RangeInclusive<usize> = 1..=64;
const MAX_INPUT_BYTES: RangeInclusive<usize> = 1024..=64 * 1024 * 1024;
const EVENT_BUFFER: RangeInclusive<usize> = 16..=1_048_576;

static SETTINGS: OnceLock<RuntimeSettings> = OnceLock::new();
static RUNTIME: OnceLock<Result<Runtime, String>> = OnceLock::new();

/// Process-wide settings of the neutral runtime: worker threads (1 to
/// 64, default 2), the largest caller string in bytes (1 KiB to 64 MiB,
/// default 1 MiB) and the stream event buffer (16 to 1 048 576, 4096).
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuntimeSettings {
    pub worker_threads: usize,
    pub max_input_bytes: usize,
    pub event_buffer: usize,
}

impl Default for RuntimeSettings {
    /// Returns the safe defaults: 2 worker threads, 1 MiB caller strings
    /// and a 4096-event stream buffer. Start from it and change fields
    /// before calling `init`.
    fn default() -> Self {
        RuntimeSettings {
            worker_threads: 2,
            max_input_bytes: 1024 * 1024,
            event_buffer: 4096,
        }
    }
}

/// Sets the runtime settings once, before the first agent or graph call.
/// Input: the settings. Output: `Ok`, also for a repeat with the same
/// settings, or `InvalidArgument` for a value out of range or other settings.
pub fn init(settings: RuntimeSettings) -> Result<(), FfiError> {
    check(&settings)?;
    let current = SETTINGS.get_or_init(|| settings.clone());
    if *current != settings {
        return Err(FfiError::new(
            ErrorCode::InvalidArgument,
            format!("the runtime is already set up with other settings: {current:?}"),
        ));
    }
    assert!(check(current).is_ok());
    Ok(())
}

/// Returns the settings in force: those of `init`, or the defaults
/// when the first call came before any `init`. From then on they
/// never change. Output: the settings.
pub(crate) fn settings() -> &'static RuntimeSettings {
    let settings = SETTINGS.get_or_init(RuntimeSettings::default);
    assert!(WORKER_THREADS.contains(&settings.worker_threads));
    assert!(MAX_INPUT_BYTES.contains(&settings.max_input_bytes));
    settings
}

/// Refuses caller text `text` for `what` when it is longer than the
/// `max_input_bytes` setting. Output: `Ok`, or `InvalidArgument` naming
/// `what`, the length and the cap, never the text itself.
pub(crate) fn check_input(what: &str, text: &str) -> Result<(), FfiError> {
    let cap = settings().max_input_bytes;
    if text.len() <= cap {
        return Ok(());
    }
    assert!(!what.is_empty());
    Err(FfiError::new(
        ErrorCode::InvalidArgument,
        format!(
            "{what} is {} bytes, over the {cap}-byte max_input_bytes cap",
            text.len()
        ),
    ))
}

/// Runs `task` on the troupekit runtime and waits for it from any
/// executor. A panic in the task becomes `Panic` with its text; the
/// runtime keeps serving later calls. Output: the task's own result.
pub(crate) async fn spawn<T, F>(task: F) -> Result<T, FfiError>
where
    T: Send + 'static,
    F: Future<Output = Result<T, FfiError>> + Send + 'static,
{
    let runtime = match RUNTIME.get_or_init(start) {
        Ok(runtime) => runtime,
        Err(message) => return Err(FfiError::new(ErrorCode::Internal, message.clone())),
    };
    match runtime.spawn(task).await {
        Ok(result) => result,
        Err(err) if err.is_panic() => Err(FfiError::new(
            ErrorCode::Panic,
            format!("troupekit task panicked: {}", panic_text(err.into_panic())),
        )),
        Err(_) => Err(FfiError::new(
            ErrorCode::Cancelled,
            "troupekit task was cancelled",
        )),
    }
}

fn start() -> Result<Runtime, String> {
    let settings = settings();
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(settings.worker_threads)
        .thread_name("troupekit-worker")
        .enable_all()
        .build()
        .map_err(|e| format!("the troupekit runtime did not start: {e}"))
}

fn check(settings: &RuntimeSettings) -> Result<(), FfiError> {
    for (field, value, range) in [
        ("worker_threads", settings.worker_threads, WORKER_THREADS),
        ("max_input_bytes", settings.max_input_bytes, MAX_INPUT_BYTES),
        ("event_buffer", settings.event_buffer, EVENT_BUFFER),
    ] {
        if !range.contains(&value) {
            return Err(FfiError::new(
                ErrorCode::InvalidArgument,
                format!(
                    "{field} must be {} to {}, got {value}",
                    range.start(),
                    range.end()
                ),
            ));
        }
    }
    Ok(())
}

fn panic_text(payload: Box<dyn Any + Send>) -> String {
    match payload.downcast::<String>() {
        Ok(text) => *text,
        Err(payload) => match payload.downcast::<&'static str>() {
            Ok(text) => (*text).to_string(),
            Err(_) => "a panic without a message".to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn panic_in_task_becomes_panic_code() {
        let err = spawn(async { panic!("host tool blew up") as Result<(), FfiError> })
            .await
            .unwrap_err();
        assert_eq!(err.code, ErrorCode::Panic);
        assert!(err.message.contains("host tool blew up"), "{err}");
        let next = spawn(async { Ok::<u32, FfiError>(7) }).await.unwrap();
        assert_eq!(next, 7);
    }
}
