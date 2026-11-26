use crate::error::FfiError;

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
/// Input: the settings. Output: `Ok`, or `InvalidArgument` for a value
/// out of range or a second call with other settings.
pub fn init(_settings: RuntimeSettings) -> Result<(), FfiError> {
    Err(FfiError::unsupported("init"))
}
