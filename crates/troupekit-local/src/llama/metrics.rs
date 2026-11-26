use super::ffi;
use crate::model::LocalMetrics;
use std::ptr::NonNull;

/// Builds the metrics of one call from the shim's performance counters.
/// Inputs: the context, time to first token in ms and reused prefix
/// tokens. Output: the metrics, peak RAM from `getrusage` when known.
pub(crate) fn collect(ctx: NonNull<ffi::TkCtx>, ttft_ms: f64, reused: usize) -> LocalMetrics {
    let (mut prompt_ms, mut eval_ms, mut n_prompt, mut n_eval) = (0.0f64, 0.0f64, 0i32, 0i32);
    unsafe {
        ffi::tk_perf(
            ctx.as_ptr(),
            &mut prompt_ms,
            &mut eval_ms,
            &mut n_prompt,
            &mut n_eval,
        )
    };
    LocalMetrics::default()
        .with_prefill_tokens_per_second(rate(n_prompt, prompt_ms))
        .with_decode_tokens_per_second(rate(n_eval, eval_ms))
        .with_time_to_first_token_ms(ttft_ms.max(0.0))
        .with_peak_ram_bytes(peak_ram_bytes())
        .with_reused_prefix_tokens(reused)
}

fn rate(tokens: i32, ms: f64) -> f64 {
    if tokens <= 0 || !ms.is_finite() || ms <= 0.0 {
        return 0.0;
    }
    f64::from(tokens) / (ms / 1_000.0)
}

#[cfg(all(unix, target_pointer_width = "64"))]
mod rusage {
    #[repr(C)]
    pub(super) struct Rusage {
        pub(super) utime: [i64; 2],
        pub(super) stime: [i64; 2],
        pub(super) maxrss: i64,
        pub(super) rest: [i64; 13],
    }

    extern "C" {
        pub(super) fn getrusage(who: i32, usage: *mut Rusage) -> i32;
    }

    pub(super) const RUSAGE_SELF: i32 = 0;
}

/// Returns the process's peak resident memory in bytes.
/// `ru_maxrss` is bytes on Apple targets and KiB elsewhere on Unix.
/// None on Windows, 32-bit targets, or when the call fails.
#[cfg(all(unix, target_pointer_width = "64"))]
pub(crate) fn peak_ram_bytes() -> Option<u64> {
    let mut usage = rusage::Rusage {
        utime: [0; 2],
        stime: [0; 2],
        maxrss: 0,
        rest: [0; 13],
    };
    let rc = unsafe { rusage::getrusage(rusage::RUSAGE_SELF, &mut usage) };
    if rc != 0 || usage.maxrss < 0 {
        return None;
    }
    let raw = usage.maxrss as u64;
    if cfg!(any(target_os = "macos", target_os = "ios")) {
        Some(raw)
    } else {
        raw.checked_mul(1_024)
    }
}

/// Returns None: this target has no peak memory query in 0.3.
/// Windows and 32-bit targets land here.
/// The metrics field stays empty.
#[cfg(not(all(unix, target_pointer_width = "64")))]
pub(crate) fn peak_ram_bytes() -> Option<u64> {
    None
}
