//! llama.cpp behind the `llama` feature, through the `tk_llama` C shim.
//! Every `unsafe` block of the crate lives in this module; Rust input is
//! checked before it crosses into C.

#[allow(dead_code)]
pub(crate) mod ffi;
mod metrics;
mod model;
mod sampler;

pub use model::{LlamaModel, LlamaSettings};

use std::ffi::CStr;
use std::sync::{Mutex, Once};

static BACKEND: Once = Once::new();

/// Initializes the llama.cpp backends once per process and silences
/// llama.cpp logging. Input: none. Output: none.
/// Every entry point that touches llama.cpp calls it first.
pub(crate) fn init_backend() {
    BACKEND.call_once(|| unsafe { ffi::tk_backend_init() });
}

/// Returns llama.cpp's backend and CPU feature summary, initializing the
/// backends first. Calls are serialized: llama.cpp rewrites one static
/// buffer per call. Input: none.
/// Output: the summary text, empty if llama.cpp returns none.
pub fn system_info() -> String {
    static NATIVE: Mutex<()> = Mutex::new(());
    init_backend();
    let _held = NATIVE
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let ptr = unsafe { ffi::tk_system_info() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
