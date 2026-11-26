//! llama.cpp behind the `llama` feature, through the `tk_llama` C shim.
//! Every `unsafe` block of the crate lives in this module; Rust input is
//! checked before it crosses into C.

#[allow(dead_code)]
pub(crate) mod ffi;

use std::ffi::CStr;
use std::sync::Once;

static BACKEND: Once = Once::new();

/// Initializes the llama.cpp backends once per process and silences
/// llama.cpp logging. Input: none. Output: none.
/// Every entry point that touches llama.cpp calls it first.
pub(crate) fn init_backend() {
    BACKEND.call_once(|| unsafe { ffi::tk_backend_init() });
}

/// Returns llama.cpp's backend and CPU feature summary, initializing the
/// backends first. Input: none.
/// Output: the summary text, empty if llama.cpp returns none.
pub fn system_info() -> String {
    init_backend();
    let ptr = unsafe { ffi::tk_system_info() };
    if ptr.is_null() {
        return String::new();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}
