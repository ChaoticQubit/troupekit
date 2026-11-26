uniffi::setup_scaffolding!();

/// Returns the troupekit version, exported to Swift and Kotlin.
/// Input: none. Output: the value of `troupekit_ffi_core::version`,
/// passed through unchanged.
#[uniffi::export]
pub fn version() -> String {
    troupekit_ffi_core::version()
}
