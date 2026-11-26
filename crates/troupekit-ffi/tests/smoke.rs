/// The UniFFI adapter exports `version` and it returns the neutral layer's value.
/// Input: none. Output: passes when both strings are equal, non-empty and the
/// workspace package version, which every troupekit crate shares.
#[test]
fn version_is_exported() {
    let exported = troupekit_ffi::version();
    let neutral = troupekit_ffi_core::version();
    assert!(!exported.is_empty());
    assert_eq!(exported, neutral);
    assert_eq!(exported, env!("CARGO_PKG_VERSION"));
}
