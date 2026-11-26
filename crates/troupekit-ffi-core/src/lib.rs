/// Returns the troupekit version this library was built from.
/// Input: none. Output: the workspace package version, for example `0.1.0`,
/// which every troupekit crate shares, so it matches `troupekit-core`.
pub fn version() -> String {
    let value = env!("CARGO_PKG_VERSION");
    assert!(!value.is_empty());
    assert!(value.split('.').count() == 3);
    value.to_owned()
}
