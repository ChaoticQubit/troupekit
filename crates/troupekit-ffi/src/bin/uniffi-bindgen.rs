/// Runs the UniFFI binding generator that ships with the pinned `uniffi`.
/// Input: the `uniffi-bindgen` command line, for example `generate --library`.
/// Output: generated Swift or Kotlin sources; built only with feature `bindgen`.
fn main() {
    uniffi::uniffi_bindgen_main()
}
