use pyo3::prelude::*;

/// Returns the troupekit version, exported to Python.
/// Input: none. Output: the value of `troupekit_ffi_core::version`,
/// passed through unchanged.
#[pyfunction]
fn version() -> String {
    troupekit_ffi_core::version()
}

/// The native module `troupekit._troupekit` that the Python package re-exports.
/// Input: the module object Python creates on import. Output: the module with
/// every exported function added, or the error that stopped the import.
#[pymodule]
fn _troupekit(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(version, module)?)?;
    Ok(())
}
