use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use troupekit_ffi_core as neutral;

create_exception!(
    troupekit,
    TroupekitError,
    PyException,
    "A troupekit failure with a stable `code`, a `message` and the `run_id`."
);

create_exception!(
    troupekit,
    HostError,
    PyException,
    "Raised by a Python host callback; the run sees its text as the failure."
);

/// Stable numeric code of every failure, mirrored from the neutral `ErrorCode`.
/// `int(ErrorCode.MissingKey)` equals `TroupekitError.code` for that failure.
/// The numbers are fixed forever.
#[pyclass(frozen, eq, eq_int, module = "troupekit")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorCode {
    Config = 1,
    MissingKey = 2,
    Catalog = 3,
    Persona = 4,
    Agent = 5,
    Tool = 6,
    Workflow = 7,
    Graph = 8,
    Store = 9,
    Context = 10,
    Failed = 11,
    Paused = 12,
    Retries = 13,
    Cancelled = 14,
    HostCallback = 15,
    InvalidArgument = 16,
    Unsupported = 17,
    Panic = 18,
    Internal = 99,
}

/// Turns a neutral error into a raised `TroupekitError`.
/// Input: the GIL token and the error. Output: the exception, with `code`
/// (int), `message` and `run_id` attributes set on its value.
pub(crate) fn raise(py: Python<'_>, error: neutral::FfiError) -> PyErr {
    let code = error.code as u32;
    let err = TroupekitError::new_err(error.message.clone());
    let value = err.value(py);
    let set = value
        .setattr("code", code)
        .and_then(|_| value.setattr("message", error.message))
        .and_then(|_| value.setattr("run_id", error.run_id));
    match set {
        Ok(()) => err,
        Err(failed) => failed,
    }
}

/// Renders a Python exception as the text a neutral `HostError` carries.
/// Input: the GIL token and the exception. Output: the neutral error with
/// `Type: message`, as `str()` of the exception shows it.
pub(crate) fn host_error(py: Python<'_>, error: PyErr) -> neutral::HostError {
    let text = error.value(py).to_string();
    let name = error
        .get_type(py)
        .name()
        .map_or_else(|_| "Exception".to_owned(), |n| n.to_string());
    neutral::HostError::new(format!("{name}: {text}"))
}
