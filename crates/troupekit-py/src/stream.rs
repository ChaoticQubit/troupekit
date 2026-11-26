use crate::block::wait;
use crate::error::raise;
use crate::host::PyListener;
use crate::records::{Event, RunResult};
use pyo3::prelude::*;
use std::sync::Arc;
use troupekit_ffi_core as neutral;

/// The events of one run, pulled with `next` or pushed to a listener.
/// Wraps the neutral `RunStream`; it is also an iterator (freeze Q4), and
/// `troupekit.aio` makes it an async iterator.
#[pyclass(frozen, module = "troupekit")]
pub struct RunStream(pub(crate) Arc<neutral::RunStream>);

#[pymethods]
impl RunStream {
    fn next(&self, py: Python<'_>) -> PyResult<Option<Event>> {
        Ok(wait(py, self.0.next())?.map(Into::into))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Event>> {
        self.next(py)
    }

    fn set_listener(&self, py: Python<'_>, listener: Py<PyAny>) -> PyResult<()> {
        self.0
            .set_listener(Arc::new(PyListener(listener)))
            .map_err(|error| raise(py, error))
    }

    fn cancel(&self) {
        self.0.cancel()
    }

    fn run_id(&self) -> String {
        self.0.run_id()
    }

    fn dropped_deltas(&self) -> u64 {
        self.0.dropped_deltas()
    }

    fn result(&self, py: Python<'_>) -> PyResult<RunResult> {
        wait(py, self.0.result()).map(Into::into)
    }
}
