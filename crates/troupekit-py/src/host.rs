use crate::error::host_error;
use crate::records::{Event, HostHeader, HostHttpRequest, HostModelCaps, HostReply, HostRequest};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::sync::Arc;
use troupekit_ffi_core as neutral;

/// Base class for a Python event listener: subclass it, or pass any object
/// with an `on_event(event)` method to `RunStream.set_listener`.
/// Input: none. Output: an empty, subclassable base.
#[pyclass(subclass, module = "troupekit")]
pub struct EventListener;

/// Base class for a Python trace listener: subclass it, or pass any object
/// with an `on_record(line)` method to `trace_listener`.
/// Input: none. Output: an empty, subclassable base.
#[pyclass(subclass, module = "troupekit")]
pub struct TraceListener;

/// Base class for a Python tool: subclass it, or pass any object with
/// `name()`, `description()`, `schema_json()` and `call(args_json)`.
/// Input: none. Output: an empty, subclassable base.
#[pyclass(subclass, module = "troupekit")]
pub struct HostTool;

/// Base class for a Python model: subclass it, or pass any object with
/// `id()`, `capabilities()` and `generate(request, sink)`.
/// Input: none. Output: an empty, subclassable base.
#[pyclass(subclass, module = "troupekit")]
pub struct HostModel;

/// Base class for a Python embedder: subclass it, or pass any object with
/// `dims()` and `embed(texts)`.
/// Input: none. Output: an empty, subclassable base.
#[cfg(feature = "knowledge")]
#[pyclass(subclass, module = "troupekit")]
pub struct HostEmbedder;

/// Base class for a Python HTTP stack: subclass it, or pass any object with
/// `send(request, sink)`. Used for model calls only.
/// Input: none. Output: an empty, subclassable base.
#[pyclass(subclass, module = "troupekit")]
pub struct HostTransport;

macro_rules! base_new {
    ($($ty:ty),*) => {
        $(
            #[pymethods]
            impl $ty {
                /// Creates the empty base. It takes and ignores any arguments,
                /// so a subclass `__init__` with its own parameters works.
                /// Input: the subclass constructor arguments. Output: the base.
                #[new]
                #[pyo3(signature = (*_args, **_kwargs))]
                fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
                    Self
                }
            }
        )*
    };
}

base_new!(
    EventListener,
    TraceListener,
    HostTool,
    HostModel,
    HostTransport
);
#[cfg(feature = "knowledge")]
base_new!(HostEmbedder);

/// Where a Python host model pushes its text deltas, wrapping the neutral sink.
/// Input: text pieces in order. Output: none; Rust caps the total bytes.
/// Created by Rust and passed to `HostModel.generate`.
#[pyclass(frozen, module = "troupekit")]
pub struct TokenSink(Arc<neutral::TokenSink>);

#[pymethods]
impl TokenSink {
    fn delta(&self, text: String) {
        self.0.delta(text)
    }
}

/// Where a Python host transport pushes the reply, wrapping the neutral sink.
/// Input: the head once, then body chunks as `bytes`. Output: none.
/// Created by Rust and passed to `HostTransport.send`.
#[pyclass(frozen, module = "troupekit")]
pub struct HttpSink(Arc<neutral::HttpSink>);

#[pymethods]
impl HttpSink {
    fn head(&self, status: u16, url: String, headers: Vec<HostHeader>) {
        self.0
            .head(status, url, headers.into_iter().map(Into::into).collect())
    }

    fn chunk(&self, bytes: Vec<u8>) {
        self.0.chunk(bytes)
    }
}

/// Presents a Python listener object as the neutral trait.
/// Input: the object. Output: a neutral listener that takes the GIL per
/// event; an exception goes to `sys.unraisablehook`, as Python does.
pub(crate) struct PyListener(pub(crate) Py<PyAny>);

impl neutral::EventListener for PyListener {
    fn on_event(&self, event: neutral::FfiEvent) {
        Python::attach(|py| {
            let event = Event::from(event);
            if let Err(error) = self.0.call_method1(py, "on_event", (event,)) {
                error.write_unraisable(py, Some(self.0.bind(py)));
            }
        })
    }
}

/// Presents a Python trace listener object as the neutral trait.
/// Input: the object. Output: a neutral listener that takes the GIL per
/// record; an exception goes to `sys.unraisablehook`, as Python does.
pub(crate) struct PyTraceListener(pub(crate) Py<PyAny>);

impl neutral::TraceListener for PyTraceListener {
    fn on_record(&self, line: String) {
        Python::attach(|py| {
            if let Err(error) = self.0.call_method1(py, "on_record", (line,)) {
                error.write_unraisable(py, Some(self.0.bind(py)));
            }
        })
    }
}

/// Presents a Python tool object as the neutral trait.
/// Name, description and schema are read once, at registration; `call`
/// takes the GIL and turns a Python exception into a `HostError`.
pub(crate) struct PyTool {
    object: Py<PyAny>,
    name: String,
    description: String,
    schema_json: String,
}

impl PyTool {
    /// Reads the fixed fields of a Python tool object.
    /// Input: the GIL token and the object. Output: the bridge, or the
    /// Python error a method raised or a non-`str` result produced.
    pub(crate) fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<PyTool> {
        let name = object.call_method0(py, "name")?.extract::<String>(py)?;
        let description = object
            .call_method0(py, "description")?
            .extract::<String>(py)?;
        let schema_json = object
            .call_method0(py, "schema_json")?
            .extract::<String>(py)?;
        Ok(PyTool {
            object,
            name,
            description,
            schema_json,
        })
    }
}

impl neutral::HostTool for PyTool {
    fn name(&self) -> String {
        self.name.clone()
    }

    fn description(&self) -> String {
        self.description.clone()
    }

    fn schema_json(&self) -> String {
        self.schema_json.clone()
    }

    fn call(&self, args_json: String) -> Result<String, neutral::HostError> {
        Python::attach(|py| {
            self.object
                .call_method1(py, "call", (args_json,))
                .and_then(|out| out.extract::<String>(py).map_err(PyErr::from))
                .map_err(|error| host_error(py, error))
        })
    }
}

/// Presents a Python model object as the neutral trait.
/// Id and capabilities are read once, at registration; `generate` takes
/// the GIL and turns a Python exception into a `HostError`.
pub(crate) struct PyModel {
    object: Py<PyAny>,
    id: String,
    caps: neutral::HostModelCaps,
}

impl PyModel {
    /// Reads the fixed fields of a Python model object.
    /// Input: the GIL token and the object. Output: the bridge, or the
    /// Python error a method raised or a wrong result type produced.
    pub(crate) fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<PyModel> {
        let id = object.call_method0(py, "id")?.extract::<String>(py)?;
        let caps = object
            .call_method0(py, "capabilities")?
            .extract::<HostModelCaps>(py)?;
        Ok(PyModel {
            object,
            id,
            caps: caps.into(),
        })
    }
}

impl neutral::HostModel for PyModel {
    fn id(&self) -> String {
        self.id.clone()
    }

    fn capabilities(&self) -> neutral::HostModelCaps {
        self.caps
    }

    fn generate(
        &self,
        request: neutral::HostRequest,
        sink: Arc<neutral::TokenSink>,
    ) -> Result<neutral::HostReply, neutral::HostError> {
        Python::attach(|py| {
            let args = (HostRequest::from(request), TokenSink(sink));
            self.object
                .call_method1(py, "generate", args)
                .and_then(|out| out.extract::<HostReply>(py).map_err(PyErr::from))
                .map(Into::into)
                .map_err(|error| host_error(py, error))
        })
    }
}

/// Presents a Python embedder object as the neutral trait.
/// `dims` is read once, at registration; `embed` takes the GIL and turns a
/// Python exception into a `HostError`.
#[cfg(feature = "knowledge")]
pub(crate) struct PyEmbedder {
    object: Py<PyAny>,
    dims: u32,
}

#[cfg(feature = "knowledge")]
impl PyEmbedder {
    /// Reads the fixed fields of a Python embedder object.
    /// Input: the GIL token and the object. Output: the bridge, or the
    /// Python error `dims()` raised or a non-`int` result produced.
    pub(crate) fn new(py: Python<'_>, object: Py<PyAny>) -> PyResult<PyEmbedder> {
        let dims = object.call_method0(py, "dims")?.extract::<u32>(py)?;
        Ok(PyEmbedder { object, dims })
    }
}

#[cfg(feature = "knowledge")]
impl neutral::HostEmbedder for PyEmbedder {
    fn dims(&self) -> u32 {
        self.dims
    }

    fn embed(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, neutral::HostError> {
        Python::attach(|py| {
            self.object
                .call_method1(py, "embed", (texts,))
                .and_then(|out| out.extract::<Vec<Vec<f32>>>(py).map_err(PyErr::from))
                .map_err(|error| host_error(py, error))
        })
    }
}

/// Presents a Python transport object as the neutral trait.
/// Input: the object. Output: a neutral transport that takes the GIL per
/// request and turns a Python exception into a `HostError`.
pub(crate) struct PyTransport(pub(crate) Py<PyAny>);

impl neutral::HostTransport for PyTransport {
    fn send(
        &self,
        request: neutral::HostHttpRequest,
        sink: Arc<neutral::HttpSink>,
    ) -> Result<(), neutral::HostError> {
        Python::attach(|py| {
            let args = (HostHttpRequest::from(request), HttpSink(sink));
            self.0
                .call_method1(py, "send", args)
                .map(|_| ())
                .map_err(|error| host_error(py, error))
        })
    }
}
