use pyo3::prelude::*;

mod agent;
mod block;
mod error;
mod graph;
mod host;
mod records;
mod stream;

/// Returns the troupekit version, exported to Python.
/// Input: none. Output: the value of `troupekit_ffi_core::version`,
/// passed through unchanged.
#[pyfunction]
fn version() -> String {
    troupekit_ffi_core::version()
}

/// Sets the process-wide runtime once, before the first agent is built.
/// Input: the settings. Output: None, or `TroupekitError` from the neutral
/// layer (for example settings out of bounds, or a second call).
#[pyfunction]
fn init(py: Python<'_>, settings: records::RuntimeSettings) -> PyResult<()> {
    troupekit_ffi_core::init(settings.into()).map_err(|e| error::raise(py, e))
}

/// The native module `troupekit._troupekit` that the Python package re-exports.
/// Input: the module object Python creates on import. Output: the module with
/// every exported function, class and exception added, or the import error.
#[pymodule]
fn _troupekit(module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = module.py();
    module.add_function(wrap_pyfunction!(version, module)?)?;
    module.add_function(wrap_pyfunction!(init, module)?)?;
    module.add("TroupekitError", py.get_type::<error::TroupekitError>())?;
    module.add("HostError", py.get_type::<error::HostError>())?;
    module.add_class::<error::ErrorCode>()?;
    module.add_class::<agent::AgentBuilder>()?;
    module.add_class::<agent::AgentHandle>()?;
    module.add_class::<graph::GraphBuilder>()?;
    module.add_class::<graph::GraphHandle>()?;
    module.add_class::<stream::RunStream>()?;
    module.add_class::<host::EventListener>()?;
    module.add_class::<host::TraceListener>()?;
    module.add_class::<host::HostTool>()?;
    module.add_class::<host::HostModel>()?;
    module.add_class::<host::HostTransport>()?;
    module.add_class::<host::TokenSink>()?;
    module.add_class::<host::HttpSink>()?;
    module.add_class::<records::RuntimeSettings>()?;
    module.add_class::<records::Usage>()?;
    module.add_class::<records::RunResult>()?;
    module.add_class::<records::GraphResult>()?;
    module.add_class::<records::Progress>()?;
    module.add_class::<records::Pause>()?;
    module.add_class::<records::EventKind>()?;
    module.add_class::<records::Event>()?;
    module.add_class::<records::HostModelCaps>()?;
    module.add_class::<records::HostMessage>()?;
    module.add_class::<records::HostRequest>()?;
    module.add_class::<records::HostReply>()?;
    module.add_class::<records::HostHeader>()?;
    module.add_class::<records::HostHttpRequest>()?;
    add_knowledge(module)
}

/// Adds the knowledge and memory names when the `knowledge` feature is on.
/// Input: the module. Output: the module with `HostEmbedder`,
/// `KnowledgeConfig` and `MemoryConfig` added, or the error that stopped it.
#[cfg(feature = "knowledge")]
fn add_knowledge(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<host::HostEmbedder>()?;
    module.add_class::<records::KnowledgeConfig>()?;
    module.add_class::<records::MemoryConfig>()?;
    Ok(())
}

/// Adds nothing: this build has no `knowledge` feature.
/// Input: the module. Output: always `Ok`, so the import still succeeds.
/// The knowledge names are then missing from the module.
#[cfg(not(feature = "knowledge"))]
fn add_knowledge(_module: &Bound<'_, PyModule>) -> PyResult<()> {
    Ok(())
}
