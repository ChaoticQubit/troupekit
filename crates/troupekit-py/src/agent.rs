use crate::block::wait;
use crate::host::{PyModel, PyTool, PyTransport};
use crate::records::{Progress, RunResult};
use crate::stream::RunStream;
use pyo3::prelude::*;
use std::sync::Arc;
use troupekit_ffi_core as neutral;

#[cfg(feature = "knowledge")]
use crate::host::PyEmbedder;
#[cfg(feature = "knowledge")]
use crate::records::{KnowledgeConfig, MemoryConfig};

/// Builds one agent in code or from a persona TOML, as the kit builder does.
/// Wraps the neutral `AgentBuilder`; every setter returns a new builder and
/// forwards unchanged. `build` validates everything.
#[pyclass(frozen, module = "troupekit")]
pub struct AgentBuilder(pub(crate) Arc<neutral::AgentBuilder>);

#[pymethods]
impl AgentBuilder {
    #[new]
    fn new(name: String) -> Self {
        AgentBuilder(neutral::AgentBuilder::new(name))
    }

    #[staticmethod]
    fn from_toml(text: String) -> Self {
        AgentBuilder(neutral::AgentBuilder::from_toml(text))
    }

    #[staticmethod]
    fn from_toml_file(path: String) -> Self {
        AgentBuilder(neutral::AgentBuilder::from_toml_file(path))
    }

    fn model(&self, selector: String) -> Self {
        AgentBuilder(self.0.model(selector))
    }

    fn instructions(&self, text: String) -> Self {
        AgentBuilder(self.0.instructions(text))
    }

    fn display_name(&self, text: String) -> Self {
        AgentBuilder(self.0.display_name(text))
    }

    fn role(&self, text: String) -> Self {
        AgentBuilder(self.0.role(text))
    }

    fn goal(&self, text: String) -> Self {
        AgentBuilder(self.0.goal(text))
    }

    fn tool(&self, py: Python<'_>, tool: Py<PyAny>) -> PyResult<Self> {
        let tool = Arc::new(PyTool::new(py, tool)?);
        Ok(AgentBuilder(self.0.tool(tool)))
    }

    fn eval(&self, call: String) -> Self {
        AgentBuilder(self.0.eval(call))
    }

    fn max_turns(&self, turns: u32) -> Self {
        AgentBuilder(self.0.max_turns(turns))
    }

    fn max_tokens(&self, tokens: u32) -> Self {
        AgentBuilder(self.0.max_tokens(tokens))
    }

    fn retries(&self, retries: u32) -> Self {
        AgentBuilder(self.0.retries(retries))
    }

    fn sqlite(&self, path: String) -> Self {
        AgentBuilder(self.0.sqlite(path))
    }

    fn api_key(&self, key: String) -> Self {
        AgentBuilder(self.0.api_key(key))
    }

    fn api_key_env(&self, name: String) -> Self {
        AgentBuilder(self.0.api_key_env(name))
    }

    fn catalog_toml(&self, text: String) -> Self {
        AgentBuilder(self.0.catalog_toml(text))
    }

    fn host_model(&self, py: Python<'_>, model: Py<PyAny>) -> PyResult<Self> {
        let model = Arc::new(PyModel::new(py, model)?);
        Ok(AgentBuilder(self.0.host_model(model)))
    }

    fn host_model_priced(
        &self,
        py: Python<'_>,
        model: Py<PyAny>,
        input_micros_per_mtok: u64,
        output_micros_per_mtok: u64,
    ) -> PyResult<Self> {
        let model = Arc::new(PyModel::new(py, model)?);
        Ok(AgentBuilder(self.0.host_model_priced(
            model,
            input_micros_per_mtok,
            output_micros_per_mtok,
        )))
    }

    fn host_transport(&self, transport: Py<PyAny>) -> Self {
        AgentBuilder(self.0.host_transport(Arc::new(PyTransport(transport))))
    }

    fn laws(&self, path: String) -> Self {
        AgentBuilder(self.0.laws(path))
    }

    fn law(&self, text: String) -> Self {
        AgentBuilder(self.0.law(text))
    }

    fn auto_laws(&self, on: bool) -> Self {
        AgentBuilder(self.0.auto_laws(on))
    }

    #[cfg(feature = "knowledge")]
    #[pyo3(signature = (config, embedder=None))]
    fn knowledge(
        &self,
        py: Python<'_>,
        config: KnowledgeConfig,
        embedder: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let embedder = match embedder {
            Some(object) => {
                let bridge: Arc<dyn neutral::HostEmbedder> = Arc::new(PyEmbedder::new(py, object)?);
                Some(bridge)
            }
            None => None,
        };
        Ok(AgentBuilder(self.0.knowledge(config.into(), embedder)))
    }

    #[cfg(feature = "knowledge")]
    fn memory(&self, config: MemoryConfig) -> Self {
        AgentBuilder(self.0.memory(config.into()))
    }

    fn build(&self, py: Python<'_>) -> PyResult<AgentHandle> {
        wait(py, self.0.build()).map(AgentHandle)
    }
}

/// A built agent: runs, starts, resumes and streams, as the kit's `BuiltAgent`.
/// Every call blocks with the GIL released; `troupekit.aio` has async forms.
/// Safe to share between Python threads.
#[pyclass(frozen, module = "troupekit")]
pub struct AgentHandle(Arc<neutral::AgentHandle>);

#[pymethods]
impl AgentHandle {
    fn run(&self, py: Python<'_>, input: String) -> PyResult<String> {
        wait(py, self.0.run(input))
    }

    fn run_outcome(&self, py: Python<'_>, input: String) -> PyResult<RunResult> {
        wait(py, self.0.run_outcome(input)).map(Into::into)
    }

    fn start(&self, py: Python<'_>, input: String) -> PyResult<String> {
        wait(py, self.0.start(input))
    }

    fn resume(&self, py: Python<'_>, run_id: String) -> PyResult<String> {
        wait(py, self.0.resume(run_id))
    }

    fn resume_for(&self, py: Python<'_>, run_id: String, budget_ms: u64) -> PyResult<Progress> {
        wait(py, self.0.resume_for(run_id, budget_ms)).map(Into::into)
    }

    fn stream(&self, input: String) -> RunStream {
        RunStream(self.0.stream(input))
    }
}
