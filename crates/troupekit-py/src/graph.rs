use crate::agent::AgentBuilder;
use crate::block::wait;
use crate::host::PyTool;
use crate::records::{GraphResult, Progress};
use crate::stream::RunStream;
use pyo3::prelude::*;
use std::sync::Arc;
use troupekit_ffi_core as neutral;

/// Builds a graph of agents and tool steps, as the kit's `Graph` does.
/// Wraps the neutral `GraphBuilder`; every setter returns a new builder and
/// forwards unchanged. `build` validates the graph.
#[pyclass(frozen, module = "troupekit")]
pub struct GraphBuilder(Arc<neutral::GraphBuilder>);

#[pymethods]
impl GraphBuilder {
    #[new]
    fn new(name: String) -> Self {
        GraphBuilder(neutral::GraphBuilder::new(name))
    }

    fn agent(&self, id: String, agent: PyRef<'_, AgentBuilder>) -> Self {
        GraphBuilder(self.0.agent(id, agent.0.clone()))
    }

    fn step(&self, py: Python<'_>, id: String, tool: Py<PyAny>) -> PyResult<Self> {
        let tool = Arc::new(PyTool::new(py, tool)?);
        Ok(GraphBuilder(self.0.step(id, tool)))
    }

    fn edge(&self, from: String, to: String) -> Self {
        GraphBuilder(self.0.edge(from, to))
    }

    fn sqlite(&self, path: String) -> Self {
        GraphBuilder(self.0.sqlite(path))
    }

    fn max_steps(&self, steps: u32) -> Self {
        GraphBuilder(self.0.max_steps(steps))
    }

    fn retries(&self, retries: u32) -> Self {
        GraphBuilder(self.0.retries(retries))
    }

    fn laws(&self, path: String) -> Self {
        GraphBuilder(self.0.laws(path))
    }

    fn law(&self, text: String) -> Self {
        GraphBuilder(self.0.law(text))
    }

    fn auto_laws(&self, on: bool) -> Self {
        GraphBuilder(self.0.auto_laws(on))
    }

    fn build(&self, py: Python<'_>) -> PyResult<GraphHandle> {
        wait(py, self.0.build()).map(GraphHandle)
    }
}

/// A built graph: runs, starts, resumes and streams, as the kit's `BuiltGraph`.
/// Every call blocks with the GIL released; `troupekit.aio` has async forms.
/// Safe to share between Python threads.
#[pyclass(frozen, module = "troupekit")]
pub struct GraphHandle(Arc<neutral::GraphHandle>);

#[pymethods]
impl GraphHandle {
    fn run(&self, py: Python<'_>, input: String) -> PyResult<String> {
        wait(py, self.0.run(input))
    }

    fn run_result(&self, py: Python<'_>, input: String) -> PyResult<GraphResult> {
        wait(py, self.0.run_result(input)).map(Into::into)
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
