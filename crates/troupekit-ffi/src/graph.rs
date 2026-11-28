use crate::agent::AgentBuilder;
use crate::error::TroupekitError;
use crate::host::{HostTool, ToolBridge, TraceBridge, TraceListener};
use crate::records::{GraphResult, Progress};
use crate::stream::RunStream;
use std::sync::Arc;
use troupekit_ffi_core as neutral;

/// Builds a graph of agents and tool steps, as the kit's `Graph` does.
/// Wraps the neutral `GraphBuilder`; every setter returns a new builder and
/// forwards unchanged. `build` validates the graph.
#[derive(Debug, uniffi::Object)]
pub struct GraphBuilder(Arc<neutral::GraphBuilder>);

impl GraphBuilder {
    fn wrap(inner: Arc<neutral::GraphBuilder>) -> Arc<GraphBuilder> {
        Arc::new(GraphBuilder(inner))
    }
}

#[uniffi::export]
impl GraphBuilder {
    #[uniffi::constructor]
    pub fn new(name: String) -> Arc<Self> {
        Self::wrap(neutral::GraphBuilder::new(name))
    }

    pub fn agent(&self, id: String, agent: Arc<AgentBuilder>) -> Arc<Self> {
        Self::wrap(self.0.agent(id, agent.0.clone()))
    }

    pub fn step(&self, id: String, tool: Arc<dyn HostTool>) -> Arc<Self> {
        Self::wrap(self.0.step(id, Arc::new(ToolBridge(tool))))
    }

    pub fn edge(&self, from: String, to: String) -> Arc<Self> {
        Self::wrap(self.0.edge(from, to))
    }

    pub fn sqlite(&self, path: String) -> Arc<Self> {
        Self::wrap(self.0.sqlite(path))
    }

    pub fn max_steps(&self, steps: u32) -> Arc<Self> {
        Self::wrap(self.0.max_steps(steps))
    }

    pub fn retries(&self, retries: u32) -> Arc<Self> {
        Self::wrap(self.0.retries(retries))
    }

    pub fn tracing(&self, on: bool) -> Arc<Self> {
        Self::wrap(self.0.tracing(on))
    }

    pub fn trace(&self, categories: Vec<String>) -> Arc<Self> {
        Self::wrap(self.0.trace(categories))
    }

    pub fn trace_content(&self, on: bool) -> Arc<Self> {
        Self::wrap(self.0.trace_content(on))
    }

    pub fn trace_jsonl(&self, path: String) -> Arc<Self> {
        Self::wrap(self.0.trace_jsonl(path))
    }

    pub fn trace_listener(&self, listener: Arc<dyn TraceListener>) -> Arc<Self> {
        Self::wrap(self.0.trace_listener(Arc::new(TraceBridge(listener))))
    }

    pub fn laws(&self, path: String) -> Arc<Self> {
        Self::wrap(self.0.laws(path))
    }

    pub fn law(&self, text: String) -> Arc<Self> {
        Self::wrap(self.0.law(text))
    }

    pub fn auto_laws(&self, on: bool) -> Arc<Self> {
        Self::wrap(self.0.auto_laws(on))
    }

    pub async fn build(&self) -> Result<Arc<GraphHandle>, TroupekitError> {
        Ok(Arc::new(GraphHandle(self.0.build().await?)))
    }
}

/// A built graph: runs, starts, resumes and streams, as the kit's `BuiltGraph`.
/// Wraps the neutral `GraphHandle`; every method forwards unchanged.
/// Safe to share between threads.
#[derive(Debug, uniffi::Object)]
pub struct GraphHandle(Arc<neutral::GraphHandle>);

#[uniffi::export]
impl GraphHandle {
    pub async fn run(&self, input: String) -> Result<String, TroupekitError> {
        Ok(self.0.run(input).await?)
    }

    pub async fn run_result(&self, input: String) -> Result<GraphResult, TroupekitError> {
        Ok(self.0.run_result(input).await?.into())
    }

    pub async fn start(&self, input: String) -> Result<String, TroupekitError> {
        Ok(self.0.start(input).await?)
    }

    pub async fn resume(&self, run_id: String) -> Result<String, TroupekitError> {
        Ok(self.0.resume(run_id).await?)
    }

    pub async fn resume_for(
        &self,
        run_id: String,
        budget_ms: u64,
    ) -> Result<Progress, TroupekitError> {
        Ok(self.0.resume_for(run_id, budget_ms).await?.into())
    }

    pub async fn stream(&self, input: String) -> Result<Arc<RunStream>, TroupekitError> {
        Ok(Arc::new(RunStream(self.0.stream(input).await?)))
    }
}
