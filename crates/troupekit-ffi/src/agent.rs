use crate::error::TroupekitError;
use crate::host::{
    HostModel, HostTool, HostTransport, ModelBridge, ToolBridge, TraceBridge, TraceListener,
    TransportBridge,
};
use crate::records::{Progress, RunResult};
use crate::stream::RunStream;
use std::sync::Arc;
use troupekit_ffi_core as neutral;

#[cfg(feature = "knowledge")]
use crate::host::{EmbedderBridge, HostEmbedder};
#[cfg(feature = "knowledge")]
use crate::records::{KnowledgeConfig, MemoryConfig};

/// Builds one agent in code or from a persona TOML, as the kit builder does.
/// Wraps the neutral `AgentBuilder`; every setter returns a new builder and
/// forwards unchanged. `build` validates everything.
#[derive(Debug, uniffi::Object)]
pub struct AgentBuilder(pub(crate) Arc<neutral::AgentBuilder>);

impl AgentBuilder {
    fn wrap(inner: Arc<neutral::AgentBuilder>) -> Arc<AgentBuilder> {
        Arc::new(AgentBuilder(inner))
    }
}

#[uniffi::export]
impl AgentBuilder {
    #[uniffi::constructor]
    pub fn new(name: String) -> Arc<Self> {
        Self::wrap(neutral::AgentBuilder::new(name))
    }

    #[uniffi::constructor]
    pub fn from_toml(text: String) -> Arc<Self> {
        Self::wrap(neutral::AgentBuilder::from_toml(text))
    }

    #[uniffi::constructor]
    pub fn from_toml_file(path: String) -> Arc<Self> {
        Self::wrap(neutral::AgentBuilder::from_toml_file(path))
    }

    pub fn model(&self, selector: String) -> Arc<Self> {
        Self::wrap(self.0.model(selector))
    }

    pub fn instructions(&self, text: String) -> Arc<Self> {
        Self::wrap(self.0.instructions(text))
    }

    pub fn display_name(&self, text: String) -> Arc<Self> {
        Self::wrap(self.0.display_name(text))
    }

    pub fn role(&self, text: String) -> Arc<Self> {
        Self::wrap(self.0.role(text))
    }

    pub fn goal(&self, text: String) -> Arc<Self> {
        Self::wrap(self.0.goal(text))
    }

    pub fn tool(&self, tool: Arc<dyn HostTool>) -> Arc<Self> {
        Self::wrap(self.0.tool(Arc::new(ToolBridge(tool))))
    }

    pub fn eval(&self, call: String) -> Arc<Self> {
        Self::wrap(self.0.eval(call))
    }

    pub fn max_turns(&self, turns: u32) -> Arc<Self> {
        Self::wrap(self.0.max_turns(turns))
    }

    pub fn max_tokens(&self, tokens: u32) -> Arc<Self> {
        Self::wrap(self.0.max_tokens(tokens))
    }

    pub fn retries(&self, retries: u32) -> Arc<Self> {
        Self::wrap(self.0.retries(retries))
    }

    pub fn sqlite(&self, path: String) -> Arc<Self> {
        Self::wrap(self.0.sqlite(path))
    }

    pub fn api_key(&self, key: String) -> Arc<Self> {
        Self::wrap(self.0.api_key(key))
    }

    pub fn api_key_env(&self, name: String) -> Arc<Self> {
        Self::wrap(self.0.api_key_env(name))
    }

    pub fn catalog_toml(&self, text: String) -> Arc<Self> {
        Self::wrap(self.0.catalog_toml(text))
    }

    pub fn host_model(&self, model: Arc<dyn HostModel>) -> Arc<Self> {
        Self::wrap(self.0.host_model(Arc::new(ModelBridge(model))))
    }

    pub fn host_model_priced(
        &self,
        model: Arc<dyn HostModel>,
        input_micros_per_mtok: u64,
        output_micros_per_mtok: u64,
    ) -> Arc<Self> {
        let model = Arc::new(ModelBridge(model));
        Self::wrap(
            self.0
                .host_model_priced(model, input_micros_per_mtok, output_micros_per_mtok),
        )
    }

    pub fn host_transport(&self, transport: Arc<dyn HostTransport>) -> Arc<Self> {
        Self::wrap(self.0.host_transport(Arc::new(TransportBridge(transport))))
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

    pub async fn build(&self) -> Result<Arc<AgentHandle>, TroupekitError> {
        Ok(Arc::new(AgentHandle(self.0.build().await?)))
    }
}

#[cfg(feature = "knowledge")]
#[uniffi::export]
impl AgentBuilder {
    pub fn knowledge(
        &self,
        config: KnowledgeConfig,
        embedder: Option<Arc<dyn HostEmbedder>>,
    ) -> Arc<Self> {
        let embedder = embedder.map(|e| {
            let bridge: Arc<dyn neutral::HostEmbedder> = Arc::new(EmbedderBridge(e));
            bridge
        });
        Self::wrap(self.0.knowledge(config.into(), embedder))
    }

    pub fn memory(&self, config: MemoryConfig) -> Arc<Self> {
        Self::wrap(self.0.memory(config.into()))
    }
}

/// A built agent: runs, starts, resumes and streams, as the kit's `BuiltAgent`.
/// Wraps the neutral `AgentHandle`; every method forwards unchanged.
/// Safe to share between threads and to run concurrently.
#[derive(Debug, uniffi::Object)]
pub struct AgentHandle(Arc<neutral::AgentHandle>);

#[uniffi::export]
impl AgentHandle {
    pub async fn run(&self, input: String) -> Result<String, TroupekitError> {
        Ok(self.0.run(input).await?)
    }

    pub async fn run_outcome(&self, input: String) -> Result<RunResult, TroupekitError> {
        Ok(self.0.run_outcome(input).await?.into())
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
