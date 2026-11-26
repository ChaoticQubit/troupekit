use crate::error::FfiError;
#[cfg(feature = "knowledge")]
use crate::host::HostEmbedder;
use crate::host::{HostModel, HostTool, HostTransport};
#[cfg(feature = "knowledge")]
use crate::records::{KnowledgeConfig, MemoryConfig};
use crate::records::{Progress, RunResult};
use crate::stream::RunStream;
use std::sync::Arc;

/// Builds an agent in steps from the host language. Every setter
/// returns the same builder, so calls chain; nothing is checked until
/// `build`, which consumes the settings.
#[derive(Debug)]
pub struct AgentBuilder {
    _private: (),
}

impl AgentBuilder {
    fn empty() -> Arc<AgentBuilder> {
        Arc::new(AgentBuilder { _private: () })
    }

    /// Starts a builder for an agent named `name`: lowercase letters,
    /// digits and `_`. Input: the name. Output: a new builder.
    /// The name keys the agent in graphs and events.
    pub fn new(_name: String) -> Arc<Self> {
        AgentBuilder::empty()
    }

    /// Starts a builder from persona TOML `text`, parsed at build.
    /// Builder settings beat the TOML; at most 1 MiB.
    /// Output: a new builder.
    pub fn from_toml(_text: String) -> Arc<Self> {
        AgentBuilder::empty()
    }

    /// Starts a builder from the persona file at `path`, read at build.
    /// The path must have no `..`, end in `.toml` and name a file of at
    /// most 1 MiB. Output: a new builder.
    pub fn from_toml_file(_path: String) -> Arc<Self> {
        AgentBuilder::empty()
    }

    /// Sets the model as `vendor/model`, resolved in the catalog at build.
    /// A host model set with `host_model` takes its place.
    /// Output: the same builder, for chaining.
    pub fn model(self: &Arc<Self>, _selector: String) -> Arc<Self> {
        self.clone()
    }

    /// Sets the instructions, the core of the system prompt.
    /// Needed unless the persona TOML has them.
    /// Output: the same builder, for chaining.
    pub fn instructions(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Sets the name the agent introduces itself by.
    /// Defaults to the builder name.
    /// Output: the same builder, for chaining.
    pub fn display_name(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Sets the one-line role in the system prompt.
    /// Has a neutral default.
    /// Output: the same builder, for chaining.
    pub fn role(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Sets the goal line in the system prompt.
    /// Has a neutral default.
    /// Output: the same builder, for chaining.
    pub fn goal(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Adds a host tool the model may call.
    /// Name and schema are checked at build; two tools may not share a name.
    /// Output: the same builder, for chaining.
    pub fn tool(self: &Arc<Self>, _tool: Arc<dyn HostTool>) -> Arc<Self> {
        self.clone()
    }

    /// Adds an eval call such as `max_words(80)`, run on every answer.
    /// A failed eval asks the model for one repair.
    /// Output: the same builder, for chaining.
    pub fn eval(self: &Arc<Self>, _call: String) -> Arc<Self> {
        self.clone()
    }

    /// Caps the model turns of one run.
    /// Zero or a value over the engine bound is refused at build.
    /// Output: the same builder, for chaining.
    pub fn max_turns(self: &Arc<Self>, _turns: u32) -> Arc<Self> {
        self.clone()
    }

    /// Caps the tokens of one model reply.
    /// Zero is refused at build.
    /// Output: the same builder, for chaining.
    pub fn max_tokens(self: &Arc<Self>, _tokens: u32) -> Arc<Self> {
        self.clone()
    }

    /// Sets how often a transient provider failure is retried per run.
    /// At most 32; counted across resume windows.
    /// Output: the same builder, for chaining.
    pub fn retries(self: &Arc<Self>, _retries: u32) -> Arc<Self> {
        self.clone()
    }

    /// Keeps checkpoints in the SQLite file at `path`, so runs survive
    /// a restart. Without it checkpoints live in memory and `start` and
    /// `resume_for` are refused. Output: the same builder, for chaining.
    pub fn sqlite(self: &Arc<Self>, _path: String) -> Arc<Self> {
        self.clone()
    }

    /// Sets the vendor API key. It is held as a secret and never appears
    /// in a handle, record, debug text or error.
    /// Output: the same builder, for chaining.
    pub fn api_key(self: &Arc<Self>, _key: String) -> Arc<Self> {
        self.clone()
    }

    /// Reads the vendor API key from environment variable `name` at build.
    /// A missing variable fails with `MissingKey` naming the variable.
    /// Output: the same builder, for chaining.
    pub fn api_key_env(self: &Arc<Self>, _name: String) -> Arc<Self> {
        self.clone()
    }

    /// Replaces the model catalog with the catalog TOML `text`.
    /// Parsed at build; a bad catalog fails with `Catalog`.
    /// Output: the same builder, for chaining.
    pub fn catalog_toml(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Runs the agent on a model the host provides, such as the phone's
    /// built-in model. Free unless priced with `host_model_priced`.
    /// Output: the same builder, for chaining.
    pub fn host_model(self: &Arc<Self>, _model: Arc<dyn HostModel>) -> Arc<Self> {
        self.clone()
    }

    /// Runs the agent on a host model with a price per million tokens,
    /// in millionths of a US dollar, for input and output.
    /// Output: the same builder, for chaining.
    pub fn host_model_priced(
        self: &Arc<Self>,
        _model: Arc<dyn HostModel>,
        _input_micros_per_mtok: u64,
        _output_micros_per_mtok: u64,
    ) -> Arc<Self> {
        self.clone()
    }

    /// Sends model and embedding HTTP calls through the host's client.
    /// Only the resolved vendor's origin is allowed; tools never use it.
    /// Output: the same builder, for chaining.
    pub fn host_transport(self: &Arc<Self>, _transport: Arc<dyn HostTransport>) -> Arc<Self> {
        self.clone()
    }

    /// Adds the law file at `path`, a `.bend` file checked at build.
    /// The path must not contain `..`.
    /// Output: the same builder, for chaining.
    pub fn laws(self: &Arc<Self>, _path: String) -> Arc<Self> {
        self.clone()
    }

    /// Adds one law as text, checked at build.
    /// Laws guard tool calls and answers.
    /// Output: the same builder, for chaining.
    pub fn law(self: &Arc<Self>, _text: String) -> Arc<Self> {
        self.clone()
    }

    /// Turns the built-in laws on or off.
    /// On by default.
    /// Output: the same builder, for chaining.
    pub fn auto_laws(self: &Arc<Self>, _on: bool) -> Arc<Self> {
        self.clone()
    }

    #[cfg(feature = "knowledge")]
    /// Ingests the documents of `config` at build and gives the agent a
    /// search tool. `embedder` replaces the default cloud embedder.
    /// Output: the same builder, for chaining.
    pub fn knowledge(
        self: &Arc<Self>,
        _config: KnowledgeConfig,
        _embedder: Option<Arc<dyn HostEmbedder>>,
    ) -> Arc<Self> {
        self.clone()
    }

    #[cfg(feature = "knowledge")]
    /// Gives the agent fact memory for the user and session of `config`,
    /// with the memory tools.
    /// Output: the same builder, for chaining.
    pub fn memory(self: &Arc<Self>, _config: MemoryConfig) -> Arc<Self> {
        self.clone()
    }

    /// Checks every setting, tool and model and returns a ready agent.
    /// Makes no model call. Output: the agent handle, or the first
    /// problem found with its code.
    pub async fn build(self: &Arc<Self>) -> Result<Arc<AgentHandle>, FfiError> {
        Err(FfiError::unsupported("AgentBuilder::build"))
    }
}

/// A built agent, ready to run many times, also at once.
/// Run ids cross as text; store one to resume the run after the app
/// restarts.
#[derive(Debug)]
pub struct AgentHandle {
    _private: (),
}

impl AgentHandle {
    /// Runs the agent on `input` to the end.
    /// Output: the answer text, or the error with its code and run id.
    /// Other runs on the same handle are not affected.
    pub async fn run(&self, _input: String) -> Result<String, FfiError> {
        Err(FfiError::unsupported("AgentHandle::run"))
    }

    /// Runs the agent on `input` to the end.
    /// Output: the whole result: text, typed output as JSON text,
    /// usage, model calls and run id.
    pub async fn run_outcome(&self, _input: String) -> Result<RunResult, FfiError> {
        Err(FfiError::unsupported("AgentHandle::run_outcome"))
    }

    /// Starts a run on `input` with no model call and saves its first
    /// checkpoint. Needs `sqlite`. Output: the run id text, to continue
    /// with `resume` or `resume_for`.
    pub async fn start(&self, _input: String) -> Result<String, FfiError> {
        Err(FfiError::unsupported("AgentHandle::start"))
    }

    /// Continues run `run_id` from its last checkpoint to the end.
    /// Output: the answer text, or the error with its code.
    /// A finished run returns its stored answer.
    pub async fn resume(&self, _run_id: String) -> Result<String, FfiError> {
        Err(FfiError::unsupported("AgentHandle::resume"))
    }

    /// Continues run `run_id` for about `budget_ms` milliseconds, saving
    /// after every step. Output: `Done` with the text, or `Pending` when
    /// time ran out; a step in flight finishes first.
    pub async fn resume_for(&self, _run_id: String, _budget_ms: u64) -> Result<Progress, FfiError> {
        Err(FfiError::unsupported("AgentHandle::resume_for"))
    }

    /// Starts a run on `input` and returns its event stream at once.
    /// Output: the stream; its `result` gives the outcome or the error.
    /// The signature waits on the leader's stream choice (eng log).
    pub fn stream(&self, _input: String) -> Arc<RunStream> {
        RunStream::unsupported()
    }
}
