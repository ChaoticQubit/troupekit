use crate::error::{ErrorCode, FfiError};
#[cfg(feature = "knowledge")]
use crate::host::HostEmbedder;
use crate::host::{HostModel, HostTool, HostTransport};
#[cfg(feature = "knowledge")]
use crate::records::{KnowledgeConfig, MemoryConfig};
use crate::records::{Progress, RunResult};
use crate::runtime::{check_input, spawn};
use crate::stream::RunStream;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;
use troupekit_core::ids::RunId;
use troupekit_core::secret::Secret;
use troupekit_kit::{Agent, BuiltAgent, StreamSettings};
use troupekit_providers::Catalog;

/// Builds an agent in steps from the host language. Every setter
/// returns the same builder, so calls chain; nothing is checked until
/// `build`, which consumes the settings.
#[derive(Debug)]
pub struct AgentBuilder {
    state: Mutex<State>,
}

#[derive(Debug)]
enum State {
    Open(Box<Agent>),
    Broken(FfiError),
    Built,
}

impl AgentBuilder {
    fn start(what: &str, text: &str, agent: impl FnOnce() -> Agent) -> Arc<AgentBuilder> {
        let state = match check_input(what, text) {
            Ok(()) => State::Open(Box::new(agent())),
            Err(err) => State::Broken(err),
        };
        Arc::new(AgentBuilder {
            state: Mutex::new(state),
        })
    }

    fn update(
        self: &Arc<Self>,
        change: impl FnOnce(Agent) -> Result<Agent, FfiError>,
    ) -> Arc<Self> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let next = match std::mem::replace(&mut *state, State::Built) {
            State::Open(agent) => match change(*agent) {
                Ok(agent) => State::Open(Box::new(agent)),
                Err(err) => State::Broken(err),
            },
            other => other,
        };
        *state = next;
        self.clone()
    }

    pub(crate) fn take(&self) -> Result<Agent, FfiError> {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        match std::mem::replace(&mut *state, State::Built) {
            State::Open(agent) => Ok(*agent),
            State::Broken(err) => Err(err),
            State::Built => Err(FfiError::new(
                ErrorCode::InvalidArgument,
                "builder already built",
            )),
        }
    }

    /// Starts a builder for an agent named `name`: lowercase letters,
    /// digits and `_`. Input: the name. Output: a new builder.
    /// The name keys the agent in graphs and events.
    pub fn new(name: String) -> Arc<Self> {
        AgentBuilder::start("name", &name, || Agent::new(name.clone()))
    }

    /// Starts a builder from persona TOML `text`, parsed at build.
    /// Builder settings beat the TOML; at most 1 MiB.
    /// Output: a new builder.
    pub fn from_toml(text: String) -> Arc<Self> {
        AgentBuilder::start("from_toml", &text, || Agent::from_toml(text.clone()))
    }

    /// Starts a builder from the persona file at `path`, read at build.
    /// The path must have no `..`, end in `.toml` and name a file of at
    /// most 1 MiB. Output: a new builder.
    pub fn from_toml_file(path: String) -> Arc<Self> {
        AgentBuilder::start("from_toml_file", &path, || Agent::from_toml_file(&path))
    }

    /// Sets the model as `vendor/model`, resolved in the catalog at build.
    /// A host model set with `host_model` takes its place.
    /// Output: the same builder, for chaining.
    pub fn model(self: &Arc<Self>, selector: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("model", &selector)?;
            Ok(agent.model(selector))
        })
    }

    /// Sets the instructions, the core of the system prompt.
    /// Needed unless the persona TOML has them.
    /// Output: the same builder, for chaining.
    pub fn instructions(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("instructions", &text)?;
            Ok(agent.instructions(text))
        })
    }

    /// Sets the name the agent introduces itself by.
    /// Defaults to the builder name.
    /// Output: the same builder, for chaining.
    pub fn display_name(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("display_name", &text)?;
            Ok(agent.display_name(text))
        })
    }

    /// Sets the one-line role in the system prompt.
    /// Has a neutral default.
    /// Output: the same builder, for chaining.
    pub fn role(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("role", &text)?;
            Ok(agent.role(text))
        })
    }

    /// Sets the goal line in the system prompt.
    /// Has a neutral default.
    /// Output: the same builder, for chaining.
    pub fn goal(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("goal", &text)?;
            Ok(agent.goal(text))
        })
    }

    /// Adds a host tool the model may call.
    /// Name and schema are checked at build; two tools may not share a name.
    /// Output: the same builder, for chaining.
    pub fn tool(self: &Arc<Self>, tool: Arc<dyn HostTool>) -> Arc<Self> {
        self.update(|agent| Ok(agent.tool_arc(crate::host::wrap_tool(tool)?)))
    }

    /// Adds an eval call such as `max_words(80)`, run on every answer.
    /// A failed eval asks the model for one repair.
    /// Output: the same builder, for chaining.
    pub fn eval(self: &Arc<Self>, call: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("eval", &call)?;
            Ok(agent.eval(call))
        })
    }

    /// Caps the model turns of one run.
    /// Zero or a value over the engine bound is refused at build.
    /// Output: the same builder, for chaining.
    pub fn max_turns(self: &Arc<Self>, turns: u32) -> Arc<Self> {
        self.update(|agent| Ok(agent.max_turns(turns)))
    }

    /// Caps the tokens of one model reply.
    /// Zero is refused at build.
    /// Output: the same builder, for chaining.
    pub fn max_tokens(self: &Arc<Self>, tokens: u32) -> Arc<Self> {
        self.update(|agent| Ok(agent.max_tokens(tokens)))
    }

    /// Sets how often a transient provider failure is retried per run.
    /// At most 32; counted across resume windows.
    /// Output: the same builder, for chaining.
    pub fn retries(self: &Arc<Self>, retries: u32) -> Arc<Self> {
        self.update(|agent| Ok(agent.retries(retries)))
    }

    /// Keeps checkpoints in the SQLite file at `path`, so runs survive
    /// a restart. Without it checkpoints live in memory and `start` and
    /// `resume_for` are refused. Output: the same builder, for chaining.
    pub fn sqlite(self: &Arc<Self>, path: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("sqlite", &path)?;
            sqlite_storage(agent, path)
        })
    }

    /// Sets the vendor API key. It is held as a secret and never appears
    /// in a handle, record, debug text or error.
    /// Output: the same builder, for chaining.
    pub fn api_key(self: &Arc<Self>, key: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("api_key", &key)?;
            match Secret::new(key) {
                Ok(secret) => Ok(agent.api_key(secret)),
                Err(_) => Err(FfiError::new(
                    ErrorCode::InvalidArgument,
                    "api_key is blank",
                )),
            }
        })
    }

    /// Reads the vendor API key from environment variable `name` at build.
    /// A missing variable fails with `MissingKey` naming the variable.
    /// Output: the same builder, for chaining.
    pub fn api_key_env(self: &Arc<Self>, name: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("api_key_env", &name)?;
            Ok(agent.api_key_env(name))
        })
    }

    /// Replaces the model catalog with the catalog TOML `text`.
    /// Parsed at build; a bad catalog fails with `Catalog`.
    /// Output: the same builder, for chaining.
    pub fn catalog_toml(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("catalog_toml", &text)?;
            match Catalog::load(&text) {
                Ok(catalog) => Ok(agent.catalog(catalog)),
                Err(e) => Err(FfiError::new(
                    ErrorCode::Catalog,
                    format!("catalog_toml: {e}"),
                )),
            }
        })
    }

    /// Runs the agent on a model the host provides, such as the phone's
    /// built-in model. Free unless priced with `host_model_priced`.
    /// Output: the same builder, for chaining.
    pub fn host_model(self: &Arc<Self>, _model: Arc<dyn HostModel>) -> Arc<Self> {
        self.update(|_| Err(FfiError::unsupported("AgentBuilder::host_model")))
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
        self.update(|_| Err(FfiError::unsupported("AgentBuilder::host_model_priced")))
    }

    /// Sends model and embedding HTTP calls through the host's client.
    /// Only the resolved vendor's origin is allowed; tools never use it.
    /// Output: the same builder, for chaining.
    pub fn host_transport(self: &Arc<Self>, _transport: Arc<dyn HostTransport>) -> Arc<Self> {
        self.update(|_| Err(FfiError::unsupported("AgentBuilder::host_transport")))
    }

    /// Adds the law file at `path`, a `.bend` file checked at build.
    /// The path must not contain `..`.
    /// Output: the same builder, for chaining.
    pub fn laws(self: &Arc<Self>, path: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("laws", &path)?;
            Ok(agent.laws(path))
        })
    }

    /// Adds one law as text, checked at build.
    /// Laws guard tool calls and answers.
    /// Output: the same builder, for chaining.
    pub fn law(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|agent| {
            check_input("law", &text)?;
            Ok(agent.law(text))
        })
    }

    /// Turns the built-in laws on or off.
    /// On by default.
    /// Output: the same builder, for chaining.
    pub fn auto_laws(self: &Arc<Self>, on: bool) -> Arc<Self> {
        self.update(|agent| Ok(agent.auto_laws(on)))
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
        self.update(|_| Err(FfiError::unsupported("AgentBuilder::knowledge")))
    }

    #[cfg(feature = "knowledge")]
    /// Gives the agent fact memory for the user and session of `config`,
    /// with the memory tools.
    /// Output: the same builder, for chaining.
    pub fn memory(self: &Arc<Self>, _config: MemoryConfig) -> Arc<Self> {
        self.update(|_| Err(FfiError::unsupported("AgentBuilder::memory")))
    }

    /// Checks every setting, tool and model and returns a ready agent.
    /// Makes no model call. Output: the agent handle, or the first
    /// problem found with its code.
    pub async fn build(self: &Arc<Self>) -> Result<Arc<AgentHandle>, FfiError> {
        let agent = self.take()?;
        let built = spawn(async move { agent.build().await.map_err(FfiError::from) }).await?;
        Ok(Arc::new(AgentHandle { agent: built }))
    }
}

/// A built agent, ready to run many times, also at once.
/// Run ids cross as text; store one to resume the run after the app
/// restarts.
#[derive(Debug)]
pub struct AgentHandle {
    agent: BuiltAgent,
}

impl AgentHandle {
    /// Runs the agent on `input` to the end.
    /// Output: the answer text, or the error with its code and run id.
    /// Other runs on the same handle are not affected.
    pub async fn run(&self, input: String) -> Result<String, FfiError> {
        check_input("input", &input)?;
        let agent = self.agent.clone();
        spawn(async move { agent.run(input).await.map_err(FfiError::from) }).await
    }

    /// Runs the agent on `input` to the end.
    /// Output: the whole result: text, typed output as JSON text,
    /// usage, model calls and run id.
    pub async fn run_outcome(&self, input: String) -> Result<RunResult, FfiError> {
        check_input("input", &input)?;
        let agent = self.agent.clone();
        let outcome =
            spawn(async move { agent.run_with(input).await.map_err(FfiError::from) }).await?;
        Ok(RunResult::from_outcome(&outcome))
    }

    /// Starts a run on `input` with no model call and saves its first
    /// checkpoint. Needs `sqlite`. Output: the run id text, to continue
    /// with `resume` or `resume_for`.
    pub async fn start(&self, input: String) -> Result<String, FfiError> {
        check_input("input", &input)?;
        let agent = self.agent.clone();
        let run_id = spawn(async move { agent.start(input).await.map_err(FfiError::from) }).await?;
        Ok(run_id.to_string())
    }

    /// Continues run `run_id` from its last checkpoint to the end.
    /// Output: the answer text, or the error with its code.
    /// A finished run returns its stored answer.
    pub async fn resume(&self, run_id: String) -> Result<String, FfiError> {
        let run_id = parse_run_id(&run_id)?;
        let agent = self.agent.clone();
        spawn(async move { agent.resume(run_id).await.map_err(FfiError::from) }).await
    }

    /// Continues run `run_id` for about `budget_ms` milliseconds, saving
    /// after every step. Output: `Done` with the text, or `Pending` when
    /// time ran out; a step in flight finishes first.
    pub async fn resume_for(&self, run_id: String, budget_ms: u64) -> Result<Progress, FfiError> {
        let run_id = parse_run_id(&run_id)?;
        let agent = self.agent.clone();
        let budget = Duration::from_millis(budget_ms);
        let progress = spawn(async move {
            agent
                .resume_for(run_id, budget)
                .await
                .map_err(FfiError::from)
        })
        .await?;
        Progress::from_kit(progress)
    }

    /// Starts a run on `input` and returns its event stream once the
    /// run id is known. Output: the stream, or the setup error (input,
    /// tools, models) with its code, as `start` reports it.
    pub async fn stream(&self, input: String) -> Result<Arc<RunStream>, FfiError> {
        check_input("input", &input)?;
        let agent = self.agent.clone();
        let mut settings = StreamSettings::default();
        settings.capacity = crate::runtime::settings().event_buffer;
        let stream = spawn(async move {
            agent
                .stream_with(input, settings)
                .await
                .map_err(FfiError::from)
        })
        .await?;
        Ok(RunStream::agent(stream))
    }
}

/// Parses run id `text`, the hyphenated UUID form a handle returned.
/// Output: the id, or `InvalidArgument` that does not echo the text,
/// since a caller may pass anything there.
pub(crate) fn parse_run_id(text: &str) -> Result<RunId, FfiError> {
    check_input("run_id", text)?;
    match serde_json::from_value::<RunId>(serde_json::Value::String(text.to_string())) {
        Ok(run_id) => Ok(run_id),
        Err(_) => Err(FfiError::new(
            ErrorCode::InvalidArgument,
            "run_id is not a run id returned by start or a stream",
        )),
    }
}

#[cfg(feature = "sqlite")]
fn sqlite_storage(agent: Agent, path: String) -> Result<Agent, FfiError> {
    Ok(agent.checkpoints(troupekit_kit::Storage::sqlite(path)))
}

#[cfg(not(feature = "sqlite"))]
fn sqlite_storage(_agent: Agent, _path: String) -> Result<Agent, FfiError> {
    Err(FfiError::new(
        ErrorCode::Unsupported,
        "sqlite needs troupekit built with the sqlite feature",
    ))
}
