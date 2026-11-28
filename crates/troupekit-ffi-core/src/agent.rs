use crate::error::{ErrorCode, FfiError};
#[cfg(feature = "knowledge")]
use crate::host::HostEmbedder;
use crate::host::{HostModel, HostTool, HostTransport};
use crate::host_transport::{origin, wrap_transport};
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
use troupekit_providers::model::MAX_REPLY_BYTES;
use troupekit_providers::wire::AuthStyle;
use troupekit_providers::Catalog;
use troupekit_providers::Price;

/// Builds an agent in steps from the host language. Every setter
/// returns the same builder, so calls chain; nothing is checked until
/// `build`, which consumes the settings.
pub struct AgentBuilder {
    state: Mutex<State>,
    transport: Mutex<Option<Arc<dyn HostTransport>>>,
}

impl std::fmt::Debug for AgentBuilder {
    /// Writes the builder state and whether a host transport is set.
    /// Agent settings and keys are not shown.
    /// Output: the debug text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let transport = self
            .transport
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        f.debug_struct("AgentBuilder")
            .field("state", &*state)
            .field("host_transport", &transport.is_some())
            .finish()
    }
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
            transport: Mutex::new(None),
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
        let transport = self
            .transport
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take();
        match std::mem::replace(&mut *state, State::Built) {
            State::Open(agent) => match transport {
                Some(host) => with_transport(*agent, host),
                None => Ok(*agent),
            },
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
        let wrapped = crate::host::guarded("tool", || crate::host::wrap_tool(tool));
        self.update(|agent| Ok(agent.tool_arc(wrapped?)))
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
    pub fn host_model(self: &Arc<Self>, model: Arc<dyn HostModel>) -> Arc<Self> {
        self.with_host_model(model, Price::FREE)
    }

    /// Runs the agent on a host model with a price per million tokens,
    /// in millionths of a US dollar, for input and output.
    /// Output: the same builder, for chaining.
    pub fn host_model_priced(
        self: &Arc<Self>,
        model: Arc<dyn HostModel>,
        input_micros_per_mtok: u64,
        output_micros_per_mtok: u64,
    ) -> Arc<Self> {
        let price = Price {
            input_usd_micros_per_mtok: input_micros_per_mtok,
            output_usd_micros_per_mtok: output_micros_per_mtok,
            ..Price::FREE
        };
        self.with_host_model(model, price)
    }

    fn with_host_model(self: &Arc<Self>, model: Arc<dyn HostModel>, price: Price) -> Arc<Self> {
        let wrapped = crate::host::guarded("model", || crate::host_model::wrap_model(model));
        self.update(|agent| {
            let (client, id) = wrapped?;
            Ok(agent.model(id).model_client(client).price(price))
        })
    }

    /// Sends model and embedding HTTP calls through the host's client.
    /// Only the resolved vendor's origin is allowed; tools never use it.
    /// Output: the same builder, for chaining.
    pub fn host_transport(self: &Arc<Self>, transport: Arc<dyn HostTransport>) -> Arc<Self> {
        *self
            .transport
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = Some(transport);
        self.clone()
    }

    /// Refuses a law file: no run enforces explicit laws yet, so `build`
    /// returns `Unsupported`. An over-cap path is `InvalidArgument` and a
    /// path with `..` is `Config`, both checked first.
    /// Output: the same builder, for chaining.
    pub fn laws(self: &Arc<Self>, path: String) -> Arc<Self> {
        self.update(|_| Err(refuse_law("laws", &path, true)))
    }

    /// Refuses a law given as text: no run enforces explicit laws yet,
    /// so `build` returns `Unsupported`. Over-cap text is
    /// `InvalidArgument`, checked first.
    /// Output: the same builder, for chaining.
    pub fn law(self: &Arc<Self>, text: String) -> Arc<Self> {
        self.update(|_| Err(refuse_law("law", &text, false)))
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
        config: KnowledgeConfig,
        embedder: Option<Arc<dyn HostEmbedder>>,
    ) -> Arc<Self> {
        let wrapped = embedder
            .map(|host| crate::host::guarded("embedder", || crate::host::wrap_embedder(host)))
            .transpose();
        self.update(|agent| {
            check_input("knowledge dir", &config.dir)?;
            let mut knowledge =
                troupekit_kit::Knowledge::dir(&config.dir).allow_failed(config.allow_failed);
            if let Some(index) = config.index {
                check_input("knowledge index", &index)?;
                knowledge = knowledge.index(index);
            }
            if let Some(store) = config.store {
                check_input("knowledge store", &store)?;
                knowledge = knowledge.store(store);
            }
            if let Some(embedder) = wrapped? {
                knowledge = knowledge.embedder(embedder);
            }
            Ok(agent.knowledge(knowledge))
        })
    }

    #[cfg(feature = "knowledge")]
    /// Gives the agent fact memory for the user and session of `config`,
    /// with the memory tools.
    /// Output: the same builder, for chaining.
    pub fn memory(self: &Arc<Self>, config: MemoryConfig) -> Arc<Self> {
        self.update(|agent| {
            check_input("memory user", &config.user)?;
            check_input("memory session", &config.session)?;
            let mut memory = troupekit_kit::AgentMemory::new(config.user, config.session)
                .preamble(config.preamble);
            if let Some(path) = config.sqlite {
                check_input("memory sqlite", &path)?;
                memory = memory.sqlite(path);
            }
            Ok(agent.memory(memory))
        })
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

fn with_transport(agent: Agent, host: Arc<dyn HostTransport>) -> Result<Agent, FfiError> {
    let Some(vendor) = troupekit_kit::__private::vendor_of(&agent) else {
        let named = match troupekit_kit::__private::selector_of(&agent) {
            Some(selector) => format!("model `{selector}` names no catalog vendor"),
            None => "no model is set".to_string(),
        };
        return Err(FfiError::new(
            ErrorCode::Config,
            format!("host_transport needs a catalog .model(\"vendor/model\"): {named}"),
        ));
    };
    if origin(&vendor.base_url).is_none() {
        return Err(FfiError::new(
            ErrorCode::Config,
            format!("vendor `{}` base_url has no plain origin", vendor.id),
        ));
    }
    let plain_ok = vendor.base_url.starts_with("http://") && vendor.auth == AuthStyle::NoKey;
    let transport = wrap_transport(host, &vendor.base_url, plain_ok, MAX_REPLY_BYTES);
    Ok(agent.transport(transport))
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

/// Builds the error for an explicit law, which no run enforces yet.
/// Input: the setter name, its text or path, and whether it is a path.
/// Output: the input check error if one fails, else `Unsupported`.
pub(crate) fn refuse_law(what: &str, text: &str, is_path: bool) -> FfiError {
    if let Err(err) = check_input(what, text) {
        return err;
    }
    let parent = std::path::Path::new(text)
        .components()
        .any(|c| c == std::path::Component::ParentDir);
    if is_path && parent {
        return FfiError::new(
            ErrorCode::Config,
            format!("law file `{text}` must not contain `..`"),
        );
    }
    FfiError::new(
        ErrorCode::Unsupported,
        format!("{what}: explicit laws are not enforced by runs yet"),
    )
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
