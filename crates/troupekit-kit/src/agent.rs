use crate::built::{BuiltAgent, Discard, Input, RunOutcome};
use crate::error::KitError;
use crate::laws::Laws;
use crate::model::{builtin_catalog, resolve, ModelChoice};
use crate::storage::Storage;
use crate::tool::AsTool;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use toml::Value as Toml;
use troupekit_agent::{
    AgentRun, EvalFn, EventSink, InMemoryLedger, Limits, ModelHandle, Persona, SimpleIo,
    UsageLedger,
};
use troupekit_core::message::{Message, Role};
use troupekit_core::secret::Secret;
use troupekit_providers::{Catalog, Model, Price, Transport};
use troupekit_tools::{Tool, ToolRegistry, Toolkit};

const MAX_RETRIES: u32 = 32;

pub(crate) struct Prepared {
    pub persona: Persona,
    pub persona_text: String,
    pub tools: Vec<Arc<dyn Tool>>,
    pub registry: ToolRegistry,
    pub handle: ModelHandle,
    pub eval_fns: Vec<(String, EvalFn)>,
    pub laws: Laws,
}
const DEFAULT_RETRIES: u32 = 3;
const DEFAULT_ROLE: &str = "An assistant.";
const DEFAULT_GOAL: &str = "Answer the user's request.";

/// Builds one agent from a name to an answer: model, instructions, tools,
/// evals, limits, storage and laws. Setters never fail; `build` checks
/// everything, reports the first problem and makes no model call.
/// `run` and `run_with` build and run in one call.
#[derive(Clone)]
pub struct Agent {
    name: String,
    model: Option<String>,
    instructions: Option<String>,
    display_name: Option<String>,
    role: Option<String>,
    goal: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
    evals: Vec<String>,
    eval_fns: Vec<(String, EvalFn)>,
    limits: Limits,
    max_turns: Option<u32>,
    max_tokens: Option<u32>,
    pub(crate) storage: Storage,
    #[cfg(feature = "knowledge")]
    pub(crate) memory: Option<crate::memory::AgentMemory>,
    #[cfg(feature = "knowledge")]
    pub(crate) knowledge: Option<crate::knowledge::Knowledge>,
    catalog: Option<Catalog>,
    api_key: Option<Secret>,
    api_key_env: Option<String>,
    transport: Option<Arc<dyn Transport>>,
    model_client: Option<Arc<dyn Model>>,
    price: Option<Price>,
    events: Option<Arc<dyn EventSink>>,
    ledger: Option<Arc<dyn UsageLedger>>,
    retries: u32,
    law_paths: Vec<PathBuf>,
    inline_laws: Vec<String>,
    auto_laws: bool,
    pub(crate) members: crate::team_agent::Members,
}

impl Agent {
    /// Starts a builder for the agent called `name`.
    /// The name must be 1 to 64 of `[a-z0-9-]`; `build` checks it.
    /// Every other setting starts at its documented default.
    pub fn new(name: impl Into<String>) -> Self {
        Agent {
            name: name.into(),
            model: None,
            instructions: None,
            display_name: None,
            role: None,
            goal: None,
            tools: Vec::new(),
            evals: Vec::new(),
            eval_fns: Vec::new(),
            limits: Limits::default(),
            max_turns: None,
            max_tokens: None,
            storage: Storage::memory(),
            #[cfg(feature = "knowledge")]
            memory: None,
            #[cfg(feature = "knowledge")]
            knowledge: None,
            catalog: None,
            api_key: None,
            api_key_env: None,
            transport: None,
            model_client: None,
            price: None,
            events: None,
            ledger: None,
            retries: DEFAULT_RETRIES,
            law_paths: Vec::new(),
            inline_laws: Vec::new(),
            auto_laws: true,
            members: crate::team_agent::Members::default(),
        }
    }

    /// Sets the model: `vendor/model` from the catalog, or an alias that
    /// needs `.model_client(..)`. Required.
    /// Returns the builder.
    pub fn model(mut self, selector: impl Into<String>) -> Self {
        self.model = Some(selector.into());
        self
    }

    /// Sets the system instructions. Required; the macro uses the doc
    /// comment. The text is kept exactly, never parsed as TOML.
    /// Returns the builder.
    pub fn instructions(mut self, text: impl Into<String>) -> Self {
        self.instructions = Some(text.into());
        self
    }

    /// Sets the display name shown in the system prompt.
    /// Defaults to the agent name.
    /// Returns the builder.
    pub fn display_name(mut self, text: impl Into<String>) -> Self {
        self.display_name = Some(text.into());
        self
    }

    /// Sets the role line of the system prompt.
    /// Defaults to `"An assistant."`.
    /// Returns the builder.
    pub fn role(mut self, text: impl Into<String>) -> Self {
        self.role = Some(text.into());
        self
    }

    /// Sets the goal line of the system prompt.
    /// Defaults to `"Answer the user's request."`.
    /// Returns the builder.
    pub fn goal(mut self, text: impl Into<String>) -> Self {
        self.goal = Some(text.into());
        self
    }

    /// Adds a tool: a `#[tool]` fn or a `tool(..)` value.
    /// A second tool under a taken name fails at `build`.
    /// Returns the builder.
    pub fn tool(mut self, tool: impl AsTool) -> Self {
        self.tools.push(tool.into_arc());
        self
    }

    /// Adds a tool already behind an `Arc<dyn Tool>`.
    /// Same checks as `tool` at `build`.
    /// Returns the builder.
    pub fn tool_arc(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Adds every tool of `tools`, in order.
    /// Same checks as `tool` at `build`.
    /// Returns the builder.
    pub fn tools(mut self, tools: impl IntoIterator<Item = Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }

    /// Adds every tool of `toolkit`, in order.
    /// Same checks as `tool` at `build`.
    /// Returns the builder.
    pub fn toolkit(mut self, toolkit: Toolkit) -> Self {
        self.tools.extend(toolkit.tools().iter().cloned());
        self
    }

    /// Adds one persona eval call, such as `min_items(items, 2)`.
    /// Parsed and checked against registered evals at `build`.
    /// Returns the builder.
    pub fn eval(mut self, call: impl Into<String>) -> Self {
        self.evals.push(call.into());
        self
    }

    /// Adds every eval call of `calls`, in order.
    /// Same checks as `eval` at `build`.
    /// Returns the builder.
    pub fn evals<S: Into<String>>(mut self, calls: impl IntoIterator<Item = S>) -> Self {
        self.evals.extend(calls.into_iter().map(Into::into));
        self
    }

    /// Registers custom eval `name` running `f`, next to the built-ins.
    /// An invalid, reserved or taken name fails at `build`.
    /// Returns the builder.
    pub fn eval_fn(mut self, name: impl Into<String>, f: EvalFn) -> Self {
        self.eval_fns.push((name.into(), f));
        self
    }

    /// Gives the agent long-term memory for one user and session: the five
    /// memory tools, bound to them, and a preamble of facts and summary.
    /// Needs `AgentMemory::sqlite(..)` or `.knowledge(..)` in 0.2.
    #[cfg(feature = "knowledge")]
    pub fn memory(mut self, memory: crate::memory::AgentMemory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Lets the agent search a folder: ingested at `build` (unchanged files
    /// skipped) and served by the `search_knowledge` tool, fenced as
    /// untrusted text. Takes `Knowledge::dir(..)` or a path.
    #[cfg(feature = "knowledge")]
    pub fn knowledge(mut self, knowledge: impl Into<crate::knowledge::Knowledge>) -> Self {
        self.knowledge = Some(knowledge.into());
        self
    }

    /// Sets the spend limits checked before every model call.
    /// Defaults to `Limits::default()`, no limit.
    /// Returns the builder.
    pub fn limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Caps model turns per run; 2 to 50, checked at `build`.
    /// Defaults to `DEFAULT_MAX_TURNS`.
    /// Returns the builder.
    pub fn max_turns(mut self, turns: u32) -> Self {
        self.max_turns = Some(turns);
        self
    }

    /// Caps output tokens per model call; 1 to 200000, checked at `build`.
    /// Defaults to `DEFAULT_MAX_TOKENS`.
    /// Returns the builder.
    pub fn max_tokens(mut self, tokens: u32) -> Self {
        self.max_tokens = Some(tokens);
        self
    }

    /// Sets where run checkpoints live.
    /// Defaults to `Storage::memory()`, fresh per run.
    /// Returns the builder.
    pub fn checkpoints(mut self, storage: Storage) -> Self {
        self.storage = storage;
        self
    }

    /// Replaces the built-in provider catalog.
    /// Model selectors and prices resolve against it at `build`.
    /// Returns the builder.
    pub fn catalog(mut self, catalog: Catalog) -> Self {
        self.catalog = Some(catalog);
        self
    }

    /// Sets the API key, beating any environment variable.
    /// The key never appears in `Debug` or an error.
    /// Returns the builder.
    pub fn api_key(mut self, key: Secret) -> Self {
        self.api_key = Some(key);
        self
    }

    /// Reads the API key from environment variable `name` at `build`.
    /// Defaults to `<VENDOR>_API_KEY`.
    /// Returns the builder.
    pub fn api_key_env(mut self, name: impl Into<String>) -> Self {
        self.api_key_env = Some(name.into());
        self
    }

    /// Sets the HTTP transport for catalog models.
    /// Defaults to a new reqwest transport (feature `reqwest`).
    /// Returns the builder.
    pub fn transport(mut self, transport: Arc<dyn Transport>) -> Self {
        self.transport = Some(transport);
        self
    }

    /// Serves `model` for the selector instead of connecting one.
    /// For tests, aliases and custom hosts; needs no key or network.
    /// Returns the builder.
    pub fn model_client(mut self, model: Arc<dyn Model>) -> Self {
        self.model_client = Some(model);
        self
    }

    /// Sets the price replies are charged at.
    /// Defaults to the catalog price, else free with a model client.
    /// Returns the builder.
    pub fn price(mut self, price: Price) -> Self {
        self.price = Some(price);
        self
    }

    /// Sends every run event to `sink`.
    /// Defaults to a sink that drops events, so memory stays bounded.
    /// Returns the builder.
    pub fn events(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.events = Some(sink);
        self
    }

    /// Records usage in `ledger` for period limits.
    /// Defaults to an in-memory ledger per built agent.
    /// Returns the builder.
    pub fn ledger(mut self, ledger: Arc<dyn UsageLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// Sets how often a transient provider failure is retried per run.
    /// At most 32, checked at `build`; defaults to 3.
    /// Returns the builder.
    pub fn retries(mut self, retries: u32) -> Self {
        self.retries = retries;
        self
    }

    /// Adds a `.bend` law file, checked at `build`.
    /// 0.2 stores laws; no run proves them yet.
    /// Returns the builder.
    pub fn laws(mut self, path: impl AsRef<Path>) -> Self {
        self.law_paths.push(path.as_ref().to_path_buf());
        self
    }

    /// Adds one inline law, checked at `build`.
    /// 0.2 stores laws; no run proves them yet.
    /// Returns the builder.
    pub fn law(mut self, law: impl Into<String>) -> Self {
        self.inline_laws.push(law.into());
        self
    }

    /// Asks for laws derived from the code as well. Defaults to `true`.
    /// 0.2 records the choice only.
    /// Returns the builder.
    pub fn auto_laws(mut self, on: bool) -> Self {
        self.auto_laws = on;
        self
    }

    /// Checks every setting and returns a ready, shareable agent.
    /// Fails with the first problem: settings, tools, persona, laws,
    /// evals, model, key or storage. Makes no model call.
    pub async fn build(self) -> Result<BuiltAgent, KitError> {
        if self.retries > MAX_RETRIES {
            return Err(KitError::config(
                &self.name,
                format!(
                    "retries must be at most {MAX_RETRIES}, got {}",
                    self.retries
                ),
            ));
        }
        if self.members.has_tools() {
            let flat = self.attach_members().await?;
            return Box::pin(flat.build()).await;
        }
        if self.members.is_team() {
            return self.build_team().await;
        }
        #[cfg(feature = "knowledge")]
        let (this, preamble) = {
            self.prepare(None)?;
            crate::memory::attach(self).await?
        };
        #[cfg(not(feature = "knowledge"))]
        let this = self;
        let prepared = this.prepare(None)?;
        let mut io = SimpleIo::new(prepared.registry);
        io.models
            .insert(prepared.persona.model.clone(), prepared.handle);
        for (eval, f) in &this.eval_fns {
            io.evals.register(eval, f.clone())?;
        }
        io.events = this.events.clone().unwrap_or_else(|| Arc::new(Discard));
        io.ledger = this
            .ledger
            .clone()
            .unwrap_or_else(|| Arc::new(InMemoryLedger::default()));
        let probe = vec![Message::text(Role::User, "build check")];
        AgentRun::new(&prepared.persona, probe, this.limits.clone(), &io).await?;
        let store = this.storage.open().await?;
        let member_tools = this.members.tool_names().to_vec();
        let built = BuiltAgent::new(
            prepared.persona,
            io,
            this.limits,
            store,
            this.retries,
            prepared.laws,
        )
        .with_member_tools(member_tools);
        #[cfg(feature = "knowledge")]
        let built = built.with_preamble(preamble);
        Ok(built)
    }

    pub(crate) fn graph_conflict(&self) -> Option<&'static str> {
        if !self.members.is_empty() {
            Some("agent tools, delegates, members or handoffs; graphs do not take them yet")
        } else if self.limits != Limits::default() {
            Some("limits; set them on the graph")
        } else if !self.law_paths.is_empty() || !self.inline_laws.is_empty() {
            Some("laws; set them on the graph")
        } else if self.ledger.is_some() {
            Some("ledger; set it on the graph")
        } else if self.has_context() {
            Some("memory or knowledge; graphs do not take them in 0.2")
        } else {
            None
        }
    }

    #[cfg(feature = "knowledge")]
    fn has_context(&self) -> bool {
        self.memory.is_some() || self.knowledge.is_some()
    }

    #[cfg(not(feature = "knowledge"))]
    fn has_context(&self) -> bool {
        false
    }

    pub(crate) fn same_model(&self, other: &Agent) -> bool {
        fn same<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
            match (a, b) {
                (None, None) => true,
                (Some(a), Some(b)) => Arc::ptr_eq(a, b),
                _ => false,
            }
        }
        self.model == other.model
            && same(&self.model_client, &other.model_client)
            && self.price == other.price
            && same(&self.transport, &other.transport)
            && self.api_key == other.api_key
            && self.api_key_env == other.api_key_env
            && self.catalog == other.catalog
    }

    #[cfg(feature = "knowledge")]
    pub(crate) fn default_embedder(
        &self,
    ) -> Result<Arc<dyn troupekit_context::embed::Embedder>, KitError> {
        crate::model::default_embedder(ModelChoice {
            agent: &self.name,
            selector: self.model.as_deref().unwrap_or_default(),
            catalog: self.catalog.as_ref().unwrap_or_else(|| builtin_catalog()),
            client: None,
            price: None,
            api_key: self.api_key.clone(),
            api_key_env: self.api_key_env.as_deref(),
            transport: self.transport.clone(),
        })
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn prepare(&self, model_key: Option<&str>) -> Result<Prepared, KitError> {
        let name = self.name.as_str();
        let Some(instructions) = self.instructions.as_deref() else {
            return Err(KitError::config(
                name,
                "needs instructions: call .instructions(..) or add a /// doc comment",
            ));
        };
        let Some(selector) = self.model.as_deref() else {
            return Err(KitError::config(
                name,
                "needs a model: call .model(\"vendor/model\")",
            ));
        };
        let mut registry = ToolRegistry::new();
        for tool in &self.tools {
            registry.register(tool.clone())?;
        }
        let tool_names: Vec<String> = registry.specs().into_iter().map(|s| s.name).collect();
        let (persona, persona_text) =
            self.persona(instructions, model_key.unwrap_or(selector), &tool_names)?;
        let laws = Laws::check(name, &self.law_paths, &self.inline_laws, self.auto_laws)?;
        let catalog = self.catalog.as_ref().unwrap_or_else(|| builtin_catalog());
        let handle = resolve(ModelChoice {
            agent: name,
            selector,
            catalog,
            client: self.model_client.clone(),
            price: self.price,
            api_key: self.api_key.clone(),
            api_key_env: self.api_key_env.as_deref(),
            transport: self.transport.clone(),
        })?;
        Ok(Prepared {
            persona,
            persona_text,
            tools: self.tools.clone(),
            registry,
            handle,
            eval_fns: self.eval_fns.clone(),
            laws,
        })
    }

    /// Builds the agent and runs it once on `input`.
    /// Returns the final answer as text.
    /// Fails like `build`, then like `BuiltAgent::run`.
    pub async fn run(self, input: impl Into<Input>) -> Result<String, KitError> {
        self.build().await?.run(input).await
    }

    /// Builds the agent and runs it once on `input`.
    /// Returns the full outcome: text, output, usage, calls, messages.
    /// Fails like `build`, then like `BuiltAgent::run_with`.
    pub async fn run_with(self, input: impl Into<Input>) -> Result<RunOutcome, KitError> {
        self.build().await?.run_with(input).await
    }

    fn persona(
        &self,
        instructions: &str,
        selector: &str,
        tools: &[String],
    ) -> Result<(Persona, String), KitError> {
        let text_or = |v: &Option<String>, default: &str| {
            Toml::String(v.clone().unwrap_or_else(|| default.to_string()))
        };
        let mut table = toml::Table::new();
        table.insert("name".into(), Toml::String(self.name.clone()));
        table.insert(
            "display_name".into(),
            text_or(&self.display_name, &self.name),
        );
        table.insert("role".into(), text_or(&self.role, DEFAULT_ROLE));
        table.insert("goal".into(), text_or(&self.goal, DEFAULT_GOAL));
        table.insert(
            "instructions".into(),
            Toml::String(instructions.to_string()),
        );
        table.insert("model".into(), Toml::String(selector.to_string()));
        let strings =
            |items: &[String]| Toml::Array(items.iter().cloned().map(Toml::String).collect());
        table.insert("tools".into(), strings(tools));
        table.insert("evals".into(), strings(&self.evals));
        if let Some(turns) = self.max_turns {
            table.insert("max_turns".into(), Toml::Integer(i64::from(turns)));
        }
        if let Some(tokens) = self.max_tokens {
            table.insert("max_tokens".into(), Toml::Integer(i64::from(tokens)));
        }
        let text = toml::to_string(&table).map_err(|e| {
            KitError::config(&self.name, format!("persona does not serialize: {e}"))
        })?;
        let persona = Persona::parse(&format!("agent:{}", self.name), &text)?;
        Ok((persona, text))
    }
}

impl fmt::Debug for Agent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let tools: Vec<&str> = self.tools.iter().map(|t| t.spec().name.as_str()).collect();
        f.debug_struct("Agent")
            .field("name", &self.name)
            .field("model", &self.model)
            .field("tools", &tools)
            .field("evals", &self.evals)
            .field("storage", &self.storage)
            .field("api_key", &self.api_key.as_ref().map(|_| "[redacted]"))
            .field("api_key_env", &self.api_key_env)
            .field("model_client", &self.model_client.is_some())
            .field("retries", &self.retries)
            .finish_non_exhaustive()
    }
}
