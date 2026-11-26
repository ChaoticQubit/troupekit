use crate::budget::Limits;
use crate::error::AgentError;
use crate::io::AgentIo;
use crate::persona::Persona;
use crate::run::{AgentRun, RunState, StepOutcome};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::CheckpointStore;
use troupekit_core::tool::ToolSpec;
use troupekit_tools::{fence_capped, valid_tool_name, Tool, ToolContext, ToolError, ToolOutput};

/// Name of the tool a leader uses to hand a task to a coworker.
/// Its arguments are `{coworker, task, context?}`; the coworker runs the
/// task as a member run and the result comes back fenced.
pub const DELEGATE_WORK: &str = "delegate_work";

/// Name of the tool a leader uses to ask a coworker a question.
/// Its arguments are `{coworker, question, context?}`; offered unless
/// `DelegateSettings::with_ask_question(false)` turns it off.
pub const ASK_QUESTION: &str = "ask_question";

const DEFAULT_MAX_DEPTH: u32 = 3;
const DEPTH_MAX: u32 = 16;
const COWORKERS_MAX: usize = 64;
const ACTIVE_MEMBERS_MAX: usize = 4096;
const ANSWER_CHARS_MAX: usize = 64 * 1024;
const STEPS_PER_TURN: u64 = 4;
const STEPS_SLACK: u64 = 8;

static ACTIVE: Mutex<BTreeMap<RunId, u32>> = Mutex::new(BTreeMap::new());

/// Bounds and options for agents called as tools: the delegation depth,
/// base limits and a spend cap for each member run, an optional store for
/// member checkpoints and whether `ask_question` is offered. `Default` holds
/// the safe values: depth 3, no base limits, the member persona budget, no
/// store, `ask_question` on.
#[non_exhaustive]
#[derive(Clone)]
pub struct DelegateSettings {
    max_depth: u32,
    max_run_usd_micros: Option<u64>,
    limits: Limits,
    store: Option<Arc<dyn CheckpointStore>>,
    ask_question: bool,
}

impl Default for DelegateSettings {
    fn default() -> Self {
        Self {
            max_depth: DEFAULT_MAX_DEPTH,
            max_run_usd_micros: None,
            limits: Limits::default(),
            store: None,
            ask_question: true,
        }
    }
}

impl std::fmt::Debug for DelegateSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DelegateSettings")
            .field("max_depth", &self.max_depth)
            .field("max_run_usd_micros", &self.max_run_usd_micros)
            .field("limits", &self.limits)
            .field("store", &self.store.is_some())
            .field("ask_question", &self.ask_question)
            .finish()
    }
}

impl DelegateSettings {
    /// Sets how deep agents may call agents, 1 to 16; a call above it is a
    /// tool error, and `AgentTool::new` refuses a value outside the range.
    /// Returns the settings.
    pub fn with_max_depth(mut self, depth: u32) -> Self {
        self.max_depth = depth;
        self
    }

    /// Caps each member run at `micros` millionths of a US dollar, below
    /// the member persona's own budget when that is lower.
    /// Returns the settings.
    pub fn with_max_run_usd_micros(mut self, micros: u64) -> Self {
        self.max_run_usd_micros = Some(micros);
        self
    }

    /// Starts each member run under `limits`, usually the caller's, so
    /// member spend is recorded under the same scope and user period keys
    /// of the shared ledger and their period limits hold. The run cap and
    /// the persona budget only tighten them. Returns the settings.
    pub fn with_limits(mut self, limits: Limits) -> Self {
        self.limits = limits;
        self
    }

    /// Checkpoints member runs in `store`, so a re-run of the same call
    /// resumes the member instead of starting it again.
    /// Returns the settings.
    pub fn with_store(mut self, store: Arc<dyn CheckpointStore>) -> Self {
        self.store = Some(store);
        self
    }

    /// Turns the `ask_question` delegation tool on or off; `delegate_work`
    /// is always offered.
    /// Returns the settings.
    pub fn with_ask_question(mut self, on: bool) -> Self {
        self.ask_question = on;
        self
    }

    /// Returns the delegation depth cap.
    /// A nested call deeper than this is a tool error.
    /// Default 3.
    pub fn max_depth(&self) -> u32 {
        self.max_depth
    }

    /// Returns the member run spend cap, if set.
    /// The member persona budget applies on top when lower.
    /// Default none.
    pub fn max_run_usd_micros(&self) -> Option<u64> {
        self.max_run_usd_micros
    }

    /// Returns the base limits of member runs.
    /// Their scope and user keys decide where member spend is recorded.
    /// Default no limits.
    pub fn limits(&self) -> &Limits {
        &self.limits
    }

    /// Returns the member checkpoint store, if set.
    /// Without one, member runs keep their checkpoints in memory for the
    /// length of the call and never resume.
    pub fn store(&self) -> Option<&Arc<dyn CheckpointStore>> {
        self.store.as_ref()
    }

    /// Returns whether the `ask_question` tool is offered.
    /// `delegate_work` is always offered.
    /// Default on.
    pub fn ask_question(&self) -> bool {
        self.ask_question
    }

    fn member_limits(&self) -> Limits {
        let mut limits = self.limits.clone();
        limits.run = match (limits.run, self.max_run_usd_micros) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        assert!(
            limits.run <= self.limits.run.or(limits.run),
            "the cap only tightens"
        );
        limits
    }
}

/// One member run in flight, entered in the process-wide table of active
/// member runs with its delegation depth and removed on drop. `ToolContext`
/// carries only the caller's run id, so this table is how a nested call
/// learns its depth, including through a cycle of agents.
struct Active {
    member: RunId,
}

impl Active {
    fn enter(
        caller: Option<RunId>,
        member: RunId,
        max_depth: u32,
    ) -> Result<(Active, u32), String> {
        let mut table = ACTIVE.lock().unwrap_or_else(|p| p.into_inner());
        let depth = caller.and_then(|c| table.get(&c).copied()).unwrap_or(0) + 1;
        if depth > max_depth {
            return Err(format!(
                "delegation depth {depth} is above max_depth {max_depth}"
            ));
        }
        if table.contains_key(&member) {
            return Err(format!("member run {member} is already running"));
        }
        if table.len() >= ACTIVE_MEMBERS_MAX {
            return Err(format!(
                "more than {ACTIVE_MEMBERS_MAX} member runs are active"
            ));
        }
        table.insert(member, depth);
        assert!(depth >= 1, "a member is at least one level deep");
        assert!(
            table.len() <= ACTIVE_MEMBERS_MAX,
            "the active table is bounded"
        );
        Ok((Active { member }, depth))
    }
}

impl Drop for Active {
    fn drop(&mut self) {
        let mut table = ACTIVE.lock().unwrap_or_else(|p| p.into_inner());
        let removed = table.remove(&self.member);
        assert!(
            removed.is_some(),
            "an active member is in the table until dropped"
        );
    }
}

/// The member run id of a call: derived from the caller's run id and the
/// call's idempotency key, so a re-run of the same call finds the same
/// member run. A context without both gets a fresh id and never resumes.
/// Returns the id.
pub fn member_run_id(ctx: &ToolContext) -> RunId {
    match (ctx.run_id, ctx.idempotency_key.as_deref()) {
        (Some(run_id), Some(key)) => RunId::derived(run_id, key),
        _ => RunId::new(),
    }
}

/// An agent called as a tool: the model sends `{task, context?}`, the
/// persona runs to its end inside the call and its final text comes back
/// fenced as untrusted output with the member run id, in a JSON object.
/// Members see only `task` and `context`, never the caller's messages.
pub struct AgentTool {
    spec: ToolSpec,
    persona: Persona,
    io: Arc<dyn AgentIo>,
    settings: DelegateSettings,
}

impl AgentTool {
    /// Builds tool `name` described by `description` that runs `persona`
    /// over `io` within `settings`. Returns `Setup` for a name outside
    /// `[A-Za-z0-9_-]{1,64}`, an empty description, a depth cap outside 1
    /// to 16 or a persona tool `io` does not register.
    pub fn new(
        name: &str,
        description: &str,
        persona: Persona,
        io: Arc<dyn AgentIo>,
        settings: DelegateSettings,
    ) -> Result<AgentTool, AgentError> {
        let setup = |m: String| Err(AgentError::Setup(format!("agent tool `{name}`: {m}")));
        if !valid_tool_name(name) {
            return setup("the name must be 1 to 64 of [A-Za-z0-9_-]".into());
        }
        if description.trim().is_empty() {
            return setup("the description is empty".into());
        }
        if !(1..=DEPTH_MAX).contains(&settings.max_depth) {
            return setup(format!("max_depth must be 1 to {DEPTH_MAX}"));
        }
        if let Some(t) = persona.tools.iter().find(|t| io.tools().get(t).is_none()) {
            return setup(format!("persona tool `{t}` is not registered"));
        }
        let spec = ToolSpec {
            name: name.to_string(),
            description: description.to_string(),
            schema: json!({
                "type": "object",
                "properties": {"task": {"type": "string"}, "context": {"type": "string"}},
                "required": ["task"],
                "additionalProperties": false
            }),
            hosted: None,
        };
        assert_eq!(spec.name, name);
        Ok(AgentTool {
            spec,
            persona,
            io,
            settings,
        })
    }

    async fn start(
        &self,
        member: RunId,
        task: &str,
        context: Option<&str>,
    ) -> Result<AgentRun, String> {
        let mut text = task.to_string();
        if let Some(c) = context.filter(|c| !c.trim().is_empty()) {
            text.push_str("\n\nContext:\n");
            text.push_str(c);
        }
        let input = vec![Message::text(Role::User, text)];
        let limits = self.settings.member_limits();
        let run = AgentRun::new(&self.persona, input, limits, self.io.as_ref())
            .await
            .map_err(|e| e.to_string())?;
        Ok(run.with_run_id(member))
    }

    async fn restore(&self, member: RunId) -> Result<Option<AgentRun>, String> {
        let Some(store) = &self.settings.store else {
            return Ok(None);
        };
        let Some(cp) = store.latest(member).await.map_err(|e| e.to_string())? else {
            return Ok(None);
        };
        let run = AgentRun::from_checkpoint(&cp).map_err(|e| e.to_string())?;
        assert_eq!(
            run.run_id(),
            member,
            "a member checkpoint holds its own run"
        );
        Ok(Some(run))
    }

    async fn drive(&self, run: &mut AgentRun) -> Result<StepOutcome, String> {
        let local = InMemCheckpoints::default();
        let store: &dyn CheckpointStore = match &self.settings.store {
            Some(s) => s.as_ref(),
            None => &local,
        };
        let err = |e: troupekit_core::error::CoreError| format!("member checkpoint: {e}");
        if store.latest(run.run_id()).await.map_err(err)?.is_none() {
            store.save(run.checkpoint()).await.map_err(err)?;
        }
        let steps_max = STEPS_PER_TURN * u64::from(self.persona.max_turns) + STEPS_SLACK;
        for _ in 0..steps_max {
            let outcome = run.step(self.io.as_ref()).await;
            store.save(run.checkpoint()).await.map_err(err)?;
            if outcome != StepOutcome::Continue {
                return Ok(outcome);
            }
        }
        Err(format!("member run took more than {steps_max} steps"))
    }

    fn reply(&self, member: RunId, outcome: StepOutcome) -> ToolOutput {
        let source = format!("agent {}", self.spec.name);
        let fence = |text: &str| fence_capped(&source, text, ANSWER_CHARS_MAX, false);
        let (key, text, is_error) = match outcome {
            StepOutcome::Done(Value::String(s)) => ("answer", s, false),
            StepOutcome::Done(v) => ("answer", v.to_string(), false),
            StepOutcome::Failed(e) => ("error", format!("member run failed: {e}"), true),
            StepOutcome::Paused(r) => {
                let reason = serde_json::to_string(&r).expect("a pause reason serializes");
                ("error", format!("member run paused: {reason}"), true)
            }
            StepOutcome::Retry { error, .. } => (
                "error",
                format!("member run hit a transient failure: {error}"),
                true,
            ),
            StepOutcome::Continue => unreachable!("drive returns only a stopping outcome"),
        };
        let body = json!({"member_run_id": member.to_string(), key: fence(&text)}).to_string();
        assert!(
            body.contains(&member.to_string()),
            "the reply names the member run"
        );
        if is_error {
            ToolOutput::error(body)
        } else {
            ToolOutput::text(body)
        }
    }

    async fn run_member(&self, member: RunId, task: &str, context: Option<&str>) -> ToolOutput {
        let restored = match self.restore(member).await {
            Ok(r) => r,
            Err(e) => return self.reply(member, StepOutcome::Failed(e)),
        };
        let mut run = match restored {
            Some(run) => run,
            None => match self.start(member, task, context).await {
                Ok(run) => run,
                Err(e) => return self.reply(member, StepOutcome::Failed(e)),
            },
        };
        let outcome = match run.state() {
            RunState::Done { output } => StepOutcome::Done(output.clone()),
            RunState::Failed { error } => StepOutcome::Failed(error.clone()),
            RunState::Paused { reason, .. } => StepOutcome::Paused(reason.clone()),
            _ => self
                .drive(&mut run)
                .await
                .unwrap_or_else(StepOutcome::Failed),
        };
        self.reply(member, outcome)
    }
}

fn text_arg<'a>(args: &'a Value, key: &str, required: bool) -> Result<Option<&'a str>, ToolError> {
    match args.get(key) {
        None | Some(Value::Null) if !required => Ok(None),
        Some(Value::String(s)) if !(required && s.trim().is_empty()) => Ok(Some(s)),
        _ => Err(ToolError::InvalidArgs(format!(
            "`{key}` must be a non-empty string"
        ))),
    }
}

#[async_trait]
impl Tool for AgentTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let task = text_arg(&args, "task", true)?.expect("a required argument is present");
        let context = text_arg(&args, "context", false)?;
        let member = member_run_id(ctx);
        let (active, depth) = Active::enter(ctx.run_id, member, self.settings.max_depth)
            .map_err(ToolError::Failed)?;
        assert!(
            depth <= self.settings.max_depth,
            "depth is checked on entry"
        );
        let out = self.run_member(member, task, context).await;
        drop(active);
        Ok(out)
    }
}

/// The `delegate_work` and `ask_question` tools of a leader over a named
/// set of coworker agents, each listed as an enum value in the specs.
/// Each call runs the named coworker's `AgentTool` with the same context.
pub struct DelegationTools {
    tools: Vec<Arc<dyn Tool>>,
}

impl DelegationTools {
    /// Builds the delegation tools over `coworkers`, each a name and its
    /// agent tool, within `settings`. Returns `Setup` for an empty set,
    /// more than 64 coworkers, an empty name or a duplicate name.
    pub fn new(
        coworkers: Vec<(String, AgentTool)>,
        settings: DelegateSettings,
    ) -> Result<DelegationTools, AgentError> {
        let setup = |m: String| Err(AgentError::Setup(format!("delegation tools: {m}")));
        if coworkers.is_empty() {
            return setup("the coworker set is empty".into());
        }
        if coworkers.len() > COWORKERS_MAX {
            return setup(format!("more than {COWORKERS_MAX} coworkers"));
        }
        let mut names: Vec<&str> = Vec::with_capacity(coworkers.len());
        for (name, _) in &coworkers {
            if name.trim().is_empty() {
                return setup("a coworker name is empty".into());
            }
            if names.contains(&name.as_str()) {
                return setup(format!("coworker `{name}` is listed twice"));
            }
            names.push(name);
        }
        let names: Vec<String> = names.into_iter().map(str::to_string).collect();
        let coworkers = Arc::new(coworkers);
        let mut tools: Vec<Arc<dyn Tool>> =
            vec![Arc::new(Delegate::new(false, &names, &coworkers))];
        if settings.ask_question {
            tools.push(Arc::new(Delegate::new(true, &names, &coworkers)));
        }
        assert!(!tools.is_empty(), "delegate_work is always offered");
        assert!(tools.len() <= 2, "at most the two delegation tools");
        Ok(DelegationTools { tools })
    }

    /// Returns the tools to register for the leader: `delegate_work`,
    /// then `ask_question` when the settings offer it.
    /// The list is never empty.
    pub fn tools(&self) -> Vec<Arc<dyn Tool>> {
        self.tools.clone()
    }
}

/// One delegation tool: `delegate_work` or, with `ask`, `ask_question`.
/// It holds the shared coworker list and calls the named coworker's
/// `AgentTool` with the caller's context.
struct Delegate {
    spec: ToolSpec,
    ask: bool,
    coworkers: Arc<Vec<(String, AgentTool)>>,
}

impl Delegate {
    fn new(ask: bool, names: &[String], coworkers: &Arc<Vec<(String, AgentTool)>>) -> Delegate {
        let (name, field, description) = if ask {
            (
                ASK_QUESTION,
                "question",
                "Ask one coworker a question and get the answer.",
            )
        } else {
            (
                DELEGATE_WORK,
                "task",
                "Hand one coworker a task and get the result.",
            )
        };
        let schema = json!({
            "type": "object",
            "properties": {
                "coworker": {"type": "string", "enum": names},
                field: {"type": "string"},
                "context": {"type": "string"}
            },
            "required": ["coworker", field],
            "additionalProperties": false
        });
        assert_eq!(names.len(), coworkers.len(), "one enum value per coworker");
        Delegate {
            spec: ToolSpec {
                name: name.to_string(),
                description: description.to_string(),
                schema,
                hosted: None,
            },
            ask,
            coworkers: coworkers.clone(),
        }
    }
}

#[async_trait]
impl Tool for Delegate {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        let name = text_arg(&args, "coworker", true)?.expect("a required argument is present");
        let Some((_, tool)) = self.coworkers.iter().find(|(n, _)| n == name) else {
            let known: Vec<&str> = self.coworkers.iter().map(|(n, _)| n.as_str()).collect();
            return Err(ToolError::InvalidArgs(format!(
                "unknown coworker `{name}`; one of {}",
                known.join(", ")
            )));
        };
        let field = if self.ask { "question" } else { "task" };
        let text = text_arg(&args, field, true)?.expect("a required argument is present");
        let task = if self.ask {
            format!("Answer this question: {text}")
        } else {
            text.to_string()
        };
        let mut member_args = json!({ "task": task });
        if let Some(c) = text_arg(&args, "context", false)? {
            member_args["context"] = json!(c);
        }
        tool.call(member_args, ctx).await
    }
}
