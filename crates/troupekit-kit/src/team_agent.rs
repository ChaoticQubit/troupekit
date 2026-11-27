use crate::agent::Agent;
use crate::built::{describe, member_settings, BuiltAgent, Input, RunOutcome};
use crate::error::KitError;
use async_trait::async_trait;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use toml::Value as Toml;
use troupekit_agent::delegate::{AgentTool, DelegateSettings, DelegationTools};
use troupekit_agent::{AgentIo, EvalFn, MemorySource, PersonaLoader, SimpleIo};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::CheckpointStore;
use troupekit_graph::flow::Flow;
use troupekit_graph::patterns;
use troupekit_graph::{GraphOutcome, GraphRun, SimpleGraphIo};
use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolRegistry, ToolSpec};

const MEMBERS_MAX: usize = 64;
const NEST_MAX: u32 = 16;

/// The member agents of a leader: agents called as tools and coworkers
/// for the delegation tools, built with the leader. After the build they
/// are gone and only the names of the tools they became remain.
#[derive(Clone, Default)]
pub(crate) struct Members {
    agent_tools: Vec<Agent>,
    delegates: Vec<Agent>,
    tool_names: Vec<String>,
    nest: u32,
    team: Option<(TeamMode, Vec<Agent>)>,
    handoffs: Vec<Agent>,
}

/// How a leader works with its `members`: `Route` picks one member,
/// `Coordinate` delegates to them, `Collaborate` briefs all and combines
/// their answers, `Broadcast` briefs all and keeps every answer. `Tasks`
/// delegates in passes until the leader says it is done, at most
/// `max_iterations` passes (5); a run still not done after the last pass
/// fails. Member spend through delegation is not counted in the run caps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TeamMode {
    Route,
    #[default]
    Coordinate,
    Collaborate,
    Broadcast,
    Tasks,
}

impl TeamMode {
    /// Returns the mode as the team pattern names it.
    /// One lowercase word per mode.
    /// The inverse of the pattern's `mode` field.
    pub fn as_str(self) -> &'static str {
        match self {
            TeamMode::Route => "route",
            TeamMode::Coordinate => "coordinate",
            TeamMode::Collaborate => "collaborate",
            TeamMode::Broadcast => "broadcast",
            TeamMode::Tasks => "tasks",
        }
    }
}

impl Members {
    /// Returns true when no member waits to be built.
    /// An agent without the 0.5 methods always returns true.
    pub(crate) fn is_empty(&self) -> bool {
        !self.has_tools() && !self.is_team()
    }

    /// Returns true when agent tools or delegates wait to be built.
    /// They build before a team does, as tools of the leader.
    /// False for an agent without them.
    pub(crate) fn has_tools(&self) -> bool {
        !self.agent_tools.is_empty() || !self.delegates.is_empty()
    }

    /// Returns true when the agent leads a team of `members` or hands
    /// off to `handoffs`. Such an agent builds into a flow run, not a
    /// single agent run. False for every other agent.
    pub(crate) fn is_team(&self) -> bool {
        self.team.is_some() || !self.handoffs.is_empty()
    }

    /// Returns the names of the tools the members became at build.
    /// Empty before the build and for an agent without members.
    pub(crate) fn tool_names(&self) -> &[String] {
        &self.tool_names
    }
}

impl Agent {
    /// Adds `member` as a tool named after it. The model sends `{task,
    /// context?}`; the member runs to its end inside the call, with its own
    /// model, tools, limits and ledger, and its answer comes back fenced as
    /// untrusted text. Transient member failures come back as a tool error.
    pub fn agent_tool(mut self, member: Agent) -> Self {
        self.members.agent_tools.push(member);
        self
    }

    /// Adds `coworkers` behind two tools, `delegate_work` and
    /// `ask_question`, each naming one coworker. Coworkers run like
    /// `agent_tool` members. `build` refuses an empty set, duplicate names
    /// and more than 64 members.
    pub fn delegates(mut self, coworkers: impl IntoIterator<Item = Agent>) -> Self {
        self.members.delegates.extend(coworkers);
        self
    }

    /// Makes this agent the leader of `members`, working in `mode`. The
    /// agent and its members build into one flow from the team pattern and
    /// run as a graph behind `run` and `run_with`. Members run under the
    /// leader's limits, ledger and storage, so `build` refuses their own.
    pub fn members(mut self, members: impl IntoIterator<Item = Agent>, mode: TeamMode) -> Self {
        let mut list = self.members.team.take().map(|(_, l)| l).unwrap_or_default();
        list.extend(members);
        self.members.team = Some((mode, list));
        self
    }

    /// Lets this agent answer a request itself or hand it to one of
    /// `agents`, which then answers it. The agents build into one flow from
    /// the handoff pattern, whose steps (not hops) are capped from its
    /// default `max_handoffs`, and run behind `run` and `run_with`.
    pub fn handoffs(mut self, agents: impl IntoIterator<Item = Agent>) -> Self {
        self.members.handoffs.extend(agents);
        self
    }

    /// Makes this agent the manager of `members`: it hands them work with
    /// `delegate_work` and `ask_question` as an agent node of a flow, and
    /// gives the final answer. Provisional: compiles to the team pattern's
    /// `coordinate` mode until crews with tasks reach the kit.
    pub fn manages(self, members: impl IntoIterator<Item = Agent>) -> Self {
        self.members(members, TeamMode::Coordinate)
    }

    /// Builds every member and registers the tools they become. Returns
    /// the leader with no members left and the tool names recorded.
    /// Fails with the first member build error or a members limit.
    pub(crate) async fn attach_members(mut self) -> Result<Agent, KitError> {
        let members = std::mem::take(&mut self.members);
        let leader = self.name().to_string();
        assert!(
            members.has_tools(),
            "attach runs only for an agent with tool members"
        );
        if members.nest >= NEST_MAX {
            let m = format!("members nest deeper than {NEST_MAX} agents");
            return Err(KitError::config(&leader, m));
        }
        if members.agent_tools.len() + members.delegates.len() > MEMBERS_MAX {
            let m = format!("more than {MEMBERS_MAX} member agents");
            return Err(KitError::config(&leader, m));
        }
        let mut names = Vec::new();
        for member in members.agent_tools {
            let built = build_member(member, members.nest, &leader).await?;
            names.push(built.persona().name.clone());
            self = self.tool_arc(built.agent_tool().clone());
        }
        if !members.delegates.is_empty() {
            let mut coworkers = Vec::with_capacity(members.delegates.len());
            for member in members.delegates {
                let built = build_member(member, members.nest, &leader).await?;
                let persona = built.persona();
                let settings = member_settings(built.limits(), built.store());
                let tool = AgentTool::new(
                    &persona.name,
                    &describe(persona),
                    persona.clone(),
                    built.io(),
                    settings,
                )?;
                coworkers.push((persona.name.clone(), tool));
            }
            let set = DelegationTools::new(coworkers, DelegateSettings::default())?;
            for tool in set.tools() {
                names.push(tool.spec().name.clone());
                self = self.tool_arc(tool);
            }
        }
        assert!(!names.is_empty(), "every member became a tool");
        self.members = Members {
            tool_names: names,
            nest: members.nest,
            team: members.team,
            handoffs: members.handoffs,
            ..Members::default()
        };
        assert!(!self.members.has_tools());
        Ok(self)
    }
}

async fn build_member(mut member: Agent, nest: u32, leader: &str) -> Result<BuiltAgent, KitError> {
    member.members.nest = nest + 1;
    let built = Box::pin(member.build()).await?;
    if built.is_team() {
        let m = format!(
            "member `{}` leads a team, which cannot run as a tool yet",
            built.persona().name
        );
        return Err(KitError::config(leader, m));
    }
    if built.has_preamble() {
        let m = format!(
            "member `{}` has memory or knowledge, which member runs do not read yet",
            built.persona().name
        );
        return Err(KitError::config(leader, m));
    }
    Ok(built)
}

#[async_trait]
impl Tool for BuiltAgent {
    fn spec(&self) -> &ToolSpec {
        self.agent_tool().spec()
    }

    async fn call(&self, args: Value, ctx: &ToolContext) -> Result<ToolOutput, ToolError> {
        if self.is_team() {
            return Err(ToolError::Failed(format!(
                "agent `{}` leads a team, which cannot run as a tool yet",
                self.persona().name
            )));
        }
        if self.has_preamble() {
            return Err(ToolError::Failed(format!(
                "agent `{}` has memory or knowledge, which a tool call does not read yet",
                self.persona().name
            )));
        }
        self.agent_tool().call(args, ctx).await
    }
}

const TEAM_MEMBERS_MAX: usize = 32;
const RETRY_WAIT_MAX_MS: i64 = 30_000;

/// A team agent's flow and the graph io its nodes run on: the leader's
/// and members' personas, models, tools and evals, the leader's ledger and
/// events, and the state key that holds the answer: `answers` for
/// broadcast, which keeps every member's answer, else `output`, and for
/// `tasks` that a run ending with `done` false fails. Built once at
/// `build`, shared by every run.
pub(crate) struct TeamFlow {
    flow: Flow,
    io: SimpleGraphIo,
    answer: &'static str,
    tasks: bool,
}

impl Agent {
    /// Builds a team agent: the leader alone, for its persona, limits and
    /// storage, then the team flow over leader and members. Fails on the
    /// first problem: mode, members, a member's own settings, or the flow.
    pub(crate) async fn build_team(mut self) -> Result<BuiltAgent, KitError> {
        let name = self.name().to_string();
        let (text, members, answer, tasks) = self.pattern()?;
        assert!(!self.members.is_team(), "the pattern took every member");
        if members.is_empty() || members.len() > TEAM_MEMBERS_MAX {
            let m = format!(
                "a team takes 1 to {TEAM_MEMBERS_MAX} members, got {}",
                members.len()
            );
            return Err(KitError::config(&name, m));
        }
        for m in &members {
            if let Some(what) = m.graph_conflict() {
                let what = what.replace("the graph", "the leader");
                return Err(KitError::config(
                    &name,
                    format!("member `{}` sets its own {what}", m.name()),
                ));
            }
        }
        let leader = Box::pin(self.clone().build()).await?;
        if leader.has_preamble() {
            let m = "a team leader with memory or knowledge is not built yet";
            return Err(KitError::config(&name, m));
        }
        let mut all: Vec<&Agent> = vec![&self];
        all.extend(members.iter());
        let io = team_io(&name, &all, &leader)?;
        let flow = patterns::load(&format!("agent:{name}"), &text)?;
        let probe = input_state("build check");
        GraphRun::start(flow.clone(), probe, leader.limits().clone(), &io).await?;
        assert!(!flow.workflow().nodes.is_empty());
        Ok(leader.with_team(TeamFlow {
            flow,
            io,
            answer,
            tasks,
        }))
    }

    /// Takes the team or the handoffs off this agent and returns the
    /// pattern TOML, the other agents, the state key of the answer
    /// (`answers` for broadcast, else `output`) and whether the mode is
    /// `tasks`. Fails for members and handoffs at once.
    fn pattern(&mut self) -> Result<(String, Vec<Agent>, &'static str, bool), KitError> {
        let name = self.name().to_string();
        let team = self.members.team.take();
        let handoffs = std::mem::take(&mut self.members.handoffs);
        match (team, handoffs.is_empty()) {
            (Some(_), false) => {
                let m = "an agent takes members or handoffs, not both";
                Err(KitError::config(&name, m))
            }
            (Some((mode, members)), true) => {
                let text = team_text(&name, mode, &members)?;
                let answer = if mode == TeamMode::Broadcast {
                    "answers"
                } else {
                    "output"
                };
                Ok((text, members, answer, mode == TeamMode::Tasks))
            }
            (None, false) => Ok((handoff_text(&name, &handoffs)?, handoffs, "output", false)),
            (None, true) => panic!("build_team runs for a team or handoffs"),
        }
    }
}

fn handoff_text(name: &str, agents: &[Agent]) -> Result<String, KitError> {
    let names: Vec<Toml> = agents
        .iter()
        .map(|a| Toml::String(a.name().to_string()))
        .collect();
    assert_eq!(names.len(), agents.len());
    let mut allow = toml::Table::new();
    allow.insert(name.into(), Toml::Array(names));
    let mut handoff = toml::Table::new();
    handoff.insert("start".into(), Toml::String(name.into()));
    handoff.insert("allow".into(), Toml::Table(allow));
    let mut table = toml::Table::new();
    table.insert("name".into(), Toml::String(name.into()));
    table.insert("handoff".into(), Toml::Table(handoff));
    toml::to_string(&table)
        .map_err(|e| KitError::config(name, format!("handoffs do not serialize: {e}")))
}

fn team_text(name: &str, mode: TeamMode, members: &[Agent]) -> Result<String, KitError> {
    let mut team = toml::Table::new();
    team.insert("mode".into(), Toml::String(mode.as_str().into()));
    team.insert("leader".into(), Toml::String(name.into()));
    let names = members
        .iter()
        .map(|m| Toml::String(m.name().to_string()))
        .collect();
    team.insert("members".into(), Toml::Array(names));
    let mut table = toml::Table::new();
    table.insert("name".into(), Toml::String(name.into()));
    table.insert("team".into(), Toml::Table(team));
    toml::to_string(&table)
        .map_err(|e| KitError::config(name, format!("team does not serialize: {e}")))
}

fn team_io(name: &str, agents: &[&Agent], leader: &BuiltAgent) -> Result<SimpleGraphIo, KitError> {
    let fail = |m: String| KitError::config(name, m);
    let mut registry = ToolRegistry::new();
    let mut agent_io = SimpleIo::new(ToolRegistry::new());
    let mut files: HashMap<String, String> = HashMap::new();
    let mut evals: HashMap<String, EvalFn> = HashMap::new();
    for (i, agent) in agents.iter().enumerate() {
        let key = format!("agent-{i}");
        let prepared = agent.prepare(Some(&key))?;
        let file = format!("{}.toml", prepared.persona.name);
        if files.insert(file, prepared.persona_text).is_some() {
            return Err(fail(format!(
                "two team agents are named `{}`",
                agent.name()
            )));
        }
        for tool in &prepared.tools {
            share(&mut registry, tool.clone()).map_err(fail)?;
        }
        agent_io.models.insert(key, prepared.handle);
        for (eval, f) in prepared.eval_fns {
            match evals.get(&eval) {
                Some(first) if Arc::ptr_eq(first, &f) => {}
                Some(_) => return Err(fail(format!("eval `{eval}` has two different callbacks"))),
                None => {
                    agent_io.evals.register(&eval, f.clone())?;
                    evals.insert(eval, f);
                }
            }
        }
    }
    assert_eq!(files.len(), agents.len(), "one persona file per team agent");
    let own = leader.io();
    agent_io.tools = registry;
    agent_io.ledger = own.ledger.clone();
    agent_io.events = own.events.clone();
    let agent_io: Arc<dyn AgentIo> = Arc::new(agent_io);
    let loader = PersonaLoader::new(vec![Arc::new(MemorySource::new("team", files))]);
    Ok(SimpleGraphIo::new(agent_io, loader))
}

fn share(registry: &mut ToolRegistry, tool: Arc<dyn Tool>) -> Result<(), String> {
    let name = tool.spec().name.clone();
    match registry.get(&name) {
        Some(existing) if Arc::ptr_eq(&existing, &tool) => Ok(()),
        Some(_) => Err(format!(
            "tool `{name}` is registered by two team agents; pass one shared tool"
        )),
        None => registry.register(tool).map_err(|e| e.to_string()),
    }
}

fn input_state(text: &str) -> Map<String, Value> {
    let mut state = Map::new();
    state.insert("input".into(), Value::String(text.to_string()));
    state
}

impl TeamFlow {
    /// Starts a team run on `input`, one user text, and runs it to its end.
    /// Returns the outcome with the `output` key as the answer; fails like
    /// `BuiltAgent::run_with`, or for an input that is not one text.
    pub(crate) async fn run(
        &self,
        agent: &BuiltAgent,
        input: Input,
    ) -> Result<RunOutcome, KitError> {
        let name = &agent.persona().name;
        let messages = input.into_messages();
        let text = match messages.as_slice() {
            [m] if m.role == Role::User && !m.text_content().is_empty() => m.text_content(),
            _ => return Err(KitError::config(name, "a team agent runs on one user text")),
        };
        let state = input_state(&text);
        let limits = agent.limits().clone();
        let mut run = GraphRun::start(self.flow.clone(), state, limits, &self.io).await?;
        let store: Arc<dyn CheckpointStore> = match agent.store() {
            Some(store) => store.clone(),
            None => Arc::new(InMemCheckpoints::default()),
        };
        self.finish(&mut run, store.as_ref(), &text).await
    }

    /// Continues team run `run_id` from its latest checkpoint in `store`
    /// and returns the answer. Fails when the checkpoint is not a run of
    /// this team's flow.
    pub(crate) async fn resume(
        &self,
        agent: &BuiltAgent,
        store: &dyn CheckpointStore,
        run_id: RunId,
    ) -> Result<String, KitError> {
        let mut run = GraphRun::restore(store, run_id).await?;
        if run.flow() != self.flow {
            let m =
                format!("run {run_id} was not started by this team; resume it with its own agent");
            return Err(KitError::config(&agent.persona().name, m));
        }
        let input = run
            .state()
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let input = input.to_string();
        Ok(self
            .finish(&mut run, store, &input)
            .await?
            .text()
            .to_string())
    }

    async fn finish(
        &self,
        run: &mut GraphRun,
        store: &dyn CheckpointStore,
        input: &str,
    ) -> Result<RunOutcome, KitError> {
        if store.latest(run.run_id()).await?.is_none() {
            store.save(run.checkpoint()).await?;
        }
        let workflow = self.flow.workflow();
        let cap = 2 * u64::from(workflow.max_steps) + u64::from(workflow.max_retries) + 8;
        for _ in 0..cap {
            let before = run.steps();
            let outcome = run.step(&self.io).await;
            if run.steps() != before {
                store.save(run.checkpoint()).await?;
            }
            match outcome {
                GraphOutcome::Continue => {}
                GraphOutcome::RetryAt { at_ms, .. } => {
                    let wait = at_ms.saturating_sub(self.io.agent.clock().now_ms());
                    let wait = u64::try_from(wait.clamp(0, RETRY_WAIT_MAX_MS)).expect("clamped");
                    tokio::time::sleep(Duration::from_millis(wait)).await;
                }
                GraphOutcome::Waiting(reason) => {
                    return Err(KitError::Paused {
                        run_id: run.run_id(),
                        reason,
                    })
                }
                GraphOutcome::Failed(reason) => {
                    return Err(KitError::Failed {
                        run_id: run.run_id(),
                        reason,
                    })
                }
                GraphOutcome::Done(state) => {
                    if self.tasks && state.get("done") != Some(&Value::Bool(true)) {
                        return Err(KitError::Failed {
                            run_id: run.run_id(),
                            reason: "the tasks leader was not done after its last pass".into(),
                        });
                    }
                    return Ok(team_outcome(run, state, self.answer, input));
                }
            }
        }
        Err(KitError::Failed {
            run_id: run.run_id(),
            reason: format!("did not finish within {cap} steps"),
        })
    }
}

fn team_outcome(
    run: &GraphRun,
    mut state: Map<String, Value>,
    key: &str,
    input: &str,
) -> RunOutcome {
    let output = state.remove(key).unwrap_or(Value::Null);
    assert!(!state.contains_key(key));
    let text = match &output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    let messages = vec![
        Message::text(Role::User, input),
        Message::text(Role::Assistant, text.clone()),
    ];
    RunOutcome::team(text, output, run.usage(), run.run_id(), messages)
}
