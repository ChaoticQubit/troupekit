use crate::error::KitError;
use crate::laws::Laws;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use troupekit_agent::delegate::{AgentTool, DelegateSettings};
use troupekit_agent::{
    drive, AgentIo, AgentRun, EventSink, Limits, Persona, RunState, SimpleIo, StepOutcome,
};
use troupekit_core::event::{Event, Usage};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::store::CheckpointStore;
use troupekit_tools::Tool;

const BASE_RETRY_MS: u64 = 500;
const MAX_RETRY_MS: u64 = 30_000;

pub(crate) struct Discard;

impl EventSink for Discard {
    fn emit(&self, _event: Event) {}
}

/// What an agent runs on: one user text, or a prepared conversation.
/// Built from `&str`, `String`, one `Message` or a `Vec<Message>`.
/// An empty conversation fails when the run starts.
#[derive(Debug, Clone, PartialEq)]
pub struct Input(Vec<Message>);

impl Input {
    pub(crate) fn into_messages(self) -> Vec<Message> {
        self.0
    }
}

impl From<&str> for Input {
    fn from(text: &str) -> Self {
        Input(vec![Message::text(Role::User, text)])
    }
}

impl From<String> for Input {
    fn from(text: String) -> Self {
        Input(vec![Message::text(Role::User, text)])
    }
}

impl From<Message> for Input {
    fn from(message: Message) -> Self {
        Input(vec![message])
    }
}

impl From<Vec<Message>> for Input {
    fn from(messages: Vec<Message>) -> Self {
        Input(messages)
    }
}

/// The result of one finished run: the answer and how it was reached.
/// `text` is the answer as text; `output` is the raw final value.
/// Read it through the getters.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct RunOutcome {
    text: String,
    output: Value,
    usage: Usage,
    model_calls: u32,
    run_id: RunId,
    messages: Vec<Message>,
    pub(crate) member_runs: Vec<RunId>,
}

impl RunOutcome {
    /// Returns the answer as text: a string output as is, any other
    /// output as JSON text.
    /// The same text `run` returns.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// Returns the final output value of the run.
    /// A string for plain agents, a JSON value for typed output.
    /// Never null for a finished run.
    pub fn output(&self) -> &Value {
        &self.output
    }

    /// Returns the tokens and cost the run used, summed over calls.
    /// Cost is in millionths of a US dollar.
    /// Includes eval and repair calls.
    pub fn usage(&self) -> Usage {
        self.usage
    }

    /// Returns how many model calls the run made.
    /// Retried calls count once per attempt that reached the model.
    /// At least one for a finished run.
    pub fn model_calls(&self) -> u32 {
        self.model_calls
    }

    /// Returns the id of the run, for `resume` and logs.
    /// Unique per run.
    /// Stable across checkpoints.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns the whole conversation, input first.
    /// Includes tool calls and tool results.
    /// The last message is the answer.
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// Returns the run ids of the member agents this run called, through
    /// `agent_tool` or `delegates`, in call order. Empty for an agent
    /// without members. Each id opens the member's own checkpoints.
    pub fn member_runs(&self) -> &[RunId] {
        &self.member_runs
    }
}

/// An agent that passed every `build` check, ready to run many times.
/// Cheap to clone and safe to share across tasks; each run gets its own
/// state, and in-memory checkpoints are dropped after the run.
#[derive(Clone)]
pub struct BuiltAgent {
    inner: Arc<Inner>,
}

struct Inner {
    persona: Persona,
    io: Arc<SimpleIo>,
    tool: Arc<AgentTool>,
    member_tools: Vec<String>,
    limits: Limits,
    store: Option<Arc<dyn CheckpointStore>>,
    retries: u32,
    laws: Laws,
    #[cfg(feature = "knowledge")]
    preamble: Option<troupekit_context::tools::KnowledgeKit>,
}

impl BuiltAgent {
    pub(crate) fn new(
        persona: Persona,
        io: SimpleIo,
        limits: Limits,
        store: Option<Arc<dyn CheckpointStore>>,
        retries: u32,
        laws: Laws,
    ) -> Self {
        let io = Arc::new(io);
        let settings = member_settings(&limits, store.as_ref());
        let tool = AgentTool::new(
            &persona.name,
            &describe(&persona),
            persona.clone(),
            io.clone(),
            settings,
        )
        .expect("a built persona is a valid agent tool");
        assert_eq!(tool.spec().name, persona.name);
        BuiltAgent {
            inner: Arc::new(Inner {
                persona,
                io,
                tool: Arc::new(tool),
                member_tools: Vec::new(),
                limits,
                store,
                retries,
                laws,
                #[cfg(feature = "knowledge")]
                preamble: None,
            }),
        }
    }

    #[cfg(feature = "knowledge")]
    pub(crate) fn with_preamble(
        mut self,
        preamble: Option<troupekit_context::tools::KnowledgeKit>,
    ) -> Self {
        let inner = Arc::get_mut(&mut self.inner).expect("a fresh BuiltAgent is not shared");
        inner.preamble = preamble;
        self
    }

    #[cfg(feature = "knowledge")]
    async fn messages(&self, input: Input) -> Result<Vec<Message>, KitError> {
        match &self.inner.preamble {
            Some(kit) => crate::memory::prepend(kit, input.0).await,
            None => Ok(input.0),
        }
    }

    #[cfg(not(feature = "knowledge"))]
    async fn messages(&self, input: Input) -> Result<Vec<Message>, KitError> {
        Ok(input.0)
    }

    /// Runs the agent once on `input` and returns the answer as text.
    /// Retries transient provider failures; fails with `Failed`,
    /// `Paused` or `Retries` when the run ends without an answer.
    pub async fn run(&self, input: impl Into<Input>) -> Result<String, KitError> {
        Ok(self.run_with(input).await?.text)
    }

    /// Runs the agent once on `input` and returns the full outcome.
    /// Same failures as `run`.
    /// The run id in the outcome works with `resume`.
    pub async fn run_with(&self, input: impl Into<Input>) -> Result<RunOutcome, KitError> {
        let inner = &self.inner;
        let messages = self.messages(input.into()).await?;
        let mut run = AgentRun::new(
            &inner.persona,
            messages,
            inner.limits.clone(),
            inner.io.as_ref(),
        )
        .await?;
        let store: Arc<dyn CheckpointStore> = match &inner.store {
            Some(store) => store.clone(),
            None => Arc::new(InMemCheckpoints::default()),
        };
        self.finish(&mut run, store.as_ref()).await
    }

    /// Continues run `run_id` from its latest checkpoint, after a crash
    /// or restart, and returns the answer. A finished run returns its
    /// answer again. Needs `Storage::sqlite(..)` or `Storage::custom(..)`.
    pub async fn resume(&self, run_id: RunId) -> Result<String, KitError> {
        let name = self.inner.persona.name.as_str();
        let Some(store) = &self.inner.store else {
            return Err(KitError::config(
                name,
                "resume needs Storage::sqlite(..) or Storage::custom(..) checkpoints",
            ));
        };
        let Some(checkpoint) = store.latest(run_id).await? else {
            return Err(KitError::config(
                name,
                format!("no checkpoint for run {run_id}"),
            ));
        };
        let mut run = AgentRun::from_checkpoint(&checkpoint)?;
        Ok(self.finish(&mut run, store.as_ref()).await?.text)
    }

    /// Returns the validated persona this agent runs.
    /// Built from the builder settings by `Persona::parse`.
    /// Read only.
    pub fn persona(&self) -> &Persona {
        &self.inner.persona
    }

    /// Returns the laws checked at `build`.
    /// 0.2 stores them; runs do not prove them yet.
    /// Read only.
    pub fn laws(&self) -> &Laws {
        &self.inner.laws
    }

    pub(crate) fn agent_tool(&self) -> &Arc<AgentTool> {
        &self.inner.tool
    }

    pub(crate) fn store(&self) -> Option<&Arc<dyn CheckpointStore>> {
        self.inner.store.as_ref()
    }

    pub(crate) fn io(&self) -> Arc<SimpleIo> {
        self.inner.io.clone()
    }

    pub(crate) fn limits(&self) -> &Limits {
        &self.inner.limits
    }

    pub(crate) fn with_member_tools(mut self, names: Vec<String>) -> Self {
        let inner = Arc::get_mut(&mut self.inner).expect("a fresh BuiltAgent is not shared");
        inner.member_tools = names;
        self
    }

    #[cfg(feature = "knowledge")]
    pub(crate) fn has_preamble(&self) -> bool {
        self.inner.preamble.is_some()
    }

    #[cfg(not(feature = "knowledge"))]
    pub(crate) fn has_preamble(&self) -> bool {
        false
    }

    pub(crate) async fn finish(
        &self,
        run: &mut AgentRun,
        store: &dyn CheckpointStore,
    ) -> Result<RunOutcome, KitError> {
        let io: &dyn AgentIo = self.inner.io.as_ref();
        match run.state().clone() {
            RunState::Done { output } => return Ok(outcome(run, output, &self.inner.member_tools)),
            RunState::Failed { error } => {
                return Err(KitError::Failed {
                    run_id: run.run_id(),
                    reason: error,
                })
            }
            RunState::Paused { reason, .. } => {
                return Err(KitError::Paused {
                    run_id: run.run_id(),
                    reason,
                })
            }
            _ => {}
        }
        let mut attempts = 0u32;
        loop {
            match drive(run, io, store).await? {
                StepOutcome::Done(output) => {
                    return Ok(outcome(run, output, &self.inner.member_tools))
                }
                StepOutcome::Retry {
                    error,
                    retry_after_ms,
                } => {
                    if attempts >= self.inner.retries {
                        return Err(KitError::Retries {
                            run_id: run.run_id(),
                            attempts,
                            last_error: error,
                        });
                    }
                    tokio::time::sleep(retry_delay(attempts, retry_after_ms)).await;
                    attempts += 1;
                }
                StepOutcome::Paused(reason) => {
                    return Err(KitError::Paused {
                        run_id: run.run_id(),
                        reason,
                    })
                }
                StepOutcome::Failed(reason) => {
                    return Err(KitError::Failed {
                        run_id: run.run_id(),
                        reason,
                    })
                }
                StepOutcome::Continue => {
                    unreachable!("drive returns only a final, paused or retry outcome")
                }
            }
            assert!(attempts <= self.inner.retries);
        }
    }
}

impl fmt::Debug for BuiltAgent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = &self.inner;
        f.debug_struct("BuiltAgent")
            .field("name", &inner.persona.name)
            .field("model", &inner.persona.model)
            .field("tools", &inner.persona.tools)
            .field("retries", &inner.retries)
            .field("durable", &inner.store.is_some())
            .field("laws", &inner.laws)
            .finish_non_exhaustive()
    }
}

fn outcome(run: &AgentRun, output: Value, member_tools: &[String]) -> RunOutcome {
    let text = match &output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    };
    RunOutcome {
        text,
        output,
        usage: run.usage(),
        model_calls: run.model_calls(),
        run_id: run.run_id(),
        messages: run.messages().to_vec(),
        member_runs: member_runs(run.messages(), member_tools),
    }
}

/// Returns the member run ids named in the replies to calls of
/// `member_tools`, in call order. A result counts only for a call of the
/// assistant message right before it, each call once; a call id repeated
/// within one message, or a reply without an id, is skipped.
fn member_runs(messages: &[Message], member_tools: &[String]) -> Vec<RunId> {
    if member_tools.is_empty() {
        return Vec::new();
    }
    let mut ids = Vec::new();
    let mut pending: Vec<&str> = Vec::new();
    let mut member_calls = 0usize;
    for message in messages {
        match message.role {
            Role::Assistant => {
                pending = turn_member_calls(message, member_tools);
                member_calls += pending.len();
            }
            Role::Tool => {
                for part in &message.parts {
                    let ContentPart::ToolResult {
                        call_id, content, ..
                    } = part
                    else {
                        continue;
                    };
                    let Some(at) = pending.iter().position(|c| c == call_id) else {
                        continue;
                    };
                    pending.swap_remove(at);
                    if let Some(id) = reply_run_id(content) {
                        ids.push(id);
                    }
                }
            }
            _ => pending.clear(),
        }
    }
    assert!(ids.len() <= member_calls, "each member call is used once");
    ids
}

/// Returns the ids of the member calls in one assistant message whose id
/// no other call in that message shares. Returns an empty list for a
/// message without member calls.
fn turn_member_calls<'a>(message: &'a Message, member_tools: &[String]) -> Vec<&'a str> {
    let calls: Vec<(&str, &str)> = message
        .parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolCall { id, name, .. } => Some((id.as_str(), name.as_str())),
            _ => None,
        })
        .collect();
    let unique = |id: &str| calls.iter().filter(|(c, _)| *c == id).count() == 1;
    let members: Vec<&str> = calls
        .iter()
        .filter(|(id, name)| member_tools.iter().any(|t| t == name) && unique(id))
        .map(|(id, _)| *id)
        .collect();
    assert!(members.len() <= calls.len());
    members
}

fn reply_run_id(content: &str) -> Option<RunId> {
    let reply: Value = serde_json::from_str(content).ok()?;
    serde_json::from_value(reply.get("member_run_id")?.clone()).ok()
}

/// Returns the delegate settings a member run of this agent uses: its own
/// limits as the base, and its store, so a member run resumes from its
/// checkpoints after a crash. The depth cap stays at its default.
pub(crate) fn member_settings(
    limits: &Limits,
    store: Option<&Arc<dyn CheckpointStore>>,
) -> DelegateSettings {
    let settings = DelegateSettings::default().with_limits(limits.clone());
    match store {
        Some(store) => settings.with_store(store.clone()),
        None => settings,
    }
}

/// Returns the tool description of an agent: its role and its goal.
/// Never empty, since `Persona::parse` requires both.
pub(crate) fn describe(persona: &Persona) -> String {
    let text = format!("{} Goal: {}", persona.role.trim(), persona.goal.trim());
    assert!(!text.trim().is_empty());
    text
}

fn retry_delay(attempt: u32, retry_after_ms: Option<u64>) -> Duration {
    let backoff = BASE_RETRY_MS.saturating_mul(1u64 << attempt.min(16));
    Duration::from_millis(retry_after_ms.unwrap_or(backoff).min(MAX_RETRY_MS))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(id: &str, name: &str) -> ContentPart {
        ContentPart::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments: json!({}),
        }
    }

    fn result(id: &str, run: RunId) -> ContentPart {
        ContentPart::ToolResult {
            call_id: id.into(),
            content: json!({"member_run_id": run.to_string(), "answer": "x"}).to_string(),
            is_error: false,
        }
    }

    fn msg(role: Role, parts: Vec<ContentPart>) -> Message {
        Message::new(role, parts).unwrap()
    }

    #[test]
    fn member_runs_pairs_results_with_calls_of_the_same_turn() {
        let (member, other, again) = (RunId::new(), RunId::new(), RunId::new());
        let tools = vec!["researcher".to_string()];
        let messages = vec![
            Message::text(Role::User, "Go"),
            msg(Role::Assistant, vec![call("call_0", "researcher")]),
            msg(Role::Tool, vec![result("call_0", member)]),
            msg(Role::Assistant, vec![call("call_0", "other")]),
            msg(Role::Tool, vec![result("call_0", other)]),
            msg(Role::Assistant, vec![call("call_0", "researcher")]),
            msg(
                Role::Tool,
                vec![result("call_0", again), result("call_0", other)],
            ),
            msg(
                Role::Assistant,
                vec![call("c1", "researcher"), call("c1", "other")],
            ),
            msg(Role::Tool, vec![result("c1", other), result("c1", other)]),
            Message::text(Role::Assistant, "Done."),
        ];
        assert_eq!(member_runs(&messages, &tools), [member, again]);
        assert!(member_runs(&messages, &[]).is_empty());
    }

    #[test]
    fn retry_delay_backs_off_and_caps() {
        assert_eq!(retry_delay(0, None), Duration::from_millis(500));
        assert_eq!(retry_delay(2, None), Duration::from_millis(2000));
        assert_eq!(retry_delay(20, None), Duration::from_millis(30_000));
        assert_eq!(retry_delay(0, Some(7)), Duration::from_millis(7));
        assert_eq!(retry_delay(0, Some(90_000)), Duration::from_millis(30_000));
    }
}
