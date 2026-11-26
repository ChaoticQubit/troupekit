use crate::error::KitError;
use crate::laws::Laws;
use serde_json::Value;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use troupekit_agent::{
    drive, AgentIo, AgentRun, EventSink, Limits, Persona, RunState, SimpleIo, StepOutcome,
};
use troupekit_core::event::{Event, Usage};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::CheckpointStore;

mod start;

pub use start::{Progress, MAX_STEPS_PER_WINDOW};

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
    io: SimpleIo,
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
        BuiltAgent {
            inner: Arc::new(Inner {
                persona,
                io,
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
        let mut run =
            AgentRun::new(&inner.persona, messages, inner.limits.clone(), &inner.io).await?;
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

    async fn finish(
        &self,
        run: &mut AgentRun,
        store: &dyn CheckpointStore,
    ) -> Result<RunOutcome, KitError> {
        let io: &dyn AgentIo = &self.inner.io;
        match run.state().clone() {
            RunState::Done { output } => return Ok(outcome(run, output)),
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
        loop {
            match drive(run, io, store).await? {
                StepOutcome::Done(output) => return Ok(outcome(run, output)),
                StepOutcome::Retry {
                    error,
                    retry_after_ms,
                } => {
                    assert!(run.retries() > 0);
                    let attempts = run.retries() - 1;
                    if attempts >= self.inner.retries {
                        return Err(KitError::Retries {
                            run_id: run.run_id(),
                            attempts,
                            last_error: error,
                        });
                    }
                    tokio::time::sleep(retry_delay(attempts, retry_after_ms)).await;
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
            assert!(run.retries() <= self.inner.retries);
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

fn outcome(run: &AgentRun, output: Value) -> RunOutcome {
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
    }
}

fn retry_delay(attempt: u32, retry_after_ms: Option<u64>) -> Duration {
    let backoff = BASE_RETRY_MS.saturating_mul(1u64 << attempt.min(16));
    Duration::from_millis(retry_after_ms.unwrap_or(backoff).min(MAX_RETRY_MS))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_delay_backs_off_and_caps() {
        assert_eq!(retry_delay(0, None), Duration::from_millis(500));
        assert_eq!(retry_delay(2, None), Duration::from_millis(2000));
        assert_eq!(retry_delay(20, None), Duration::from_millis(30_000));
        assert_eq!(retry_delay(0, Some(7)), Duration::from_millis(7));
        assert_eq!(retry_delay(0, Some(90_000)), Duration::from_millis(30_000));
    }
}
