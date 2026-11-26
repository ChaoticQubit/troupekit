use super::{outcome, retry_delay, BuiltAgent, Input};
use crate::error::KitError;
use std::sync::Arc;
use std::time::{Duration, Instant};
use troupekit_agent::{AgentIo, AgentRun, RunState, StepOutcome};
use troupekit_core::ids::RunId;
use troupekit_core::store::CheckpointStore;

/// Most steps one `resume_for` call takes before it returns `Pending`,
/// whatever the budget. Bounds a window even when steps are instant.
/// Fixed at 10 000.
pub const MAX_STEPS_PER_WINDOW: u32 = 10_000;

/// Where a run stands after one `resume_for` window.
/// `Done` holds the answer as text; `Pending` means the budget or the
/// step bound ran out first, and the run resumes from its last save.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Progress {
    Done(String),
    Pending { run_id: RunId },
}

impl BuiltAgent {
    /// Starts a run on `input`: checks it, saves the first checkpoint and
    /// returns the run id, with no model call. Continue it with `resume`
    /// or `resume_for`. Needs `Storage::sqlite(..)` or `Storage::custom(..)`.
    pub async fn start(&self, input: impl Into<Input>) -> Result<RunId, KitError> {
        let store = self.durable("start")?;
        let messages = self.messages(input.into()).await?;
        let inner = &self.inner;
        let run = AgentRun::new(&inner.persona, messages, inner.limits.clone(), &inner.io).await?;
        store.save(run.checkpoint()).await?;
        Ok(run.run_id())
    }

    /// Continues run `run_id` for about `budget`, saving after every step.
    /// A step in flight finishes, so leave one model call of slack in a
    /// background window. Fails with `Failed`, `Paused` or `Retries` as
    /// `resume` does; a retry wait that would pass the budget is `Pending`.
    pub async fn resume_for(&self, run_id: RunId, budget: Duration) -> Result<Progress, KitError> {
        let deadline = Instant::now().checked_add(budget);
        let spent = |extra: Duration| match deadline {
            Some(deadline) => Instant::now() + extra >= deadline,
            None => false,
        };
        let store = self.durable("resume_for")?;
        let Some(checkpoint) = store.latest(run_id).await? else {
            return Err(KitError::config(
                &self.inner.persona.name,
                format!("no checkpoint for run {run_id}"),
            ));
        };
        let mut run = AgentRun::from_checkpoint(&checkpoint)?;
        match run.state().clone() {
            RunState::Done { output } => return Ok(Progress::Done(outcome(&run, output).text)),
            RunState::Failed { error } => {
                return Err(KitError::Failed {
                    run_id,
                    reason: error,
                })
            }
            RunState::Paused { reason, .. } => return Err(KitError::Paused { run_id, reason }),
            _ => {}
        }
        let io: &dyn AgentIo = &self.inner.io;
        let mut steps = 0u32;
        while steps < MAX_STEPS_PER_WINDOW && !spent(Duration::ZERO) {
            let step = run.step(io).await;
            store.save(run.checkpoint()).await?;
            steps += 1;
            match step {
                StepOutcome::Continue => {}
                StepOutcome::Done(output) => return Ok(Progress::Done(outcome(&run, output).text)),
                StepOutcome::Retry {
                    error,
                    retry_after_ms,
                } => {
                    assert!(run.retries() > 0);
                    let attempts = run.retries() - 1;
                    if attempts >= self.inner.retries {
                        return Err(KitError::Retries {
                            run_id,
                            attempts,
                            last_error: error,
                        });
                    }
                    let wait = retry_delay(attempts, retry_after_ms);
                    if spent(wait) {
                        return Ok(Progress::Pending { run_id });
                    }
                    tokio::time::sleep(wait).await;
                }
                StepOutcome::Paused(reason) => return Err(KitError::Paused { run_id, reason }),
                StepOutcome::Failed(reason) => return Err(KitError::Failed { run_id, reason }),
            }
            assert!(run.retries() <= self.inner.retries);
        }
        assert!(steps <= MAX_STEPS_PER_WINDOW);
        Ok(Progress::Pending { run_id })
    }

    fn durable(&self, what: &str) -> Result<Arc<dyn CheckpointStore>, KitError> {
        match &self.inner.store {
            Some(store) => Ok(store.clone()),
            None => Err(KitError::config(
                &self.inner.persona.name,
                format!("{what} needs Storage::sqlite(..) or Storage::custom(..) checkpoints"),
            )),
        }
    }
}
