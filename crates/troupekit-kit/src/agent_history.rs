use crate::built::{BuiltAgent, Input, RunOutcome};
use crate::error::KitError;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use troupekit_agent::AgentRun;
use troupekit_core::ids::RunId;
use troupekit_core::store::{Checkpoint, CheckpointStore, ForkLink};

const CONCURRENCY_MAX: usize = 64;
const INPUTS_MAX: usize = 10_000;

impl BuiltAgent {
    /// Runs the agent once on each of `inputs`, at most `concurrency` runs
    /// at a time, and returns one result per input, in input order. Fails
    /// as a whole only for `concurrency` outside 1 to 64, more than 10 000
    /// inputs, or a run task that panicked.
    pub async fn batch<I: Into<Input>>(
        &self,
        inputs: impl IntoIterator<Item = I>,
        concurrency: usize,
    ) -> Result<Vec<Result<RunOutcome, KitError>>, KitError> {
        let name = self.persona().name.clone();
        if !(1..=CONCURRENCY_MAX).contains(&concurrency) {
            let m = format!("batch concurrency must be 1 to {CONCURRENCY_MAX}, got {concurrency}");
            return Err(KitError::config(&name, m));
        }
        let inputs: Vec<Input> = inputs
            .into_iter()
            .take(INPUTS_MAX + 1)
            .map(Into::into)
            .collect();
        if inputs.len() > INPUTS_MAX {
            let m = format!("batch takes at most {INPUTS_MAX} inputs");
            return Err(KitError::config(&name, m));
        }
        let count = inputs.len();
        let gate = Arc::new(Semaphore::new(concurrency));
        let mut runs = JoinSet::new();
        for (index, input) in inputs.into_iter().enumerate() {
            let (agent, gate) = (self.clone(), gate.clone());
            runs.spawn(async move {
                let _permit = gate
                    .acquire_owned()
                    .await
                    .expect("the batch gate is never closed");
                (index, agent.run_with(input).await)
            });
        }
        let mut slots: Vec<Option<Result<RunOutcome, KitError>>> =
            (0..count).map(|_| None).collect();
        while let Some(joined) = runs.join_next().await {
            let (index, result) =
                joined.map_err(|e| KitError::config(&name, format!("a batch run stopped: {e}")))?;
            assert!(slots[index].is_none(), "each input runs once");
            slots[index] = Some(result);
        }
        let results: Vec<_> = slots
            .into_iter()
            .map(|s| s.expect("every input ran"))
            .collect();
        assert_eq!(results.len(), count);
        Ok(results)
    }

    /// Returns every kept checkpoint of run `run_id`, oldest step first.
    /// A store without history returns only the latest step. Needs
    /// `Storage::sqlite(..)` or `Storage::custom(..)` checkpoints.
    pub async fn history(&self, run_id: RunId) -> Result<Vec<Checkpoint>, KitError> {
        let store = self.durable("history")?;
        let history = store.history(run_id).await?;
        assert!(history.iter().all(|cp| cp.run_id == run_id));
        Ok(history)
    }

    /// Starts a new run from step `step` of run `run_id`, saves its first
    /// checkpoint and links it to the parent. Returns the new run id; pass
    /// it to `resume` to continue. Fails when that step is not kept.
    pub async fn fork(&self, run_id: RunId, step: u64) -> Result<RunId, KitError> {
        let store = self.durable("fork")?;
        let history = store.history(run_id).await?;
        let Some(cp) = history.into_iter().find(|cp| cp.step == step) else {
            let m = format!("no checkpoint at step {step} of run {run_id}");
            return Err(KitError::config(&self.persona().name, m));
        };
        let child = AgentRun::from_checkpoint(&cp)?.fork();
        assert_eq!(child.parent(), Some((run_id, step)));
        assert_ne!(child.run_id(), run_id);
        store.save(child.checkpoint()).await?;
        store
            .link_fork(ForkLink::new(child.run_id(), run_id, step))
            .await?;
        Ok(child.run_id())
    }

    /// Continues run `run_id` as a new run that keeps its first
    /// `message_index` messages, then `input` (empty or one user message),
    /// and runs to the end. The kept prefix must end on a user or tool
    /// message. Returns the outcome of the new, linked run.
    pub async fn continue_from(
        &self,
        run_id: RunId,
        message_index: usize,
        input: impl Into<Input>,
    ) -> Result<RunOutcome, KitError> {
        let store = self.durable("continue_from")?;
        let name = &self.persona().name;
        let mut messages = input.into().into_messages();
        if messages.len() > 1 {
            let m = "continue_from takes at most one message as input";
            return Err(KitError::config(name, m));
        }
        let Some(cp) = store.latest(run_id).await? else {
            return Err(KitError::config(
                name,
                format!("no checkpoint for run {run_id}"),
            ));
        };
        let mut child = AgentRun::fork_at_message(&cp, message_index, messages.pop())?;
        assert_eq!(child.parent(), Some((run_id, cp.step)));
        store.save(child.checkpoint()).await?;
        store
            .link_fork(ForkLink::new(child.run_id(), run_id, cp.step))
            .await?;
        self.finish(&mut child, store.as_ref()).await
    }

    fn durable(&self, what: &str) -> Result<Arc<dyn CheckpointStore>, KitError> {
        if self.is_team() {
            let m = format!("{what} is not built yet for a team agent");
            return Err(KitError::config(&self.persona().name, m));
        }
        let Some(store) = self.store() else {
            let m = format!("{what} needs Storage::sqlite(..) or Storage::custom(..) checkpoints");
            return Err(KitError::config(&self.persona().name, m));
        };
        Ok(store.clone())
    }
}
