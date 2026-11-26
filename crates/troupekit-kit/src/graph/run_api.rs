use super::{input_state, BuiltGraph, Graph, GraphReport};
use crate::built::{Progress, Queue, RunStream, StreamIo, StreamSettings, MAX_STEPS_PER_WINDOW};
use crate::error::KitError;
use async_trait::async_trait;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use troupekit_agent::{AgentError, AgentIo, EventSink, Persona};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::CheckpointStore;
use troupekit_graph::{Conditions, GraphIo};
use troupekit_graph::{GraphOutcome, GraphRun};

impl Graph {
    /// Sends every event of the graph and of its agent nodes to `sink`.
    /// Agent nodes run under their own run ids, so order by `seq` only
    /// within one run id. Defaults to a sink that drops events.
    pub fn events(mut self, sink: Arc<dyn EventSink>) -> Self {
        self.events = Some(sink);
        self
    }

    /// Builds the graph and starts a streamed run on `input`.
    /// Output: the stream once the run id is known, or the first build
    /// or setup error. Keep a `BuiltGraph` to run many times.
    pub async fn stream(
        self,
        input: impl Into<String>,
    ) -> Result<RunStream<GraphReport>, KitError> {
        self.build().await?.stream(input).await
    }
}

impl BuiltGraph {
    /// Starts a graph run on `input`: saves the first checkpoint and
    /// returns the run id, with no node run yet. Continue it with
    /// `resume` or `resume_for`. Needs durable storage.
    pub async fn start(&self, input: impl Into<String>) -> Result<RunId, KitError> {
        let inner = &self.inner;
        let store = self.durable("start")?;
        let state = input_state(&input.into());
        let run = GraphRun::new(
            inner.workflow.clone(),
            state,
            inner.limits.clone(),
            &inner.io,
        )
        .await?;
        let checkpoint = run.checkpoint();
        assert_eq!(checkpoint.step, 0);
        store.save(checkpoint).await?;
        Ok(run.run_id())
    }

    /// Continues graph run `run_id` for about `budget` on the io clock,
    /// saving after every step. A node in flight finishes, so leave one
    /// model call of slack. Returns the last node's output as text when
    /// done; a retry wait that would pass the budget is `Pending`.
    pub async fn resume_for(&self, run_id: RunId, budget: Duration) -> Result<Progress, KitError> {
        let inner = &self.inner;
        let clock = inner.io.agent.clock();
        let budget_ms = i64::try_from(budget.as_millis()).unwrap_or(i64::MAX);
        let deadline_ms = clock.now_ms().saturating_add(budget_ms);
        let mut run = self.load(run_id).await?;
        let store = self.durable("resume_for")?;
        let mut steps = 0u32;
        while steps < MAX_STEPS_PER_WINDOW && clock.now_ms() < deadline_ms {
            let before = run.steps();
            let outcome = run.step(&inner.io).await;
            if run.steps() != before {
                assert_eq!(run.steps(), before + 1);
                store.save(run.checkpoint()).await?;
            }
            steps += 1;
            match outcome {
                GraphOutcome::Continue => {}
                GraphOutcome::RetryAt { at_ms, .. } if at_ms >= deadline_ms => {
                    return Ok(Progress::Pending { run_id });
                }
                GraphOutcome::RetryAt { at_ms, .. } => {
                    super::wait_until(clock, at_ms, super::RETRY_WAIT_CHUNK_MS).await;
                }
                GraphOutcome::Waiting(reason) => return Err(KitError::Paused { run_id, reason }),
                GraphOutcome::Failed(reason) => return Err(KitError::Failed { run_id, reason }),
                GraphOutcome::Done(state) => {
                    let output = state
                        .get(&inner.last)
                        .expect("a finished chain run wrote its last node");
                    return Ok(Progress::Done(text_of(output)));
                }
            }
        }
        assert!(steps <= MAX_STEPS_PER_WINDOW);
        Ok(Progress::Pending { run_id })
    }

    /// Starts a graph run on `input` and returns the stream of its events
    /// and its agent nodes' events, with the default buffer. Fails at the
    /// call on a bad setup. Needs no durable storage.
    pub async fn stream(
        &self,
        input: impl Into<String>,
    ) -> Result<RunStream<GraphReport>, KitError> {
        let inner = &self.inner;
        let settings = StreamSettings::default();
        let queue = Arc::new(Queue::new(settings.capacity));
        let io = StreamGraphIo {
            graph: self.clone(),
            agent: StreamIo {
                base: inner.io.agent.clone(),
                queue: queue.clone(),
            },
        };
        let state = input_state(&input.into());
        let mut run =
            GraphRun::new(inner.workflow.clone(), state, inner.limits.clone(), &io).await?;
        let run_id = run.run_id();
        let store: Arc<dyn CheckpointStore> = match &inner.store {
            Some(store) => store.clone(),
            None => Arc::new(InMemCheckpoints::default()),
        };
        store.save(run.checkpoint()).await?;
        let graph = self.clone();
        let task = async move { graph.finish(&io, &mut run, store.as_ref()).await };
        assert_eq!(queue.dropped(), 0);
        Ok(RunStream::spawn(run_id, queue, task))
    }

    async fn load(&self, run_id: RunId) -> Result<GraphRun, KitError> {
        let inner = &self.inner;
        let store = self.durable("resume_for")?;
        let Some(checkpoint) = store.latest(run_id).await? else {
            return Err(KitError::config(
                &inner.owner,
                format!("no checkpoint for run {run_id}"),
            ));
        };
        let run = GraphRun::from_checkpoint(&checkpoint)?;
        if run.workflow() != &inner.workflow {
            return Err(KitError::config(
                &inner.owner,
                format!(
                    "run {run_id} was started by a different workflow; resume it with the graph that started it"
                ),
            ));
        }
        assert_eq!(run.run_id(), run_id);
        Ok(run)
    }

    fn durable(&self, what: &str) -> Result<Arc<dyn CheckpointStore>, KitError> {
        match &self.inner.store {
            Some(store) => Ok(store.clone()),
            None => Err(KitError::config(
                &self.inner.owner,
                format!("{what} needs Storage::sqlite(..) or Storage::custom(..) checkpoints"),
            )),
        }
    }
}

fn text_of(output: &Value) -> String {
    match output {
        Value::String(text) => text.clone(),
        other => other.to_string(),
    }
}

struct StreamGraphIo {
    graph: BuiltGraph,
    agent: StreamIo<Arc<dyn AgentIo>>,
}

#[async_trait]
impl GraphIo for StreamGraphIo {
    fn agent(&self) -> &dyn AgentIo {
        &self.agent
    }

    async fn persona(&self, name: &str) -> Result<Persona, AgentError> {
        self.graph.inner.io.persona(name).await
    }

    fn conditions(&self) -> &Conditions {
        self.graph.inner.io.conditions()
    }
}
