use super::{BuiltAgent, Input, RunOutcome};
use crate::error::KitError;
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, PoisonError};
use tokio::sync::Notify;
use tokio::task::JoinHandle;
use troupekit_agent::{
    AgentError, AgentIo, AgentRun, EvalRegistry, EventSink, ModelHandle, UsageLedger,
};
use troupekit_core::clock::Clock;
use troupekit_core::event::{Event, EventBody};
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::store::CheckpointStore;
use troupekit_tools::ToolRegistry;

/// Buffer settings of one streamed run. `capacity` (default 4096, from
/// `MIN_CAPACITY` to `MAX_CAPACITY`) bounds queued events: when full a new
/// text delta is dropped and counted; lifecycle events are always kept.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamSettings {
    pub capacity: usize,
}

impl StreamSettings {
    /// Smallest `capacity` `stream_with` accepts.
    /// Below it a burst of lifecycle events would crowd out every delta.
    /// Fixed at 16.
    pub const MIN_CAPACITY: usize = 16;

    /// Largest `capacity` `stream_with` accepts.
    /// Bounds the memory one unread stream can hold.
    /// Fixed at 1 048 576 events.
    pub const MAX_CAPACITY: usize = 1_048_576;

    fn check(&self, agent: &str) -> Result<(), KitError> {
        let range = StreamSettings::MIN_CAPACITY..=StreamSettings::MAX_CAPACITY;
        if range.contains(&self.capacity) {
            return Ok(());
        }
        Err(KitError::config(
            agent,
            format!(
                "stream capacity must be {} to {}, got {}",
                range.start(),
                range.end(),
                self.capacity
            ),
        ))
    }
}

impl Default for StreamSettings {
    /// Returns the default buffer of 4096 events.
    /// Enough for a long answer read at UI speed.
    /// Change `capacity` before passing it to `stream_with`.
    fn default() -> Self {
        StreamSettings { capacity: 4096 }
    }
}

/// A run in progress that yields its events in `seq` order. `next`
/// returns `None` once the run has ended, been cancelled or failed;
/// `finish` then gives the result. Dropping it cancels the run.
pub struct RunStream<T = RunOutcome> {
    run_id: RunId,
    queue: Arc<Queue>,
    task: Option<JoinHandle<Result<T, KitError>>>,
}

impl<T> RunStream<T> {
    pub(crate) fn spawn<F>(run_id: RunId, queue: Arc<Queue>, task: F) -> RunStream<T>
    where
        F: std::future::Future<Output = Result<T, KitError>> + Send + 'static,
        T: Send + 'static,
    {
        let closer = Closer(queue.clone());
        let task = tokio::spawn(async move {
            let _closer = closer;
            task.await
        });
        RunStream {
            run_id,
            queue,
            task: Some(task),
        }
    }

    /// Waits for the next event of the run.
    /// Output: the event, or `None` once the run is over and every
    /// queued event has been read.
    pub async fn next(&mut self) -> Option<Event> {
        self.queue.pop().await
    }

    /// Returns the run id, known before the first event.
    /// Store it to resume the run after a cancel or a restart.
    /// Output: the id.
    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    /// Returns how many text deltas were dropped because the buffer was
    /// full. The answer in `finish` is complete either way.
    /// Output: the count.
    pub fn dropped_deltas(&self) -> u64 {
        self.queue.dropped()
    }

    /// Stops the run at its next await point. The last saved checkpoint
    /// stays, so `resume` can continue it; `finish` returns `Cancelled`.
    /// Calling it twice, or after the end, does nothing.
    pub fn cancel(&self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }

    /// Waits for the run to end and returns its result.
    /// Output: the outcome, the run's error, or `Cancelled` after
    /// `cancel`. A panic in the run is resumed in the caller.
    pub async fn finish(mut self) -> Result<T, KitError> {
        let task = self
            .task
            .take()
            .expect("a stream has its task until finish");
        match task.await {
            Ok(result) => result,
            Err(err) if err.is_panic() => std::panic::resume_unwind(err.into_panic()),
            Err(_) => Err(KitError::Cancelled {
                run_id: self.run_id,
            }),
        }
    }
}

impl<T> Drop for RunStream<T> {
    /// Cancels the run when the stream is dropped before `finish`, so an
    /// unread stream never keeps calling the model. The checkpoint stays.
    /// Does nothing after `finish`.
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

impl<T> std::fmt::Debug for RunStream<T> {
    /// Writes the run id and the dropped delta count.
    /// The queued events are not shown.
    /// Output: the debug text.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunStream")
            .field("run_id", &self.run_id)
            .field("dropped_deltas", &self.dropped_deltas())
            .finish_non_exhaustive()
    }
}

impl BuiltAgent {
    /// Starts a run on `input` and returns its event stream once the run
    /// id is known, with the default buffer. Fails at the call on a bad
    /// input or setup, as `start` does. Needs no durable storage.
    pub async fn stream(&self, input: impl Into<Input>) -> Result<RunStream, KitError> {
        self.stream_with(input, StreamSettings::default()).await
    }

    /// Starts a streamed run on `input` with `settings`. Events also reach
    /// the agent's own `.events(..)` sink. Fails with `Config` when the
    /// capacity is out of range, else as `stream`.
    pub async fn stream_with(
        &self,
        input: impl Into<Input>,
        settings: StreamSettings,
    ) -> Result<RunStream, KitError> {
        let inner = &self.inner;
        settings.check(&inner.persona.name)?;
        let messages = self.messages(input.into()).await?;
        let queue = Arc::new(Queue::new(settings.capacity));
        let io = StreamIo {
            agent: self.clone(),
            queue: queue.clone(),
        };
        let mut run = AgentRun::new(&inner.persona, messages, inner.limits.clone(), &io).await?;
        let run_id = run.run_id();
        let store: Arc<dyn CheckpointStore> = match &inner.store {
            Some(store) => store.clone(),
            None => Arc::new(InMemCheckpoints::default()),
        };
        store.save(run.checkpoint()).await?;
        let agent = self.clone();
        let task = async move { agent.finish(&io, &mut run, store.as_ref()).await };
        assert_eq!(queue.dropped(), 0);
        Ok(RunStream::spawn(run_id, queue, task))
    }
}

pub(crate) struct Queue {
    state: Mutex<QueueState>,
    ready: Notify,
    capacity: usize,
}

struct QueueState {
    events: VecDeque<Event>,
    dropped: u64,
    closed: bool,
}

impl Queue {
    pub(crate) fn new(capacity: usize) -> Queue {
        assert!(capacity >= StreamSettings::MIN_CAPACITY);
        assert!(capacity <= StreamSettings::MAX_CAPACITY);
        Queue {
            state: Mutex::new(QueueState {
                events: VecDeque::new(),
                dropped: 0,
                closed: false,
            }),
            ready: Notify::new(),
            capacity,
        }
    }

    fn push(&self, event: Event) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        if state.closed {
            return;
        }
        let delta = matches!(event.body, EventBody::TextDelta { .. });
        if delta && state.events.len() >= self.capacity {
            state.dropped += 1;
            return;
        }
        state.events.push_back(event);
        drop(state);
        self.ready.notify_one();
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        state.closed = true;
        drop(state);
        self.ready.notify_one();
    }

    async fn pop(&self) -> Option<Event> {
        loop {
            {
                let mut state = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(event) = state.events.pop_front() {
                    return Some(event);
                }
                if state.closed {
                    return None;
                }
            }
            self.ready.notified().await;
        }
    }

    fn dropped(&self) -> u64 {
        self.state
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .dropped
    }
}

struct Closer(Arc<Queue>);

impl Drop for Closer {
    /// Closes the queue when the run task ends for any reason: an answer,
    /// an error with no final event such as `Retries`, a panic or a
    /// cancel. `next` then returns `None` after the queued events.
    fn drop(&mut self) {
        self.0.close();
    }
}

impl EventSink for Queue {
    fn emit(&self, event: Event) {
        self.push(event);
    }
}

struct StreamIo {
    agent: BuiltAgent,
    queue: Arc<Queue>,
}

impl EventSink for StreamIo {
    fn emit(&self, event: Event) {
        self.queue.push(event.clone());
        self.agent.inner.io.events.emit(event);
    }
}

#[async_trait]
impl AgentIo for StreamIo {
    async fn model(&self, selector: &str) -> Result<ModelHandle, AgentError> {
        self.agent.inner.io.model(selector).await
    }

    fn tools(&self) -> &ToolRegistry {
        self.agent.inner.io.tools()
    }

    fn evals(&self) -> &EvalRegistry {
        self.agent.inner.io.evals()
    }

    fn ledger(&self) -> &dyn UsageLedger {
        self.agent.inner.io.ledger()
    }

    fn events(&self) -> &dyn EventSink {
        self
    }

    fn clock(&self) -> &dyn Clock {
        self.agent.inner.io.clock()
    }
}
