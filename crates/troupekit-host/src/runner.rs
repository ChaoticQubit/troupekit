use crate::error::HostError;
use crate::queue::{JobQueue, Lease};
use futures_util::future::join_all;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::watch;
use troupekit_agent::Limits;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::store::CheckpointStore;
use troupekit_graph::{GraphIo, GraphOutcome, GraphRun, Stop};
use troupekit_trace::{trace_span, Trace};

const DEFAULT_LEASE_MS: i64 = 300_000;
const STORE_RETRY_MS: i64 = 1_000;

/// What `process` did with one lease. `Stepped`: the run moved and was
/// saved. `Unchanged`: it was waiting or final, nothing saved. `Lost`:
/// another worker advanced the run first; the job was left alone.
#[derive(Debug, Clone, PartialEq)]
pub enum Processed {
    Stepped(GraphOutcome),
    Unchanged(GraphOutcome),
    Lost,
}

/// Limits for one `tick`: at most `max_jobs` processed, and no new claim
/// once `budget_ms` has passed. The defaults (16 jobs, 250 s) fit a
/// serverless request capped at 300 s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TickOptions {
    pub max_jobs: usize,
    pub budget_ms: i64,
}

impl Default for TickOptions {
    fn default() -> Self {
        Self {
            max_jobs: 16,
            budget_ms: 250_000,
        }
    }
}

/// What one `tick` did: jobs processed, how many of them were lost to
/// another worker, and the error text of each that failed.
/// A quiet tick processes nothing.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TickReport {
    pub processed: usize,
    pub lost: usize,
    pub errors: Vec<String>,
}

/// Worker mode settings: the options of every tick and how long to sleep
/// when a tick found nothing to do or its claim failed.
/// The default sleeps one second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkOptions {
    pub tick: TickOptions,
    pub idle_ms: u64,
}

impl Default for WorkOptions {
    fn default() -> Self {
        Self {
            tick: TickOptions::default(),
            idle_ms: 1_000,
        }
    }
}

/// Runs graph runs from a job queue: claims due jobs, loads each run's
/// latest checkpoint, steps it once, saves it and reschedules, completes
/// or fails the job. Time comes from the io's clock.
pub struct Runner {
    queue: Arc<dyn JobQueue>,
    checkpoints: Arc<dyn CheckpointStore>,
    io: Arc<dyn GraphIo>,
    lease_ms: i64,
}

impl Runner {
    /// Builds a runner over `queue`, `checkpoints` and `io` with a 300 s
    /// lease. Several runners may share the same three.
    /// Returns the runner.
    pub fn new(
        queue: Arc<dyn JobQueue>,
        checkpoints: Arc<dyn CheckpointStore>,
        io: Arc<dyn GraphIo>,
    ) -> Self {
        Self {
            queue,
            checkpoints,
            io,
            lease_ms: DEFAULT_LEASE_MS,
        }
    }

    /// Sets the lease length to `lease_ms`, which must be positive and
    /// longer than one step takes.
    /// Returns the runner.
    pub fn with_lease_ms(mut self, lease_ms: i64) -> Self {
        assert!(lease_ms > 0, "a lease lasts some time");
        self.lease_ms = lease_ms;
        self
    }

    fn now(&self) -> i64 {
        self.io.agent().clock().now_ms()
    }

    /// Saves `run` as step 0 when it has no checkpoint yet and enqueues
    /// its job now. Returns the store or queue error.
    /// Call it once after `GraphRun::new`.
    pub async fn enqueue(&self, run: &GraphRun) -> Result<(), HostError> {
        if self.checkpoints.latest(run.run_id()).await?.is_none() {
            self.checkpoints.save(run.checkpoint()).await?;
        }
        self.queue.enqueue(run.run_id(), self.now()).await?;
        Ok(())
    }

    /// Processes one claimed `lease`: load, step (timed as active work, or
    /// stopped instead once over its time limit), save when it moved, then
    /// queue the outcome. A save conflict returns `Lost`; store errors
    /// reschedule the job after one second and return the error.
    pub async fn process(&self, lease: &Lease) -> Result<Processed, HostError> {
        let mut span = trace_span!(Trace::Graphs, "graph.job", run = lease.run_id.to_string());
        let result = span.instrument(self.process_leased(lease)).await;
        match &result {
            Ok(Processed::Stepped(_)) => span.record("outcome", "stepped"),
            Ok(Processed::Unchanged(_)) => span.record("outcome", "unchanged"),
            Ok(Processed::Lost) => span.record("outcome", "lost"),
            Err(e) => span.fail(e.to_string()),
        }
        result
    }

    async fn process_leased(&self, lease: &Lease) -> Result<Processed, HostError> {
        let cp = match self.checkpoints.latest(lease.run_id).await {
            Ok(Some(cp)) => cp,
            Ok(None) => {
                self.queue.fail(lease, "no checkpoint").await?;
                return Err(HostError::NoCheckpoint(lease.run_id));
            }
            Err(e) => return Err(self.store_failed(lease, e).await),
        };
        let mut run = match GraphRun::from_checkpoint(&cp) {
            Ok(run) => run,
            Err(e) => {
                self.queue.fail(lease, &e.to_string()).await?;
                return Err(e.into());
            }
        };
        let before = run.steps();
        let outcome = if run.over_time() {
            let limit_ms = run.time_limit_ms().expect("only a limited run is over time");
            let _finished = run.stop(Stop::TimeLimit { limit_ms });
            run.step(self.io.as_ref()).await
        } else {
            let started = self.now();
            let outcome = run.step(self.io.as_ref()).await;
            run.add_active_ms(u64::try_from(self.now() - started).unwrap_or(0));
            outcome
        };
        let moved = run.steps() != before;
        if moved {
            match self.checkpoints.save(run.checkpoint()).await {
                Ok(()) => {}
                Err(CoreError::Conflict { .. }) => return Ok(Processed::Lost),
                Err(e) => return Err(self.store_failed(lease, e).await),
            }
        }
        let queued = match &outcome {
            GraphOutcome::Continue => self.queue.reschedule(lease, self.now()).await,
            GraphOutcome::RetryAt { at_ms, .. } => self.queue.reschedule(lease, *at_ms).await,
            GraphOutcome::Waiting(_) | GraphOutcome::Done(_) => self.queue.complete(lease).await,
            GraphOutcome::Failed(error) => self.queue.fail(lease, error).await,
        };
        match queued {
            Ok(()) => {}
            Err(CoreError::Conflict { .. }) => return Ok(Processed::Lost),
            Err(e) => return Err(e.into()),
        }
        Ok(if moved {
            Processed::Stepped(outcome)
        } else {
            Processed::Unchanged(outcome)
        })
    }

    async fn store_failed(&self, lease: &Lease, e: CoreError) -> HostError {
        let _ = self
            .queue
            .reschedule(lease, self.now() + STORE_RETRY_MS)
            .await;
        e.into()
    }

    /// Step mode: claims due jobs and processes each batch concurrently,
    /// until nothing is due, `max_jobs` were processed or `budget_ms` has
    /// passed (checked before each claim). Returns the report or the claim error.
    pub async fn tick(&self, opts: TickOptions) -> Result<TickReport, HostError> {
        let start = self.now();
        let mut report = TickReport::default();
        while report.processed < opts.max_jobs && self.now() - start < opts.budget_ms {
            let leases = self
                .queue
                .claim_due(opts.max_jobs - report.processed, self.lease_ms)
                .await?;
            if leases.is_empty() {
                break;
            }
            assert!(leases.len() <= opts.max_jobs - report.processed);
            for result in join_all(leases.iter().map(|l| self.process(l))).await {
                report.processed += 1;
                match result {
                    Ok(Processed::Lost) => report.lost += 1,
                    Ok(_) => {}
                    Err(e) => report.errors.push(e.to_string()),
                }
            }
        }
        Ok(report)
    }

    /// Worker mode: ticks in a loop, sleeping `idle_ms` after a tick that
    /// processed nothing or failed to claim. Returns once `shutdown` is
    /// true (or its sender is gone), after the tick in progress.
    pub async fn work(&self, opts: WorkOptions, mut shutdown: watch::Receiver<bool>) {
        while !*shutdown.borrow() {
            let idle = self
                .tick(opts.tick)
                .await
                .map_or(true, |r| r.processed == 0);
            if !idle || *shutdown.borrow() {
                continue;
            }
            tokio::select! {
                _ = tokio::time::sleep(Duration::from_millis(opts.idle_ms)) => {}
                changed = shutdown.changed() => {
                    if changed.is_err() {
                        return;
                    }
                }
            }
        }
    }

    /// Answers human node `node` of `run_id` with `answer`, saves the run
    /// under the next-step rule and enqueues it now. Returns `Resume` when
    /// the answer does not fit, or `Conflict` when the run moved meanwhile.
    pub async fn resume(&self, run_id: RunId, node: &str, answer: Value) -> Result<(), HostError> {
        let mut run = self.load(run_id).await?;
        run.resume(node, answer)?;
        self.save_and_enqueue(&run).await
    }

    /// Replaces the limits of `run_id` with `limits`, resumes its budget
    /// pauses, saves it under the next-step rule and enqueues it now.
    /// Returns the graph, store or queue error.
    pub async fn raise_budget(&self, run_id: RunId, limits: Limits) -> Result<(), HostError> {
        let mut run = self.load(run_id).await?;
        run.raise_budget(limits)?;
        self.save_and_enqueue(&run).await
    }

    async fn load(&self, run_id: RunId) -> Result<GraphRun, HostError> {
        let cp = self
            .checkpoints
            .latest(run_id)
            .await?
            .ok_or(HostError::NoCheckpoint(run_id))?;
        Ok(GraphRun::from_checkpoint(&cp)?)
    }

    async fn save_and_enqueue(&self, run: &GraphRun) -> Result<(), HostError> {
        self.checkpoints.save(run.checkpoint()).await?;
        self.queue.enqueue(run.run_id(), self.now()).await?;
        Ok(())
    }
}
