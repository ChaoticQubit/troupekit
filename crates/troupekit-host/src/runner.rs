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
use troupekit_core::store::{CheckpointStore, ForkLink};
use troupekit_graph::{GraphError, GraphIo, GraphOutcome, GraphRun, Stop};

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

    /// Processes one claimed `lease`: restore with its kept branch writes,
    /// step durably (timed as active work, or stopped once over its time
    /// limit), save when it moved, then queue the outcome. A conflict
    /// returns `Lost`; store errors reschedule after one second, unsaved.
    pub async fn process(&self, lease: &Lease) -> Result<Processed, HostError> {
        match self.checkpoints.latest(lease.run_id).await {
            Ok(Some(_)) => {}
            Ok(None) => {
                self.queue.fail(lease, "no checkpoint").await?;
                return Err(HostError::NoCheckpoint(lease.run_id));
            }
            Err(e) => return Err(self.store_failed(lease, e).await),
        }
        let mut run = match GraphRun::restore(self.checkpoints.as_ref(), lease.run_id).await {
            Ok(run) => run,
            Err(GraphError::Store(e)) => return Err(self.store_failed(lease, e).await),
            Err(e) => {
                self.queue.fail(lease, &e.to_string()).await?;
                return Err(e.into());
            }
        };
        let before = run.steps();
        let stepped = if run.over_time() {
            let limit_ms = run
                .time_limit_ms()
                .expect("only a limited run is over time");
            let _finished = run.stop(Stop::TimeLimit { limit_ms });
            run.step_durable(self.io.as_ref(), self.checkpoints.as_ref())
                .await
        } else {
            let started = self.now();
            let stepped = run
                .step_durable(self.io.as_ref(), self.checkpoints.as_ref())
                .await;
            run.add_active_ms(u64::try_from(self.now() - started).unwrap_or(0));
            stepped
        };
        let outcome = match stepped {
            Ok(outcome) => outcome,
            Err(GraphError::Store(CoreError::Conflict { .. })) => return Ok(Processed::Lost),
            Err(GraphError::Store(e)) => return Err(self.store_failed(lease, e).await),
            Err(e) => return Err(self.step_failed(lease, e).await),
        };
        let moved = run.steps() != before;
        if moved {
            match self.save_run(&run).await {
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
        self.step_failed(lease, e.into()).await
    }

    async fn step_failed(&self, lease: &Lease, e: GraphError) -> HostError {
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

    /// Forks run `run_id` at its kept step `step` into a new run: saves
    /// the child as its step 0, links it to the parent and enqueues it now.
    /// Returns the child id, or `NoCheckpoint` for a missing run or step.
    /// The parent run and its checkpoints never change.
    pub async fn fork(&self, run_id: RunId, step: u64) -> Result<RunId, HostError> {
        let history = self.checkpoints.history(run_id).await?;
        let cp = history
            .iter()
            .find(|cp| cp.step == step)
            .ok_or(HostError::NoCheckpoint(run_id))?;
        let child = GraphRun::fork(cp)?;
        assert_eq!(
            child.parent(),
            Some((run_id, step)),
            "a fork links to its source"
        );
        assert_eq!(child.steps(), 0, "a fork starts at step 0");
        self.checkpoints.save(child.checkpoint()).await?;
        let _kept = self
            .checkpoints
            .link_fork(ForkLink::new(child.run_id(), run_id, step))
            .await?;
        self.queue.enqueue(child.run_id(), self.now()).await?;
        Ok(child.run_id())
    }

    /// Loads run `run_id` with `GraphRun::restore`, so the branch writes
    /// its latest checkpoint carries stay pending across the change the
    /// caller makes next. Returns `NoCheckpoint` for an unknown run.
    async fn load(&self, run_id: RunId) -> Result<GraphRun, HostError> {
        if self.checkpoints.latest(run_id).await?.is_none() {
            return Err(HostError::NoCheckpoint(run_id));
        }
        Ok(GraphRun::restore(self.checkpoints.as_ref(), run_id).await?)
    }

    /// Saves `run`'s checkpoint, then drops the branch writes below the
    /// lowest step the saved run still reads. Input: a run between steps.
    /// Output: the save error, or the drop error after a durable save.
    async fn save_run(&self, run: &GraphRun) -> Result<(), CoreError> {
        self.checkpoints.save(run.checkpoint()).await?;
        let _dropped = self
            .checkpoints
            .drop_branch_writes(run.run_id(), run.kept_writes_from())
            .await?;
        Ok(())
    }

    async fn save_and_enqueue(&self, run: &GraphRun) -> Result<(), HostError> {
        self.save_run(run).await?;
        self.queue.enqueue(run.run_id(), self.now()).await?;
        Ok(())
    }
}
