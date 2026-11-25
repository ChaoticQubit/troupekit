use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use troupekit_core::clock::Clock;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;

pub use troupekit_core::store::{JobQueue, JobState, Lease};

#[derive(Debug)]
struct Job {
    state: JobState,
    attempts: u32,
}

#[derive(Debug, Default)]
struct Jobs {
    map: HashMap<RunId, Job>,
    next_token: u64,
}

/// In-memory `JobQueue` for tests, examples and single-process hosts.
/// Times come from the injected clock, so tests move leases forward
/// with a `FixedClock`.
pub struct InMemQueue {
    clock: Arc<dyn Clock>,
    jobs: Mutex<Jobs>,
}

impl InMemQueue {
    /// Builds an empty queue reading time from `clock`.
    /// Share the clock with the runner's io.
    /// Returns the queue.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            jobs: Mutex::new(Jobs::default()),
        }
    }

    /// Returns the state of `run_id`'s job.
    /// `None` when the run was never enqueued.
    /// Tests read it to check what the runner did.
    pub fn job(&self, run_id: RunId) -> Option<JobState> {
        self.lock().map.get(&run_id).map(|j| j.state.clone())
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Jobs> {
        self.jobs.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn end_lease(
        &self,
        lease: &Lease,
        f: impl FnOnce(Option<i64>) -> JobState,
    ) -> Result<(), CoreError> {
        let mut jobs = self.lock();
        let job = jobs
            .map
            .get_mut(&lease.run_id)
            .ok_or_else(|| CoreError::NotFound(format!("job for run {}", lease.run_id)))?;
        let JobState::Leased {
            token, next_due_ms, ..
        } = job.state
        else {
            return Err(CoreError::Conflict {
                expected: 0,
                actual: lease.token,
            });
        };
        if token != lease.token {
            return Err(CoreError::Conflict {
                expected: token,
                actual: lease.token,
            });
        }
        job.state = f(next_due_ms);
        job.attempts = 0;
        Ok(())
    }
}

fn earlier(a: Option<i64>, b: i64) -> i64 {
    a.map_or(b, |a| a.min(b))
}

#[async_trait]
impl JobQueue for InMemQueue {
    async fn enqueue(&self, run_id: RunId, due_ms: i64) -> Result<(), CoreError> {
        let mut jobs = self.lock();
        let job = jobs.map.entry(run_id).or_insert(Job {
            state: JobState::Done,
            attempts: 0,
        });
        job.state = match job.state.clone() {
            JobState::Pending { due_ms: d } => JobState::Pending {
                due_ms: d.min(due_ms),
            },
            JobState::Leased {
                token,
                until_ms,
                next_due_ms,
            } => JobState::Leased {
                token,
                until_ms,
                next_due_ms: Some(earlier(next_due_ms, due_ms)),
            },
            JobState::Done | JobState::Failed { .. } => JobState::Pending { due_ms },
        };
        Ok(())
    }

    async fn claim_due(&self, limit: usize, lease_ms: i64) -> Result<Vec<Lease>, CoreError> {
        assert!(lease_ms > 0, "a lease lasts some time");
        let now = self.clock.now_ms();
        let mut jobs = self.lock();
        let mut due: Vec<(i64, RunId)> = jobs
            .map
            .iter()
            .filter_map(|(id, j)| match j.state {
                JobState::Pending { due_ms } if due_ms <= now => Some((due_ms, *id)),
                JobState::Leased { until_ms, .. } if until_ms <= now => Some((until_ms, *id)),
                _ => None,
            })
            .collect();
        due.sort();
        due.truncate(limit);
        let mut leases = Vec::with_capacity(due.len());
        for (_, run_id) in due {
            jobs.next_token += 1;
            let token = jobs.next_token;
            let job = jobs.map.get_mut(&run_id).expect("job listed above");
            let next_due_ms = match job.state {
                JobState::Leased { next_due_ms, .. } => next_due_ms,
                _ => None,
            };
            job.state = JobState::Leased {
                token,
                until_ms: now + lease_ms,
                next_due_ms,
            };
            job.attempts += 1;
            leases.push(Lease {
                run_id,
                token,
                attempts: job.attempts,
            });
        }
        Ok(leases)
    }

    async fn complete(&self, lease: &Lease) -> Result<(), CoreError> {
        self.end_lease(lease, |next| match next {
            Some(due_ms) => JobState::Pending { due_ms },
            None => JobState::Done,
        })
    }

    async fn fail(&self, lease: &Lease, error: &str) -> Result<(), CoreError> {
        self.end_lease(lease, |_| JobState::Failed {
            error: error.to_string(),
        })
    }

    async fn reschedule(&self, lease: &Lease, due_ms: i64) -> Result<(), CoreError> {
        self.end_lease(lease, |next| JobState::Pending {
            due_ms: earlier(next, due_ms),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use troupekit_core::clock::FixedClock;

    fn queue() -> (Arc<FixedClock>, InMemQueue) {
        let clock = Arc::new(FixedClock::new(1_000));
        (clock.clone(), InMemQueue::new(clock))
    }

    #[tokio::test]
    async fn one_job_per_run() {
        let (_, q) = queue();
        let run = RunId::new();
        q.enqueue(run, 900).await.unwrap();
        q.enqueue(run, 950).await.unwrap();
        assert_eq!(q.job(run), Some(JobState::Pending { due_ms: 900 }));
        q.enqueue(run, 100).await.unwrap();
        assert_eq!(q.job(run), Some(JobState::Pending { due_ms: 100 }));
        let leases = q.claim_due(10, 500).await.unwrap();
        assert_eq!(leases.len(), 1);
        assert!(q.claim_due(10, 500).await.unwrap().is_empty());
        assert_eq!(q.job(RunId::new()), None);
    }

    #[tokio::test]
    async fn claims_due_in_order_with_leases() {
        let (clock, q) = queue();
        let (a, b, c) = (RunId::new(), RunId::new(), RunId::new());
        q.enqueue(a, 800).await.unwrap();
        q.enqueue(b, 700).await.unwrap();
        q.enqueue(c, 2_000).await.unwrap();
        let first = q.claim_due(1, 500).await.unwrap();
        assert_eq!(first.len(), 1);
        assert_eq!(first[0].run_id, b);
        assert_eq!(first[0].attempts, 1);
        let rest = q.claim_due(10, 500).await.unwrap();
        assert_eq!(rest.iter().map(|l| l.run_id).collect::<Vec<_>>(), vec![a]);
        assert_ne!(rest[0].token, first[0].token);
        assert_eq!(
            q.job(a),
            Some(JobState::Leased {
                token: rest[0].token,
                until_ms: 1_500,
                next_due_ms: None
            })
        );
        q.complete(&first[0]).await.unwrap();
        assert_eq!(q.job(b), Some(JobState::Done));
        q.fail(&rest[0], "boom").await.unwrap();
        assert_eq!(
            q.job(a),
            Some(JobState::Failed {
                error: "boom".into()
            })
        );
        clock.advance(1_000);
        let late = q.claim_due(10, 500).await.unwrap();
        assert_eq!(late.iter().map(|l| l.run_id).collect::<Vec<_>>(), vec![c]);
        q.reschedule(&late[0], 2_500).await.unwrap();
        assert_eq!(q.job(c), Some(JobState::Pending { due_ms: 2_500 }));
    }

    #[tokio::test]
    async fn expired_lease_is_reclaimed() {
        let (clock, q) = queue();
        let run = RunId::new();
        q.enqueue(run, 1_000).await.unwrap();
        let first = q.claim_due(10, 500).await.unwrap().remove(0);
        clock.advance(499);
        assert!(q.claim_due(10, 500).await.unwrap().is_empty());
        clock.advance(1);
        let second = q.claim_due(10, 500).await.unwrap().remove(0);
        assert_eq!(second.run_id, run);
        assert_eq!(second.attempts, 2);
        assert_ne!(second.token, first.token);
        q.reschedule(&second, 0).await.unwrap();
        assert_eq!(q.claim_due(10, 500).await.unwrap()[0].attempts, 1);
    }

    #[tokio::test]
    async fn stale_token_changes_nothing() {
        let (clock, q) = queue();
        let run = RunId::new();
        q.enqueue(run, 1_000).await.unwrap();
        let stale = q.claim_due(10, 500).await.unwrap().remove(0);
        clock.advance(500);
        let fresh = q.claim_due(10, 500).await.unwrap().remove(0);
        let held = q.job(run);
        for result in [
            q.complete(&stale).await,
            q.fail(&stale, "x").await,
            q.reschedule(&stale, 0).await,
        ] {
            assert!(
                matches!(result, Err(CoreError::Conflict { .. })),
                "{result:?}"
            );
        }
        assert_eq!(q.job(run), held);
        q.complete(&fresh).await.unwrap();
        assert!(matches!(
            q.complete(&fresh).await,
            Err(CoreError::Conflict { .. })
        ));
    }

    #[tokio::test]
    async fn enqueue_while_leased_applies_on_complete() {
        let (_, q) = queue();
        let run = RunId::new();
        q.enqueue(run, 1_000).await.unwrap();
        let lease = q.claim_due(10, 500).await.unwrap().remove(0);
        q.enqueue(run, 1_200).await.unwrap();
        q.enqueue(run, 1_100).await.unwrap();
        assert!(q.claim_due(10, 500).await.unwrap().is_empty());
        q.complete(&lease).await.unwrap();
        assert_eq!(q.job(run), Some(JobState::Pending { due_ms: 1_100 }));
        let lease = q.claim_due(10, 500).await.unwrap();
        assert!(lease.is_empty(), "not due until 1100");
        let lease2 = {
            q.enqueue(run, 1_000).await.unwrap();
            q.claim_due(10, 500).await.unwrap().remove(0)
        };
        q.enqueue(run, 5_000).await.unwrap();
        q.reschedule(&lease2, 3_000).await.unwrap();
        assert_eq!(q.job(run), Some(JobState::Pending { due_ms: 3_000 }));
    }
}
