use async_trait::async_trait;
use std::sync::Arc;
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;

fn take_lease(lease: troupekit_core::store::Lease) -> u64 {
    lease.token
}

fn take_state(state: troupekit_core::store::JobState) -> troupekit_core::store::JobState {
    state
}

struct NoQueue;

#[async_trait]
impl troupekit_host::JobQueue for NoQueue {
    async fn enqueue(&self, _run_id: RunId, _due_ms: i64) -> Result<(), CoreError> {
        Ok(())
    }
    async fn claim_due(
        &self,
        _limit: usize,
        _lease_ms: i64,
    ) -> Result<Vec<troupekit_host::Lease>, CoreError> {
        Ok(Vec::new())
    }
    async fn complete(&self, _lease: &troupekit_host::Lease) -> Result<(), CoreError> {
        Ok(())
    }
    async fn fail(&self, _lease: &troupekit_host::Lease, _error: &str) -> Result<(), CoreError> {
        Ok(())
    }
    async fn reschedule(
        &self,
        _lease: &troupekit_host::Lease,
        _due_ms: i64,
    ) -> Result<(), CoreError> {
        Ok(())
    }
}

#[tokio::test]
async fn old_queue_paths_name_the_core_types() {
    let run = RunId::new();
    assert_eq!(
        take_lease(troupekit_host::Lease {
            run_id: run,
            token: 2,
            attempts: 0
        }),
        2
    );
    assert_eq!(
        take_lease(troupekit_host::queue::Lease {
            run_id: run,
            token: 3,
            attempts: 1
        }),
        3
    );
    assert_eq!(
        take_state(troupekit_host::JobState::Done),
        troupekit_core::store::JobState::Done
    );
    assert_eq!(
        take_state(troupekit_host::queue::JobState::Pending { due_ms: 1 }),
        troupekit_core::store::JobState::Pending { due_ms: 1 }
    );
    let queue: Arc<dyn troupekit_core::store::JobQueue> = Arc::new(NoQueue);
    assert!(queue.claim_due(1, 10).await.unwrap().is_empty());
}
