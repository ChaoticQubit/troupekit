use crate::error::TroupekitError;
use crate::host::{EventListener, ListenerBridge};
use crate::records::{Event, RunResult};
use std::sync::Arc;
use troupekit_ffi_core as neutral;

/// The events of one run, pulled with `next` or pushed to a listener.
/// Wraps the neutral `RunStream`; every method forwards unchanged.
/// Created by `AgentHandle::stream` and `GraphHandle::stream`.
#[derive(Debug, uniffi::Object)]
pub struct RunStream(pub(crate) Arc<neutral::RunStream>);

#[uniffi::export]
impl RunStream {
    pub async fn next(&self) -> Result<Option<Event>, TroupekitError> {
        Ok(self.0.next().await?.map(Into::into))
    }

    pub fn set_listener(&self, listener: Arc<dyn EventListener>) -> Result<(), TroupekitError> {
        Ok(self.0.set_listener(Arc::new(ListenerBridge(listener)))?)
    }

    pub fn cancel(&self) {
        self.0.cancel()
    }

    pub fn run_id(&self) -> String {
        self.0.run_id()
    }

    pub fn dropped_deltas(&self) -> u64 {
        self.0.dropped_deltas()
    }

    pub async fn result(&self) -> Result<RunResult, TroupekitError> {
        Ok(self.0.result().await?.into())
    }
}
