use std::collections::BTreeMap;
use std::sync::Mutex;
use troupekit_agent::EventSink;
use troupekit_core::event::Event;
use troupekit_core::ids::RunId;

/// An in-memory event store that upserts on `(run_id, seq)`, as a durable
/// one does with a unique key: a step replayed after a crash re-emits the
/// same numbers, which replace the earlier copies instead of duplicating.
#[derive(Debug, Default)]
pub struct EventLog(Mutex<BTreeMap<(RunId, u64), Event>>);

impl EventLog {
    /// Returns every stored event of `run_id` in `seq` order.
    /// A replayed `seq` appears once, holding its latest copy.
    /// Empty for an unknown run.
    pub fn events(&self, run_id: RunId) -> Vec<Event> {
        let map = self.0.lock().unwrap_or_else(|p| p.into_inner());
        map.range((run_id, 0)..=(run_id, u64::MAX))
            .map(|(_, e)| e.clone())
            .collect()
    }
}

impl EventSink for EventLog {
    fn emit(&self, event: Event) {
        let mut map = self.0.lock().unwrap_or_else(|p| p.into_inner());
        map.insert((event.run_id, event.seq), event);
    }
}
