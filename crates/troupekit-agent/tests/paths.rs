use std::sync::Arc;
use troupekit_agent::budget::InMemoryLedger;
use troupekit_core::event::Usage;

#[tokio::test]
async fn old_ledger_paths_name_the_core_trait() {
    let ledger = Arc::new(InMemoryLedger::default());
    let core: Arc<dyn troupekit_core::store::UsageLedger> = ledger.clone();
    let root: Arc<dyn troupekit_agent::UsageLedger> = ledger.clone();
    let module: Arc<dyn troupekit_agent::budget::UsageLedger> = ledger;
    let usage = Usage {
        input_tokens: 1,
        output_tokens: 2,
        cost_usd_micros: 3,
    };
    core.record("k", usage).await.unwrap();
    root.record("k", usage).await.unwrap();
    assert_eq!(module.period_total("k").await.unwrap(), 6);
}
