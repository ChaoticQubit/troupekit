use crate::budget::{InMemoryLedger, UsageLedger};
use crate::error::AgentError;
use crate::eval::EvalRegistry;
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use troupekit_core::clock::{Clock, SystemClock};
use troupekit_core::event::Event;
use troupekit_providers::{Model, Price};
use troupekit_tools::ToolRegistry;

/// A model ready to call plus the price its replies are charged at.
/// The price travels with the model because a type-erased model hides
/// it, and budgets need it before the call.
#[derive(Clone)]
pub struct ModelHandle {
    pub model: Arc<dyn Model>,
    pub price: Price,
}

/// Receives every event a run emits, in `seq` order.
/// Synchronous so it can be called while a reply streams; hosts that
/// write to a database queue the event and return.
pub trait EventSink: Send + Sync {
    fn emit(&self, event: Event);
}

/// An `EventSink` that keeps every event in memory.
/// For tests and examples; `events` returns a copy.
/// Safe to share across threads.
#[derive(Debug, Default)]
pub struct VecSink(Mutex<Vec<Event>>);

impl VecSink {
    /// Copies every event received so far.
    /// Returns them oldest first, in `seq` order for a single run.
    /// The sink keeps its own copy.
    pub fn events(&self) -> Vec<Event> {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }
}

impl EventSink for VecSink {
    fn emit(&self, event: Event) {
        self.0.lock().unwrap_or_else(|p| p.into_inner()).push(event);
    }
}

/// Everything live a run needs, supplied by the host on every step:
/// model resolution by persona selector (`vendor/model` or a role alias),
/// tools, evals, the usage ledger, the event sink and the clock.
#[async_trait]
pub trait AgentIo: Send + Sync {
    async fn model(&self, selector: &str) -> Result<ModelHandle, AgentError>;
    fn tools(&self) -> &ToolRegistry;
    fn evals(&self) -> &EvalRegistry;
    fn ledger(&self) -> &dyn UsageLedger;
    fn events(&self) -> &dyn EventSink;
    fn clock(&self) -> &dyn Clock;
}

/// An `AgentIo` built from plain fields: models by exact selector, a tool
/// registry, evals, a ledger, a sink and a clock. Good for tests, examples
/// and single-user hosts; `new` starts with in-memory parts.
pub struct SimpleIo {
    pub models: HashMap<String, ModelHandle>,
    pub tools: ToolRegistry,
    pub evals: EvalRegistry,
    pub ledger: Arc<dyn UsageLedger>,
    pub events: Arc<dyn EventSink>,
    pub clock: Arc<dyn Clock>,
}

impl SimpleIo {
    /// Builds an io over `tools` with no models, the built-in evals, an
    /// in-memory ledger, a `VecSink` and the system clock.
    /// Returns the io; add models with `with_model`.
    pub fn new(tools: ToolRegistry) -> Self {
        Self {
            models: HashMap::new(),
            tools,
            evals: EvalRegistry::new(),
            ledger: Arc::new(InMemoryLedger::default()),
            events: Arc::new(VecSink::default()),
            clock: Arc::new(SystemClock),
        }
    }

    /// Serves `model` at `price` for persona selector `selector`,
    /// replacing any model already served for it.
    /// Returns the io.
    pub fn with_model(mut self, selector: &str, model: Arc<dyn Model>, price: Price) -> Self {
        self.models
            .insert(selector.to_string(), ModelHandle { model, price });
        self
    }
}

#[async_trait]
impl AgentIo for SimpleIo {
    async fn model(&self, selector: &str) -> Result<ModelHandle, AgentError> {
        self.models
            .get(selector)
            .cloned()
            .ok_or_else(|| AgentError::Setup(format!("no model for `{selector}`")))
    }

    fn tools(&self) -> &ToolRegistry {
        &self.tools
    }

    fn evals(&self) -> &EvalRegistry {
        &self.evals
    }

    fn ledger(&self) -> &dyn UsageLedger {
        self.ledger.as_ref()
    }

    fn events(&self) -> &dyn EventSink {
        self.events.as_ref()
    }

    fn clock(&self) -> &dyn Clock {
        self.clock.as_ref()
    }
}
