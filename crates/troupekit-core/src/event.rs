use crate::ids::RunId;
use serde::{Deserialize, Serialize};
use std::ops::{Add, AddAssign};

/// Token and cost totals for a step or a run.
/// Cost is in millionths of a US dollar so sums stay exact.
/// Totals combine with `+` and `+=`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cost_usd_micros: u64,
}

impl Add for Usage {
    type Output = Usage;
    fn add(self, o: Usage) -> Usage {
        Usage {
            input_tokens: self.input_tokens + o.input_tokens,
            output_tokens: self.output_tokens + o.output_tokens,
            cost_usd_micros: self.cost_usd_micros + o.cost_usd_micros,
        }
    }
}

impl AddAssign for Usage {
    fn add_assign(&mut self, o: Usage) {
        *self = *self + o;
    }
}

/// Why a run stopped and waits.
/// A human approval at a node, a human answer at a node, or a budget limit.
/// Serializes with a `kind` tag in snake case.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PauseReason {
    Approval { node: String },
    Question { node: String },
    Budget { limit: String },
}

/// What happened in a run.
/// One enum feeds the UI stream, telemetry and checkpoints.
/// Serializes with a `type` tag in snake case; `StepStarted` names the
/// persona slug of an agent node's step, left out when there is none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventBody {
    RunStarted {
        workflow: String,
    },
    StepStarted {
        step: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        persona: Option<String>,
    },
    TextDelta {
        text: String,
    },
    ToolCallStarted {
        call_id: String,
        name: String,
    },
    ToolCallFinished {
        call_id: String,
        is_error: bool,
    },
    StepFinished {
        step: String,
        usage: Usage,
    },
    RunPaused {
        reason: PauseReason,
    },
    RunFinished {
        usage: Usage,
    },
    RunFailed {
        error: String,
    },
}

/// One numbered event of one run.
/// `seq` increases by one per event within a run and `at_ms` is the
/// Unix time in milliseconds when it happened.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub run_id: RunId,
    pub seq: u64,
    pub at_ms: i64,
    pub body: EventBody,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn usage_adds() {
        let mut u = Usage { input_tokens: 10, output_tokens: 5, cost_usd_micros: 30 };
        u += Usage { input_tokens: 1, output_tokens: 2, cost_usd_micros: 3 };
        assert_eq!(u, Usage { input_tokens: 11, output_tokens: 7, cost_usd_micros: 33 });
    }

    #[test]
    fn event_json_is_tagged() {
        let id = RunId::new();
        let e = Event { run_id: id, seq: 3, at_ms: 9, body: EventBody::RunPaused { reason: PauseReason::Approval { node: "approve_plan".into() } } };
        let v = serde_json::to_value(&e).unwrap();
        assert_eq!(v["body"], json!({"type":"run_paused","reason":{"kind":"approval","node":"approve_plan"}}));
        let back: Event = serde_json::from_value(v).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn step_started_persona_is_optional() {
        let old: EventBody = serde_json::from_value(json!({"type":"step_started","step":"plan/call_model"})).unwrap();
        assert_eq!(old, EventBody::StepStarted { step: "plan/call_model".into(), persona: None });
        assert_eq!(serde_json::to_value(&old).unwrap(), json!({"type":"step_started","step":"plan/call_model"}));
        let new = EventBody::StepStarted { step: "plan/call_model".into(), persona: Some("goal-planner".into()) };
        let v = serde_json::to_value(&new).unwrap();
        assert_eq!(v, json!({"type":"step_started","step":"plan/call_model","persona":"goal-planner"}));
        assert_eq!(serde_json::from_value::<EventBody>(v).unwrap(), new);
    }
}
