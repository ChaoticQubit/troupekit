use crate::error::WorkflowError;
use crate::flow_check::{self, FlowFile};
use crate::workflow::{Ask, NodeKind, Workflow};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use troupekit_agent::dsl::Call;
use troupekit_agent::persona::toml_error;

/// Checkpoint format of a graph run that uses a 0.5 flow feature or 0.5
/// run state. Holds the format 1 keys plus the new private keys. A run
/// that uses neither keeps writing `GRAPH_FORMAT` 1.
pub const GRAPH_FORMAT_FLOW: u32 = 2;

/// How a join closes its fan-out: `All` waits for every branch, `Any`
/// continues with the first arrival and drops the other branches.
/// Written `mode = "all"` or `mode = "any"`; the default is `All`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JoinMode {
    #[default]
    All,
    Any,
}

/// How writes of several branches to one state key combine within one
/// step, always applied in ascending branch index. Written under
/// `[reducers]` as `append`, `extend`, `merge`, `sum`, `min`, `max`, `last`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Reducer {
    Append,
    Extend,
    Merge,
    Sum,
    Min,
    Max,
    Last,
}

/// How concurrent agent branches share the remaining run budget: `Even`
/// gives each an equal share, `Shared` lets each see the whole remainder
/// as in 0.1. Written under `[budget]` as `split`; the default is `Even`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BudgetSplit {
    #[default]
    Even,
    Shared,
}

/// The work a flow node does: every `NodeKind` plus switch, map and
/// subflow nodes, goto targets, delegates, model overrides, deferred
/// entry and join modes. Each struct-like variant is non-exhaustive.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FlowKind {
    #[non_exhaustive]
    Agent {
        persona: String,
        prompt: Option<String>,
        goto: Vec<String>,
        delegates: Vec<String>,
        model: Option<String>,
        defer: bool,
    },
    #[non_exhaustive]
    Tool {
        tool: String,
        defer: bool,
    },
    #[non_exhaustive]
    Human {
        ask: Ask,
        prompt: Option<String>,
    },
    #[non_exhaustive]
    Router {
        condition: Call,
    },
    #[non_exhaustive]
    Switch {
        on: String,
    },
    #[non_exhaustive]
    Map {
        over: String,
        item: String,
        body: String,
        max_items: u32,
        concurrency: u32,
    },
    #[non_exhaustive]
    Flow {
        flow: String,
        returns: String,
        defer: bool,
    },
    #[non_exhaustive]
    Join {
        mode: JoinMode,
    },
    End,
}

impl FlowKind {
    /// Converts a 0.1 node kind into its flow kind with every 0.5 option
    /// at its default: no goto, delegates or model override, no deferred
    /// entry, an all-join. Returns the flow kind.
    fn from_v1(kind: &NodeKind) -> FlowKind {
        match kind {
            NodeKind::Agent { persona, prompt } => FlowKind::Agent {
                persona: persona.clone(),
                prompt: prompt.clone(),
                goto: Vec::new(),
                delegates: Vec::new(),
                model: None,
                defer: false,
            },
            NodeKind::Tool { tool } => FlowKind::Tool {
                tool: tool.clone(),
                defer: false,
            },
            NodeKind::Human { ask, prompt } => FlowKind::Human {
                ask: *ask,
                prompt: prompt.clone(),
            },
            NodeKind::Router { condition } => FlowKind::Router {
                condition: condition.clone(),
            },
            NodeKind::Join => FlowKind::Join {
                mode: JoinMode::All,
            },
            NodeKind::End => FlowKind::End,
        }
    }
}

/// A validated flow: the superset of a workflow that every 0.5 pattern
/// compiles to. Holds the outline `Workflow` (read-only, never executed)
/// plus per-node flow kinds, per-key reducers, untracked keys and the
/// budget split. Plain data, so a graph run keeps a snapshot of it.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Flow {
    workflow: Workflow,
    kinds: BTreeMap<String, FlowKind>,
    reducers: BTreeMap<String, Reducer>,
    untracked: Vec<String>,
    budget_split: BudgetSplit,
    nested: bool,
}

impl Flow {
    /// Parses and validates flow TOML `text` read from `file`, the
    /// superset of workflow TOML. A file with no flow key goes through
    /// `Workflow::parse` first, so it gives an equal flow or the same error
    /// unless only a flow feature (a nested fan-out) makes it valid.
    pub fn parse(file: &str, text: &str) -> Result<Flow, WorkflowError> {
        let raw: FlowFile = match toml::from_str(text) {
            Ok(raw) => raw,
            Err(e) => {
                let p = toml_error(file, text, &e);
                let flow = WorkflowError::new(&p.file, &p.field, p.message);
                return Err(match Workflow::parse(file, text) {
                    Err(w) if w.field == flow.field => w,
                    _ => flow,
                });
            }
        };
        if !raw.is_workflow_file() {
            return flow_check::check(file, raw);
        }
        match Workflow::parse(file, text) {
            Ok(workflow) => Ok(Flow::from(workflow)),
            Err(e) => flow_check::check(file, raw).map_err(|_| e),
        }
    }

    /// Builds a flow from validated parts; only the flow checker calls it.
    /// `kinds` holds one kind per outline node.
    /// Returns the flow.
    pub(crate) fn from_parts(
        workflow: Workflow,
        kinds: BTreeMap<String, FlowKind>,
        reducers: BTreeMap<String, Reducer>,
        untracked: Vec<String>,
        budget_split: BudgetSplit,
        nested: bool,
    ) -> Flow {
        assert_eq!(kinds.len(), workflow.nodes.len(), "one kind per node");
        assert!(
            workflow.nodes.iter().all(|n| kinds.contains_key(&n.id)),
            "every node has a kind"
        );
        Flow {
            workflow,
            kinds,
            reducers,
            untracked,
            budget_split,
            nested,
        }
    }

    /// Returns the outline workflow: same name, start, caps, state, node
    /// ids, reads, writes and edges, with 0.5-only kinds shown as
    /// `flow-*` placeholder tools. For reading only.
    pub fn workflow(&self) -> &Workflow {
        &self.workflow
    }

    /// Returns the flow kind of node `node`.
    /// The kind holds every 0.5 option of the node.
    /// `None` for an unknown id.
    pub fn kind(&self, node: &str) -> Option<&FlowKind> {
        self.kinds.get(node)
    }

    /// Returns the reducer declared for state key `key`.
    /// Writes of several branches to that key combine through it.
    /// `None` when the key has no reducer.
    pub fn reducer(&self, key: &str) -> Option<Reducer> {
        self.reducers.get(key).copied()
    }

    /// Returns true when the flow uses no 0.5 feature: every kind is a
    /// 0.1 kind at its defaults, no fan-out is nested and no reducer or
    /// untracked key exists. The budget split is a run setting and does
    /// not count.
    pub fn is_v1(&self) -> bool {
        !self.nested
            && self.reducers.is_empty()
            && self.untracked.is_empty()
            && self.workflow.nodes.len() == self.kinds.len()
            && self
                .workflow
                .nodes
                .iter()
                .all(|n| self.kinds.get(&n.id) == Some(&FlowKind::from_v1(&n.kind)))
    }

    /// Returns the flow's name.
    /// It is the outline workflow's name.
    /// Events of a run carry it.
    pub fn name(&self) -> &str {
        &self.workflow.name
    }

    /// Returns the id of the start node.
    /// A run's first branch enters it.
    /// It is a node of the outline.
    pub fn start(&self) -> &str {
        &self.workflow.start
    }

    /// Returns the cap on steps of one run.
    /// A run that reaches it fails.
    /// Defaults to 200.
    pub fn max_steps(&self) -> u32 {
        self.workflow.max_steps
    }

    /// Returns the cap on transient retries in a row.
    /// A run past it fails as retryable.
    /// Defaults to 5.
    pub fn max_retries(&self) -> u32 {
        self.workflow.max_retries
    }

    /// Returns the state keys that are never checkpointed.
    /// Each is nullable and reads `null` after a resume.
    /// Empty when the flow has none.
    pub fn untracked(&self) -> &[String] {
        &self.untracked
    }

    /// Returns how concurrent agent branches share the run budget.
    /// Written `[budget] split`.
    /// Defaults to `BudgetSplit::Even`.
    pub fn budget_split(&self) -> BudgetSplit {
        self.budget_split
    }
}

impl From<Workflow> for Flow {
    /// Wraps a validated 0.1 workflow as a flow: each node keeps its kind
    /// with every 0.5 option at its default, no reducers, no untracked
    /// keys and the default budget split. The workflow is the outline.
    fn from(workflow: Workflow) -> Flow {
        let kinds: BTreeMap<String, FlowKind> = workflow
            .nodes
            .iter()
            .map(|n| (n.id.clone(), FlowKind::from_v1(&n.kind)))
            .collect();
        assert_eq!(
            kinds.len(),
            workflow.nodes.len(),
            "workflow node ids are unique"
        );
        Flow {
            workflow,
            kinds,
            reducers: BTreeMap::new(),
            untracked: Vec::new(),
            budget_split: BudgetSplit::default(),
            nested: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINEAR: &str = r#"
name = "two-steps"
start = "plan"

[state]
goal = "string"
plan = "string"

[[nodes]]
id = "plan"
kind = "agent"
persona = "planner"
reads = ["goal"]
writes = "plan"

[[nodes]]
id = "done"
kind = "end"

[[edges]]
from = "plan"
to = "done"
"#;

    #[test]
    fn flow_from_workflow_keeps_the_workflow() {
        let workflow = Workflow::parse("two.toml", LINEAR).unwrap();
        let flow = Flow::from(workflow.clone());
        assert_eq!(flow.workflow(), &workflow);
        assert!(flow.is_v1());
        assert_eq!(flow.name(), "two-steps");
        assert_eq!(flow.start(), "plan");
        assert_eq!(flow.max_steps(), workflow.max_steps);
        assert_eq!(flow.max_retries(), workflow.max_retries);
        assert_eq!(flow.reducer("plan"), None);
        assert!(matches!(
            flow.kind("plan"),
            Some(FlowKind::Agent { persona, goto, defer: false, .. }) if persona == "planner" && goto.is_empty()
        ));
        assert_eq!(flow.kind("done"), Some(&FlowKind::End));
        assert_eq!(flow.kind("missing"), None);
        let back: Flow = serde_json::from_value(serde_json::to_value(&flow).unwrap()).unwrap();
        assert_eq!(back, flow);
    }
}
