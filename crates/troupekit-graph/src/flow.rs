use crate::error::WorkflowError;
use crate::flow_check::{self, FlowFile};
use crate::workflow::{Ask, NodeKind, Workflow};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
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

impl Reducer {
    /// Combines the writes of one step to one key with its `current` value
    /// (`None` or `null` when unset). `writes` holds `(branch index, value)`
    /// in ascending branch index; the caller checks the result's type.
    /// Returns the new value, or why a write does not fit.
    pub fn reduce(
        self,
        current: Option<&Value>,
        writes: &[(usize, Value)],
    ) -> Result<Value, String> {
        assert!(
            writes.windows(2).all(|w| w[0].0 < w[1].0),
            "writes are applied in ascending branch index"
        );
        let current = current.filter(|v| !v.is_null()).cloned();
        let Some((_, last)) = writes.last() else {
            return Ok(current.unwrap_or(Value::Null));
        };
        match self {
            Reducer::Last => Ok(last.clone()),
            Reducer::Append => {
                let mut list = list_of(current)?;
                list.extend(writes.iter().map(|(_, v)| v.clone()));
                Ok(Value::Array(list))
            }
            Reducer::Extend => {
                let mut list = list_of(current)?;
                for (b, v) in writes {
                    let Value::Array(items) = v else {
                        return Err(format!("`extend` takes a list; branch {b} wrote {v}"));
                    };
                    list.extend(items.iter().cloned());
                }
                Ok(Value::Array(list))
            }
            Reducer::Merge => {
                let mut out = match current {
                    None => Map::new(),
                    Some(Value::Object(m)) => m,
                    Some(v) => return Err(format!("`merge` needs an object; the key holds {v}")),
                };
                let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
                for (b, v) in writes {
                    let Value::Object(m) = v else {
                        return Err(format!("`merge` takes an object; branch {b} wrote {v}"));
                    };
                    for (k, item) in m {
                        if let Some(other) = seen.insert(k, *b) {
                            return Err(format!(
                                "`merge`: branches {other} and {b} both write `{k}` in one step; use `last` to keep the higher branch"
                            ));
                        }
                        out.insert(k.clone(), item.clone());
                    }
                }
                Ok(Value::Object(out))
            }
            Reducer::Sum | Reducer::Min | Reducer::Max => {
                let mut values: Vec<(Option<usize>, &Value)> =
                    current.iter().map(|v| (None, v)).collect();
                values.extend(writes.iter().map(|(b, v)| (Some(*b), v)));
                for (b, v) in &values {
                    if !v.is_number() {
                        let who =
                            b.map_or("the key holds".to_string(), |b| format!("branch {b} wrote"));
                        return Err(format!("`{}` takes a number; {who} {v}", self.name()));
                    }
                }
                match self {
                    Reducer::Sum => sum(values.iter().map(|(_, v)| *v)),
                    _ => {
                        let better =
                            |a: f64, b: f64| if self == Reducer::Min { b < a } else { b > a };
                        let mut best = values[0].1;
                        for (_, v) in &values[1..] {
                            if better(num(best), num(v)) {
                                best = v;
                            }
                        }
                        Ok(best.clone())
                    }
                }
            }
        }
    }

    fn name(self) -> &'static str {
        match self {
            Reducer::Append => "append",
            Reducer::Extend => "extend",
            Reducer::Merge => "merge",
            Reducer::Sum => "sum",
            Reducer::Min => "min",
            Reducer::Max => "max",
            Reducer::Last => "last",
        }
    }
}

fn list_of(current: Option<Value>) -> Result<Vec<Value>, String> {
    match current {
        None => Ok(Vec::new()),
        Some(Value::Array(items)) => Ok(items),
        Some(v) => Err(format!("the key holds {v}, not a list")),
    }
}

fn num(v: &Value) -> f64 {
    v.as_f64()
        .expect("reducer values are checked to be numbers")
}

fn sum<'v>(values: impl Iterator<Item = &'v Value> + Clone) -> Result<Value, String> {
    if values.clone().all(|v| v.as_i64().is_some()) {
        let total = values
            .map(|v| v.as_i64().expect("checked integer"))
            .try_fold(0i64, i64::checked_add)
            .ok_or("`sum` overflows a 64-bit integer")?;
        return Ok(Value::from(total));
    }
    let total: f64 = values.map(num).sum();
    serde_json::Number::from_f64(total)
        .map(Value::Number)
        .ok_or_else(|| "`sum` is not a finite number".to_string())
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
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        fields: BTreeMap<String, String>,
        #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
        sets: BTreeMap<String, String>,
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
    pub(crate) fn from_v1(kind: &NodeKind) -> FlowKind {
        match kind {
            NodeKind::Agent { persona, prompt } => FlowKind::Agent {
                persona: persona.clone(),
                prompt: prompt.clone(),
                goto: Vec::new(),
                delegates: Vec::new(),
                model: None,
                defer: false,
                fields: BTreeMap::new(),
                sets: BTreeMap::new(),
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
    /// `Workflow::parse` first, so it gives an equal flow or the same error,
    /// except where a nested fan-out makes the 0.1 verdict wrong.
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
            return flow_check::check(file, raw).map_err(|r| r.error);
        }
        match Workflow::parse(file, text) {
            Ok(workflow) => Ok(Flow::from(workflow)),
            Err(e) => flow_check::check(file, raw).map_err(|r| if r.nested { r.error } else { e }),
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

    fn writes(list: &[(usize, Value)]) -> Vec<(usize, Value)> {
        list.to_vec()
    }

    #[test]
    fn append_keeps_branch_index_order() {
        use serde_json::json;
        let w = writes(&[(0, json!("a")), (1, json!("b")), (3, json!("d"))]);
        assert_eq!(
            Reducer::Append.reduce(None, &w).unwrap(),
            json!(["a", "b", "d"])
        );
        assert_eq!(
            Reducer::Append.reduce(Some(&json!(["x"])), &w).unwrap(),
            json!(["x", "a", "b", "d"])
        );
        assert_eq!(
            Reducer::Append.reduce(Some(&Value::Null), &w[..1]).unwrap(),
            json!(["a"])
        );
        let lists = writes(&[(0, json!([1, 2])), (2, json!([3]))]);
        assert_eq!(
            Reducer::Extend.reduce(None, &lists).unwrap(),
            json!([1, 2, 3])
        );
        assert!(Reducer::Extend
            .reduce(None, &w)
            .unwrap_err()
            .contains("branch 0"));
        assert!(Reducer::Append.reduce(Some(&json!("s")), &w).is_err());
        assert_eq!(
            Reducer::Append.reduce(Some(&json!(["x"])), &[]).unwrap(),
            json!(["x"])
        );
    }

    #[test]
    #[should_panic(expected = "ascending branch index")]
    fn reduce_refuses_writes_out_of_branch_order() {
        let w = writes(&[(1, Value::from(1)), (0, Value::from(2))]);
        let _ = Reducer::Sum.reduce(None, &w);
    }

    #[test]
    fn merge_conflict_takes_the_higher_branch_index_only_with_last() {
        use serde_json::json;
        let clash = writes(&[(0, json!({"a": 1, "b": 2})), (2, json!({"a": 3}))]);
        let e = Reducer::Merge.reduce(None, &clash).unwrap_err();
        assert!(e.contains("branches 0 and 2 both write `a`"), "{e}");
        assert_eq!(Reducer::Last.reduce(None, &clash).unwrap(), json!({"a": 3}));
        assert_eq!(
            Reducer::Last
                .reduce(Some(&json!({"z": 0})), &clash)
                .unwrap(),
            json!({"a": 3})
        );
        let apart = writes(&[(0, json!({"a": 1})), (1, json!({"b": 2}))]);
        assert_eq!(
            Reducer::Merge
                .reduce(Some(&json!({"a": 0, "c": 9})), &apart)
                .unwrap(),
            json!({"a": 1, "b": 2, "c": 9})
        );
        assert!(Reducer::Merge
            .reduce(None, &writes(&[(0, json!(1))]))
            .is_err());
    }

    #[test]
    fn number_reducers_fold_in_branch_order() {
        use serde_json::json;
        let w = writes(&[(0, json!(3)), (1, json!(-2)), (2, json!(5))]);
        assert_eq!(
            Reducer::Sum.reduce(Some(&json!(10)), &w).unwrap(),
            json!(16)
        );
        assert_eq!(Reducer::Min.reduce(None, &w).unwrap(), json!(-2));
        assert_eq!(Reducer::Max.reduce(Some(&json!(4)), &w).unwrap(), json!(5));
        assert_eq!(
            Reducer::Sum
                .reduce(None, &writes(&[(0, json!(1.5)), (1, json!(1))]))
                .unwrap(),
            json!(2.5)
        );
        assert!(Reducer::Sum
            .reduce(None, &writes(&[(0, json!(i64::MAX)), (1, json!(1))]))
            .is_err());
        assert!(Reducer::Max
            .reduce(None, &writes(&[(0, json!("x"))]))
            .unwrap_err()
            .contains("branch 0"));
        assert_eq!(
            Reducer::Max
                .reduce(None, &writes(&[(0, json!(2)), (1, json!(2.0))]))
                .unwrap(),
            json!(2)
        );
    }

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
