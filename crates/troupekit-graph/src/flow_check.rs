use crate::error::WorkflowError;
use crate::flow::{BudgetSplit, Flow, FlowKind, JoinMode, Reducer};
use crate::workflow::{Ask, Edge, Node, NodeKind, Workflow};
use serde::Deserialize;
use serde_json::{json, Value};
use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use troupekit_agent::dsl::{is_ident, parse_call};
use troupekit_agent::persona::valid_name;
use troupekit_tools::valid_tool_name;

const MAX_FAN_OUT_DEPTH: usize = 4;
const DEFAULT_MAX_ITEMS: u32 = 256;
const MAX_ITEMS_RANGE: std::ops::RangeInclusive<i64> = 1..=10_000;
const DEFAULT_CONCURRENCY: u32 = 8;
const CONCURRENCY_RANGE: std::ops::RangeInclusive<i64> = 1..=256;
const MAX_LIST: usize = 256;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct FlowFile {
    name: String,
    start: String,
    max_steps: Option<i64>,
    max_retries: Option<i64>,
    #[serde(default)]
    state: toml::Table,
    untracked: Option<Vec<String>>,
    reducers: Option<BTreeMap<String, String>>,
    budget: Option<BudgetFile>,
    #[serde(default)]
    nodes: Vec<NodeFile>,
    #[serde(default)]
    edges: Vec<Edge>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct BudgetFile {
    split: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct NodeFile {
    id: String,
    kind: String,
    persona: Option<String>,
    prompt: Option<String>,
    tool: Option<String>,
    ask: Option<String>,
    condition: Option<String>,
    reads: Option<Vec<String>>,
    writes: Option<String>,
    on: Option<String>,
    over: Option<String>,
    item: Option<String>,
    body: Option<String>,
    max_items: Option<i64>,
    concurrency: Option<i64>,
    flow: Option<String>,
    returns: Option<String>,
    defer: Option<bool>,
    mode: Option<String>,
    goto: Option<Vec<String>>,
    delegates: Option<Vec<String>>,
    model: Option<String>,
}

impl NodeFile {
    fn uses_flow_fields(&self) -> bool {
        let flow_kind = matches!(self.kind.as_str(), "switch" | "map" | "flow");
        flow_kind
            || self.on.is_some()
            || self.over.is_some()
            || self.item.is_some()
            || self.body.is_some()
            || self.max_items.is_some()
            || self.concurrency.is_some()
            || self.flow.is_some()
            || self.returns.is_some()
            || self.defer.is_some()
            || self.mode.is_some()
            || self.goto.is_some()
            || self.delegates.is_some()
            || self.model.is_some()
    }
}

impl FlowFile {
    /// Returns true when the file uses no key beyond workflow TOML, so
    /// `Workflow::parse` reads it with the same result and messages.
    /// Any flow key, node kind or node field makes it false.
    pub(crate) fn is_workflow_file(&self) -> bool {
        self.untracked.is_none()
            && self.reducers.is_none()
            && self.budget.is_none()
            && !self.nodes.iter().any(NodeFile::uses_flow_fields)
    }
}

type Problem = (String, String);

/// Why the flow checker refused a file: the error, and whether the graph
/// check walked into a nested fan-out before failing, so `Flow::parse`
/// knows a 0.1 error would be false for that file.
pub(crate) struct Rejected {
    pub(crate) error: WorkflowError,
    pub(crate) nested: bool,
}

struct Spec {
    id: String,
    kind: FlowKind,
    reads: Vec<String>,
    writes: Option<String>,
}

/// Validates a parsed flow file `raw` read from `file` and builds the
/// flow: header, nodes, reducers, untracked keys, budget, then the graph.
/// Returns the flow, or the first problem naming file, dotted field and
/// what is wrong.
pub(crate) fn check(file: &str, raw: FlowFile) -> Result<Flow, Rejected> {
    let err = |(field, message): Problem| Rejected {
        error: WorkflowError::new(file, &field, message),
        nested: false,
    };
    let header = header(file, &raw).map_err(|error| Rejected {
        error,
        nested: false,
    })?;
    let state = &header.state;
    if raw.nodes.is_empty() || raw.nodes.len() > 256 {
        return Err(err((
            "nodes".into(),
            "a workflow has 1 to 256 nodes".into(),
        )));
    }
    let mut specs: Vec<Spec> = Vec::with_capacity(raw.nodes.len());
    for (i, n) in raw.nodes.into_iter().enumerate() {
        if !is_ident(&n.id) || n.id.len() > 64 {
            return Err(err((
                format!("nodes[{i}].id"),
                format!("`{}` must be 1 to 64 of [a-z_][a-z0-9_]*", n.id),
            )));
        }
        if specs.iter().any(|s| s.id == n.id) {
            return Err(err((
                format!("nodes[{i}].id"),
                format!("duplicate node `{}`", n.id),
            )));
        }
        let id = n.id.clone();
        specs.push(spec(n, state).map_err(|(f, m)| err((format!("nodes.{id}.{f}"), m)))?);
    }
    let reducers = reducers(raw.reducers.unwrap_or_default(), state).map_err(err)?;
    let untracked = untracked(raw.untracked.unwrap_or_default(), state).map_err(err)?;
    let budget_split = budget(raw.budget).map_err(err)?;
    let graph = Graph::new(&specs, &raw.edges, &header.start);
    let nested = graph.check(state, &reducers).map_err(|p| Rejected {
        nested: graph.nested_seen.get(),
        ..err(p)
    })?;
    let workflow = Workflow {
        name: header.name,
        start: header.start,
        max_steps: header.max_steps,
        max_retries: header.max_retries,
        state: header.state,
        nodes: specs.iter().map(outline_node).collect(),
        edges: outline_edges(&specs, raw.edges),
    };
    let kinds = specs.into_iter().map(|s| (s.id, s.kind)).collect();
    Ok(Flow::from_parts(
        workflow,
        kinds,
        reducers,
        untracked,
        budget_split,
        nested,
    ))
}

/// Validates the header (name, caps and state types) through
/// `Workflow::parse` of a one-node workflow, so messages match workflow
/// files exactly. Returns that workflow, whose start is a placeholder.
fn header(file: &str, raw: &FlowFile) -> Result<Workflow, WorkflowError> {
    let mut table = toml::Table::new();
    table.insert("name".into(), raw.name.clone().into());
    table.insert("start".into(), "end".into());
    if let Some(n) = raw.max_steps {
        table.insert("max_steps".into(), n.into());
    }
    if let Some(n) = raw.max_retries {
        table.insert("max_retries".into(), n.into());
    }
    table.insert("state".into(), raw.state.clone().into());
    let mut end = toml::Table::new();
    end.insert("id".into(), "end".into());
    end.insert("kind".into(), "end".into());
    table.insert("nodes".into(), toml::Value::Array(vec![end.into()]));
    let text = toml::to_string(&table).expect("a toml table serializes");
    let mut wf = Workflow::parse(file, &text)?;
    assert_eq!(wf.nodes.len(), 1, "the header workflow holds one node");
    wf.start = raw.start.clone();
    Ok(wf)
}

fn types(schema: &Value) -> Vec<&str> {
    match &schema["type"] {
        Value::String(t) => vec![t.as_str()],
        Value::Array(ts) => ts.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn has_type(schema: &Value, wanted: &[&str]) -> bool {
    types(schema).iter().any(|t| wanted.contains(t))
}

fn list_field(list: Vec<String>, field: &str) -> Result<Vec<String>, Problem> {
    let e = |m: String| (field.to_string(), m);
    if list.is_empty() || list.len() > MAX_LIST {
        return Err(e(format!("lists 1 to {MAX_LIST} entries")));
    }
    for (i, v) in list.iter().enumerate() {
        if list[..i].contains(v) {
            return Err(e(format!("`{v}` repeats")));
        }
    }
    Ok(list)
}

fn ranged(
    v: Option<i64>,
    range: std::ops::RangeInclusive<i64>,
    default: u32,
    field: &str,
) -> Result<u32, Problem> {
    match v {
        None => Ok(default),
        Some(n) if range.contains(&n) => Ok(u32::try_from(n).expect("range fits in u32")),
        Some(n) => Err((
            field.to_string(),
            format!("{n} must be between {} and {}", range.start(), range.end()),
        )),
    }
}

fn spec(n: NodeFile, state: &BTreeMap<String, Value>) -> Result<Spec, Problem> {
    let e = |f: &str, m: String| (f.to_string(), m);
    let set = [
        ("persona", n.persona.is_some()),
        ("prompt", n.prompt.is_some()),
        ("tool", n.tool.is_some()),
        ("ask", n.ask.is_some()),
        ("condition", n.condition.is_some()),
        ("reads", n.reads.is_some()),
        ("writes", n.writes.is_some()),
        ("on", n.on.is_some()),
        ("over", n.over.is_some()),
        ("item", n.item.is_some()),
        ("body", n.body.is_some()),
        ("max_items", n.max_items.is_some()),
        ("concurrency", n.concurrency.is_some()),
        ("flow", n.flow.is_some()),
        ("returns", n.returns.is_some()),
        ("defer", n.defer.is_some()),
        ("mode", n.mode.is_some()),
        ("goto", n.goto.is_some()),
        ("delegates", n.delegates.is_some()),
        ("model", n.model.is_some()),
    ];
    let allowed: &[&str] = match n.kind.as_str() {
        "agent" => &["persona", "prompt", "reads", "writes", "goto", "delegates", "model", "defer"],
        "tool" => &["tool", "reads", "writes", "defer"],
        "human" => &["ask", "prompt", "reads", "writes"],
        "router" => &["condition"],
        "switch" => &["on"],
        "map" => &["over", "item", "body", "writes", "max_items", "concurrency"],
        "flow" => &["flow", "reads", "returns", "writes", "defer"],
        "join" => &["mode"],
        "end" => &[],
        other => {
            return Err(e(
                "kind",
                format!(
                    "unknown kind `{other}`; use agent, tool, human, router, switch, map, flow, join or end"
                ),
            ))
        }
    };
    if let Some((f, _)) = set.iter().find(|(f, s)| *s && !allowed.contains(f)) {
        return Err(e(f, format!("is not a field of a {} node", n.kind)));
    }
    if n.prompt.as_ref().is_some_and(|p| p.trim().is_empty()) {
        return Err(e("prompt", "must not be blank".into()));
    }
    let reads = n.reads.unwrap_or_default();
    for (i, r) in reads.iter().enumerate() {
        if !state.contains_key(r) {
            return Err(e("reads", format!("`{r}` is not declared in [state]")));
        }
        if reads[..i].contains(r) {
            return Err(e("reads", format!("`{r}` repeats")));
        }
    }
    if let Some(w) = n.writes.as_ref().filter(|w| !state.contains_key(*w)) {
        return Err(e("writes", format!("`{w}` is not declared in [state]")));
    }
    let need = |v: Option<String>, f: &str| {
        v.ok_or_else(|| e(f, format!("a {} node needs `{f}`", n.kind)))
    };
    let declared = |key: &str, f: &str| -> Result<&Value, Problem> {
        state
            .get(key)
            .ok_or_else(|| e(f, format!("`{key}` is not declared in [state]")))
    };
    let defer = n.defer.unwrap_or(false);
    let kind = match n.kind.as_str() {
        "agent" => {
            let persona = need(n.persona, "persona")?;
            if !valid_name(&persona) {
                return Err(e(
                    "persona",
                    format!("`{persona}` must be 1 to 64 of [a-z0-9-], not starting with `-`"),
                ));
            }
            if n.prompt.is_none() && reads.is_empty() {
                return Err(e("prompt", "an agent node needs a prompt or reads".into()));
            }
            let goto = n
                .goto
                .map(|g| list_field(g, "goto"))
                .transpose()?
                .unwrap_or_default();
            let delegates = n
                .delegates
                .map(|d| list_field(d, "delegates"))
                .transpose()?
                .unwrap_or_default();
            if let Some(d) = delegates.iter().find(|d| !valid_name(d)) {
                return Err(e(
                    "delegates",
                    format!("`{d}` must be 1 to 64 of [a-z0-9-], not starting with `-`"),
                ));
            }
            if n.model.as_ref().is_some_and(|m| m.trim().is_empty()) {
                return Err(e("model", "must not be blank".into()));
            }
            FlowKind::Agent {
                persona,
                prompt: n.prompt,
                goto,
                delegates,
                model: n.model,
                defer,
            }
        }
        "tool" => {
            let tool = need(n.tool, "tool")?;
            if !valid_tool_name(&tool) {
                return Err(e("tool", format!("`{tool}` is not a valid tool name")));
            }
            FlowKind::Tool { tool, defer }
        }
        "human" => {
            let ask = match need(n.ask, "ask")?.as_str() {
                "approval" => Ask::Approval,
                "question" => Ask::Question,
                other => {
                    return Err(e(
                        "ask",
                        format!("`{other}` must be `approval` or `question`"),
                    ))
                }
            };
            let w = n
                .writes
                .as_ref()
                .ok_or_else(|| e("writes", "a human node needs `writes`".into()))?;
            if ask == Ask::Approval && state[w] != json!({"type": "boolean"}) {
                return Err(e(
                    "writes",
                    format!("an approval writes a bool; `{w}` is not"),
                ));
            }
            FlowKind::Human {
                ask,
                prompt: n.prompt,
            }
        }
        "router" => {
            let condition =
                parse_call(&need(n.condition, "condition")?).map_err(|m| e("condition", m))?;
            if condition.name == "flag" {
                let bool_key = condition
                    .args
                    .first()
                    .and_then(Value::as_str)
                    .filter(|k| state.get(*k) == Some(&json!({"type": "boolean"})));
                if condition.args.len() != 1 || bool_key.is_none() {
                    return Err(e(
                        "condition",
                        "flag takes one bool key declared in [state]".into(),
                    ));
                }
            }
            FlowKind::Router { condition }
        }
        "switch" => {
            let on = need(n.on, "on")?;
            if !has_type(declared(&on, "on")?, &["string", "boolean"]) {
                return Err(e(
                    "on",
                    format!("a switch routes on a string, enum or bool key; `{on}` is not"),
                ));
            }
            FlowKind::Switch { on }
        }
        "map" => {
            let over = need(n.over, "over")?;
            if !has_type(declared(&over, "over")?, &["array"]) {
                return Err(e(
                    "over",
                    format!("a map runs over a list key; `{over}` is not"),
                ));
            }
            let item = need(n.item, "item")?;
            declared(&item, "item")?;
            let body = need(n.body, "body")?;
            if n.writes.is_none() {
                return Err(e("writes", "a map node needs `writes`".into()));
            }
            FlowKind::Map {
                over,
                item,
                body,
                max_items: ranged(n.max_items, MAX_ITEMS_RANGE, DEFAULT_MAX_ITEMS, "max_items")?,
                concurrency: ranged(
                    n.concurrency,
                    CONCURRENCY_RANGE,
                    DEFAULT_CONCURRENCY,
                    "concurrency",
                )?,
            }
        }
        "flow" => {
            let flow = need(n.flow, "flow")?;
            if !valid_name(&flow) {
                return Err(e(
                    "flow",
                    format!("`{flow}` must be 1 to 64 of [a-z0-9-], not starting with `-`"),
                ));
            }
            let returns = need(n.returns, "returns")?;
            if !is_ident(&returns) {
                return Err(e(
                    "returns",
                    format!("`{returns}` must match [a-z_][a-z0-9_]*"),
                ));
            }
            if n.writes.is_none() {
                return Err(e("writes", "a flow node needs `writes`".into()));
            }
            FlowKind::Flow {
                flow,
                returns,
                defer,
            }
        }
        "join" => {
            let mode = match n.mode.as_deref() {
                None | Some("all") => JoinMode::All,
                Some("any") => JoinMode::Any,
                Some(other) => return Err(e("mode", format!("`{other}` must be `all` or `any`"))),
            };
            FlowKind::Join { mode }
        }
        "end" => FlowKind::End,
        _ => unreachable!("kind checked above"),
    };
    Ok(Spec {
        id: n.id,
        kind,
        reads,
        writes: n.writes,
    })
}

fn reducers(
    raw: BTreeMap<String, String>,
    state: &BTreeMap<String, Value>,
) -> Result<BTreeMap<String, Reducer>, Problem> {
    let mut out = BTreeMap::new();
    for (key, name) in raw {
        let field = format!("reducers.{key}");
        let e = |m: String| (field.clone(), m);
        let schema = state
            .get(&key)
            .ok_or_else(|| e(format!("`{key}` is not declared in [state]")))?;
        let (reducer, needs): (Reducer, &[&str]) = match name.as_str() {
            "append" => (Reducer::Append, &["array"]),
            "extend" => (Reducer::Extend, &["array"]),
            "merge" => (Reducer::Merge, &["object"]),
            "sum" => (Reducer::Sum, &["number", "integer"]),
            "min" => (Reducer::Min, &["number", "integer"]),
            "max" => (Reducer::Max, &["number", "integer"]),
            "last" => (Reducer::Last, &[]),
            other => {
                return Err(e(format!(
                    "unknown reducer `{other}`; use append, extend, merge, sum, min, max or last"
                )))
            }
        };
        if !needs.is_empty() && !has_type(schema, needs) {
            let what = match reducer {
                Reducer::Append | Reducer::Extend => "a list",
                Reducer::Merge => "an object",
                _ => "a number",
            };
            return Err(e(format!("`{name}` needs {what} key; `{key}` is not")));
        }
        out.insert(key, reducer);
    }
    Ok(out)
}

fn untracked(raw: Vec<String>, state: &BTreeMap<String, Value>) -> Result<Vec<String>, Problem> {
    let e = |m: String| ("untracked".to_string(), m);
    if raw.is_empty() {
        return Ok(raw);
    }
    let raw = list_field(raw, "untracked")?;
    for key in &raw {
        let schema = state
            .get(key)
            .ok_or_else(|| e(format!("`{key}` is not declared in [state]")))?;
        if !has_type(schema, &["null"]) {
            return Err(e(format!("`{key}` must be nullable, such as `string?`")));
        }
    }
    Ok(raw)
}

fn budget(raw: Option<BudgetFile>) -> Result<BudgetSplit, Problem> {
    match raw.and_then(|b| b.split).as_deref() {
        None | Some("even") => Ok(BudgetSplit::Even),
        Some("shared") => Ok(BudgetSplit::Shared),
        Some(other) => Err((
            "budget.split".into(),
            format!("`{other}` must be `even` or `shared`"),
        )),
    }
}

fn outline_node(s: &Spec) -> Node {
    let kind = match &s.kind {
        FlowKind::Agent {
            persona, prompt, ..
        } => NodeKind::Agent {
            persona: persona.clone(),
            prompt: prompt.clone(),
        },
        FlowKind::Tool { tool, .. } => NodeKind::Tool { tool: tool.clone() },
        FlowKind::Human { ask, prompt } => NodeKind::Human {
            ask: *ask,
            prompt: prompt.clone(),
        },
        FlowKind::Router { condition } => NodeKind::Router {
            condition: condition.clone(),
        },
        FlowKind::Switch { .. } => NodeKind::Tool {
            tool: "flow-switch".into(),
        },
        FlowKind::Map { .. } => NodeKind::Tool {
            tool: "flow-map".into(),
        },
        FlowKind::Flow { .. } => NodeKind::Tool {
            tool: "flow-flow".into(),
        },
        FlowKind::Join { .. } => NodeKind::Join,
        FlowKind::End => NodeKind::End,
    };
    Node {
        id: s.id.clone(),
        kind,
        reads: s.reads.clone(),
        writes: s.writes.clone(),
    }
}

fn outline_edges(specs: &[Spec], mut edges: Vec<Edge>) -> Vec<Edge> {
    for s in specs {
        if let FlowKind::Agent { goto, .. } = &s.kind {
            edges.extend(goto.iter().map(|t| Edge {
                from: s.id.clone(),
                to: t.clone(),
                when: Some(t.clone()),
            }));
        }
    }
    edges
}

struct Graph<'a> {
    specs: &'a [Spec],
    edges: &'a [Edge],
    start: &'a str,
    outs: HashMap<&'a str, Vec<&'a str>>,
    ins: HashMap<&'a str, usize>,
    bodies: BTreeMap<&'a str, &'a str>,
    nested_seen: Cell<bool>,
}

struct FanOut<'a> {
    join: &'a str,
    writes: BTreeSet<&'a str>,
    depth: usize,
}

impl<'a> Graph<'a> {
    fn new(specs: &'a [Spec], edges: &'a [Edge], start: &'a str) -> Graph<'a> {
        let mut outs: HashMap<&str, Vec<&str>> = HashMap::new();
        let mut ins: HashMap<&str, usize> = HashMap::new();
        for e in edges {
            outs.entry(e.from.as_str()).or_default().push(e.to.as_str());
            *ins.entry(e.to.as_str()).or_default() += 1;
        }
        let mut bodies = BTreeMap::new();
        for s in specs {
            match &s.kind {
                FlowKind::Agent { goto, .. } => {
                    for t in goto {
                        outs.entry(s.id.as_str()).or_default().push(t.as_str());
                        *ins.entry(t.as_str()).or_default() += 1;
                    }
                }
                FlowKind::Map { body, .. } => {
                    bodies.insert(body.as_str(), s.id.as_str());
                }
                _ => {}
            }
        }
        Graph {
            specs,
            edges,
            start,
            outs,
            ins,
            bodies,
            nested_seen: Cell::new(false),
        }
    }

    fn spec(&self, id: &str) -> Option<&'a Spec> {
        self.specs.iter().find(|s| s.id == id)
    }

    fn outs(&self, id: &str) -> &[&'a str] {
        self.outs.get(id).map(Vec::as_slice).unwrap_or(&[])
    }

    fn in_degree(&self, id: &str) -> usize {
        self.ins.get(id).copied().unwrap_or(0)
    }

    fn is_choice(&self, id: &str) -> bool {
        self.spec(id).is_some_and(|s| match &s.kind {
            FlowKind::Router { .. } | FlowKind::Switch { .. } => true,
            FlowKind::Agent { goto, .. } => !goto.is_empty(),
            _ => false,
        })
    }

    fn is_fan_out(&self, id: &str) -> bool {
        !self.is_choice(id) && self.outs(id).len() >= 2
    }

    fn check(
        &self,
        state: &BTreeMap<String, Value>,
        reducers: &BTreeMap<String, Reducer>,
    ) -> Result<bool, Problem> {
        let nf = |id: &str, m: String| (format!("nodes.{id}"), m);
        if self.spec(self.start).is_none() {
            return Err(("start".into(), format!("unknown node `{}`", self.start)));
        }
        if self.bodies.contains_key(self.start) {
            return Err(("start".into(), format!("`{}` is a map body", self.start)));
        }
        self.check_edges(state)?;
        self.check_targets(state, reducers)?;
        for s in self.specs {
            let out = self.outs(&s.id).len();
            let body = self.bodies.contains_key(s.id.as_str());
            match s.kind {
                FlowKind::End if out > 0 => {
                    return Err(nf(&s.id, "an end node has no out-edges".into()))
                }
                FlowKind::End => {}
                _ if body => {}
                _ if out == 0 => return Err(nf(&s.id, "has no out-edges".into())),
                _ => {}
            }
        }
        if !self.specs.iter().any(|s| s.kind == FlowKind::End) {
            return Err(("nodes".into(), "a workflow needs an end node".into()));
        }
        let reached = self.reachable();
        if let Some(s) = self.specs.iter().find(|s| !reached.contains(s.id.as_str())) {
            return Err(nf(&s.id, format!("is unreachable from `{}`", self.start)));
        }
        if let Some(id) = self.choice_free_cycle() {
            return Err(nf(
                &id,
                "is on a cycle without a router, switch or goto node".into(),
            ));
        }
        self.check_fan_outs(reducers)
    }

    fn check_edges(&self, state: &BTreeMap<String, Value>) -> Result<(), Problem> {
        let mut seen = BTreeSet::new();
        let mut labels: BTreeMap<&str, BTreeSet<&str>> = BTreeMap::new();
        for (i, e) in self.edges.iter().enumerate() {
            let field = |f: &str| format!("edges[{i}].{f}");
            let Some(from) = self.spec(&e.from) else {
                return Err((field("from"), format!("unknown node `{}`", e.from)));
            };
            if self.spec(&e.to).is_none() {
                return Err((field("to"), format!("unknown node `{}`", e.to)));
            }
            if !seen.insert((e.from.as_str(), e.to.as_str())) {
                return Err((
                    format!("edges[{i}]"),
                    format!("duplicate edge `{}` to `{}`", e.from, e.to),
                ));
            }
            if self.bodies.contains_key(e.from.as_str()) || self.bodies.contains_key(e.to.as_str())
            {
                return Err((
                    format!("edges[{i}]"),
                    "a map body has no edges in or out".into(),
                ));
            }
            if matches!(&from.kind, FlowKind::Agent { goto, .. } if !goto.is_empty()) {
                return Err((
                    field("from"),
                    format!("`{}` has goto targets, so no [[edges]] leave it", e.from),
                ));
            }
            let labelled = matches!(from.kind, FlowKind::Router { .. } | FlowKind::Switch { .. });
            match (labelled, &e.when) {
                (true, None) => {
                    return Err((field("when"), "a router or switch edge needs `when`".into()))
                }
                (true, Some(w)) => {
                    if w.trim().is_empty() {
                        return Err((field("when"), "must not be blank".into()));
                    }
                    if !labels
                        .entry(e.from.as_str())
                        .or_default()
                        .insert(w.as_str())
                    {
                        return Err((field("when"), format!("label `{w}` repeats")));
                    }
                }
                (false, Some(_)) => {
                    return Err((
                        field("when"),
                        "only router and switch edges have `when`".into(),
                    ))
                }
                (false, None) => {}
            }
        }
        for s in self.specs {
            let FlowKind::Switch { on } = &s.kind else {
                continue;
            };
            let got = labels.get(s.id.as_str()).cloned().unwrap_or_default();
            let schema = &state[on];
            let values: Option<BTreeSet<String>> = if let Some(e) = schema["enum"].as_array() {
                Some(
                    e.iter()
                        .filter_map(Value::as_str)
                        .map(str::to_string)
                        .collect(),
                )
            } else if has_type(schema, &["boolean"]) {
                Some(["false".to_string(), "true".to_string()].into())
            } else {
                None
            };
            let Some(values) = values else {
                continue;
            };
            if let Some(l) = got.iter().find(|l| !values.contains(**l)) {
                return Err((
                    format!("nodes.{}", s.id),
                    format!("label `{l}` is not a value of `{on}`"),
                ));
            }
            if let Some(v) = values.iter().find(|v| !got.contains(v.as_str())) {
                return Err((
                    format!("nodes.{}", s.id),
                    format!("switch edges do not cover value `{v}` of `{on}`"),
                ));
            }
        }
        Ok(())
    }

    fn check_targets(
        &self,
        state: &BTreeMap<String, Value>,
        reducers: &BTreeMap<String, Reducer>,
    ) -> Result<(), Problem> {
        let nf = |id: &str, f: &str, m: String| (format!("nodes.{id}.{f}"), m);
        let mut used: BTreeSet<&str> = BTreeSet::new();
        for s in self.specs {
            match &s.kind {
                FlowKind::Agent { goto, .. } => {
                    for t in goto {
                        if self.spec(t).is_none() {
                            return Err(nf(&s.id, "goto", format!("unknown node `{t}`")));
                        }
                        if self.bodies.contains_key(t.as_str()) {
                            return Err(nf(&s.id, "goto", format!("`{t}` is a map body")));
                        }
                    }
                }
                FlowKind::Map { item, body, .. } => {
                    let Some(b) = self.spec(body) else {
                        return Err(nf(&s.id, "body", format!("unknown node `{body}`")));
                    };
                    if b.id == s.id || !used.insert(body.as_str()) {
                        return Err(nf(&s.id, "body", format!("`{body}` is already a map body")));
                    }
                    if !matches!(
                        b.kind,
                        FlowKind::Agent { .. } | FlowKind::Tool { .. } | FlowKind::Flow { .. }
                    ) {
                        return Err(nf(
                            &s.id,
                            "body",
                            format!("`{body}` must be an agent, tool or flow node"),
                        ));
                    }
                    if self.is_choice(body) {
                        return Err(nf(&s.id, "body", format!("`{body}` has goto targets")));
                    }
                    if !b.reads.contains(item) {
                        return Err(nf(
                            &s.id,
                            "body",
                            format!("`{body}` must read the item key `{item}`"),
                        ));
                    }
                    if b.writes.is_none() {
                        return Err(nf(&s.id, "body", format!("`{body}` must write one value")));
                    }
                    let w = s.writes.as_deref().expect("map nodes write");
                    if !reducers.contains_key(w) && !has_type(&state[w], &["array"]) {
                        return Err(nf(
                            &s.id,
                            "writes",
                            format!("a map writes a list of results; `{w}` is not a list and has no reducer"),
                        ));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn reachable(&self) -> BTreeSet<&'a str> {
        let start = self.spec(self.start).expect("start checked").id.as_str();
        let mut seen = BTreeSet::from([start]);
        let mut queue = VecDeque::from([start]);
        while let Some(id) = queue.pop_front() {
            let spec = self.spec(id).expect("reached nodes exist");
            let body = match &spec.kind {
                FlowKind::Map { body, .. } => Some(body.as_str()),
                _ => None,
            };
            for to in self.outs(id).iter().copied().chain(body) {
                let to = self.spec(to).expect("targets checked").id.as_str();
                if seen.insert(to) {
                    queue.push_back(to);
                }
            }
        }
        seen
    }

    fn choice_free_cycle(&self) -> Option<String> {
        let plain = |id: &str| self.spec(id).is_some() && !self.is_choice(id);
        let mut color: HashMap<&str, u8> = HashMap::new();
        for s in self.specs.iter().filter(|s| plain(&s.id)) {
            if color.contains_key(s.id.as_str()) {
                continue;
            }
            let mut stack: Vec<(&str, usize)> = vec![(s.id.as_str(), 0)];
            color.insert(s.id.as_str(), 1);
            while let Some((id, i)) = stack.pop() {
                let out: Vec<&str> = self.outs(id).iter().copied().filter(|t| plain(t)).collect();
                let Some(&to) = out.get(i) else {
                    color.insert(id, 2);
                    continue;
                };
                stack.push((id, i + 1));
                match color.get(to) {
                    Some(1) => return Some(to.to_string()),
                    Some(_) => {}
                    None => {
                        color.insert(to, 1);
                        stack.push((to, 0));
                    }
                }
            }
        }
        None
    }

    fn check_fan_outs(&self, reducers: &BTreeMap<String, Reducer>) -> Result<bool, Problem> {
        let mut closes: HashMap<&str, usize> = HashMap::new();
        let mut nested = false;
        for s in self.specs {
            if !self.is_fan_out(&s.id) {
                continue;
            }
            let fan = self.fan_out(&s.id, 1, reducers)?;
            nested |= fan.depth > 1;
            *closes.entry(fan.join).or_default() += 1;
        }
        for s in self
            .specs
            .iter()
            .filter(|s| matches!(s.kind, FlowKind::Join { .. }))
        {
            if closes.get(s.id.as_str()).copied().unwrap_or(0) != 1 {
                return Err((
                    format!("nodes.{}", s.id),
                    "a join closes exactly one fan-out".into(),
                ));
            }
        }
        Ok(nested)
    }

    fn fan_out(
        &self,
        f: &'a str,
        depth: usize,
        reducers: &BTreeMap<String, Reducer>,
    ) -> Result<FanOut<'a>, Problem> {
        assert!(
            depth <= MAX_FAN_OUT_DEPTH,
            "deeper fan-outs are refused before recursing"
        );
        let nf = |id: &str, m: String| (format!("nodes.{id}"), m);
        let mut join: Option<&'a str> = None;
        let mut tails: Vec<&'a str> = Vec::new();
        let mut owner: BTreeMap<&'a str, usize> = BTreeMap::new();
        let limit = self.specs.len() + 1;
        let mut deepest = depth;
        for (b, &first) in self.outs(f).iter().enumerate() {
            let (mut prev, mut cur) = (f, first);
            let mut writes: BTreeSet<&'a str> = BTreeSet::new();
            let mut walked = 0;
            loop {
                walked += 1;
                assert!(
                    walked <= limit,
                    "choice-free cycles are rejected, so every walk ends"
                );
                let s = self.spec(cur).expect("edge ends checked");
                if matches!(s.kind, FlowKind::Join { .. }) {
                    break;
                }
                let walkable = !self.is_choice(cur)
                    && matches!(
                        s.kind,
                        FlowKind::Agent { .. }
                            | FlowKind::Tool { .. }
                            | FlowKind::Human { .. }
                            | FlowKind::Flow { .. }
                            | FlowKind::Map { .. }
                    );
                if !walkable || self.in_degree(cur) != 1 {
                    return Err(nf(
                        f,
                        format!(
                            "fan-out branch reaches `{cur}` before a join; a branch is a chain of agent, tool, human, flow or map nodes ending at one join"
                        ),
                    ));
                }
                if let Some(w) = &s.writes {
                    writes.insert(w.as_str());
                }
                let mut next = cur;
                if self.outs(cur).len() >= 2 {
                    self.nested_seen.set(true);
                    if depth + 1 > MAX_FAN_OUT_DEPTH {
                        return Err(nf(
                            f,
                            format!("nested fan-out deeper than {MAX_FAN_OUT_DEPTH}"),
                        ));
                    }
                    let inner = self.fan_out(cur, depth + 1, reducers)?;
                    writes.extend(inner.writes);
                    deepest = deepest.max(inner.depth);
                    next = inner.join;
                }
                let outs = self.outs(next);
                if outs.len() != 1 {
                    return Err(nf(
                        f,
                        format!("fan-out branch reaches `{next}`, which must lead on by exactly one edge"),
                    ));
                }
                prev = next;
                cur = outs[0];
            }
            for w in &writes {
                if owner.insert(w, b).is_some_and(|other| other != b) && !reducers.contains_key(*w)
                {
                    return Err(nf(f, format!("two branches of the fan-out write `{w}`")));
                }
            }
            if join.is_some_and(|j| j != cur) {
                return Err(nf(f, "fan-out branches meet at different joins".into()));
            }
            join = Some(cur);
            tails.push(prev);
        }
        let j = join.expect("a fan-out has branches");
        let inputs: Vec<&str> = self
            .specs
            .iter()
            .flat_map(|s| {
                let from = s.id.as_str();
                self.outs(from)
                    .iter()
                    .filter(|t| **t == j)
                    .map(move |_| from)
            })
            .collect();
        assert_eq!(
            inputs.len(),
            self.in_degree(j),
            "join inputs count every edge, goto edges included"
        );
        if inputs.len() != tails.len() || inputs.iter().any(|i| !tails.contains(i)) {
            return Err(nf(
                j,
                format!("join has inputs from outside the fan-out at `{f}`"),
            ));
        }
        Ok(FanOut {
            join: j,
            writes: owner.into_keys().collect(),
            depth: deepest,
        })
    }
}
