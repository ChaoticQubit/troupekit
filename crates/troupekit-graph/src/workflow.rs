use crate::error::WorkflowError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use troupekit_agent::dsl::{is_ident, parse_call, Call};
use troupekit_agent::persona::{toml_error, valid_name};
use troupekit_agent::{compile_output, validate};
use troupekit_tools::valid_tool_name;

const MAX_NODES: usize = 256;
const MAX_STEPS_RANGE: std::ops::RangeInclusive<i64> = 1..=10_000;
const MAX_RETRIES_RANGE: std::ops::RangeInclusive<i64> = 0..=50;
const DEFAULT_MAX_STEPS: u32 = 200;
const DEFAULT_MAX_RETRIES: u32 = 5;

/// What a human node asks for: a yes or no approval, written as a bool,
/// or a free answer typed by the node's `writes` key.
/// Written `ask = "approval"` or `ask = "question"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ask {
    Approval,
    Question,
}

/// The work a node does. Agent nodes run a persona, tool nodes call one
/// tool, human nodes wait for an answer, routers pick one out-edge by a
/// condition, joins wait for every branch of a fan-out, end nodes finish.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum NodeKind {
    Agent {
        persona: String,
        prompt: Option<String>,
    },
    Tool {
        tool: String,
    },
    Human {
        ask: Ask,
        prompt: Option<String>,
    },
    Router {
        condition: Call,
    },
    Join,
    End,
}

/// One validated node: its id, kind, the state keys it reads and the
/// state key it writes. Routers, joins and ends read and write nothing;
/// a human node always writes its answer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Node {
    pub id: String,
    pub kind: NodeKind,
    pub reads: Vec<String>,
    pub writes: Option<String>,
}

/// One edge from node `from` to node `to`. `when` is the label a router
/// condition returns to take it; only router edges have one.
/// A node with several plain out-edges fans out into parallel branches.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Edge {
    pub from: String,
    pub to: String,
    pub when: Option<String>,
}

/// A validated workflow: name, start node, step and retry caps, the
/// compiled JSON Schema of every state key, nodes and edges. Plain data,
/// so a graph run keeps a snapshot of the workflow it started with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Workflow {
    pub name: String,
    pub start: String,
    pub max_steps: u32,
    pub max_retries: u32,
    pub state: BTreeMap<String, Value>,
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkflowFile {
    name: String,
    start: String,
    max_steps: Option<i64>,
    max_retries: Option<i64>,
    #[serde(default)]
    state: toml::Table,
    #[serde(default)]
    nodes: Vec<NodeFile>,
    #[serde(default)]
    edges: Vec<Edge>,
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
}

impl Workflow {
    /// Parses and validates workflow TOML `text` read from `file`: fields,
    /// node kinds, state keys, edges, reachability, router-free cycles and
    /// fan-out joins. Returns the first problem found.
    pub fn parse(file: &str, text: &str) -> Result<Workflow, WorkflowError> {
        let raw: WorkflowFile = toml::from_str(text).map_err(|e| {
            let p = toml_error(file, text, &e);
            WorkflowError::new(&p.file, &p.field, p.message)
        })?;
        let err = |field: &str, msg: String| WorkflowError::new(file, field, msg);
        if !valid_name(&raw.name) {
            return Err(err(
                "name",
                "must be 1 to 64 of [a-z0-9-], not starting with `-`".into(),
            ));
        }
        let max_steps = ranged(raw.max_steps, MAX_STEPS_RANGE, DEFAULT_MAX_STEPS)
            .map_err(|m| err("max_steps", m))?;
        let max_retries = ranged(raw.max_retries, MAX_RETRIES_RANGE, DEFAULT_MAX_RETRIES)
            .map_err(|m| err("max_retries", m))?;
        let state = state_schemas(&raw.state).map_err(|m| err("state", m))?;
        if raw.nodes.is_empty() || raw.nodes.len() > MAX_NODES {
            return Err(err(
                "nodes",
                format!("a workflow has 1 to {MAX_NODES} nodes"),
            ));
        }
        let mut nodes: Vec<Node> = Vec::with_capacity(raw.nodes.len());
        for (i, n) in raw.nodes.into_iter().enumerate() {
            if !is_ident(&n.id) || n.id.len() > 64 {
                return Err(err(
                    &format!("nodes[{i}].id"),
                    format!("`{}` must be 1 to 64 of [a-z_][a-z0-9_]*", n.id),
                ));
            }
            if nodes.iter().any(|m| m.id == n.id) {
                return Err(err(
                    &format!("nodes[{i}].id"),
                    format!("duplicate node `{}`", n.id),
                ));
            }
            let id = n.id.clone();
            nodes.push(node(n, &state).map_err(|(f, m)| err(&format!("nodes.{id}.{f}"), m))?);
        }
        let wf = Workflow {
            name: raw.name,
            start: raw.start,
            max_steps,
            max_retries,
            state,
            nodes,
            edges: raw.edges,
        };
        wf.check_graph().map_err(|(f, m)| err(&f, m))?;
        Ok(wf)
    }

    /// Returns the node with id `id`.
    /// `None` when the workflow has none.
    /// Ids are unique, so at most one matches.
    pub fn node(&self, id: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }

    /// Returns the out-edges of node `id` in file order.
    /// Fan-out branches and router choices follow this order.
    /// Empty for end nodes and unknown ids.
    pub fn next(&self, id: &str) -> Vec<&Edge> {
        self.edges.iter().filter(|e| e.from == id).collect()
    }

    /// Returns how many edges lead into node `id`.
    /// A join waits for this many branches.
    /// Zero for unknown ids.
    pub fn in_degree(&self, id: &str) -> usize {
        self.edges.iter().filter(|e| e.to == id).count()
    }

    /// Checks `value` against the declared type of state key `key`.
    /// Returns the problems as `"/<key>...: <message>"` lines, or an empty
    /// list when it fits; an undeclared key is one problem.
    pub fn check_value(&self, key: &str, value: &Value) -> Vec<String> {
        let Some(schema) = self.state.get(key) else {
            return vec![format!("`{key}` is not declared in [state]")];
        };
        let wrapped = serde_json::json!({
            "type": "object",
            "properties": {key: schema},
            "required": [key],
            "additionalProperties": false
        });
        validate(&wrapped, &serde_json::json!({key: value}))
    }

    fn check_graph(&self) -> Result<(), (String, String)> {
        let nf = |id: &str, m: String| (format!("nodes.{id}"), m);
        if self.node(&self.start).is_none() {
            return Err(("start".into(), format!("unknown node `{}`", self.start)));
        }
        let mut seen = BTreeSet::new();
        let mut labels: HashMap<&str, BTreeSet<&str>> = HashMap::new();
        for (i, e) in self.edges.iter().enumerate() {
            let field = |f: &str| format!("edges[{i}].{f}");
            let Some(from) = self.node(&e.from) else {
                return Err((field("from"), format!("unknown node `{}`", e.from)));
            };
            if self.node(&e.to).is_none() {
                return Err((field("to"), format!("unknown node `{}`", e.to)));
            }
            if !seen.insert((e.from.as_str(), e.to.as_str())) {
                return Err((
                    format!("edges[{i}]"),
                    format!("duplicate edge `{}` to `{}`", e.from, e.to),
                ));
            }
            match (&from.kind, &e.when) {
                (NodeKind::Router { .. }, None) => {
                    return Err((field("when"), "a router edge needs `when`".into()))
                }
                (NodeKind::Router { .. }, Some(w)) => {
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
                (_, Some(_)) => {
                    return Err((field("when"), "only router edges have `when`".into()))
                }
                (_, None) => {}
            }
        }
        for n in &self.nodes {
            let out = self.next(&n.id).len();
            match n.kind {
                NodeKind::End if out > 0 => {
                    return Err(nf(&n.id, "an end node has no out-edges".into()))
                }
                NodeKind::End => {}
                _ if out == 0 => return Err(nf(&n.id, "has no out-edges".into())),
                _ => {}
            }
        }
        if !self.nodes.iter().any(|n| n.kind == NodeKind::End) {
            return Err(("nodes".into(), "a workflow needs an end node".into()));
        }
        let reached = self.reachable();
        if let Some(n) = self.nodes.iter().find(|n| !reached.contains(n.id.as_str())) {
            return Err(nf(&n.id, format!("is unreachable from `{}`", self.start)));
        }
        if let Some(id) = self.router_free_cycle() {
            return Err(nf(&id, "is on a cycle without a router".into()));
        }
        self.check_fan_outs()
    }

    fn reachable(&self) -> BTreeSet<&str> {
        let mut seen = BTreeSet::from([self.start.as_str()]);
        let mut queue = VecDeque::from([self.start.as_str()]);
        while let Some(id) = queue.pop_front() {
            for e in self.next(id) {
                if seen.insert(e.to.as_str()) {
                    queue.push_back(e.to.as_str());
                }
            }
        }
        seen
    }

    fn router_free_cycle(&self) -> Option<String> {
        let plain = |id: &str| {
            self.node(id)
                .is_some_and(|n| !matches!(n.kind, NodeKind::Router { .. }))
        };
        let mut color: HashMap<&str, u8> = HashMap::new();
        for n in self.nodes.iter().filter(|n| plain(&n.id)) {
            if color.contains_key(n.id.as_str()) {
                continue;
            }
            let mut stack: Vec<(&str, usize)> = vec![(n.id.as_str(), 0)];
            color.insert(n.id.as_str(), 1);
            while let Some((id, i)) = stack.pop() {
                let out: Vec<&str> = self
                    .next(id)
                    .into_iter()
                    .map(|e| e.to.as_str())
                    .filter(|t| plain(t))
                    .collect();
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

    fn check_fan_outs(&self) -> Result<(), (String, String)> {
        let nf = |id: &str, m: String| (format!("nodes.{id}"), m);
        let mut closes: HashMap<&str, usize> = HashMap::new();
        for f in &self.nodes {
            let out = self.next(&f.id);
            if matches!(f.kind, NodeKind::Router { .. }) || out.len() < 2 {
                continue;
            }
            let mut join: Option<&str> = None;
            let mut tails: Vec<&str> = Vec::with_capacity(out.len());
            let mut writes: BTreeMap<&str, usize> = BTreeMap::new();
            for (b, e) in out.iter().enumerate() {
                let (mut prev, mut cur) = (f.id.as_str(), e.to.as_str());
                for _ in 0..=self.nodes.len() {
                    let n = self.node(cur).expect("edge ends checked");
                    let walkable = matches!(
                        n.kind,
                        NodeKind::Agent { .. } | NodeKind::Tool { .. } | NodeKind::Human { .. }
                    );
                    if n.kind == NodeKind::Join {
                        break;
                    }
                    if !walkable || self.next(cur).len() != 1 || self.in_degree(cur) != 1 {
                        return Err(nf(
                            &f.id,
                            format!(
                                "fan-out branch reaches `{cur}` before a join; a branch is a chain of agent, tool or human nodes ending at one join"
                            ),
                        ));
                    }
                    if let Some(w) = &n.writes {
                        if writes.insert(w.as_str(), b).is_some_and(|other| other != b) {
                            return Err(nf(
                                &f.id,
                                format!("two branches of the fan-out write `{w}`"),
                            ));
                        }
                    }
                    prev = cur;
                    cur = self.next(cur)[0].to.as_str();
                }
                assert_eq!(
                    self.node(cur).map(|n| &n.kind),
                    Some(&NodeKind::Join),
                    "router-free cycles are rejected, so every walk ends"
                );
                if join.is_some_and(|j| j != cur) {
                    return Err(nf(&f.id, "fan-out branches meet at different joins".into()));
                }
                join = Some(cur);
                tails.push(prev);
            }
            let j = join.expect("a fan-out has branches");
            let inputs: Vec<&str> = self
                .edges
                .iter()
                .filter(|e| e.to == j)
                .map(|e| e.from.as_str())
                .collect();
            if inputs.len() != tails.len() || inputs.iter().any(|i| !tails.contains(i)) {
                return Err(nf(
                    j,
                    format!("join has inputs from outside the fan-out at `{}`", f.id),
                ));
            }
            *closes.entry(j).or_default() += 1;
        }
        for n in self.nodes.iter().filter(|n| n.kind == NodeKind::Join) {
            if closes.get(n.id.as_str()).copied().unwrap_or(0) != 1 {
                return Err(nf(&n.id, "a join closes exactly one fan-out".into()));
            }
        }
        Ok(())
    }
}

fn ranged(
    v: Option<i64>,
    range: std::ops::RangeInclusive<i64>,
    default: u32,
) -> Result<u32, String> {
    match v {
        None => Ok(default),
        Some(n) if range.contains(&n) => Ok(u32::try_from(n).expect("range fits in u32")),
        Some(n) => Err(format!(
            "{n} must be between {} and {}",
            range.start(),
            range.end()
        )),
    }
}

fn state_schemas(table: &toml::Table) -> Result<BTreeMap<String, Value>, String> {
    if table.is_empty() {
        return Ok(BTreeMap::new());
    }
    let schema = compile_output(table)?;
    let props = schema["properties"]
        .as_object()
        .expect("compiled output is an object schema");
    assert_eq!(props.len(), table.len(), "one schema per state key");
    Ok(props.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
}

fn node(n: NodeFile, state: &BTreeMap<String, Value>) -> Result<Node, (String, String)> {
    let e = |f: &str, m: String| (f.to_string(), m);
    let set = [
        ("persona", n.persona.is_some()),
        ("prompt", n.prompt.is_some()),
        ("tool", n.tool.is_some()),
        ("ask", n.ask.is_some()),
        ("condition", n.condition.is_some()),
        ("reads", n.reads.is_some()),
        ("writes", n.writes.is_some()),
    ];
    let allowed: &[&str] = match n.kind.as_str() {
        "agent" => &["persona", "prompt", "reads", "writes"],
        "tool" => &["tool", "reads", "writes"],
        "human" => &["ask", "prompt", "reads", "writes"],
        "router" => &["condition"],
        "join" | "end" => &[],
        other => {
            return Err(e(
                "kind",
                format!("unknown kind `{other}`; use agent, tool, human, router, join or end"),
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
            NodeKind::Agent {
                persona,
                prompt: n.prompt,
            }
        }
        "tool" => {
            let tool = need(n.tool, "tool")?;
            if !valid_tool_name(&tool) {
                return Err(e("tool", format!("`{tool}` is not a valid tool name")));
            }
            NodeKind::Tool { tool }
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
            if ask == Ask::Approval && state[w] != serde_json::json!({"type": "boolean"}) {
                return Err(e(
                    "writes",
                    format!("an approval writes a bool; `{w}` is not"),
                ));
            }
            NodeKind::Human {
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
                    .filter(|k| state.get(*k) == Some(&serde_json::json!({"type": "boolean"})));
                if condition.args.len() != 1 || bool_key.is_none() {
                    return Err(e(
                        "condition",
                        "flag takes one bool key declared in [state]".into(),
                    ));
                }
            }
            NodeKind::Router { condition }
        }
        "join" => NodeKind::Join,
        "end" => NodeKind::End,
        _ => unreachable!("kind checked above"),
    };
    Ok(Node {
        id: n.id,
        kind,
        reads,
        writes: n.writes,
    })
}
