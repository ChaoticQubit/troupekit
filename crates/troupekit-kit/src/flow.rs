use crate::error::KitError;
use crate::graph::{step_name, Graph, Node, END, INPUT, MAX_NODES};
use std::collections::{BTreeMap, BTreeSet};
use toml::Value as Toml;
use troupekit_graph::Flow;

/// One flow part added by a builder method: a fan-out into parallel
/// branches, a join that merges them, or a human node (an approval when
/// `routes` is empty, else a question). Parts turn the chain into a flow.
#[derive(Clone, Debug)]
pub(crate) enum Part {
    Human {
        id: String,
        prompt: String,
        routes: Vec<(String, String)>,
    },
    Parallel {
        from: String,
        to: Vec<String>,
    },
    Join {
        id: String,
        from: Vec<String>,
        any: bool,
    },
}

/// When the node after `Graph::after` runs: `All` once every listed
/// branch arrives (`join`), `Any` as soon as the first one does
/// (`join_any`). The default is `All`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum Trigger {
    #[default]
    All,
    Any,
}

/// The checked shape of a graph with parts: every node, join and human
/// node in topological order, the keys each one reads, the edges (each
/// last node's edge goes to `end`), the keys an any-join may leave unset
/// and the nodes whose output is the answer. Built by `topology`.
pub(crate) struct Topology {
    pub(crate) order: Vec<String>,
    pub(crate) reads: BTreeMap<String, Vec<String>>,
    pub(crate) human_reads: BTreeMap<String, Vec<String>>,
    pub(crate) nullable: BTreeSet<String>,
    pub(crate) edges: Vec<(String, String)>,
    pub(crate) lasts: Vec<String>,
}

const GATE: &str = "__gate";
const ROUTE: &str = "__route";
const MAX_ID: usize = 64;

impl Graph {
    /// Fans node `from` out into the parallel branches `to`, which run in
    /// the same step. Each branch is a chain of nodes that ends at one
    /// `join`. Plain `.edge` calls never fan out. Returns the builder.
    pub fn parallel<I, S>(mut self, from: impl Into<String>, to: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let to = to.into_iter().map(Into::into).collect();
        self.parts.push(Part::Parallel {
            from: from.into(),
            to,
        });
        self
    }

    /// Adds join `id`, which waits for every branch in `from` and then
    /// continues on its one `.edge`. The node after a join reads each
    /// branch's output by its node id, in `from` order. Returns the builder.
    pub fn join<I, S>(mut self, id: impl Into<String>, from: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let from = from.into_iter().map(Into::into).collect();
        self.parts.push(Part::Join {
            id: id.into(),
            from,
            any: false,
        });
        self
    }

    /// Adds any-join `id`: the first branch in `from` to arrive continues
    /// on its one `.edge` and the others are dropped (on a tie, the branch
    /// listed first in `parallel` wins). The node after it reads each branch
    /// by node id; a dropped branch reads null, so steps take `Option`.
    pub fn join_any<I, S>(mut self, id: impl Into<String>, from: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let from = from.into_iter().map(Into::into).collect();
        self.parts.push(Part::Join {
            id: id.into(),
            from,
            any: true,
        });
        self
    }

    /// Adds join `id` over the branches `from` that fires on `trigger`:
    /// `Trigger::All` is `join`, `Trigger::Any` is `join_any`.
    /// Returns the builder.
    pub fn after<I, S>(self, id: impl Into<String>, from: I, trigger: Trigger) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        match trigger {
            Trigger::All => self.join(id, from),
            Trigger::Any => self.join_any(id, from),
        }
    }

    /// Adds approval node `id`: the run pauses with `prompt`, and
    /// `BuiltGraph::answer(run, id, true)` continues on its one `.edge`;
    /// `false` ends the run, which then fails as declined. The node after
    /// it reads what fed the approval. Returns the builder.
    pub fn approval(mut self, id: impl Into<String>, prompt: impl Into<String>) -> Self {
        self.parts.push(Part::Human {
            id: id.into(),
            prompt: prompt.into(),
            routes: Vec::new(),
        });
        self
    }

    /// Adds question node `id`: the run pauses with `prompt`, and
    /// `BuiltGraph::answer(run, id, label)` continues at the target of
    /// that label in `routes`. A question has no `.edge`; each target reads
    /// what fed the question. Returns the builder.
    pub fn question<I, L, T>(
        mut self,
        id: impl Into<String>,
        prompt: impl Into<String>,
        routes: I,
    ) -> Self
    where
        I: IntoIterator<Item = (L, T)>,
        L: Into<String>,
        T: Into<String>,
    {
        let routes = routes
            .into_iter()
            .map(|(l, t)| (l.into(), t.into()))
            .collect();
        self.parts.push(Part::Human {
            id: id.into(),
            prompt: prompt.into(),
            routes,
        });
        self
    }

    /// Checks the nodes, joins, human nodes and edges of a graph with parts
    /// and orders them. A node fans out only through `parallel`, merges only
    /// at a `join` and branches only at a `question`; one node starts it.
    /// Returns the shape or the first problem found.
    pub(crate) fn topology(&self) -> Result<Topology, String> {
        let nodes = self.ids()?;
        let mut joins: BTreeMap<&str, &[String]> = BTreeMap::new();
        let mut humans: BTreeMap<&str, &[(String, String)]> = BTreeMap::new();
        let mut edges: Vec<(String, String)> = self.edges.clone();
        let mut sources: BTreeSet<&str> = BTreeSet::new();
        let taken = |id: &str,
                     joins: &BTreeMap<&str, &[String]>,
                     humans: &BTreeMap<&str, &[(String, String)]>| {
            id == INPUT
                || id == END
                || nodes.contains(&id)
                || joins.contains_key(id)
                || humans.contains_key(id)
                || id.ends_with(GATE)
                || id.ends_with(ROUTE)
        };
        for part in &self.parts {
            match part {
                Part::Parallel { from, to } => {
                    if to.len() < 2 {
                        return Err(format!(
                            "parallel from `{from}` needs at least two branches"
                        ));
                    }
                    if !sources.insert(from) {
                        return Err(format!("node `{from}` fans out twice"));
                    }
                    edges.extend(to.iter().map(|t| (from.clone(), t.clone())));
                }
                Part::Join { id, from, .. } => {
                    if taken(id, &joins, &humans) {
                        return Err(format!("join id `{id}` is taken"));
                    }
                    if from.len() < 2 {
                        return Err(format!("join `{id}` needs at least two branches"));
                    }
                    joins.insert(id, from);
                    edges.extend(from.iter().map(|f| (f.clone(), id.clone())));
                }
                Part::Human { id, routes, .. } => {
                    if taken(id, &joins, &humans) || id.len() + GATE.len().max(ROUTE.len()) > MAX_ID
                    {
                        return Err(format!("human node id `{id}` is taken or too long"));
                    }
                    let mut labels = BTreeSet::new();
                    let mut targets = BTreeSet::new();
                    for (label, target) in routes {
                        if !labels.insert(label) || !targets.insert(target) {
                            return Err(format!(
                                "question `{id}` repeats label `{label}` or target `{target}`"
                            ));
                        }
                    }
                    if routes.len() == 1 {
                        return Err(format!("question `{id}` needs at least two routes"));
                    }
                    if !routes.is_empty() && self.edges.iter().any(|e| e.0 == *id) {
                        return Err(format!(
                            "question `{id}` routes by its answers; give it no .edge"
                        ));
                    }
                    humans.insert(id, routes);
                    edges.extend(routes.iter().map(|(_, t)| (id.clone(), t.clone())));
                }
            }
        }
        let all: Vec<&str> = nodes
            .iter()
            .copied()
            .chain(joins.keys().copied())
            .chain(humans.keys().copied())
            .collect();
        if all.len() > MAX_NODES {
            return Err(format!(
                "takes at most {MAX_NODES} nodes, joins and human nodes, got {}",
                all.len()
            ));
        }
        for (from, to) in &edges {
            if let Some(unknown) = [from, to].into_iter().find(|n| !all.contains(&n.as_str())) {
                return Err(format!("edge names unknown node `{unknown}`"));
            }
        }
        let outs = |id: &str| edges.iter().filter(|e| e.0 == id).count();
        let ins = |id: &str| edges.iter().filter(|e| e.1 == id).count();
        for id in &all {
            let explicit = self.edges.iter().filter(|e| e.0 == *id).count();
            if explicit > 1 || (explicit == 1 && sources.contains(id)) {
                return Err(format!(
                    "node `{id}` has two outgoing edges; use .parallel(from, [..]) for fan-out"
                ));
            }
            if ins(id) > 1 && !joins.contains_key(id) {
                return Err(format!(
                    "node `{id}` has two incoming edges; use .join(id, [..]) to merge branches"
                ));
            }
        }
        let starts: Vec<&str> = all.iter().copied().filter(|id| ins(id) == 0).collect();
        let start = match starts[..] {
            [] => return Err("edges form a cycle".into()),
            [one] if joins.contains_key(one) => {
                return Err(format!("join `{one}` has no branches"))
            }
            [one] => one,
            _ => {
                return Err(format!(
                    "nodes {} start the graph; one node starts it",
                    starts.join(", ")
                ))
            }
        };
        let mut order: Vec<String> = Vec::with_capacity(all.len());
        let mut left: BTreeMap<&str, usize> = all.iter().map(|id| (*id, ins(id))).collect();
        let mut ready = vec![start];
        while let Some(id) = ready.pop() {
            order.push(id.to_string());
            for (_, to) in edges.iter().filter(|e| e.0 == id) {
                let n = left.get_mut(to.as_str()).expect("edge ends are known");
                *n -= 1;
                if *n == 0 {
                    ready.push(to);
                }
            }
        }
        if order.len() != all.len() {
            return Err("edges form a cycle".into());
        }
        let sinks: Vec<&str> = all.iter().copied().filter(|id| outs(id) == 0).collect();
        if let Some(open) = sinks
            .iter()
            .find(|id| joins.contains_key(*id) || humans.contains_key(*id))
        {
            return Err(format!("`{open}` needs an .edge to the node after it"));
        }
        let questions = humans.values().any(|r| !r.is_empty());
        if sinks.len() > 1 && !questions {
            return Err(format!(
                "nodes {} end the graph; join their branches",
                sinks.join(", ")
            ));
        }
        assert!(!sinks.is_empty(), "an acyclic graph has a last node");
        let mut pass: BTreeMap<&str, Vec<String>> =
            joins.iter().map(|(j, b)| (*j, b.to_vec())).collect();
        for h in humans.keys() {
            let feeds: Vec<String> = edges
                .iter()
                .filter(|e| e.1 == *h)
                .map(|e| e.0.clone())
                .collect();
            pass.insert(
                h,
                if feeds.is_empty() {
                    vec![INPUT.to_string()]
                } else {
                    feeds
                },
            );
        }
        let reads_of = |id: &str| {
            let mut keys = Vec::new();
            for (from, _) in edges.iter().filter(|e| e.1 == id) {
                expand(from, &pass, &mut keys, 0);
            }
            if keys.is_empty() {
                keys.push(INPUT.to_string());
            }
            keys
        };
        let reads = nodes
            .iter()
            .map(|id| (id.to_string(), reads_of(id)))
            .collect();
        let human_reads = humans
            .keys()
            .map(|id| (id.to_string(), reads_of(id)))
            .collect();
        let mut nullable = BTreeSet::new();
        for part in &self.parts {
            if let Part::Join { id, any: true, .. } = part {
                let mut keys = Vec::new();
                expand(id, &pass, &mut keys, 0);
                nullable.extend(keys);
            }
        }
        let lasts: Vec<String> = sinks.iter().map(|s| s.to_string()).collect();
        edges.extend(lasts.iter().map(|l| (l.clone(), END.to_string())));
        assert_eq!(order.len(), all.len(), "every node is ordered once");
        Ok(Topology {
            order,
            reads,
            human_reads,
            nullable,
            edges,
            lasts,
        })
    }

    /// Serializes the graph with parts as flow TOML and parses it, so the
    /// engine checks fan-outs, joins, routes and state. Each node writes
    /// its own id; an approval gates on a router, a question on a switch.
    /// Returns the flow.
    pub(crate) fn flow(&self, topo: &Topology) -> Result<Flow, KitError> {
        let owner = format!("graph:{}", self.name);
        let mut state = toml::Table::new();
        state.insert(INPUT.into(), text(INPUT));
        let mut nodes = Vec::with_capacity(topo.order.len() + 1);
        let mut edges = Vec::with_capacity(topo.edges.len());
        let human = |id: &str| {
            self.parts.iter().find_map(|p| match p {
                Part::Human {
                    id: h,
                    prompt,
                    routes,
                } if h == id => Some((prompt, routes)),
                _ => None,
            })
        };
        let join = |id: &str| {
            self.parts.iter().find_map(|p| match p {
                Part::Join { id: j, any, .. } if j == id => Some(*any),
                _ => None,
            })
        };
        for id in &topo.order {
            let mut node = toml::Table::new();
            node.insert("id".into(), text(id));
            if let Some(any) = join(id) {
                node.insert("kind".into(), text("join"));
                if any {
                    node.insert("mode".into(), text("any"));
                }
            } else if let Some((prompt, routes)) = human(id) {
                node.insert("kind".into(), text("human"));
                node.insert("prompt".into(), text(prompt));
                node.insert("reads".into(), keys(&topo.human_reads[id]));
                node.insert("writes".into(), text(id));
                let mut next = toml::Table::new();
                if routes.is_empty() {
                    node.insert("ask".into(), text("approval"));
                    state.insert(id.clone(), text("bool"));
                    next.insert("id".into(), text(&format!("{id}{GATE}")));
                    next.insert("kind".into(), text("router"));
                    next.insert("condition".into(), text(&format!("flag({id})")));
                    edges.push(edge(id, &format!("{id}{GATE}"), None));
                    edges.push(edge(&format!("{id}{GATE}"), END, Some("false")));
                } else {
                    node.insert("ask".into(), text("question"));
                    let labels: Vec<&str> = routes.iter().map(|(l, _)| l.as_str()).collect();
                    state.insert(id.clone(), text(&format!("enum({})", labels.join(", "))));
                    next.insert("id".into(), text(&format!("{id}{ROUTE}")));
                    next.insert("kind".into(), text("switch"));
                    next.insert("on".into(), text(id));
                    edges.push(edge(id, &format!("{id}{ROUTE}"), None));
                }
                nodes.push(Toml::Table(node));
                nodes.push(Toml::Table(next));
                continue;
            } else {
                self.node_table(id, &mut node);
                node.insert("reads".into(), keys(&topo.reads[id]));
                node.insert("writes".into(), text(id));
                let kind = if topo.nullable.contains(id) {
                    "string?"
                } else {
                    "string"
                };
                state.insert(id.clone(), text(kind));
            }
            nodes.push(Toml::Table(node));
        }
        let mut end = toml::Table::new();
        end.insert("id".into(), text(END));
        end.insert("kind".into(), text("end"));
        nodes.push(Toml::Table(end));
        for (from, to) in &topo.edges {
            match human(from) {
                Some((_, routes)) if routes.is_empty() => {
                    edges.push(edge(&format!("{from}{GATE}"), to, Some("true")));
                }
                Some((_, routes)) => {
                    let label = &routes
                        .iter()
                        .find(|(_, t)| t == to)
                        .expect("route edges come from routes")
                        .0;
                    edges.push(edge(&format!("{from}{ROUTE}"), to, Some(label)));
                }
                None => edges.push(edge(from, to, None)),
            }
        }
        let table = self.file_table(&topo.order[0], state, nodes, edges);
        let text = toml::to_string(&table)
            .map_err(|e| KitError::config(&owner, format!("flow does not serialize: {e}")))?;
        Ok(Flow::parse(&owner, &text)?)
    }

    fn node_table(&self, id: &str, node: &mut toml::Table) {
        match self.node(id) {
            Node::Agent(agent) => {
                node.insert("kind".into(), Toml::String("agent".into()));
                node.insert("persona".into(), Toml::String(agent.name().to_string()));
            }
            Node::Step(_) => {
                node.insert("kind".into(), Toml::String("tool".into()));
                node.insert("tool".into(), Toml::String(step_name(id)));
            }
        }
    }
}

/// Pushes the keys a node fed by `from` reads: `from` itself, or for a
/// join or human node what it passes through, in order (nested ones
/// expand in turn). A key already pushed is skipped. Depth is bounded.
fn expand(from: &str, pass: &BTreeMap<&str, Vec<String>>, keys: &mut Vec<String>, depth: usize) {
    assert!(
        depth <= MAX_NODES,
        "pass-through nesting is bounded by the node count"
    );
    match pass.get(from) {
        Some(through) => {
            for t in through {
                expand(t, pass, keys, depth + 1);
            }
        }
        None if !keys.iter().any(|k| k == from) => keys.push(from.to_string()),
        None => {}
    }
}

fn text(s: &str) -> Toml {
    Toml::String(s.to_string())
}

fn keys(list: &[String]) -> Toml {
    Toml::Array(list.iter().map(|k| text(k)).collect())
}

fn edge(from: &str, to: &str, when: Option<&str>) -> Toml {
    let mut edge = toml::Table::new();
    edge.insert("from".into(), text(from));
    edge.insert("to".into(), text(to));
    if let Some(when) = when {
        edge.insert("when".into(), text(when));
    }
    Toml::Table(edge)
}
