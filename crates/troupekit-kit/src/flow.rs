use crate::error::KitError;
use crate::graph::{step_name, Graph, Node, END, INPUT, MAX_NODES};
use std::collections::{BTreeMap, BTreeSet};
use toml::Value as Toml;
use troupekit_graph::Flow;

/// One flow part added by a builder method: a fan-out from one node into
/// parallel branches, or a join that merges branches. Parts turn the
/// chain into a flow; a graph without parts builds the 0.2 workflow.
#[derive(Clone, Debug)]
pub(crate) enum Part {
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

/// The checked shape of a graph with parts: every node and join in
/// topological order, the keys each agent or step node reads, the edges
/// (the last node's edge goes to `end`) and the node whose output is the
/// answer. Built by `Graph::topology`.
pub(crate) struct Topology {
    pub(crate) order: Vec<String>,
    pub(crate) reads: BTreeMap<String, Vec<String>>,
    pub(crate) edges: Vec<(String, String)>,
    pub(crate) last: String,
}

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

    /// Checks the nodes, joins and edges of a graph with parts and orders
    /// them. A node fans out only through `parallel` and merges only at a
    /// `join`; there is one start and one last node. Returns the shape or
    /// the first problem found.
    pub(crate) fn topology(&self) -> Result<Topology, String> {
        let nodes = self.ids()?;
        let mut joins: BTreeMap<&str, &[String]> = BTreeMap::new();
        let mut edges: Vec<(String, String)> = self.edges.clone();
        let mut sources: BTreeSet<&str> = BTreeSet::new();
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
                    let taken = id == INPUT || id == END || nodes.contains(&id.as_str());
                    if taken || joins.contains_key(id.as_str()) {
                        return Err(format!("join id `{id}` is taken"));
                    }
                    if from.len() < 2 {
                        return Err(format!("join `{id}` needs at least two branches"));
                    }
                    joins.insert(id, from);
                    edges.extend(from.iter().map(|f| (f.clone(), id.clone())));
                }
            }
        }
        let all: Vec<&str> = nodes.iter().copied().chain(joins.keys().copied()).collect();
        if all.len() > MAX_NODES {
            return Err(format!(
                "takes at most {MAX_NODES} nodes and joins, got {}",
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
        let last = match sinks[..] {
            [one] if joins.contains_key(one) => {
                return Err(format!("join `{one}` needs an .edge to the node after it"))
            }
            [one] => one.to_string(),
            _ => {
                return Err(format!(
                    "nodes {} end the graph; join their branches",
                    sinks.join(", ")
                ))
            }
        };
        let mut reads = BTreeMap::new();
        for id in &nodes {
            let mut keys = Vec::new();
            for (from, _) in edges.iter().filter(|e| e.1 == *id) {
                expand(from, &joins, &mut keys, 0);
            }
            if keys.is_empty() {
                keys.push(INPUT.to_string());
            }
            reads.insert(id.to_string(), keys);
        }
        edges.push((last.clone(), END.to_string()));
        assert_eq!(
            order.len(),
            all.len(),
            "every node and join is ordered once"
        );
        Ok(Topology {
            order,
            reads,
            edges,
            last,
        })
    }

    /// Serializes the graph with parts as flow TOML and parses it, so the
    /// engine checks fan-outs, joins and state. Each node writes its own
    /// id as a string key and reads `topo.reads`. Returns the flow.
    pub(crate) fn flow(&self, topo: &Topology) -> Result<Flow, KitError> {
        let owner = format!("graph:{}", self.name);
        let mut state = toml::Table::new();
        state.insert(INPUT.into(), Toml::String("string".into()));
        let mut nodes = Vec::with_capacity(topo.order.len() + 1);
        for id in &topo.order {
            let mut node = toml::Table::new();
            node.insert("id".into(), Toml::String(id.clone()));
            let join = self.parts.iter().find_map(|p| match p {
                Part::Join { id: j, any, .. } if j == id => Some(*any),
                _ => None,
            });
            match join {
                Some(any) => {
                    node.insert("kind".into(), Toml::String("join".into()));
                    if any {
                        node.insert("mode".into(), Toml::String("any".into()));
                    }
                }
                None => {
                    self.node_table(id, &mut node);
                    let reads = topo.reads[id]
                        .iter()
                        .map(|k| Toml::String(k.clone()))
                        .collect();
                    node.insert("reads".into(), Toml::Array(reads));
                    node.insert("writes".into(), Toml::String(id.clone()));
                    state.insert(id.clone(), Toml::String("string".into()));
                }
            }
            nodes.push(Toml::Table(node));
        }
        let mut end = toml::Table::new();
        end.insert("id".into(), Toml::String(END.into()));
        end.insert("kind".into(), Toml::String("end".into()));
        nodes.push(Toml::Table(end));
        let edges = topo
            .edges
            .iter()
            .map(|(from, to)| {
                let mut edge = toml::Table::new();
                edge.insert("from".into(), Toml::String(from.clone()));
                edge.insert("to".into(), Toml::String(to.clone()));
                Toml::Table(edge)
            })
            .collect();
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
/// join each of its branches in order (nested joins expand in turn).
/// A key already pushed is skipped. Depth is bounded by the node cap.
fn expand(from: &str, joins: &BTreeMap<&str, &[String]>, keys: &mut Vec<String>, depth: usize) {
    assert!(
        depth <= MAX_NODES,
        "join nesting is bounded by the node count"
    );
    match joins.get(from) {
        Some(branches) => {
            for b in branches.iter() {
                expand(b, joins, keys, depth + 1);
            }
        }
        None if !keys.iter().any(|k| k == from) => keys.push(from.to_string()),
        None => {}
    }
}
