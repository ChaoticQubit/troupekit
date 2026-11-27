use crate::error::WorkflowError;
use crate::flow::Flow;
use std::collections::BTreeSet;
use toml::{Table, Value};
use troupekit_agent::persona::{toml_error, valid_name};

pub mod crew;
pub mod handoff;
pub mod steps;
pub mod team;

const PATTERN_KEYS: [&str; 5] = ["nodes", "team", "crew", "handoff", "steps"];
const RESERVED_KEYS: [&str; 2] = ["input", "output"];
const END: &str = "end";

/// Loads pattern TOML `text` read from `file` (a flow, team, crew, handoff
/// or steps pattern) and compiles it into a validated `Flow`. Every flow
/// rule applies, because the compiled table goes through `Flow::parse`.
/// Returns the flow or the first problem found.
pub fn load(file: &str, text: &str) -> Result<Flow, WorkflowError> {
    let table = compile(file, text)?;
    let flow_text = toml::to_string(&table)
        .map_err(|e| WorkflowError::new(file, "", format!("compiled flow: {e}")))?;
    Flow::parse(file, &flow_text)
}

/// Compiles pattern TOML `text` read from `file` into the flow TOML table
/// that `Flow::parse` reads, so builders can produce the same table. A flow
/// file comes back as it was. Returns the table, or the first problem with
/// the pattern field named; more than one pattern in a file is refused.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let raw: Table = toml::from_str(text).map_err(|e| from_toml(file, text, &e))?;
    let found: Vec<&str> = PATTERN_KEYS
        .into_iter()
        .filter(|k| raw.contains_key(*k))
        .collect();
    match found.as_slice() {
        ["nodes"] => Ok(raw),
        ["team"] => team::compile(file, text),
        ["crew"] => crew::compile(file, text),
        ["handoff"] => handoff::compile(file, text),
        ["steps"] => steps::compile(file, text),
        [other] => unreachable!("`{other}` is not a pattern key"),
        [] => Err(WorkflowError::new(
            file,
            "",
            "a pattern file holds one of nodes, team, crew, handoff or steps",
        )),
        [first, second, ..] => Err(WorkflowError::new(
            file,
            second,
            format!("a pattern file holds one pattern; found `{first}` and `{second}`"),
        )),
    }
}

/// Turns a TOML decode error into a `WorkflowError` naming the field, the
/// same way workflow files report theirs.
/// Returns the error.
pub(crate) fn from_toml(file: &str, text: &str, e: &toml::de::Error) -> WorkflowError {
    let p = toml_error(file, text, e);
    WorkflowError::new(&p.file, &p.field, p.message)
}

/// The node id of persona `name`: `-` becomes `_`, and a name starting with
/// a digit gets a `p_` prefix, so every valid persona name maps to a valid
/// node id. Two names can map to one id (`1` and `p-1` both give `p_1`);
/// the table builder refuses duplicate ids. Returns the id.
pub(crate) fn node_id(name: &str) -> String {
    assert!(valid_name(name), "persona names are checked before mapping");
    let id = name.replace('-', "_");
    let id = if id.starts_with(|c: char| c.is_ascii_digit()) {
        format!("p_{id}")
    } else {
        id
    };
    assert!(!id.is_empty(), "a valid name maps to a non-empty id");
    assert!(!id.contains('-'), "node ids carry no `-`");
    id
}

/// Checks persona name `name` at pattern field `field`.
/// Returns the error naming the field when it is not 1 to 64 of
/// `[a-z0-9-]` or starts with `-`.
pub(crate) fn check_persona(file: &str, field: &str, name: &str) -> Result<(), WorkflowError> {
    if valid_name(name) {
        return Ok(());
    }
    Err(WorkflowError::new(
        file,
        field,
        format!("`{name}` must be 1 to 64 of [a-z0-9-], not starting with `-`"),
    ))
}

/// A flow TOML table under construction: header, state, reducers, nodes
/// and edges, built with the TOML serializer and never by splicing text,
/// so pattern text stays text. Refuses a duplicate node id or state key
/// with the pattern field that caused it.
pub(crate) struct FlowTable {
    file: String,
    head: Table,
    state: Table,
    reducers: Table,
    nodes: Vec<Value>,
    edges: Vec<Value>,
    ids: BTreeSet<String>,
}

impl FlowTable {
    /// Starts the table of flow `name` beginning at node `start`, with the
    /// `input` and `output` string keys every pattern flow has.
    /// Returns the table.
    pub(crate) fn new(file: &str, name: &str, start: &str) -> FlowTable {
        let mut head = Table::new();
        head.insert("name".into(), Value::String(name.into()));
        head.insert("start".into(), Value::String(start.into()));
        let mut state = Table::new();
        for key in RESERVED_KEYS {
            state.insert(key.into(), Value::String("string".into()));
        }
        assert_eq!(state.len(), RESERVED_KEYS.len());
        FlowTable {
            file: file.to_string(),
            head,
            state,
            reducers: Table::new(),
            nodes: Vec::new(),
            edges: Vec::new(),
            ids: BTreeSet::new(),
        }
    }

    /// Sets the flow's `max_steps`, the bound on every step the run takes.
    /// Patterns with a cycle set it from their own limit.
    /// `steps` must be 1 to 10 000, the range `Flow::parse` accepts.
    pub(crate) fn max_steps(&mut self, steps: i64) {
        assert!(
            (1..=10_000).contains(&steps),
            "max_steps fits the flow range"
        );
        self.head.insert("max_steps".into(), Value::Integer(steps));
    }

    /// Declares state key `key` of type `ty` for pattern field `field`.
    /// Returns an error naming the field for a key already declared,
    /// including the reserved `input` and `output`.
    pub(crate) fn state(&mut self, field: &str, key: &str, ty: Value) -> Result<(), WorkflowError> {
        if self.state.contains_key(key) {
            let m = format!("state key `{key}` is already used by the pattern");
            return Err(WorkflowError::new(&self.file, field, m));
        }
        self.state.insert(key.into(), ty);
        assert!(self.state.contains_key(key));
        Ok(())
    }

    /// Declares reducer `reducer` on state key `key`, so branches of one
    /// fan-out may all write it.
    /// The key must be declared first.
    pub(crate) fn reducer(&mut self, key: &str, reducer: &str) {
        assert!(
            self.state.contains_key(key),
            "a reducer names a declared key"
        );
        let old = self
            .reducers
            .insert(key.into(), Value::String(reducer.into()));
        assert!(old.is_none(), "one reducer per key");
    }

    /// Adds `node`, whose `id` came from pattern field `field`.
    /// Node ids are unique across the whole pattern flow.
    /// Returns an error naming the field for an id already taken.
    pub(crate) fn node(&mut self, field: &str, node: Table) -> Result<(), WorkflowError> {
        let id = node
            .get("id")
            .and_then(Value::as_str)
            .expect("every node has an id")
            .to_string();
        if !self.ids.insert(id.clone()) {
            let m = format!("node id `{id}` is already used by the pattern");
            return Err(WorkflowError::new(&self.file, field, m));
        }
        self.nodes.push(Value::Table(node));
        assert_eq!(self.ids.len(), self.nodes.len(), "one id per node");
        Ok(())
    }

    /// Adds a plain edge from `from` to `to`.
    /// Several plain edges from one node fan out; node ids are checked by
    /// `Flow::parse`.
    pub(crate) fn edge(&mut self, from: &str, to: &str) {
        self.edges.push(Value::Table(edge(from, to, None)));
    }

    /// Adds an edge from `from` to `to` taken on label `when`.
    /// Only switch and router nodes have labelled edges.
    /// Node ids are checked by `Flow::parse`.
    pub(crate) fn routed(&mut self, from: &str, to: &str, when: &str) {
        self.edges.push(Value::Table(edge(from, to, Some(when))));
    }

    /// Adds the end node every pattern flow finishes at, with id `end`
    /// and no out-edges.
    /// Returns an error when a persona already maps to the id `end`.
    pub(crate) fn end(&mut self, field: &str) -> Result<(), WorkflowError> {
        let mut t = Table::new();
        t.insert("id".into(), Value::String(END.into()));
        t.insert("kind".into(), Value::String("end".into()));
        self.node(field, t)
    }

    /// Finishes the flow table in the shape `Flow::parse` reads.
    /// Returns header, state, reducers when any, nodes and edges.
    /// The end node must have been added.
    pub(crate) fn finish(self) -> Table {
        assert!(!self.nodes.is_empty(), "a pattern flow has nodes");
        assert!(self.ids.contains(END), "a pattern flow has its end node");
        let mut t = self.head;
        t.insert("state".into(), Value::Table(self.state));
        if !self.reducers.is_empty() {
            t.insert("reducers".into(), Value::Table(self.reducers));
        }
        t.insert("nodes".into(), Value::Array(self.nodes));
        if !self.edges.is_empty() {
            t.insert("edges".into(), Value::Array(self.edges));
        }
        t
    }
}

fn edge(from: &str, to: &str, when: Option<&str>) -> Table {
    let mut t = Table::new();
    t.insert("from".into(), Value::String(from.into()));
    t.insert("to".into(), Value::String(to.into()));
    if let Some(w) = when {
        t.insert("when".into(), Value::String(w.into()));
    }
    t
}

/// Builds an agent node table: `id`, `persona`, an optional `prompt`, the
/// state keys it `reads` and the key it `writes`, if any.
/// Returns the table; callers add `goto`, `delegates` or `model`.
pub(crate) fn agent(
    id: &str,
    persona: &str,
    prompt: Option<&str>,
    reads: &[String],
    writes: Option<&str>,
) -> Table {
    assert!(
        prompt.is_some() || !reads.is_empty(),
        "an agent node has a prompt or reads"
    );
    let mut t = Table::new();
    t.insert("id".into(), Value::String(id.into()));
    t.insert("kind".into(), Value::String("agent".into()));
    t.insert("persona".into(), Value::String(persona.into()));
    if let Some(p) = prompt {
        t.insert("prompt".into(), Value::String(p.into()));
    }
    if !reads.is_empty() {
        t.insert("reads".into(), strings(reads));
    }
    if let Some(w) = writes {
        t.insert("writes".into(), Value::String(w.into()));
    }
    t
}

/// Turns `items` into a TOML array of strings, such as a node's reads,
/// goto targets or delegates.
/// Returns the array.
pub(crate) fn strings(items: &[String]) -> Value {
    Value::Array(items.iter().cloned().map(Value::String).collect())
}
