use super::parse::Json;
use crate::category::Trace;
use std::collections::HashMap;

/// A field value read back from a trace file.
/// `Withheld` is content written only as its byte count; `Other` stands
/// for an array or object, which the writer never produces.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum FieldValue {
    Str(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Null,
    Withheld(u64),
    Other,
}

/// Whether a node is a span with a duration or a point event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Span,
    Event,
}

/// One span or event of a `SpanTree`, with its children in start order.
/// Input: built by `read_jsonl`. Output: getters only.
#[derive(Clone, Debug)]
pub struct Node {
    kind: NodeKind,
    category: Trace,
    name: String,
    span: u64,
    parent: u64,
    start_us: u64,
    end_us: Option<u64>,
    dur_us: Option<u64>,
    ok: Option<bool>,
    err: Option<String>,
    fields: Vec<(String, FieldValue)>,
    children: Vec<usize>,
}

impl Node {
    /// Returns span or event. Input: none.
    pub fn kind(&self) -> NodeKind {
        self.kind
    }

    /// Returns the category. Input: none.
    pub fn category(&self) -> Trace {
        self.category
    }

    /// Returns the span or event name. Input: none.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the span id, 0 for an event. Input: none.
    pub fn span_id(&self) -> u64 {
        self.span
    }

    /// Returns the parent span id, 0 at the root. Input: none.
    pub fn parent_id(&self) -> u64 {
        self.parent
    }

    /// Returns the start time in microseconds since the Unix epoch.
    pub fn start_us(&self) -> u64 {
        self.start_us
    }

    /// Returns the end time, `None` for an event or an open span.
    pub fn end_us(&self) -> Option<u64> {
        self.end_us
    }

    /// Returns the duration in microseconds when the span ended.
    pub fn dur_us(&self) -> Option<u64> {
        self.dur_us
    }

    /// Returns whether the span ended without error; `None` while open.
    pub fn ok(&self) -> Option<bool> {
        self.ok
    }

    /// Returns the error text of a failed span. Input: none.
    pub fn err(&self) -> Option<&str> {
        self.err.as_deref()
    }

    /// Returns the start fields followed by the end fields. Input: none.
    pub fn fields(&self) -> &[(String, FieldValue)] {
        &self.fields
    }

    /// Returns the last value written under `key`. Input: the key.
    pub fn field(&self, key: &str) -> Option<&FieldValue> {
        self.fields
            .iter()
            .rev()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Returns the indexes of the children, in start order. Input: none.
    pub fn children(&self) -> &[usize] {
        &self.children
    }

    /// Returns whether this is a span whose end record never came.
    pub fn is_open(&self) -> bool {
        self.kind == NodeKind::Span && self.end_us.is_none()
    }
}

/// Every span and event of a trace file as a tree, plus counters for
/// every line or limit the reader refused. Input: built by `read_jsonl`.
/// Output: roots, nodes by index, per-category counts and the time range.
#[derive(Clone, Debug, Default)]
pub struct SpanTree {
    name: Option<String>,
    trace_id: Option<String>,
    nodes: Vec<Node>,
    roots: Vec<usize>,
    counts: [u64; Trace::ALL.len()],
    range_us: Option<(u64, u64)>,
    pub(super) bad_lines: u64,
    pub(super) oversized_lines: u64,
    pub(super) deep_lines: u64,
    pub(super) record_limit_hit: bool,
    unmatched_ends: u64,
    truncated: bool,
    footer: Option<(u64, u64)>,
    index: HashMap<(String, u64), usize>,
}

impl SpanTree {
    /// Returns the trace name from the header. Input: none.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Returns the trace id from the header, 16 hex characters.
    pub fn trace_id(&self) -> Option<&str> {
        self.trace_id.as_deref()
    }

    /// Returns the indexes of the root nodes, in start order.
    pub fn roots(&self) -> &[usize] {
        &self.roots
    }

    /// Returns the node at `index`, from `roots` or `Node::children`.
    /// Input: an index this tree gave out. Panics on any other index.
    pub fn node(&self, index: usize) -> &Node {
        &self.nodes[index]
    }

    /// Returns every node in file order. Input: none.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// Returns how many spans and events have category `t`.
    pub fn count(&self, t: Trace) -> u64 {
        self.counts[t.index()]
    }

    /// Returns the first and last timestamp in microseconds, if any node.
    pub fn time_range_us(&self) -> Option<(u64, u64)> {
        self.range_us
    }

    /// Returns how many lines were not valid records. Input: none.
    pub fn bad_lines(&self) -> u64 {
        self.bad_lines
    }

    /// Returns how many lines were longer than the line limit.
    pub fn oversized_lines(&self) -> u64 {
        self.oversized_lines
    }

    /// Returns how many lines nested deeper than the depth limit.
    pub fn deep_lines(&self) -> u64 {
        self.deep_lines
    }

    /// Returns whether reading stopped at the record limit.
    pub fn record_limit_hit(&self) -> bool {
        self.record_limit_hit
    }

    /// Returns how many end records matched no start. Input: none.
    pub fn unmatched_ends(&self) -> u64 {
        self.unmatched_ends
    }

    /// Returns how many spans never ended. Input: none.
    pub fn open_spans(&self) -> u64 {
        self.nodes.iter().filter(|n| n.is_open()).count() as u64
    }

    /// Returns whether the writer stopped at its byte limit.
    pub fn truncated(&self) -> bool {
        self.truncated
    }

    /// Returns the footer's dropped-field and cut-string counts, if any.
    pub fn footer_counts(&self) -> Option<(u64, u64)> {
        self.footer
    }

    pub(super) fn apply(&mut self, record: Json) -> Result<(), ()> {
        let Json::Obj(fields) = record else {
            return Err(());
        };
        let rec = Record(&fields);
        match rec.str("t")? {
            "header" => {
                self.name = Some(rec.str("name")?.to_owned());
                self.trace_id = Some(rec.str("trace")?.to_owned());
            }
            "start" => self.add(&rec, NodeKind::Span)?,
            "event" => self.add(&rec, NodeKind::Event)?,
            "end" => self.end(&rec)?,
            "truncated" => self.truncated = true,
            "footer" => self.footer = Some((rec.num("dropped")?, rec.num("truncated")?)),
            _ => return Err(()),
        }
        Ok(())
    }

    fn add(&mut self, rec: &Record<'_>, kind: NodeKind) -> Result<(), ()> {
        let trace = rec.str("trace")?;
        let span = rec.num("span")?;
        let parent = rec.num("parent")?;
        let category: Trace = rec.str("cat")?.parse().map_err(|_| ())?;
        let start_us = rec.num("ts_us")?;
        if kind == NodeKind::Span
            && (span == 0 || self.index.contains_key(&(trace.to_owned(), span)))
        {
            return Err(());
        }
        let index = self.nodes.len();
        self.nodes.push(Node {
            kind,
            category,
            name: rec.str("name")?.to_owned(),
            span,
            parent,
            start_us,
            end_us: None,
            dur_us: None,
            ok: None,
            err: None,
            fields: rec.fields(),
            children: Vec::new(),
        });
        match self
            .index
            .get(&(trace.to_owned(), parent))
            .filter(|_| parent != 0)
        {
            Some(&up) => self.nodes[up].children.push(index),
            None => self.roots.push(index),
        }
        if kind == NodeKind::Span {
            self.index.insert((trace.to_owned(), span), index);
        }
        self.counts[category.index()] += 1;
        self.widen(start_us);
        Ok(())
    }

    fn end(&mut self, rec: &Record<'_>) -> Result<(), ()> {
        let key = (rec.str("trace")?.to_owned(), rec.num("span")?);
        let end_us = rec.num("ts_us")?;
        let dur_us = rec.num("dur_us")?;
        let ok = rec.bool("ok")?;
        let Some(&index) = self.index.get(&key) else {
            self.unmatched_ends += 1;
            return Ok(());
        };
        let node = &mut self.nodes[index];
        if node.end_us.is_some() {
            return Err(());
        }
        node.end_us = Some(end_us);
        node.dur_us = Some(dur_us);
        node.ok = Some(ok);
        node.err = rec.str("err").ok().map(str::to_owned);
        node.fields.extend(rec.fields());
        self.widen(end_us);
        Ok(())
    }

    fn widen(&mut self, ts_us: u64) {
        let (first, last) = self.range_us.unwrap_or((ts_us, ts_us));
        self.range_us = Some((first.min(ts_us), last.max(ts_us)));
    }
}

struct Record<'a>(&'a [(String, Json)]);

impl Record<'_> {
    fn get(&self, key: &str) -> Option<&Json> {
        self.0.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    fn str(&self, key: &str) -> Result<&str, ()> {
        match self.get(key) {
            Some(Json::Str(text)) => Ok(text),
            _ => Err(()),
        }
    }

    fn num(&self, key: &str) -> Result<u64, ()> {
        match self.get(key) {
            Some(Json::UInt(n)) => Ok(*n),
            _ => Err(()),
        }
    }

    fn bool(&self, key: &str) -> Result<bool, ()> {
        match self.get(key) {
            Some(Json::Bool(b)) => Ok(*b),
            _ => Err(()),
        }
    }

    fn fields(&self) -> Vec<(String, FieldValue)> {
        let Some(Json::Obj(fields)) = self.get("fields") else {
            return Vec::new();
        };
        fields
            .iter()
            .map(|(k, v)| (k.clone(), field_value(v)))
            .collect()
    }
}

fn field_value(value: &Json) -> FieldValue {
    match value {
        Json::Str(text) => FieldValue::Str(text.clone()),
        Json::Int(n) => FieldValue::Int(*n),
        Json::UInt(n) => FieldValue::UInt(*n),
        Json::Float(n) => FieldValue::Float(*n),
        Json::Bool(b) => FieldValue::Bool(*b),
        Json::Null => FieldValue::Null,
        Json::Obj(inner) => match inner.as_slice() {
            [(key, Json::UInt(bytes))] if key == "bytes" => FieldValue::Withheld(*bytes),
            _ => FieldValue::Other,
        },
        Json::Arr(_) => FieldValue::Other,
    }
}
