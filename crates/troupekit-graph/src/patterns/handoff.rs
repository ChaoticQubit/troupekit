use super::{agent, check_persona, from_toml, node_id, strings, FlowTable, END};
use crate::error::WorkflowError;
use serde::Deserialize;
use std::collections::BTreeMap;
use toml::{Table, Value};

const AGENTS_MAX: usize = 32;
const DEFAULT_MAX_HANDOFFS: i64 = 8;
const MAX_HANDOFFS_RANGE: std::ops::RangeInclusive<i64> = 1..=100;
const STEPS_PER_HANDOFF: i64 = 16;
const HANDOFFS_KEY: &str = "handoffs";
const RECEIPTS_TYPE: &str = "list<{next, reason, answer}>?";
const HANDOFF_PROMPT: &str =
    "Answer the request yourself, or hand it to a teammate who fits it better.";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffFile {
    name: String,
    handoff: HandoffRaw,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HandoffRaw {
    start: String,
    max_handoffs: Option<i64>,
    receipts: Option<bool>,
    require_reason: Option<bool>,
    #[serde(default)]
    allow: BTreeMap<String, Vec<String>>,
}

/// Compiles handoff pattern `text` read from `file` into a flow table:
/// one goto agent per agent, whose targets are its `allow` list and
/// `end`, its `answer` set to `output`, and, with `receipts`, its reply
/// appended to `handoffs`. Returns the table or the first problem.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let raw: HandoffFile = toml::from_str(text).map_err(|e| from_toml(file, text, &e))?;
    let h = raw.handoff;
    let err = |field: &str, m: String| WorkflowError::new(file, &format!("handoff.{field}"), m);
    check_persona(file, "handoff.start", &h.start)?;
    let max = h.max_handoffs.unwrap_or(DEFAULT_MAX_HANDOFFS);
    if !MAX_HANDOFFS_RANGE.contains(&max) {
        return Err(err("max_handoffs", format!("must be 1 to 100, got {max}")));
    }
    if h.require_reason == Some(true) {
        let m = "is not built yet: the goto `reason` field cannot be required by schema";
        return Err(err("require_reason", m.into()));
    }
    let agents = check_allow(file, &h.start, &h.allow)?;
    if agents.len() > AGENTS_MAX {
        let m = format!("a handoff takes at most {AGENTS_MAX} agents");
        return Err(err("allow", m));
    }
    let receipts = h.receipts.unwrap_or(true);
    let mut t = FlowTable::new(file, &raw.name, &node_id(&h.start));
    t.max_steps((max + 1) * STEPS_PER_HANDOFF);
    if receipts {
        t.state(
            "handoff.receipts",
            HANDOFFS_KEY,
            Value::String(RECEIPTS_TYPE.into()),
        )?;
        t.reducer(HANDOFFS_KEY, "append");
    }
    for name in &agents {
        let targets = h.allow.get(name).cloned().unwrap_or_default();
        t.node(
            &format!("handoff.allow.{name}"),
            handoff_agent(name, &targets, receipts),
        )?;
    }
    t.end("handoff")?;
    assert_eq!(agents[0], h.start, "the start agent comes first");
    Ok(t.finish())
}

/// Checks every `allow` list and returns the agents: `start` first, then
/// every other named agent in name order. Each name is a valid persona,
/// no list names its own agent, and no list repeats a target.
fn check_allow(
    file: &str,
    start: &str,
    allow: &BTreeMap<String, Vec<String>>,
) -> Result<Vec<String>, WorkflowError> {
    let mut others: Vec<String> = Vec::new();
    for (from, targets) in allow {
        check_persona(file, &format!("handoff.allow.{from}"), from)?;
        for (i, to) in targets.iter().enumerate() {
            let field = format!("handoff.allow.{from}[{i}]");
            check_persona(file, &field, to)?;
            if to == from {
                return Err(WorkflowError::new(
                    file,
                    &field,
                    format!("`{from}` hands off to itself"),
                ));
            }
            if targets[..i].contains(to) {
                return Err(WorkflowError::new(file, &field, format!("`{to}` repeats")));
            }
        }
        others.extend(std::iter::once(from).chain(targets).cloned());
    }
    others.sort();
    others.dedup();
    others.retain(|a| a != start);
    let mut agents = vec![start.to_string()];
    agents.extend(others);
    assert!(agents.iter().skip(1).all(|a| a != start));
    Ok(agents)
}

fn handoff_agent(name: &str, targets: &[String], receipts: bool) -> Table {
    let mut reads = vec!["input".to_string()];
    if receipts {
        reads.push(HANDOFFS_KEY.to_string());
    }
    let writes = receipts.then_some(HANDOFFS_KEY);
    let mut node = agent(&node_id(name), name, Some(HANDOFF_PROMPT), &reads, writes);
    let mut goto: Vec<String> = targets.iter().map(|t| node_id(t)).collect();
    goto.push(END.to_string());
    node.insert("goto".into(), strings(&goto));
    let mut fields = Table::new();
    fields.insert("answer".into(), Value::String("string".into()));
    node.insert("fields".into(), Value::Table(fields));
    let mut sets = Table::new();
    sets.insert("answer".into(), Value::String("output".into()));
    node.insert("sets".into(), Value::Table(sets));
    assert_eq!(goto.last().map(String::as_str), Some(END));
    node
}
