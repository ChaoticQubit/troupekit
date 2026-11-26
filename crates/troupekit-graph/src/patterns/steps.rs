use super::{agent, check_persona, from_toml, node_id, FlowTable, END};
use crate::error::WorkflowError;
use serde::Deserialize;
use std::collections::BTreeMap;
use toml::{Table, Value};
use troupekit_agent::dsl::is_ident;

const STEPS_MAX: usize = 64;
const PARALLEL_MAX: usize = 32;
const ROUTES_MAX: usize = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct StepsFile {
    name: String,
    #[serde(default)]
    state: Table,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Step {
    id: String,
    agent: Option<String>,
    prompt: Option<String>,
    parallel: Option<Vec<String>>,
    #[serde(rename = "loop")]
    repeat: Option<Value>,
    feedback: Option<Feedback>,
    after: Option<Vec<String>>,
    trigger: Option<String>,
    when: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Feedback {
    prompt: String,
    routes: BTreeMap<String, String>,
}

/// What a checked step does.
/// Exactly one of `agent`, `parallel` or `feedback` is set on each step.
/// `loop` and `after` are refused until the pattern IR can express them.
enum Kind<'a> {
    Agent(&'a str),
    Parallel(&'a [String]),
    Feedback(&'a Feedback),
}

/// Compiles steps pattern `text` read from `file` into a flow table:
/// `agent` steps run in order, `parallel` fans out to several agents and
/// joins, `when` skips a step on a bool key, `feedback` asks a human and
/// routes back, to the next step or to `end`. Returns the table or the
/// first problem, naming the `steps[i].*` field.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let raw: StepsFile = toml::from_str(text).map_err(|e| from_toml(file, text, &e))?;
    let steps = &raw.steps;
    if steps.is_empty() || steps.len() > STEPS_MAX {
        let m = format!("lists 1 to {STEPS_MAX} steps");
        return Err(WorkflowError::new(file, "steps", m));
    }
    let mut kinds = Vec::with_capacity(steps.len());
    for (i, s) in steps.iter().enumerate() {
        kinds.push(check_step(file, i, s, steps, &raw.state)?);
    }
    assert_eq!(kinds.len(), steps.len());
    build(file, &raw, &kinds)
}

fn check_step<'a>(
    file: &str,
    i: usize,
    s: &'a Step,
    steps: &[Step],
    state: &Table,
) -> Result<Kind<'a>, WorkflowError> {
    let field = |f: &str| format!("steps[{i}].{f}");
    let err = |f: &str, m: String| WorkflowError::new(file, &field(f), m);
    let reserved = ["input", "output", END];
    if !is_ident(&s.id) || s.id.len() > 64 || reserved.contains(&s.id.as_str()) {
        let m = format!(
            "`{}` must be 1 to 64 of [a-z_][a-z0-9_]*, not {reserved:?}",
            s.id
        );
        return Err(err("id", m));
    }
    if steps[..i].iter().any(|o| o.id == s.id) {
        return Err(err("id", format!("`{}` repeats", s.id)));
    }
    if s.repeat.is_some() || s.after.is_some() || s.trigger.is_some() {
        let f = if s.repeat.is_some() { "loop" } else { "after" };
        return Err(err(
            f,
            "loop, after and trigger steps are not built yet".into(),
        ));
    }
    if let Some(w) = &s.when {
        if !state.contains_key(w) {
            return Err(err("when", format!("`{w}` is not declared in [state]")));
        }
    }
    let kind = match (&s.agent, &s.parallel, &s.feedback) {
        (Some(a), None, None) => {
            check_persona(file, &field("agent"), a)?;
            Kind::Agent(a)
        }
        (None, Some(p), None) => check_parallel(file, i, s, p, steps)?,
        (None, None, Some(f)) => check_feedback(file, i, s, f, steps)?,
        _ => {
            return Err(WorkflowError::new(
                file,
                &format!("steps[{i}]"),
                "takes exactly one of agent, parallel, loop or feedback",
            ))
        }
    };
    if s.prompt.is_some() && !matches!(kind, Kind::Agent(_)) {
        return Err(err("prompt", "is only for agent steps".into()));
    }
    Ok(kind)
}

fn check_parallel<'a>(
    file: &str,
    i: usize,
    s: &Step,
    agents: &'a [String],
    steps: &[Step],
) -> Result<Kind<'a>, WorkflowError> {
    let err = |f: &str, m: &str| Err(WorkflowError::new(file, &format!("steps[{i}].{f}"), m));
    if agents.len() < 2 || agents.len() > PARALLEL_MAX {
        return err("parallel", "lists 2 to 32 agents");
    }
    for (j, a) in agents.iter().enumerate() {
        check_persona(file, &format!("steps[{i}].parallel[{j}]"), a)?;
        if agents[..j].contains(a) {
            return err(&format!("parallel[{j}]"), "repeats");
        }
    }
    if i == 0 {
        return err("parallel", "the first step has nothing to fan out from");
    }
    let prev = &steps[i - 1];
    if prev.when.is_some() || prev.feedback.is_some() {
        return err(
            "parallel",
            "the step before a parallel step must not choose a route",
        );
    }
    if s.when.is_some() {
        return err("when", "a parallel step cannot have `when`");
    }
    assert!(agents.len() >= 2, "a fan-out has two or more branches");
    Ok(Kind::Parallel(agents))
}

fn check_feedback<'a>(
    file: &str,
    i: usize,
    s: &Step,
    f: &'a Feedback,
    steps: &[Step],
) -> Result<Kind<'a>, WorkflowError> {
    let err = |fld: String, m: String| Err(WorkflowError::new(file, &fld, m));
    if f.prompt.trim().is_empty() {
        return err(format!("steps[{i}].feedback.prompt"), "is empty".into());
    }
    if f.routes.is_empty() || f.routes.len() > ROUTES_MAX {
        return err(
            format!("steps[{i}].feedback.routes"),
            format!("lists 1 to {ROUTES_MAX} routes"),
        );
    }
    if s.when.is_some() {
        return err(
            format!("steps[{i}].when"),
            "a feedback step cannot have `when`".into(),
        );
    }
    for (label, target) in &f.routes {
        let fld = format!("steps[{i}].feedback.routes.{label}");
        if !is_ident(label) || label.len() > 64 {
            return err(
                fld,
                format!("label `{label}` must be 1 to 64 of [a-z_][a-z0-9_]*"),
            );
        }
        let to = steps.iter().find(|o| o.id == *target);
        match to {
            None if target != END => return err(fld, format!("`{target}` is not a step or `end`")),
            Some(o) if o.parallel.is_some() => {
                return err(
                    fld,
                    format!("`{target}` is a parallel step; route to the step before it"),
                )
            }
            _ => {}
        }
        let at = steps.iter().position(|o| o.id == *target);
        if at.is_some_and(|j| j > i + 1) {
            let next = &steps[i + 1].id;
            let m = format!(
                "`{target}` jumps past step `{next}`, whose keys later steps read; \
                 route back, to the next step or to `end`"
            );
            return err(fld, m);
        }
    }
    Ok(Kind::Feedback(f))
}

fn entry(s: &Step) -> String {
    match s.when {
        Some(_) => format!("when_{}", s.id),
        None => s.id.clone(),
    }
}

fn target_entry(steps: &[Step], target: &str) -> String {
    steps
        .iter()
        .find(|s| s.id == target)
        .map_or_else(|| END.to_string(), entry)
}

/// Builds the flow table. Each step reads `input` and the keys the step
/// before it wrote; a `when` step can be skipped, so it hands on the keys
/// it read instead of its own. An agent step writes a key named after it;
/// the last agent step in the list writes `output`.
fn build(file: &str, raw: &StepsFile, kinds: &[Kind<'_>]) -> Result<Table, WorkflowError> {
    let steps = &raw.steps;
    let mut t = FlowTable::new(file, &raw.name, &entry(&steps[0]));
    for (key, ty) in &raw.state {
        t.state(&format!("state.{key}"), key, ty.clone())?;
    }
    let mut pending: Vec<(String, Option<String>)> = Vec::new();
    let mut prev_keys: Vec<String> = Vec::new();
    let last_agent = kinds.iter().rposition(|k| matches!(k, Kind::Agent(_)));
    for (i, (s, kind)) in steps.iter().zip(kinds).enumerate() {
        let field = format!("steps[{i}]");
        let mut reads = vec!["input".to_string()];
        reads.extend(prev_keys.iter().cloned());
        if let Some(on) = &s.when {
            let id = format!("when_{}", s.id);
            t.node(&format!("{field}.when"), switch(&id, on))?;
            t.routed(&id, &s.id, "true");
        }
        let fan_source = match kind {
            Kind::Parallel(_) => {
                let (source, _) = pending.pop().expect("a parallel step follows a step");
                assert!(
                    pending.is_empty(),
                    "the step before a parallel step has one exit"
                );
                Some(source)
            }
            _ => {
                connect(&mut t, &mut pending, &entry(s));
                None
            }
        };
        prev_keys = match kind {
            Kind::Agent(persona) => {
                let key = if Some(i) == last_agent {
                    "output".to_string()
                } else {
                    s.id.clone()
                };
                if key != "output" {
                    t.state(&field, &key, Value::String("string?".into()))?;
                }
                let node = agent(&s.id, persona, s.prompt.as_deref(), &reads, Some(&key));
                t.node(&field, node)?;
                if s.when.is_some() {
                    pending.push((format!("when_{}", s.id), Some("false".into())));
                }
                pending.push((s.id.clone(), None));
                if s.when.is_some() {
                    prev_keys
                } else {
                    vec![key]
                }
            }
            Kind::Parallel(agents) => {
                let source = fan_source.expect("set for parallel steps");
                parallel(&mut t, &field, s, agents, &reads, &source, &mut pending)?
            }
            Kind::Feedback(f) => feedback(&mut t, &field, s, f, steps, &reads)?,
        };
    }
    t.end("steps")?;
    connect(&mut t, &mut pending, END);
    Ok(t.finish())
}

fn connect(t: &mut FlowTable, pending: &mut Vec<(String, Option<String>)>, to: &str) {
    for (from, when) in pending.drain(..) {
        match when {
            Some(w) => t.routed(&from, to, &w),
            None => t.edge(&from, to),
        }
    }
}

fn switch(id: &str, on: &str) -> Table {
    let mut s = Table::new();
    s.insert("id".into(), Value::String(id.into()));
    s.insert("kind".into(), Value::String("switch".into()));
    s.insert("on".into(), Value::String(on.into()));
    s
}

/// Adds a parallel step: one agent node `<step>_<persona>` per agent, each
/// writing its own key, fanned out from `source` and joined at a join node
/// named after the step. Returns the keys the branches wrote.
fn parallel(
    t: &mut FlowTable,
    field: &str,
    s: &Step,
    agents: &[String],
    reads: &[String],
    source: &str,
    pending: &mut Vec<(String, Option<String>)>,
) -> Result<Vec<String>, WorkflowError> {
    let mut keys = Vec::with_capacity(agents.len());
    for (j, a) in agents.iter().enumerate() {
        let id = format!("{}_{}", s.id, node_id(a));
        let f = format!("{field}.parallel[{j}]");
        t.state(&f, &id, Value::String("string?".into()))?;
        t.node(&f, agent(&id, a, None, reads, Some(&id)))?;
        t.edge(source, &id);
        t.edge(&id, &s.id);
        keys.push(id);
    }
    let mut join = Table::new();
    join.insert("id".into(), Value::String(s.id.clone()));
    join.insert("kind".into(), Value::String("join".into()));
    t.node(field, join)?;
    pending.push((s.id.clone(), None));
    assert_eq!(keys.len(), agents.len(), "one key per branch");
    Ok(keys)
}

/// Adds a feedback step: a human question named after the step whose
/// answer is one of the route labels, then a switch `route_<step>` that
/// takes each label to its step. Returns the answer key.
fn feedback(
    t: &mut FlowTable,
    field: &str,
    s: &Step,
    f: &Feedback,
    steps: &[Step],
    reads: &[String],
) -> Result<Vec<String>, WorkflowError> {
    let labels: Vec<&str> = f.routes.keys().map(String::as_str).collect();
    let ty = format!("enum({})", labels.join(", "));
    t.state(field, &s.id, Value::String(ty))?;
    let mut human = Table::new();
    human.insert("id".into(), Value::String(s.id.clone()));
    human.insert("kind".into(), Value::String("human".into()));
    human.insert("ask".into(), Value::String("question".into()));
    human.insert("prompt".into(), Value::String(f.prompt.clone()));
    human.insert("reads".into(), super::strings(reads));
    human.insert("writes".into(), Value::String(s.id.clone()));
    t.node(field, human)?;
    let route = format!("route_{}", s.id);
    t.node(&format!("{field}.feedback"), switch(&route, &s.id))?;
    t.edge(&s.id, &route);
    for (label, target) in &f.routes {
        t.routed(&route, &target_entry(steps, target), label);
    }
    assert!(!labels.is_empty(), "a feedback step routes somewhere");
    Ok(vec![s.id.clone()])
}
