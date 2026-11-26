use super::{agent, check_persona, from_toml, strings, FlowTable, END};
use crate::error::WorkflowError;
use serde::Deserialize;
use toml::{Table, Value};
use troupekit_agent::dsl::is_ident;

const TASKS_MAX: usize = 64;
const BUILTIN_MANAGER: &str = "crew-manager";
const MANAGER_ID: &str = "manager";
const MANAGER_PROMPT: &str = "Run these tasks with your coworkers, in order, then answer:";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrewFile {
    name: String,
    crew: CrewRaw,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CrewRaw {
    process: String,
    manager: Option<String>,
    manager_model: Option<String>,
    tasks: Vec<Task>,
    #[serde(default)]
    state: Table,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Task {
    id: String,
    agent: String,
    description: String,
    expected_output: Option<String>,
    #[serde(default)]
    context: Vec<String>,
    #[serde(default, rename = "async")]
    run_async: bool,
    #[serde(default)]
    human_input: bool,
    when: Option<String>,
}

/// Compiles crew pattern `text` read from `file` into a flow table.
/// `sequential` chains the tasks, `hierarchical` hands them all to one
/// manager with the task agents as delegates. Returns the table or the
/// first problem, naming the `crew.*` field.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let raw: CrewFile = toml::from_str(text).map_err(|e| from_toml(file, text, &e))?;
    let crew = raw.crew;
    let err = |field: &str, m: String| WorkflowError::new(file, field, m);
    if crew.tasks.is_empty() || crew.tasks.len() > TASKS_MAX {
        return Err(err("crew.tasks", format!("lists 1 to {TASKS_MAX} tasks")));
    }
    for (i, t) in crew.tasks.iter().enumerate() {
        check_task(file, i, t, &crew.tasks, &crew.state)?;
    }
    match crew.process.as_str() {
        "sequential" => {
            if crew.manager.is_some() || crew.manager_model.is_some() {
                let m = "a manager is only for process `hierarchical`".to_string();
                return Err(err("crew.manager", m));
            }
            sequential(file, &raw.name, &crew)
        }
        "hierarchical" => hierarchical(file, &raw.name, &crew),
        other => Err(err(
            "crew.process",
            format!("`{other}` must be `sequential` or `hierarchical`"),
        )),
    }
}

fn check_task(
    file: &str,
    i: usize,
    t: &Task,
    tasks: &[Task],
    state: &Table,
) -> Result<(), WorkflowError> {
    let field = |f: &str| format!("crew.tasks[{i}].{f}");
    let err = |f: &str, m: String| Err(WorkflowError::new(file, &field(f), m));
    let reserved = ["input", "output", END, MANAGER_ID];
    if !is_ident(&t.id) || t.id.len() > 64 || reserved.contains(&t.id.as_str()) {
        let m = format!(
            "`{}` must be 1 to 64 of [a-z_][a-z0-9_]*, not {reserved:?}",
            t.id
        );
        return err("id", m);
    }
    if tasks[..i].iter().any(|o| o.id == t.id) {
        return err("id", format!("`{}` repeats", t.id));
    }
    check_persona(file, &field("agent"), &t.agent)?;
    if t.description.trim().is_empty() {
        return err("description", "is empty".into());
    }
    for (c, name) in t.context.iter().enumerate() {
        let f = format!("context[{c}]");
        match tasks.iter().position(|o| o.id == *name) {
            None => return err(&f, format!("`{name}` is not a task")),
            Some(p) if p >= i => {
                return err(
                    &f,
                    format!("`{name}` is not an earlier task; context reads only earlier tasks"),
                )
            }
            Some(p) if tasks[p].when.is_some() => {
                let m = format!("`{name}` can be skipped by `when`, so it cannot be read");
                return err(&f, m);
            }
            Some(_) if t.context[..c].contains(name) => {
                return err(&f, format!("`{name}` repeats"))
            }
            Some(_) => {}
        }
    }
    if let Some(w) = &t.when {
        if !state.contains_key(w) {
            return err("when", format!("`{w}` is not declared in [crew.state]"));
        }
    }
    if t.when.is_some() && t.human_input && i + 1 < tasks.len() {
        let m = "the next task reads this feedback, but `when` can skip it".to_string();
        return err("human_input", m);
    }
    assert!(t.context.len() <= i, "context names distinct earlier tasks");
    Ok(())
}

/// The task index that closes the async span opened by task `i`: the
/// first later task whose context names it, else the task count.
/// Returns the index; a join goes in front of that task or of the end.
fn closer(tasks: &[Task], i: usize) -> usize {
    assert!(tasks[i].run_async, "only async tasks open a span");
    let id = &tasks[i].id;
    let c = (i + 1..tasks.len())
        .find(|&j| tasks[j].context.contains(id))
        .unwrap_or(tasks.len());
    assert!(c > i, "a span closes after its task");
    c
}

/// Checks the async spans: no async first task, no `human_input` or
/// `when` on an async task or inside its span, no span inside another,
/// and no `when` on the task a fan-out starts from. Returns the closer of
/// each task that opens a span.
fn check_spans(file: &str, tasks: &[Task]) -> Result<Vec<Option<usize>>, WorkflowError> {
    let err = |i: usize, f: &str, m: &str| {
        Err(WorkflowError::new(
            file,
            &format!("crew.tasks[{i}].{f}"),
            m.to_string(),
        ))
    };
    let mut closers = vec![None; tasks.len()];
    let mut open_until = 0;
    for (i, t) in tasks.iter().enumerate() {
        let inside = i < open_until;
        if inside && t.when.is_some() {
            return err(i, "when", "a task inside an async span cannot have `when`");
        }
        if !t.run_async {
            continue;
        }
        if i == 0 {
            return err(i, "async", "the first task has nothing to run beside");
        }
        if inside {
            return err(
                i,
                "async",
                "an async task cannot start inside another async span",
            );
        }
        if t.human_input || t.when.is_some() {
            return err(
                i,
                "async",
                "an async task cannot have `human_input` or `when`",
            );
        }
        if tasks[i - 1].when.is_some() {
            return err(
                i,
                "async",
                "the task before an async task cannot have `when`",
            );
        }
        let c = closer(tasks, i);
        closers[i] = Some(c);
        open_until = c;
    }
    assert_eq!(closers.len(), tasks.len());
    Ok(closers)
}

fn task_prompt(t: &Task) -> String {
    let mut p = t.description.clone();
    if let Some(e) = &t.expected_output {
        p.push_str("\n\nExpected output: ");
        p.push_str(e);
    }
    p
}

fn key_of(tasks: &[Task], i: usize) -> String {
    if i + 1 == tasks.len() {
        "output".to_string()
    } else {
        tasks[i].id.clone()
    }
}

/// Edges waiting for the next node: each from-node with its label, such
/// as a task's exit and a `when` switch's `false` edge.
/// `connect` points them all at one node.
struct Wiring {
    pending: Vec<(String, Option<String>)>,
}

impl Wiring {
    fn connect(&mut self, t: &mut FlowTable, to: &str) {
        assert!(
            !self.pending.is_empty(),
            "every node after the start has an in-edge"
        );
        for (from, when) in self.pending.drain(..) {
            match when {
                Some(w) => t.routed(&from, to, &w),
                None => t.edge(&from, to),
            }
        }
    }
}

fn sequential(file: &str, name: &str, crew: &CrewRaw) -> Result<Table, WorkflowError> {
    let tasks = &crew.tasks;
    let closers = check_spans(file, tasks)?;
    let first = entry(&tasks[0]);
    let mut t = FlowTable::new(file, name, &first);
    for (key, ty) in &crew.state {
        t.state(&format!("crew.state.{key}"), key, ty.clone())?;
    }
    let mut wiring = Wiring {
        pending: Vec::new(),
    };
    let mut open: Option<(String, usize)> = None;
    for (i, task) in tasks.iter().enumerate() {
        if open.as_ref().is_some_and(|(_, c)| *c == i) {
            close_join(&mut t, &mut wiring, &open.take().expect("checked").0)?;
        }
        let field = format!("crew.tasks[{i}]");
        if i + 1 < tasks.len() {
            t.state(&field, &task.id, Value::String("string?".into()))?;
        }
        add_task(&mut t, &field, tasks, i)?;
        if let Some(c) = closers[i] {
            let join = format!("join_{}", task.id);
            let (source, _) = wiring.pending.pop().expect("an async task follows a task");
            assert!(
                wiring.pending.is_empty(),
                "the task before an async task has one exit"
            );
            t.edge(&source, &task.id);
            t.edge(&exit(task), &join);
            wiring.pending.push((source, None));
            open = Some((join, c));
            continue;
        }
        if i > 0 {
            wiring.connect(&mut t, &entry(task));
        }
        if task.when.is_some() {
            wiring
                .pending
                .push((format!("when_{}", task.id), Some("false".into())));
        }
        wiring.pending.push((exit(task), None));
    }
    if let Some((join, _)) = open.take() {
        close_join(&mut t, &mut wiring, &join)?;
    }
    t.end("crew")?;
    wiring.connect(&mut t, END);
    Ok(t.finish())
}

fn entry(task: &Task) -> String {
    match task.when {
        Some(_) => format!("when_{}", task.id),
        None => task.id.clone(),
    }
}

fn exit(task: &Task) -> String {
    if task.human_input {
        format!("review_{}", task.id)
    } else {
        task.id.clone()
    }
}

fn close_join(t: &mut FlowTable, wiring: &mut Wiring, join: &str) -> Result<(), WorkflowError> {
    let mut node = Table::new();
    node.insert("id".into(), Value::String(join.into()));
    node.insert("kind".into(), Value::String("join".into()));
    t.node("crew.tasks", node)?;
    wiring.connect(t, join);
    wiring.pending.push((join.to_string(), None));
    assert_eq!(wiring.pending.len(), 1, "a join leads on by one edge");
    Ok(())
}

/// Adds task `i`: its `when` switch, its agent node and its review node,
/// with the edges between them. The agent reads `input`, its context keys
/// and the previous task's feedback, and writes its own key, or `output`
/// for the last task.
fn add_task(t: &mut FlowTable, field: &str, tasks: &[Task], i: usize) -> Result<(), WorkflowError> {
    let task = &tasks[i];
    let mut reads = vec!["input".to_string()];
    reads.extend(task.context.iter().cloned());
    if i > 0 && tasks[i - 1].human_input {
        reads.push(format!("{}_feedback", tasks[i - 1].id));
    }
    if let Some(on) = &task.when {
        let id = format!("when_{}", task.id);
        let mut switch = Table::new();
        switch.insert("id".into(), Value::String(id.clone()));
        switch.insert("kind".into(), Value::String("switch".into()));
        switch.insert("on".into(), Value::String(on.clone()));
        t.node(&format!("{field}.when"), switch)?;
        t.routed(&id, &task.id, "true");
    }
    let key = key_of(tasks, i);
    let prompt = task_prompt(task);
    t.node(
        field,
        agent(&task.id, &task.agent, Some(&prompt), &reads, Some(&key)),
    )?;
    if task.human_input {
        let (id, feedback) = (
            format!("review_{}", task.id),
            format!("{}_feedback", task.id),
        );
        t.state(
            &format!("{field}.human_input"),
            &feedback,
            Value::String("string".into()),
        )?;
        let mut human = Table::new();
        human.insert("id".into(), Value::String(id.clone()));
        human.insert("kind".into(), Value::String("human".into()));
        human.insert("ask".into(), Value::String("question".into()));
        let ask = format!("Review the result of task `{}`.", task.id);
        human.insert("prompt".into(), Value::String(ask));
        human.insert("reads".into(), strings(&[key]));
        human.insert("writes".into(), Value::String(feedback));
        t.node(&format!("{field}.human_input"), human)?;
        t.edge(&task.id, &id);
    }
    assert!(
        reads.len() > task.context.len(),
        "every task reads its input"
    );
    Ok(())
}

fn hierarchical(file: &str, name: &str, crew: &CrewRaw) -> Result<Table, WorkflowError> {
    for (i, task) in crew.tasks.iter().enumerate() {
        if task.run_async || task.human_input || task.when.is_some() {
            let field = format!("crew.tasks[{i}]");
            let m = "`async`, `human_input` and `when` are only for process `sequential`";
            return Err(WorkflowError::new(file, &field, m));
        }
    }
    let persona = crew.manager.as_deref().unwrap_or(BUILTIN_MANAGER);
    check_persona(file, "crew.manager", persona)?;
    let mut agents: Vec<String> = Vec::new();
    let mut prompt = MANAGER_PROMPT.to_string();
    for (i, task) in crew.tasks.iter().enumerate() {
        if !agents.contains(&task.agent) {
            agents.push(task.agent.clone());
        }
        prompt.push_str(&format!(
            "\n{}. {} [{}]: {}",
            i + 1,
            task.id,
            task.agent,
            task_prompt(task)
        ));
        if !task.context.is_empty() {
            let uses = task.context.join(", ");
            prompt.push_str(&format!(" Uses the results of: {uses}."));
        }
    }
    let mut t = FlowTable::new(file, name, MANAGER_ID);
    for (key, ty) in &crew.state {
        t.state(&format!("crew.state.{key}"), key, ty.clone())?;
    }
    let reads = ["input".to_string()];
    let mut manager = agent(MANAGER_ID, persona, Some(&prompt), &reads, Some("output"));
    manager.insert("delegates".into(), strings(&agents));
    if let Some(m) = &crew.manager_model {
        manager.insert("model".into(), Value::String(m.clone()));
    }
    t.node("crew.manager", manager)?;
    t.edge(MANAGER_ID, END);
    t.end("crew")?;
    assert!(!agents.is_empty() && agents.len() <= crew.tasks.len());
    Ok(t.finish())
}
