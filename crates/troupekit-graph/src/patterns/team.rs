use super::{agent, check_persona, from_toml, node_id, strings, FlowTable, END};
use crate::error::WorkflowError;
use serde::Deserialize;
use toml::{Table, Value};
use troupekit_agent::dsl::is_ident;

const MEMBERS_MAX: usize = 32;
const DEFAULT_MAX_ITERATIONS: i64 = 5;
const MAX_ITERATIONS_RANGE: std::ops::RangeInclusive<i64> = 1..=100;
const ROUTE_PROMPT: &str = "Pick the team member who should handle the request.";
const BRIEF_PROMPT: &str = "Write a short brief of the request for your team.";
const SYNTHESIZE_PROMPT: &str = "Combine your team's answers into one final answer.";
const COORDINATE_PROMPT: &str =
    "Hand parts of the request to your coworkers as needed, then give the final answer.";
const TASKS_PROMPT: &str = "Work through the request with your coworkers. `results` holds \
    what earlier passes finished. Put what this pass finished in `results` and the tasks still \
    open in `tasks`. Set `done` to true and give the final answer in `answer` once nothing is \
    left.";
const STEPS_PER_ITERATION: i64 = 16;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TeamFile {
    name: String,
    team: TeamRaw,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TeamRaw {
    mode: String,
    leader: String,
    members: Vec<String>,
    #[serde(default)]
    approve: Vec<String>,
    max_iterations: Option<i64>,
    #[serde(default)]
    state: Table,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Route,
    Coordinate,
    Collaborate,
    Broadcast,
    Tasks,
}

impl Mode {
    fn parse(s: &str) -> Option<Mode> {
        match s {
            "route" => Some(Mode::Route),
            "coordinate" => Some(Mode::Coordinate),
            "collaborate" => Some(Mode::Collaborate),
            "broadcast" => Some(Mode::Broadcast),
            "tasks" => Some(Mode::Tasks),
            _ => None,
        }
    }
}

/// A checked team: its mode, leader, members, approvals and the extra
/// state keys every member reads, in key order.
/// Persona names are valid and distinct, so every one maps to a node id.
struct Team {
    name: String,
    mode: Mode,
    leader: String,
    members: Vec<String>,
    approve: Vec<String>,
    max_iterations: i64,
    state: Vec<(String, Value)>,
}

/// Compiles team pattern `text` read from `file` into a flow table.
/// `route` gives the leader goto targets, `coordinate` gives it the
/// members as delegates. Returns the table or the first problem, naming
/// the `team.*` field.
pub fn compile(file: &str, text: &str) -> Result<Table, WorkflowError> {
    let raw: TeamFile = toml::from_str(text).map_err(|e| from_toml(file, text, &e))?;
    let team = check(file, raw)?;
    match team.mode {
        Mode::Route => route(file, &team),
        Mode::Coordinate => coordinate(file, &team),
        Mode::Collaborate | Mode::Broadcast => fan_out(file, &team),
        Mode::Tasks => tasks(file, &team),
    }
}

fn check(file: &str, raw: TeamFile) -> Result<Team, WorkflowError> {
    let err = |field: &str, m: String| WorkflowError::new(file, field, m);
    let t = raw.team;
    let mode = Mode::parse(&t.mode).ok_or_else(|| {
        let m = format!(
            "`{}` must be one of route, coordinate, collaborate, broadcast, tasks",
            t.mode
        );
        err("team.mode", m)
    })?;
    check_persona(file, "team.leader", &t.leader)?;
    if t.members.is_empty() || t.members.len() > MEMBERS_MAX {
        return Err(err(
            "team.members",
            format!("lists 1 to {MEMBERS_MAX} members"),
        ));
    }
    for (i, m) in t.members.iter().enumerate() {
        let field = format!("team.members[{i}]");
        check_persona(file, &field, m)?;
        if *m == t.leader {
            return Err(err(&field, format!("`{m}` is the leader")));
        }
        if t.members[..i].contains(m) {
            return Err(err(&field, format!("`{m}` repeats")));
        }
    }
    if matches!(mode, Mode::Collaborate | Mode::Broadcast) && t.members.len() < 2 {
        return Err(err(
            "team.members",
            format!("mode `{}` needs 2 or more members", t.mode),
        ));
    }
    check_approve(file, mode, &t.approve, &t.members)?;
    let max_iterations = check_iterations(file, mode, t.max_iterations)?;
    let state = check_state(file, t.state)?;
    assert!(!t.members.is_empty(), "a team has members");
    assert!(
        t.members.len() <= MEMBERS_MAX,
        "the member count is bounded"
    );
    Ok(Team {
        name: raw.name,
        mode,
        leader: t.leader,
        members: t.members,
        approve: t.approve,
        max_iterations,
        state,
    })
}

fn check_approve(
    file: &str,
    mode: Mode,
    approve: &[String],
    members: &[String],
) -> Result<(), WorkflowError> {
    let err = |field: String, m: String| Err(WorkflowError::new(file, &field, m));
    if !approve.is_empty() && mode != Mode::Route {
        return err(
            "team.approve".into(),
            "approvals work with mode `route`".into(),
        );
    }
    for (i, a) in approve.iter().enumerate() {
        if !members.contains(a) {
            return err(
                format!("team.approve[{i}]"),
                format!("`{a}` is not a member"),
            );
        }
        if approve[..i].contains(a) {
            return err(format!("team.approve[{i}]"), format!("`{a}` repeats"));
        }
    }
    assert!(
        approve.len() <= members.len(),
        "approvals name distinct members"
    );
    Ok(())
}

/// Checks `max_iterations`: only for mode `tasks`, 1 to 100. Returns the
/// value, or the default 5 when it is not set.
fn check_iterations(file: &str, mode: Mode, n: Option<i64>) -> Result<i64, WorkflowError> {
    let field = "team.max_iterations";
    let n = match n {
        None => return Ok(DEFAULT_MAX_ITERATIONS),
        Some(_) if mode != Mode::Tasks => {
            return Err(WorkflowError::new(file, field, "is only for mode `tasks`"))
        }
        Some(n) => n,
    };
    if !MAX_ITERATIONS_RANGE.contains(&n) {
        let (lo, hi) = (MAX_ITERATIONS_RANGE.start(), MAX_ITERATIONS_RANGE.end());
        return Err(WorkflowError::new(
            file,
            field,
            format!("must be between {lo} and {hi}"),
        ));
    }
    assert!(MAX_ITERATIONS_RANGE.contains(&DEFAULT_MAX_ITERATIONS));
    Ok(n)
}

fn check_state(file: &str, state: Table) -> Result<Vec<(String, Value)>, WorkflowError> {
    let mut out = Vec::with_capacity(state.len());
    for (key, ty) in state {
        let field = format!("team.state.{key}");
        if !is_ident(&key) || key.len() > 64 {
            let m = format!("`{key}` must be 1 to 64 of [a-z_][a-z0-9_]*");
            return Err(WorkflowError::new(file, &field, m));
        }
        if !ty.is_str() {
            let m = "must be a type string such as \"string?\"".to_string();
            return Err(WorkflowError::new(file, &field, m));
        }
        out.push((key, ty));
    }
    out.sort_by(|a, b| a.0.cmp(&b.0));
    assert!(
        out.windows(2).all(|w| w[0].0 < w[1].0),
        "table keys are distinct, so sorted keys are strictly ordered"
    );
    Ok(out)
}

/// Starts the team's flow table at `start` with the team state declared.
/// Returns the table and the keys every member reads: `input`, then the
/// team state keys in key order.
fn start(file: &str, team: &Team, start: &str) -> Result<(FlowTable, Vec<String>), WorkflowError> {
    let mut t = FlowTable::new(file, &team.name, start);
    let mut reads = vec!["input".to_string()];
    for (key, ty) in &team.state {
        t.state(&format!("team.state.{key}"), key, ty.clone())?;
        reads.push(key.clone());
    }
    assert_eq!(
        reads.len(),
        team.state.len() + 1,
        "members read input and team state"
    );
    Ok((t, reads))
}

fn route(file: &str, team: &Team) -> Result<Table, WorkflowError> {
    let lead = node_id(&team.leader);
    let (mut t, reads) = start(file, team, &lead)?;
    let targets: Vec<String> = team
        .members
        .iter()
        .map(|m| {
            if team.approve.contains(m) {
                format!("approve_{}", node_id(m))
            } else {
                node_id(m)
            }
        })
        .collect();
    let mut leader = agent(&lead, &team.leader, Some(ROUTE_PROMPT), &reads, None);
    leader.insert("goto".into(), strings(&targets));
    t.node("team.leader", leader)?;
    for (i, m) in team.members.iter().enumerate() {
        let id = node_id(m);
        let node = agent(&id, m, None, &reads, Some("output"));
        t.node(&format!("team.members[{i}]"), node)?;
        t.edge(&id, END);
    }
    for (i, m) in team.approve.iter().enumerate() {
        approval(&mut t, &format!("team.approve[{i}]"), m)?;
    }
    t.end("team")?;
    assert_eq!(
        targets.len(),
        team.members.len(),
        "one goto target per member"
    );
    Ok(t.finish())
}

/// Puts a human approval in front of member `member`: the approval writes
/// `<id>_approved`, a switch on it continues to the member on `true` and
/// ends the run with no member answer on `false`.
fn approval(t: &mut FlowTable, field: &str, member: &str) -> Result<(), WorkflowError> {
    let id = node_id(member);
    let (gate, check, key) = (
        format!("approve_{id}"),
        format!("check_{id}"),
        format!("{id}_approved"),
    );
    t.state(field, &key, Value::String("bool".into()))?;
    let mut human = Table::new();
    human.insert("id".into(), Value::String(gate.clone()));
    human.insert("kind".into(), Value::String("human".into()));
    human.insert("ask".into(), Value::String("approval".into()));
    let prompt = format!("Approve handing the request to `{member}`?");
    human.insert("prompt".into(), Value::String(prompt));
    human.insert("writes".into(), Value::String(key.clone()));
    t.node(field, human)?;
    let mut switch = Table::new();
    switch.insert("id".into(), Value::String(check.clone()));
    switch.insert("kind".into(), Value::String("switch".into()));
    switch.insert("on".into(), Value::String(key));
    t.node(field, switch)?;
    t.edge(&gate, &check);
    t.routed(&check, &id, "true");
    t.routed(&check, END, "false");
    Ok(())
}

fn coordinate(file: &str, team: &Team) -> Result<Table, WorkflowError> {
    let lead = node_id(&team.leader);
    let (mut t, reads) = start(file, team, &lead)?;
    let mut leader = agent(
        &lead,
        &team.leader,
        Some(COORDINATE_PROMPT),
        &reads,
        Some("output"),
    );
    leader.insert("delegates".into(), strings(&team.members));
    t.node("team.leader", leader)?;
    t.edge(&lead, END);
    t.end("team")?;
    assert!(
        !team.members.is_empty(),
        "a coordinating leader has delegates"
    );
    Ok(t.finish())
}

/// Compiles `tasks` as `max_iterations` leader passes in a row with no
/// edge back, so the pass count is bounded by the flow itself. Each pass
/// delegates to the members, reads the open `tasks` and every earlier
/// `results`, and sets `done`, `tasks`, `results` (extended) and `output`.
/// A switch on `done` ends the run or enters the next pass; the last pass
/// ends it, leaving `done` false when the leader is not done.
fn tasks(file: &str, team: &Team) -> Result<Table, WorkflowError> {
    let lead = node_id(&team.leader);
    let (mut t, mut reads) = start(file, team, &lead)?;
    assert!(MAX_ITERATIONS_RANGE.contains(&team.max_iterations));
    t.max_steps((team.max_iterations + 1) * STEPS_PER_ITERATION);
    t.state("team.mode", "done", Value::String("bool".into()))?;
    t.state("team.mode", "tasks", Value::String("list?".into()))?;
    t.state("team.mode", "results", Value::String("list?".into()))?;
    t.reducer("results", "extend");
    reads.insert(1, "tasks".to_string());
    reads.insert(2, "results".to_string());
    let mut fields = Table::new();
    fields.insert("done".into(), Value::String("bool".into()));
    fields.insert("tasks".into(), Value::String("list<string>".into()));
    fields.insert("results".into(), Value::String("list<string>".into()));
    fields.insert("answer".into(), Value::String("string".into()));
    let mut sets = Table::new();
    sets.insert("done".into(), Value::String("done".into()));
    sets.insert("tasks".into(), Value::String("tasks".into()));
    sets.insert("results".into(), Value::String("results".into()));
    sets.insert("answer".into(), Value::String("output".into()));
    let passes = usize::try_from(team.max_iterations).expect("max_iterations is in range");
    let ids: Vec<String> = (1..=passes)
        .map(|k| match k {
            1 => lead.clone(),
            _ => format!("{lead}_pass_{k}"),
        })
        .collect();
    for (k, id) in ids.iter().enumerate() {
        let mut leader = agent(id, &team.leader, Some(TASKS_PROMPT), &reads, None);
        leader.insert("delegates".into(), strings(&team.members));
        leader.insert("fields".into(), Value::Table(fields.clone()));
        leader.insert("sets".into(), Value::Table(sets.clone()));
        t.node("team.leader", leader)?;
        let Some(next) = ids.get(k + 1) else {
            t.edge(id, END);
            continue;
        };
        let check = format!("check_pass_{}", k + 1);
        let mut switch = Table::new();
        switch.insert("id".into(), Value::String(check.clone()));
        switch.insert("kind".into(), Value::String("switch".into()));
        switch.insert("on".into(), Value::String("done".into()));
        t.node("team.mode", switch)?;
        t.edge(id, &check);
        t.routed(&check, END, "true");
        t.routed(&check, next, "false");
    }
    t.end("team")?;
    assert_eq!(ids.len(), passes, "one leader node per pass");
    assert_eq!(ids[0], lead, "the first pass is the start node");
    assert_eq!(reads[1], "tasks", "the leader reads the open tasks");
    assert_eq!(reads[2], "results", "the leader reads earlier results");
    Ok(t.finish())
}

/// Compiles `collaborate` and `broadcast`. The flow needs a node to fan out
/// from, so the leader first writes a `brief`; every member then answers
/// in its own branch. `collaborate` writes `<id>_answer` per member and the
/// leader combines them into `output`; `broadcast` appends every answer to
/// `answers` in member order and ends at the join.
fn fan_out(file: &str, team: &Team) -> Result<Table, WorkflowError> {
    let (mut t, reads) = start(file, team, "brief")?;
    t.state("team", "brief", Value::String("string".into()))?;
    let brief = agent(
        "brief",
        &team.leader,
        Some(BRIEF_PROMPT),
        &reads,
        Some("brief"),
    );
    t.node("team.leader", brief)?;
    let mut member_reads = reads.clone();
    member_reads.insert(1, "brief".to_string());
    let broadcast = team.mode == Mode::Broadcast;
    if broadcast {
        t.state("team", "answers", Value::String("list".into()))?;
        t.reducer("answers", "append");
    }
    let mut answers = vec!["input".to_string()];
    for (i, m) in team.members.iter().enumerate() {
        let field = format!("team.members[{i}]");
        let id = node_id(m);
        let key = if broadcast {
            "answers".to_string()
        } else {
            format!("{id}_answer")
        };
        if !broadcast {
            t.state(&field, &key, Value::String("string".into()))?;
            answers.push(key.clone());
        }
        t.node(&field, agent(&id, m, None, &member_reads, Some(&key)))?;
        t.edge("brief", &id);
        t.edge(&id, "gather");
    }
    let mut join = Table::new();
    join.insert("id".into(), Value::String("gather".into()));
    join.insert("kind".into(), Value::String("join".into()));
    t.node("team", join)?;
    if broadcast {
        t.edge("gather", END);
    } else {
        let lead = agent(
            "synthesize",
            &team.leader,
            Some(SYNTHESIZE_PROMPT),
            &answers,
            Some("output"),
        );
        t.node("team.leader", lead)?;
        t.edge("gather", "synthesize");
        t.edge("synthesize", END);
    }
    t.end("team")?;
    assert!(
        team.members.len() >= 2,
        "a fan-out has two or more branches"
    );
    Ok(t.finish())
}
