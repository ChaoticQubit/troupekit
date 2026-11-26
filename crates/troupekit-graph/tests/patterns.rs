use toml::{Table, Value};
use troupekit_graph::patterns::compile;
use troupekit_graph::WorkflowError;

const FILE: &str = "test/team.toml";

const ROUTE: &str = "name = 'triage'\n[team]\nmode = 'route'\nleader = 'triage-lead'\n\
                     members = ['billing', 'tech', 'sales']\napprove = ['billing']\n\
                     [team.state]\ncustomer = 'string?'\n";

const COORDINATE: &str =
    "name = 'desk'\n[team]\nmode = 'coordinate'\nleader = 'lead'\nmembers = ['billing', 'tech']\n";

fn golden(name: &str) -> Table {
    let path = format!("{}/tests/fixtures/patterns/{name}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap();
    toml::from_str(&text).unwrap()
}

fn refused(text: &str, field: &str, part: &str) -> WorkflowError {
    let e = compile(FILE, text).expect_err(&format!("expected `{field}` error for:\n{text}"));
    assert_eq!(e.file, FILE, "{e}");
    assert_eq!(e.field, field, "{e}");
    assert!(e.message.contains(part), "`{part}` not in: {e}");
    e
}

fn nodes(table: &Table) -> &Vec<Value> {
    table["nodes"].as_array().unwrap()
}

fn node<'a>(table: &'a Table, id: &str) -> &'a Table {
    nodes(table)
        .iter()
        .filter_map(Value::as_table)
        .find(|n| n["id"].as_str() == Some(id))
        .unwrap_or_else(|| panic!("no node `{id}`"))
}

fn edge_list(table: &Table) -> Vec<(&str, &str, Option<&str>)> {
    table["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let e = e.as_table().unwrap();
            let when = e.get("when").and_then(Value::as_str);
            (e["from"].as_str().unwrap(), e["to"].as_str().unwrap(), when)
        })
        .collect()
}

fn strs(v: &Value) -> Vec<&str> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| s.as_str().unwrap())
        .collect()
}

#[test]
fn route_compiles_to_goto_leader_and_members() {
    let table = compile(FILE, ROUTE).unwrap();
    assert_eq!(table, golden("team_route.toml"));
    let lead = node(&table, "triage_lead");
    assert_eq!(lead["persona"].as_str(), Some("triage-lead"));
    assert_eq!(strs(&lead["goto"]), ["approve_billing", "tech", "sales"]);
    assert!(lead.get("writes").is_none(), "a router leader writes nothing");
    for id in ["billing", "tech", "sales"] {
        assert_eq!(node(&table, id)["writes"].as_str(), Some("output"));
    }
}

#[test]
fn coordinate_compiles_to_leader_with_delegates() {
    let table = compile(FILE, COORDINATE).unwrap();
    assert_eq!(table, golden("team_coordinate.toml"));
    let lead = node(&table, "lead");
    assert_eq!(strs(&lead["delegates"]), ["billing", "tech"]);
    assert_eq!(lead["writes"].as_str(), Some("output"));
    assert_eq!(nodes(&table).len(), 2, "members are delegates, not nodes");
}

const PANEL: &str = "name = 'panel'\n[team]\nmode = 'collaborate'\nleader = 'chair'\n\
                     members = ['law', 'tax']\n";

#[test]
fn collaborate_compiles_to_fan_out_join_and_synthesis() {
    let table = compile(FILE, PANEL).unwrap();
    assert_eq!(table, golden("team_collaborate.toml"));
    assert_eq!(node(&table, "brief")["persona"].as_str(), Some("chair"));
    assert_eq!(node(&table, "gather")["kind"].as_str(), Some("join"));
    let synth = node(&table, "synthesize");
    assert_eq!(strs(&synth["reads"]), ["input", "law_answer", "tax_answer"]);
    assert_eq!(synth["writes"].as_str(), Some("output"));
    refused(
        &PANEL.replace("['law', 'tax']", "['law']"),
        "team.members",
        "2 or more members",
    );
}

#[test]
fn broadcast_compiles_to_fan_out_and_merge() {
    let table = compile(FILE, &PANEL.replace("collaborate", "broadcast")).unwrap();
    assert_eq!(table, golden("team_broadcast.toml"));
    assert_eq!(table["reducers"]["answers"].as_str(), Some("append"));
    for id in ["law", "tax"] {
        assert_eq!(node(&table, id)["writes"].as_str(), Some("answers"));
    }
    assert!(nodes(&table).iter().all(|n| n["id"].as_str() != Some("synthesize")));
}

#[test]
fn team_state_keys_reach_every_member() {
    let text = ROUTE.replace(
        "customer = 'string?'\n",
        "customer = 'string?'\nplan = 'enum(free, pro)?'\n",
    );
    let table = compile(FILE, &text).unwrap();
    let state = table["state"].as_table().unwrap();
    assert_eq!(state["plan"].as_str(), Some("enum(free, pro)?"));
    for id in ["triage_lead", "billing", "tech", "sales"] {
        assert_eq!(strs(&node(&table, id)["reads"]), ["input", "customer", "plan"], "{id}");
    }
    let coordinated = compile(FILE, &format!("{COORDINATE}[team.state]\nplan = 'string?'\n"));
    let coordinated = coordinated.unwrap();
    assert_eq!(strs(&node(&coordinated, "lead")["reads"]), ["input", "plan"]);
    refused(
        &format!("{COORDINATE}[team.state]\noutput = 'string'\n"),
        "team.state.output",
        "already used",
    );
    refused(
        &format!("{COORDINATE}[team.state]\nBad = 'string'\n"),
        "team.state.Bad",
        "[a-z_][a-z0-9_]*",
    );
    refused(
        &format!("{COORDINATE}[team.state]\nplan = 3\n"),
        "team.state.plan",
        "type string",
    );
}

#[test]
fn approve_inserts_a_human_approval_before_member_answer() {
    let table = compile(FILE, ROUTE).unwrap();
    let gate = node(&table, "approve_billing");
    assert_eq!(gate["kind"].as_str(), Some("human"));
    assert_eq!(gate["ask"].as_str(), Some("approval"));
    assert_eq!(gate["writes"].as_str(), Some("billing_approved"));
    let check = node(&table, "check_billing");
    assert_eq!(check["kind"].as_str(), Some("switch"));
    assert_eq!(check["on"].as_str(), Some("billing_approved"));
    let edges = edge_list(&table);
    assert!(edges.contains(&("approve_billing", "check_billing", None)));
    assert!(edges.contains(&("check_billing", "billing", Some("true"))));
    assert!(edges.contains(&("check_billing", "end", Some("false"))));
    assert!(!edges.iter().any(|e| e.1 == "billing" && e.0 != "check_billing"));
    refused(
        &ROUTE.replace("approve = ['billing']", "approve = ['nobody']"),
        "team.approve[0]",
        "not a member",
    );
    refused(
        &format!("{COORDINATE}approve = ['billing']\n"),
        "team.approve",
        "mode `route`",
    );
}

#[test]
fn team_with_unknown_mode_is_refused() {
    let e = refused(
        &COORDINATE.replace("coordinate", "anarchy"),
        "team.mode",
        "`anarchy` must be one of route, coordinate, collaborate, broadcast, tasks",
    );
    assert!(!e.message.is_empty());
    refused(&format!("{COORDINATE}colour = 'red'\n"), "colour", "unknown field");
    refused(
        &COORDINATE.replace("members = ['billing', 'tech']", "members = []"),
        "team.members",
        "1 to 32",
    );
    refused(
        &COORDINATE.replace("['billing', 'tech']", "['billing', 'billing']"),
        "team.members[1]",
        "repeats",
    );
    refused(
        &COORDINATE.replace("['billing', 'tech']", "['billing', 'lead']"),
        "team.members[1]",
        "is the leader",
    );
    refused(
        &COORDINATE.replace("leader = 'lead'", "leader = 'Lead'"),
        "team.leader",
        "[a-z0-9-]",
    );
    refused(
        &format!("{COORDINATE}max_iterations = 3\n"),
        "team.max_iterations",
        "only for mode `tasks`",
    );
}

#[test]
fn persona_names_map_to_node_ids_and_collisions_are_refused() {
    let text = ROUTE.replace("['billing', 'tech', 'sales']", "['2fa', 'tech', 'sales']");
    let text = text.replace("approve = ['billing']", "approve = []");
    let table = compile(FILE, &text).unwrap();
    assert_eq!(node(&table, "p_2fa")["persona"].as_str(), Some("2fa"));
    refused(
        &text.replace("['2fa', 'tech', 'sales']", "['1', 'p-1']"),
        "team.members[1]",
        "node id `p_1` is already used",
    );
    refused(
        &COORDINATE.replace("leader = 'lead'", "leader = 'end'"),
        "team",
        "node id `end` is already used",
    );
}

#[test]
fn pattern_text_with_toml_syntax_stays_text() {
    let hostile = "string\"\n[[nodes]]\nid = \"evil\"\nkind = \"end\"\n";
    let text = format!("{COORDINATE}[team.state]\nnote = '''{hostile}'''\n");
    let table = compile(FILE, &text).unwrap();
    let round: Table = toml::from_str(&toml::to_string(&table).unwrap()).unwrap();
    assert_eq!(round, table);
    assert_eq!(nodes(&round).len(), 2, "no node is injected");
    assert_eq!(round["state"]["note"].as_str(), Some(hostile));
}

#[test]
fn load_refuses_two_patterns_in_one_file() {
    refused(
        &format!("{COORDINATE}[crew]\nprocess = 'sequential'\n"),
        "crew",
        "found `team` and `crew`",
    );
    refused("name = 'x'\n", "", "one of nodes, team, crew, handoff or steps");
    let flow = "name = 'f'\nstart = 'e'\n[[nodes]]\nid = 'e'\nkind = 'end'\n";
    assert_eq!(compile(FILE, flow).unwrap(), toml::from_str::<Table>(flow).unwrap());
}

#[test]
fn defaults_are_the_safe_values() {
    let table = compile(FILE, COORDINATE).unwrap();
    let state = table["state"].as_table().unwrap();
    assert_eq!(state.len(), 2, "only input and output");
    assert_eq!(state["input"].as_str(), Some("string"));
    assert_eq!(state["output"].as_str(), Some("string"));
    let route = compile(FILE, &ROUTE.replace("approve = ['billing']\n", "")).unwrap();
    assert!(nodes(&route)
        .iter()
        .all(|n| n["kind"].as_str() != Some("human")));
    assert!(table.get("max_steps").is_none(), "the workflow step cap applies");
}

const REPORT: &str = "name = 'report'\n[crew]\nprocess = 'sequential'\n\
    [crew.state]\nneeds_report = 'bool'\n\
    [[crew.tasks]]\nid = 'research'\nagent = 'researcher'\n\
    description = 'Collect facts about {input}.'\nexpected_output = 'A list of facts.'\n\
    [[crew.tasks]]\nid = 'fact_check'\nagent = 'checker'\ndescription = 'Check the facts.'\n\
    context = ['research']\nasync = true\n\
    [[crew.tasks]]\nid = 'write'\nagent = 'writer'\ndescription = 'Write the report.'\n\
    context = ['research', 'fact_check']\nhuman_input = true\nwhen = 'needs_report'\n";

fn crew(tasks: &[&str]) -> String {
    let mut text = "name = 'c'\n[crew]\nprocess = 'sequential'\n".to_string();
    for t in tasks {
        text.push_str("[[crew.tasks]]\n");
        text.push_str(t);
        text.push('\n');
    }
    text
}

fn task(id: &str, extra: &str) -> String {
    format!("id = '{id}'\nagent = 'a-{id}'\ndescription = 'Do {id}.'\n{extra}")
}

#[test]
fn sequential_crew_chains_tasks_in_order() {
    let table = compile(FILE, &crew(&[&task("a", ""), &task("b", ""), &task("c", "")])).unwrap();
    assert_eq!(table["start"].as_str(), Some("a"));
    assert_eq!(
        edge_list(&table),
        [("a", "b", None), ("b", "c", None), ("c", "end", None)]
    );
    assert_eq!(node(&table, "a")["writes"].as_str(), Some("a"));
    assert_eq!(node(&table, "b")["writes"].as_str(), Some("b"));
    assert_eq!(node(&table, "c")["writes"].as_str(), Some("output"));
    assert_eq!(node(&table, "b")["persona"].as_str(), Some("a-b"));
    assert_eq!(node(&table, "b")["prompt"].as_str(), Some("Do b."));
    assert_eq!(table["state"]["a"].as_str(), Some("string?"));
}

#[test]
fn task_context_feeds_named_outputs_only() {
    let text = crew(&[
        &task("a", ""),
        &task("b", ""),
        &task("c", "context = ['a']\nexpected_output = 'One line.'"),
    ]);
    let table = compile(FILE, &text).unwrap();
    assert_eq!(strs(&node(&table, "a")["reads"]), ["input"]);
    assert_eq!(strs(&node(&table, "b")["reads"]), ["input"]);
    assert_eq!(strs(&node(&table, "c")["reads"]), ["input", "a"]);
    let prompt = node(&table, "c")["prompt"].as_str().unwrap();
    assert_eq!(prompt, "Do c.\n\nExpected output: One line.");
}

#[test]
fn async_tasks_compile_to_a_fan_out_joined_before_first_reader() {
    let text = crew(&[
        &task("a", ""),
        &task("check", "context = ['a']\nasync = true"),
        &task("outline", "context = ['a']"),
        &task("write", "context = ['check', 'outline']"),
    ]);
    let table = compile(FILE, &text).unwrap();
    assert_eq!(node(&table, "join_check")["kind"].as_str(), Some("join"));
    assert_eq!(
        edge_list(&table),
        [
            ("a", "check", None),
            ("check", "join_check", None),
            ("a", "outline", None),
            ("outline", "join_check", None),
            ("join_check", "write", None),
            ("write", "end", None),
        ]
    );
    let unread = crew(&[&task("a", ""), &task("b", "async = true"), &task("c", "")]);
    let unread = compile(FILE, &unread).unwrap();
    let edges = edge_list(&unread);
    assert!(edges.contains(&("c", "join_b", None)));
    assert!(edges.contains(&("join_b", "end", None)));
    refused(
        &crew(&[&task("a", "async = true"), &task("b", "")]),
        "crew.tasks[0].async",
        "nothing to run beside",
    );
    refused(
        &crew(&[
            &task("a", ""),
            &task("b", "async = true"),
            &task("c", "async = true"),
            &task("d", "context = ['b', 'c']"),
        ]),
        "crew.tasks[2].async",
        "inside another async span",
    );
    assert_eq!(compile(FILE, REPORT).unwrap(), golden("crew_sequential.toml"));
}

#[test]
fn conditional_task_compiles_to_switch() {
    let text = crew(&[&task("a", ""), &task("b", "when = 'go'"), &task("c", "")]).replacen(
        "[[crew.tasks]]",
        "[crew.state]\ngo = 'bool'\n[[crew.tasks]]",
        1,
    );
    let table = compile(FILE, &text).unwrap();
    let switch = node(&table, "when_b");
    assert_eq!(switch["kind"].as_str(), Some("switch"));
    assert_eq!(switch["on"].as_str(), Some("go"));
    let edges = edge_list(&table);
    assert!(edges.contains(&("a", "when_b", None)));
    assert!(edges.contains(&("when_b", "b", Some("true"))));
    assert!(edges.contains(&("when_b", "c", Some("false"))));
    assert!(edges.contains(&("b", "c", None)));
    refused(
        &crew(&[&task("a", ""), &task("b", "when = 'nope'")]),
        "crew.tasks[1].when",
        "not declared in [crew.state]",
    );
}

#[test]
fn human_input_task_keeps_an_approval_node() {
    let text = crew(&[&task("a", "human_input = true"), &task("b", "")]);
    let table = compile(FILE, &text).unwrap();
    let review = node(&table, "review_a");
    assert_eq!(review["kind"].as_str(), Some("human"));
    assert_eq!(review["ask"].as_str(), Some("question"));
    assert_eq!(strs(&review["reads"]), ["a"]);
    assert_eq!(review["writes"].as_str(), Some("a_feedback"));
    assert_eq!(strs(&node(&table, "b")["reads"]), ["input", "a_feedback"]);
    let edges = edge_list(&table);
    assert!(edges.contains(&("a", "review_a", None)));
    assert!(edges.contains(&("review_a", "b", None)));
}

const HIERARCHY: &str = "name = 'report'\n[crew]\nprocess = 'hierarchical'\n\
    manager_model = 'openai/gpt-6-luna'\n\
    [[crew.tasks]]\nid = 'research'\nagent = 'researcher'\n\
    description = 'Collect facts about {input}.'\nexpected_output = 'A list of facts.'\n\
    [[crew.tasks]]\nid = 'fact_check'\nagent = 'checker'\ndescription = 'Check the facts.'\n\
    [[crew.tasks]]\nid = 'write'\nagent = 'writer'\ndescription = 'Write the report.'\n";

#[test]
fn hierarchical_crew_uses_builtin_manager_with_members_as_delegates() {
    let table = compile(FILE, HIERARCHY).unwrap();
    assert_eq!(table, golden("crew_hierarchical.toml"));
    let manager = node(&table, "manager");
    assert_eq!(manager["persona"].as_str(), Some("crew-manager"));
    assert_eq!(strs(&manager["delegates"]), ["researcher", "checker", "writer"]);
    let async_task = HIERARCHY.replace("description = 'Check the facts.'\n", "description = 'Check the facts.'\nasync = true\n");
    refused(&async_task, "crew.tasks[1]", "only for process `sequential`");
}

#[test]
fn custom_manager_persona_replaces_builtin() {
    let text = HIERARCHY.replace("manager_model = 'openai/gpt-6-luna'", "manager = 'editor'");
    let table = compile(FILE, &text).unwrap();
    let manager = node(&table, "manager");
    assert_eq!(manager["persona"].as_str(), Some("editor"));
    assert!(manager.get("model").is_none());
    refused(
        &crew(&[&task("a", "")]).replace("process = 'sequential'", "process = 'sequential'\nmanager = 'editor'"),
        "crew.manager",
        "only for process `hierarchical`",
    );
    refused(
        &HIERARCHY.replace("hierarchical", "chaotic"),
        "crew.process",
        "`chaotic` must be `sequential` or `hierarchical`",
    );
}

#[test]
fn context_naming_a_later_task_is_refused() {
    refused(
        &crew(&[&task("a", "context = ['b']"), &task("b", "")]),
        "crew.tasks[0].context[0]",
        "not an earlier task",
    );
    refused(
        &crew(&[&task("a", "context = ['a']")]),
        "crew.tasks[0].context[0]",
        "not an earlier task",
    );
    refused(
        &crew(&[&task("a", ""), &task("b", "context = ['zz']")]),
        "crew.tasks[1].context[0]",
        "`zz` is not a task",
    );
    refused(
        &crew(&[&task("a", ""), &task("a", "")]),
        "crew.tasks[1].id",
        "repeats",
    );
    refused(&crew(&[&task("output", "")]), "crew.tasks[0].id", "must be 1 to 64");
}
