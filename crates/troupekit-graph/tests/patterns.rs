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
    let edges: Vec<(&str, &str, Option<&str>)> = table["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            let e = e.as_table().unwrap();
            let when = e.get("when").and_then(Value::as_str);
            (e["from"].as_str().unwrap(), e["to"].as_str().unwrap(), when)
        })
        .collect();
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
