use serde_json::{json, Map, Value};
use troupekit_graph::{Ask, Conditions, Flow, NodeKind, Workflow, WorkflowError};

pub(crate) const FULL: &str = r#"
name = "plan-week"
start = "plan"
max_steps = 50
max_retries = 3

[state]
goal = "string"
plan = "{steps: list}"
approved = "bool"
note = "string"

[[nodes]]
id = "plan"
kind = "agent"
persona = "planner"
prompt = "Plan the week."
reads = ["goal"]
writes = "plan"

[[nodes]]
id = "approve"
kind = "human"
ask = "approval"
prompt = "Approve this plan?"
reads = ["plan"]
writes = "approved"

[[nodes]]
id = "route"
kind = "router"
condition = "flag(approved)"

[[nodes]]
id = "save"
kind = "tool"
tool = "save_plan"
reads = ["plan"]
writes = "note"

[[nodes]]
id = "done"
kind = "end"

[[edges]]
from = "plan"
to = "approve"

[[edges]]
from = "approve"
to = "route"

[[edges]]
from = "route"
to = "save"
when = "true"

[[edges]]
from = "route"
to = "plan"
when = "false"

[[edges]]
from = "save"
to = "done"
"#;

const HEAD: &str =
    "name = 'w'\nstart = 'a'\n[state]\nx = 'string'\ny = 'string'\nok = 'bool'\nn = 'int'\n";

fn parse(text: &str) -> Result<Workflow, WorkflowError> {
    let wf = Workflow::parse("test/w.toml", text);
    let flow = Flow::parse("test/w.toml", text);
    match (&wf, &flow) {
        (Ok(w), Ok(f)) => assert_eq!(&Flow::from(w.clone()), f, "{text}"),
        (Err(a), Err(b)) => assert_eq!(a, b),
        _ => panic!("parsers disagree on:\n{text}\nworkflow: {wf:?}\nflow: {flow:?}"),
    }
    wf
}

fn nodes(list: &[&str]) -> String {
    list.iter()
        .map(|n| format!("[[nodes]]\n{n}\n"))
        .collect::<String>()
}

fn edges(list: &[(&str, &str)]) -> String {
    list.iter()
        .map(|(f, t)| format!("[[edges]]\nfrom = '{f}'\nto = '{t}'\n"))
        .collect::<String>()
}

fn routed(from: &str, to: &str, when: &str) -> String {
    format!("[[edges]]\nfrom = '{from}'\nto = '{to}'\nwhen = '{when}'\n")
}

const TOOL_A: &str = "id = 'a'\nkind = 'tool'\ntool = 't'";
const END: &str = "id = 'e'\nkind = 'end'";

fn field_of(text: &str) -> String {
    parse(text).expect_err("workflow must be rejected").field
}

#[test]
fn parses_full_workflow() {
    let wf = parse(FULL).unwrap();
    assert_eq!(wf.name, "plan-week");
    assert_eq!(wf.start, "plan");
    assert_eq!((wf.max_steps, wf.max_retries), (50, 3));
    assert_eq!(wf.state["approved"], json!({"type": "boolean"}));
    assert_eq!(wf.state["plan"]["properties"]["steps"]["type"], "array");
    let plan = wf.node("plan").unwrap();
    assert_eq!(
        plan.kind,
        NodeKind::Agent {
            persona: "planner".into(),
            prompt: Some("Plan the week.".into())
        }
    );
    assert_eq!(plan.reads, vec!["goal".to_string()]);
    assert_eq!(plan.writes.as_deref(), Some("plan"));
    assert!(matches!(
        wf.node("approve").unwrap().kind,
        NodeKind::Human {
            ask: Ask::Approval,
            ..
        }
    ));
    let NodeKind::Router { condition } = &wf.node("route").unwrap().kind else {
        panic!("route is a router");
    };
    assert_eq!(condition.name, "flag");
    assert_eq!(wf.next("route").len(), 2);
    assert_eq!(wf.in_degree("plan"), 1);
    let back: Workflow = serde_json::from_value(serde_json::to_value(&wf).unwrap()).unwrap();
    assert_eq!(back, wf);
    let defaults = parse(&format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, END]),
        edges(&[("a", "e")])
    ))
    .unwrap();
    assert_eq!((defaults.max_steps, defaults.max_retries), (200, 5));
}

#[test]
fn rejects_unknown_node() {
    let text = format!("{HEAD}{}{}", nodes(&[TOOL_A, END]), edges(&[("a", "zz")]));
    assert_eq!(field_of(&text), "edges[0].to");
    let text = format!("{HEAD}{}{}", nodes(&[TOOL_A, END]), edges(&[("zz", "e")]));
    assert_eq!(field_of(&text), "edges[0].from");
    let text = format!(
        "{}{}{}",
        HEAD.replace("start = 'a'", "start = 'q'"),
        nodes(&[TOOL_A, END]),
        edges(&[("a", "e")])
    );
    assert_eq!(field_of(&text), "start");
}

#[test]
fn rejects_unreachable_node() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let text = format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, b, END]),
        edges(&[("a", "e"), ("b", "e")])
    );
    let e = parse(&text).unwrap_err();
    assert_eq!(e.field, "nodes.b");
    assert!(e.message.contains("unreachable"), "{e}");
}

#[test]
fn rejects_cycle_without_router() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let r = "id = 'r'\nkind = 'router'\ncondition = 'flag(ok)'";
    let text = format!(
        "{HEAD}{}{}{}{}",
        nodes(&[TOOL_A, b, r, END]),
        edges(&[("a", "b"), ("b", "a")]),
        routed("r", "e", "true"),
        routed("r", "a", "false")
    );
    let text = text.replace("start = 'a'", "start = 'r'");
    let e = parse(&text).unwrap_err();
    assert!(e.message.contains("cycle without a router"), "{e}");
}

#[test]
fn accepts_cycle_through_router() {
    let r = "id = 'r'\nkind = 'router'\ncondition = 'flag(ok)'";
    let text = format!(
        "{HEAD}{}{}{}{}",
        nodes(&[TOOL_A, r, END]),
        edges(&[("a", "r")]),
        routed("r", "a", "false"),
        routed("r", "e", "true")
    );
    let wf = parse(&text).unwrap();
    assert_eq!(wf.next("r").len(), 2);
}

#[test]
fn rejects_missing_join() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'";
    let j = "id = 'j'\nkind = 'join'";
    let k = "id = 'k'\nkind = 'join'";
    let ok = format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, b, c, j, END]),
        edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "j"), ("j", "e")])
    );
    parse(&ok).unwrap();
    let direct = format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, b, j, END]),
        edges(&[("a", "b"), ("a", "j"), ("b", "j"), ("j", "e")])
    );
    parse(&direct).unwrap();
    let to_end = format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, b, c, j, END]),
        edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "e"), ("j", "e")])
    );
    let e = parse(&to_end).unwrap_err();
    assert_eq!(e.field, "nodes.a", "{e}");
    assert!(e.message.contains("before a join"), "{e}");
    let two_joins = format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, b, c, j, k, END]),
        edges(&[
            ("a", "b"),
            ("a", "c"),
            ("b", "j"),
            ("c", "k"),
            ("j", "e"),
            ("k", "e")
        ])
    );
    assert!(parse(&two_joins)
        .unwrap_err()
        .message
        .contains("different joins"));
    let r = "id = 'r'\nkind = 'router'\ncondition = 'flag(ok)'";
    let outside = format!(
        "{}{}{}{}{}",
        HEAD.replace("start = 'a'", "start = 'r'"),
        nodes(&[r, TOOL_A, b, c, j, END]),
        routed("r", "a", "true"),
        routed("r", "j", "false"),
        edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "j"), ("j", "e")])
    );
    let e = parse(&outside).unwrap_err();
    assert_eq!(e.field, "nodes.j", "{e}");
    assert!(e.message.contains("outside the fan-out at `a`"), "{e}");
    let loop_in = format!(
        "{HEAD}{}{}{}{}",
        nodes(&[TOOL_A, b, c, j, r, END]),
        edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "j"), ("j", "r")]),
        routed("r", "b", "false"),
        routed("r", "e", "true")
    );
    assert!(parse(&loop_in)
        .unwrap_err()
        .message
        .contains("before a join"));
    let no_fan_out = format!(
        "{HEAD}{}{}",
        nodes(&[TOOL_A, j, END]),
        edges(&[("a", "j"), ("j", "e")])
    );
    let e = parse(&no_fan_out).unwrap_err();
    assert_eq!(e.field, "nodes.j");
    assert!(e.message.contains("exactly one fan-out"), "{e}");
}

#[test]
fn rejects_bad_fields() {
    let ok_tail = format!("{}{}", nodes(&[END]), edges(&[("a", "e")]));
    let with_a = |a: &str| format!("{HEAD}{}{ok_tail}", nodes(&[a]));
    let r = "id = 'r'\nkind = 'router'\ncondition = 'flag(ok)'";
    let h = "id = 'h'\nkind = 'human'\nask = 'approval'\nwrites = 'x'";
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'\nwrites = 'x'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'\nwrites = 'x'";
    let j = "id = 'j'\nkind = 'join'";
    let cases: Vec<(String, &str)> = vec![
        (format!("colour = 1\n{}", with_a(TOOL_A)), "colour"),
        (
            with_a("id = 'a'\nkind = 'tool'\ntool = 't'\npersona = 'p'"),
            "nodes.a.persona",
        ),
        (
            with_a("id = 'a'\nkind = 'tool'\ntool = 't'\nreads = ['zz']"),
            "nodes.a.reads",
        ),
        (
            with_a("id = 'a'\nkind = 'tool'\ntool = 't'\nwrites = 'zz'"),
            "nodes.a.writes",
        ),
        (
            with_a("id = 'a'\nkind = 'human'\nask = 'approval'\nwrites = 'x'"),
            "nodes.a.writes",
        ),
        (
            with_a("id = 'a'\nkind = 'human'\nask = 'maybe'\nwrites = 'ok'"),
            "nodes.a.ask",
        ),
        (
            with_a("id = 'a'\nkind = 'agent'\npersona = 'p'"),
            "nodes.a.prompt",
        ),
        (
            with_a("id = 'a'\nkind = 'agent'\npersona = 'Bad'\nprompt = 'x'"),
            "nodes.a.persona",
        ),
        (
            with_a("id = 'a'\nkind = 'tool'\ntool = 'bad tool'"),
            "nodes.a.tool",
        ),
        (
            with_a("id = 'a'\nkind = 'router'\ncondition = 'flag('"),
            "nodes.a.condition",
        ),
        (
            with_a("id = 'a'\nkind = 'router'\ncondition = 'flag(x)'"),
            "nodes.a.condition",
        ),
        (with_a("id = 'a'\nkind = 'wait'"), "nodes.a.kind"),
        (
            with_a("id = 'a'\nkind = 'join'\nreads = ['x']"),
            "nodes.a.reads",
        ),
        (
            format!("{HEAD}{}{}", nodes(&[r, END]), edges(&[("r", "e")]))
                .replace("start = 'a'", "start = 'r'"),
            "edges[0].when",
        ),
        (
            format!("{HEAD}{}{}", nodes(&[TOOL_A, END]), routed("a", "e", "yes")),
            "edges[0].when",
        ),
        (
            format!(
                "{HEAD}{}{}{}{}",
                nodes(&[TOOL_A, r, END]),
                edges(&[("a", "r")]),
                routed("r", "e", "true"),
                routed("r", "a", "true")
            ),
            "edges[2].when",
        ),
        (
            format!(
                "{HEAD}{}{}",
                nodes(&[TOOL_A, END]),
                edges(&[("a", "e"), ("a", "e")])
            ),
            "edges[1]",
        ),
        (
            format!(
                "{HEAD}{}{}",
                nodes(&[TOOL_A, END]),
                edges(&[("a", "e"), ("e", "a")])
            ),
            "nodes.e",
        ),
        (format!("{HEAD}{}", nodes(&[TOOL_A, END])), "nodes.a"),
        (
            format!(
                "{HEAD}{}{}{}",
                nodes(&[TOOL_A, r]),
                edges(&[("a", "r")]),
                routed("r", "a", "true")
            ),
            "nodes",
        ),
        (format!("max_steps = 0\n{}", with_a(TOOL_A)), "max_steps"),
        (
            format!("max_retries = 51\n{}", with_a(TOOL_A)),
            "max_retries",
        ),
        (
            format!("{HEAD}{}{ok_tail}", nodes(&[TOOL_A, TOOL_A])),
            "nodes[1].id",
        ),
        (
            format!("{HEAD}{}{ok_tail}", nodes(&["id = 'A'\nkind = 'end'"])),
            "nodes[0].id",
        ),
        (
            format!(
                "{}{}{ok_tail}",
                HEAD.replace("name = 'w'", "name = 'W'"),
                nodes(&[TOOL_A])
            ),
            "name",
        ),
        (
            format!(
                "{}{}{ok_tail}",
                HEAD.replace("x = 'string'", "x = 'lst'"),
                nodes(&[TOOL_A])
            ),
            "state",
        ),
        (
            format!(
                "{HEAD}{}{}",
                nodes(&[TOOL_A, b, c, j, END]),
                edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "j"), ("j", "e")])
            ),
            "nodes.a",
        ),
        (
            with_a(h).replace("start = 'a'", "start = 'h'"),
            "nodes.h.writes",
        ),
        (format!("{HEAD}id = \"a\n"), "toml"),
    ];
    for (text, field) in cases {
        let got = parse(&text);
        let e = got.expect_err(&format!("expected `{field}` error for:\n{text}"));
        assert_eq!(e.file, "test/w.toml");
        if field != "toml" {
            assert_eq!(e.field, field, "{e}\n{text}");
        }
    }
    let wrong_kind = parse(&with_a("id = 'a'\nkind = 'end'\ntool = 't'")).unwrap_err();
    assert!(
        wrong_kind.message.contains("not a field of a end node"),
        "{wrong_kind}"
    );
}

#[test]
fn flag_condition_routes_on_bool() {
    let c = Conditions::new();
    let flag = c.get("flag").unwrap();
    let mut state = Map::new();
    state.insert("ok".into(), json!(true));
    state.insert("n".into(), json!(3));
    assert_eq!(flag(&state, &[json!("ok")]).unwrap(), "true");
    state.insert("ok".into(), json!(false));
    assert_eq!(flag(&state, &[json!("ok")]).unwrap(), "false");
    assert!(flag(&state, &[json!("n")]).is_err());
    assert!(flag(&state, &[json!("zz")]).is_err());
    assert!(flag(&state, &[]).is_err());
    let mut c = c;
    let always: troupekit_graph::ConditionFn =
        std::sync::Arc::new(|_: &Map<String, Value>, _: &[Value]| Ok("yes".into()));
    c.register("always", always.clone()).unwrap();
    assert!(c.register("always", always.clone()).is_err());
    assert!(c.register("Bad", always).is_err());
    assert!(c.get("always").is_some());
}
