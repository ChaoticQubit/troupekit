use troupekit_graph::{
    BudgetSplit, Flow, FlowKind, JoinMode, NodeKind, Reducer, Workflow, WorkflowError,
};

const HEAD: &str = "name = 'f'\nstart = 'a'\n[state]\nx = 'string'\ny = 'string'\nxs = 'list'\nout = 'list'\nn = 'number'\nobj = '{a: string}'\nnext = 'enum(left, right)'\nok = 'bool'\ntmp = 'string?'\n";

fn parse(text: &str) -> Result<Flow, WorkflowError> {
    Flow::parse("test/f.toml", text)
}

fn nodes(list: &[&str]) -> String {
    list.iter().map(|n| format!("[[nodes]]\n{n}\n")).collect()
}

fn edges(list: &[(&str, &str)]) -> String {
    list.iter()
        .map(|(f, t)| format!("[[edges]]\nfrom = '{f}'\nto = '{t}'\n"))
        .collect()
}

fn routed(from: &str, to: &str, when: &str) -> String {
    format!("[[edges]]\nfrom = '{from}'\nto = '{to}'\nwhen = '{when}'\n")
}

fn refused(text: &str, field: &str, part: &str) {
    let e = parse(text).expect_err(&format!("expected `{field}` error for:\n{text}"));
    assert_eq!(e.file, "test/f.toml", "{e}");
    assert_eq!(e.field, field, "{e}");
    assert!(e.message.contains(part), "`{part}` not in: {e}");
}

fn head(start: &str, extra: &str) -> String {
    format!(
        "{}{extra}",
        HEAD.replace("start = 'a'", &format!("start = '{start}'"))
    )
}

const TOOL_A: &str = "id = 'a'\nkind = 'tool'\ntool = 't'";
const END: &str = "id = 'e'\nkind = 'end'";
const JOIN: &str = "id = 'j'\nkind = 'join'";

fn fan(extra: &str, b: &str, c: &str, join: &str) -> String {
    format!(
        "{}{}{}",
        head("a", extra),
        nodes(&[TOOL_A, b, c, join, END]),
        edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "j"), ("j", "e")])
    )
}

fn nested(depth: usize) -> String {
    fn level(
        i: usize,
        depth: usize,
        ns: &mut Vec<String>,
        es: &mut Vec<(String, String)>,
    ) -> (String, String) {
        let (f, l, j) = (format!("f{i}"), format!("l{i}"), format!("j{i}"));
        ns.push(format!("id = '{f}'\nkind = 'tool'\ntool = 't'"));
        ns.push(format!("id = '{l}'\nkind = 'tool'\ntool = 't'"));
        ns.push(format!("id = '{j}'\nkind = 'join'"));
        let (entry, exit) = if i < depth {
            level(i + 1, depth, ns, es)
        } else {
            let r = format!("r{i}");
            ns.push(format!("id = '{r}'\nkind = 'tool'\ntool = 't'"));
            (r.clone(), r)
        };
        es.extend([
            (f.clone(), l.clone()),
            (f.clone(), entry),
            (l, j.clone()),
            (exit, j.clone()),
        ]);
        (f, j)
    }
    let (mut ns, mut es) = (Vec::new(), Vec::new());
    let (_, exit) = level(1, depth, &mut ns, &mut es);
    ns.push(END.to_string());
    es.push((exit, "e".into()));
    let ns: Vec<&str> = ns.iter().map(String::as_str).collect();
    let es: Vec<(&str, &str)> = es.iter().map(|(a, b)| (a.as_str(), b.as_str())).collect();
    format!("{}{}{}", head("f1", ""), nodes(&ns), edges(&es))
}

#[test]
fn every_v1_workflow_parses_as_an_equal_flow() {
    let texts = [
        crate::workflow::FULL.to_string(),
        fan(
            "",
            "id = 'b'\nkind = 'tool'\ntool = 't'",
            "id = 'c'\nkind = 'tool'\ntool = 't'",
            JOIN,
        ),
    ];
    for text in &texts {
        let wf = Workflow::parse("test/f.toml", text).unwrap();
        let flow = parse(text).unwrap();
        assert_eq!(flow, Flow::from(wf.clone()));
        assert_eq!(flow.workflow(), &wf);
        assert!(flow.is_v1());
    }
}

#[test]
fn same_key_from_two_branches_needs_a_reducer() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'\nwrites = 'n'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'\nwrites = 'n'";
    refused(
        &fan("", b, c, JOIN),
        "nodes.a",
        "two branches of the fan-out write `n`",
    );
    let flow = parse(&fan("[reducers]\nn = 'sum'\n", b, c, JOIN)).unwrap();
    assert_eq!(flow.reducer("n"), Some(Reducer::Sum));
    assert_eq!(flow.reducer("x"), None);
    assert!(!flow.is_v1());
}

#[test]
fn reducer_on_an_undeclared_key_is_refused() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'";
    refused(
        &fan("[reducers]\nzz = 'sum'\n", b, c, JOIN),
        "reducers.zz",
        "not declared",
    );
}

#[test]
fn reducer_type_must_fit_the_key() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'";
    let with = |r: &str| fan(&format!("[reducers]\n{r}\n"), b, c, JOIN);
    refused(&with("y = 'extend'"), "reducers.y", "needs a list key");
    refused(&with("y = 'append'"), "reducers.y", "needs a list key");
    refused(&with("xs = 'sum'"), "reducers.xs", "needs a number key");
    refused(&with("n = 'merge'"), "reducers.n", "needs an object key");
    refused(
        &with("n = 'concat'"),
        "reducers.n",
        "unknown reducer `concat`",
    );
    let flow = parse(&with(
        "xs = 'extend'\nout = 'append'\nobj = 'merge'\nn = 'max'\ny = 'last'",
    ))
    .unwrap();
    assert_eq!(flow.reducer("xs"), Some(Reducer::Extend));
    assert_eq!(flow.reducer("out"), Some(Reducer::Append));
    assert_eq!(flow.reducer("obj"), Some(Reducer::Merge));
    assert_eq!(flow.reducer("n"), Some(Reducer::Max));
    assert_eq!(flow.reducer("y"), Some(Reducer::Last));
}

fn map_flow(map: &str, body: &str, extra_edges: &str) -> String {
    format!(
        "{}{}{}{extra_edges}",
        head("m", ""),
        nodes(&[map, body, END]),
        edges(&[("m", "e")])
    )
}

const MAP: &str = "id = 'm'\nkind = 'map'\nover = 'xs'\nitem = 'x'\nbody = 'w'\nwrites = 'out'";
const BODY: &str = "id = 'w'\nkind = 'tool'\ntool = 't'\nreads = ['x']\nwrites = 'y'";

#[test]
fn map_needs_a_list_key_and_a_body_without_edges() {
    let flow = parse(&map_flow(MAP, BODY, "")).unwrap();
    assert!(matches!(
        flow.kind("m"),
        Some(FlowKind::Map { over, item, body, .. }) if over == "xs" && item == "x" && body == "w"
    ));
    assert!(!flow.is_v1());
    refused(
        &map_flow(&MAP.replace("'xs'", "'y'"), BODY, ""),
        "nodes.m.over",
        "list key",
    );
    refused(
        &map_flow(MAP, BODY, &edges(&[("w", "e")])),
        "edges[1]",
        "map body has no edges",
    );
    refused(
        &map_flow(&MAP.replace("body = 'w'", "body = 'zz'"), BODY, ""),
        "nodes.m.body",
        "unknown node `zz`",
    );
    refused(
        &map_flow(MAP, &BODY.replace("reads = ['x']", "reads = ['y']"), ""),
        "nodes.m.body",
        "item key `x`",
    );
    refused(
        &map_flow(&MAP.replace("'out'", "'y'"), BODY, ""),
        "nodes.m.writes",
        "not a list",
    );
    refused(
        &map_flow(
            MAP,
            &BODY.replace(
                "kind = 'tool'\ntool = 't'",
                "kind = 'human'\nask = 'question'",
            ),
            "",
        ),
        "nodes.m.body",
        "agent, tool or flow",
    );
    refused(
        &map_flow(MAP, &BODY.replace("\nwrites = 'y'", ""), ""),
        "nodes.m.body",
        "must write one value",
    );
    refused(
        &map_flow(&format!("{MAP}\nmax_items = 0"), BODY, ""),
        "nodes.m.max_items",
        "between 1 and 10000",
    );
    refused(
        &map_flow(&format!("{MAP}\nconcurrency = 257"), BODY, ""),
        "nodes.m.concurrency",
        "between 1 and 256",
    );
}

fn switch_flow(on: &str, routes: &[(&str, &str)]) -> String {
    let l = "id = 'l'\nkind = 'tool'\ntool = 't'";
    let r = "id = 'r'\nkind = 'tool'\ntool = 't'";
    let routes: String = routes
        .iter()
        .map(|(to, when)| routed("s", to, when))
        .collect();
    format!(
        "{}{}{routes}{}",
        head("s", ""),
        nodes(&[
            &format!("id = 's'\nkind = 'switch'\non = '{on}'"),
            l,
            r,
            END
        ]),
        edges(&[("l", "e"), ("r", "e")])
    )
}

#[test]
fn switch_edges_need_labels_and_cover_enum_values() {
    let flow = parse(&switch_flow("next", &[("l", "left"), ("r", "right")])).unwrap();
    assert!(matches!(flow.kind("s"), Some(FlowKind::Switch { on, .. }) if on == "next"));
    parse(&switch_flow("ok", &[("l", "true"), ("r", "false")])).unwrap();
    parse(&switch_flow("y", &[("l", "a"), ("r", "b")])).unwrap();
    refused(
        &switch_flow("next", &[("l", "left"), ("r", "up")]),
        "nodes.s",
        "label `up` is not a value of `next`",
    );
    refused(
        &switch_flow("ok", &[("l", "true"), ("r", "maybe")]),
        "nodes.s",
        "label `maybe`",
    );
    let missing =
        switch_flow("next", &[("l", "left"), ("r", "right")]).replace("when = 'right'\n", "");
    refused(&missing, "edges[1].when", "needs `when`");
    let half = format!(
        "{}{}{}{}",
        head("s", ""),
        nodes(&[
            "id = 's'\nkind = 'switch'\non = 'next'",
            "id = 'l'\nkind = 'tool'\ntool = 't'",
            END
        ]),
        routed("s", "l", "left"),
        edges(&[("l", "e")])
    );
    refused(&half, "nodes.s", "do not cover value `right`");
    refused(
        &switch_flow("xs", &[("l", "a"), ("r", "b")]),
        "nodes.s.on",
        "string, enum or bool",
    );
    let labelled_tool = format!(
        "{}{}{}",
        head("a", "[budget]\n"),
        nodes(&[TOOL_A, END]),
        routed("a", "e", "x")
    );
    refused(
        &labelled_tool,
        "edges[0].when",
        "only router and switch edges",
    );
}

fn goto_flow(targets: &str, extra_edges: &str) -> String {
    format!(
        "{}{}{}{extra_edges}",
        head("g", ""),
        nodes(&[
            &format!("id = 'g'\nkind = 'agent'\npersona = 'p'\nprompt = 'Go.'\ngoto = {targets}"),
            "id = 'l'\nkind = 'tool'\ntool = 't'",
            END
        ]),
        edges(&[("l", "g")])
    )
}

#[test]
fn goto_targets_must_exist_and_have_no_plain_edges() {
    let flow = parse(&goto_flow("['l', 'e']", "")).unwrap();
    assert!(matches!(flow.kind("g"), Some(FlowKind::Agent { goto, .. }) if goto == &["l", "e"]));
    let out: Vec<_> = flow
        .workflow()
        .next("g")
        .into_iter()
        .map(|e| (e.to.as_str(), e.when.as_deref()))
        .collect();
    assert_eq!(out, vec![("l", Some("l")), ("e", Some("e"))]);
    refused(
        &goto_flow("['l', 'zz']", ""),
        "nodes.g.goto",
        "unknown node `zz`",
    );
    refused(&goto_flow("['l', 'l']", ""), "nodes.g.goto", "`l` repeats");
    refused(&goto_flow("[]", ""), "nodes.g.goto", "1 to 256");
    refused(
        &goto_flow("['l', 'e']", &edges(&[("g", "e")])),
        "edges[1].from",
        "goto targets",
    );
}

#[test]
fn cycle_without_switch_router_or_goto_is_refused() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let loop_text = format!(
        "{}{}{}",
        head("a", "[budget]\nsplit = 'even'\n"),
        nodes(&[TOOL_A, b, END]),
        edges(&[("a", "b"), ("b", "a"), ("b", "e")])
    );
    refused(
        &loop_text,
        "nodes.a",
        "cycle without a router, switch or goto",
    );
    parse(&goto_flow("['l', 'e']", "")).unwrap();
    let through_switch = format!(
        "{}{}{}{}",
        head("a", ""),
        nodes(&[TOOL_A, "id = 's'\nkind = 'switch'\non = 'ok'", END]),
        edges(&[("a", "s")]),
        format!("{}{}", routed("s", "a", "false"), routed("s", "e", "true"))
    );
    parse(&through_switch).unwrap();
}

#[test]
fn nested_fan_out_within_depth_parses() {
    let one = parse(&nested(1)).unwrap();
    assert!(one.is_v1());
    for depth in 2..=4 {
        let text = nested(depth);
        assert!(
            Workflow::parse("test/f.toml", &text).is_err(),
            "0.1 refuses nesting"
        );
        let flow = parse(&text).unwrap_or_else(|e| panic!("depth {depth}: {e}"));
        assert!(!flow.is_v1(), "depth {depth}");
        assert_eq!(flow.workflow().nodes.len(), 3 * depth + 2);
    }
}

#[test]
fn nested_fan_out_deeper_than_4_is_refused() {
    refused(&nested(5), "nodes.f4", "nested fan-out deeper than 4");
}

#[test]
fn any_join_closes_one_fan_out() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'";
    let flow = parse(&fan("", b, c, "id = 'j'\nkind = 'join'\nmode = 'any'")).unwrap();
    assert!(matches!(
        flow.kind("j"),
        Some(FlowKind::Join {
            mode: JoinMode::Any,
            ..
        })
    ));
    assert_eq!(flow.workflow().node("j").unwrap().kind, NodeKind::Join);
    assert!(!flow.is_v1());
    refused(
        &fan("", b, c, "id = 'j'\nkind = 'join'\nmode = 'some'"),
        "nodes.j.mode",
        "`all` or `any`",
    );
    let lone = format!(
        "{}{}{}",
        head("a", ""),
        nodes(&[TOOL_A, "id = 'j'\nkind = 'join'\nmode = 'any'", END]),
        edges(&[("a", "j"), ("j", "e")])
    );
    refused(&lone, "nodes.j", "exactly one fan-out");
}

#[test]
fn deferred_flag_only_on_agent_tool_or_flow_nodes() {
    let one = |node: &str| {
        format!(
            "{}{}{}",
            head("a", ""),
            nodes(&[node, END]),
            edges(&[("a", "e")])
        )
    };
    let tool = parse(&one("id = 'a'\nkind = 'tool'\ntool = 't'\ndefer = true")).unwrap();
    assert!(matches!(
        tool.kind("a"),
        Some(FlowKind::Tool { defer: true, .. })
    ));
    parse(&one(
        "id = 'a'\nkind = 'agent'\npersona = 'p'\nprompt = 'Go.'\ndefer = true",
    ))
    .unwrap();
    parse(&one(
        "id = 'a'\nkind = 'flow'\nflow = 'polish'\nreturns = 'summary'\nwrites = 'y'\ndefer = true",
    ))
    .unwrap();
    refused(
        &one("id = 'a'\nkind = 'human'\nask = 'question'\nwrites = 'y'\ndefer = true"),
        "nodes.a.defer",
        "not a field of a human node",
    );
    refused(
        &one("id = 'a'\nkind = 'switch'\non = 'ok'\ndefer = true"),
        "nodes.a.defer",
        "not a field of a switch node",
    );
}

#[test]
fn untracked_key_must_be_nullable() {
    let text = format!(
        "untracked = ['tmp']\n{}{}{}",
        HEAD,
        nodes(&[TOOL_A, END]),
        edges(&[("a", "e")])
    );
    let flow = parse(&text).unwrap();
    assert_eq!(flow.untracked(), &["tmp".to_string()]);
    assert!(!flow.is_v1());
    refused(
        &text.replace("['tmp']", "['y']"),
        "untracked",
        "`y` must be nullable",
    );
    refused(
        &text.replace("['tmp']", "['zz']"),
        "untracked",
        "not declared",
    );
    refused(
        &text.replace("['tmp']", "['tmp', 'tmp']"),
        "untracked",
        "repeats",
    );
}

#[test]
fn subflow_names_are_valid() {
    let one = |node: &str| {
        format!(
            "{}{}{}",
            head("a", ""),
            nodes(&[node, END]),
            edges(&[("a", "e")])
        )
    };
    let good = "id = 'a'\nkind = 'flow'\nflow = 'polish'\nreads = ['x']\nreturns = 'summary'\nwrites = 'y'";
    let flow = parse(&one(good)).unwrap();
    assert!(matches!(
        flow.kind("a"),
        Some(FlowKind::Flow { flow, returns, defer: false, .. }) if flow == "polish" && returns == "summary"
    ));
    refused(
        &one(&good.replace("'polish'", "'Polish'")),
        "nodes.a.flow",
        "must be 1 to 64",
    );
    refused(
        &one(&good.replace("'summary'", "'Sum mary'")),
        "nodes.a.returns",
        "must match",
    );
    refused(
        &one(&good.replace("\nwrites = 'y'", "")),
        "nodes.a.writes",
        "needs `writes`",
    );
    refused(
        &one(&good.replace("\nreturns = 'summary'", "")),
        "nodes.a.returns",
        "needs `returns`",
    );
}

#[test]
fn outline_is_a_valid_workflow_with_placeholder_kinds() {
    let text = format!(
        "{}{}{}",
        head("m", ""),
        nodes(&[
            MAP,
            BODY,
            "id = 's'\nkind = 'switch'\non = 'ok'",
            "id = 'sub'\nkind = 'flow'\nflow = 'polish'\nreturns = 'summary'\nwrites = 'x'",
            END
        ]),
        format!(
            "{}{}{}",
            edges(&[("m", "s")]),
            routed("s", "sub", "true"),
            routed("s", "e", "false")
        )
    ) + &edges(&[("sub", "e")]);
    let flow = parse(&text).unwrap();
    let outline = flow.workflow();
    let tool = |id: &str| match &outline.node(id).unwrap().kind {
        NodeKind::Tool { tool } => tool.clone(),
        other => panic!("{id}: {other:?}"),
    };
    assert_eq!(tool("m"), "flow-map");
    assert_eq!(tool("s"), "flow-switch");
    assert_eq!(tool("sub"), "flow-flow");
    assert_eq!(tool("w"), "t");
    assert_eq!(outline.node("m").unwrap().writes.as_deref(), Some("out"));
    assert_eq!(outline.node("w").unwrap().reads, vec!["x".to_string()]);
    assert_eq!(outline.next("s").len(), 2);
    assert_eq!(outline.start, "m");
    assert!(outline
        .check_value("out", &serde_json::json!(["a"]))
        .is_empty());
    let back: Workflow = serde_json::from_value(serde_json::to_value(outline).unwrap()).unwrap();
    assert_eq!(&back, outline);
    assert_eq!(flow.name(), "f");
    assert_eq!(flow.start(), "m");
}

#[test]
fn defaults_are_the_safe_values() {
    assert_eq!(JoinMode::default(), JoinMode::All);
    assert_eq!(BudgetSplit::default(), BudgetSplit::Even);
    let flow = parse(&map_flow(MAP, BODY, "")).unwrap();
    assert!(matches!(
        flow.kind("m"),
        Some(FlowKind::Map {
            max_items: 256,
            concurrency: 8,
            ..
        })
    ));
    assert_eq!(flow.budget_split(), BudgetSplit::Even);
    assert!(flow.untracked().is_empty());
    assert_eq!(flow.max_steps(), 200);
    assert_eq!(flow.max_retries(), 5);
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'";
    let joined = parse(&fan("[budget]\n", b, c, JOIN)).unwrap();
    assert!(matches!(
        joined.kind("j"),
        Some(FlowKind::Join {
            mode: JoinMode::All,
            ..
        })
    ));
    assert!(matches!(
        joined.kind("b"),
        Some(FlowKind::Tool { defer: false, .. })
    ));
    let shared = parse(&fan("[budget]\nsplit = 'shared'\n", b, c, JOIN)).unwrap();
    assert_eq!(shared.budget_split(), BudgetSplit::Shared);
    assert!(shared.is_v1());
    refused(
        &fan("[budget]\nsplit = 'half'\n", b, c, JOIN),
        "budget.split",
        "`even` or `shared`",
    );
}

#[test]
fn flow_parse_rejects_unknown_keys() {
    let b = "id = 'b'\nkind = 'tool'\ntool = 't'";
    let c = "id = 'c'\nkind = 'tool'\ntool = 't'";
    let base = fan("[reducers]\nn = 'sum'\n", b, c, JOIN);
    assert!(parse(&format!("colour = 1\n{base}")).is_err());
    assert!(parse(&base.replace(
        "tool = 't'\n[[nodes]]\nid = 'c'",
        "tool = 't'\nspeed = 2\n[[nodes]]\nid = 'c'"
    ))
    .is_err());
    assert!(parse(&fan("[budget]\ncap = 1\n", b, c, JOIN)).is_err());
    let wrong_field = fan(
        "",
        b,
        "id = 'c'\nkind = 'tool'\ntool = 't'\non = 'ok'",
        JOIN,
    );
    refused(&wrong_field, "nodes.c.on", "not a field of a tool node");
    refused(
        &fan("", b, "id = 'c'\nkind = 'wait'", JOIN),
        "nodes.c.kind",
        "unknown kind",
    );
    refused(
        &fan("[budget]\n", b, "id = 'c'\nkind = 'wait'", JOIN),
        "nodes.c.kind",
        "switch, map, flow",
    );
}

#[test]
fn goto_into_a_join_is_an_input_from_outside_the_fan_out() {
    let text = |target: &str| {
        format!(
            "{}{}{}{}{}",
            head("r", ""),
            nodes(&[
                "id = 'r'\nkind = 'router'\ncondition = 'flag(ok)'",
                TOOL_A,
                "id = 'b'\nkind = 'tool'\ntool = 't'",
                "id = 'c'\nkind = 'tool'\ntool = 't'",
                JOIN,
                &format!(
                    "id = 'g'\nkind = 'agent'\npersona = 'p'\nprompt = 'Go.'\ngoto = ['{target}']"
                ),
                END
            ]),
            routed("r", "a", "true"),
            routed("r", "g", "false"),
            edges(&[("a", "b"), ("a", "c"), ("b", "j"), ("c", "j"), ("j", "e")])
        )
    };
    parse(&text("e")).unwrap();
    refused(
        &text("j"),
        "nodes.j",
        "inputs from outside the fan-out at `a`",
    );
}
