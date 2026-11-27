use crate::common::{calc_model, echo_model, greeter};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use troupekit::{Agent, AsTool, KitError, Storage, TeamMode, Tool, ToolContext};
use troupekit_agent::Limits;
use troupekit_core::event::Usage;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::store::CheckpointStore;
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_providers::{CompletionResponse, Model, StopReason};

fn call_then_answer(tool: &'static str, args: Value) -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", move |req| {
        let result = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .find_map(|p| match p {
                ContentPart::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            });
        match result {
            Some(content) => text_reply(&format!("Lead got: {content}"), 10),
            None => Ok(CompletionResponse {
                message: Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: tool.into(),
                        arguments: args.clone(),
                    }],
                )
                .unwrap(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cost_usd_micros: 0,
                },
                stop: StopReason::ToolUse,
            }),
        }
    }))
}

fn member(name: &str) -> Agent {
    Agent::new(name)
        .instructions("Answer the task.")
        .role("A researcher.")
        .goal("Find facts.")
        .model("fake/m")
        .model_client(echo_model())
}

fn lead(model: Arc<dyn Model>) -> Agent {
    Agent::new("lead")
        .instructions("Lead the work.")
        .model("fake/m")
        .model_client(model)
}

fn config_message(err: KitError) -> String {
    match err {
        KitError::Config { message, .. } => message,
        other => panic!("expected a config error, got {other:?}"),
    }
}

#[tokio::test]
async fn agent_tool_in_one_line() {
    let model = call_then_answer("researcher", json!({"task": "find facts"}));
    let built = lead(model)
        .agent_tool(member("researcher"))
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().tools, ["researcher"]);
    let out = built.run_with("Go").await.unwrap();
    assert!(out.text().starts_with("Lead got: "), "{}", out.text());
    assert!(
        out.text().contains("Hello! You said: find facts"),
        "{}",
        out.text()
    );
    assert_eq!(out.member_runs().len(), 1);
    assert_ne!(out.member_runs()[0], out.run_id());
    assert!(greeter().build().await.unwrap().persona().tools.is_empty());
}

#[tokio::test]
async fn built_agent_is_a_tool() {
    let built = greeter().build().await.unwrap();
    assert_eq!(built.spec().name, "greeter");
    assert_eq!(built.spec().schema["required"], json!(["task"]));
    let ctx = ToolContext::for_call(RunId::new(), 1, 0);
    let out = built.call(json!({"task": "hi"}), &ctx).await.unwrap();
    assert!(!out.is_error, "{}", out.content);
    let reply: Value = serde_json::from_str(&out.content).unwrap();
    assert!(reply["member_run_id"].is_string());
    let answer = reply["answer"].as_str().unwrap();
    assert!(answer.contains("Hello! You said: hi"), "{answer}");
    let model = call_then_answer("greeter", json!({"task": "say hi"}));
    let leader = lead(model).tool(built.clone()).build().await.unwrap();
    assert_eq!(leader.persona().tools, ["greeter"]);
    let text = leader.run("Go").await.unwrap();
    assert!(text.contains("Hello! You said: say hi"), "{text}");
    let arc = built.into_arc();
    assert_eq!(arc.spec().name, "greeter");
}

#[tokio::test]
async fn delegates_adds_delegate_work_and_ask_question() {
    let model = call_then_answer(
        "delegate_work",
        json!({"coworker": "checker", "task": "check facts"}),
    );
    let built = lead(model)
        .delegates([member("researcher"), member("checker")])
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().tools, ["delegate_work", "ask_question"]);
    let out = built.run_with("Go").await.unwrap();
    assert!(
        out.text().contains("Hello! You said: check facts"),
        "{}",
        out.text()
    );
    assert_eq!(out.member_runs().len(), 1);
    let again = built.run_with("Go").await.unwrap();
    assert_ne!(again.member_runs(), out.member_runs());
    let twice = lead(echo_model())
        .delegates([member("researcher"), member("researcher")])
        .build()
        .await
        .unwrap_err();
    assert!(format!("{twice}").contains("listed twice"), "{twice}");
    let clash = lead(echo_model())
        .agent_tool(member("researcher"))
        .agent_tool(member("researcher"))
        .build()
        .await
        .unwrap_err();
    assert!(format!("{clash}").contains("researcher"), "{clash}");
}

#[tokio::test]
async fn graph_refuses_agent_with_members() {
    let leader = lead(echo_model()).agent_tool(member("researcher"));
    let err = troupekit::Graph::new("g")
        .agent("lead", leader)
        .build()
        .await
        .unwrap_err();
    let message = format!("{err}");
    assert!(
        message.contains("agent tools, delegates, members or handoffs"),
        "{message}"
    );
}

#[tokio::test]
async fn batch_returns_results_in_input_order() {
    let built = greeter().build().await.unwrap();
    let results = built.batch(["a", "b", "c", "d"], 2).await.unwrap();
    let texts: Vec<String> = results
        .into_iter()
        .map(|r| r.unwrap().text().to_string())
        .collect();
    assert_eq!(
        texts,
        [
            "Hello! You said: a",
            "Hello! You said: b",
            "Hello! You said: c",
            "Hello! You said: d"
        ]
    );
    assert!(built.batch(Vec::<&str>::new(), 2).await.unwrap().is_empty());
    let zero = built.batch(["a"], 0).await.unwrap_err();
    assert!(config_message(zero).contains("concurrency"));
    let many = built.batch(["a"], 65).await.unwrap_err();
    assert!(config_message(many).contains("concurrency"));
}

#[tokio::test]
async fn batch_concurrency_cap_holds() {
    let live = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let (l, p) = (live.clone(), peak.clone());
    let slow = troupekit::tool("add", "Adds slowly", move |x: crate::common::AddArgs| {
        let (l, p) = (l.clone(), p.clone());
        async move {
            let now = l.fetch_add(1, Ordering::SeqCst) + 1;
            p.fetch_max(now, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(20)).await;
            l.fetch_sub(1, Ordering::SeqCst);
            x.a + x.b
        }
    });
    let built = Agent::new("calc")
        .instructions("Add.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(slow)
        .build()
        .await
        .unwrap();
    let results = built
        .batch(["1", "2", "3", "4", "5", "6"], 2)
        .await
        .unwrap();
    assert_eq!(results.len(), 6);
    assert!(results
        .iter()
        .all(|r| r.as_ref().unwrap().text() == "The sum is 42"));
    assert_eq!(peak.load(Ordering::SeqCst), 2);
    assert_eq!(live.load(Ordering::SeqCst), 0);
}

fn durable_calc(store: Arc<InMemCheckpoints>) -> Agent {
    Agent::new("calc")
        .instructions("Add.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(crate::common::add_tool())
        .checkpoints(Storage::custom(store))
}

#[tokio::test]
async fn agent_history_and_fork() {
    let store = Arc::new(InMemCheckpoints::with_history());
    let built = durable_calc(store.clone()).build().await.unwrap();
    let out = built.run_with("Add 2 and 40").await.unwrap();
    let history = built.history(out.run_id()).await.unwrap();
    assert!(history.len() >= 3, "{}", history.len());
    assert!(history.windows(2).all(|w| w[0].step + 1 == w[1].step));
    let first = history[0].step;
    let child = built.fork(out.run_id(), first).await.unwrap();
    assert_ne!(child, out.run_id());
    let links = store.forks(out.run_id()).await.unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].child(), child);
    assert_eq!(links[0].parent_step(), first);
    assert_eq!(built.resume(child).await.unwrap(), "The sum is 42");
    let past = built.fork(out.run_id(), 10_000).await.unwrap_err();
    assert!(config_message(past).contains("no checkpoint at step"));
    let memory = greeter().build().await.unwrap();
    let none = memory.history(out.run_id()).await.unwrap_err();
    assert!(config_message(none).contains("Storage::sqlite"));
}

#[tokio::test]
async fn continue_from_message() {
    let store = Arc::new(InMemCheckpoints::with_history());
    let built = durable_calc(store.clone()).build().await.unwrap();
    let out = built.run_with("Add 2 and 40").await.unwrap();
    assert_eq!(out.messages().len(), 4);
    let next = built
        .continue_from(out.run_id(), 3, "And again?")
        .await
        .unwrap();
    assert_ne!(next.run_id(), out.run_id());
    assert_eq!(next.messages()[..3], out.messages()[..3]);
    assert_eq!(next.messages()[3].text_content(), "And again?");
    assert_eq!(next.text(), "The sum is 42");
    let links = store.forks(out.run_id()).await.unwrap();
    assert_eq!(links.len(), 1);
    assert_eq!(links[0].child(), next.run_id());
    let inside = built.continue_from(out.run_id(), 2, "x").await.unwrap_err();
    assert!(
        format!("{inside}").contains("user or tool message"),
        "{inside}"
    );
    let two = vec![
        Message::text(Role::User, "a"),
        Message::text(Role::User, "b"),
    ];
    let err = built.continue_from(out.run_id(), 3, two).await.unwrap_err();
    assert!(config_message(err).contains("at most one message"));
}

fn counted_echo(count: Arc<AtomicUsize>) -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", move |req| {
        count.fetch_add(1, Ordering::SeqCst);
        let asked = req
            .messages
            .last()
            .map(|m| m.text_content())
            .unwrap_or_default();
        text_reply(&format!("Hello! You said: {asked}"), 10)
    }))
}

fn team_agent(name: &str, model: Arc<dyn Model>) -> Agent {
    Agent::new(name)
        .instructions("Do the work.")
        .model("fake/m")
        .model_client(model)
}

fn delegate_then_answer() -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", |req| {
        let result = req
            .messages
            .iter()
            .flat_map(|m| &m.parts)
            .find_map(|p| match p {
                ContentPart::ToolResult { content, .. } => Some(content.clone()),
                _ => None,
            });
        match result {
            Some(content) => text_reply(&format!("Manager got: {content}"), 10),
            None => Ok(CompletionResponse {
                message: Message::new(
                    Role::Assistant,
                    vec![ContentPart::ToolCall {
                        id: "c1".into(),
                        name: "delegate_work".into(),
                        arguments: json!({"coworker": "tech", "task": "check the cable"}),
                    }],
                )
                .unwrap(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cost_usd_micros: 0,
                },
                stop: StopReason::ToolUse,
            }),
        }
    }))
}

fn team_config(err: KitError) -> String {
    match err {
        KitError::Config { message, .. } => message,
        other => panic!("expected a config error, got {other:?}"),
    }
}

#[tokio::test]
async fn members_route_in_one_line() {
    let (billing, tech) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let router = Arc::new(FnModel::new("fake/m", |_req| {
        text_reply(r#"{"next":"tech","reason":"broken device"}"#, 10)
    }));
    let built = team_agent("desk", router)
        .members(
            [
                team_agent("billing", counted_echo(billing.clone())),
                team_agent("tech", counted_echo(tech.clone())),
            ],
            TeamMode::Route,
        )
        .build()
        .await
        .unwrap();
    let out = built.run_with("My laptop broke").await.unwrap();
    assert!(
        out.text().starts_with("Hello! You said: "),
        "{}",
        out.text()
    );
    assert!(out.text().contains("My laptop broke"), "{}", out.text());
    assert_eq!(tech.load(Ordering::SeqCst), 1);
    assert_eq!(billing.load(Ordering::SeqCst), 0);
    assert_eq!(out.model_calls(), 0);
    assert_eq!(out.messages().len(), 2);
    assert!(out.member_runs().is_empty());
    let plain = team_agent("desk", counted_echo(Arc::new(AtomicUsize::new(0))))
        .build()
        .await
        .unwrap();
    assert_eq!(plain.run("hi").await.unwrap(), "Hello! You said: hi");
}

fn counted_reply(count: Arc<AtomicUsize>, reply: &'static str) -> Arc<dyn Model> {
    Arc::new(FnModel::new("fake/m", move |_req| {
        count.fetch_add(1, Ordering::SeqCst);
        text_reply(reply, 10)
    }))
}

#[tokio::test]
async fn handoffs_in_one_line() {
    let (desk, billing, tech) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
        Arc::new(AtomicUsize::new(0)),
    );
    let to_tech = r#"{"next":"tech","reason":"hardware","answer":""}"#;
    let fixed = r#"{"next":"end","reason":"done","answer":"Replace the screen."}"#;
    let built = team_agent("desk", counted_reply(desk.clone(), to_tech))
        .handoffs([
            team_agent("billing", counted_reply(billing.clone(), fixed)),
            team_agent("tech", counted_reply(tech.clone(), fixed)),
        ])
        .build()
        .await
        .unwrap();
    let out = built.run_with("My laptop broke").await.unwrap();
    assert_eq!(out.text(), "Replace the screen.");
    assert_eq!(desk.load(Ordering::SeqCst), 1);
    assert_eq!(tech.load(Ordering::SeqCst), 1);
    assert_eq!(billing.load(Ordering::SeqCst), 0);
    assert_eq!(out.messages().len(), 2);
    let both = team_agent("desk", counted_reply(desk.clone(), to_tech))
        .handoffs([team_agent("tech", counted_reply(tech.clone(), fixed))])
        .members(
            [team_agent("billing", counted_reply(billing, fixed))],
            TeamMode::Route,
        )
        .build()
        .await;
    let message = team_config(both.err().expect("members and handoffs together"));
    assert!(
        message.contains("members or handoffs, not both"),
        "{message}"
    );
}

#[tokio::test]
async fn manages_runs_hierarchical_crew() {
    let tech = Arc::new(AtomicUsize::new(0));
    let store = Arc::new(InMemCheckpoints::with_history());
    let built = team_agent("manager", delegate_then_answer())
        .checkpoints(Storage::custom(store.clone()))
        .manages([team_agent("tech", counted_echo(tech.clone()))])
        .build()
        .await
        .unwrap();
    let out = built.run_with("Fix my network").await.unwrap();
    assert!(out.text().starts_with("Manager got: "), "{}", out.text());
    assert!(
        out.text().contains("Hello! You said: check the cable"),
        "{}",
        out.text()
    );
    assert_eq!(tech.load(Ordering::SeqCst), 1);
    assert_eq!(built.resume(out.run_id()).await.unwrap(), out.text());
    let history = built.history(out.run_id()).await.unwrap_err();
    assert!(team_config(history).contains("team agent"));
    let two = vec![
        Message::text(Role::User, "a"),
        Message::text(Role::User, "b"),
    ];
    let err = built.run_with(two).await.unwrap_err();
    assert!(team_config(err).contains("one user text"));
}

#[tokio::test]
async fn members_tasks_loops_until_done() {
    let passes = Arc::new(AtomicUsize::new(0));
    let count = passes.clone();
    let seen = Arc::new(std::sync::Mutex::new(String::new()));
    let saw = seen.clone();
    let leader = Arc::new(FnModel::new("fake/m", move |req| {
        if count.fetch_add(1, Ordering::SeqCst) == 0 {
            text_reply(
                r#"{"done":false,"tasks":["check the tax"],"finished":["the bill is 40 dollars"],"answer":""}"#,
                10,
            )
        } else {
            let asked: String = req.messages.iter().map(|m| m.text_content()).collect();
            *saw.lock().unwrap() = asked;
            text_reply(
                r#"{"done":true,"tasks":[],"finished":["the tax is 4 dollars"],"answer":"All settled."}"#,
                10,
            )
        }
    }));
    let member = Arc::new(AtomicUsize::new(0));
    let built = team_agent("lead", leader)
        .members(
            [team_agent("billing", counted_echo(member.clone()))],
            TeamMode::Tasks,
        )
        .build()
        .await
        .unwrap();
    let out = built.run_with("Sort out my account").await.unwrap();
    assert_eq!(out.text(), "All settled.");
    assert_eq!(passes.load(Ordering::SeqCst), 2);
    assert_eq!(member.load(Ordering::SeqCst), 0);
    let second = seen.lock().unwrap().clone();
    assert!(second.contains("the bill is 40 dollars"), "{second}");
    assert!(second.contains("check the tax"), "{second}");
    let stuck_passes = Arc::new(AtomicUsize::new(0));
    let stuck_count = stuck_passes.clone();
    let stuck = Arc::new(FnModel::new("fake/m", move |_req| {
        stuck_count.fetch_add(1, Ordering::SeqCst);
        text_reply(
            r#"{"done":false,"tasks":["more"],"finished":[],"answer":""}"#,
            10,
        )
    }));
    let looping = team_agent("lead", stuck)
        .members(
            [team_agent("billing", counted_echo(member))],
            TeamMode::Tasks,
        )
        .build()
        .await
        .unwrap();
    let err = looping.run_with("Never done").await.unwrap_err();
    assert!(
        matches!(&err, KitError::Failed { reason, .. } if reason.contains("last pass")),
        "{err}"
    );
    assert_eq!(
        stuck_passes.load(Ordering::SeqCst),
        5,
        "a leader never done runs max_iterations passes"
    );
}

#[tokio::test]
async fn team_build_refusals() {
    let echo0 = || counted_echo(Arc::new(AtomicUsize::new(0)));
    let none = team_agent("lead", echo0()).members(Vec::<Agent>::new(), TeamMode::Route);
    assert!(team_config(none.build().await.unwrap_err()).contains("1 to 32 members"));
    let limited = team_agent("a", echo0()).limits(Limits {
        run: Some(5),
        ..Limits::default()
    });
    let own = team_agent("lead", echo0()).members([limited], TeamMode::Broadcast);
    let message = team_config(own.build().await.unwrap_err());
    assert!(
        message.contains("member `a` sets its own limits"),
        "{message}"
    );
    assert!(message.contains("on the leader"), "{message}");
    let team = team_agent("inner", echo0()).members([team_agent("a", echo0())], TeamMode::Route);
    let outer = team_agent("lead", echo0()).agent_tool(team.clone());
    let message = team_config(outer.build().await.unwrap_err());
    assert!(message.contains("leads a team"), "{message}");
    let built = team.build().await.unwrap();
    let ctx = ToolContext::for_call(RunId::new(), 1, 0);
    let refused = built.call(json!({"task": "x"}), &ctx).await.unwrap_err();
    assert!(refused.to_string().contains("leads a team"), "{refused}");
}
