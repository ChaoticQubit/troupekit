use crate::common::{calc_model, echo_model, greeter};
use serde_json::{json, Value};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;
use troupekit::{Agent, AsTool, KitError, Storage, Tool, ToolContext};
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
    assert!(message.contains("agent tools or delegates"), "{message}");
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
