use async_trait::async_trait;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use troupekit_agent::io::{AgentIo, SimpleIo, VecSink};
use troupekit_agent::run::{drive, AgentRun, RunState, StepOutcome};
use troupekit_agent::{AgentError, InMemoryLedger, Limits, PeriodLimit, Persona, UsageLedger};
use troupekit_core::clock::FixedClock;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::{EventBody, PauseReason, Usage};
use troupekit_core::inmem::InMemCheckpoints;
use troupekit_core::message::{ContentPart, Message, Role};
use troupekit_core::secret::Secret;
use troupekit_core::store::CheckpointStore;
use troupekit_providers::model::{Endpoint, HttpModel};
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::wire::{AuthStyle, Protocol, Quirks};
use troupekit_providers::{CompletionRequest, CompletionResponse, Delta, Model, Price, StopReason};
use troupekit_tools::{hosted, typed_tool, HostedTool, ToolContext, ToolOutput, ToolRegistry};

const PRICE: Price = Price {
    input_usd_micros_per_mtok: 1_000_000,
    output_usd_micros_per_mtok: 2_000_000,
    verified: true,
    long_context: None,
    search_usd_micros: 0,
};

type Reply = Result<CompletionResponse, ProviderError>;

struct FakeModel {
    replies: Mutex<VecDeque<Reply>>,
    seen: Mutex<Vec<CompletionRequest>>,
}

impl FakeModel {
    fn new(replies: Vec<Reply>) -> Arc<Self> {
        Arc::new(Self {
            replies: Mutex::new(replies.into()),
            seen: Mutex::new(Vec::new()),
        })
    }

    fn seen(&self) -> Vec<CompletionRequest> {
        self.seen.lock().unwrap().clone()
    }

    fn next(&self, req: &CompletionRequest) -> Reply {
        self.seen.lock().unwrap().push(req.clone());
        self.replies.lock().unwrap().pop_front().unwrap_or_else(|| {
            Err(ProviderError::new(
                ProviderErrorKind::BadRequest,
                "script empty",
            ))
        })
    }
}

#[async_trait]
impl Model for FakeModel {
    fn id(&self) -> &str {
        "fake/m"
    }

    async fn complete(&self, req: &CompletionRequest) -> Reply {
        self.next(req)
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Reply {
        let reply = self.next(req)?;
        let text = reply.message.text_content();
        if !text.is_empty() {
            on_delta(&Delta::TextDelta { text });
        }
        Ok(reply)
    }
}

fn usage(input: u64, output: u64) -> Usage {
    Usage {
        input_tokens: input,
        output_tokens: output,
        cost_usd_micros: 0,
    }
}

fn text(s: &str) -> Reply {
    Ok(CompletionResponse {
        message: Message::text(Role::Assistant, s),
        usage: usage(10, 5),
        stop: StopReason::EndTurn,
    })
}

fn calls(list: &[(&str, &str)]) -> Reply {
    let parts = list
        .iter()
        .map(|(id, name)| ContentPart::ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            arguments: json!({"q": id}),
        })
        .collect();
    Ok(CompletionResponse {
        message: Message::new(Role::Assistant, parts).unwrap(),
        usage: usage(10, 5),
        stop: StopReason::ToolUse,
    })
}

fn persona(extra: &str) -> Persona {
    let text = format!("name = 'coach'\ndisplay_name = 'Coach'\nrole = 'Plans.'\ngoal = 'A plan.'\ninstructions = 'Plan it.'\nmodel = 'fake/m'\n{extra}");
    Persona::parse("test/coach.toml", &text).unwrap()
}

#[derive(Deserialize, schemars::JsonSchema)]
struct Q {
    q: String,
}

fn counted(name: &str, count: Arc<AtomicUsize>) -> Arc<dyn troupekit_tools::Tool> {
    Arc::new(typed_tool(
        name,
        "test tool",
        move |args: Q, _ctx: ToolContext| {
            let count = count.clone();
            async move {
                count.fetch_add(1, Ordering::SeqCst);
                Ok(ToolOutput::text(format!("got {}", args.q)))
            }
        },
    ))
}

struct Setup {
    io: SimpleIo,
    model: Arc<FakeModel>,
    sink: Arc<VecSink>,
    ledger: Arc<InMemoryLedger>,
    count: Arc<AtomicUsize>,
}

fn setup(replies: Vec<Reply>, price: Price) -> Setup {
    let count = Arc::new(AtomicUsize::new(0));
    let mut tools = ToolRegistry::new();
    tools.register(counted("echo", count.clone())).unwrap();
    tools.register(counted("secret", count.clone())).unwrap();
    tools
        .register(hosted(
            counted("web_search", count.clone()),
            HostedTool::WebSearch { max_uses: 2 },
        ))
        .unwrap();
    let model = FakeModel::new(replies);
    let sink = Arc::new(VecSink::default());
    let ledger = Arc::new(InMemoryLedger::default());
    let mut io = SimpleIo::new(tools).with_model("fake/m", model.clone(), price);
    io.events = sink.clone();
    io.ledger = ledger.clone();
    io.clock = Arc::new(FixedClock::new(1_000));
    Setup {
        io,
        model,
        sink,
        ledger,
        count,
    }
}

fn ask() -> Vec<Message> {
    vec![Message::text(Role::User, "Plan my week")]
}

async fn run_to_end(run: &mut AgentRun, io: &dyn AgentIo) -> StepOutcome {
    drive(run, io, &InMemCheckpoints::default()).await.unwrap()
}

async fn step_to_end(run: &mut AgentRun, io: &dyn AgentIo) -> StepOutcome {
    loop {
        let out = run.step(io).await;
        if out != StepOutcome::Continue {
            return out;
        }
    }
}

#[tokio::test]
async fn answers_without_tools_and_streams_events() {
    let s = setup(vec![text("hello")], PRICE);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("hello"))
    );
    let events = s.sink.events();
    let bodies: Vec<&EventBody> = events.iter().map(|e| &e.body).collect();
    assert!(matches!(bodies[0], EventBody::RunStarted { workflow } if workflow == "coach"));
    assert!(bodies.contains(&&EventBody::TextDelta {
        text: "hello".into()
    }));
    assert!(matches!(
        bodies.last().unwrap(),
        EventBody::RunFinished { .. }
    ));
    assert!(events
        .iter()
        .enumerate()
        .all(|(i, e)| e.seq == i as u64 && e.run_id == run.run_id() && e.at_ms == 1_000));
    let cost = PRICE.cost_usd_micros(&usage(10, 5));
    assert_eq!(
        run.usage(),
        Usage {
            input_tokens: 10,
            output_tokens: 5,
            cost_usd_micros: cost
        }
    );
    let req = &s.model.seen()[0];
    assert!(req
        .system
        .as_deref()
        .unwrap()
        .starts_with("You are Coach. Plans."));
    assert!(req.tools.is_empty() && req.output_schema.is_none());
}

#[tokio::test]
async fn tool_turn_then_answer() {
    let s = setup(vec![calls(&[("c1", "echo")]), text("done")], PRICE);
    let mut run = AgentRun::new(
        &persona("tools = ['echo']"),
        ask(),
        Limits::default(),
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("done"))
    );
    let roles: Vec<Role> = run.messages().iter().map(|m| m.role).collect();
    assert_eq!(
        roles,
        [Role::User, Role::Assistant, Role::Tool, Role::Assistant]
    );
    assert_eq!(
        run.messages()[2].parts,
        vec![ContentPart::ToolResult {
            call_id: "c1".into(),
            content: "got c1".into(),
            is_error: false
        }]
    );
    assert_eq!(s.model.seen()[1].messages.len(), 3);
    assert_eq!(
        s.model.seen()[0]
            .tools
            .iter()
            .map(|t| t.name.as_str())
            .collect::<Vec<_>>(),
        ["echo"]
    );
    let bodies: Vec<EventBody> = s.sink.events().into_iter().map(|e| e.body).collect();
    assert!(bodies.contains(&EventBody::ToolCallStarted {
        call_id: "c1".into(),
        name: "echo".into()
    }));
    assert!(bodies.contains(&EventBody::ToolCallFinished {
        call_id: "c1".into(),
        is_error: false
    }));
}

#[tokio::test]
async fn tools_run_concurrently_in_order() {
    let barrier = Arc::new(tokio::sync::Barrier::new(2));
    let mut tools = ToolRegistry::new();
    for name in ["a", "b"] {
        let barrier = barrier.clone();
        tools
            .register(Arc::new(typed_tool(
                name,
                "waits for the other",
                move |args: Q, _ctx: ToolContext| {
                    let barrier = barrier.clone();
                    async move {
                        barrier.wait().await;
                        Ok(ToolOutput::text(args.q))
                    }
                },
            )))
            .unwrap();
    }
    let model = FakeModel::new(vec![calls(&[("c1", "a"), ("c2", "b")]), text("ok")]);
    let io = SimpleIo::new(tools).with_model("fake/m", model, PRICE);
    let mut run = AgentRun::new(
        &persona("tools = ['a', 'b']"),
        ask(),
        Limits::default(),
        &io,
    )
    .await
    .unwrap();
    let out = tokio::time::timeout(Duration::from_secs(2), run_to_end(&mut run, &io))
        .await
        .expect("tools must run concurrently");
    assert_eq!(out, StepOutcome::Done(json!("ok")));
    let ids: Vec<&str> = run.messages()[2]
        .parts
        .iter()
        .map(|p| match p {
            ContentPart::ToolResult { call_id, .. } => call_id.as_str(),
            _ => panic!("tool message holds only results"),
        })
        .collect();
    assert_eq!(ids, ["c1", "c2"]);
}

#[tokio::test]
async fn disallowed_tool_is_never_called() {
    let s = setup(vec![calls(&[("c1", "secret")]), text("fine")], PRICE);
    let mut run = AgentRun::new(
        &persona("tools = ['echo']"),
        ask(),
        Limits::default(),
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("fine"))
    );
    assert_eq!(s.count.load(Ordering::SeqCst), 0);
    match &run.messages()[2].parts[0] {
        ContentPart::ToolResult {
            is_error, content, ..
        } => assert!(*is_error && content.contains("not allowed"), "{content}"),
        other => panic!("{other:?}"),
    }
}

const TYPED: &str = "[output]\nn = 'int'\nitems = 'list'\n";

#[tokio::test]
async fn schema_failure_repairs_once() {
    let s = setup(
        vec![
            text("{\"n\": \"x\", \"items\": []}"),
            text("{\"n\": 1, \"items\": []}"),
        ],
        PRICE,
    );
    let mut run = AgentRun::new(&persona(TYPED), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!({"n": 1, "items": []}))
    );
    assert_eq!(run.model_calls(), 2);
    let repair = run.messages()[2].text_content();
    assert!(
        repair.starts_with("Your answer failed these checks:") && repair.contains("/n"),
        "{repair}"
    );
    let req = &s.model.seen()[0];
    assert!(req.output_schema.is_some() && req.system.as_deref().unwrap().contains("JSON Schema"));
}

fn searched(before: &str, after: &str, stop: StopReason) -> Reply {
    let mut parts = Vec::new();
    if !before.is_empty() {
        parts.push(ContentPart::Text {
            text: before.into(),
        });
    }
    parts.push(ContentPart::Hosted {
        tool: Some("web_search".into()),
        block: json!({"type": "server_tool_use", "id": "s", "name": "web_search", "input": {"query": "q"}}),
    });
    parts.push(ContentPart::Hosted {
        tool: None,
        block: json!({"type": "web_search_tool_result", "tool_use_id": "s"}),
    });
    if !after.is_empty() {
        parts.push(ContentPart::Text { text: after.into() });
    }
    Ok(CompletionResponse {
        message: Message::new(Role::Assistant, parts).unwrap(),
        usage: usage(10, 5),
        stop,
    })
}

fn search_cap(req: &CompletionRequest) -> Option<u32> {
    req.tools
        .iter()
        .find(|t| t.name == "web_search")
        .map(|t| t.hosted.unwrap().max_uses())
}

#[tokio::test]
async fn hosted_search_pauses_resumes_and_stops_at_the_cap() {
    let s = setup(
        vec![
            searched("", "", StopReason::PauseTurn),
            searched("Let me look.", "not json", StopReason::EndTurn),
            text("{\"n\": 1, \"items\": []}"),
        ],
        PRICE,
    );
    let p = persona(&format!("tools = ['web_search']\n{TYPED}"));
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!({"n": 1, "items": []}))
    );
    let seen = s.model.seen();
    assert_eq!(seen.len(), 3);
    assert_eq!(
        seen.iter().map(search_cap).collect::<Vec<_>>(),
        vec![Some(2), Some(1), None]
    );
    assert_eq!(seen[1].messages.len(), 2);
    assert_eq!(seen[1].messages[1].hosted_calls("web_search"), 1);
    let repair = seen[2].messages.last().unwrap().text_content();
    assert!(repair.contains("not JSON"), "{repair}");
    assert!(!repair.contains("Let me look"), "{repair}");
    assert_eq!(s.count.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn answer_is_the_text_after_the_last_search() {
    let s = setup(
        vec![searched(
            "Searching first.",
            "{\"n\": 2, \"items\": []}",
            StopReason::EndTurn,
        )],
        PRICE,
    );
    let p = persona(&format!("tools = ['web_search']\n{TYPED}"));
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!({"n": 2, "items": []}))
    );
}

#[tokio::test]
async fn blank_reply_is_never_sent_back() {
    let s = setup(vec![text(""), text("{\"n\": 1, \"items\": []}")], PRICE);
    let mut run = AgentRun::new(&persona(TYPED), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!({"n": 1, "items": []}))
    );
    let seen = s.model.seen();
    assert_eq!(seen.len(), 2);
    let blank: Vec<&Message> = seen[1]
        .messages
        .iter()
        .filter(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, ContentPart::Text { text } if text.trim().is_empty()))
        })
        .collect();
    assert!(blank.is_empty(), "{blank:?}");
    assert_eq!(seen[1].messages.len(), 2);
}

#[tokio::test]
async fn repair_then_eval_failure_fails() {
    let s = setup(
        vec![
            text("{\"n\": \"x\", \"items\": []}"),
            text("{\"n\": 1, \"items\": [\"a\"]}"),
            text("never"),
        ],
        PRICE,
    );
    let p = persona(&format!("evals = ['min_items(items, 2)']\n{TYPED}"));
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    let out = run_to_end(&mut run, &s.io).await;
    assert!(
        matches!(&out, StepOutcome::Failed(e) if e.contains("min_items: items has 1 items")),
        "{out:?}"
    );
    assert_eq!(s.model.seen().len(), 2);
    assert!(matches!(run.state(), RunState::Failed { .. }));
    assert!(matches!(
        s.sink.events().last().unwrap().body,
        EventBody::RunFailed { .. }
    ));
}

#[tokio::test]
async fn non_json_is_a_validation_failure() {
    let s = setup(
        vec![text("Sure! Here it is."), text("{\"n\": 2, \"items\": []}")],
        PRICE,
    );
    let mut run = AgentRun::new(&persona(TYPED), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!({"n": 2, "items": []}))
    );
    assert!(run.messages()[2]
        .text_content()
        .contains("answer is not JSON"));
}

#[tokio::test]
async fn judge_eval_runs_and_counts_usage() {
    let judge = Ok(CompletionResponse {
        message: Message::text(Role::Assistant, "{\"score\": 8, \"reason\": \"short\"}"),
        usage: usage(3, 2),
        stop: StopReason::EndTurn,
    });
    let s = setup(vec![text("a short plan"), judge], PRICE);
    let limits = Limits {
        scope: Some(PeriodLimit {
            key: "goal:1".into(),
            max_usd_micros: 1_000_000,
        }),
        ..Limits::default()
    };
    let mut run = AgentRun::new(
        &persona("evals = [\"judge('Plan is short', 7)\"]"),
        ask(),
        limits,
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("a short plan"))
    );
    assert_eq!(run.model_calls(), 1);
    let total = PRICE.cost_usd_micros(&usage(10, 5)) + PRICE.cost_usd_micros(&usage(3, 2));
    assert_eq!(
        run.usage(),
        Usage {
            input_tokens: 13,
            output_tokens: 7,
            cost_usd_micros: total
        }
    );
    assert_eq!(s.ledger.period_total("goal:1").await.unwrap(), total);
    let judge_req = &s.model.seen()[1];
    assert!(
        judge_req.messages[0]
            .text_content()
            .contains("Plan is short")
            && judge_req.tools.is_empty()
    );
}

#[tokio::test]
async fn low_judge_score_repairs() {
    let low = Ok(CompletionResponse {
        message: Message::text(Role::Assistant, "{\"score\": 2, \"reason\": \"long\"}"),
        usage: usage(1, 1),
        stop: StopReason::EndTurn,
    });
    let high = Ok(CompletionResponse {
        message: Message::text(Role::Assistant, "{\"score\": 9, \"reason\": \"ok\"}"),
        usage: usage(1, 1),
        stop: StopReason::EndTurn,
    });
    let s = setup(vec![text("long plan"), low, text("short"), high], PRICE);
    let mut run = AgentRun::new(
        &persona("evals = [\"judge('Plan is short', 7)\"]"),
        ask(),
        Limits::default(),
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("short"))
    );
    assert!(run.messages()[2]
        .text_content()
        .contains("judge: score 2 is below 7: long"));
}

#[tokio::test]
async fn max_turns_counts_model_calls() {
    let replies = (0..20)
        .map(|i| calls(&[(&format!("c{i}"), "echo")]))
        .collect();
    let s = setup(replies, PRICE);
    let mut run = AgentRun::new(
        &persona("tools = ['echo']"),
        ask(),
        Limits::default(),
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run_to_end(&mut run, &s.io).await,
        StepOutcome::Failed("max_turns 8 reached".into())
    );
    assert_eq!((s.model.seen().len(), run.model_calls()), (8, 8));
}

#[tokio::test]
async fn budget_pauses_and_resumes() {
    let s = setup(vec![text("hi")], PRICE);
    let limits = Limits {
        run: Some(10),
        ..Limits::default()
    };
    let mut run = AgentRun::new(&persona(""), ask(), limits, &s.io)
        .await
        .unwrap();
    let paused = StepOutcome::Paused(PauseReason::Budget {
        limit: "run".into(),
    });
    assert_eq!(run.step(&s.io).await, paused);
    assert_eq!(run.step(&s.io).await, paused);
    assert!(s.model.seen().is_empty());
    assert!(s.sink.events().iter().any(|e| e.body
        == EventBody::RunPaused {
            reason: PauseReason::Budget {
                limit: "run".into()
            }
        }));
    run.resume(Limits {
        run: Some(1_000_000),
        ..Limits::default()
    })
    .unwrap();
    assert!(run.resume(Limits::default()).is_err());
    assert_eq!(
        step_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("hi"))
    );

    let tight = setup(vec![text("hi")], PRICE);
    let p = persona("budget = { call_usd = 0.000001 }");
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &tight.io)
        .await
        .unwrap();
    assert_eq!(
        run.step(&tight.io).await,
        StepOutcome::Paused(PauseReason::Budget {
            limit: "call".into()
        })
    );

    let unverified = setup(
        vec![text("hi")],
        Price {
            verified: false,
            ..PRICE
        },
    );
    let mut run = AgentRun::new(
        &persona(""),
        ask(),
        Limits {
            run: Some(1_000_000),
            ..Limits::default()
        },
        &unverified.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run.step(&unverified.io).await,
        StepOutcome::Paused(PauseReason::Budget {
            limit: "unverified_price".into()
        })
    );

    let spent = setup(vec![text("hi")], PRICE);
    spent
        .ledger
        .record(
            "user:1",
            Usage {
                cost_usd_micros: 999_999,
                ..Usage::default()
            },
        )
        .await
        .unwrap();
    let limits = Limits {
        user: Some(PeriodLimit {
            key: "user:1".into(),
            max_usd_micros: 1_000_000,
        }),
        ..Limits::default()
    };
    let mut run = AgentRun::new(&persona(""), ask(), limits, &spent.io)
        .await
        .unwrap();
    assert_eq!(
        run.step(&spent.io).await,
        StepOutcome::Paused(PauseReason::Budget {
            limit: "user".into()
        })
    );
}

#[tokio::test]
async fn checkpoint_every_step_matches_uninterrupted() {
    let script = || {
        vec![
            calls(&[("c1", "echo"), ("c2", "echo")]),
            text("{\"n\": \"bad\", \"items\": []}"),
            text("{\"n\": 3, \"items\": [\"x\"]}"),
        ]
    };
    let p = persona(&format!("tools = ['echo']\n{TYPED}"));
    let a = setup(script(), PRICE);
    let start = AgentRun::new(&p, ask(), Limits::default(), &a.io)
        .await
        .unwrap();
    let mut straight = start.clone();
    let out_a = run_to_end(&mut straight, &a.io).await;

    let b = setup(script(), PRICE);
    let store = InMemCheckpoints::default();
    store.save(start.checkpoint()).await.unwrap();
    let out_b = loop {
        let cp = store.latest(start.run_id()).await.unwrap().unwrap();
        let json = serde_json::to_string(&cp.state).unwrap();
        let mut resumed = AgentRun::from_checkpoint(&troupekit_core::store::Checkpoint {
            state: serde_json::from_str(&json).unwrap(),
            ..cp
        })
        .unwrap();
        let out = resumed.step(&b.io).await;
        store.save(resumed.checkpoint()).await.unwrap();
        if out != StepOutcome::Continue {
            break out;
        }
    };
    assert_eq!(out_a, StepOutcome::Done(json!({"n": 3, "items": ["x"]})));
    assert_eq!(out_a, out_b);
    let last =
        AgentRun::from_checkpoint(&store.latest(start.run_id()).await.unwrap().unwrap()).unwrap();
    assert_eq!(
        serde_json::to_value(&last).unwrap(),
        serde_json::to_value(&straight).unwrap()
    );
    assert_eq!(a.sink.events(), b.sink.events());
    assert!(b
        .sink
        .events()
        .iter()
        .enumerate()
        .all(|(i, e)| e.seq == i as u64));
}

#[tokio::test]
async fn paused_run_survives_checkpoint_and_resumes() {
    let judge = Ok(CompletionResponse {
        message: Message::text(Role::Assistant, "{\"score\": 9, \"reason\": \"ok\"}"),
        usage: usage(1, 1),
        stop: StopReason::EndTurn,
    });
    let s = setup(vec![text("plan"), judge], PRICE);
    let p = persona("evals = [\"judge('Plan is short', 7)\"]");
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(run.step(&s.io).await, StepOutcome::Continue);
    run.resume(Limits::default()).unwrap_err();
    let tight = Limits {
        run: Some(run.usage().cost_usd_micros + 1),
        ..Limits::default()
    };
    let mut cp = run.checkpoint();
    cp.state["limits"] = serde_json::to_value(&tight).unwrap();
    let mut run = AgentRun::from_checkpoint(&cp).unwrap();
    let paused = run.step(&s.io).await;
    assert_eq!(
        paused,
        StepOutcome::Paused(PauseReason::Budget {
            limit: "run".into()
        })
    );
    let mut back = AgentRun::from_checkpoint(&run.checkpoint()).unwrap();
    assert!(
        matches!(back.state(), RunState::Paused { resume, .. } if matches!(**resume, RunState::Check { .. }))
    );
    back.resume(Limits::default()).unwrap();
    assert_eq!(
        step_to_end(&mut back, &s.io).await,
        StepOutcome::Done(json!("plan"))
    );
    assert_eq!(s.model.seen().len(), 2);
}

#[tokio::test]
async fn max_tokens_fits_a_tight_call_budget() {
    let opus = Price {
        input_usd_micros_per_mtok: 4_000_000,
        output_usd_micros_per_mtok: 20_000_000,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    };
    let s = setup(vec![text("hi")], opus);
    let mut run = AgentRun::new(
        &persona("budget = { call_usd = 0.05 }"),
        ask(),
        Limits::default(),
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        run.step(&s.io).await,
        StepOutcome::Paused(PauseReason::Budget {
            limit: "call".into()
        })
    );
    let mut run = AgentRun::new(
        &persona("budget = { call_usd = 0.05 }\nmax_tokens = 1000"),
        ask(),
        Limits::default(),
        &s.io,
    )
    .await
    .unwrap();
    assert_eq!(
        step_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("hi"))
    );
    assert_eq!(s.model.seen()[0].max_tokens, 1000);
}

#[tokio::test]
async fn checkpoint_refuses_other_formats() {
    let s = setup(vec![], PRICE);
    let run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    let mut cp = run.checkpoint();
    cp.state["format"] = json!(99);
    assert!(matches!(
        AgentRun::from_checkpoint(&cp),
        Err(AgentError::Checkpoint(_))
    ));
    let mut cp = run.checkpoint();
    cp.state["surprise"] = json!(1);
    assert!(matches!(
        AgentRun::from_checkpoint(&cp),
        Err(AgentError::Checkpoint(_))
    ));
}

#[tokio::test]
async fn retryable_error_keeps_state() {
    let limited =
        Err(ProviderError::new(ProviderErrorKind::RateLimited, "slow down").with_retry_after(1500));
    let s = setup(vec![limited, text("hi")], PRICE);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        run.step(&s.io).await,
        StepOutcome::Retry {
            error: "RateLimited: slow down".into(),
            retry_after_ms: Some(1500)
        }
    );
    assert_eq!((run.state(), run.model_calls()), (&RunState::CallModel, 0));
    assert_eq!(
        step_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("hi"))
    );
    let fatal = setup(
        vec![Err(ProviderError::new(ProviderErrorKind::Auth, "bad key"))],
        PRICE,
    );
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &fatal.io)
        .await
        .unwrap();
    assert_eq!(
        run.step(&fatal.io).await,
        StepOutcome::Failed("Auth: bad key".into())
    );
}

#[tokio::test]
async fn unknown_tool_or_eval_fails_at_new() {
    let s = setup(vec![], PRICE);
    for extra in [
        "tools = ['nope']",
        "evals = ['no_overlap(actions)']",
        "evals = [\"judge('x', 5, 'other/model')\"]",
    ] {
        let err = AgentRun::new(&persona(extra), ask(), Limits::default(), &s.io)
            .await
            .err()
            .unwrap();
        assert!(
            matches!(err, AgentError::Setup(ref m) if m.contains("test/coach.toml")),
            "{extra}: {err}"
        );
    }
    let unloaded = Persona {
        instructions: String::new(),
        ..persona("")
    };
    assert!(AgentRun::new(&unloaded, ask(), Limits::default(), &s.io)
        .await
        .is_err());
    assert!(
        AgentRun::new(&persona(""), vec![], Limits::default(), &s.io)
            .await
            .is_err()
    );
    let mut io = SimpleIo::new(ToolRegistry::new()).with_model(
        "fake/m",
        FakeModel::new(vec![text("{}"), text("{}")]),
        PRICE,
    );
    io.evals
        .register(
            "no_overlap",
            Arc::new(|_: &Value, _: &[Value]| Err("overlap at 9:00".to_string())),
        )
        .unwrap();
    let mut run = AgentRun::new(
        &persona("evals = ['no_overlap()']\nmax_turns = 2"),
        ask(),
        Limits::default(),
        &io,
    )
    .await
    .unwrap();
    assert!(
        matches!(run_to_end(&mut run, &io).await, StepOutcome::Failed(e) if e.contains("no_overlap: overlap at 9:00"))
    );
}

#[tokio::test]
async fn http_model_end_to_end() {
    let sse = "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hel\"}}]}\n\n\
               data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"lo\"},\"finish_reason\":\"stop\"}]}\n\n\
               data: [DONE]\n\n";
    let transport = Arc::new(ScriptedTransport::new(vec![Scripted::chunked(sse, 7)]));
    let endpoint = Endpoint {
        protocol: Protocol::OpenAi,
        base_url: "https://h.test/".into(),
        auth: AuthStyle::Bearer,
        quirks: Quirks::default(),
    };
    let key = Some(Secret::new("sk-test-not-real".into()).unwrap());
    let model = Arc::new(HttpModel::new("fake/m", "m", endpoint, key, transport.clone()).unwrap());
    let sink = Arc::new(VecSink::default());
    let mut io = SimpleIo::new(ToolRegistry::new()).with_model("fake/m", model, Price::FREE);
    io.events = sink.clone();
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &io)
        .await
        .unwrap();
    assert_eq!(
        run_to_end(&mut run, &io).await,
        StepOutcome::Done(json!("hello"))
    );
    let deltas: Vec<EventBody> = sink
        .events()
        .into_iter()
        .map(|e| e.body)
        .filter(|b| matches!(b, EventBody::TextDelta { .. }))
        .collect();
    assert_eq!(
        deltas,
        [
            EventBody::TextDelta { text: "hel".into() },
            EventBody::TextDelta { text: "lo".into() }
        ]
    );
    assert_eq!(transport.sent()[0].url, "https://h.test/chat/completions");
    assert!(!serde_json::to_string(&run)
        .unwrap()
        .contains("sk-test-not-real"));
}

#[tokio::test]
async fn tool_calls_carry_idempotency_keys() {
    let keys = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = keys.clone();
    let mut tools = ToolRegistry::new();
    tools
        .register(Arc::new(typed_tool(
            "echo",
            "test tool",
            move |_args: Q, ctx: ToolContext| {
                let seen = seen.clone();
                async move {
                    seen.lock()
                        .unwrap()
                        .push(ctx.idempotency_key.clone().unwrap());
                    Ok(ToolOutput::text("ok"))
                }
            },
        )))
        .unwrap();
    let model = FakeModel::new(vec![
        calls(&[("c1", "echo"), ("c2", "echo")]),
        calls(&[("c3", "echo")]),
        text("done"),
    ]);
    let io = SimpleIo::new(tools).with_model("fake/m", model, PRICE);
    let p = persona("tools = ['echo']");
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &io)
        .await
        .unwrap();
    let id = run.run_id();
    assert_eq!(run.step(&io).await, StepOutcome::Continue);
    let before_tools = run.checkpoint();
    assert_eq!(run.step(&io).await, StepOutcome::Continue);
    assert_eq!(
        *keys.lock().unwrap(),
        vec![format!("{id}:2:0"), format!("{id}:2:1")]
    );
    let mut replay = AgentRun::from_checkpoint(&before_tools).unwrap();
    assert_eq!(replay.step(&io).await, StepOutcome::Continue);
    let after_replay = keys.lock().unwrap().clone();
    assert_eq!(after_replay[2..], after_replay[..2]);
    assert_eq!(run.step(&io).await, StepOutcome::Continue);
    assert_eq!(run.step(&io).await, StepOutcome::Continue);
    assert_eq!(keys.lock().unwrap()[4], format!("{id}:4:0"));
}

#[tokio::test]
async fn retries_count_survives_checkpoints() {
    let limited =
        || Err(ProviderError::new(ProviderErrorKind::RateLimited, "slow down").with_retry_after(1));
    let s = setup(vec![limited(), limited(), text("hi")], PRICE);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(run.retries(), 0);
    assert!(matches!(run.step(&s.io).await, StepOutcome::Retry { .. }));
    assert_eq!(run.retries(), 1);
    let mut run = AgentRun::from_checkpoint(&run.checkpoint()).unwrap();
    assert_eq!(run.retries(), 1);
    assert!(matches!(run.step(&s.io).await, StepOutcome::Retry { .. }));
    assert_eq!(run.retries(), 2);
    assert_eq!(
        step_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("hi"))
    );
    assert_eq!(run.retries(), 2);
}

#[tokio::test]
async fn checkpoint_without_retries_still_loads() {
    let s = setup(vec![], PRICE);
    let run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    let mut cp = run.checkpoint();
    let state = cp.state.as_object_mut().unwrap();
    assert!(state.remove("retries").is_some());
    let old = AgentRun::from_checkpoint(&cp).unwrap();
    assert_eq!(old.retries(), 0);
    assert_eq!(old.run_id(), run.run_id());
}

fn traced(content: bool) -> (troupekit_trace::Tracer, Arc<troupekit_trace::MemorySink>) {
    let sink = Arc::new(troupekit_trace::MemorySink::default());
    let tracer = troupekit_trace::Tracer::builder()
        .content(content)
        .sink(sink.clone())
        .build()
        .unwrap();
    (tracer, sink)
}

fn trace_records(sink: &troupekit_trace::MemorySink) -> Vec<Value> {
    sink.lines()
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn starts<'a>(records: &'a [Value], name: &str) -> Vec<&'a Value> {
    records
        .iter()
        .filter(|r| r["t"] != "end" && r["name"] == name)
        .collect()
}

fn end_of<'a>(records: &'a [Value], start: &Value) -> &'a Value {
    records
        .iter()
        .find(|r| r["t"] == "end" && r["span"] == start["span"])
        .unwrap()
}

#[tokio::test]
async fn model_call_span_has_tokens_and_no_content_by_default() {
    let s = setup(vec![text("hello there")], PRICE);
    let (tracer, sink) = traced(false);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    tracer.in_scope(run_to_end(&mut run, &s.io)).await;
    drop(tracer);
    let records = trace_records(&sink);
    let step = starts(&records, "agent.step")[0];
    assert_eq!(step["cat"], "agents");
    assert_eq!(step["fields"]["run"], run.run_id().to_string().as_str());
    assert_eq!(step["fields"]["state"], "call_model");
    assert_eq!(step["fields"]["persona"], "coach");
    let call = starts(&records, "model.call")[0];
    assert_eq!(call["parent"], step["span"]);
    assert_eq!(call["cat"], "models");
    assert_eq!(call["fields"]["model"], "fake/m");
    assert_eq!(call["fields"]["stream"], true);
    assert!(call["fields"]["messages"]["bytes"].as_u64().unwrap() > 0);
    let end = end_of(&records, call);
    assert_eq!(end["ok"], true);
    assert_eq!(end["fields"]["input_tokens"], 10);
    assert_eq!(end["fields"]["output_tokens"], 5);
    assert_eq!(end["fields"]["cost_usd_micros"], 20);
    assert_eq!(end["fields"]["stop"], "EndTurn");
    assert!(end["fields"]["ttft_us"].is_u64());
    assert!(end["fields"]["reply"]["bytes"].is_u64());
    for line in sink.lines() {
        assert!(!line.contains("hello there") && !line.contains("Plan my week"), "{line}");
    }
}

#[tokio::test]
async fn tool_arguments_are_content() {
    for content in [false, true] {
        let s = setup(vec![calls(&[("c1", "echo")]), text("done")], PRICE);
        let (tracer, sink) = traced(content);
        let mut run = AgentRun::new(&persona("tools = ['echo']"), ask(), Limits::default(), &s.io)
            .await
            .unwrap();
        tracer.in_scope(run_to_end(&mut run, &s.io)).await;
        drop(tracer);
        let records = trace_records(&sink);
        let tool = starts(&records, "tool.call")[0];
        assert_eq!(tool["cat"], "tools");
        assert_eq!(tool["fields"]["name"], "echo");
        assert_eq!(tool["fields"]["call_id"], "c1");
        let step = starts(&records, "agent.step")
            .into_iter()
            .find(|r| r["fields"]["state"] == "call_tools")
            .unwrap();
        assert_eq!(tool["parent"], step["span"]);
        let end = end_of(&records, tool);
        assert_eq!(end["fields"]["is_error"], false);
        if content {
            assert_eq!(tool["fields"]["args"], "{\"q\":\"c1\"}");
            assert_eq!(end["fields"]["output"], "got c1");
        } else {
            assert_eq!(tool["fields"]["args"]["bytes"], 10);
            assert!(sink.lines().iter().all(|l| !l.contains("got c1")));
        }
    }
}

#[tokio::test]
async fn max_turns_law_event_is_emitted() {
    let replies = (0..20)
        .map(|i| calls(&[(&format!("c{i}"), "echo")]))
        .collect();
    let s = setup(replies, PRICE);
    let (tracer, sink) = traced(false);
    let mut run = AgentRun::new(&persona("tools = ['echo']"), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    tracer.in_scope(run_to_end(&mut run, &s.io)).await;
    drop(tracer);
    let records = trace_records(&sink);
    let laws = starts(&records, "law.check");
    assert!(laws.iter().all(|r| r["t"] == "event" && r["cat"] == "laws"));
    assert!(laws
        .iter()
        .all(|r| r["fields"]["law"] == "step_keeps_model_calls_within_max_turns"));
    assert_eq!(laws.iter().filter(|r| r["fields"]["held"] == true).count(), 8);
    assert_eq!(laws.iter().filter(|r| r["fields"]["held"] == false).count(), 1);
}

#[tokio::test]
async fn budget_check_emits_span() {
    let s = setup(vec![text("hi")], PRICE);
    let (tracer, sink) = traced(false);
    let limits = Limits {
        run: Some(10),
        ..Limits::default()
    };
    let mut run = AgentRun::new(&persona(""), ask(), limits, &s.io)
        .await
        .unwrap();
    tracer.in_scope(run.step(&s.io)).await;
    drop(tracer);
    let records = trace_records(&sink);
    let check = starts(&records, "budget.check")[0];
    assert_eq!(check["cat"], "budget");
    assert!(check["fields"]["estimate"].as_u64().unwrap() > 10);
    assert_eq!(check["fields"]["run_spent"], 0);
    assert_eq!(end_of(&records, check)["fields"]["crossed"], "run");
    assert!(starts(&records, "model.call").is_empty());
}

#[tokio::test]
async fn eval_emits_span() {
    let s = setup(vec![text("{\"n\": 1, \"items\": [\"a\", \"b\"]}")], PRICE);
    let (tracer, sink) = traced(false);
    let p = persona(&format!("evals = ['min_items(items, 2)']\n{TYPED}"));
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    let out = tracer.in_scope(run_to_end(&mut run, &s.io)).await;
    drop(tracer);
    assert!(matches!(out, StepOutcome::Done(_)), "{out:?}");
    let records = trace_records(&sink);
    let eval = starts(&records, "eval.run")[0];
    assert_eq!(eval["cat"], "evals");
    assert_eq!(eval["fields"]["name"], "min_items");
    assert_eq!(end_of(&records, eval)["fields"]["passed"], true);
}

#[tokio::test]
async fn failed_step_error_is_content() {
    let s = setup(
        vec![
            text("{\"n\": \"x\", \"items\": []}"),
            text("{\"n\": 1, \"items\": [\"a\"]}"),
        ],
        PRICE,
    );
    let (tracer, sink) = traced(false);
    let p = persona(&format!("evals = ['min_items(items, 2)']\n{TYPED}"));
    let mut run = AgentRun::new(&p, ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    let out = tracer.in_scope(run_to_end(&mut run, &s.io)).await;
    drop(tracer);
    assert!(matches!(out, StepOutcome::Failed(_)), "{out:?}");
    let records = trace_records(&sink);
    let failed: Vec<&Value> = records
        .iter()
        .filter(|r| r["t"] == "end" && r["ok"] == false)
        .collect();
    assert_eq!(failed.len(), 1);
    assert_eq!(failed[0]["err"], "step failed");
    assert!(failed[0]["fields"]["error"]["bytes"].as_u64().unwrap() > 0);
    assert!(sink.lines().iter().all(|l| !l.contains("has 1 items")));
}

#[tokio::test]
async fn checkpoint_without_trace_has_no_trace_key() {
    let s = setup(vec![text("hi")], PRICE);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(run.trace(), None);
    assert!(run.checkpoint().state.get("trace").is_none());
    run.set_trace(troupekit_trace::TraceSpec::default().with_content(true));
    assert_eq!(run.checkpoint().state["trace"]["content"], true);
}

#[tokio::test]
async fn resumed_run_keeps_its_trace_spec() {
    use troupekit_trace::{Trace, TraceSet, TraceSpec};
    let s = setup(vec![calls(&[("c1", "echo")]), text("done")], PRICE);
    let store = InMemCheckpoints::default();
    let spec = TraceSpec::new(TraceSet::none().with(Trace::Tools)).with_file("traces/coach.jsonl");
    let mut run = AgentRun::new(&persona("tools = ['echo']"), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    run.set_trace(spec.clone());
    store.save(run.checkpoint()).await.unwrap();
    let (tracer, sink) = traced(false);
    assert_eq!(tracer.in_scope(run.step(&s.io)).await, StepOutcome::Continue);
    store.save(run.checkpoint()).await.unwrap();
    let cp = store.latest(run.run_id()).await.unwrap().unwrap();
    let mut resumed = AgentRun::from_checkpoint(&cp).unwrap();
    assert_eq!(resumed.trace(), Some(spec));
    tracer.in_scope(step_to_end(&mut resumed, &s.io)).await;
    drop(tracer);
    let records = trace_records(&sink);
    assert_eq!(starts(&records, "tool.call").len(), 1, "the resumed run still traces tools");
}

#[tokio::test]
async fn set_trace_accepts_an_absolute_file() {
    let s = setup(vec![text("hi")], PRICE);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    let spec = troupekit_trace::TraceSpec::default().with_file("/tmp/app sandbox/run.jsonl");
    run.set_trace(spec.clone());
    assert_eq!(run.trace(), Some(spec.clone()));
    let restored = AgentRun::from_checkpoint(&run.checkpoint()).unwrap();
    assert_eq!(restored.trace(), Some(spec));
}

struct FailingLedger;

#[async_trait]
impl UsageLedger for FailingLedger {
    async fn period_total(&self, key: &str) -> Result<u64, troupekit_core::error::CoreError> {
        Err(troupekit_core::error::CoreError::Store(format!(
            "no row for {key}"
        )))
    }

    async fn record(
        &self,
        _key: &str,
        _usage: Usage,
    ) -> Result<(), troupekit_core::error::CoreError> {
        Ok(())
    }
}

async fn ledger_failure_trace(content: bool) -> Vec<String> {
    let (tracer, sink) = traced(content);
    let limits = Limits {
        user: Some(PeriodLimit {
            key: "user-sk-PRIVATE-42".into(),
            max_usd_micros: 1_000,
        }),
        ..Limits::default()
    };
    let checked = tracer
        .in_scope(troupekit_agent::budget::check(
            &limits,
            &PRICE,
            0,
            1,
            &FailingLedger,
        ))
        .await;
    assert!(checked.is_err());
    drop(tracer);
    sink.lines()
}

#[tokio::test]
async fn ledger_error_detail_stays_out_of_a_trace_without_content() {
    let lines = ledger_failure_trace(false).await;
    assert!(
        lines.iter().any(|l| l.contains("\"ok\":false")),
        "{lines:?}"
    );
    assert!(
        lines.iter().all(|l| !l.contains("sk-PRIVATE-42")),
        "{lines:?}"
    );
    let lines = ledger_failure_trace(true).await;
    assert!(
        lines.iter().any(|l| l.contains("sk-PRIVATE-42")),
        "{lines:?}"
    );
}
