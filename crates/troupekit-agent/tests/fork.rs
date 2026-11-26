use crate::run::{calls, persona, setup, step_to_end, text, Q, PRICE};
use serde_json::json;
use std::sync::{Arc, Mutex};
use troupekit_agent::run::{AgentRun, RunState, StepOutcome};
use troupekit_agent::{AgentError, Limits};
use troupekit_core::ids::RunId;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::Checkpoint;
use troupekit_tools::{typed_tool, ToolContext, ToolOutput};

fn ask() -> Vec<Message> {
    vec![Message::text(Role::User, "Plan my week")]
}

#[tokio::test]
async fn fork_has_new_id_and_parent() {
    let s = setup(vec![text("hello")], PRICE);
    let mut run = AgentRun::new(&persona(""), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(run.parent(), None);
    assert_eq!(run.step(&s.io).await, StepOutcome::Continue);
    let fork = run.fork();
    assert_ne!(fork.run_id(), run.run_id());
    assert_eq!(fork.parent(), Some((run.run_id(), 1)));
    assert_eq!(fork.messages(), run.messages());
    assert_eq!(fork.state(), run.state());
    assert_eq!(fork.usage(), run.usage());
    let cp = fork.checkpoint();
    assert_eq!(cp.step, 0);
    assert_eq!(cp.run_id, fork.run_id());
    let back = AgentRun::from_checkpoint(&cp).unwrap();
    assert_eq!(back.parent(), Some((run.run_id(), 1)));
    assert_eq!(back, fork);
}

#[test]
fn v1_agent_checkpoint_reserializes_identically() {
    let run_id = "0190a5b2-7c3d-7e4f-8a1b-2c3d4e5f6a7b";
    let state = json!({
        "format": 1,
        "run_id": run_id,
        "spec": {
            "persona": "coach",
            "model": "fake/m",
            "system": "You are Coach.",
            "tools": ["echo"],
            "evals": [],
            "output": null,
            "max_turns": 8,
            "max_tokens": 1024
        },
        "limits": {"call": null, "run": 5000, "scope": null, "user": null},
        "messages": [
            {"role": "user", "parts": [{"type": "text", "text": "Plan my week"}]},
            {"role": "assistant", "parts": [{"type": "text", "text": "Done."}]}
        ],
        "state": {"state": "check", "answer": "Done."},
        "model_calls": 1,
        "repaired": false,
        "usage": {"input_tokens": 10, "output_tokens": 5, "cost_usd_micros": 20},
        "seq": 6,
        "steps": 1
    });
    let cp = Checkpoint {
        run_id: serde_json::from_value::<RunId>(json!(run_id)).unwrap(),
        step: 1,
        state: state.clone(),
    };
    let run = AgentRun::from_checkpoint(&cp).unwrap();
    assert_eq!(run.parent(), None);
    let again = run.checkpoint();
    assert_eq!(again.state, state);
    assert!(again.state.get("parent").is_none());
    assert_eq!(again.step, cp.step);
}

async fn tool_turn_then_answer() -> (Checkpoint, crate::run::Setup) {
    let s = setup(
        vec![calls(&[("c1", "echo")]), text("first"), text("second")],
        PRICE,
    );
    let mut run = AgentRun::new(&persona("tools = ['echo']"), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(
        step_to_end(&mut run, &s.io).await,
        StepOutcome::Done(json!("first"))
    );
    assert_eq!(run.messages().len(), 4);
    (run.checkpoint(), s)
}

#[tokio::test]
async fn fork_at_message_truncates_and_continues() {
    let (cp, s) = tool_turn_then_answer().await;
    let extra = Message::text(Role::User, "Add the gym too");
    let mut fork = AgentRun::fork_at_message(&cp, 3, Some(extra.clone())).unwrap();
    assert_eq!(fork.parent(), Some((cp.run_id, cp.step)));
    assert_ne!(fork.run_id(), cp.run_id);
    assert_eq!(fork.state(), &RunState::CallModel);
    assert_eq!(fork.messages().len(), 4);
    assert_eq!(fork.messages()[2].role, Role::Tool);
    assert_eq!(fork.messages()[3], extra);
    assert_eq!(
        step_to_end(&mut fork, &s.io).await,
        StepOutcome::Done(json!("second"))
    );
    let seen = s.model.seen();
    assert_eq!(seen.len(), 3);
    assert_eq!(seen[2].messages[..3], seen[1].messages[..3]);
    assert_eq!(seen[2].messages[3], extra);
    assert_eq!(AgentRun::from_checkpoint(&cp).unwrap().messages().len(), 4);
}

#[tokio::test]
async fn fork_at_message_inside_tool_exchange_is_refused() {
    let (cp, _s) = tool_turn_then_answer().await;
    let refused = |r: Result<AgentRun, AgentError>| match r {
        Err(AgentError::Setup(m)) => m,
        other => panic!("expected a setup error, got {other:?}"),
    };
    let inside = refused(AgentRun::fork_at_message(&cp, 2, None));
    assert!(inside.contains("user or tool message"), "{inside}");
    let after_answer = refused(AgentRun::fork_at_message(&cp, 4, None));
    assert!(after_answer.contains("user or tool message"), "{after_answer}");
    let zero = refused(AgentRun::fork_at_message(&cp, 0, None));
    assert!(zero.contains("1 to 4"), "{zero}");
    let past = refused(AgentRun::fork_at_message(&cp, 5, None));
    assert!(past.contains("1 to 4"), "{past}");
    let not_user = Message::text(Role::Assistant, "I said so");
    let role = refused(AgentRun::fork_at_message(&cp, 1, Some(not_user)));
    assert!(role.contains("user message"), "{role}");
    let ok = AgentRun::fork_at_message(&cp, 1, None).unwrap();
    assert_eq!(ok.messages().len(), 1);
}

#[tokio::test]
async fn forked_tool_calls_get_new_idempotency_keys() {
    let keys = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen = keys.clone();
    let mut s = setup(vec![calls(&[("c1", "keys")])], PRICE);
    s.io.tools
        .register(Arc::new(typed_tool(
            "keys",
            "records keys",
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
    let mut run = AgentRun::new(&persona("tools = ['keys']"), ask(), Limits::default(), &s.io)
        .await
        .unwrap();
    assert_eq!(run.step(&s.io).await, StepOutcome::Continue);
    assert!(matches!(run.state(), RunState::CallTools { .. }));
    let mut fork = run.fork();
    assert_eq!(run.step(&s.io).await, StepOutcome::Continue);
    assert_eq!(fork.step(&s.io).await, StepOutcome::Continue);
    let keys = keys.lock().unwrap().clone();
    assert_eq!(keys.len(), 2);
    assert_eq!(keys[0], format!("{}:2:0", run.run_id()));
    assert_eq!(keys[1], format!("{}:1:0", fork.run_id()));
    assert_ne!(keys[0], keys[1]);
}
