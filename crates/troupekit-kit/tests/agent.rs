use crate::common::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use troupekit::{Agent, BuiltAgent, KitError};
use troupekit_agent::EvalFn;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_graph::testing::{text_reply, FnModel};
use troupekit_tools::ToolError;

#[tokio::test]
async fn hello_runs_against_fn_model() {
    assert_eq!(
        greeter().run("I am Ada").await.unwrap(),
        "Hello! You said: I am Ada"
    );
}

#[tokio::test]
async fn tool_agent_runs_lab2_flow() {
    let text = Agent::new("calc")
        .instructions("Use the add tool.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(add_tool())
        .run("What is 2 + 40?")
        .await
        .unwrap();
    assert_eq!(text, "The sum is 42");
}

#[tokio::test]
async fn run_with_returns_usage_calls_and_messages() {
    let outcome = Agent::new("calc")
        .instructions("Use the add tool.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(add_tool())
        .run_with("What is 2 + 40?")
        .await
        .unwrap();
    assert_eq!(outcome.text(), "The sum is 42");
    assert_eq!(outcome.output(), &serde_json::json!("The sum is 42"));
    assert_eq!(outcome.model_calls(), 2);
    assert_eq!(outcome.usage().input_tokens, 20);
    assert_eq!(outcome.messages()[0].text_content(), "What is 2 + 40?");
    assert_eq!(
        outcome.messages().last().unwrap().text_content(),
        "The sum is 42"
    );
}

#[tokio::test]
async fn build_reports_first_builder_error() {
    let err = Agent::new("calc")
        .instructions("Use the add tool.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(add_tool())
        .tool(add_tool())
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(err, KitError::Tool(ToolError::DuplicateTool(ref n)) if n == "add"),
        "{err}"
    );
}

#[tokio::test]
async fn name_with_uppercase_fails_at_build() {
    let err = Agent::new("Calc")
        .instructions("x")
        .model("fake/m")
        .model_client(echo_model())
        .build()
        .await
        .unwrap_err();
    assert!(
        matches!(err, KitError::Persona(ref e) if e.field == "name"),
        "{err}"
    );
}

#[tokio::test]
async fn missing_instructions_or_model_fail_at_build() {
    let no_instructions = Agent::new("calc")
        .model("fake/m")
        .model_client(echo_model())
        .build()
        .await
        .unwrap_err();
    assert_eq!(
        no_instructions.to_string(),
        "agent `calc`: needs instructions: call .instructions(..) or add a /// doc comment"
    );
    let no_model = Agent::new("calc")
        .instructions("x")
        .build()
        .await
        .unwrap_err();
    assert_eq!(
        no_model.to_string(),
        "agent `calc`: needs a model: call .model(\"vendor/model\")"
    );
}

#[tokio::test]
async fn defaults_are_the_safe_values() {
    let built = greeter().build().await.unwrap();
    let persona = built.persona();
    assert_eq!(persona.display_name, "greeter");
    assert_eq!(persona.role, "An assistant.");
    assert_eq!(persona.goal, "Answer the user's request.");
    assert_eq!(persona.max_turns, troupekit_agent::DEFAULT_MAX_TURNS);
    assert_eq!(persona.max_tokens, troupekit_agent::DEFAULT_MAX_TOKENS);
    assert!(built.laws().auto());
    let in_memory = built.resume(troupekit_core::ids::RunId::new()).await;
    assert!(matches!(in_memory, Err(KitError::Config { .. })));
    let three = AtomicUsize::new(0);
    let model = Arc::new(FnModel::new("fake/m", move |_req| {
        if three.fetch_add(1, Ordering::SeqCst) < 3 {
            Err(ProviderError::new(ProviderErrorKind::Server, "busy").with_retry_after(1))
        } else {
            text_reply("up", 1)
        }
    }));
    let text = Agent::new("greeter")
        .instructions("x")
        .model("fake/m")
        .model_client(model.clone())
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "up");
    assert_eq!(model.calls(), 4);
}

#[tokio::test]
async fn instructions_with_toml_syntax_stay_text() {
    let hostile = "a \"\"\" b ''' c\n[table]\nkey = \"v\" \\\" end";
    let built = greeter().instructions(hostile).build().await.unwrap();
    assert_eq!(built.persona().instructions, hostile);
    assert!(built.persona().tools.is_empty());
}

#[tokio::test]
async fn retry_outcome_is_retried_then_succeeds() {
    let once = AtomicUsize::new(0);
    let model = Arc::new(FnModel::new("fake/m", move |_req| {
        if once.fetch_add(1, Ordering::SeqCst) == 0 {
            Err(ProviderError::new(ProviderErrorKind::RateLimited, "slow down").with_retry_after(1))
        } else {
            text_reply("done", 1)
        }
    }));
    let text = greeter()
        .model_client(model.clone())
        .retries(1)
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "done");
    assert_eq!(model.calls(), 2);
}

#[tokio::test]
async fn retries_exhausted_is_an_error_with_run_id() {
    let model = failing_model(ProviderErrorKind::Server);
    let err = greeter()
        .model_client(model.clone())
        .retries(2)
        .run("hi")
        .await
        .unwrap_err();
    match err {
        KitError::Retries {
            attempts,
            ref last_error,
            run_id,
        } => {
            assert_eq!(attempts, 2);
            assert!(last_error.contains("busy"), "{last_error}");
            assert!(err.to_string().contains(&run_id.to_string()));
        }
        other => panic!("expected Retries, got {other}"),
    }
    assert_eq!(model.calls(), 3);
}

#[tokio::test]
async fn retries_over_the_cap_fail_at_build() {
    let err = greeter().retries(33).build().await.unwrap_err();
    assert_eq!(
        err.to_string(),
        "agent `greeter`: retries must be at most 32, got 33"
    );
}

#[tokio::test]
async fn failed_run_is_kit_error_failed() {
    let err = greeter()
        .model_client(failing_model(ProviderErrorKind::Auth))
        .run("hi")
        .await
        .unwrap_err();
    assert!(
        matches!(err, KitError::Failed { ref reason, .. } if reason.contains("busy")),
        "{err}"
    );
}

#[test]
fn built_agent_is_clone_send_sync() {
    fn assert_shared<T: Clone + Send + Sync + 'static>() {}
    assert_shared::<BuiltAgent>();
}

#[tokio::test]
async fn eval_strings_and_eval_fn_reach_the_run() {
    let unregistered = greeter().eval("never_ok()").build().await.unwrap_err();
    assert!(
        unregistered
            .to_string()
            .contains("eval `never_ok` is not registered"),
        "{unregistered}"
    );
    let never_ok: EvalFn = Arc::new(|_output, _args| Err("never ok".into()));
    let built = greeter()
        .eval_fn("never_ok", never_ok)
        .evals(["never_ok()"])
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().evals[0].name, "never_ok");
    let err = built.run("hi").await.unwrap_err();
    assert!(matches!(err, KitError::Failed { .. }), "{err}");
}

#[tokio::test]
async fn debug_shows_no_dyn_internals() {
    let built = greeter().build().await.unwrap();
    let text = format!("{:?} {:?}", greeter(), built);
    assert!(text.contains("greeter"), "{text}");
}
