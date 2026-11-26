use crate::common::{calc_model, echo_model};
use troupekit::{agent, Agent, Storage};
use troupekit_agent::Limits;

/// Use the add tool.
#[agent(model = "fake/m", tools = [crate::tools_mod::add])]
fn calc() {}

/// Answer briefly.
/// Never guess.
#[agent(
    model = "fake/m",
    role = "Checks facts.",
    goal = "A short answer.",
    display_name = "Fact Checker",
    evals = ["min_items(items, 1)"],
    limits = Limits { run: Some(5_000), ..Limits::default() },
    checkpoints = Storage::memory(),
    max_turns = 3,
    max_tokens = 256,
    retries = 1,
    laws = "tests/fixtures/rules.bend",
    auto_laws = false,
)]
pub(crate) fn fact_checker() {}

/// Ignored when instructions is set.
#[agent(model = "fake/m", name = "renamed", instructions = "From the key.")]
fn original() {}

#[tokio::test]
async fn agent_with_tool_from_other_module_runs_offline() {
    let text = calc()
        .model_client(calc_model())
        .run("What is 2 + 40?")
        .await
        .unwrap();
    assert_eq!(text, "The sum is 42");
}

#[tokio::test]
async fn doc_comment_becomes_instructions() {
    let built = fact_checker()
        .model_client(echo_model())
        .build()
        .await
        .unwrap();
    assert_eq!(
        built.persona().instructions,
        "Answer briefly.\nNever guess."
    );
    assert_eq!(built.persona().name, "fact-checker");
    let renamed = original().model_client(echo_model()).build().await.unwrap();
    assert_eq!(renamed.persona().name, "renamed");
    assert_eq!(renamed.persona().instructions, "From the key.");
}

#[tokio::test]
async fn attribute_values_reach_the_builder() {
    let built = fact_checker()
        .model_client(echo_model())
        .build()
        .await
        .unwrap();
    let persona = built.persona();
    assert_eq!(persona.role, "Checks facts.");
    assert_eq!(persona.goal, "A short answer.");
    assert_eq!(persona.display_name, "Fact Checker");
    assert_eq!(persona.max_turns, 3);
    assert_eq!(persona.max_tokens, 256);
    assert_eq!(persona.evals[0].name, "min_items");
    assert!(!built.laws().auto());
    assert_eq!(built.laws().paths().len(), 1);
    assert!(built.laws().paths()[0].ends_with("tests/fixtures/rules.bend"));
}

#[tokio::test]
async fn macro_and_builder_build_the_same_persona() {
    let from_macro = calc().model_client(calc_model()).build().await.unwrap();
    let from_builder = Agent::new("calc")
        .instructions("Use the add tool.")
        .model("fake/m")
        .tool(crate::tools_mod::add)
        .model_client(calc_model())
        .build()
        .await
        .unwrap();
    assert_eq!(from_macro.persona(), from_builder.persona());
}
