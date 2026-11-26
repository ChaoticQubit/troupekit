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

/// Find facts about the task.
#[agent(model = "fake/m", role = "A researcher.", goal = "Find facts.")]
fn researcher() {}

/// Check the facts in the task.
#[agent(model = "fake/m")]
fn checker() {}

fn echo_researcher() -> Agent {
    researcher().model_client(echo_model())
}

mod crew {
    pub fn echo_checker() -> troupekit::Agent {
        super::checker().model_client(crate::common::echo_model())
    }
}

/// Lead with agent tools.
#[agent(model = "fake/m", agents = [echo_researcher, crew::echo_checker])]
fn tool_lead() {}

/// Lead with coworkers.
#[agent(model = "fake/m", delegates = [echo_researcher, self::crew::echo_checker])]
fn delegate_lead() {}

/// Lead with a member that has no model client.
#[agent(model = "fake/m", agents = [checker])]
fn broken_lead() {}

#[tokio::test]
async fn macro_agents_key_reaches_builder() {
    let built = tool_lead()
        .model_client(echo_model())
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().tools, ["researcher", "checker"]);
    let text = built.run("Go").await.unwrap();
    assert_eq!(text, "Hello! You said: Go");
    let err = broken_lead()
        .model_client(echo_model())
        .build()
        .await
        .unwrap_err();
    assert!(format!("{err:?}").contains("checker"), "{err:?}");
}

#[tokio::test]
async fn macro_delegates_key_reaches_builder() {
    let built = delegate_lead()
        .model_client(echo_model())
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona().tools, ["delegate_work", "ask_question"]);
    let by_hand = Agent::new("delegate-lead")
        .instructions("Lead with coworkers.")
        .model("fake/m")
        .model_client(echo_model())
        .delegates([echo_researcher(), crew::echo_checker()])
        .build()
        .await
        .unwrap();
    assert_eq!(by_hand.persona(), built.persona());
}

/// Manage the crew.
#[agent(model = "fake/m", manages = [echo_researcher, crew::echo_checker])]
fn boss() {}

#[tokio::test]
async fn macro_manages_key_reaches_builder() {
    let built = boss().model_client(echo_model()).build().await.unwrap();
    let by_hand = Agent::new("boss")
        .instructions("Manage the crew.")
        .model("fake/m")
        .model_client(echo_model())
        .manages([echo_researcher(), crew::echo_checker()])
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona(), by_hand.persona());
    let text = built.run("Go").await.unwrap();
    assert!(text.starts_with("Hello! You said: "), "{text}");
}

/// Answer or hand the request to a desk.
#[agent(model = "fake/m", handoffs = [echo_researcher, crew::echo_checker])]
fn reception() {}

#[tokio::test]
async fn macro_handoffs_key_reaches_builder() {
    let built = reception().model_client(echo_model()).build().await.unwrap();
    let by_hand = Agent::new("reception")
        .instructions("Answer or hand the request to a desk.")
        .model("fake/m")
        .model_client(echo_model())
        .handoffs([echo_researcher(), crew::echo_checker()])
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona(), by_hand.persona());
    let err = reception()
        .model_client(echo_model())
        .handoffs([echo_researcher()])
        .build()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("two team agents are named"), "{err}");
}

/// Send each request to the right desk.
#[agent(model = "fake/m", members = [echo_researcher, crew::echo_checker], team = "broadcast")]
fn front_desk() {}

/// Lead the default team.
#[agent(model = "fake/m", members = [echo_researcher])]
fn default_lead() {}

#[tokio::test]
async fn macro_members_and_team_keys_reach_builder() {
    let built = front_desk()
        .model_client(echo_model())
        .build()
        .await
        .unwrap();
    let by_hand = Agent::new("front-desk")
        .instructions("Send each request to the right desk.")
        .model("fake/m")
        .model_client(echo_model())
        .members(
            [echo_researcher(), crew::echo_checker()],
            troupekit::TeamMode::Broadcast,
        )
        .build()
        .await
        .unwrap();
    assert_eq!(built.persona(), by_hand.persona());
    let out = built.run_with("Hi").await.unwrap();
    assert_eq!(
        out.output().as_array().map(Vec::len),
        Some(2),
        "{:?}",
        out.output()
    );
    let text = default_lead()
        .model_client(echo_model())
        .run("Go")
        .await
        .unwrap();
    assert!(text.starts_with("Hello! You said: "), "{text}");
}
