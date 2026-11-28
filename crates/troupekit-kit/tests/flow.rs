use crate::common::echo_model;
use troupekit::{tool, Agent, Graph, Reducer, Trigger};
use troupekit_graph::Workflow;

/// The workflow a two-node chain built at 2343a4f, frozen as TOML.
/// A chain graph must keep building exactly this value once flow parts
/// exist, so 0.1 and 0.2 users see no change.
const CHAIN_AT_0_2: &str = r#"
name = "pipeline"
start = "draft"
max_retries = 3
[state]
input = "string"
draft = "string"
upper = "string"
[[nodes]]
id = "draft"
kind = "agent"
persona = "drafter"
reads = ["input"]
writes = "draft"
[[nodes]]
id = "upper"
kind = "tool"
tool = "step__upper"
reads = ["draft"]
writes = "upper"
[[nodes]]
id = "end"
kind = "end"
[[edges]]
from = "draft"
to = "upper"
[[edges]]
from = "upper"
to = "end"
"#;

/// Uppercases the draft.
#[tool]
fn louder_draft(draft: String) -> String {
    draft.to_uppercase()
}

#[tokio::test]
async fn chain_graph_still_builds_the_same_workflow() {
    let drafter = Agent::new("drafter")
        .instructions("Draft.")
        .model("fake/m")
        .model_client(echo_model());
    let built = Graph::new("pipeline")
        .agent("draft", drafter)
        .step("upper", louder_draft)
        .build()
        .await
        .unwrap();
    let expected = Workflow::parse("frozen/chain.toml", CHAIN_AT_0_2).unwrap();
    assert_eq!(built.workflow(), &expected);
}

/// Trims the input.
#[tool]
fn trim_input(input: String) -> String {
    input.trim().to_string()
}

/// Uppercases the cleaned text.
#[tool]
fn upper_clean(clean: String) -> String {
    clean.to_uppercase()
}

/// Counts characters of the cleaned text.
#[tool]
fn count_clean(clean: String) -> usize {
    clean.chars().count()
}

/// Joins both branch outputs.
#[tool]
fn report_both(upper: String, count: String) -> String {
    format!("{upper}:{count}")
}

#[tokio::test]
async fn parallel_then_join_in_one_line() {
    let built = Graph::new("fan")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("count", count_clean)
        .step("report", report_both)
        .parallel("clean", ["upper", "count"])
        .join("both", ["upper", "count"])
        .edge("both", "report")
        .build()
        .await
        .unwrap();
    assert_eq!(
        built.run("  hi  ").await.unwrap(),
        serde_json::json!("HI:2")
    );
    let outline = built.workflow();
    let join = outline.nodes.iter().find(|n| n.id == "both").unwrap();
    assert_eq!(join.kind, troupekit_graph::NodeKind::Join);
    let report = outline.nodes.iter().find(|n| n.id == "report").unwrap();
    assert_eq!(
        report.reads,
        ["upper", "count"],
        "a node after a join reads each branch by id, in join order"
    );
}

#[tokio::test]
async fn edge_fan_out_still_needs_parallel() {
    let err = Graph::new("fan")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("count", count_clean)
        .edge("clean", "upper")
        .edge("clean", "count")
        .join("both", ["upper", "count"])
        .build()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("use .parallel"), "{err}");
}

#[tokio::test]
async fn join_of_unknown_branch_fails_build() {
    let err = Graph::new("fan")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("count", count_clean)
        .parallel("clean", ["upper", "count"])
        .join("both", ["upper", "ghost"])
        .build()
        .await
        .unwrap_err();
    assert!(err.to_string().contains("unknown node `ghost`"), "{err}");
}

/// Shows which branch reached an any-join; a dropped branch reads null.
#[tool]
fn pick_first(upper: Option<String>, count: Option<String>) -> String {
    let show = |v: Option<String>| v.unwrap_or_else(|| "-".into());
    format!("{}|{}", show(upper), show(count))
}

#[tokio::test]
async fn join_any_takes_first() {
    let built = Graph::new("race")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("count", count_clean)
        .step("pick", pick_first)
        .parallel("clean", ["upper", "count"])
        .join_any("first", ["upper", "count"])
        .edge("first", "pick")
        .build()
        .await
        .unwrap();
    let report = built.run_with("  hi  ").await.unwrap();
    assert_eq!(
        report.output(),
        &serde_json::json!("HI|-"),
        "the lowest branch continues"
    );
    assert!(
        report.state().get("count").is_none_or(|v| v.is_null()),
        "{:?}",
        report.state()
    );
    let pick = built
        .workflow()
        .nodes
        .iter()
        .find(|n| n.id == "pick")
        .unwrap();
    assert_eq!(pick.reads, ["upper", "count"]);
}

#[tokio::test]
async fn after_names_the_trigger() {
    let all = Graph::new("all")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("count", count_clean)
        .step("report", report_both)
        .parallel("clean", ["upper", "count"])
        .after("both", ["upper", "count"], Trigger::All)
        .edge("both", "report")
        .run("  hi  ")
        .await
        .unwrap();
    assert_eq!(all, serde_json::json!("HI:2"));
    let any = Graph::new("any")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("count", count_clean)
        .step("pick", pick_first)
        .parallel("clean", ["upper", "count"])
        .after("first", ["upper", "count"], Trigger::Any)
        .edge("first", "pick")
        .run("  hi  ")
        .await
        .unwrap();
    assert_eq!(any, serde_json::json!("HI|-"));
}

fn paused(err: troupekit::KitError) -> troupekit_core::ids::RunId {
    match err {
        troupekit::KitError::Paused { run_id, .. } => run_id,
        other => panic!("expected a pause, got {other}"),
    }
}

fn kept() -> troupekit::Storage {
    troupekit::Storage::custom(std::sync::Arc::new(
        troupekit_core::inmem::InMemCheckpoints::default(),
    ))
}

#[tokio::test]
async fn approval_node_pauses_and_answer_resumes() {
    let built = Graph::new("gate")
        .step("clean", trim_input)
        .approval("ok", "Ship it?")
        .step("upper", upper_clean)
        .edge("clean", "ok")
        .edge("ok", "upper")
        .checkpoints(kept())
        .build()
        .await
        .unwrap();
    let upper = built
        .workflow()
        .nodes
        .iter()
        .find(|n| n.id == "upper")
        .unwrap();
    assert_eq!(
        upper.reads,
        ["clean"],
        "an approval passes its feeder through"
    );
    let run = paused(built.run_with("  hi  ").await.unwrap_err());
    assert_eq!(
        built
            .answer(run, "ok", serde_json::json!(true))
            .await
            .unwrap(),
        serde_json::json!("HI")
    );
    let declined = paused(built.run_with("  no  ").await.unwrap_err());
    let err = built
        .answer(declined, "ok", serde_json::json!(false))
        .await
        .unwrap_err();
    assert!(matches!(err, troupekit::KitError::Failed { .. }), "{err}");
    assert!(err.to_string().contains("declined"), "{err}");
}

#[tokio::test]
async fn question_node_routes_on_answer() {
    let built = Graph::new("ask")
        .step("clean", trim_input)
        .question(
            "how",
            "Loud or count?",
            [("loud", "upper"), ("count", "count")],
        )
        .step("upper", upper_clean)
        .step("count", count_clean)
        .edge("clean", "how")
        .checkpoints(kept())
        .build()
        .await
        .unwrap();
    let run = paused(built.run_with("  hi  ").await.unwrap_err());
    assert_eq!(
        built
            .answer(run, "how", serde_json::json!("count"))
            .await
            .unwrap(),
        serde_json::json!("2")
    );
    let run = paused(built.run_with("  hi  ").await.unwrap_err());
    let wrong = built.answer(run, "how", serde_json::json!("maybe")).await;
    assert!(wrong.is_err(), "an answer outside the routes is refused");
    assert_eq!(
        built
            .answer(run, "how", serde_json::json!("loud"))
            .await
            .unwrap(),
        serde_json::json!("HI")
    );
}

#[tokio::test]
async fn answer_needs_kept_checkpoints() {
    let built = Graph::new("gate")
        .step("clean", trim_input)
        .approval("ok", "Ship it?")
        .step("upper", upper_clean)
        .edge("clean", "ok")
        .edge("ok", "upper")
        .build()
        .await
        .unwrap();
    let run = paused(built.run_with("hi").await.unwrap_err());
    let err = built
        .answer(run, "ok", serde_json::json!(true))
        .await
        .unwrap_err();
    assert!(matches!(err, troupekit::KitError::Config { .. }), "{err}");
}

/// Lowercases the cleaned text.
#[tool]
fn lower_clean(clean: String) -> String {
    clean.to_lowercase()
}

/// Joins the reduced list of branch outputs.
#[tool]
fn glue(both: Vec<String>) -> String {
    both.join("+")
}

#[tokio::test]
async fn reducer_append_from_two_branches() {
    let built = Graph::new("merge")
        .step("clean", trim_input)
        .step("upper", upper_clean)
        .step("lower", lower_clean)
        .step("glue", glue)
        .parallel("clean", ["upper", "lower"])
        .join("both", ["upper", "lower"])
        .reducer("both", Reducer::Append)
        .edge("both", "glue")
        .build()
        .await
        .unwrap();
    let report = built.run_with("  Hi  ").await.unwrap();
    assert_eq!(
        report.output(),
        &serde_json::json!("HI+hi"),
        "writes apply in branch order"
    );
    assert_eq!(report.state()["both"], serde_json::json!(["HI", "hi"]));
    let after = built
        .workflow()
        .nodes
        .iter()
        .find(|n| n.id == "glue")
        .unwrap();
    assert_eq!(
        after.reads,
        ["both"],
        "the node after a reducing join reads its one key"
    );
}

#[tokio::test]
async fn reducer_needs_a_join_and_a_text_reducer() {
    let base = || {
        Graph::new("merge")
            .step("clean", trim_input)
            .step("upper", upper_clean)
            .step("lower", lower_clean)
            .step("glue", glue)
            .parallel("clean", ["upper", "lower"])
            .join("both", ["upper", "lower"])
            .edge("both", "glue")
    };
    let not_join = base()
        .reducer("upper", Reducer::Append)
        .build()
        .await
        .unwrap_err();
    assert!(
        not_join.to_string().contains("`upper` is not a join"),
        "{not_join}"
    );
    let sum = base()
        .reducer("both", Reducer::Sum)
        .build()
        .await
        .unwrap_err();
    assert!(sum.to_string().contains("append or last"), "{sum}");
}
