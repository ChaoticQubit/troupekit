use crate::common::echo_model;
use troupekit::{tool, Agent, Graph};
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
