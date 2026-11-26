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
