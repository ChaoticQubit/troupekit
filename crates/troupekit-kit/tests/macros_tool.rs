use crate::common::calc_model;
use crate::tools_mod::{add, div, echo_type, has_run, shout, small};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use troupekit::Agent;
use troupekit_core::ids::RunId;
use troupekit_tools::{typed_tool, Tool, ToolContext, ToolError, ToolOutput, ToolRegistry};

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
#[allow(dead_code)]
struct AddOracle {
    a: i64,
    b: i64,
}

fn registry() -> ToolRegistry {
    let mut registry = ToolRegistry::new();
    for tool in [
        Arc::new(add) as Arc<dyn Tool>,
        Arc::new(shout),
        Arc::new(has_run),
        Arc::new(div),
        Arc::new(small),
        Arc::new(echo_type),
    ] {
        registry.register(tool).unwrap();
    }
    registry
}

#[tokio::test]
async fn tool_from_other_module_registers_in_0_1_registry() {
    let out = registry()
        .call("add", json!({"a": 2, "b": 40}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("42"));
    let text = Agent::new("calc")
        .instructions("Use the add tool.")
        .model("fake/m")
        .model_client(calc_model())
        .tool(add)
        .run("What is 2 + 40?")
        .await
        .unwrap();
    assert_eq!(text, "The sum is 42");
}

#[test]
fn tool_schema_matches_oracle() {
    let oracle = typed_tool::<AddOracle, _, _>("add", "Adds two numbers.", |_a, _c| async {
        Ok(ToolOutput::text(""))
    });
    assert_eq!(add.spec(), oracle.spec());
}

#[test]
fn doc_comment_becomes_description() {
    assert_eq!(add.spec().name, "add");
    assert_eq!(add.spec().description, "Adds two numbers.");
    assert_eq!(div.spec().description, "Divides a by b.");
}

#[tokio::test]
async fn sync_tool_works() {
    let out = registry()
        .call("shout", json!({"text": "hi"}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("HI"));
}

#[tokio::test]
async fn context_param_is_not_in_schema() {
    let schema = &has_run.spec().schema;
    assert_eq!(schema["properties"].as_object().unwrap().len(), 1);
    assert!(schema["properties"].get("label").is_some());
    let ctx = ToolContext::for_call(RunId::new(), 1, 0);
    let out = registry()
        .call("has_run", json!({"label": "x"}), &ctx)
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("true:x"));
    let out = registry()
        .call("has_run", json!({}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("false:"));
}

#[tokio::test]
async fn invoke_keeps_the_body_callable() {
    assert_eq!(add::invoke(2, 40).await, 42);
    assert_eq!(shout::invoke("a".into()), "A");
}

#[tokio::test]
async fn unknown_arg_key_is_refused() {
    let args = json!({"a": 1, "b": 2, "c": 3});
    let via_registry = registry()
        .call("add", args.clone(), &ToolContext::default())
        .await;
    assert!(matches!(via_registry, Err(ToolError::InvalidArgs(_))));
    let direct = add.call(args, &ToolContext::default()).await;
    assert!(
        matches!(direct, Err(ToolError::InvalidArgs(_))),
        "{direct:?}"
    );
}

#[tokio::test]
async fn u8_out_of_range_is_refused() {
    let via_registry = registry()
        .call("small", json!({"n": 300}), &ToolContext::default())
        .await;
    assert!(matches!(via_registry, Err(ToolError::InvalidArgs(_))));
    let direct = small.call(json!({"n": 300}), &ToolContext::default()).await;
    assert!(
        matches!(direct, Err(ToolError::InvalidArgs(_))),
        "{direct:?}"
    );
}

#[tokio::test]
async fn err_return_reaches_model_as_error_output() {
    let out = registry()
        .call("div", json!({"a": 1, "b": 0}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::error("division by zero"));
}

#[tokio::test]
async fn raw_ident_arg_reads_plain_key() {
    assert!(echo_type.spec().schema["properties"].get("type").is_some());
    let out = registry()
        .call("echo_type", json!({"type": "x"}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("x"));
}

/// Echoes the query.
#[troupekit::tool]
fn query(query: String, __troupekit_ctx: i64, ctx: &ToolContext) -> String {
    format!("{query} {__troupekit_ctx} {}", ctx.run_id.is_some())
}

#[tokio::test]
async fn params_named_like_the_tool_or_its_internals_still_bind() {
    let out = query
        .call(
            json!({"query": "q", "__troupekit_ctx": 7}),
            &ToolContext::for_call(RunId::new(), 3, 0),
        )
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("q 7 true"));
    assert_eq!(
        query::invoke("x".into(), 1, &ToolContext::default()),
        "x 1 false"
    );
}

/// Counts down to zero.
#[troupekit::tool]
fn countdown(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        1 + Self::invoke(n - 1)
    }
}

/// Counts down to zero by name.
#[troupekit::tool]
fn descend(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        1 + descend(n - 1)
    }
}

/// Counts down to zero by name, asynchronously.
#[troupekit::tool]
async fn descend_later(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        Box::pin(descend_later(n - 1)).await + 1
    }
}

#[tokio::test]
async fn tool_bodies_call_the_tool_by_name() {
    assert_eq!(descend::invoke(3), 3);
    assert_eq!(descend_later::invoke(3).await, 3);
    let out = descend
        .call(json!({"n": 4}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("4"));
}

#[tokio::test]
async fn tool_bodies_keep_self_access() {
    assert_eq!(countdown::invoke(3), 3);
    let out = countdown
        .call(json!({"n": 4}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("4"));
}
