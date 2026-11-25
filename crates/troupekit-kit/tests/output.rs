use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::sync::Arc;
use troupekit::__private::spec;
use troupekit::{tool, IntoToolOutput, Json};
use troupekit_tools::{typed_tool, Tool, ToolContext, ToolOutput, ToolRegistry};

#[derive(Deserialize, schemars::JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[allow(dead_code)]
struct Oracle {
    count: i64,
    small: u8,
    label: Option<String>,
    values: Vec<f64>,
    flags: HashMap<String, bool>,
    raw: Value,
}

#[derive(serde::Serialize)]
struct Point {
    x: i32,
    y: i32,
}

#[tokio::test]
async fn tool_fn_form_registers_and_runs() {
    let mut registry = ToolRegistry::new();
    let add = tool("add", "Adds a and b", |x: AddArgs| async move { x.a + x.b });
    registry.register(Arc::new(add)).unwrap();
    let out = registry
        .call("add", json!({"a": 2, "b": 40}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::text("42"));
}

#[tokio::test]
async fn tool_fn_form_err_reaches_model_as_error_output() {
    let mut registry = ToolRegistry::new();
    let div = tool("div", "Divides a by b", |x: AddArgs| async move {
        if x.b == 0 {
            Err("division by zero")
        } else {
            Ok(x.a / x.b)
        }
    });
    registry.register(Arc::new(div)).unwrap();
    let out = registry
        .call("div", json!({"a": 1, "b": 0}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::error("division by zero"));
}

#[test]
fn spec_matches_typed_tool_oracle() {
    let oracle = typed_tool::<Oracle, _, _>("oracle", "Every supported type", |_a, _c| async {
        Ok(ToolOutput::text(""))
    });
    assert_eq!(
        &spec::<Oracle>("oracle", "Every supported type"),
        oracle.spec()
    );
}

#[test]
fn result_err_is_a_tool_error_output() {
    let failed: Result<i64, String> = Err("no such city".into());
    assert_eq!(failed.into_output(), ToolOutput::error("no such city"));
    let passed: Result<i64, String> = Ok(7);
    assert_eq!(passed.into_output(), ToolOutput::text("7"));
}

#[test]
fn option_none_is_null() {
    assert_eq!(None::<String>.into_output(), ToolOutput::text("null"));
    assert_eq!(Some("x").into_output(), ToolOutput::text("x"));
}

#[test]
fn json_wrapper_serializes() {
    assert_eq!(
        Json(Point { x: 1, y: -2 }).into_output(),
        ToolOutput::text(r#"{"x":1,"y":-2}"#)
    );
    assert_eq!(
        json!({"k": [1, 2]}).into_output(),
        ToolOutput::text(r#"{"k":[1,2]}"#)
    );
}

#[test]
fn unit_is_ok() {
    assert_eq!(().into_output(), ToolOutput::text("ok"));
    assert_eq!(true.into_output(), ToolOutput::text("true"));
    assert_eq!(2.5f64.into_output(), ToolOutput::text("2.5"));
    assert_eq!(ToolOutput::error("e").into_output(), ToolOutput::error("e"));
}

#[test]
fn stateless_tools_share_one_arc_and_stateful_ones_do_not() {
    use troupekit::AsTool;
    let first = crate::tools_mod::shout.into_arc();
    let again = crate::tools_mod::shout.into_arc();
    assert!(std::sync::Arc::ptr_eq(&first, &again));
    let make = || troupekit::tool("t", "T.", |a: crate::common::AddArgs| async move { a.a });
    assert!(!std::sync::Arc::ptr_eq(
        &make().into_arc(),
        &make().into_arc()
    ));
}
