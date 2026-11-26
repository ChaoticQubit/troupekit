#![cfg(feature = "sqlite")]

#[path = "../examples/teams/flows.rs"]
mod flows;

use serde_json::{json, Value};
use std::sync::Arc;
use troupekit::core::secret::Secret;
use troupekit::graph::GraphStatus;
use troupekit::Troupe;
use troupekit_providers::model::{Endpoint, HttpModel};
use troupekit_providers::transport::{Scripted, ScriptedTransport};
use troupekit_providers::wire::{AuthStyle, Protocol, Quirks};
use troupekit_providers::Price;

const FACTS: &str = "The first bicycle was built in 1817.";
const CHECKED: &str = "Confirmed: the first bicycle was built in 1817.";
const REPORT: &str = "Bicycles date back to 1817.";

fn sse(events: &[Value]) -> String {
    events.iter().map(|e| format!("data: {e}\n\n")).collect::<String>() + "data: [DONE]\n\n"
}

fn reply(content: &str) -> Scripted {
    let chunk = |delta: Value, finish: Option<&str>| json!({ "id": "c", "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }], "usage": null });
    let usage = json!({ "id": "c", "choices": [], "usage": { "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15 } });
    let text = sse(&[chunk(json!({ "role": "assistant", "content": content }), None), chunk(json!({}), Some("stop")), usage]);
    Scripted::chunked(&text, 64)
}

async fn desk(replies: &[&str]) -> (tempfile::TempDir, Troupe, Arc<troupekit::graph::SimpleGraphIo>, Arc<ScriptedTransport>) {
    let dir = tempfile::tempdir().unwrap();
    let troupe = Troupe::builder().sqlite(dir.path().join("teams.db")).build().await.unwrap();
    let transport = Arc::new(ScriptedTransport::new(replies.iter().map(|r| reply(r)).collect()));
    let endpoint = Endpoint { protocol: Protocol::OpenAi, base_url: "https://h.test/".into(), auth: AuthStyle::Bearer, quirks: Quirks::default() };
    let key = Some(Secret::new("sk-test-not-real".into()).unwrap());
    let model = Arc::new(HttpModel::new(flows::PERSONA_MODEL, "teams", endpoint, key, transport.clone()).unwrap());
    let io = flows::io(&troupe, model, Price::FREE);
    (dir, troupe, io, transport)
}

fn body(transport: &ScriptedTransport, index: usize) -> String {
    String::from_utf8(transport.sent()[index].body.clone().unwrap()).unwrap()
}

#[tokio::test]
async fn crew_writes_a_report() {
    let (_dir, troupe, io, transport) = desk(&[FACTS, CHECKED, REPORT]).await;
    let id = flows::start(&troupe, io.clone(), "report.toml", flows::REPORT, "the bicycle").await.unwrap();
    let run = flows::work(&troupe, io, id).await.unwrap();
    assert_eq!(flows::answer(&run).unwrap(), REPORT);
    assert_eq!(run.state()["research"], json!(FACTS));
    assert_eq!(run.state()["fact_check"], json!(CHECKED));
    assert_eq!(transport.sent().len(), 3);
    assert!(body(&transport, 0).contains("the bicycle"));
    assert!(body(&transport, 1).contains(FACTS));
    let writer = body(&transport, 2);
    assert!(writer.contains(FACTS) && writer.contains(CHECKED), "{writer}");
}

#[tokio::test]
async fn team_triage_routes_and_answers() {
    let fix = "Hold the power button for ten seconds, then restart.";
    let (_dir, troupe, io, transport) = desk(&[r#"{"next":"tech","reason":"broken device"}"#, fix]).await;
    let id = flows::start(&troupe, io.clone(), "triage.toml", flows::TRIAGE, "My laptop does not start").await.unwrap();
    let run = flows::work(&troupe, io, id).await.unwrap();
    assert_eq!(flows::answer(&run).unwrap(), fix);
    assert_eq!(transport.sent().len(), 2);
    assert!(body(&transport, 1).contains("My laptop does not start"));
}

#[tokio::test]
async fn team_triage_waits_for_refund_approval() {
    let refund = "Your refund is on its way and arrives in five days.";
    let to_billing = r#"{"next":"approve_billing","reason":"refund request"}"#;
    let (_dir, troupe, io, transport) = desk(&[to_billing, refund, to_billing]).await;
    let id = flows::start(&troupe, io.clone(), "triage.toml", flows::TRIAGE, "Refund my last order").await.unwrap();
    let waiting = flows::work(&troupe, io.clone(), id).await.unwrap();
    assert!(matches!(waiting.status(), GraphStatus::WaitingHuman { .. }), "{:?}", waiting.status());
    assert_eq!(transport.sent().len(), 1, "billing waits for the approval");
    assert!(flows::answer(&waiting).is_err());
    flows::approve(&troupe, io.clone(), id, "approve_billing", true).await.unwrap();
    let run = flows::work(&troupe, io.clone(), id).await.unwrap();
    assert_eq!(flows::answer(&run).unwrap(), refund);
    assert_eq!(transport.sent().len(), 2);
    let denied = flows::start(&troupe, io.clone(), "triage.toml", flows::TRIAGE, "Refund my order twice").await.unwrap();
    flows::work(&troupe, io.clone(), denied).await.unwrap();
    flows::approve(&troupe, io.clone(), denied, "approve_billing", false).await.unwrap();
    let run = flows::work(&troupe, io, denied).await.unwrap();
    assert!(matches!(run.status(), GraphStatus::Done), "{:?}", run.status());
    assert!(run.state().get("output").is_none_or(Value::is_null), "no billing answer");
    assert_eq!(transport.sent().len(), 3, "billing never ran");
}
