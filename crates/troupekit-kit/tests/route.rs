use crate::common::*;
use troupekit::{Agent, Graph, KitError};
use troupekit_providers::route::{DeviceState, NetworkState, RoutePolicy};

fn echo() -> Agent {
    Agent::new("echo")
        .instructions("Echo the user.")
        .model("fake/m")
        .model_client(echo_model())
}

fn policy(text: &str) -> RoutePolicy {
    RoutePolicy::parse("route.toml", text).unwrap()
}

async fn config_error(agent: Agent) -> String {
    match agent.build().await {
        Err(err @ KitError::Config { .. }) => err.to_string(),
        Err(other) => panic!("expected a Config error, got {other}"),
        Ok(_) => panic!("expected build to fail"),
    }
}

#[tokio::test]
async fn route_binds_cloud_and_refuses_unknown_targets() {
    let cloud_only = policy("name = \"office\"\n[[target]]\nid = \"cloud\"\n");
    let network = NetworkState::new();
    network.set(Some(true));
    let built = echo()
        .route(cloud_only)
        .network(network)
        .device(DeviceState::new())
        .build()
        .await
        .unwrap();
    assert_eq!(built.run("hi").await.unwrap(), "Hello! You said: hi");

    let edge = policy("name = \"office\"\n[[target]]\nid = \"edge\"\n");
    let text = config_error(echo().route(edge)).await;
    assert!(text.contains("edge"), "{text}");
    assert!(
        text.contains("`local`") && text.contains("`cloud`"),
        "{text}"
    );

    let with_local =
        policy("name = \"office\"\n[[target]]\nid = \"local\"\n[[target]]\nid = \"cloud\"\n");
    let text = config_error(echo().route(with_local)).await;
    assert!(text.contains(".local("), "{text}");
}

#[tokio::test]
async fn route_file_reads_and_bounds_the_policy() {
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("route.toml");
    std::fs::write(&good, "name = \"office\"\n[[target]]\nid = \"cloud\"\n").unwrap();
    let built = echo().route_file(&good).build().await.unwrap();
    assert_eq!(built.run("hi").await.unwrap(), "Hello! You said: hi");

    let missing = dir.path().join("missing.toml");
    let text = config_error(echo().route_file(&missing)).await;
    assert!(text.contains("missing.toml"), "{text}");

    let huge = dir.path().join("huge.toml");
    std::fs::write(&huge, " ".repeat(RoutePolicy::MAX_BYTES + 1)).unwrap();
    let text = config_error(echo().route_file(&huge)).await;
    assert!(text.contains("huge.toml"), "{text}");

    let bad = dir.path().join("bad.toml");
    std::fs::write(&bad, "name = \"office\"\ncolour = 1\n").unwrap();
    let err = echo().route_file(&bad).build().await.unwrap_err();
    assert!(matches!(err, KitError::Route(_)), "{err}");
    assert!(err.to_string().contains("colour"), "{err}");
}

#[cfg(feature = "knowledge")]
#[tokio::test]
async fn private_agent_refuses_cloud_embedder() {
    use std::sync::Arc;
    use troupekit::Knowledge;
    use troupekit_context::embed::HashingEmbedder;
    let dir = tempfile::tempdir().unwrap();
    let text = config_error(echo().private(true).knowledge(Knowledge::dir(dir.path()))).await;
    assert!(text.contains("on_device_embedder"), "{text}");
    let plain = Knowledge::dir(dir.path()).embedder(Arc::new(HashingEmbedder::new(32)));
    let text = config_error(echo().private(true).knowledge(plain)).await;
    assert!(text.contains("on_device_embedder"), "{text}");
    let marked = Knowledge::dir(dir.path()).on_device_embedder(Arc::new(HashingEmbedder::new(32)));
    let text = config_error(echo().private(true).knowledge(marked)).await;
    assert!(!text.contains("embedder"), "{text}");
    assert!(text.contains("on-device"), "{text}");
}

#[tokio::test]
async fn private_agent_refuses_cloud_judge_model() {
    let text = config_error(
        echo()
            .private(true)
            .eval("judge('Answer is kind', 7, 'openai/gpt-6-luna')"),
    )
    .await;
    assert!(text.contains("judge"), "{text}");
    assert!(text.contains("local/"), "{text}");
}

#[tokio::test]
async fn private_agent_without_on_device_target_fails_build() {
    let text = config_error(echo().private(true)).await;
    assert!(text.contains("on-device"), "{text}");
    assert!(text.contains(".local("), "{text}");
    let cloud_only = policy("name = \"office\"\n[[target]]\nid = \"cloud\"\n");
    let text = config_error(echo().route(cloud_only).private(true)).await;
    assert!(text.contains("on-device"), "{text}");
}

#[tokio::test]
async fn tier_needs_a_route_that_serves_it() {
    let text = config_error(echo().tier("small")).await;
    assert!(text.contains(".route("), "{text}");
    let cloud = policy("name = \"office\"\n[[target]]\nid = \"cloud\"\ntiers = [\"large\"]\n");
    let text = config_error(echo().route(cloud).tier("small")).await;
    assert!(text.contains("small"), "{text}");
    let cloud = policy("name = \"office\"\n[[target]]\nid = \"cloud\"\ntiers = [\"large\"]\n");
    let built = echo().route(cloud).tier("large").build().await.unwrap();
    assert_eq!(built.run("hi").await.unwrap(), "Hello! You said: hi");
}

#[tokio::test]
async fn graph_private_applies_to_agent_nodes() {
    let err = Graph::new("pipe")
        .agent("first", echo())
        .private(true)
        .build()
        .await
        .unwrap_err();
    let text = err.to_string();
    assert!(text.contains("on-device"), "{text}");

    let built = Graph::new("pipe")
        .agent("first", echo())
        .private(false)
        .build()
        .await;
    assert!(built.is_ok());
}

#[test]
fn defaults_are_the_safe_values() {
    let text = format!("{:?}", echo());
    assert!(text.contains("private: false"), "{text}");
    assert!(text.contains("tier: None"), "{text}");
    assert!(text.contains("route: false"), "{text}");
    let graph = format!("{:?}", Graph::new("pipe"));
    assert!(graph.contains("private: false"), "{graph}");
}

#[tokio::test]
async fn cloud_key_is_needed_only_when_a_cloud_target_can_run() {
    let keyless = || {
        Agent::new("keyless")
            .instructions("Echo the user.")
            .model("openai/gpt-6-luna")
            .api_key_env("TROUPEKIT_TEST_KEY_THAT_IS_NEVER_SET")
    };
    let text = config_error(keyless().private(true)).await;
    assert!(text.contains("on-device"), "{text}");
    let local_only = policy("name = \"office\"\n[[target]]\nid = \"local\"\n");
    let text = config_error(keyless().route(local_only)).await;
    assert!(text.contains(".local("), "{text}");
    let cloud = policy("name = \"office\"\n[[target]]\nid = \"cloud\"\n");
    let err = keyless().route(cloud).build().await.unwrap_err();
    assert!(matches!(err, KitError::MissingKey { .. }), "{err}");
    let err = keyless().build().await.unwrap_err();
    assert!(matches!(err, KitError::MissingKey { .. }), "{err}");
}

#[tokio::test]
async fn route_cap_is_shared_through_the_agent_ledger() {
    use std::sync::Arc;
    use troupekit_core::store::UsageLedger;
    let capped = policy(
        "name = \"office\"\n[budget]\ncloud_usd_per_day = 0.0001\n[[target]]\nid = \"cloud\"\n",
    );
    let paid = troupekit_providers::Price {
        input_usd_micros_per_mtok: 100_000_000,
        ..troupekit_providers::Price::FREE
    };
    let ledger: Arc<dyn UsageLedger> = Arc::new(troupekit_agent::InMemoryLedger::default());
    let agent = || {
        echo()
            .price(paid)
            .retries(0)
            .route(capped.clone())
            .ledger(ledger.clone())
    };
    let first = agent().build().await.unwrap();
    assert_eq!(first.run("hi").await.unwrap(), "Hello! You said: hi");
    let second = agent().build().await.unwrap();
    assert!(
        second.run("hi").await.is_err(),
        "the shared ledger already holds the day's cap"
    );
    let fresh = echo()
        .price(paid)
        .retries(0)
        .route(capped)
        .build()
        .await
        .unwrap();
    assert_eq!(fresh.run("hi").await.unwrap(), "Hello! You said: hi");
}
