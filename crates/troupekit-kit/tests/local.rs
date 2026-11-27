use crate::common::*;
use std::path::PathBuf;
use std::sync::Arc;
use troupekit::{Agent, Graph, KitError};
use troupekit_local::{
    ChatMessage, Flow, GenParams, GenStop, Generation, LocalError, LocalInfo, LocalMetrics,
    LocalModel,
};
use troupekit_providers::route::RoutePolicy;

struct Scripted {
    info: LocalInfo,
    reply: String,
}

impl LocalModel for Scripted {
    fn info(&self) -> &LocalInfo {
        &self.info
    }

    fn tokenize(&self, text: &str, _special: bool) -> Result<Vec<u32>, LocalError> {
        Ok(text.bytes().map(u32::from).collect())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, LocalError> {
        Ok(tokens.iter().map(|t| char::from(*t as u8)).collect())
    }

    fn render_chat(&self, msgs: &[ChatMessage], add: bool) -> Result<String, LocalError> {
        let mut out: String = msgs.iter().map(|m| m.content().to_string()).collect();
        if add {
            out.push_str("<assistant>");
        }
        Ok(out)
    }

    fn generate(
        &self,
        _prompt: &[u32],
        _params: &GenParams,
        on_token: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<Generation, LocalError> {
        on_token(&self.reply);
        Ok(Generation::new(
            self.reply.clone(),
            1,
            GenStop::EndOfText,
            None,
            LocalMetrics::default(),
        ))
    }

    fn score(&self, _tokens: &[u32]) -> Result<f64, LocalError> {
        Ok(0.0)
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LocalError> {
        let dims = self.info.embedding_dims();
        if dims == 0 {
            return Err(LocalError::Unsupported("embeddings"));
        }
        Ok(texts
            .iter()
            .map(|_| {
                let mut v = vec![0.0; dims];
                v[0] = 1.0;
                v
            })
            .collect())
    }

    fn reset(&self) {}
}

fn on_device(reply: &str, dims: usize) -> Arc<dyn LocalModel> {
    Arc::new(Scripted {
        info: LocalInfo::new(4_096, 256, dims, true),
        reply: reply.to_string(),
    })
}

fn cloud_agent(name: &str) -> Agent {
    Agent::new(name)
        .instructions("Answer.")
        .model("openai/gpt-6-luna")
        .api_key(secret("sk-local-test"))
}

async fn config_error(agent: Agent) -> String {
    match agent.build().await {
        Err(err @ KitError::Config { .. }) => err.to_string(),
        Err(other) => panic!("expected a Config error, got {other}"),
        Ok(_) => panic!("expected build to fail"),
    }
}

#[tokio::test]
async fn private_agent_never_calls_cloud_transport() {
    let transport = scripted();
    let text = cloud_agent("private")
        .transport(transport.clone())
        .local(on_device("from the device", 0))
        .private(true)
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "from the device");
    assert!(transport.sent().is_empty(), "no request left the device");

    let both = RoutePolicy::parse(
        "office.toml",
        "name = \"office\"\ndefault = \"cloud\"\n[[target]]\nid = \"local\"\n[[target]]\nid = \"cloud\"\n",
    )
    .unwrap();
    let routed = scripted();
    let text = cloud_agent("private")
        .transport(routed.clone())
        .local(on_device("from the device", 0))
        .route(both)
        .private(true)
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "from the device");
    assert!(
        routed.sent().is_empty(),
        "the private view skips target cloud"
    );

    let open = scripted();
    let text = cloud_agent("open")
        .transport(open.clone())
        .local(on_device("from the device", 0))
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "Hi there");
    assert_eq!(open.sent().len(), 1, "a non-private agent uses .model(..)");
}

#[tokio::test]
async fn private_local_agent_needs_no_cloud_key() {
    let built = Agent::new("keyless")
        .instructions("Answer.")
        .model("openai/gpt-6-luna")
        .api_key_env("TROUPEKIT_TEST_KEY_THAT_IS_NEVER_SET")
        .local(on_device("from the device", 0))
        .private(true)
        .build()
        .await
        .unwrap();
    assert_eq!(built.run("hi").await.unwrap(), "from the device");
    let local_only = Agent::new("solo")
        .instructions("Answer.")
        .local(on_device("solo answer", 0));
    assert_eq!(local_only.run("hi").await.unwrap(), "solo answer");
}

#[cfg(feature = "knowledge")]
#[tokio::test]
async fn local_only_agent_defaults_to_local_embedder() {
    use troupekit::Knowledge;
    let dir = tempfile::tempdir().unwrap();
    let docs = dir.path().join("docs");
    std::fs::create_dir(&docs).unwrap();
    std::fs::write(docs.join("a.md"), "The office opens at nine.").unwrap();
    let knowledge = || Knowledge::dir(&docs).store(dir.path().join("k.db"));
    let solo = |dims: usize| {
        Agent::new("solo")
            .instructions("Answer.")
            .local(on_device("ok", dims))
            .knowledge(knowledge())
    };
    solo(8).build().await.unwrap();
    solo(8).private(true).build().await.unwrap();
    let text = config_error(solo(0)).await;
    assert!(
        text.contains("Knowledge::dir(..).local_embedder("),
        "{text}"
    );

    let marked = cloud_agent("mixed")
        .local(on_device("ok", 0))
        .private(true)
        .knowledge(knowledge().local_embedder(on_device("emb", 8)));
    marked.build().await.unwrap();
    let cloud_default = cloud_agent("mixed")
        .local(on_device("ok", 8))
        .private(true)
        .knowledge(knowledge());
    let text = config_error(cloud_default).await;
    assert!(text.contains("on_device_embedder"), "{text}");
}

#[tokio::test]
async fn local_selector_needs_file_or_download() {
    let dir = tempfile::tempdir().unwrap();
    std::env::set_var("TROUPEKIT_MODEL_DIR", dir.path());
    let agent = Agent::new("solo")
        .instructions("Answer.")
        .model("local/smollm2-135m-instruct-q2k");
    let text = config_error(agent).await;
    assert!(text.contains(".download(true)"), "{text}");
    assert!(text.contains("ModelCache::fetch"), "{text}");
    let unknown = Agent::new("solo")
        .instructions("Answer.")
        .local("no-such-model");
    let text = config_error(unknown).await;
    assert!(text.contains("LocalCatalog::builtin()"), "{text}");
    let twice = Agent::new("solo")
        .instructions("Answer.")
        .model("local/smollm2-135m-instruct-q2k")
        .local(on_device("x", 0));
    let text = config_error(twice).await;
    assert!(text.contains("once"), "{text}");
}

#[cfg(not(feature = "llama"))]
#[tokio::test]
async fn local_without_llama_feature_names_it() {
    let agent = Agent::new("solo")
        .instructions("Answer.")
        .local(PathBuf::from("model.gguf"));
    let text = config_error(agent).await;
    assert!(text.contains("`llama`"), "{text}");
    assert!(text.contains("model.gguf"), "{text}");
}

#[tokio::test]
async fn route_file_binds_local_and_cloud() {
    let dir = tempfile::tempdir().unwrap();
    let path = |name: &str, default: &str| {
        let file = dir.path().join(name);
        std::fs::write(
            &file,
            format!(
                "name = \"office\"\ndefault = \"{default}\"\n\
                 [[target]]\nid = \"local\"\n[[target]]\nid = \"cloud\"\n"
            ),
        )
        .unwrap();
        file
    };
    let agent = |file: PathBuf| {
        Agent::new("routed")
            .instructions("Echo the user.")
            .model("fake/m")
            .model_client(echo_model())
            .local(on_device("from the device", 0))
            .route_file(file)
    };
    let local_first = agent(path("local.toml", "local")).run("hi").await.unwrap();
    assert_eq!(local_first, "from the device");
    let cloud_first = agent(path("cloud.toml", "cloud")).run("hi").await.unwrap();
    assert_eq!(cloud_first, "Hello! You said: hi");

    let solo = Agent::new("solo")
        .instructions("Answer.")
        .local(on_device("x", 0))
        .route_file(path("solo.toml", "local"));
    let text = config_error(solo).await;
    assert!(text.contains("cloud model"), "{text}");
}

#[tokio::test]
async fn graph_private_runs_local_nodes() {
    let node = Agent::new("first")
        .instructions("Answer.")
        .model("fake/m")
        .model_client(echo_model())
        .local(on_device("from the device", 0));
    let text = Graph::new("pipe")
        .agent("first", node)
        .private(true)
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "from the device");
}

#[tokio::test]
async fn graph_nodes_share_one_local_agent() {
    let agent = Agent::new("helper")
        .instructions("Answer.")
        .local(on_device("from the device", 0));
    let text = Graph::new("pipe")
        .agent("first", agent.clone())
        .agent("second", agent)
        .edge("first", "second")
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "from the device");
}

#[tokio::test]
async fn private_route_needs_no_cloud_key() {
    let both = RoutePolicy::parse(
        "office.toml",
        "name = \"office\"\ndefault = \"cloud\"\n[[target]]\nid = \"local\"\n[[target]]\nid = \"cloud\"\n",
    )
    .unwrap();
    let keyless = || {
        Agent::new("keyless")
            .instructions("Answer.")
            .model("openai/gpt-6-luna")
            .api_key_env("TROUPEKIT_TEST_KEY_THAT_IS_NEVER_SET")
            .route(both.clone())
    };
    let text = keyless()
        .local(on_device("from the device", 0))
        .private(true)
        .run("hi")
        .await
        .unwrap();
    assert_eq!(text, "from the device");
    let text = config_error(keyless().private(true)).await;
    assert!(text.contains(".local("), "{text}");
    let err = keyless()
        .local(on_device("from the device", 0))
        .build()
        .await
        .unwrap_err();
    assert!(matches!(err, KitError::MissingKey { .. }), "{err}");
}
