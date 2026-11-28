use std::sync::{Arc, Mutex};
use troupekit_context::embed::{embed_checked, Embedder};
use troupekit_local::{
    ChatMessage, Flow, GenParams, Generation, LocalEmbedder, LocalError, LocalInfo, LocalModel,
};

/// A model double that embeds each text as a unit vector of `dims`
/// floats and records the size of every `embed` call.
/// A real model needs the `llama` feature and a GGUF file (L8).
struct VecModel {
    info: LocalInfo,
    calls: Mutex<Vec<usize>>,
}

impl VecModel {
    fn new(dims: usize) -> Self {
        Self {
            info: LocalInfo::new(64, 100, dims, false),
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl LocalModel for VecModel {
    fn info(&self) -> &LocalInfo {
        &self.info
    }

    fn tokenize(&self, _text: &str, _special: bool) -> Result<Vec<u32>, LocalError> {
        Ok(Vec::new())
    }

    fn detokenize(&self, _tokens: &[u32]) -> Result<String, LocalError> {
        Ok(String::new())
    }

    fn render_chat(&self, _msgs: &[ChatMessage], _add: bool) -> Result<String, LocalError> {
        Err(LocalError::Unsupported("chat template"))
    }

    fn generate(
        &self,
        _prompt: &[u32],
        _params: &GenParams,
        _on_token: &mut dyn FnMut(&str) -> Flow,
    ) -> Result<Generation, LocalError> {
        Err(LocalError::Unsupported("generation"))
    }

    fn score(&self, _tokens: &[u32]) -> Result<f64, LocalError> {
        Ok(0.0)
    }

    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, LocalError> {
        let dims = self.info.embedding_dims();
        if dims == 0 {
            return Err(LocalError::Unsupported("embeddings"));
        }
        self.calls.lock().expect("lock").push(texts.len());
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

#[test]
fn embedder_rejects_model_without_embeddings() {
    let model: Arc<dyn LocalModel> = Arc::new(VecModel::new(0));
    let err = LocalEmbedder::new(model).expect_err("no embeddings");
    assert_eq!(err, LocalError::Unsupported("embeddings"));
}

#[tokio::test]
async fn embed_checked_accepts_local_output() {
    let model: Arc<dyn LocalModel> = Arc::new(VecModel::new(8));
    let embedder = LocalEmbedder::new(model).expect("embedder");
    assert_eq!(embedder.dims(), 8);
    let texts: Vec<String> = ["a", "b", "c"].iter().map(|s| s.to_string()).collect();
    let vectors = embed_checked(&embedder, &texts).await.expect("checked");
    assert_eq!(vectors.len(), 3);
    assert!(vectors.iter().all(|v| v.len() == 8));
}

#[tokio::test]
async fn embed_batches_at_most_32_texts() {
    let model = Arc::new(VecModel::new(4));
    let embedder = LocalEmbedder::new(model.clone()).expect("embedder");
    let texts: Vec<String> = (0..70).map(|i| format!("t{i}")).collect();
    let vectors = embedder.embed(&texts).await.expect("embeds");
    assert_eq!(vectors.len(), 70);
    assert_eq!(*model.calls.lock().expect("lock"), [32, 32, 6]);
}

#[tokio::test]
async fn empty_input_makes_no_model_call() {
    let model = Arc::new(VecModel::new(4));
    let embedder = LocalEmbedder::new(model.clone()).expect("embedder");
    assert!(embedder.embed(&[]).await.expect("empty").is_empty());
    assert!(model.calls.lock().expect("lock").is_empty());
}
