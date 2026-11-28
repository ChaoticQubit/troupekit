use crate::error::LocalError;
use crate::model::LocalModel;
use async_trait::async_trait;
use std::sync::Arc;
use troupekit_context::embed::Embedder;
use troupekit_context::error::ContextError;

const BATCH: usize = 32;

/// On-device embeddings from a `LocalModel`, for knowledge search with
/// no network. Mean-pooled, L2-normalized vectors (the mean of the
/// token rows); batches of at most 32 texts run off the async runtime.
pub struct LocalEmbedder {
    model: Arc<dyn LocalModel>,
    dims: usize,
}

impl LocalEmbedder {
    /// Wraps a model that can embed.
    /// Input: the model. Output: the embedder, or
    /// `Unsupported("embeddings")` when its embedding size is 0.
    pub fn new(model: Arc<dyn LocalModel>) -> Result<Self, LocalError> {
        let dims = model.info().embedding_dims();
        if dims == 0 {
            return Err(LocalError::Unsupported("embeddings"));
        }
        Ok(Self { model, dims })
    }
}

impl std::fmt::Debug for LocalEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalEmbedder")
            .field("dims", &self.dims)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl Embedder for LocalEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        let mut out = Vec::with_capacity(texts.len());
        for batch in texts.chunks(BATCH) {
            let model = Arc::clone(&self.model);
            let batch = batch.to_vec();
            let n = batch.len();
            let vectors = tokio::task::spawn_blocking(move || model.embed(&batch))
                .await
                .map_err(|e| ContextError::Embed(format!("local embed task failed: {e}")))?
                .map_err(|e| ContextError::Embed(e.to_string()))?;
            if vectors.len() != n {
                return Err(ContextError::Embed(format!(
                    "local model returned {} vectors for {n} texts",
                    vectors.len()
                )));
            }
            out.extend(vectors);
        }
        assert!(out.len() == texts.len(), "one vector per text");
        Ok(out)
    }
}
