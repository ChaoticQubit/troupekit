use crate::embed::Embedder;
use crate::error::ContextError;
use async_trait::async_trait;
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Vector length of `intfloat/multilingual-e5-small`, the model `OnnxEmbedder` loads.
/// Vector stores and database columns that hold its output use this dimension;
/// a model change needs a new store and a re-embed.
pub const ONNX_DIMS: usize = 384;

const E5_PREFIX: &str = "query: ";

/// Local embedder running `intfloat/multilingual-e5-small` through ONNX Runtime.
/// The model downloads into the cache on first `new`, then runs offline.
/// Every input gets the E5 `query: ` prefix, the model card's rule for symmetric use.
pub struct OnnxEmbedder {
    model: Arc<Mutex<TextEmbedding>>,
}

impl OnnxEmbedder {
    /// Loads the model, downloading it into `cache_dir` (or fastembed's default
    /// cache) when it is not there yet. Returns `ContextError::Embed` when the
    /// download or the ONNX session fails, otherwise the ready embedder.
    pub fn new(cache_dir: Option<PathBuf>) -> Result<Self, ContextError> {
        let mut options = TextInitOptions::new(EmbeddingModel::MultilingualE5Small).with_show_download_progress(false);
        if let Some(dir) = cache_dir {
            options = options.with_cache_dir(dir);
        }
        let model = TextEmbedding::try_new(options).map_err(|e| ContextError::Embed(e.to_string()))?;
        Ok(Self { model: Arc::new(Mutex::new(model)) })
    }
}

#[async_trait]
impl Embedder for OnnxEmbedder {
    fn dims(&self) -> usize {
        ONNX_DIMS
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        let inputs: Vec<String> = texts.iter().map(|t| format!("{E5_PREFIX}{t}")).collect();
        let model = self.model.clone();
        tokio::task::spawn_blocking(move || {
            let mut model = model.lock().map_err(|_| ContextError::Embed("embedding model lock poisoned".into()))?;
            model.embed(inputs, None).map_err(|e| ContextError::Embed(e.to_string()))
        })
        .await
        .map_err(|e| ContextError::Embed(e.to_string()))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::embed_checked;
    use troupekit_core::store::cosine;

    #[tokio::test]
    #[ignore = "downloads the model; run by hand with --features onnx -- --ignored"]
    async fn embeds_multilingual_text() {
        let e = OnnxEmbedder::new(None).unwrap();
        let texts = ["protein breakfast", "desayuno con proteína", "car insurance renewal"].map(String::from);
        let v = embed_checked(&e, &texts).await.unwrap();
        assert_eq!(v[0].len(), ONNX_DIMS);
        assert!(cosine(&v[0], &v[1]) > cosine(&v[0], &v[2]));
    }
}
