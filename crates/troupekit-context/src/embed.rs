use crate::error::ContextError;
use crate::text::{stable_hash, terms};
use async_trait::async_trait;

/// Turns texts into fixed-length vectors for a `VectorStore`.
/// Object safe, so the model can be chosen at runtime as `Arc<dyn Embedder>`.
/// `embed` returns one vector of `dims()` floats per input text, in order.
#[async_trait]
pub trait Embedder: Send + Sync {
    fn dims(&self) -> usize;
    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError>;
}

/// Calls `embedder` on `texts` and checks the shape of what comes back.
/// Returns `ContextError::Embed` when the vector count differs from the text
/// count or any vector is not `dims()` long; otherwise the vectors.
pub async fn embed_checked(embedder: &dyn Embedder, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
    let vectors = embedder.embed(texts).await?;
    if vectors.len() != texts.len() {
        return Err(ContextError::Embed(format!("expected {} vectors, got {}", texts.len(), vectors.len())));
    }
    if let Some(v) = vectors.iter().find(|v| v.len() != embedder.dims()) {
        return Err(ContextError::Embed(format!("expected {} dimensions, got {}", embedder.dims(), v.len())));
    }
    Ok(vectors)
}

/// Deterministic embedder for tests and examples, with no model.
/// Each term adds plus or minus one to a bucket picked by its stable hash,
/// then the vector is L2 normalised, so shared terms give a positive cosine.
#[derive(Debug, Clone, Copy)]
pub struct HashingEmbedder {
    dims: usize,
}

impl HashingEmbedder {
    /// Creates an embedder producing vectors of `dims` floats.
    /// Panics when `dims` is zero, which no vector index accepts.
    /// Returns the new embedder.
    pub fn new(dims: usize) -> Self {
        assert!(dims > 0, "embedding dimension must be positive");
        Self { dims }
    }

    /// Hashes the terms of `text` into one normalised vector.
    /// Text without terms gives the zero vector.
    /// Returns a vector of `dims` floats.
    fn vector(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dims];
        for term in terms(text) {
            let h = stable_hash(&[&term]);
            let sign = if h >> 63 == 1 { -1.0 } else { 1.0 };
            v[(h % self.dims as u64) as usize] += sign;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            v.iter_mut().for_each(|x| *x /= norm);
        }
        v
    }
}

#[async_trait]
impl Embedder for HashingEmbedder {
    fn dims(&self) -> usize {
        self.dims
    }

    async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
        Ok(texts.iter().map(|t| self.vector(t)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use troupekit_core::store::cosine;

    struct Bad {
        count: usize,
        len: usize,
    }

    #[async_trait]
    impl Embedder for Bad {
        fn dims(&self) -> usize {
            4
        }

        async fn embed(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
            Ok(vec![vec![0.5; self.len]; self.count])
        }
    }

    fn one(e: &HashingEmbedder, text: &str) -> Vec<f32> {
        tokio_test_block(e.embed(&[text.to_string()])).unwrap().remove(0)
    }

    fn tokio_test_block<F: std::future::Future>(f: F) -> F::Output {
        tokio::runtime::Builder::new_current_thread().build().unwrap().block_on(f)
    }

    #[test]
    fn hashing_is_deterministic_and_normalised() {
        let e = HashingEmbedder::new(64);
        let a = one(&e, "Protein breakfast at 8");
        assert_eq!(a, one(&e, "Protein breakfast at 8"));
        let norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
        assert!(one(&e, " ... ").iter().all(|x| *x == 0.0));
    }

    #[test]
    fn shared_terms_score_higher() {
        let e = HashingEmbedder::new(256);
        let q = one(&e, "protein breakfast");
        assert!(cosine(&q, &one(&e, "protein shake")) > cosine(&q, &one(&e, "running shoes")));
    }

    #[tokio::test]
    async fn checked_rejects_wrong_shape() {
        let texts = vec!["a".to_string(), "b".to_string()];
        assert!(matches!(embed_checked(&Bad { count: 1, len: 4 }, &texts).await, Err(ContextError::Embed(_))));
        assert!(matches!(embed_checked(&Bad { count: 2, len: 3 }, &texts).await, Err(ContextError::Embed(_))));
        assert_eq!(embed_checked(&Bad { count: 2, len: 4 }, &texts).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn trait_is_object_safe() {
        let e: Arc<dyn Embedder> = Arc::new(HashingEmbedder::new(8));
        let out = embed_checked(e.as_ref(), &["hello world".to_string()]).await.unwrap();
        assert_eq!(out[0].len(), 8);
    }
}
