//! One running summary per session, stored as a single replaced blob.
//! A replace keeps one row per session, so the summary never grows a log.

use crate::error::ContextError;
use std::sync::Arc;
use troupekit_core::error::CoreError;
use troupekit_core::store::BlobStore;

const SUMMARY_PREFIX: &str = "summary/";

/// Session summaries over a `BlobStore`, one blob per session at
/// `summary/<session>`. `set` replaces the text, `get` reads it back.
/// The blob store can be shared with other blobs; the prefix keeps them apart.
pub struct SessionSummary {
    blobs: Arc<dyn BlobStore>,
}

impl SessionSummary {
    /// Creates summaries over `blobs`.
    /// Input: the blob store. Output: the summaries handle.
    /// Nothing is read or written until `set` or `get`.
    pub fn new(blobs: Arc<dyn BlobStore>) -> Self {
        Self { blobs }
    }

    /// Replaces the summary of `session` with `text`.
    /// Input: a non-blank session and the text. Output: `Ok`, or
    /// `ContextError::Invalid` for a blank session; store errors pass through.
    pub async fn set(&self, session: &str, text: &str) -> Result<(), ContextError> {
        let path = path_of(session)?;
        self.blobs.put(&path, text.as_bytes().to_vec()).await?;
        Ok(())
    }

    /// Reads the summary of `session`.
    /// Input: a non-blank session. Output: the text, `None` when no summary
    /// was set, or `CoreError::Store` for a blob that is not UTF-8.
    pub async fn get(&self, session: &str) -> Result<Option<String>, ContextError> {
        let path = path_of(session)?;
        match self.blobs.get(&path).await {
            Ok(bytes) => {
                let text = String::from_utf8(bytes).map_err(|_| CoreError::Store(format!("corrupt summary at {path}")))?;
                Ok(Some(text))
            }
            Err(CoreError::NotFound(_)) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

/// Builds the blob path of `session`.
/// Input: the session. Output: `summary/<session>`, or
/// `ContextError::Invalid` when the session is blank.
fn path_of(session: &str) -> Result<String, ContextError> {
    if session.trim().is_empty() {
        return Err(ContextError::Invalid("session must not be empty".into()));
    }
    Ok(format!("{SUMMARY_PREFIX}{session}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use troupekit_core::inmem::InMemBlobs;

    #[tokio::test]
    async fn set_replaces_and_get_reads_back() {
        let s = SessionSummary::new(Arc::new(InMemBlobs::default()));
        assert_eq!(s.get("s1").await.unwrap(), None);
        s.set("s1", "first").await.unwrap();
        s.set("s1", "second").await.unwrap();
        assert_eq!(s.get("s1").await.unwrap().as_deref(), Some("second"));
        assert_eq!(s.get("s2").await.unwrap(), None);
        assert!(matches!(s.set(" ", "x").await, Err(ContextError::Invalid(_))));
    }
}
