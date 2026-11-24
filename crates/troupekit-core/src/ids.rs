use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Identifier of one workflow run.
/// A UUID v7, so ids sort by creation time as values and as strings.
/// Serializes as the plain hyphenated UUID string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(pub Uuid);

impl RunId {
    /// Creates a new run id from the current time.
    /// Ids made later in the same process sort after earlier ones.
    /// Returns the new id.
    pub fn new() -> Self {
        Self(Uuid::now_v7())
    }
}

impl Default for RunId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_are_unique_and_time_ordered() {
        let a = RunId::new();
        let b = RunId::new();
        assert_ne!(a, b);
        assert!(a.to_string() < b.to_string());
        let back: RunId = serde_json::from_str(&serde_json::to_string(&a).unwrap()).unwrap();
        assert_eq!(back, a);
    }
}
