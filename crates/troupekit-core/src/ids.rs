use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Identifier of one workflow run.
/// `new` makes a UUID v7, so ids sort by creation time as values and as
/// strings; `derived` makes a stable UUID v8. Serializes as the plain
/// hyphenated UUID string.
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

    /// Derives a stable id from run `parent` and `key`, such as a member
    /// run from its caller's run and the call's idempotency key. Equal
    /// inputs give equal ids. Hashes with 128-bit FNV-1a and sets the UUID
    /// version 8 and variant bits. Returns the derived id.
    pub fn derived(parent: RunId, key: &str) -> Self {
        const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
        const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
        let mut hash = OFFSET;
        for byte in parent.0.as_bytes().iter().chain(key.as_bytes()) {
            hash ^= u128::from(*byte);
            hash = hash.wrapping_mul(PRIME);
        }
        let mut bytes = hash.to_be_bytes();
        bytes[6] = (bytes[6] & 0x0f) | 0x80;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        let id = Uuid::from_bytes(bytes);
        assert_eq!(id.get_version_num(), 8, "derived ids are UUID v8");
        Self(id)
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

    #[test]
    fn derived_run_id_is_stable_and_distinct() {
        let parent = RunId::new();
        let other = RunId::new();
        let a = RunId::derived(parent, "run:3:1");
        assert_eq!(a, RunId::derived(parent, "run:3:1"));
        assert_ne!(a, RunId::derived(parent, "run:3:2"));
        assert_ne!(a, RunId::derived(other, "run:3:1"));
        assert_ne!(a, parent);
        assert_eq!(a.0.get_variant(), uuid::Variant::RFC4122);
        let mut seen = std::collections::HashSet::new();
        for i in 0..10_000 {
            assert!(seen.insert(RunId::derived(parent, &format!("key-{i}"))));
        }
        assert_eq!(seen.len(), 10_000);
    }
}
