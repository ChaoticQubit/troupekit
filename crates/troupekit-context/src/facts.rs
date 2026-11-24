use crate::error::ContextError;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use troupekit_core::error::CoreError;
use troupekit_core::message::{Message, Role};
use troupekit_core::store::MemoryLog;

const FACT_LOG_PREFIX: &str = "facts/";
const MAX_ATTEMPTS: usize = 3;

/// One remembered fact about a subject, such as a user's weight or diet.
/// `source` says where it came from, `at_ms` when it was learned and
/// `confidence` how sure the writer was, from 0.0 to 1.0.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Fact {
    pub key: String,
    pub value: String,
    pub source: String,
    pub at_ms: i64,
    pub confidence: f32,
}

/// One stored edit in a subject's fact log.
/// `Set` writes a fact; `Remove` hides a key from that time on.
/// Serializes with an `op` tag of `set` or `remove`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum FactEntry {
    Set { fact: Fact },
    Remove { key: String, at_ms: i64 },
}

impl FactEntry {
    /// Returns the fact key and the time of this edit.
    /// Both variants carry them; the fold in `FactBook::list`
    /// compares these to pick the winning edit per key.
    fn key_and_time(&self) -> (&str, i64) {
        match self {
            FactEntry::Set { fact } => (&fact.key, fact.at_ms),
            FactEntry::Remove { key, at_ms } => (key, *at_ms),
        }
    }
}

/// Profile facts per subject, stored as an append-only `MemoryLog` per subject.
/// An edit is an append, so history is kept and two writers never lose one;
/// `list` folds the log to the newest value per key.
pub struct FactBook {
    log: Arc<dyn MemoryLog>,
}

impl FactBook {
    /// Creates a fact book over `log`; subject logs are named `facts/<subject>`.
    /// The log can be shared with run history, whose names never collide.
    /// Returns the new book.
    pub fn new(log: Arc<dyn MemoryLog>) -> Self {
        Self { log }
    }

    /// Records `fact` for `subject`. A fact older than the current value of its
    /// key is kept in history but does not win. Returns `ContextError::Invalid`
    /// for an empty subject or key, or a confidence outside 0.0..=1.0.
    pub async fn put(&self, subject: &str, fact: Fact) -> Result<(), ContextError> {
        check_key(&fact.key)?;
        if !(0.0..=1.0).contains(&fact.confidence) {
            return Err(ContextError::Invalid("confidence must be within 0.0..=1.0".into()));
        }
        self.append(subject, FactEntry::Set { fact }).await
    }

    /// Hides `key` for `subject` as of `at_ms`, until a newer `put`.
    /// Returns `ContextError::Invalid` for an empty subject or key,
    /// and passes store errors through.
    pub async fn remove(&self, subject: &str, key: &str, at_ms: i64) -> Result<(), ContextError> {
        check_key(key)?;
        self.append(subject, FactEntry::Remove { key: key.to_string(), at_ms }).await
    }

    /// Lists the current facts of `subject`, sorted by key.
    /// Per key the edit with the highest `at_ms` wins, ties go to the later
    /// edit, and a winning remove hides the key. Errors on a corrupt entry.
    pub async fn list(&self, subject: &str) -> Result<Vec<Fact>, ContextError> {
        let name = log_name(subject)?;
        let mut current: BTreeMap<String, FactEntry> = BTreeMap::new();
        for message in self.log.read(&name).await? {
            let entry: FactEntry = serde_json::from_str(&message.text_content())
                .map_err(|e| CoreError::Store(format!("corrupt fact entry in {name}: {e}")))?;
            let (key, at_ms) = entry.key_and_time();
            if current.get(key).is_none_or(|old| at_ms >= old.key_and_time().1) {
                current.insert(key.to_string(), entry);
            }
        }
        Ok(current
            .into_values()
            .filter_map(|e| match e {
                FactEntry::Set { fact } => Some(fact),
                FactEntry::Remove { .. } => None,
            })
            .collect())
    }

    /// Appends `entry` to the subject's log with a conditional append.
    /// Retries on `CoreError::Conflict` up to `MAX_ATTEMPTS` times in total,
    /// then returns the conflict; other store errors return at once.
    async fn append(&self, subject: &str, entry: FactEntry) -> Result<(), ContextError> {
        let name = log_name(subject)?;
        let json = serde_json::to_string(&entry).expect("fact entry serializes");
        for attempt in 1..=MAX_ATTEMPTS {
            let len = self.log.read(&name).await?.len() as u64;
            match self.log.append(&name, len, vec![Message::text(Role::System, json.clone())]).await {
                Ok(_) => return Ok(()),
                Err(CoreError::Conflict { .. }) if attempt < MAX_ATTEMPTS => continue,
                Err(e) => return Err(e.into()),
            }
        }
        unreachable!("the last attempt always returns")
    }
}

/// Builds the log name for `subject`.
/// Returns `ContextError::Invalid` when the subject is blank,
/// otherwise `facts/<subject>`.
fn log_name(subject: &str) -> Result<String, ContextError> {
    if subject.trim().is_empty() {
        return Err(ContextError::Invalid("subject must not be empty".into()));
    }
    Ok(format!("{FACT_LOG_PREFIX}{subject}"))
}

/// Checks that a fact `key` is not blank.
/// Returns `ContextError::Invalid` for a blank key,
/// otherwise `Ok(())`.
fn check_key(key: &str) -> Result<(), ContextError> {
    if key.trim().is_empty() {
        return Err(ContextError::Invalid("fact key must not be empty".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use troupekit_core::inmem::InMemMemory;
    use troupekit_core::message::Message;

    fn fact(key: &str, value: &str, at_ms: i64) -> Fact {
        Fact { key: key.into(), value: value.into(), source: "chat".into(), at_ms, confidence: 0.9 }
    }

    fn book() -> FactBook {
        FactBook::new(Arc::new(InMemMemory::default()))
    }

    fn values(facts: &[Fact]) -> Vec<(&str, &str)> {
        facts.iter().map(|f| (f.key.as_str(), f.value.as_str())).collect()
    }

    #[tokio::test]
    async fn newest_wins_per_key() {
        let b = book();
        b.put("u1", fact("weight", "80", 10)).await.unwrap();
        b.put("u1", fact("weight", "78", 20)).await.unwrap();
        b.put("u1", fact("height", "180", 5)).await.unwrap();
        assert_eq!(values(&b.list("u1").await.unwrap()), vec![("height", "180"), ("weight", "78")]);
    }

    #[tokio::test]
    async fn older_edit_does_not_override() {
        let b = book();
        b.put("u1", fact("weight", "78", 20)).await.unwrap();
        b.put("u1", fact("weight", "80", 10)).await.unwrap();
        assert_eq!(values(&b.list("u1").await.unwrap()), vec![("weight", "78")]);
    }

    #[tokio::test]
    async fn equal_time_later_write_wins() {
        let b = book();
        b.put("u1", fact("diet", "vegan", 10)).await.unwrap();
        b.put("u1", fact("diet", "vegetarian", 10)).await.unwrap();
        assert_eq!(values(&b.list("u1").await.unwrap()), vec![("diet", "vegetarian")]);
    }

    #[tokio::test]
    async fn remove_hides_key_until_newer_put() {
        let b = book();
        b.put("u1", fact("diet", "vegan", 10)).await.unwrap();
        b.remove("u1", "diet", 20).await.unwrap();
        assert!(b.list("u1").await.unwrap().is_empty());
        b.put("u1", fact("diet", "keto", 15)).await.unwrap();
        assert!(b.list("u1").await.unwrap().is_empty());
        b.put("u1", fact("diet", "keto", 30)).await.unwrap();
        assert_eq!(values(&b.list("u1").await.unwrap()), vec![("diet", "keto")]);
    }

    #[tokio::test]
    async fn subjects_are_separate() {
        let b = book();
        b.put("u1", fact("diet", "vegan", 10)).await.unwrap();
        assert!(b.list("u2").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn rejects_invalid() {
        let b = book();
        assert!(matches!(b.put("", fact("k", "v", 1)).await, Err(ContextError::Invalid(_))));
        assert!(matches!(b.put("u1", fact(" ", "v", 1)).await, Err(ContextError::Invalid(_))));
        for confidence in [-0.1, 1.5, f32::NAN] {
            let f = Fact { confidence, ..fact("k", "v", 1) };
            assert!(matches!(b.put("u1", f).await, Err(ContextError::Invalid(_))));
        }
        assert!(matches!(b.remove("u1", "", 1).await, Err(ContextError::Invalid(_))));
        assert!(matches!(b.list("").await, Err(ContextError::Invalid(_))));
    }

    struct Flaky {
        inner: InMemMemory,
        conflicts: AtomicUsize,
    }

    #[async_trait]
    impl MemoryLog for Flaky {
        async fn append(&self, log: &str, expected_len: u64, entries: Vec<Message>) -> Result<u64, CoreError> {
            if self.conflicts.load(Ordering::SeqCst) > 0 {
                self.conflicts.fetch_sub(1, Ordering::SeqCst);
                return Err(CoreError::Conflict { expected: expected_len, actual: expected_len + 1 });
            }
            self.inner.append(log, expected_len, entries).await
        }

        async fn read(&self, log: &str) -> Result<Vec<Message>, CoreError> {
            self.inner.read(log).await
        }
    }

    #[tokio::test]
    async fn retries_on_conflict_then_gives_up() {
        let b = FactBook::new(Arc::new(Flaky { inner: InMemMemory::default(), conflicts: AtomicUsize::new(2) }));
        b.put("u1", fact("diet", "vegan", 10)).await.unwrap();
        assert_eq!(b.list("u1").await.unwrap().len(), 1);
        let b = FactBook::new(Arc::new(Flaky { inner: InMemMemory::default(), conflicts: AtomicUsize::new(9) }));
        assert!(matches!(b.put("u1", fact("diet", "vegan", 10)).await, Err(ContextError::Core(CoreError::Conflict { .. }))));
    }
}
