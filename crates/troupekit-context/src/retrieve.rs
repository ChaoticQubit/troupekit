use crate::embed::{embed_checked, Embedder};
use crate::error::ContextError;
use crate::ingest::{META_AT, META_SCOPE};
use crate::text::terms;
use std::collections::{BTreeMap, BTreeSet};
use troupekit_core::clock::Clock;
use troupekit_core::store::{Filter, Scored, VectorStore};

const BM25_K1: f32 = 1.2;
const BM25_B: f32 = 0.75;
const DAY_MS: i64 = 86_400_000;

/// Tuning for `retrieve`.
/// `candidates` hits are fetched per scope, `top_k` are returned, and the
/// three weights mix cosine, normalised BM25 and recency, where recency halves
/// every `half_life_ms`. The default is 20, 8, 0.6, 0.3, 0.1 and 30 days.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrievalConfig {
    pub candidates: usize,
    pub top_k: usize,
    pub vector_weight: f32,
    pub keyword_weight: f32,
    pub recency_weight: f32,
    pub half_life_ms: i64,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self { candidates: 20, top_k: 8, vector_weight: 0.6, keyword_weight: 0.3, recency_weight: 0.1, half_life_ms: 30 * DAY_MS }
    }
}

impl RetrievalConfig {
    /// Checks the config before a search.
    /// Returns `ContextError::Invalid` for `top_k` 0, `candidates` below
    /// `top_k`, a negative or non-finite weight, or a non-positive half-life.
    fn validate(&self) -> Result<(), ContextError> {
        if self.top_k == 0 || self.candidates < self.top_k {
            return Err(ContextError::Invalid("need 0 < top_k <= candidates".into()));
        }
        if [self.vector_weight, self.keyword_weight, self.recency_weight].iter().any(|w| !w.is_finite() || *w < 0.0) {
            return Err(ContextError::Invalid("weights must be finite and non-negative".into()));
        }
        if self.half_life_ms <= 0 {
            return Err(ContextError::Invalid("half_life_ms must be positive".into()));
        }
        Ok(())
    }
}

/// One retrieved chunk with its scores.
/// `record.score` is the cosine similarity, `keyword` the BM25 score divided by
/// the best in the candidate set, `recency` in 0.0..=1.0, and `score` the mix.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub record: Scored,
    pub score: f32,
    pub keyword: f32,
    pub recency: f32,
}

/// Searches each of `scopes` under `base` plus its scope key and ranks the merged
/// candidates by vector, keyword and recency score, highest first, ties by id.
/// Returns at most `cfg.top_k` hits, or `ContextError::Invalid` for a blank query,
/// no or a blank scope, a `base` that sets the scope key, or a bad `cfg`.
pub async fn retrieve(
    vectors: &dyn VectorStore,
    embedder: &dyn Embedder,
    clock: &dyn Clock,
    query: &str,
    scopes: &[&str],
    base: &Filter,
    cfg: &RetrievalConfig,
) -> Result<Vec<Hit>, ContextError> {
    cfg.validate()?;
    if query.trim().is_empty() {
        return Err(ContextError::Invalid("query must not be empty".into()));
    }
    if scopes.is_empty() || scopes.iter().any(|s| s.trim().is_empty()) {
        return Err(ContextError::Invalid("need at least one non-empty scope".into()));
    }
    if base.0.contains_key(META_SCOPE) {
        return Err(ContextError::Invalid("base filter must not set the scope key".into()));
    }
    let query_vec = embed_checked(embedder, &[query.to_string()]).await?.remove(0);
    let mut merged: BTreeMap<String, Scored> = BTreeMap::new();
    for scope in scopes {
        let mut filter = base.clone();
        filter.0.insert(META_SCOPE.to_string(), scope.to_string());
        for hit in vectors.search(&query_vec, cfg.candidates, &filter).await? {
            merged.insert(hit.id.clone(), hit);
        }
    }
    let candidates: Vec<Scored> = merged.into_values().collect();
    let keyword = bm25(query, &candidates.iter().map(|c| c.text.as_str()).collect::<Vec<_>>());
    let best = keyword.iter().copied().fold(0.0f32, f32::max);
    let now = clock.now_ms();
    let mut hits: Vec<Hit> = candidates
        .into_iter()
        .zip(keyword)
        .map(|(record, kw)| {
            let keyword = if best > 0.0 { kw / best } else { 0.0 };
            let recency = recency(&record.meta, now, cfg.half_life_ms);
            let score = cfg.vector_weight * record.score + cfg.keyword_weight * keyword + cfg.recency_weight * recency;
            Hit { record, score, keyword, recency }
        })
        .collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.record.id.cmp(&b.record.id)));
    hits.truncate(cfg.top_k);
    Ok(hits)
}

/// Scores each of `texts` against `query` with Okapi BM25 (k1 1.2, b 0.75).
/// Document frequencies come from `texts` alone, the candidate set.
/// Returns one non-negative score per text, in order.
fn bm25(query: &str, texts: &[&str]) -> Vec<f32> {
    let docs: Vec<Vec<String>> = texts.iter().map(|t| terms(t)).collect();
    let n = docs.len() as f32;
    let total: usize = docs.iter().map(Vec::len).sum();
    let avg_len = if total == 0 { 1.0 } else { total as f32 / n };
    let query_terms: BTreeSet<String> = terms(query).into_iter().collect();
    let mut scores = vec![0.0f32; docs.len()];
    for q in &query_terms {
        let df = docs.iter().filter(|d| d.contains(q)).count() as f32;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        for (score, doc) in scores.iter_mut().zip(&docs) {
            let tf = doc.iter().filter(|t| *t == q).count() as f32;
            if tf > 0.0 {
                let norm = 1.0 - BM25_B + BM25_B * doc.len() as f32 / avg_len;
                *score += idf * tf * (BM25_K1 + 1.0) / (tf + BM25_K1 * norm);
            }
        }
    }
    for s in &scores {
        assert!(s.is_finite() && *s >= 0.0, "bm25 score out of range");
    }
    scores
}

/// Recency weight of a record written at its `META_AT` time, seen at `now`.
/// Halves every `half_life_ms`; a future time counts as now and gives 1.0,
/// a missing or unparseable time gives 0.0.
fn recency(meta: &BTreeMap<String, String>, now: i64, half_life_ms: i64) -> f32 {
    assert!(half_life_ms > 0, "half-life checked by validate");
    match meta.get(META_AT).and_then(|v| v.parse::<i64>().ok()) {
        Some(at) => 0.5f64.powf(now.saturating_sub(at).max(0) as f64 / half_life_ms as f64) as f32,
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::ChunkConfig;
    use crate::embed::HashingEmbedder;
    use crate::ingest::{ingest, Document, Origin, META_SOURCE};
    use async_trait::async_trait;
    use std::collections::BTreeMap;
    use troupekit_core::clock::FixedClock;
    use troupekit_core::inmem::InMemVectors;
    use troupekit_core::store::VectorRecord;

    const DIMS: usize = 64;
    const NOW: i64 = 1_000_000_000;

    struct Env {
        store: InMemVectors,
        embedder: HashingEmbedder,
        clock: FixedClock,
    }

    impl Env {
        fn new() -> Self {
            Self { store: InMemVectors::new(DIMS), embedder: HashingEmbedder::new(DIMS), clock: FixedClock::new(NOW) }
        }

        async fn add(&self, scope: &str, source: &str, text: &str, at_ms: i64, user: &str) {
            let doc = Document {
                scope: scope.into(),
                source: source.into(),
                text: text.into(),
                at_ms,
                origin: Origin::Internal,
                meta: BTreeMap::from([("user".to_string(), user.to_string())]),
            };
            ingest(&self.store, &self.embedder, &doc, &ChunkConfig { max_tokens: 200, overlap_tokens: 0 }).await.unwrap();
        }

        async fn run(&self, query: &str, scopes: &[&str], base: &Filter, cfg: &RetrievalConfig) -> Result<Vec<Hit>, ContextError> {
            retrieve(&self.store, &self.embedder, &self.clock, query, scopes, base, cfg).await
        }
    }

    fn only(vector: f32, keyword: f32, recency: f32) -> RetrievalConfig {
        RetrievalConfig { vector_weight: vector, keyword_weight: keyword, recency_weight: recency, ..RetrievalConfig::default() }
    }

    fn sources(hits: &[Hit]) -> Vec<&str> {
        hits.iter().map(|h| h.record.meta[META_SOURCE].as_str()).collect()
    }

    fn user(u: &str) -> Filter {
        Filter(BTreeMap::from([("user".to_string(), u.to_string())]))
    }

    #[tokio::test]
    async fn searches_every_linked_scope() {
        let env = Env::new();
        env.add("goal:1", "a", "protein breakfast plan", NOW, "1").await;
        env.add("goal:2", "b", "protein dinner plan", NOW, "1").await;
        env.add("goal:3", "c", "protein lunch plan", NOW, "1").await;
        let hits = env.run("protein plan", &["goal:1", "goal:2"], &Filter::default(), &RetrievalConfig::default()).await.unwrap();
        let mut got = sources(&hits);
        got.sort();
        assert_eq!(got, vec!["a", "b"]);
    }

    #[tokio::test]
    async fn base_filter_isolates_tenants() {
        let env = Env::new();
        env.add("goal:1", "mine", "protein plan", NOW, "1").await;
        env.add("goal:1", "theirs", "protein plan", NOW, "2").await;
        let hits = env.run("protein plan", &["goal:1"], &user("2"), &RetrievalConfig::default()).await.unwrap();
        assert_eq!(sources(&hits), vec!["theirs"]);
    }

    struct Fixed;

    #[async_trait]
    impl Embedder for Fixed {
        fn dims(&self) -> usize {
            2
        }

        async fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ContextError> {
            Ok(texts.iter().map(|t| if t.starts_with("q:") || t.contains("alpha") { vec![1.0, 0.0] } else { vec![0.0, 1.0] }).collect())
        }
    }

    #[tokio::test]
    async fn keyword_weight_reorders() {
        let store = InMemVectors::new(2);
        let clock = FixedClock::new(NOW);
        let meta = |source: &str| BTreeMap::from([(META_SCOPE.to_string(), "s".to_string()), (META_SOURCE.to_string(), source.to_string())]);
        store
            .upsert(vec![
                VectorRecord { id: "1".into(), vector: vec![0.0, 1.0], text: "rare word here".into(), meta: meta("rare") },
                VectorRecord { id: "2".into(), vector: vec![1.0, 0.0], text: "common alpha".into(), meta: meta("alpha") },
            ])
            .await
            .unwrap();
        let run = |cfg: RetrievalConfig| {
            let (store, clock) = (&store, &clock);
            async move { retrieve(store, &Fixed, clock, "q: rare", &["s"], &Filter::default(), &cfg).await.unwrap() }
        };
        assert_eq!(sources(&run(only(1.0, 0.0, 0.0)).await), vec!["alpha", "rare"]);
        let by_keyword = run(only(0.0, 1.0, 0.0)).await;
        assert_eq!(sources(&by_keyword), vec!["rare", "alpha"]);
        assert_eq!(by_keyword[0].keyword, 1.0);
        assert_eq!(by_keyword[1].keyword, 0.0);
    }

    #[tokio::test]
    async fn recency_weight_reorders() {
        let env = Env::new();
        let day = 86_400_000;
        env.add("goal:1", "old", "protein plan", NOW - 60 * day, "1").await;
        env.add("goal:1", "new", "protein plan", NOW - day, "1").await;
        env.add("goal:1", "future", "protein plan", NOW + day, "1").await;
        env.store
            .upsert(vec![VectorRecord {
                id: "undated".into(),
                vector: env.embedder.embed(&["protein plan".to_string()]).await.unwrap().remove(0),
                text: "protein plan".into(),
                meta: BTreeMap::from([(META_SCOPE.to_string(), "goal:1".to_string()), (META_SOURCE.to_string(), "undated".to_string())]),
            }])
            .await
            .unwrap();
        let hits = env.run("protein plan", &["goal:1"], &Filter::default(), &only(0.0, 0.0, 1.0)).await.unwrap();
        assert_eq!(sources(&hits), vec!["future", "new", "old", "undated"]);
        assert_eq!(hits[0].recency, 1.0);
        assert!((hits[2].recency - 0.25).abs() < 1e-4);
        assert_eq!(hits[3].recency, 0.0);
    }

    #[tokio::test]
    async fn deterministic_tie_break_by_id() {
        let env = Env::new();
        for s in ["x", "y", "z"] {
            env.add("goal:1", s, "same text", NOW, "1").await;
        }
        let hits = env.run("same text", &["goal:1"], &Filter::default(), &RetrievalConfig::default()).await.unwrap();
        let ids: Vec<&str> = hits.iter().map(|h| h.record.id.as_str()).collect();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
    }

    #[tokio::test]
    async fn top_k_bounds_the_result() {
        let env = Env::new();
        for s in ["a", "b", "c", "d"] {
            env.add("goal:1", s, "protein plan", NOW, "1").await;
        }
        let cfg = RetrievalConfig { top_k: 2, ..RetrievalConfig::default() };
        assert_eq!(env.run("protein", &["goal:1"], &Filter::default(), &cfg).await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn rejects_bad_input() {
        let env = Env::new();
        let d = RetrievalConfig::default();
        let f = Filter::default();
        let scoped = Filter(BTreeMap::from([(META_SCOPE.to_string(), "goal:9".to_string())]));
        let bad_cfgs = [
            RetrievalConfig { top_k: 0, ..d.clone() },
            RetrievalConfig { candidates: 3, top_k: 4, ..d.clone() },
            RetrievalConfig { vector_weight: -1.0, ..d.clone() },
            RetrievalConfig { keyword_weight: f32::NAN, ..d.clone() },
            RetrievalConfig { half_life_ms: 0, ..d.clone() },
        ];
        for cfg in &bad_cfgs {
            assert!(matches!(env.run("q", &["goal:1"], &f, cfg).await, Err(ContextError::Invalid(_))), "{cfg:?}");
        }
        assert!(matches!(env.run(" ", &["goal:1"], &f, &d).await, Err(ContextError::Invalid(_))));
        assert!(matches!(env.run("q", &[], &f, &d).await, Err(ContextError::Invalid(_))));
        assert!(matches!(env.run("q", &[""], &f, &d).await, Err(ContextError::Invalid(_))));
        assert!(matches!(env.run("q", &["goal:1"], &scoped, &d).await, Err(ContextError::Invalid(_))));
    }
}
