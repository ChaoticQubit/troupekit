//! Hybrid retrieval: a vector list and a keyword list per scope, fused
//! with reciprocal-rank fusion. A new path next to `retrieve`, which
//! keeps its weighted-sum behaviour unchanged.

use crate::embed::{embed_checked, Embedder};
use crate::error::ContextError;
use crate::ingest::META_SCOPE;
use std::collections::BTreeMap;
use troupekit_core::store::{Filter, KeywordStore, Scored, VectorStore};

const K_RRF_DEFAULT: u32 = 60;
const CANDIDATES_DEFAULT: usize = 50;
const TOP_K_DEFAULT: usize = 8;
const MAX_FILTER_PAIRS_DEFAULT: usize = 16;
const MAX_QUERY_CHARS_DEFAULT: usize = 2_000;
const CANDIDATES_MAX: usize = 1_000;

/// Which lists `hybrid_search` fuses. `Hybrid` fuses the vector and the
/// keyword list, `Vector` and `Keyword` use one list alone, in its own
/// order. Default `Hybrid`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum HybridMode {
    #[default]
    Hybrid,
    Vector,
    Keyword,
}

/// Tuning for `hybrid_search`. Defaults: mode `Hybrid`, `k_rrf` 60, 50
/// candidates per list per scope, 8 results, 16 filter pairs and 2 000
/// query chars. Built with `Default` and the `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridConfig {
    mode: HybridMode,
    k_rrf: u32,
    candidates: usize,
    top_k: usize,
    max_filter_pairs: usize,
    max_query_chars: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            mode: HybridMode::default(),
            k_rrf: K_RRF_DEFAULT,
            candidates: CANDIDATES_DEFAULT,
            top_k: TOP_K_DEFAULT,
            max_filter_pairs: MAX_FILTER_PAIRS_DEFAULT,
            max_query_chars: MAX_QUERY_CHARS_DEFAULT,
        }
    }
}

impl HybridConfig {
    /// Sets which lists are fused.
    /// `Vector` and `Keyword` skip the other list entirely.
    /// Default `HybridMode::Hybrid`.
    pub fn with_mode(mut self, mode: HybridMode) -> Self {
        self.mode = mode;
        self
    }

    /// Sets the RRF constant: a hit at rank `r` adds `1 / (k_rrf + r)`.
    /// Higher values flatten the gap between top and lower ranks.
    /// Default 60.
    pub fn with_k_rrf(mut self, k_rrf: u32) -> Self {
        self.k_rrf = k_rrf;
        self
    }

    /// Sets how many hits each list fetches per scope before fusion.
    /// Valid from `top_k` to 1 000.
    /// Default 50.
    pub fn with_candidates(mut self, candidates: usize) -> Self {
        self.candidates = candidates;
        self
    }

    /// Sets how many fused hits are returned.
    /// Valid from 1 to `candidates`.
    /// Default 8.
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Sets the most filter pairs, the scope key included.
    /// A base filter that leaves no room for the scope key is refused.
    /// Default 16.
    pub fn with_max_filter_pairs(mut self, pairs: usize) -> Self {
        self.max_filter_pairs = pairs;
        self
    }

    /// Sets the longest query in chars.
    /// A longer query is refused before any store call.
    /// Default 2 000.
    pub fn with_max_query_chars(mut self, chars: usize) -> Self {
        self.max_query_chars = chars;
        self
    }

    /// Returns which lists are fused.
    /// One list alone keeps its own order.
    /// Default `HybridMode::Hybrid`.
    pub fn mode(&self) -> HybridMode {
        self.mode
    }

    /// Returns the RRF constant.
    /// A hit at rank `r` adds `1 / (k_rrf + r)`.
    /// Default 60.
    pub fn k_rrf(&self) -> u32 {
        self.k_rrf
    }

    /// Returns the hits fetched per list per scope.
    /// Fusion only sees these.
    /// Default 50.
    pub fn candidates(&self) -> usize {
        self.candidates
    }

    /// Returns the most fused hits returned.
    /// Cut after fusion across every scope.
    /// Default 8.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// Returns the most filter pairs, the scope key included.
    /// Checked before any store call.
    /// Default 16.
    pub fn max_filter_pairs(&self) -> usize {
        self.max_filter_pairs
    }

    /// Returns the longest query in chars.
    /// Checked before any store call.
    /// Default 2 000.
    pub fn max_query_chars(&self) -> usize {
        self.max_query_chars
    }
}

/// One fused hit. `record` is the stored chunk; its `score` is the
/// cosine when the vector list found it, else the BM25 score. `score` is
/// the RRF score, and each rank, from 1, is where the list placed it.
#[derive(Debug, Clone, PartialEq)]
pub struct HybridHit {
    pub record: Scored,
    pub score: f32,
    pub vector_rank: Option<u32>,
    pub keyword_rank: Option<u32>,
}

/// Fuses ranked id lists with reciprocal-rank fusion: an id at rank `r`,
/// from 1, adds `1 / (k_rrf + r)`, counted once per list. Input: the lists
/// and `k_rrf`. Output: each id once with its score, highest first, ties
/// by id; the result does not depend on the order of `lists`.
pub fn rrf(lists: &[Vec<String>], k_rrf: u32) -> Vec<(String, f32)> {
    let mut ranks: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for list in lists {
        let mut seen = std::collections::HashSet::new();
        for (index, id) in list.iter().enumerate() {
            if seen.insert(id.as_str()) {
                ranks.entry(id.as_str()).or_default().push(index + 1);
            }
        }
    }
    let mut fused: Vec<(String, f32)> = ranks
        .into_iter()
        .map(|(id, mut found)| {
            found.sort_unstable();
            let score: f64 = found.iter().map(|rank| 1.0 / (f64::from(k_rrf) + *rank as f64)).sum();
            (id.to_string(), score as f32)
        })
        .collect();
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    assert!(fused.windows(2).all(|w| w[0].0 != w[1].0), "rrf output has a duplicate id");
    assert!(fused.windows(2).all(|w| w[0].1 >= w[1].1), "rrf output is not sorted");
    fused
}

/// Searches each of `scopes` under `base` plus its scope key with the
/// lists `cfg.mode` names, fuses each scope with `rrf` and returns the
/// best `cfg.top_k` across scopes, highest first, ties by id. Errors
/// like `retrieve` on a blank query, no or a blank scope, or a `base`
/// that sets the scope key, and on a query or filter over its cap.
pub async fn hybrid_search(
    vectors: &dyn VectorStore,
    keywords: &dyn KeywordStore,
    embedder: &dyn Embedder,
    query: &str,
    scopes: &[&str],
    base: &Filter,
    cfg: &HybridConfig,
) -> Result<Vec<HybridHit>, ContextError> {
    validate(cfg)?;
    if query.trim().is_empty() {
        return Err(ContextError::Invalid("query must not be empty".into()));
    }
    if query.chars().count() > cfg.max_query_chars {
        return Err(ContextError::Invalid(format!("query is above max_query_chars {}", cfg.max_query_chars)));
    }
    if scopes.is_empty() || scopes.iter().any(|s| s.trim().is_empty()) {
        return Err(ContextError::Invalid("need at least one non-empty scope".into()));
    }
    if base.0.contains_key(META_SCOPE) {
        return Err(ContextError::Invalid("base filter must not set the scope key".into()));
    }
    if base.0.len() + 1 > cfg.max_filter_pairs {
        return Err(ContextError::Invalid(format!("base filter has {} pairs, above max_filter_pairs {} with the scope key", base.0.len(), cfg.max_filter_pairs)));
    }
    let use_vector = cfg.mode != HybridMode::Keyword;
    let use_keyword = cfg.mode != HybridMode::Vector;
    let query_vec = if use_vector { Some(embed_checked(embedder, &[query.to_string()]).await?.remove(0)) } else { None };
    let mut best: BTreeMap<String, HybridHit> = BTreeMap::new();
    for scope in scopes {
        let mut filter = base.clone();
        filter.0.insert(META_SCOPE.to_string(), scope.to_string());
        let vector_hits = match &query_vec {
            Some(v) => vectors.search(v, cfg.candidates, &filter).await?,
            None => Vec::new(),
        };
        let keyword_hits = if use_keyword { keywords.search_text(query, cfg.candidates, &filter).await? } else { Vec::new() };
        for hit in fuse(vector_hits, keyword_hits, cfg.k_rrf)? {
            let keep = best.get(&hit.record.id).is_none_or(|old| hit.score > old.score);
            if keep {
                best.insert(hit.record.id.clone(), hit);
            }
        }
    }
    let mut hits: Vec<HybridHit> = best.into_values().collect();
    hits.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.record.id.cmp(&b.record.id)));
    hits.truncate(cfg.top_k);
    assert!(hits.iter().all(|h| h.vector_rank.is_some() || h.keyword_rank.is_some()), "a fused hit came from no list");
    Ok(hits)
}

/// Fuses one scope's vector and keyword hits. Input: both lists in
/// rank order and `k_rrf`. Output: one `HybridHit` per id with its RRF
/// score and ranks, highest first, ties by id.
fn fuse(vector_hits: Vec<Scored>, keyword_hits: Vec<Scored>, k_rrf: u32) -> Result<Vec<HybridHit>, ContextError> {
    let vector_ids: Vec<String> = vector_hits.iter().map(|h| h.id.clone()).collect();
    let keyword_ids: Vec<String> = keyword_hits.iter().map(|h| h.id.clone()).collect();
    let mut records: BTreeMap<String, Scored> = BTreeMap::new();
    for hit in keyword_hits.into_iter().chain(vector_hits) {
        records.insert(hit.id.clone(), hit);
    }
    let rank_of = |ids: &[String], id: &str| -> Result<Option<u32>, ContextError> {
        match ids.iter().position(|x| x == id) {
            Some(index) => u32::try_from(index + 1).map(Some).map_err(|_| ContextError::Invalid("rank out of range".into())),
            None => Ok(None),
        }
    };
    let mut out = Vec::new();
    for (id, score) in rrf(&[vector_ids.clone(), keyword_ids.clone()], k_rrf) {
        let record = records.remove(&id).expect("every fused id has a record");
        out.push(HybridHit { score, vector_rank: rank_of(&vector_ids, &id)?, keyword_rank: rank_of(&keyword_ids, &id)?, record });
    }
    Ok(out)
}

/// Checks a config before a search. Input: the config. Output: `Ok`,
/// or `ContextError::Invalid` for `top_k` 0, `candidates` below `top_k`
/// or above 1 000, or a zero query or filter cap.
fn validate(cfg: &HybridConfig) -> Result<(), ContextError> {
    if cfg.top_k == 0 || cfg.candidates < cfg.top_k {
        return Err(ContextError::Invalid("need 0 < top_k <= candidates".into()));
    }
    if cfg.candidates > CANDIDATES_MAX {
        return Err(ContextError::Invalid(format!("candidates must be at most {CANDIDATES_MAX}")));
    }
    if cfg.max_filter_pairs == 0 || cfg.max_query_chars == 0 {
        return Err(ContextError::Invalid("max_filter_pairs and max_query_chars must be positive".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn rrf_known_values() {
        let fused = rrf(&[list(&["a", "b", "c"]), list(&["b", "d"])], 60);
        let want = [
            ("b", 1.0 / 62.0 + 1.0 / 61.0),
            ("a", 1.0 / 61.0),
            ("c", 1.0 / 63.0),
            ("d", 1.0 / 62.0),
        ];
        let mut want: Vec<(&str, f64)> = want.to_vec();
        want.sort_by(|x, y| y.1.total_cmp(&x.1).then_with(|| x.0.cmp(y.0)));
        assert_eq!(fused.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), want.iter().map(|(id, _)| *id).collect::<Vec<_>>());
        for ((_, got), (_, w)) in fused.iter().zip(&want) {
            assert!((f64::from(*got) - w).abs() < 1e-7, "{got} vs {w}");
        }
        assert_eq!(fused[0].0, "b");
        assert!(rrf(&[], 60).is_empty());
    }

    #[test]
    fn rrf_is_order_independent() {
        let lists = [list(&["a", "b", "c", "a"]), list(&["c", "x"]), list(&["b", "c", "y", "z"])];
        let forward = rrf(&lists, 60);
        let mut reversed = lists.to_vec();
        reversed.reverse();
        assert_eq!(rrf(&reversed, 60), forward);
        let rotated = [lists[1].clone(), lists[2].clone(), lists[0].clone()];
        assert_eq!(rrf(&rotated, 60), forward);
        assert_eq!(forward.iter().filter(|(id, _)| id == "a").count(), 1);
    }

    #[test]
    fn rrf_ties_by_id() {
        let fused = rrf(&[list(&["z", "m"]), list(&["a", "q"])], 60);
        assert_eq!(fused.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["a", "z", "m", "q"]);
        assert_eq!(fused[0].1, fused[1].1);
    }
}
