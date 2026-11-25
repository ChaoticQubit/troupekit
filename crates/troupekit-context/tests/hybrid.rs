use std::collections::BTreeMap;
use troupekit_context::embed::{Embedder, HashingEmbedder};
use troupekit_context::hybrid::{hybrid_search, rrf, HybridConfig, HybridMode};
use troupekit_context::ContextError;
use troupekit_core::store::{Filter, KeywordStore, VectorRecord, VectorStore};
use troupekit_store_sqlite::{SqliteStore, SqliteVectors, VectorSettings};

const DIMS: usize = 64;

const TEXTS: [(&str, &str, &str); 12] = [
    ("refund", "a", "Refunds are paid to the original card within 14 days of the return."),
    ("refund_late", "a", "A late refund can take one more billing cycle to show on the card."),
    ("shipping", "a", "Standard shipping takes two to four working days."),
    ("express", "a", "Express shipping arrives the next working day when ordered before noon."),
    ("returns", "a", "Returns need the original label and must arrive within 30 days."),
    ("warranty", "a", "The warranty covers defects for one year from the purchase date."),
    ("password", "a", "Reset a forgotten password from the sign in page."),
    ("invoice", "a", "Invoices are emailed after each payment and listed in the account."),
    ("b_refund", "b", "Tenant b refunds go to store credit, never to the card."),
    ("b_shipping", "b", "Tenant b ships only on Mondays."),
    ("b_returns", "b", "Tenant b accepts returns for 14 days."),
    ("b_card", "b", "Tenant b stores no card details."),
];

struct Kit {
    _dir: tempfile::TempDir,
    _store: SqliteStore,
    vectors: SqliteVectors,
    embedder: HashingEmbedder,
}

async fn kit() -> Kit {
    let dir = tempfile::tempdir().unwrap();
    let store = SqliteStore::open(dir.path().join("kb.db")).await.unwrap();
    let vectors = store.vectors("docs", VectorSettings::new(DIMS)).await.unwrap();
    let embedder = HashingEmbedder::new(DIMS);
    let texts: Vec<String> = TEXTS.iter().map(|(_, _, t)| t.to_string()).collect();
    let embedded = embedder.embed(&texts).await.unwrap();
    let records = TEXTS
        .iter()
        .zip(embedded)
        .map(|((id, tenant, text), vector)| VectorRecord {
            id: id.to_string(),
            vector,
            text: text.to_string(),
            meta: BTreeMap::from([("scope".to_string(), "docs".to_string()), ("tenant".to_string(), tenant.to_string())]),
        })
        .collect();
    vectors.upsert(records).await.unwrap();
    Kit { _dir: dir, _store: store, vectors, embedder }
}

fn tenant(t: &str) -> Filter {
    Filter(BTreeMap::from([("tenant".to_string(), t.to_string())]))
}

fn scoped(base: &Filter) -> Filter {
    let mut f = base.clone();
    f.0.insert("scope".into(), "docs".into());
    f
}

async fn run(k: &Kit, query: &str, base: &Filter, cfg: &HybridConfig) -> Result<Vec<String>, ContextError> {
    let hits = hybrid_search(&k.vectors, &k.vectors, &k.embedder, query, &["docs"], base, cfg).await?;
    Ok(hits.into_iter().map(|h| h.record.id).collect())
}

#[tokio::test]
async fn hybrid_modes_return_their_list_order() {
    let k = kit().await;
    let base = tenant("a");
    let cfg = HybridConfig::default().with_top_k(5).with_candidates(10);
    for query in ["refund to the card", "shipping days", "password"] {
        let qv = k.embedder.embed(&[query.to_string()]).await.unwrap().remove(0);
        let vector_ids: Vec<String> = k.vectors.search(&qv, 10, &scoped(&base)).await.unwrap().into_iter().map(|h| h.id).collect();
        let keyword_ids: Vec<String> = k.vectors.search_text(query, 10, &scoped(&base)).await.unwrap().into_iter().map(|h| h.id).collect();
        assert!(!keyword_ids.is_empty(), "{query}");
        let vector = run(&k, query, &base, &cfg.clone().with_mode(HybridMode::Vector)).await.unwrap();
        assert_eq!(vector, vector_ids.iter().take(5).cloned().collect::<Vec<_>>(), "{query}");
        let keyword = run(&k, query, &base, &cfg.clone().with_mode(HybridMode::Keyword)).await.unwrap();
        assert_eq!(keyword, keyword_ids.iter().take(5).cloned().collect::<Vec<_>>(), "{query}");
        let fused: Vec<String> = rrf(&[vector_ids, keyword_ids], 60).into_iter().take(5).map(|(id, _)| id).collect();
        assert_eq!(run(&k, query, &base, &cfg).await.unwrap(), fused, "{query}");
    }
    let hits = hybrid_search(&k.vectors, &k.vectors, &k.embedder, "refund", &["docs"], &base, &cfg).await.unwrap();
    let top = &hits[0];
    assert!(top.vector_rank.is_some() || top.keyword_rank.is_some());
    assert!(hits.windows(2).all(|w| w[0].score >= w[1].score));
}

#[tokio::test]
async fn base_filter_isolates_tenants() {
    let k = kit().await;
    for mode in [HybridMode::Hybrid, HybridMode::Vector, HybridMode::Keyword] {
        let cfg = HybridConfig::default().with_mode(mode);
        for query in ["refund card", "Tenant b returns", "shipping"] {
            for id in run(&k, query, &tenant("a"), &cfg).await.unwrap() {
                assert!(!id.starts_with("b_"), "{mode:?} {query}: {id}");
            }
            for id in run(&k, query, &tenant("b"), &cfg).await.unwrap() {
                assert!(id.starts_with("b_"), "{mode:?} {query}: {id}");
            }
        }
        assert!(run(&k, "refund", &tenant("x' OR '1'='1"), &cfg).await.unwrap().is_empty());
        let other_scope = hybrid_search(&k.vectors, &k.vectors, &k.embedder, "refund", &["other"], &tenant("a"), &cfg).await.unwrap();
        assert!(other_scope.is_empty());
    }
}

#[tokio::test]
async fn rejects_bad_input_like_retrieve() {
    let k = kit().await;
    let cfg = HybridConfig::default();
    let invalid = |r: Result<Vec<String>, ContextError>| matches!(r, Err(ContextError::Invalid(_)));
    assert!(invalid(run(&k, "  ", &tenant("a"), &cfg).await));
    let none: [&str; 0] = [];
    let r = hybrid_search(&k.vectors, &k.vectors, &k.embedder, "refund", &none, &tenant("a"), &cfg).await;
    assert!(matches!(r, Err(ContextError::Invalid(_))));
    let r = hybrid_search(&k.vectors, &k.vectors, &k.embedder, "refund", &["docs", " "], &tenant("a"), &cfg).await;
    assert!(matches!(r, Err(ContextError::Invalid(_))));
    let sets_scope = Filter(BTreeMap::from([("scope".to_string(), "other".to_string())]));
    assert!(invalid(run(&k, "refund", &sets_scope, &cfg).await));
    assert!(invalid(run(&k, "refund", &tenant("a"), &cfg.clone().with_top_k(0)).await));
    assert!(invalid(run(&k, "refund", &tenant("a"), &cfg.clone().with_candidates(4)).await));
    assert!(invalid(run(&k, "refund", &tenant("a"), &cfg.clone().with_candidates(1_001)).await));
    assert!(invalid(run(&k, "refund", &tenant("a"), &cfg.clone().with_max_filter_pairs(1)).await));
    assert!(invalid(run(&k, &"x".repeat(2_001), &tenant("a"), &cfg).await));
    assert!(run(&k, &"é".repeat(2_000), &tenant("a"), &cfg).await.is_ok());
}

#[test]
fn defaults_are_the_safe_values() {
    let cfg = HybridConfig::default();
    assert_eq!(cfg.mode(), HybridMode::Hybrid);
    assert_eq!(cfg.k_rrf(), 60);
    assert_eq!(cfg.candidates(), 50);
    assert_eq!(cfg.top_k(), 8);
    assert_eq!(cfg.max_filter_pairs(), 16);
    assert_eq!(cfg.max_query_chars(), 2_000);
    assert_eq!(HybridMode::default(), HybridMode::Hybrid);
}
