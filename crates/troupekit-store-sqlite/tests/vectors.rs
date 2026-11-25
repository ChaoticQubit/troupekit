use serde_json::json;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::time::{Duration, Instant};
use troupekit_core::error::CoreError;
use troupekit_core::ids::RunId;
use troupekit_core::inmem::InMemVectors;
use troupekit_core::store::{BlobStore, Checkpoint, CheckpointStore, Filter, KeywordStore, Scored, VectorRecord, VectorStore};
use troupekit_store_sqlite::{HnswParams, IndexKind, Quantization, SqliteStore, SqliteVectors, VectorSettings};

const DIMS: usize = 8;
const CHILD_DB: &str = "TROUPEKIT_VECTORS_CHILD_DB";
const DEADLINE: Duration = Duration::from_secs(30);

struct Temp {
    _dir: tempfile::TempDir,
    path: PathBuf,
}

fn temp() -> Temp {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("store.db");
    Temp { _dir: dir, path }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn vector(&mut self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| (self.next() % 2001) as f32 / 1000.0 - 1.0).collect()
    }
}

fn meta(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
}

fn rec(id: &str, vector: Vec<f32>, text: &str, pairs: &[(&str, &str)]) -> VectorRecord {
    VectorRecord { id: id.into(), vector, text: text.into(), meta: meta(pairs) }
}

fn sourced(id: &str, rng: &mut Rng, scope: &str, source: &str) -> VectorRecord {
    rec(id, rng.vector(DIMS), &format!("text of {id}"), &[("scope", scope), ("source", source)])
}

fn filter(pairs: &[(&str, &str)]) -> Filter {
    Filter(meta(pairs))
}

fn count(path: &Path, sql: &str) -> i64 {
    let conn = rusqlite::Connection::open(path).unwrap();
    conn.query_row(sql, [], |row| row.get(0)).unwrap()
}

fn ids(hits: &[Scored]) -> Vec<&str> {
    hits.iter().map(|h| h.id.as_str()).collect()
}

async fn docs(store: &SqliteStore) -> SqliteVectors {
    store.vectors("docs", VectorSettings::new(DIMS)).await.unwrap()
}

fn store_error(result: Result<impl std::fmt::Debug, CoreError>, needle: &str) {
    match result {
        Err(CoreError::Store(message)) => assert!(message.contains(needle), "{message} lacks {needle}"),
        other => panic!("expected Store error with {needle}, got {other:?}"),
    }
}

#[tokio::test]
async fn upsert_then_search_matches_brute_force_in_memory() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = store.vectors("docs", VectorSettings::new(16)).await.unwrap();
    let oracle = InMemVectors::new(16);
    let mut rng = Rng(7);
    let records: Vec<VectorRecord> = (0..2000)
        .map(|i| {
            let group = (i % 5).to_string();
            rec(&format!("r{i:05}"), rng.vector(16), &format!("chunk {i}"), &[("group", &group)])
        })
        .collect();
    for batch in records.chunks(500) {
        vectors.upsert(batch.to_vec()).await.unwrap();
        oracle.upsert(batch.to_vec()).await.unwrap();
    }
    for q in 0..20 {
        let query = rng.vector(16);
        for f in [filter(&[]), filter(&[("group", "2")])] {
            let got = vectors.search(&query, 10, &f).await.unwrap();
            let want = oracle.search(&query, 10, &f).await.unwrap();
            assert_eq!(ids(&got), ids(&want), "query {q}");
            for (g, w) in got.iter().zip(&want) {
                assert!((g.score - w.score).abs() < 1e-5, "query {q}: {} vs {}", g.score, w.score);
                assert_eq!((&g.text, &g.meta), (&w.text, &w.meta));
            }
        }
    }
    assert_eq!(vectors.len().await.unwrap(), 2000);
}

#[tokio::test]
async fn text_and_meta_round_trip() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let text = "Größe: 10 cm\n\t\"quoted\" 'single' \u{1F600} end";
    let pairs = [("ключ", "значение"), ("empty", ""), ("quote", "x' OR '1'='1")];
    vectors.upsert(vec![rec("a", vec![1.0; DIMS], text, &pairs)]).await.unwrap();
    drop(vectors);
    drop(store);
    let store = SqliteStore::open(&t.path).await.unwrap();
    let hits = docs(&store).await.search(&[1.0; DIMS], 1, &filter(&[])).await.unwrap();
    assert_eq!(hits[0].text, text);
    assert_eq!(hits[0].meta, meta(&pairs));
}

#[tokio::test]
async fn dims_mismatch_on_reopen_is_refused() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let err = vectors.upsert(vec![rec("a", vec![1.0; 3], "t", &[])]).await.unwrap_err();
    assert_eq!(err, CoreError::DimensionMismatch { expected: DIMS, actual: 3 });
    let err = vectors.search(&[1.0; 9], 1, &filter(&[])).await.unwrap_err();
    assert_eq!(err, CoreError::DimensionMismatch { expected: DIMS, actual: 9 });
    drop(vectors);
    drop(store);
    let store = SqliteStore::open(&t.path).await.unwrap();
    let err = store.vectors("docs", VectorSettings::new(16)).await.unwrap_err();
    assert_eq!(err, CoreError::DimensionMismatch { expected: DIMS, actual: 16 });
    assert_eq!(docs(&store).await.len().await.unwrap(), 0);
}

#[tokio::test]
async fn upsert_is_atomic_across_rows_fts_and_pages() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = store.vectors("docs", VectorSettings::new(DIMS).with_max_text_bytes(8)).await.unwrap();
    vectors.upsert(vec![rec("keep", vec![1.0; DIMS], "kept", &[("k", "v")])]).await.unwrap();
    let batch = vec![
        rec("a", vec![0.5; DIMS], "short", &[("k", "v")]),
        rec("b", vec![0.5; DIMS], "far too long", &[("k", "v")]),
        rec("c", vec![0.5; DIMS], "short", &[("k", "v")]),
    ];
    store_error(vectors.upsert(batch).await, "max_text_bytes");
    let rows = |sql| count(&t.path, sql);
    assert_eq!(rows("SELECT COUNT(*) FROM chunk"), 1);
    assert_eq!(rows("SELECT COUNT(*) FROM chunk_fts_docsize"), 1);
    assert_eq!(rows("SELECT COUNT(*) FROM chunk_meta"), 1);
    assert_eq!(rows("SELECT COUNT(*) FROM index_page"), 0);
    assert_eq!(vectors.len().await.unwrap(), 1);

    {
        let conn = rusqlite::Connection::open(&t.path).unwrap();
        conn.execute_batch("CREATE TRIGGER block BEFORE UPDATE ON vector_index BEGIN SELECT RAISE(ABORT, 'blocked'); END;")
            .unwrap();
    }
    let loads = vectors.open_report().loads();
    store_error(vectors.upsert(vec![rec("late", vec![0.25; DIMS], "late", &[])]).await, "blocked");
    assert_eq!(rows("SELECT COUNT(*) FROM chunk"), 1);
    {
        let conn = rusqlite::Connection::open(&t.path).unwrap();
        conn.execute_batch("DROP TRIGGER block").unwrap();
    }
    let hits = vectors.search(&[0.25; DIMS], 10, &filter(&[])).await.unwrap();
    assert_eq!(ids(&hits), ["keep"]);
    assert_eq!(vectors.len().await.unwrap(), 1);
    assert_eq!(vectors.open_report().loads(), loads + 1, "a failed write reloads from the file");
}

struct Worker(Child);

impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.0.try_wait(), Ok(None)) {
            let _ = self.0.kill();
        }
        let _ = self.0.wait();
    }
}

#[cfg(unix)]
#[tokio::test]
async fn killed_mid_upsert_leaves_a_consistent_index() {
    let t = temp();
    drop(docs(&SqliteStore::open(&t.path).await.unwrap()).await);
    let mut worker = Worker(
        Command::new(std::env::current_exe().unwrap())
            .args(["--ignored", "--exact", "vectors::child_upserts_until_killed", "--test-threads=1", "--quiet"])
            .env(CHILD_DB, &t.path)
            .stdout(std::process::Stdio::null())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    while count(&t.path, "SELECT generation FROM vector_index WHERE name = 'docs'") < 3 {
        assert!(start.elapsed() < DEADLINE, "child wrote no batches");
        assert!(matches!(worker.0.try_wait(), Ok(None)), "child exited early");
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    worker.0.kill().unwrap();
    let status = worker.0.wait().unwrap();
    assert!(!status.success(), "a killed child does not exit cleanly");
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let chunks = count(&t.path, "SELECT COUNT(*) FROM chunk");
    assert!(chunks >= 2 * 2000, "{chunks}");
    assert_eq!(chunks % 2000, 0, "only whole batches commit");
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk_fts_docsize"), chunks);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk_meta"), chunks);
    assert_eq!(vectors.len().await.unwrap() as i64, chunks);
    assert_eq!(vectors.open_report().rebuilds(), 0);
}

#[tokio::test]
#[ignore = "child process of killed_mid_upsert_leaves_a_consistent_index"]
async fn child_upserts_until_killed() {
    let Ok(path) = std::env::var(CHILD_DB) else {
        return;
    };
    let store = SqliteStore::open(&path).await.unwrap();
    let vectors = docs(&store).await;
    let mut rng = Rng(11);
    for batch in 0..10_000 {
        let records = (0..2000)
            .map(|i| rec(&format!("b{batch}-{i}"), rng.vector(DIMS), "child text to index", &[("batch", "x")]))
            .collect();
        vectors.upsert(records).await.unwrap();
    }
}

#[tokio::test]
async fn corrupt_page_rebuilds_on_open() {
    let t = temp();
    let mut rng = Rng(3);
    let query = rng.vector(DIMS);
    let before = {
        let store = SqliteStore::open(&t.path).await.unwrap();
        let vectors = docs(&store).await;
        let records = (0..50).map(|i| rec(&format!("r{i}"), rng.vector(DIMS), "t", &[])).collect();
        vectors.upsert(records).await.unwrap();
        vectors.search(&query, 5, &filter(&[])).await.unwrap()
    };
    {
        let conn = rusqlite::Connection::open(&t.path).unwrap();
        conn.execute("INSERT INTO index_page (index_name, page, bytes) VALUES ('docs', 0, X'DEADBEEF')", []).unwrap();
    }
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    assert_eq!(vectors.open_report().rebuilds(), 1);
    assert_eq!(vectors.open_report().loads(), 1);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM index_page WHERE index_name = 'docs'"), 0);
    assert_eq!(vectors.search(&query, 5, &filter(&[])).await.unwrap(), before);
    assert_eq!(vectors.len().await.unwrap(), 50);
}

#[tokio::test]
async fn switching_kind_rebuilds_and_keeps_results() {
    let t = temp();
    let mut rng = Rng(5);
    let queries: Vec<Vec<f32>> = (0..10).map(|_| rng.vector(DIMS)).collect();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let brute = docs(&store).await;
    let records = (0..200).map(|i| rec(&format!("r{i:03}"), rng.vector(DIMS), "t", &[])).collect();
    brute.upsert(records).await.unwrap();
    let mut exact = Vec::new();
    for q in &queries {
        exact.push(ids(&brute.search(q, 5, &filter(&[])).await.unwrap()).iter().map(|s| s.to_string()).collect::<Vec<_>>());
    }
    let hnsw = store.vectors("docs", VectorSettings::new(DIMS).with_kind(IndexKind::Hnsw(HnswParams::default()))).await.unwrap();
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM vector_index WHERE kind = 'hnsw'"), 1);
    assert!(count(&t.path, "SELECT COUNT(*) FROM index_page WHERE index_name = 'docs'") > 0);
    assert_eq!(hnsw.len().await.unwrap(), 200);
    for (q, want) in queries.iter().zip(&exact) {
        assert_eq!(&ids(&hnsw.search(q, 5, &filter(&[])).await.unwrap()), want);
    }
    let back = docs(&store).await;
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM index_page WHERE index_name = 'docs'"), 0);
    for (q, want) in queries.iter().zip(&exact) {
        assert_eq!(&ids(&back.search(q, 5, &filter(&[])).await.unwrap()), want);
    }
}

#[tokio::test]
async fn replace_source_drops_orphans() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let mut rng = Rng(9);
    let first: Vec<VectorRecord> = (0..3).map(|i| sourced(&format!("a{i}"), &mut rng, "docs", "a.md")).collect();
    assert_eq!(vectors.replace_source("docs", "a.md", first, "h1").await.unwrap(), 3);
    let other: Vec<VectorRecord> = (0..2).map(|i| sourced(&format!("b{i}"), &mut rng, "docs", "b.md")).collect();
    vectors.replace_source("docs", "b.md", other, "hb").await.unwrap();
    let shorter = vec![sourced("a0", &mut rng, "docs", "a.md")];
    assert_eq!(vectors.replace_source("docs", "a.md", shorter, "h2").await.unwrap(), 1);
    assert_eq!(vectors.len().await.unwrap(), 3);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk"), 3);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk_fts_docsize"), 3);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk_meta"), 6);
    let mut found = ids(&vectors.search(&rng.vector(DIMS), 10, &filter(&[])).await.unwrap()).iter().map(|s| s.to_string()).collect::<Vec<_>>();
    found.sort();
    assert_eq!(found, ["a0", "b0", "b1"]);
    assert_eq!(vectors.source_hash("docs", "a.md").await.unwrap().as_deref(), Some("h2"));
    assert_eq!(vectors.source_hash("docs", "b.md").await.unwrap().as_deref(), Some("hb"));
    assert_eq!(vectors.source_hash("docs", "c.md").await.unwrap(), None);
    assert_eq!(count(&t.path, "SELECT chunks FROM ingest_source WHERE source = 'a.md'"), 1);
}

#[tokio::test]
async fn reopen_after_replace_source_does_not_rebuild() {
    let t = temp();
    let mut rng = Rng(13);
    {
        let store = SqliteStore::open(&t.path).await.unwrap();
        let vectors = docs(&store).await;
        let first = (0..4).map(|i| sourced(&format!("a{i}"), &mut rng, "docs", "a.md")).collect();
        vectors.replace_source("docs", "a.md", first, "h1").await.unwrap();
        vectors.replace_source("docs", "a.md", vec![sourced("a9", &mut rng, "docs", "a.md")], "h2").await.unwrap();
    }
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    assert_eq!(vectors.open_report().rebuilds(), 0);
    assert_eq!(vectors.len().await.unwrap(), 1);
}

#[tokio::test]
async fn replace_source_refuses_records_without_scope_and_source() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let mut rng = Rng(17);
    let missing = vec![sourced("a0", &mut rng, "docs", "a.md"), rec("a1", rng.vector(DIMS), "t", &[("scope", "docs")])];
    store_error(vectors.replace_source("docs", "a.md", missing, "h").await, "record 1");
    let wrong_scope = vec![sourced("a0", &mut rng, "other", "a.md")];
    store_error(vectors.replace_source("docs", "a.md", wrong_scope, "h").await, "record 0");
    let wrong_source = vec![sourced("a0", &mut rng, "docs", "b.md")];
    store_error(vectors.replace_source("docs", "a.md", wrong_source, "h").await, "record 0");
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk"), 0);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM ingest_source"), 0);
    assert_eq!(vectors.len().await.unwrap(), 0);
}

#[tokio::test]
async fn failed_source_is_retried_and_cleared_by_replace() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let mut rng = Rng(19);
    let first = (0..2).map(|i| sourced(&format!("a{i}"), &mut rng, "docs", "a.md")).collect();
    vectors.replace_source("docs", "a.md", first, "h1").await.unwrap();
    let long_error = "é".repeat(1500);
    vectors.record_failed_source("docs", "a.md", &long_error).await.unwrap();
    assert_eq!(vectors.source_hash("docs", "a.md").await.unwrap(), None);
    assert_eq!(vectors.len().await.unwrap(), 2);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk"), 2);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM ingest_source WHERE status = 'failed'"), 1);
    let error_bytes = count(&t.path, "SELECT length(CAST(error AS BLOB)) FROM ingest_source");
    assert!((1..=1024).contains(&error_bytes), "{error_bytes}");
    vectors.record_failed_source("docs", "new.md", "unreadable").await.unwrap();
    assert_eq!(vectors.source_hash("docs", "new.md").await.unwrap(), None);
    let again = vec![sourced("a0", &mut rng, "docs", "a.md")];
    vectors.replace_source("docs", "a.md", again, "h2").await.unwrap();
    assert_eq!(vectors.source_hash("docs", "a.md").await.unwrap().as_deref(), Some("h2"));
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM ingest_source WHERE source = 'a.md' AND status = 'done' AND error IS NULL"), 1);
}

#[tokio::test]
async fn filter_values_are_bound_not_spliced() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let hostile = "x' OR '1'='1";
    vectors
        .upsert(vec![
            rec("tenant_a", vec![1.0; DIMS], "a", &[("tenant", "a")]),
            rec("tenant_b", vec![1.0; DIMS], "b", &[("tenant", "b")]),
            rec("literal", vec![1.0; DIMS], "l", &[("tenant", hostile)]),
        ])
        .await
        .unwrap();
    let hits = vectors.search(&[1.0; DIMS], 10, &filter(&[("tenant", "x")])).await.unwrap();
    assert!(hits.is_empty());
    let hits = vectors.search(&[1.0; DIMS], 10, &filter(&[("tenant", "x' OR '1'='1' --")])).await.unwrap();
    assert!(hits.is_empty());
    let hits = vectors.search(&[1.0; DIMS], 10, &filter(&[("tenant", hostile)])).await.unwrap();
    assert_eq!(ids(&hits), ["literal"]);
    let hits = vectors.search(&[1.0; DIMS], 10, &filter(&[("tenant\" OR 1=1 --", "a")])).await.unwrap();
    assert!(hits.is_empty());
    let hits = vectors.search(&[1.0; DIMS], 10, &filter(&[("tenant", "a")])).await.unwrap();
    assert_eq!(ids(&hits), ["tenant_a"]);
}

#[tokio::test]
async fn same_store_handles_share_one_index() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let a = docs(&store).await;
    let b = docs(&store.clone()).await;
    a.upsert(vec![rec("x", vec![1.0; DIMS], "t", &[])]).await.unwrap();
    assert_eq!(ids(&b.search(&[1.0; DIMS], 1, &filter(&[])).await.unwrap()), ["x"]);
    assert_eq!(a.open_report().loads(), 1);
    assert_eq!(b.open_report().loads(), 1);
    let other = store.vectors("other", VectorSettings::new(4)).await.unwrap();
    assert_eq!(other.len().await.unwrap(), 0);
    assert_eq!(other.dims(), 4);
}

#[tokio::test]
async fn other_connection_sees_new_records() {
    let t = temp();
    let first = SqliteStore::open(&t.path).await.unwrap();
    let second = SqliteStore::open(&t.path).await.unwrap();
    let mine = docs(&first).await;
    let theirs = docs(&second).await;
    assert!(mine.search(&[1.0; DIMS], 5, &filter(&[])).await.unwrap().is_empty());
    theirs.upsert(vec![rec("new", vec![1.0; DIMS], "t", &[])]).await.unwrap();
    assert_eq!(ids(&mine.search(&[1.0; DIMS], 5, &filter(&[])).await.unwrap()), ["new"]);
    assert_eq!(mine.open_report().loads(), 2);
    mine.upsert(vec![rec("mine", vec![0.5; DIMS], "t", &[])]).await.unwrap();
    assert_eq!(theirs.len().await.unwrap(), 2);
}

#[tokio::test]
async fn checkpoint_saves_do_not_reload_the_index() {
    let t = temp();
    let first = SqliteStore::open(&t.path).await.unwrap();
    let second = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&first).await;
    vectors.upsert(vec![rec("x", vec![1.0; DIMS], "t", &[])]).await.unwrap();
    let run = RunId::new();
    for step in 0..5 {
        second.save(Checkpoint { run_id: run, step, state: json!({ "step": step }) }).await.unwrap();
        second.put("blob", vec![step as u8]).await.unwrap();
        vectors.search(&[1.0; DIMS], 1, &filter(&[])).await.unwrap();
    }
    assert_eq!(vectors.open_report().loads(), 1);
}

#[tokio::test]
async fn every_bound_is_enforced() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    store_error(store.vectors("Bad-Name", VectorSettings::new(DIMS)).await, "name");
    store_error(store.vectors("docs", VectorSettings::new(0)).await, "dims");
    store_error(store.vectors("docs", VectorSettings::new(65_537)).await, "dims");
    store_error(store.vectors("docs", VectorSettings::new(DIMS).with_max_batch(0)).await, "max_batch");
    let settings = VectorSettings::new(DIMS)
        .with_max_batch(2)
        .with_max_meta_pairs(1)
        .with_max_meta_key_bytes(3)
        .with_max_meta_value_bytes(3)
        .with_max_text_bytes(4);
    let v = store.vectors("docs", settings).await.unwrap();
    let one = |id: &str, text: &str, pairs: &[(&str, &str)]| rec(id, vec![1.0; DIMS], text, pairs);
    store_error(v.upsert(vec![one("a", "t", &[]), one("b", "t", &[]), one("c", "t", &[])]).await, "max_batch");
    store_error(v.upsert(vec![one("a", "t", &[("k", "v"), ("l", "w")])]).await, "max_meta_pairs");
    store_error(v.upsert(vec![one("a", "t", &[("long", "v")])]).await, "meta key");
    store_error(v.upsert(vec![one("a", "t", &[("", "v")])]).await, "meta key");
    store_error(v.upsert(vec![one("a", "t", &[("k", "long")])]).await, "meta value");
    store_error(v.upsert(vec![one("a", "texts", &[])]).await, "max_text_bytes");
    store_error(v.upsert(vec![one("", "t", &[])]).await, "id");
    store_error(v.upsert(vec![one(&"i".repeat(257), "t", &[])]).await, "id");
    let mut nan = vec![1.0; DIMS];
    nan[2] = f32::NAN;
    store_error(v.upsert(vec![rec("a", nan.clone(), "t", &[])]).await, "non-finite");
    store_error(v.search(&nan, 1, &filter(&[])).await, "finite");
    store_error(v.search(&[1.0; DIMS], 1_001, &filter(&[])).await, "k");
    store_error(v.search(&[1.0; DIMS], 1, &filter(&[("a", "1"), ("b", "2")])).await, "max_meta_pairs");
    store_error(v.replace_source("", "a.md", vec![], "h").await, "scope");
    store_error(v.replace_source("d", "a.md", vec![], "").await, "content_hash");
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk"), 0);
}

#[tokio::test]
async fn defaults_are_the_safe_values() {
    let s = VectorSettings::new(384);
    assert_eq!(s.dims(), 384);
    assert_eq!(s.kind(), &IndexKind::BruteForce);
    assert_eq!(s.quantization(), Quantization::F32);
    assert_eq!(s.max_text_bytes(), 1024 * 1024);
    assert_eq!(s.max_meta_pairs(), 64);
    assert_eq!(s.max_meta_key_bytes(), 256);
    assert_eq!(s.max_meta_value_bytes(), 4096);
    assert_eq!(s.max_batch(), 10_000);
    assert_eq!(s.max_query_terms(), 32);
    assert_eq!(s.max_query_chars(), 2_000);
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let v = store.vectors("docs", s).await.unwrap();
    assert_eq!(v.open_report().loads(), 1);
    assert_eq!(v.open_report().rebuilds(), 0);
    let conn = rusqlite::Connection::open(&t.path).unwrap();
    let row: String = conn.query_row("SELECT kind || ':' || params_json || ':' || generation FROM vector_index", [], |r| r.get(0)).unwrap();
    assert_eq!(row, "brute_force:{}:0");
}

#[tokio::test]
async fn keyword_search_ranks_by_bm25() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    let other = store.vectors("other", VectorSettings::new(DIMS)).await.unwrap();
    vectors
        .upsert(vec![
            rec("once", vec![1.0; DIMS], "a refund is paid once the parcel is back at the warehouse", &[]),
            rec("twice", vec![1.0; DIMS], "refund policy: refund in 14 days", &[]),
            rec("none", vec![1.0; DIMS], "shipping takes two days", &[]),
            rec("résumé", vec![1.0; DIMS], "Résumé of the REFUND rules", &[]),
        ])
        .await
        .unwrap();
    let fillers = (0..8).map(|i| rec(&format!("filler{i}"), vec![1.0; DIMS], &format!("shipping note number {i}"), &[])).collect();
    vectors.upsert(fillers).await.unwrap();
    other.upsert(vec![rec("elsewhere", vec![1.0; DIMS], "refund refund refund", &[])]).await.unwrap();
    let hits = vectors.search_text("Refund", 10, &filter(&[])).await.unwrap();
    let conn = rusqlite::Connection::open(&t.path).unwrap();
    let mut stmt = conn
        .prepare(
            "SELECT c.id, -bm25(chunk_fts) FROM chunk_fts JOIN chunk c ON c.key = chunk_fts.rowid \
             WHERE chunk_fts MATCH 'refund' AND c.index_name = 'docs' ORDER BY bm25(chunk_fts), c.id",
        )
        .unwrap();
    let want: Vec<(String, f64)> = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap().map(Result::unwrap).collect();
    assert_eq!(want.len(), 3);
    assert_eq!(ids(&hits), want.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>());
    assert_eq!(hits[0].id, "twice");
    for (hit, (_, score)) in hits.iter().zip(&want) {
        assert!(hit.score > 0.0);
        assert!((f64::from(hit.score) - score).abs() < 1e-4, "{} vs {score}", hit.score);
    }
    assert!(ids(&vectors.search_text("resume", 10, &filter(&[])).await.unwrap()).contains(&"résumé"));
    assert_eq!(ids(&vectors.search_text("refund", 1, &filter(&[])).await.unwrap()), ["twice"]);
    assert!(vectors.search_text("refund", 0, &filter(&[])).await.unwrap().is_empty());
    assert!(vectors.search_text("absent", 10, &filter(&[])).await.unwrap().is_empty());
}

#[tokio::test]
async fn fts_operators_in_query_are_plain_text() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    vectors
        .upsert(vec![
            rec("a", vec![1.0; DIMS], "a near b and text secret", &[]),
            rec("b", vec![1.0; DIMS], "nothing to see", &[]),
        ])
        .await
        .unwrap();
    for query in ["NEAR(a b)", "a*", "text:secret", "\"", "AND OR NOT", "^a", "\"unclosed", "a + b - c", "{text}: a", "col:* OR", "(((", "'; DROP TABLE chunk; --", ""] {
        let result = vectors.search_text(query, 10, &filter(&[])).await;
        assert!(result.is_ok(), "{query}: {result:?}");
    }
    assert_eq!(ids(&vectors.search_text("text:secret", 10, &filter(&[])).await.unwrap()), ["a"]);
    assert_eq!(count(&t.path, "SELECT COUNT(*) FROM chunk"), 2);
}

#[tokio::test]
async fn keyword_filter_respects_scope() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let vectors = docs(&store).await;
    vectors
        .upsert(vec![
            rec("a1", vec![1.0; DIMS], "refund policy", &[("scope", "tenant_a")]),
            rec("b1", vec![1.0; DIMS], "refund policy", &[("scope", "tenant_b")]),
            rec("b2", vec![1.0; DIMS], "refund refund policy", &[("scope", "tenant_b"), ("lang", "en")]),
        ])
        .await
        .unwrap();
    assert_eq!(ids(&vectors.search_text("refund", 10, &filter(&[("scope", "tenant_a")])).await.unwrap()), ["a1"]);
    let b = vectors.search_text("refund", 10, &filter(&[("scope", "tenant_b")])).await.unwrap();
    assert_eq!(ids(&b), ["b2", "b1"]);
    let both = vectors.search_text("refund", 10, &filter(&[("scope", "tenant_b"), ("lang", "en")])).await.unwrap();
    assert_eq!(ids(&both), ["b2"]);
    assert!(vectors.search_text("refund", 10, &filter(&[("scope", "x' OR '1'='1")])).await.unwrap().is_empty());
}

#[tokio::test]
async fn query_term_and_length_caps() {
    let t = temp();
    let store = SqliteStore::open(&t.path).await.unwrap();
    let settings = VectorSettings::new(DIMS).with_max_query_terms(3).with_max_query_chars(20).with_max_meta_pairs(1);
    let vectors = store.vectors("docs", settings).await.unwrap();
    vectors.upsert(vec![rec("a", vec![1.0; DIMS], "one two three", &[])]).await.unwrap();
    assert_eq!(ids(&vectors.search_text("one, two; three", 10, &filter(&[])).await.unwrap()), ["a"]);
    store_error(vectors.search_text("one two three four", 10, &filter(&[])).await, "max_query_terms");
    store_error(vectors.search_text(&"x".repeat(21), 10, &filter(&[])).await, "max_query_chars");
    assert!(vectors.search_text(&"é".repeat(20), 10, &filter(&[])).await.is_ok());
    store_error(vectors.search_text("one", 1_001, &filter(&[])).await, "k must be");
    store_error(vectors.search_text("one", 1, &filter(&[("a", "1"), ("b", "2")])).await, "max_meta_pairs");
    store_error(store.vectors("docs", VectorSettings::new(DIMS).with_max_query_terms(0)).await, "max_query_terms");
}
