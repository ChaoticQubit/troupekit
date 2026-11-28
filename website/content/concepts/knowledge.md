---
title: Knowledge
description: Let an agent search a folder of documents by keyword, vector or both.
---

# Knowledge

With the `knowledge` feature, `.knowledge(Knowledge::dir(path))` lets an agent search a folder. The folder is ingested at `build`, unchanged files are skipped, and the agent gets one tool, `search_knowledge`, whose results are fenced as untrusted text.

Source: `crates/troupekit-kit/src/agent.rs`, `crates/troupekit-kit/src/knowledge.rs`

## A searchable folder

```rust
let reader = || {
    Agent::new("reader")
        .instructions("Read.")
        .model("fake/m")
        .model_client(echo_user_text())
        .checkpoints(Storage::sqlite(dir.path().join("kb.db")))
        .knowledge(Knowledge::dir(&docs).embedder(embedder.clone()))
};
let built = reader().build().await.unwrap();
assert_eq!(built.persona().tools, ["search_knowledge".to_string()]);
let after_first = embedder.calls.load(Ordering::SeqCst);
assert!(after_first > 0);
reader().build().await.unwrap();
assert_eq!(embedder.calls.load(Ordering::SeqCst), after_first);
```

Source: `crates/troupekit-kit/tests/memory.rs`, test `knowledge_dir_ingests_once_then_skips`. The second `build` embeds nothing, because no file changed.

## Settings

| Method | Meaning | Default |
|---|---|---|
| `index(name)` | vector index and scope the folder is stored under, 1 to 64 of `[a-z0-9_]` | `docs` |
| `store(path)` | SQLite file the index lives in | the agent's `Storage::sqlite` file, else `./troupekit.db` |
| `embedder(e)` | embedder for the folder and for searches | the catalog embedder of the agent's vendor |
| `on_device_embedder(e)` | an embedder marked as on-device, for private agents | none |
| `local_embedder(choice)` | embed with a local model: catalog id, GGUF file or loaded model | none |
| `chunk(cfg)` | chunk size and overlap, in estimated tokens | 200 tokens, 20 of overlap |
| `load(settings)` | folder walk bounds and filters | `LoadSettings::default()` |
| `allow_failed(on)` | let `build` succeed when some files fail; they retry on the next build | `false` |

Source: `crates/troupekit-kit/src/knowledge.rs`

An agent whose vendor has no embedding row in the catalog needs `.embedder(..)`; `build` says so.

## What is read

A walk reads `txt`, `md`, `markdown`, `html` and `htm` files. The defaults are 10 000 files, 8 MiB per file, depth 16 and 100 000 directory entries. Symlinks are not followed, because a link can point outside the folder the user chose. Hidden entries such as `.git` are skipped. Going over a cap is an error that names it, so a walk never stops halfway in silence.

Source: `crates/troupekit-context/src/load.rs`

Text is split into chunks of at most `max_tokens` estimated tokens of four chars each, preferring paragraph breaks, then sentence ends, then spaces.

Source: `crates/troupekit-context/src/chunk.rs`

## Hybrid search

`hybrid_search` fuses a vector list and a keyword list with reciprocal rank fusion. `HybridMode::Vector` or `HybridMode::Keyword` uses one list alone. Defaults: mode `Hybrid`, `k_rrf` 60, 50 candidates per list per scope, 8 results, 16 filter pairs and 2 000 query chars.

Source: `crates/troupekit-context/src/hybrid.rs`

The keyword list is SQLite FTS5. The vector list comes from `troupekit-index`:

- `IndexKind::BruteForce`, exact, the default;
- `IndexKind::Hnsw(params)`, approximate;
- with `Quantization::F32` (default), `F16` or `I8`.

Source: `crates/troupekit-index/src/lib.rs`, `crates/troupekit-store-sqlite/src/vectors.rs`

## The engine API

Below the builder, the support example opens the store, ingests a folder and binds a knowledge kit to one user and session:

```rust
pub async fn open(db: &Path, docs: &Path, embedder: &dyn Embedder) -> Outcome<(SqliteStore, Arc<SqliteVectors>, IngestReport)> {
    let store = SqliteStore::open(db).await?;
    let vectors = Arc::new(store.vectors(INDEX, VectorSettings::new(embedder.dims())).await?);
    let report = ingest_dir(&vectors, embedder, docs, &IngestSettings::default()).await?;
    Ok((store, vectors, report))
}
```

Source: `crates/troupekit/examples/support/agent.rs`

See the [Support agent](../guides/support-agent.md) guide for the whole program.

## From the command line

`troupekit ingest` puts a folder into one SQLite file:

```text
usage: troupekit ingest <dir> [--db FILE] [--index NAME] [--scope NAME] --embedder SELECTOR [--catalog FILE] [--key-env VAR] [--chunk-tokens N] [--overlap-tokens N] [--kind brute|hnsw] [--quant f32|f16|i8]
```

Source: `crates/troupekit-cli/src/args.rs`

`--embedder` is a catalog `vendor/model` embedding row or `hashing:<dims>`, an offline hashing embedder.
