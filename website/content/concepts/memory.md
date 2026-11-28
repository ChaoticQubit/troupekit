---
title: Memory and storage
description: Long-term memory per user and session, run checkpoints, and the one-file SQLite store.
---

# Memory and storage

Troupekit keeps five things: run checkpoints, memory logs, blobs, a job queue and a usage ledger. `Troupe::builder()` keeps all five in memory by default, or in one SQLite file with `sqlite(path)`.

Source: `crates/troupekit/src/lib.rs`

## Long-term memory

With the `knowledge` feature, `.memory(AgentMemory::new(user, session))` gives an agent facts about one user and a summary of one session. The agent gets five tools bound to them:

- `remember_fact`
- `forget_fact`
- `recall_facts`
- `update_session_summary`
- `read_session_summary`

By default the facts and the summary are prepended to every run's input. The host picks `user` and `session`, never the model, so one user can never read another's facts:

```rust
let dir = tempfile::tempdir().unwrap();
let remember = calling("remember_fact", json!({"key": "color", "value": "blue"}));
let ada = || AgentMemory::new("ada", "s1").sqlite(db(&dir));
rememberer(remember, ada()).run("note it").await.unwrap();
let recall = || calling("recall_facts", json!({}));
let bob = AgentMemory::new("bob", "s1").sqlite(db(&dir));
let seen = rememberer(recall(), bob).run("what?").await.unwrap();
assert!(!seen.contains("blue"), "{seen}");
let seen = rememberer(recall(), ada()).run("what?").await.unwrap();
assert!(seen.contains("blue"), "{seen}");
```

Source: `crates/troupekit-kit/tests/memory.rs`, test `two_users_never_share_memory`. `calling` and `rememberer` are helpers in the same file.

| Method | Meaning | Default |
|---|---|---|
| `sqlite(path)` | the SQLite file facts and summaries live in | required, unless the agent has `.knowledge(..)`, whose store is used |
| `preamble(on)` | prepend facts and summary to each run | `true` |
| `fact_caps(bytes, facts, log)` | bytes per fact, facts per user, fact log length | 1024, 200, 1000 |
| `max_summary_bytes(n)` | session summary size | 4096 |
| `preamble_budget_tokens(n)` | preamble size; facts that do not fit are left out | 800 |

`user` and `session` are 1 to 256 bytes each, checked at `build`.

Source: `crates/troupekit-kit/src/memory.rs`

## Run checkpoints

`.checkpoints(storage)` picks where an agent or graph keeps run checkpoints:

- `Storage::memory()`, the default: a fresh in-memory store per run. Runs cannot be resumed.
- `Storage::sqlite(path)`: the SQLite file at `path`, opened at `build`. Enables `resume(run_id)`, also from a later process.
- `Storage::custom(store)`: any `Arc<dyn CheckpointStore>`, such as a store the host already opened.

Source: `crates/troupekit-kit/src/storage.rs`

## The SQLite store

`SqliteStore` holds checkpoints with history, memory logs, blobs, the job queue and the usage ledger in one file, in WAL mode, with its own migrations. A file at a newer schema version is refused; an older one is migrated on open.

| Setting | Default |
|---|---|
| `durability` | `Durability::Full`: every commit uses `fullfsync` and survives power loss. `Normal` is faster and can lose the last commits on power loss, never on a process kill. |
| `max_state_bytes` | 64 MiB per checkpoint state |
| `max_blob_bytes` | 64 MiB per blob |
| `max_history_steps` | 100 000 |
| `busy_timeout_ms` | 5 000 |
| `secure_delete` | on: deleted rows, including a removed chunk's text and vector, are overwritten |
| `key` | none; with the `sqlcipher` feature, opens the file encrypted with SQLCipher |

Source: `crates/troupekit-store-sqlite/src/lib.rs`, `README.md`

Open one with `SqliteStore::open(path)` or `SqliteStore::builder(path)`, and hand it to `Troupe::builder().sqlite_with(builder)` or `Storage::custom(..)`.
