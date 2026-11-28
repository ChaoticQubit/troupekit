---
title: Kill and resume a local model call
description: Kill a process in the middle of on-device generation and resume the agent run from its SQLite checkpoint.
---

# Kill and resume a local model call

The `kill_model_call` example runs an agent on a local GGUF model with checkpoints in SQLite. One process starts a long story and is killed in the middle of the model call; a second process resumes the run by its id.

Source: `crates/troupekit/examples/kill_model_call.rs`

## Run it

The example needs the `llama` and `sqlite` features. First fetch llama.cpp and the test model:

```sh
sh scripts/fetch-llama.sh
sh scripts/fetch-test-model.sh
```

`fetch-test-model.sh` downloads `smollm2-135m-instruct-q2k`, checks its SHA-256 and prints its path in the model folder.

Source: `scripts/fetch-test-model.sh`

```text
usage: kill_model_call start <db> <gguf> | resume <db> <gguf> <run id>
```

Source: `crates/troupekit/examples/kill_model_call.rs`

- `start` runs a fresh story and prints `RUN <id>` on the first event, then `GENERATING` when the local model call starts.
- `resume` continues that run id from its checkpoint and prints `RESUMED <bytes>`.

The integration test `crates/troupekit/tests/kill_model_call.rs` drives both halves and kills the first process.

## The agent

```rust
async fn agent(db: &Path, gguf: &Path) -> Result<BuiltAgent, Box<dyn Error>> {
    let model = LlamaModel::load(gguf, LlamaSettings::default().with_context(CONTEXT))?;
    let settings = LocalChatSettings::default().with_ban_eos(true);
    let chat = LocalChat::new(Arc::new(model), "local/test").with_settings(settings);
    let store = SqliteStore::open(db).await?;
```

Source: `crates/troupekit/examples/kill_model_call.rs`

```rust
    let built = Agent::new("storyteller")
        .instructions("Write a very long story.")
        .model("local/test")
        .model_client(Arc::new(chat))
        .max_tokens(MAX_TOKENS)
        .checkpoints(Storage::custom(Arc::new(store)))
        .events(Arc::new(PrintRun(AtomicBool::new(false))))
        .trace([Trace::Models])
        .trace_to(Arc::new(sink))
        .build()
        .await?;
    Ok(built)
}
```

Source: `crates/troupekit/examples/kill_model_call.rs`

End of text is banned, so generation runs until `max_tokens` (960) and the kill lands mid call. The context is 1 024 tokens.

- `LlamaModel::load` opens the GGUF file with llama.cpp.
- `LocalChat` serves it as a chat `Model` under the selector `local/test`.
- `Storage::custom` keeps checkpoints in the SQLite file, so `BuiltAgent::resume(run_id)` works from the second process.
- A `CallbackSink` on the `Models` trace category prints `GENERATING` once, on the start of the first `model.local` span.

See [Local models and routing](../concepts/local.md).
