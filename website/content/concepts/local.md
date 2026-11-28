---
title: Local models and routing
description: Run an agent on a GGUF model with llama.cpp, and route each call to the device or the cloud by policy.
---

# Local models and routing

With the `local` and `llama` features, an agent can run on a GGUF model on the device. A route policy then decides, per model call, whether the device or the cloud answers.

Source: `crates/troupekit/Cargo.toml`, `crates/troupekit-kit/src/local.rs`, `crates/troupekit-kit/src/route.rs`

## Choose the on-device model

`.local(choice)` takes one of three things:

- a row id of the built-in local catalog, as a `&str`;
- a GGUF file, as a `PathBuf`;
- a loaded model, as an `Arc<dyn LocalModel>`.

Without `.model(..)` the agent is local only. A catalog model that is not in the model cache fails `build` unless `.download(true)` is set. Loading a file needs the `llama` feature.

Source: `crates/troupekit-kit/src/local.rs`

The local catalog has one row in 0.3, `smollm2-135m-instruct-q2k`, a 135M-parameter model for phones with an 8192-token context:

```toml
[[model]]
id = "smollm2-135m-instruct-q2k"
url = "https://huggingface.co/bartowski/SmolLM2-135M-Instruct-GGUF/resolve/09816acd5d99df7be770d85ea30822623dab342c/SmolLM2-135M-Instruct-Q2_K.gguf"
sha256 = "741ad12b64088fedc17c33aacb22e48be1972ef36a39f03666dd68bd15614fb9"
bytes = 88202080
ram_bytes = 268435456
context = 8192
class = "phone"
licence = "apache-2.0"
embeddings = false
recommended = false
```

Source: `crates/troupekit-local/catalog/local.toml`

## The model cache

Downloaded files live in one folder, one `<id>-<sha256>.gguf` per row. A file is used only when its size matches, and a download is checked against its SHA-256 before it gets its final name. The default folder is `$TROUPEKIT_MODEL_DIR`, else `$HOME/.cache/troupekit/models`, else `%LOCALAPPDATA%\troupekit\models`. A mobile app passes its own folder to `ModelCache::new(dir)`.

Source: `crates/troupekit-local/src/cache.rs`

## A loaded model as the client

This example loads a GGUF file with llama.cpp and serves it as the agent's model:

```rust
async fn agent(db: &Path, gguf: &Path) -> Result<BuiltAgent, Box<dyn Error>> {
    let model = LlamaModel::load(gguf, LlamaSettings::default().with_context(CONTEXT))?;
    let settings = LocalChatSettings::default().with_ban_eos(true);
    let chat = LocalChat::new(Arc::new(model), "local/test").with_settings(settings);
    let store = SqliteStore::open(db).await?;
    let said = AtomicBool::new(false);
    let sink = CallbackSink::new(move |line| {
        let local_start =
            line.contains("\"t\":\"start\"") && line.contains("\"name\":\"model.local\"");
        if local_start && !said.swap(true, Ordering::SeqCst) {
            println!("GENERATING");
            std::io::stdout().flush().expect("stdout flushes");
        }
    });
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

## Route policies

`.route(policy)` or `.route_file(path)` routes each model call. Target `cloud` is the `.model(..)` selector and target `local` is the `.local(..)` model; any other target id fails `build`.

```rust
let both = RoutePolicy::parse(
    "office.toml",
    "name = \"office\"\ndefault = \"cloud\"\n[[target]]\nid = \"local\"\n[[target]]\nid = \"cloud\"\n",
)
.unwrap();
```

Source: `crates/troupekit-kit/tests/local.rs`, test `private_agent_never_calls_cloud_transport`

### Policy keys

| Key | Meaning | Default |
|---|---|---|
| `name` | policy name, `[a-z0-9_-]` | required |
| `default` | target tried first whenever it is eligible | none |
| `order` | target ids in try order; targets left out follow in file order | file order |
| `[[target]] id` | target id, unique in the policy | required |
| `[[target]] tiers` | task tiers the target serves | none |
| `[[target]] context_tokens` | context window; a request that cannot fit skips the target | no limit |
| `[network] mode` | `auto`, `online` or `offline` | `auto` |
| `[network] offline_after_failures` | network errors in a row before the probe reports offline, 1 to 1 000 | 1 |
| `[network] retry_online_after_ms` | wait before an offline probe tries online, 1 to 86 400 000 | 30 000 |
| `[budget] cloud_usd_per_day` | daily cap on remote spend; at the cap, remote targets are skipped | no cap |
| `[device] min_battery_percent` | below it, on-device targets are skipped, 0 to 100 | no limit |
| `[device] max_thermal` | `nominal`, `fair`, `serious` or `critical`; hotter, on-device targets are skipped | no limit |
| `[fallback] on_error` | whether an error moves the request to the next target | `true` |

A policy holds at most 16 targets and 64 KiB of text. Fallback never switches after the first streamed delta.

Source: `crates/troupekit-providers/src/route/policy.rs`

Whether a target runs on the device is never read from TOML. Code declares it when it binds a model to the target id, since only code knows whether a model touches the network.

Source: `crates/troupekit-providers/src/route/decide.rs`

## Private agents

`.private(true)` keeps every request on the device. `build` refuses a private agent with no on-device model, a cloud knowledge embedder, or a judge eval naming a cloud model. A private request never leaves the device, even on fallback.

Source: `crates/troupekit-kit/src/route.rs`, `crates/troupekit-providers/src/route/policy.rs`

## Tiers, network and device state

- `.tier(name)` sends requests only to targets that list that tier. A tier no target serves fails `build`.
- `.network(state)` reads the network state from a `NetworkState` handle the app keeps and sets. Without it, a passive probe infers the state from network errors.
- `.device(state)` reads battery and thermal state from a `DeviceState` handle. Without it both are unknown and the device rules are skipped.

Source: `crates/troupekit-kit/src/route.rs`

Coming in 0.4: Apple and Android on-device models.
