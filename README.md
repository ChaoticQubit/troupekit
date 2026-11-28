# Troupekit

Troupekit is an open-source Rust framework for LLM agents. You write an agent as a Rust function with `#[agent]`, or as a persona TOML file, give it tools, memory and documents, and run it in durable graph workflows that survive a restart.

## Why Troupekit

- **One Rust core.** Agents, workflows, retrieval, memory and tracing in one workspace, with no Python runtime.
- **Durable by default.** Runs checkpoint into one SQLite file, so a killed process resumes where it stopped.
- **Local or cloud.** Call Anthropic, OpenAI, Gemini or any OpenAI-compatible server, or run a GGUF model on the device through llama.cpp.
- **Your documents.** Keyword, vector and hybrid search over local files, with HNSW and quantized indexes.
- **Built for devices.** CI builds for Linux x86_64 and aarch64, Windows, macOS, Android and iOS.

## Quick start

Troupekit is not on crates.io yet. Run the bundled example from a clone:

```sh
git clone https://github.com/ChaoticQubit/troupekit
cd troupekit
export OPENAI_API_KEY=sk-...
cargo run -p troupekit --example hello
```

The example is the whole program:

```rust
use troupekit::prelude::*;

/// Greet the user by name.
#[agent(model = "openai/gpt-6-luna")]
fn greeter() {}

#[tokio::main]
async fn main() -> Result<(), KitError> {
    println!("{}", greeter().run("I am Ada").await?);
    Ok(())
}
```

The doc comment is the agent's instructions. The model is `vendor/model` from the built-in catalog in `catalog/providers.toml`, and the key comes from `<VENDOR>_API_KEY`. `crates/troupekit/examples/` has more: `tool_agent` (an agent with a `#[tool]`), `kill_resume` (a SQLite-backed workflow that resumes after its process is killed) and `support_agent` (answers from a folder of markdown articles, with `--features knowledge`).

To use it in your own project, depend on the repository:

```toml
[dependencies]
troupekit = { git = "https://github.com/ChaoticQubit/troupekit" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Crates

| Crate | What |
|---|---|
| `troupekit` | Entry crate. `use troupekit::prelude::*` for the macros and builders; `Troupe::builder()` wires in-memory or SQLite stores and hands out a runner; re-exports the engines |
| `troupekit-kit` | Ergonomic API: one builder or one macro per agent, tool and graph |
| `troupekit-macros` | The `#[agent]` and `#[tool]` attribute macros |
| `troupekit-core` | Messages, secrets, errors, events, ids, clock, storage traits and in-memory stores |
| `troupekit-providers` | Anthropic, OpenAI, Gemini and OpenAI-compatible wire protocols, provider catalog with prices, retry, fallback and cost metering |
| `troupekit-local` | On-device models: the `LocalModel` trait, a model catalog and llama.cpp |
| `troupekit-tools` | Tool trait, validating registry, SSRF-guarded web fetch, web search over Brave and SearXNG, declarative HTTP tools |
| `troupekit-context` | Memory scopes, profile facts, chunking, embeddings, hybrid retrieval and context blocks |
| `troupekit-index` | Vector indexes: exact brute force and HNSW, f16 and i8 quantization, persistable pages |
| `troupekit-agent` | Persona TOML agents: layered loader, checkpointable run, typed output, evals and LLM judge, budgets |
| `troupekit-graph` | TOML workflows of agent, tool, human, router, join and end nodes, with parallel branches, human pauses and retries |
| `troupekit-host` | Leased job queue, event log and a runner with worker and serverless step modes, crash-safe resume |
| `troupekit-store-sqlite` | One SQLite file for checkpoints, memory logs, blobs, the job queue and the usage ledger; optional SQLCipher |
| `troupekit-trace` | Zero-dependency trace spans, sinks and viewers |
| `troupekit-cli` | The `troupekit` command: `troupekit ingest <dir>` builds a searchable index in one SQLite file, `troupekit trace <file>` shows a trace |

## Feature flags

Features of the `troupekit` crate:

| Feature | Default | What |
|---|---|---|
| `kit` | yes | The `#[agent]` and `#[tool]` macros and the `Agent` and `Graph` builders |
| `sqlite` | yes | SQLite stores through `Troupe::builder().sqlite(path)` |
| `reqwest` | yes | The HTTP transport for cloud providers. Turn it off to build without the TLS stack |
| `knowledge` | no | Document loading, embeddings and hybrid search for agents |
| `sqlcipher` | no | Encrypted SQLite with a vendored OpenSSL |
| `sqlcipher-system` | no | Encrypted SQLite with the system OpenSSL |
| `local` | no | The `LocalModel` trait and model catalog |
| `llama` | no | llama.cpp models. Needs CMake and the llama.cpp source: run `sh scripts/fetch-llama.sh` or set `TROUPEKIT_LLAMA_SRC` |

## Supported Rust

Rust 1.89 or newer for `troupekit`, `troupekit-kit`, `troupekit-store-sqlite` and `troupekit-cli`; 1.85 or newer for the other crates.

## Status and roadmap

Troupekit is at 0.3, before 1.0: the API can change between minor versions.

| Release | Theme | A user can |
|---|---|---|
| 0.1 | Survives a restart | Kill the app mid-run and resume from SQLite |
| 0.2 | Knows your docs | Search local documents by keyword, vector or hybrid, and remember facts |
| 0.3 (current) | Thinks offline | Run on a local GGUF model or switch to cloud by policy, and trace a run |
| 0.4 | In your app | Call Troupekit from Swift, Kotlin and Python, and use Apple and Android on-device models |
| 0.5 | Works in teams | Use team, crew and handoff patterns in TOML, parallel branches, and fork a run |
| 0.6 | Plugs in anywhere | Use MCP tools, and expose Troupekit as an MCP server |
| 0.7 | Proven | Prove workflow properties with Bend, replay runs deterministically and run evals |
| 1.0 | On-device core | Meet the 1.0 acceptance criteria on Linux, Windows, macOS, Android and iOS |

## Documentation

API documentation is in the crate docs: `cargo doc --open -p troupekit`. A documentation site is in progress.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Report security issues as described in [SECURITY.md](SECURITY.md). Everyone taking part follows the [Code of Conduct](CODE_OF_CONDUCT.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE), at your option. Unless you state otherwise, any contribution you submit for inclusion is dual licensed the same way, without any additional terms or conditions.
