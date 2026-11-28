---
title: Install
description: Add troupekit 0.3 to a Rust project and pick its features.
---

# Install

## Requirements

- Rust 1.89 or newer. That is the `rust-version` of the `troupekit` crate; the engine crates need 1.85.
- A tokio runtime. The examples use `#[tokio::main]`.
- For the `llama` feature only: `cmake` 3.14 or newer and a C and C++ compiler. The default build runs no C or C++ compiler.

Source: `Cargo.toml`, `crates/troupekit/Cargo.toml`, `crates/troupekit-local/build.rs`, `crates/troupekit-local/build/plan.rs`

## Add the dependency

Troupekit is not on crates.io yet. Use the git tag:

```toml
[dependencies]
troupekit = { git = "https://github.com/ChaoticQubit/troupekit", tag = "v0.3.0" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Source: composed from the `repository` key of `Cargo.toml`, the `v0.3.0` tag and the tokio features of `crates/troupekit/Cargo.toml` `[dev-dependencies]`.

## Pick features

The default features are `sqlite`, `reqwest` and `kit`. Add others in the dependency line:

```toml
troupekit = { git = "https://github.com/ChaoticQubit/troupekit", tag = "v0.3.0", features = ["knowledge"] }
```

Source: composed from the dependency line above and `crates/troupekit/Cargo.toml` `[features]`.

| Feature | Adds |
|---|---|
| `kit` | The `#[agent]` and `#[tool]` macros and the `Agent` and `Graph` builders. Default. |
| `sqlite` | All five stores in one SQLite file. Default. |
| `reqwest` | The HTTP transport for hosted models and web tools. Default. |
| `knowledge` | Folder search and long-term memory: `Knowledge`, `AgentMemory`, hybrid search. |
| `local` | On-device models through the `LocalModel` trait. |
| `llama` | llama.cpp behind `local`, for GGUF files. |
| `sqlcipher` | An encrypted SQLite file, with vendored OpenSSL. |
| `sqlcipher-system` | An encrypted SQLite file, with the system crypto library. |

Source: `crates/troupekit/Cargo.toml` `[features]`, `crates/troupekit-store-sqlite/Cargo.toml` `[features]`

See [Feature flags](reference/feature-flags.md) for every crate's flags.

## Build llama.cpp

With the `llama` feature, the build script needs the llama.cpp source. In a clone of the repository:

```sh
sh scripts/fetch-llama.sh
```

The script downloads one pinned llama.cpp release, checks its SHA-256 and unpacks it into `crates/troupekit-local/llama.cpp`. Outside the repository, set one of these instead:

- `TROUPEKIT_LLAMA_SRC`: a llama.cpp checkout to build.
- `TROUPEKIT_LLAMA_LIB_DIR`: a folder with a prebuilt `libllama` and `tk_llama`.

On macOS and iOS, Metal is on unless `TROUPEKIT_LLAMA_METAL=0`. An Android build needs `ANDROID_NDK_HOME`.

Source: `scripts/fetch-llama.sh`, `crates/troupekit-local/build.rs`, `crates/troupekit-local/build/plan.rs`

## Install the command line

The `troupekit-cli` crate builds a binary called `troupekit`:

```sh
cargo install --git https://github.com/ChaoticQubit/troupekit --tag v0.3.0 troupekit-cli
```

Source: `crates/troupekit-cli/Cargo.toml` `[[bin]]`

It has two commands, `troupekit ingest` and `troupekit trace`. See [Knowledge](concepts/knowledge.md) and [Tracing and budgets](concepts/tracing.md).
