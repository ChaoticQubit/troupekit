---
title: Introduction
description: What troupekit is and what ships in 0.3.
---

# Introduction

Troupekit is a Rust framework for building LLM agents from persona TOML files: providers, tools, memory, RAG and durable graph workflows.

One crate, `troupekit`, is the entry point. `use troupekit::prelude::*;` gives the `#[agent]` and `#[tool]` macros and the `Agent` and `Graph` builders. `Troupe::builder()` wires the five stores, in memory by default or in one SQLite file, and hands out a runner over them.

Source: `crates/troupekit/src/lib.rs`

## A first agent

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

Source: `crates/troupekit/examples/hello.rs`

The [Quick start](quick-start.md) runs this program and adds a tool.

## What 0.3 does

- Agents from a macro, a builder or a persona TOML file, with tools, typed output, evals and spend limits.
- Hosted models from nine catalog vendors, plus any OpenAI-compatible server.
- Local GGUF models through llama.cpp, and a route that picks local or cloud by policy.
- Folder search by keyword, vector or both, and long-term memory per user and session.
- Durable TOML workflows that survive a crash and resume from SQLite.
- One flag to trace every model call, tool call and graph step.

## The crates

Version 0.3.0 is a workspace of fifteen crates. You normally depend on `troupekit` only; it re-exports the engines as modules.

| Crate | What it does |
|---|---|
| `troupekit` | Entry crate: one builder that wires the stores, the runner and the engines. |
| `troupekit-kit` | Ergonomic API: one builder or one macro per agent, tool and graph. |
| `troupekit-macros` | The `#[tool]` and `#[agent]` attribute macros. |
| `troupekit-core` | Core types and storage traits. |
| `troupekit-agent` | Persona TOML agents: loader, checkpointable run state machine, typed output, evals and budgets. |
| `troupekit-graph` | Durable TOML workflows of agent, tool, human, router, join and end nodes. |
| `troupekit-host` | A leased job queue, an event log and a runner with worker and serverless step modes. |
| `troupekit-providers` | Model providers: wire protocols, catalog, transports and middleware. |
| `troupekit-tools` | Tool trait, registry, guarded web fetch, web search and declarative HTTP tools. |
| `troupekit-context` | Memory scopes, chunking, embeddings and hybrid retrieval. |
| `troupekit-index` | Vector indexes: exact brute force and HNSW, with f16 and i8 quantization. |
| `troupekit-store-sqlite` | Checkpoints, memory logs, blobs, a job queue and a usage ledger in one SQLite file. |
| `troupekit-local` | On-device models: the `LocalModel` trait, model catalog and llama.cpp. |
| `troupekit-trace` | Zero-dependency trace spans, sinks and viewers. |
| `troupekit-cli` | Command line: ingest a folder of documents into a searchable index in one SQLite file. |

Source: the `description` field of each `crates/*/Cargo.toml`.

## What is coming

- Coming in 0.4: Swift, Kotlin and Python bindings, and Apple and Android on-device models.
- Coming in 0.5: team, crew and handoff patterns in TOML, fan-out and forking a run.

The [Roadmap](roadmap.md) lists every planned release.

## License

Licensed under either of MIT or Apache-2.0, at your option.
