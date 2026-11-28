---
title: Feature flags
description: The Cargo features of every troupekit crate and what each one turns on.
---

# Feature flags

Most programs depend on `troupekit` and set features there; it forwards them to the engine crates.

## `troupekit`

| Feature | Turns on | Default |
|---|---|---|
| `kit` | `troupekit-kit`: macros, `Agent`, `Graph` | yes |
| `sqlite` | `troupekit-store-sqlite` | yes |
| `reqwest` | the reqwest HTTP transport in the agent, graph, host and kit crates | yes |
| `knowledge` | `sqlite`, `kit`, `troupekit-context` and the kit's `knowledge` | no |
| `sqlcipher` | `sqlite` with SQLCipher and vendored OpenSSL | no |
| `sqlcipher-system` | `sqlite` with SQLCipher and the system OpenSSL | no |
| `local` | `kit`, the kit's `local` and `troupekit-local` | no |
| `llama` | `local` and llama.cpp | no |

Source: `crates/troupekit/Cargo.toml`

## Engine crates

| Crate | Features | Default |
|---|---|---|
| `troupekit-kit` | `sqlite`, `knowledge`, `local`, `llama`, `reqwest` | `sqlite`, `reqwest` |
| `troupekit-agent` | `reqwest` | `reqwest` |
| `troupekit-graph` | `reqwest` | `reqwest` |
| `troupekit-host` | `reqwest` | `reqwest` |
| `troupekit-tools` | `reqwest` | `reqwest` |
| `troupekit-providers` | `reqwest`, `embeddings` | `reqwest` |
| `troupekit-store-sqlite` | `sqlcipher`, `sqlcipher-system` | none |
| `troupekit-local` | `llama`, `llama-vulkan`, `embed` | none |
| `troupekit-context` | `onnx`, the fastembed ONNX embedder | none |
| `troupekit-core`, `troupekit-index`, `troupekit-trace`, `troupekit-macros`, `troupekit-cli` | none | none |

Source: the `[features]` table of each `crates/*/Cargo.toml`

## Notes

- Without `llama`, the build runs no C or C++ compiler. With it, see [Install](../install.md).
- `llama-vulkan` builds llama.cpp with the Vulkan backend.
- `sqlite` uses rusqlite with the bundled SQLite, so no system SQLite is needed.
- `troupekit` without default features and with `sqlite` only is the phone build CI checks, with no TLS stack.

Source: `crates/troupekit-local/build.rs`, `crates/troupekit-local/build/plan.rs`, `crates/troupekit-store-sqlite/Cargo.toml`, `.github/workflows/ci.yml`
