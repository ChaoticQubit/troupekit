---
title: Platform support
description: The targets CI builds and tests troupekit 0.3 on.
---

# Platform support

CI builds the whole workspace on pull requests for these targets:

| Target | Runner | Also checked |
|---|---|---|
| `x86_64-unknown-linux-gnu` | `ubuntu-latest` | tests of the changed crates |
| `aarch64-unknown-linux-gnu` | `ubuntu-24.04-arm` | |
| `aarch64-linux-android` | `ubuntu-latest`, cross build with the NDK | a build without the TLS stack; a SQLCipher build |
| `x86_64-pc-windows-msvc` | `windows-latest` | the SQLite store smoke tests |
| macOS | `macos-latest` | tests of the changed crates |
| `aarch64-apple-ios` | `macos-latest`, cross build | workspace without `troupekit-cli`; a build without the TLS stack; a SQLCipher build |

Source: `.github/workflows/ci.yml`

"Without the TLS stack" is `cargo build -p troupekit --no-default-features --features sqlite`: no `reqwest`, for an app that brings its own transport.

Source: `.github/workflows/ci.yml`, `crates/troupekit/Cargo.toml`

## Feature-specific jobs

- `llama`: a Linux job fetches llama.cpp and an embedding GGUF and runs the pooled-embedding test of `troupekit-local`.
- `sqlcipher` and `sqlcipher-system`: the SQLite store tests run with each feature on Linux.

Source: `.github/workflows/ci.yml`

## Toolchain

Rust 1.89 or newer for `troupekit` and `troupekit-cli`; Rust 1.85 or newer for the other crates. Edition 2021.

Source: `Cargo.toml`, `crates/troupekit/Cargo.toml`, `crates/troupekit-cli/Cargo.toml`

Coming in 0.4: Swift, Kotlin and Python bindings for iOS and Android apps.
