---
title: API reference
description: Links to the generated Rust API docs of every troupekit crate.
---

# API reference

The API docs are generated from the doc comments of each crate.

Troupekit 0.3 is not on crates.io yet, so these docs.rs pages go live only once the crates are published. Until then, build the same docs locally from a clone:

```sh
cargo doc --workspace --no-deps --open
```

| Crate | docs.rs |
|---|---|
| `troupekit` | <https://docs.rs/troupekit> |
| `troupekit-kit` | <https://docs.rs/troupekit-kit> |
| `troupekit-macros` | <https://docs.rs/troupekit-macros> |
| `troupekit-core` | <https://docs.rs/troupekit-core> |
| `troupekit-agent` | <https://docs.rs/troupekit-agent> |
| `troupekit-graph` | <https://docs.rs/troupekit-graph> |
| `troupekit-host` | <https://docs.rs/troupekit-host> |
| `troupekit-providers` | <https://docs.rs/troupekit-providers> |
| `troupekit-tools` | <https://docs.rs/troupekit-tools> |
| `troupekit-context` | <https://docs.rs/troupekit-context> |
| `troupekit-index` | <https://docs.rs/troupekit-index> |
| `troupekit-store-sqlite` | <https://docs.rs/troupekit-store-sqlite> |
| `troupekit-local` | <https://docs.rs/troupekit-local> |
| `troupekit-trace` | <https://docs.rs/troupekit-trace> |
| `troupekit-cli` | <https://docs.rs/troupekit-cli> |

Source: the package names in `crates/*/Cargo.toml`

Start with `troupekit` and its `prelude` module. The engine crates are re-exported from it as `troupekit::agent`, `troupekit::core`, `troupekit::graph`, `troupekit::host`, `troupekit::providers` and `troupekit::trace`, plus `troupekit::context` with `knowledge` and `troupekit::local` with `local`.

Source: `crates/troupekit/src/lib.rs`
