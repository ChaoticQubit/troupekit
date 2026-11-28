---
title: Benchmarks
description: The benchmark examples of the engine crates and how to run them.
---

# Benchmarks

The engine crates ship small benchmark programs as examples. Run each with `--release`, once on a quiet machine and once under a normal working load, and report both.

## Vector index

```sh
cargo run --release -p troupekit-index --example recall -- 100000
cargo run --release -p troupekit-index --example latency
```

- `recall` measures recall@10 of HNSW against brute force on clustered 384-dimension data and asserts that the default `ef_search` reaches 0.95.
- `latency` measures index-only query latency and memory at 50 000 vectors of 384 dimensions, and prints p50, p99 and heap bytes per kind and quantization.

Source: `crates/troupekit-index/examples/recall.rs`, `crates/troupekit-index/examples/latency.rs`

## Hybrid search

See [Hybrid search latency](hybrid-latency.md).

## SQLite checkpoints

```sh
cargo run --release -p troupekit-store-sqlite --example bench_checkpoint -- full
```

Times checkpoint writes per state size at one durability level, `full` (the default) or `normal`, and prints p50, p99 and max per size plus the file size.

Source: `crates/troupekit-store-sqlite/examples/bench_checkpoint.rs`

## Agent loop overhead

```sh
cargo run --release -p troupekit-kit --example kit_loop
cargo run --release -p troupekit-kit --example kit_loop_traced
cargo run --release -p troupekit-kit --example kit_overhead
```

- `kit_loop` runs a built agent 10 000 times against a scripted model, after a warm-up, and prints the median and p99 in microseconds.
- `kit_loop_traced` does the same with tracing on, once into a `JsonlSink` file and once into a `MemorySink`.
- `kit_overhead` compares a run through the `Agent` builder with a direct `AgentRun` on the same persona and prints the median overhead per run.

Source: `crates/troupekit-kit/examples/kit_loop.rs`, `crates/troupekit-kit/examples/kit_loop_traced.rs`, `crates/troupekit-kit/examples/kit_overhead.rs`

## Trace points

```sh
cargo run --release -p troupekit-trace --example point_cost
```

Measures the cost of one disabled `trace!` point with no tracer, with a live tracer on another category and with a live tracer on the same category outside its scope, plus one disabled `trace_span!` with `Span::instrument`. Prints the median and spread in nanoseconds per row.

Source: `crates/troupekit-trace/examples/point_cost.rs`

## Local generation

```sh
cargo run --release -p troupekit-local --features llama --example local_bench -- model.gguf
```

```text
usage: local_bench <model.gguf> [--runs N] [--prompt-tokens N] [--gen-tokens N] [--context N]
```

Measures local generation speed for one GGUF model: a cold call and a prefix-reuse call per run, at a fixed length with end of text banned. Prints a table of medians.

Source: `crates/troupekit-local/examples/local_bench.rs`
