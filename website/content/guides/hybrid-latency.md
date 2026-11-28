---
title: Hybrid search latency
description: Time hybrid_search over one SQLite file for brute force and HNSW indexes in f32 and i8.
---

# Hybrid search latency

The `hybrid_latency` example seeds one SQLite file with chunks that carry clustered 384-dimension vectors, the same data as the index benchmarks (100 clusters, spread 0.05). It then times `hybrid_search` calls with `top_k` 8, for brute force and HNSW in f32 and i8.

Source: `crates/troupekit/examples/hybrid_latency.rs`

## Run it

```sh
cargo run --release -p troupekit --features knowledge --example hybrid_latency -- 50000 2000
```

Both arguments are optional: chunks, default 50 000, and queries, default 2 000.

Source: `crates/troupekit/examples/hybrid_latency.rs`, constants `CHUNKS_DEFAULT` and `QUERIES_DEFAULT`

## Output

The example prints one header line and a Markdown table, one row per index kind and quantization:

```text
chunks={chunks} dims={DIMS} queries={queries} top_k={TOP_K}
| index | quant | p50 us | p99 us |
|---|---|---|---|
```

Source: `crates/troupekit/examples/hybrid_latency.rs`

Run it on a quiet machine and again under load to see both the best and the worst case.

See [Knowledge](../concepts/knowledge.md) for index kinds and quantization.
