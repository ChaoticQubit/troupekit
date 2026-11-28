---
title: Offline support agent
description: The support agent on a local model with the network off. A stub in 0.3.
---

# Offline support agent

The `support_agent_offline` example is a stub in 0.3. It compiles with the `llama` and `knowledge` features and does nothing yet.

```rust
/// Runs the support agent on a local model with the network off.
/// Input: none yet. Output: none yet.
/// This is a stub until the offline example lands.
fn main() {}
```

Source: `crates/troupekit/examples/support_agent_offline.rs`

## What works today

The pieces are in 0.3:

- A local chat model through `.local(..)` or `LocalChat`. See [Local models and routing](../concepts/local.md).
- An on-device embedder through `Knowledge::local_embedder(..)` or `on_device_embedder(..)`.
- `.private(true)`, which refuses at `build` any setting that would send a request off the device.

The [Support agent](support-agent.md) guide shows the online version, which already runs offline when its embedder is `hashing:256`; only the chat model needs the network.
