---
title: FAQ
description: Short answers to common questions about troupekit 0.3.
---

# FAQ

## Is troupekit on crates.io?

Not yet. Use the git dependency on the `v0.3.0` tag. See [Install](install.md).

## Can I run an agent with no network?

Yes, with the `llama` feature: give it a local model with `.local(..)` and leave out `.model(..)`, or set `.private(true)`, which refuses at `build` any setting that would send a request off the device. See [Local models and routing](concepts/local.md).

## Can I test an agent without an API key?

Yes. `.model_client(m)` serves the model selector from any `Arc<dyn Model>`, such as the scripted `FnModel` in `troupekit_graph::testing`. It needs no key and no network. See [Agents and personas](concepts/agents-and-personas.md).

## Where does my data go?

Into the stores you choose. By default the five stores live in memory and nothing survives a restart. `Troupe::builder().sqlite(path)` or `Storage::sqlite(path)` keeps them in one SQLite file on your disk, encrypted with the `sqlcipher` feature and a key. Model calls go to the vendor of the selector you set, embeddings to the embedder's vendor, and web tools to the pages and search services they call. See [Memory and storage](concepts/memory.md).

## Does a crash lose a run?

Not with SQLite checkpoints. `BuiltAgent::resume(run_id)` and `BuiltGraph::resume(run_id)` continue from the latest checkpoint, also from a new process. See [Kill and resume a workflow](guides/kill-resume.md).

## Are API keys logged?

No. A key never appears in `Debug` output or an error, and tracing scrubs secrets whether or not content is kept. See [Models and providers](concepts/models.md) and [Tracing and budgets](concepts/tracing.md).

## Can a web page or document steer my agent?

Fetched pages, search results and knowledge chunks are fenced as untrusted text before the model sees them, and the built-in personas tell the model to treat them as data. Fencing lowers the risk; it does not remove it. Keep tools with side effects narrow.

## How do I cap spend?

Set `Limits` on the agent or graph, a `budget` in the persona, or `[budget] cloud_usd_per_day` in a route policy. See [Tracing and budgets](concepts/tracing.md).

## Can I call troupekit from Swift, Kotlin or Python?

Coming in 0.4.

## Does troupekit support MCP?

Coming in 0.6.
