---
title: Outline
description: Information architecture of the troupekit 0.3 documentation, one row per page with its sources.
---

# Outline

Every page is tool-neutral Markdown with front matter `title` and `description` only. Every code sample is copied from an example, a doc test, a test or a shipped TOML file, with a `Source:` line under the block. Anything planned for 0.4 or later is marked "coming in 0.x".

## Start here

| Page | File | Sources |
|---|---|---|
| Introduction | `index.md` | `crates/*/Cargo.toml`, `crates/troupekit/src/lib.rs` |
| Install | `install.md` | `Cargo.toml`, `crates/troupekit/Cargo.toml` |
| Quick start | `quick-start.md` | `crates/troupekit/examples/hello.rs`, `crates/troupekit/examples/tool_agent/` |

## Concepts

| Page | File | Sources |
|---|---|---|
| Agents and personas | `concepts/agents-and-personas.md` | `crates/troupekit-kit/src/agent.rs`, `crates/troupekit-macros/src/agent.rs`, `crates/troupekit-agent/` |
| Tools | `concepts/tools.md` | `crates/troupekit-kit/src/tool.rs`, `crates/troupekit-macros/src/tool.rs`, `crates/troupekit-tools/` |
| Models and providers | `concepts/models.md` | `catalog/providers.toml`, `crates/troupekit-providers/`, `crates/troupekit-kit/src/model.rs` |
| Local models and routing | `concepts/local.md` | `crates/troupekit-local/`, `crates/troupekit-kit/src/local.rs`, `crates/troupekit-kit/src/route.rs` |
| Knowledge | `concepts/knowledge.md` | `crates/troupekit-kit/src/knowledge.rs`, `crates/troupekit-context/`, `crates/troupekit-index/` |
| Memory and storage | `concepts/memory.md` | `crates/troupekit-kit/src/memory.rs`, `crates/troupekit-kit/src/storage.rs`, `crates/troupekit-store-sqlite/` |
| Workflows and graphs | `concepts/workflows.md` | `crates/troupekit-kit/src/graph.rs`, `crates/troupekit-graph/`, `crates/troupekit-host/` |
| Tracing and budgets | `concepts/tracing.md` | `crates/troupekit-trace/`, `crates/troupekit-kit/src/trace.rs`, `crates/troupekit-agent/src/budget.rs` |

## Guides

One page per example of the `troupekit` crate, plus one page for the benchmark examples of the other crates.

| Page | File | Source example |
|---|---|---|
| Hello agent | `guides/hello.md` | `crates/troupekit/examples/hello.rs` |
| Tool agent | `guides/tool-agent.md` | `crates/troupekit/examples/tool_agent/` |
| Support agent | `guides/support-agent.md` | `crates/troupekit/examples/support_agent.rs` |
| Kill and resume a workflow | `guides/kill-resume.md` | `crates/troupekit/examples/kill_resume.rs` |
| Kill and resume a local model call | `guides/kill-model-call.md` | `crates/troupekit/examples/kill_model_call.rs` |
| Hybrid search latency | `guides/hybrid-latency.md` | `crates/troupekit/examples/hybrid_latency.rs` |
| Offline support agent | `guides/support-agent-offline.md` | `crates/troupekit/examples/support_agent_offline.rs` (a stub in 0.3) |
| Benchmarks | `guides/benchmarks.md` | `crates/troupekit-index/examples/`, `crates/troupekit-kit/examples/`, `crates/troupekit-local/examples/`, `crates/troupekit-store-sqlite/examples/`, `crates/troupekit-trace/examples/` |

## Reference

| Page | File | Sources |
|---|---|---|
| Configuration (TOML) | `reference/configuration.md` | `crates/troupekit-agent/src/persona.rs`, `crates/troupekit-graph/src/workflow.rs`, `catalog/*.toml` |
| Feature flags | `reference/feature-flags.md` | `[features]` of every `crates/*/Cargo.toml` |
| Platform support | `reference/platforms.md` | `.github/workflows/ci.yml`, `Cargo.toml` |
| API reference | `reference/api.md` | docs.rs links, one per published crate |

## Project

| Page | File | Sources |
|---|---|---|
| Roadmap | `roadmap.md` | the release sequence |
| FAQ | `faq.md` | the pages above |
| Contributing | `contributing.md` | a link to `CONTRIBUTING.md` |

## PDF

The PDF holds the same pages in this order. Every page links it from the menu bar, next to the print button.
