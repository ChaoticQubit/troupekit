# Troupekit

Troupekit is a Rust framework for building LLM agents from persona TOML files: providers, tools, memory, RAG and durable graph workflows.

| Crate | Status | What |
|---|---|---|
| `troupekit-core` | available | Messages, secrets, errors, events, ids, clock, storage traits and in-memory stores |
| `troupekit-providers` | available | Anthropic, OpenAI, Gemini and OpenAI-compatible wire protocols, provider catalog with prices, injected transports, retry, fallback and cost-meter middleware |
| `troupekit-tools` | available | Tool trait, validating registry, typed tools, SSRF-guarded web fetch, web search over Brave and SearXNG, declarative HTTP tools |
| `troupekit-context` | available | Memory scopes, profile facts, chunking, embeddings, hybrid retrieval and context blocks |
| `troupekit-agent` | available | Persona TOML agents: layered loader, checkpointable run state machine, concurrent allowed tools, typed output with one repair round, evals and LLM judge, layered budgets, events |
| `troupekit-graph` | available | TOML workflows of agent, tool, human, router, join and end nodes over typed shared state; checkpointable graph run with parallel branches, human pauses, graph budgets and retries |
| `troupekit-plugins` | planned | Plugin loading |
| `troupekit-host` | available | Leased job queue, event log deduped on run and seq, runner with worker mode and serverless step mode, crash-safe resume |

## License

Licensed under either of MIT or Apache-2.0, at your option.
