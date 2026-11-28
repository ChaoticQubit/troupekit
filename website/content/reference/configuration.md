---
title: Configuration (TOML)
description: Every key of persona files, workflow files, route policies and the provider, search and local model catalogs.
---

# Configuration (TOML)

Every troupekit TOML file refuses unknown keys, so a typo fails at load with the file and key named.

Source: `#[serde(deny_unknown_fields)]` in `crates/troupekit-agent/src/persona.rs`, `crates/troupekit-graph/src/workflow.rs` and `crates/troupekit-providers/src/route/policy.rs`

## Persona files

```toml
name = "coach"
display_name = "Coach"
role = "Plans training weeks."
goal = "A safe weekly plan."
instructions = "Plan the week."
context = "The owner trains at home."
model = "anthropic/claude-opus-5-5"
tools = ["web_search"]
guardrails = ["No medical advice."]
evals = ["min_items(actions, 1)", "judge('Plan is safe', 7)"]
max_turns = 6
budget = { call_usd = 0.05, run_usd = 0.5 }

[output]
actions = "list<{name, minutes: int}>"
```

Source: `crates/troupekit-agent/src/persona.rs`, unit test constant `FULL`

| Key | Rule | Default |
|---|---|---|
| `name` | 1 to 64 of `[a-z0-9-]`, not starting with `-` | required |
| `display_name` | shown in the system prompt | required |
| `role` | role line | required |
| `goal` | goal line | required |
| `instructions` | the instructions text | one of `instructions` or `instructions_file` is required |
| `instructions_file` | a relative path inside the persona folder | see `instructions` |
| `context` | extra context for the system prompt | empty |
| `model` | `vendor/model`, or a role alias of `[a-z0-9_-]` such as `default` | required |
| `tools` | tool names, 1 to 64 of `[A-Za-z0-9_-]`, no repeats | none |
| `guardrails` | rules added to the system prompt, not blank | none |
| `evals` | eval calls such as `min_items(sources, 1)` or `judge('Plan is safe', 7)` | none |
| `[output]` | typed output fields, see [Agents and personas](../concepts/agents.md) | plain text |
| `max_turns` | 2 to 50 | 8 |
| `max_tokens` | 1 to 200 000 | 4096 |
| `budget` | `{ call_usd, run_usd }` in dollars | no budget |
| `trace`, `trace_content`, `trace_file` | see [Tracing and budgets](../concepts/tracing.md) | off |

Source: `crates/troupekit-agent/src/persona.rs`

## Workflow files

See [Workflows and graphs](../concepts/workflows.md) for the full example. Top-level keys: `name`, `start`, `max_steps` (1 to 10 000, default 200), `max_retries` (0 to 50, default 5), `[state]`, `[[nodes]]`, `[[edges]]`, `trace`, `trace_content` and `trace_file`.

| Node key | Used by |
|---|---|
| `id`, `kind` | every node |
| `persona`, `prompt` | `agent` |
| `tool` | `tool` |
| `ask`, `prompt` | `human` |
| `condition` | `router` |
| `reads`, `writes` | `agent`, `tool`, `human` |

Edges have `from`, `to` and, after a router, `when`.

Source: `crates/troupekit-graph/src/workflow.rs`

## Route policies

See [Local models and routing](../concepts/local.md) for every key: `name`, `default`, `order`, `[[target]]`, `[network]`, `[budget]`, `[device]` and `[fallback]`.

## Provider catalog

`catalog/providers.toml` has three kinds of rows.

```toml
[[vendor]]
id = "anthropic"
name = "Anthropic"
protocol = "anthropic"
base_url = "https://api.anthropic.com"
auth = "x-api-key"
price_source = "https://platform.claude.com/docs/en/about-claude/pricing"
quirks = { temperature = false }
search_usd_micros = 10000
```

Source: `catalog/providers.toml`

- `[[vendor]]`: `id`, `name`, `protocol` (`anthropic`, `openai`, `gemini` or `openai_compatible`), `base_url`, `auth`, `price_source`, `quirks`, `search_usd_micros` and `open_models`.
- `[[vendor.model]]`: `id`, `context_window`, `input_usd_micros_per_mtok`, `output_usd_micros_per_mtok`, `price_verified` and `long_context`.
- `[[embedding]]`: `vendor`, `id`, `dims`, `input_usd_micros_per_mtok` and `price_verified`.

Source: `catalog/providers.toml`, `crates/troupekit-providers/src/catalog.rs`

## Search catalog

`catalog/search.toml` lists `[[instance]]` rows with `name`, `adapter` (`brave` or `searxng`), `base_url`, and either `key_env` or `allow_private`. See [Tools](../concepts/tools.md).

Source: `catalog/search.toml`

## Local model catalog

`crates/troupekit-local/catalog/local.toml` lists `[[model]]` rows with `id`, `url`, `sha256`, `bytes`, `ram_bytes`, `context`, `class`, `licence`, `embeddings` and `recommended`. See [Local models and routing](../concepts/local.md).

Source: `crates/troupekit-local/catalog/local.toml`
