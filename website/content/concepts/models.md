---
title: Models and providers
description: Model selectors, the built-in provider catalog, API keys, prices, retries and fallback.
---

# Models and providers

An agent names its model with a selector, `vendor/model`, such as `openai/gpt-6-luna`. The selector resolves against the provider catalog at `build`, which gives the wire protocol, the base URL, the auth header and the price.

Source: `crates/troupekit-kit/src/agent.rs`, `crates/troupekit-providers/src/catalog.rs`

## The built-in catalog

| Vendor id | Protocol | Auth |
|---|---|---|
| `anthropic` | `anthropic` | `x-api-key` |
| `openai` | `openai` | `bearer` |
| `google` | `gemini` | `x-goog-api-key` |
| `openrouter` | `openai_compatible` | `bearer` |
| `groq` | `openai_compatible` | `bearer` |
| `deepseek` | `openai_compatible` | `bearer` |
| `mistral` | `openai_compatible` | `bearer` |
| `ollama` | `openai_compatible` | `none` |
| `lmstudio` | `openai_compatible` | `none` |

Source: `catalog/providers.toml`

Each vendor lists its models with a context window and a price in millionths of a dollar per million tokens. `ollama` and `lmstudio` are local servers marked `open_models`: they also accept model ids the catalog does not list, at a zero price.

A model row looks like this:

```toml
[[vendor.model]]
id = "gpt-6-luna"
context_window = 1050000
input_usd_micros_per_mtok = 100000
output_usd_micros_per_mtok = 500000
price_verified = true
long_context = { above_input_tokens = 272000, input_usd_micros_per_mtok = 200000, output_usd_micros_per_mtok = 750000 }
```

Source: `catalog/providers.toml`

The catalog also has `[[embedding]]` rows for the embedding models used by [Knowledge](knowledge.md): `openai/text-embedding-3-small`, `openai/text-embedding-3-large` and `google/gemini-embedding-001`.

## API keys

The key is read from `<VENDOR>_API_KEY` at `build`, with the vendor id upper-cased and `-` read as `_`: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`. Vendors with `auth = "none"` need no key.

- `.api_key(secret)` sets the key directly and beats any environment variable.
- `.api_key_env(name)` reads another variable.

The key never appears in `Debug` output or an error.

Source: `crates/troupekit-kit/src/model.rs`, `crates/troupekit-kit/src/agent.rs`

## Your own catalog

`.catalog(catalog)` replaces the built-in one. `Catalog::load(text)` parses catalog TOML in the same shape as `catalog/providers.toml`. Selectors and prices then resolve against it.

Source: `crates/troupekit-kit/src/agent.rs`, `crates/troupekit-providers/src/catalog.rs`

## Transports and test models

- `.transport(t)` sets the HTTP transport for catalog models. The default is a reqwest transport, behind the `reqwest` feature.
- `.model_client(m)` serves the selector from any `Arc<dyn Model>` instead of connecting one. It needs no key and no network, which is how the tests and the local model examples run.
- `.price(p)` overrides the price replies are charged at.

Source: `crates/troupekit-kit/src/agent.rs`

## Retries and fallback

`troupekit-providers` ships three middleware models:

- `RetryModel` retries transient failures and honours server retry delays. Defaults: 3 tries, a 500 ms base wait, a 30 s cap. A stream is retried only while no delta has reached the caller.
- `FallbackModel` tries an ordered list of models and returns the first success.
- `MeteredModel` fills in the cost of every reply from catalog prices. `Catalog::connect` wraps each model in one.

The agent's `.retries(n)` sets how often a transient failure is retried per run, 3 by default.

Source: `crates/troupekit-providers/src/middleware.rs`, `crates/troupekit-kit/src/agent.rs`

To switch between a local and a cloud model by policy, see [Local models and routing](local.md).
