---
title: Tools
description: Tool fns, the tool(..) form, the registry, web fetch, web search and declarative HTTP tools.
---

# Tools

A tool is anything that implements the `Tool` trait. An agent or graph accepts a `#[tool]` fn, a `tool(..)` value or an `Arc<dyn Tool>`. The registry checks every call's arguments against the tool's JSON Schema before the tool runs.

Source: `crates/troupekit-tools/src/lib.rs`, `crates/troupekit-kit/src/tool.rs`

## `#[tool]` fns

`#[tool]` turns a fn into a unit struct that implements `Tool`. The doc comment is the description and the arguments are the schema. The body stays callable as `add::invoke(..)`.

```rust
use troupekit::prelude::*;

/// Adds two numbers.
#[tool]
pub async fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

Source: `crates/troupekit/examples/tool_agent/tools.rs`

The fn may be sync or async. Its keys are `name`, `description` and `crate`.

Source: `crates/troupekit-macros/src/lib.rs`

## The `tool(..)` form

`tool(name, description, f)` builds a tool without a macro. `f` takes the parsed arguments, any `Deserialize + JsonSchema` type, and returns any `IntoToolOutput` value. An `Err` reaches the model as error output, not as a failed run:

```rust
#[derive(Deserialize, schemars::JsonSchema)]
struct AddArgs {
    a: i64,
    b: i64,
}

#[tokio::test]
async fn tool_fn_form_err_reaches_model_as_error_output() {
    let mut registry = ToolRegistry::new();
    let div = tool("div", "Divides a by b", |x: AddArgs| async move {
        if x.b == 0 {
            Err("division by zero")
        } else {
            Ok(x.a / x.b)
        }
    });
    registry.register(Arc::new(div)).unwrap();
    let out = registry
        .call("div", json!({"a": 1, "b": 0}), &ToolContext::default())
        .await
        .unwrap();
    assert_eq!(out, ToolOutput::error("division by zero"));
}
```

Source: `crates/troupekit-kit/tests/output.rs`

## Adding tools to an agent

- `.tool(t)` adds one `#[tool]` fn or `tool(..)` value.
- `.tool_arc(t)` adds an `Arc<dyn Tool>`.
- `.tools(iter)` and `.toolkit(kit)` add many, in order.

A second tool under a taken name fails at `build`. When a model asks for several tools in one turn, the agent runs those calls concurrently.

Source: `crates/troupekit-kit/src/agent.rs`, `crates/troupekit-agent/src/run.rs`

## Tool context

Every call gets a `ToolContext`. `run_id` names the run that made the call. `idempotency_key` is the same for a replayed call and new for every other call, so a tool with side effects can dedupe after a resume.

Source: `crates/troupekit-tools/src/tool.rs`

## Web fetch

`web_fetch` GETs one page through a `GuardedHttp` client, keeps the visible text of HTML, caps it at `max_chars` characters and returns it fenced as untrusted, with the final URL as its source. Only `text/html` and `text/plain` are accepted. The guarded client refuses private addresses, so a model cannot reach your network through it.

Source: `crates/troupekit-tools/src/fetch.rs`

## Web search

`web_search` tries search instances in order and moves to the next only when one fails. Two adapters ship: Brave Search and SearXNG. The default instances live in `catalog/search.toml`:

```toml
[[instance]]
name = "brave"
adapter = "brave"
base_url = "https://api.search.brave.com"
key_env = "BRAVE_SEARCH_API_KEY"

[[instance]]
name = "searxng-local"
adapter = "searxng"
base_url = "http://127.0.0.1:8888"
allow_private = true
```

Source: `catalog/search.toml`, `crates/troupekit-tools/src/search.rs`

An instance whose key is missing is skipped at load. Results are third-party text and are fenced before the model sees them.

## Declarative HTTP tools

`http_toolkit_from_toml` loads HTTP endpoints from TOML. Each `[[tool]]` has a `name`, a `description`, a `method`, a `url` with `{arg}` path placeholders, optional `headers` and a JSON `schema`. Header values may hold `${secret:NAME}`, resolved when the file loads:

```toml
[[tool]]
name = "forecast"
description = "Weather forecast for a city"
method = "GET"
url = "https://api.example.com/v1/{city}/forecast"
headers = { Authorization = "Bearer ${secret:WEATHER_KEY}", Accept = "application/json" }
schema = { type = "object", properties = { city = { type = "string" }, days = { type = "integer" } }, required = ["city"] }
```

Source: `crates/troupekit-tools/src/declarative.rs`, unit tests

Path arguments fill the placeholders; the rest go to the query string or a JSON body. Pass a `GuardedHttp`, never an unguarded transport.
