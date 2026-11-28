---
title: Quick start
description: Build a one-line agent, run it against a hosted model, then give it a tool.
---

# Quick start

## Requirements

- Rust 1.89 or newer.
- An API key for a catalog vendor. The key is read from `<VENDOR>_API_KEY`, so `OPENAI_API_KEY` for an `openai/...` model.

Source: `crates/troupekit/Cargo.toml`, `crates/troupekit-kit/src/model.rs`

## Create the project

```sh
cargo new hello-agent
cd hello-agent
```

```toml
[dependencies]
troupekit = { git = "https://github.com/ChaoticQubit/troupekit", tag = "v0.3.0" }
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

Troupekit is not on crates.io yet, so 0.3 is used as a git dependency on the `v0.3.0` tag.

## Write the agent

The doc comment becomes the instructions and the fn name becomes the agent name.

```rust
use troupekit::prelude::*;

/// Greet the user by name.
#[agent(model = "openai/gpt-6-luna")]
fn greeter() {}

#[tokio::main]
async fn main() -> Result<(), KitError> {
    println!("{}", greeter().run("I am Ada").await?);
    Ok(())
}
```

Source: `crates/troupekit/examples/hello.rs`

## Run it

```sh
export OPENAI_API_KEY="sk-..."
cargo run
```

`run` builds the agent, checks every setting, makes the model call and returns the final answer as a `String`.

## Give it a tool

`#[tool]` turns a fn into a tool: the doc comment is the description and the arguments are the JSON Schema the model sees. Put it in `src/tools.rs`:

```rust
use troupekit::prelude::*;

/// Adds two numbers.
#[tool]
pub async fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

Source: `crates/troupekit/examples/tool_agent/tools.rs`

Then list it in the agent's `tools` key:

```rust
mod tools;
use tools::add;
use troupekit::prelude::*;

/// Adds numbers with the add tool.
#[agent(model = "openai/gpt-6-luna", tools = [add])]
fn calc() {}

#[tokio::main]
async fn main() -> Result<(), KitError> {
    println!("{}", calc().run("What is 2 + 40?").await?);
    Ok(())
}
```

Source: `crates/troupekit/examples/tool_agent/main.rs`

The model asks for `add`, the agent calls it, and the model answers with the result.

## Run the repository examples

Both programs ship as examples:

```sh
cargo run -p troupekit --example hello
cargo run -p troupekit --example tool_agent
```

## Next

- [Agents and personas](concepts/agents.md): every agent setting.
- [Tools](concepts/tools.md): tool fns, the `tool(..)` form, web and HTTP tools.
- [Guides](guides/hello.md): one page per example.
