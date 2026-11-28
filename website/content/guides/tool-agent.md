---
title: Tool agent
description: An agent that calls a #[tool] fn to answer.
---

# Tool agent

The `tool_agent` example gives an agent one tool, `add`, and asks it a sum.

```rust
use troupekit::prelude::*;

/// Adds two numbers.
#[tool]
pub async fn add(a: i64, b: i64) -> i64 {
    a + b
}
```

Source: `crates/troupekit/examples/tool_agent/tools.rs`

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

## Run it

```sh
export OPENAI_API_KEY="sk-..."
cargo run -p troupekit --example tool_agent
```

## What happens

1. The model sees the `add` tool with the schema `{a: integer, b: integer}`, built from the fn arguments.
2. The model asks for `add` with `a = 2`, `b = 40`. The registry checks the arguments against the schema and calls the fn.
3. The result goes back to the model, which answers with the sum.

The kit test `tool_agent_runs_lab2_flow` runs the same flow against a scripted model and expects `The sum is 42`. The test `run_with_returns_usage_calls_and_messages` counts two model calls: one that asks for the tool, one that answers.

Source: `crates/troupekit-kit/tests/agent.rs`

See [Tools](../concepts/tools.md) for the `tool(..)` form and the web and HTTP tools.
