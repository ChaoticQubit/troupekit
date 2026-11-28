---
title: Hello agent
description: The smallest troupekit program, one agent from a macro and one hosted model call.
---

# Hello agent

The `hello` example is one agent defined by `#[agent]` and run once.

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
cargo run -p troupekit --example hello
```

The example needs the `kit` feature, which is on by default.

Source: `crates/troupekit/Cargo.toml` `[[example]]`

## What happens

1. `greeter()` returns an `Agent` named `greeter` whose instructions are the doc comment.
2. `run` builds it: the selector `openai/gpt-6-luna` resolves against the catalog and the key is read from `OPENAI_API_KEY`.
3. One model call answers, and `run` returns the text.

## Change the model

Any catalog selector works, such as `anthropic/claude-haiku-5-5` with `ANTHROPIC_API_KEY`. See [Models and providers](../concepts/models.md).
