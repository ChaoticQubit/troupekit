extern crate troupekit_kit as troupekit;
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
