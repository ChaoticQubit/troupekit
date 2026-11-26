extern crate troupekit_kit as troupekit;
use troupekit::prelude::*;

/// Greet the user by name.
#[agent(model = "openai/gpt-6-luna")]
fn greeter() {}

#[tokio::main]
async fn main() -> Result<(), KitError> {
    println!("{}", greeter().run("I am Ada").await?);
    Ok(())
}
