extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

mod tools {
    use troupekit::tool;

    /// Adds.
    #[tool]
    pub async fn add(a: i64, b: i64) -> i64 {
        a + b
    }
}

/// Uses add.
#[agent(model = "openai/gpt-6-luna", tools = [tools::add])]
fn calc() {}

fn main() {
    let _ = calc();
}
