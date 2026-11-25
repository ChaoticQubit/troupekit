extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// Greets.
#[tool]
async fn greet(ctx: &troupekit::ToolContext, name: Option<String>) -> String {
    format!("{} {:?}", name.unwrap_or_default(), ctx.run_id)
}

fn main() {
    let _ = greet;
}
