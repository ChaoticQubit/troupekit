extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// Says hi.
#[agent(model = "openai/gpt-6-luna")]
fn hello() {}

fn main() {
    let _ = hello();
}
