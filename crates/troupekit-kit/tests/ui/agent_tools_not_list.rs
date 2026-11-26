extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// D.
#[agent(model = "fake/m", tools = add)]
fn calc() {}

fn main() {}
