extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

struct NotATool;

/// D.
#[agent(model = "fake/m", tools = [NotATool])]
fn calc() {}

fn main() {}
