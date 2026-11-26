extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// D.
#[agent(model = "fake/m", evals = [check])]
fn calc() {}

fn main() {}
