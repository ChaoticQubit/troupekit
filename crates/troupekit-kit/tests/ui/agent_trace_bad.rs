extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// D.
#[agent(model = "fake/m", trace = everything)]
fn calc() {}

fn main() {}
