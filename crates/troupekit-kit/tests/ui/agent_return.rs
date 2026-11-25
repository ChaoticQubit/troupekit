extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// D.
#[agent(model = "fake/m")]
fn calc() -> u8 {
    0
}

fn main() {}
