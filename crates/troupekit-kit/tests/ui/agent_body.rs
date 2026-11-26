extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// D.
#[agent(model = "fake/m")]
fn calc() {
    let _x = 1;
}

fn main() {}
