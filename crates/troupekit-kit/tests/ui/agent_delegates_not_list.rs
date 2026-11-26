extern crate troupekit_kit as troupekit;
use troupekit::agent;

/// D.
#[agent(model = "fake/m", delegates = writer)]
fn lead() {}

fn main() {}
