extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// Echoes.
#[tool]
fn echo(r#type: String) -> String {
    r#type
}

fn main() {
    let _ = echo;
}
