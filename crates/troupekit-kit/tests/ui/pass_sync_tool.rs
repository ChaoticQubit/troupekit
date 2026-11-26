extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// Shouts.
#[tool]
pub fn shout(text: String) -> String {
    text.to_uppercase()
}

fn main() {
    let _ = shout::invoke("a".into());
}
