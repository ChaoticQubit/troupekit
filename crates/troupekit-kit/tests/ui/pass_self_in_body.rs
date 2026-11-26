extern crate troupekit_kit as troupekit;
use troupekit::tool;

/// Counts down to zero.
#[tool]
fn countdown(n: u32) -> u32 {
    if n == 0 {
        0
    } else {
        1 + Self::invoke(n - 1)
    }
}

/// Echoes the query.
#[tool]
fn query(query: String) -> String {
    query
}

fn main() {
    let _ = (countdown::invoke(2), query::invoke(String::new()));
}
