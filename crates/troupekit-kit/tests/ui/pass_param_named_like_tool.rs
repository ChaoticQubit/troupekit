extern crate troupekit_kit as troupekit;
use troupekit::tool;

/// Echoes the query.
#[tool]
fn query(query: String) -> String {
    query
}

fn main() {
    let _ = query::invoke(String::new());
}
