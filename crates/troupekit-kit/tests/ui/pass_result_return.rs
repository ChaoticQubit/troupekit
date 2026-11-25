extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

/// Divides.
#[tool]
fn div(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        Err("zero".into())
    } else {
        Ok(a / b)
    }
}

fn main() {
    let _ = div;
}
