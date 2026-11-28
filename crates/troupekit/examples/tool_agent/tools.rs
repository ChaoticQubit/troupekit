use troupekit::prelude::*;

/// Adds two numbers.
#[tool]
pub async fn add(a: i64, b: i64) -> i64 {
    a + b
}
