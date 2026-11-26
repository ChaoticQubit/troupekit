use troupekit::tool;
use troupekit_tools::ToolContext;

/// Adds two numbers.
#[tool]
pub async fn add(a: i64, b: i64) -> i64 {
    a + b
}

/// Shouts the text.
#[tool]
pub fn shout(text: String) -> String {
    text.to_uppercase()
}

/// Reports whether the call has a run id.
#[tool]
pub async fn has_run(ctx: &ToolContext, label: Option<String>) -> String {
    format!("{}:{}", ctx.run_id.is_some(), label.unwrap_or_default())
}

/// Divides a by b.
#[tool]
pub async fn div(a: i64, b: i64) -> Result<i64, String> {
    if b == 0 {
        Err("division by zero".into())
    } else {
        Ok(a / b)
    }
}

/// Returns a small number.
#[tool]
pub fn small(n: u8) -> u8 {
    n
}

/// Echoes the type field.
#[tool]
pub fn echo_type(r#type: String) -> String {
    r#type
}
