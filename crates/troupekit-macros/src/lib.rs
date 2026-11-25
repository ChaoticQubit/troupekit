use proc_macro::TokenStream;

mod args;
mod doc;
mod item;
#[cfg(test)]
mod lex;
mod lit;
mod tokens;
mod tool;

/// Turns a fn into a tool value: `#[tool] async fn add(a: i64, b: i64)`
/// makes `add` a unit struct implementing `Tool`, with the doc comment as
/// description and the arguments as the JSON schema. The body stays
/// callable as `add::invoke(..)`. Keys: `name`, `description`, `crate`.
#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut source = tokens::Source::new();
    let attr = source.add(attr);
    let item = source.add(item);
    match tool::parse_tool(&attr, &item) {
        Ok(parsed) => tokens::emit(&tool::expand_tool(&parsed), &source),
        Err(error) => tokens::emit_error(&error, &source),
    }
}
