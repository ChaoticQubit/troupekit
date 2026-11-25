use proc_macro::TokenStream;

mod agent;
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

/// Turns `fn calc() {}` into a fn returning a configured `Agent`: the doc
/// comment becomes the instructions and the fn name, `_` read as `-`,
/// the agent name. Keys such as `model`, `tools = [add]` and `evals`
/// map to the builder methods of the same name.
#[proc_macro_attribute]
pub fn agent(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut source = tokens::Source::new();
    let attr = source.add(attr);
    let item = source.add(item);
    match agent::parse_agent(&attr, &item) {
        Ok(parsed) => tokens::emit(&agent::expand_agent(&parsed), &source),
        Err(error) => tokens::emit_error(&error, &source),
    }
}
