const HELLO: &str = include_str!("../examples/hello.rs");
const TOOL_MAIN: &str = include_str!("../examples/tool_agent/main.rs");
const TOOL_MOD: &str = include_str!("../examples/tool_agent/tools.rs");
const OWNER_SNIPPET: &str = r#"
/// Adds two numbers.
#[tool]
pub async fn add(a: i64, b: i64) -> i64 {
    a + b
}

mod tools;
use tools::add;
use troupekit::{agent, KitError};

/// Adds numbers with the add tool.
#[agent(model = "openai/gpt-6-luna", tools = [add])]
fn calc() {}

#[tokio::main]
async fn main() -> Result<(), KitError> {
    println!("{}", calc().run("What is 2 + 40?").await?);
    Ok(())
}
"#;

fn user_lines(src: &str) -> usize {
    src.lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .filter(|l| {
            !["use ", "pub use ", "mod ", "extern crate "]
                .iter()
                .any(|p| l.starts_with(p))
        })
        .filter(|l| *l != "#[tokio::main]" && *l != "Ok(())")
        .filter(|l| !l.starts_with("async fn main"))
        .filter(|l| !l.chars().all(|c| ")]};,".contains(c)))
        .count()
}

#[test]
fn count_rule_matches_hand_count() {
    assert_eq!(user_lines(OWNER_SNIPPET), 8);
    assert_eq!(user_lines("use a;\n\n}\n);\n]),\nlet x = 1;\n"), 1);
}

#[test]
fn hello_is_at_most_5_user_lines() {
    let count = user_lines(HELLO);
    assert!(count <= 5, "hello has {count} user lines");
}

#[test]
fn tool_agent_is_at_most_8_user_lines() {
    let count = user_lines(TOOL_MAIN) + user_lines(TOOL_MOD);
    assert!(count <= 8, "tool agent has {count} user lines");
}

#[test]
fn tool_lives_in_its_own_module_file() {
    assert!(TOOL_MOD.contains("#[tool]"));
    assert!(!TOOL_MAIN.contains("#[tool]"));
    assert!(TOOL_MAIN.lines().any(|l| l.trim() == "mod tools;"));
}

#[test]
fn examples_have_no_expect_or_unwrap() {
    for src in [HELLO, TOOL_MAIN, TOOL_MOD] {
        assert!(!src.contains(".expect(") && !src.contains(".unwrap("));
    }
}
