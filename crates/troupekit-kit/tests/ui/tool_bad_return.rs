extern crate troupekit_kit as troupekit;
use troupekit::{agent, tool};

struct Foo;

/// D.
#[tool]
fn f() -> Foo {
    Foo
}

fn main() {}
