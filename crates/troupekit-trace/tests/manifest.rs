const MANIFEST: &str = include_str!("../Cargo.toml");

/// Returns the body lines of the `[dependencies]` table of `manifest`.
/// Input: the manifest text. Output: every non-blank line after the header
/// and before the next table header. Panics when the table is missing.
fn dependency_lines(manifest: &str) -> Vec<&str> {
    let mut lines = manifest.lines().skip_while(|line| line.trim() != "[dependencies]");
    assert_eq!(lines.next().map(str::trim), Some("[dependencies]"), "no [dependencies] table");
    lines
        .take_while(|line| !line.trim_start().starts_with('['))
        .filter(|line| !line.trim().is_empty())
        .collect()
}

#[test]
fn crate_has_no_dependencies() {
    assert_eq!(dependency_lines(MANIFEST), Vec::<&str>::new());
}

#[test]
fn dependency_lines_sees_a_dependency() {
    let text = "[dependencies]\nserde = \"1\"\n\n[dev-dependencies]\nx = \"1\"\n";
    assert_eq!(dependency_lines(text), vec!["serde = \"1\""]);
}
