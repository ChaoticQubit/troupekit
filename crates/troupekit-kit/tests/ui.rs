use std::path::{Path, PathBuf};
use std::process::Command;

const MAX_CASES: usize = 128;

struct Case {
    name: String,
    source: String,
    expected: Option<String>,
}

fn manifest_dir() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
}

fn cases() -> Vec<Case> {
    let dir = manifest_dir().join("tests/ui");
    let mut names: Vec<String> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".rs").map(str::to_string)
        })
        .collect();
    names.sort();
    assert!(
        !names.is_empty() && names.len() <= MAX_CASES,
        "{}",
        names.len()
    );
    names
        .into_iter()
        .map(|name| Case {
            source: std::fs::read_to_string(dir.join(format!("{name}.rs"))).unwrap(),
            expected: std::fs::read_to_string(dir.join(format!("{name}.expected"))).ok(),
            name,
        })
        .collect()
}

fn scratch(kind: &str, cases: &[&Case]) -> PathBuf {
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("kit-ui");
    let root = base.join(kind);
    let _ = std::fs::remove_dir_all(root.join("src"));
    std::fs::create_dir_all(root.join("src/bin")).unwrap();
    let kit = manifest_dir().display().to_string().replace('\\', "/");
    std::fs::write(
        root.join("Cargo.toml"),
        format!(
            "[package]\nname = \"kit-ui-{kind}\"\nversion = \"0.0.0\"\nedition = \"2021\"\npublish = false\n\n\
             [workspace]\n\n[dependencies]\ntroupekit-kit = {{ path = \"{kit}\", default-features = false }}\n"
        ),
    )
    .unwrap();
    std::fs::copy(
        manifest_dir().join("../../Cargo.lock"),
        root.join("Cargo.lock"),
    )
    .unwrap();
    for case in cases {
        std::fs::write(root.join(format!("src/bin/{}.rs", case.name)), &case.source).unwrap();
    }
    root
}

fn check(root: &Path, name: &str) -> (bool, String) {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".into());
    let target = Path::new(env!("CARGO_TARGET_TMPDIR")).join("kit-ui/target");
    let output = Command::new(cargo)
        .args([
            "check",
            "--offline",
            "--message-format",
            "short",
            "--bin",
            name,
        ])
        .arg("--manifest-path")
        .arg(root.join("Cargo.toml"))
        .env("CARGO_TARGET_DIR", target)
        .output()
        .unwrap();
    (
        output.status.success(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
    )
}

#[test]
fn ui_cases_match_expected_errors() {
    let all = cases();
    let failing: Vec<&Case> = all
        .iter()
        .filter(|c| !c.name.starts_with("pass_"))
        .collect();
    for case in &failing {
        assert!(
            case.expected.is_some(),
            "{} has no .expected file",
            case.name
        );
    }
    let root = scratch("fail", &failing);
    let mut problems = Vec::new();
    for case in &failing {
        let (ok, stderr) = check(&root, &case.name);
        if ok {
            problems.push(format!("{}: compiled, expected an error", case.name));
            continue;
        }
        let expected = case.expected.as_deref().unwrap_or_default();
        for line in expected.lines().filter(|l| !l.trim().is_empty()) {
            if !stderr.contains(line) {
                problems.push(format!("{}: missing `{line}` in:\n{stderr}", case.name));
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n\n"));
}

#[test]
fn pass_cases_compile() {
    let all = cases();
    let passing: Vec<&Case> = all.iter().filter(|c| c.name.starts_with("pass_")).collect();
    assert!(!passing.is_empty());
    let root = scratch("pass", &passing);
    let mut problems = Vec::new();
    for case in &passing {
        let (ok, stderr) = check(&root, &case.name);
        if !ok {
            problems.push(format!("{}:\n{stderr}", case.name));
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n\n"));
}
