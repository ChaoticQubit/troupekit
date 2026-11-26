use crate::common::*;
use std::path::PathBuf;
use troupekit::KitError;

fn bend_file(dir: &tempfile::TempDir, name: &str, bytes: usize) -> PathBuf {
    let path = dir.path().join(name);
    std::fs::write(&path, "x".repeat(bytes)).unwrap();
    path
}

async fn config_error(agent: troupekit::Agent) -> String {
    match agent.build().await {
        Err(e @ KitError::Config { .. }) => e.to_string(),
        Err(other) => panic!("expected Config, got {other}"),
        Ok(_) => panic!("expected Config, got a built agent"),
    }
}

#[tokio::test]
async fn auto_laws_defaults_to_true() {
    assert!(greeter().build().await.unwrap().laws().auto());
    let off = greeter().auto_laws(false).build().await.unwrap();
    assert!(!off.laws().auto());
}

#[tokio::test]
async fn laws_path_must_be_bend_file_that_exists() {
    let dir = tempfile::tempdir().unwrap();
    let txt = bend_file(&dir, "laws.txt", 4);
    let text = config_error(greeter().laws(&txt)).await;
    assert!(text.contains("must be a .bend file"), "{text}");
    let missing = dir.path().join("absent.bend");
    let text = config_error(greeter().laws(&missing)).await;
    assert!(text.contains("does not exist"), "{text}");
    let folder = dir.path().join("dir.bend");
    std::fs::create_dir(&folder).unwrap();
    let text = config_error(greeter().laws(&folder)).await;
    assert!(text.contains("is not a regular file"), "{text}");
    let text = config_error(greeter().laws("")).await;
    assert!(text.contains("is empty"), "{text}");
}

#[tokio::test]
async fn laws_path_with_dotdot_is_refused() {
    let text = config_error(greeter().laws("rules/../../etc/x.bend")).await;
    assert!(text.starts_with("agent `greeter`: "), "{text}");
    assert!(text.contains("must not contain `..`"), "{text}");
}

#[tokio::test]
async fn laws_file_over_1_mib_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let big = bend_file(&dir, "big.bend", 1024 * 1024 + 1);
    let text = config_error(greeter().laws(&big)).await;
    assert!(text.contains("over 1 MiB"), "{text}");
    let exact = bend_file(&dir, "exact.bend", 1024 * 1024);
    assert!(greeter().laws(&exact).build().await.is_ok());
    let binary = dir.path().join("bin.bend");
    std::fs::write(&binary, [0xff, 0xfe]).unwrap();
    let text = config_error(greeter().laws(&binary)).await;
    assert!(text.contains("not valid UTF-8"), "{text}");
}

#[tokio::test]
async fn inline_law_blank_or_with_nul_is_refused() {
    let text = config_error(greeter().law("  ")).await;
    assert!(text.contains("inline law 1 is blank"), "{text}");
    let text = config_error(greeter().law("ok").law("a\0b")).await;
    assert!(text.contains("inline law 2 contains a NUL byte"), "{text}");
    let text = config_error(greeter().law("x".repeat(64 * 1024 + 1))).await;
    assert!(text.contains("inline law 1 is over 64 KiB"), "{text}");
}

#[tokio::test]
async fn more_than_64_laws_is_refused() {
    let mut agent = greeter();
    for i in 0..65 {
        agent = agent.law(format!("law {i}"));
    }
    let text = config_error(agent).await;
    assert!(text.contains("at most 64 laws, got 65"), "{text}");
}

#[tokio::test]
async fn laws_are_stored_and_run_ignores_them() {
    let dir = tempfile::tempdir().unwrap();
    let file = bend_file(&dir, "rules.bend", 10);
    let built = greeter()
        .laws(&file)
        .law("def main = 1")
        .build()
        .await
        .unwrap();
    assert_eq!(built.laws().paths(), [file.clone()]);
    assert_eq!(built.laws().inline(), ["def main = 1".to_string()]);
    assert_eq!(built.run("x").await.unwrap(), "Hello! You said: x");
}
