use std::path::Path;
use std::process::{Command, Output};

const SAMPLE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../troupekit-trace/tests/data/sample.jsonl"
);

fn troupekit(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_troupekit"))
        .args(args)
        .output()
        .expect("run binary")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn sample_copy(dir: &Path) -> std::path::PathBuf {
    let path = dir.join("run.jsonl");
    std::fs::copy(SAMPLE, &path).unwrap();
    path
}

#[test]
fn trace_prints_tree_for_sample_file() {
    let out = troupekit(&["trace", SAMPLE, "--no-color"]);
    let stdout = text(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", text(&out.stderr));
    assert!(
        stdout.starts_with("trace demo 00000000000000ab\n"),
        "{stdout}"
    );
    assert!(
        stdout.contains("\nagent.run [agents] ok 1.000ms run=r1\n"),
        "{stdout}"
    );
    assert!(stdout.contains("\n  tool.call [tools] open\n"), "{stdout}");
    assert!(
        !stdout.contains('\u{1b}'),
        "--no-color and a pipe give plain text"
    );
    let only = text(&troupekit(&["trace", SAMPLE, "--only", "tools", "--no-color"]).stdout);
    assert!(
        only.contains("\ntool.call [tools] open\n") && !only.contains("agent.run"),
        "{only}"
    );
}

#[test]
fn trace_html_writes_self_contained_file() {
    let tmp = tempfile::tempdir().unwrap();
    let file = sample_copy(tmp.path());
    let out = troupekit(&["trace", file.to_str().unwrap(), "--html"]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", text(&out.stderr));
    let page = std::fs::read_to_string(tmp.path().join("run.html")).unwrap();
    assert!(page.contains("Content-Security-Policy"));
    assert!(page.contains("agent.step"));
    assert!(!page.contains("https:") && !page.contains("http:"));
    let named = tmp.path().join("named.html");
    let out = troupekit(&[
        "trace",
        "--html",
        named.to_str().unwrap(),
        file.to_str().unwrap(),
    ]);
    assert_eq!(out.status.code(), Some(0), "stderr: {}", text(&out.stderr));
    assert!(named.is_file());
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mode = std::fs::metadata(&named).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}

#[test]
fn trace_refuses_to_overwrite_html() {
    let tmp = tempfile::tempdir().unwrap();
    let file = sample_copy(tmp.path());
    let html = tmp.path().join("run.html");
    std::fs::write(&html, "keep me").unwrap();
    let out = troupekit(&["trace", file.to_str().unwrap(), "--html"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(
        text(&out.stderr).contains("run.html"),
        "{}",
        text(&out.stderr)
    );
    assert_eq!(std::fs::read_to_string(&html).unwrap(), "keep me");
}

#[test]
fn trace_missing_file_exits_1_and_bad_usage_exits_2() {
    let missing = troupekit(&["trace", "/nonexistent/troupekit/trace.jsonl"]);
    assert_eq!(missing.status.code(), Some(1));
    let bad = troupekit(&["trace", SAMPLE, "--only", "bogus"]);
    assert_eq!(bad.status.code(), Some(2));
    let err = text(&bad.stderr);
    assert!(err.contains("usage: troupekit ingest <dir>"), "{err}");
    assert!(
        err.contains("usage: troupekit trace <file> [--html [OUT]] [--only CATS] [--no-color]"),
        "{err}"
    );
}
