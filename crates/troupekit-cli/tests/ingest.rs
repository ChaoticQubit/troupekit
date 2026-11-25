use std::process::Command;

#[test]
fn no_arguments_prints_usage_and_exits_2() {
    let out = Command::new(env!("CARGO_BIN_EXE_troupekit")).output().expect("run binary");
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("usage: troupekit ingest <dir>"), "stderr: {err}");
}
