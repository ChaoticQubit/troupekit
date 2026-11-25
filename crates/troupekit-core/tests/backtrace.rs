use std::process::Command;

const ROLE: &str = "TROUPEKIT_BACKTRACE_ROLE";
const FRAME: &str = "tests/backtrace.rs:";

/// Panics from a frame of its own, so the backtrace must name this file.
/// Input: none. Output: never returns.
/// `inline(never)` keeps the frame even when the optimizer would fold it.
#[inline(never)]
fn panic_in_core_frame() -> ! {
    panic!("backtrace probe");
}

/// The exact test name of `backtrace_child` in whatever binary holds it.
/// Input: none. Output: `backtrace_child`, or `<module>::backtrace_child`
/// when this file is a module of a merged `tests/main.rs`.
fn child_name() -> String {
    match module_path!().split_once("::") {
        Some((_, module)) => format!("{module}::backtrace_child"),
        None => "backtrace_child".to_string(),
    }
}

#[test]
#[ignore = "child process of backtrace_keeps_file_and_line, started by it"]
fn backtrace_child() {
    if std::env::var_os(ROLE).is_some() {
        panic_in_core_frame();
    }
}

#[test]
fn backtrace_keeps_file_and_line() {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "--ignored",
            "--exact",
            &child_name(),
            "--test-threads=1",
            "--nocapture",
        ])
        .env(ROLE, "child")
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !out.status.success(),
        "child did not panic\n{stdout}\n{stderr}"
    );
    assert!(
        stderr.contains("backtrace probe"),
        "child panicked elsewhere\n{stderr}"
    );
    let frame = stderr.lines().map(str::trim_start).any(|line| {
        line.strip_prefix("at ")
            .and_then(|at| at.split_once(FRAME))
            .is_some_and(|(_, rest)| rest.starts_with(|c: char| c.is_ascii_digit()))
    });
    assert!(frame, "no backtrace frame with file and line\n{stderr}");
}
