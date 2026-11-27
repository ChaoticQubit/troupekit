use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::{Duration, Instant};

const START_DEADLINE: Duration = Duration::from_secs(120);
const RESUME_DEADLINE: Duration = Duration::from_secs(600);
const FETCH: &str =
    "sh scripts/fetch-test-model.sh, then set TROUPEKIT_TEST_GGUF to the printed path";

fn runs() -> usize {
    std::env::var("TROUPEKIT_E2E_RUNS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

fn gguf() -> PathBuf {
    let path = std::env::var_os("TROUPEKIT_TEST_GGUF")
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("TROUPEKIT_TEST_GGUF is not set: run {FETCH}"));
    assert!(path.is_file(), "{} is missing: run {FETCH}", path.display());
    path
}

fn example() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let profile = exe.parent().and_then(|deps| deps.parent()).unwrap();
    let path = profile.join("examples").join("kill_model_call");
    assert!(
        path.is_file(),
        "{} is missing: build the example",
        path.display()
    );
    path
}

/// Kills and reaps the child when dropped, so a failed assertion never
/// leaves a process alive. Holds the receiver of its stdout lines.
struct Worker {
    child: Child,
    lines: Receiver<String>,
}

impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

impl Worker {
    fn spawn(args: &[&str]) -> Worker {
        let mut child = Command::new(example())
            .args(args)
            .stdout(Stdio::piped())
            .spawn()
            .unwrap();
        let stdout = child.stdout.take().unwrap();
        let (send, lines) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if send.send(line).is_err() {
                    return;
                }
            }
        });
        Worker { child, lines }
    }

    fn wait_for(&self, prefix: &str, deadline: Duration) -> String {
        let start = Instant::now();
        loop {
            let left = deadline.saturating_sub(start.elapsed());
            assert!(!left.is_zero(), "no `{prefix}` line within {deadline:?}");
            match self.lines.recv_timeout(left) {
                Ok(line) if line.starts_with(prefix) => return line,
                Ok(line) => assert!(!line.starts_with("DONE"), "run finished before the kill"),
                Err(e) => panic!("waiting for `{prefix}`: {e}"),
            }
        }
    }
}

#[test]
fn killed_mid_model_call_resumes_from_checkpoint() {
    let gguf = gguf();
    let gguf = gguf.to_str().unwrap();
    for round in 0..runs() {
        let dir = tempfile::tempdir().unwrap();
        let db = dir.path().join("kill.db");
        let db = db.to_str().unwrap();

        let mut first = Worker::spawn(&["start", db, gguf]);
        let run = first.wait_for("RUN ", START_DEADLINE);
        let run_id = run.strip_prefix("RUN ").unwrap().to_string();
        first.wait_for("GENERATING", START_DEADLINE);
        first.child.kill().unwrap();
        let status = first.child.wait().unwrap();
        assert!(
            !status.success(),
            "round {round}: the killed run exits uncleanly"
        );
        assert!(
            first.lines.try_iter().all(|l| !l.starts_with("DONE")),
            "round {round}: the kill landed after the model call"
        );

        let second = Worker::spawn(&["resume", db, gguf, &run_id]);
        let resumed = second.wait_for("RESUMED ", RESUME_DEADLINE);
        let bytes: usize = resumed.strip_prefix("RESUMED ").unwrap().parse().unwrap();
        assert!(bytes > 0, "round {round}: the resumed run answers");
    }
}
