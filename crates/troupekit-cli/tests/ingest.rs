use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../troupekit/examples/support/docs");
const UNSET_KEY_VAR: &str = "TROUPEKIT_CLI_TEST_UNSET_KEY";
const SET_KEY_VAR: &str = "TROUPEKIT_CLI_TEST_SET_KEY";
const UNREACHABLE: &str = "[[vendor]]\nid = \"v\"\nname = \"V\"\nprotocol = \"openai_compatible\"\nbase_url = \"https://127.0.0.1:1/v1\"\nauth = \"bearer\"\n\n[[embedding]]\nvendor = \"v\"\nid = \"e\"\ndims = 8\nmax_batch = 8\ninput_usd_micros_per_mtok = 0\nprice_verified = false\n";

fn troupekit(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_troupekit")).args(args).env_remove(UNSET_KEY_VAR).output().expect("run binary")
}

fn text(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

fn corpus_copy(tmp: &Path) -> (PathBuf, usize) {
    let dir = tmp.join("docs");
    std::fs::create_dir(&dir).unwrap();
    let mut n = 0;
    for entry in std::fs::read_dir(CORPUS).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "md") {
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
            n += 1;
        }
    }
    assert!(n >= 20, "corpus has {n} files");
    (dir, n)
}

fn ingest(dir: &Path, db: &Path) -> Output {
    troupekit(&["ingest", dir.to_str().unwrap(), "--db", db.to_str().unwrap(), "--embedder", "hashing:256"])
}

#[test]
fn no_arguments_prints_usage_and_exits_2() {
    let out = troupekit(&[]);
    assert_eq!(out.status.code(), Some(2));
    let err = text(&out.stderr);
    assert!(err.contains("usage: troupekit ingest <dir>"), "stderr: {err}");
}

#[test]
fn ingest_folder_reports_counts() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, n) = corpus_copy(tmp.path());
    let out = ingest(&dir, &tmp.path().join("kb.db"));
    let stdout = text(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "stdout: {stdout} stderr: {}", text(&out.stderr));
    assert!(stdout.contains(&format!("{n} added, 0 replaced, 0 unchanged, 0 failed")), "{stdout}");
    assert_eq!(stdout.lines().count(), 1, "{stdout}");
    assert!(tmp.path().join("kb.db").is_file());
}

#[test]
fn ingest_twice_skips_unchanged_files() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, n) = corpus_copy(tmp.path());
    let db = tmp.path().join("kb.db");
    assert_eq!(ingest(&dir, &db).status.code(), Some(0));
    let out = ingest(&dir, &db);
    let stdout = text(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains(&format!("0 added, 0 replaced, {n} unchanged, 0 failed, 0 chunks")), "{stdout}");
    std::fs::write(dir.join("shipping-times.md"), "# Shipping times\n\nOrders now ship within one working day.\n").unwrap();
    let out = ingest(&dir, &db);
    let stdout = text(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(stdout.contains(&format!("0 added, 1 replaced, {} unchanged, 0 failed", n - 1)), "{stdout}");
}

#[test]
fn failed_files_are_listed_and_exit_1() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, n) = corpus_copy(tmp.path());
    let catalog = tmp.path().join("catalog.toml");
    std::fs::write(&catalog, UNREACHABLE).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_troupekit"))
        .args(["ingest", dir.to_str().unwrap(), "--db", tmp.path().join("kb.db").to_str().unwrap(), "--embedder", "v/e", "--catalog", catalog.to_str().unwrap(), "--key-env", SET_KEY_VAR])
        .env(SET_KEY_VAR, "sk-unused")
        .output()
        .expect("run binary");
    let (stdout, stderr) = (text(&out.stdout), text(&out.stderr));
    assert_eq!(out.status.code(), Some(1), "{stdout} {stderr}");
    assert!(stdout.contains(&format!("0 added, 0 replaced, 0 unchanged, {n} failed")), "{stdout} {stderr}");
    assert_eq!(stderr.lines().filter(|l| l.starts_with("failed: ") && l.contains(".md")).count(), n, "{stderr}");
}

#[test]
fn non_utf8_file_exits_1_and_names_the_file() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _) = corpus_copy(tmp.path());
    std::fs::write(dir.join("broken.md"), [0xff, 0xfe, 0x00, 0x41]).unwrap();
    let out = ingest(&dir, &tmp.path().join("kb.db"));
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("broken.md") && stderr.contains("UTF-8"), "{stderr}");
}

#[test]
fn missing_key_env_is_a_clear_error() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _) = corpus_copy(tmp.path());
    let out = troupekit(&["ingest", dir.to_str().unwrap(), "--db", tmp.path().join("kb.db").to_str().unwrap(), "--embedder", "openai/text-embedding-3-small", "--key-env", UNSET_KEY_VAR]);
    let stderr = text(&out.stderr);
    assert_eq!(out.status.code(), Some(1), "{stderr}");
    assert!(stderr.contains("needs an API key") && stderr.contains(UNSET_KEY_VAR), "{stderr}");
    assert!(text(&out.stdout).is_empty());
    assert!(!tmp.path().join("kb.db").exists(), "no store before the embedder is ready");
}

#[test]
fn key_never_appears_in_output() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _) = corpus_copy(tmp.path());
    let catalog = tmp.path().join("catalog.toml");
    std::fs::write(&catalog, UNREACHABLE).unwrap();
    let secret = "sk-cli-secret-789";
    let out = Command::new(env!("CARGO_BIN_EXE_troupekit"))
        .args(["ingest", dir.to_str().unwrap(), "--db", tmp.path().join("kb.db").to_str().unwrap(), "--embedder", "v/e", "--catalog", catalog.to_str().unwrap(), "--key-env", SET_KEY_VAR])
        .env(SET_KEY_VAR, secret)
        .output()
        .expect("run binary");
    let all = format!("{}{}", text(&out.stdout), text(&out.stderr));
    assert_eq!(out.status.code(), Some(1), "{all}");
    assert!(!all.contains(secret) && !all.contains("sk-cli"), "{all}");
}

#[test]
fn unknown_selector_and_bad_catalog_exit_1() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _) = corpus_copy(tmp.path());
    let db = tmp.path().join("kb.db");
    let out = troupekit(&["ingest", dir.to_str().unwrap(), "--db", db.to_str().unwrap(), "--embedder", "nobody/none"]);
    assert_eq!(out.status.code(), Some(1), "{}", text(&out.stderr));
    assert!(text(&out.stderr).contains("unknown vendor"), "{}", text(&out.stderr));
    let missing = tmp.path().join("absent.toml");
    let out = troupekit(&["ingest", dir.to_str().unwrap(), "--db", db.to_str().unwrap(), "--embedder", "v/e", "--catalog", missing.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("absent.toml"), "{}", text(&out.stderr));
}

#[test]
fn overlap_not_below_chunk_size_fails_before_the_store_opens() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, _) = corpus_copy(tmp.path());
    let db = tmp.path().join("kb.db");
    let out = troupekit(&["ingest", dir.to_str().unwrap(), "--db", db.to_str().unwrap(), "--embedder", "hashing:64", "--chunk-tokens", "50", "--overlap-tokens", "50"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(text(&out.stderr).contains("--overlap-tokens 50 must be below the chunk size 50"), "{}", text(&out.stderr));
    assert!(!db.exists());
}

#[test]
fn hnsw_i8_run_then_reopen_skips_unchanged() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, n) = corpus_copy(tmp.path());
    let db = tmp.path().join("kb.db");
    let run = || troupekit(&["ingest", dir.to_str().unwrap(), "--db", db.to_str().unwrap(), "--embedder", "hashing:128", "--kind", "hnsw", "--quant", "i8", "--index", "kb", "--scope", "help"]);
    let out = run();
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains(&format!("index kb: {n} added")), "{}", text(&out.stdout));
    let out = run();
    assert_eq!(out.status.code(), Some(0), "{}", text(&out.stderr));
    assert!(text(&out.stdout).contains(&format!("{n} unchanged, 0 failed")), "{}", text(&out.stdout));
}

#[test]
fn removal_overrides_reach_the_ingest() {
    let tmp = tempfile::tempdir().unwrap();
    let (dir, n) = corpus_copy(tmp.path());
    let db = tmp.path().join("kb.db");
    assert_eq!(ingest(&dir, &db).status.code(), Some(0));
    let mut names: Vec<PathBuf> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    names.sort();
    for path in &names[1..] {
        std::fs::remove_file(path).unwrap();
    }
    let refused = ingest(&dir, &db);
    assert_eq!(refused.status.code(), Some(1), "{}", text(&refused.stdout));
    assert!(text(&refused.stderr).contains("max_remove_share"), "{}", text(&refused.stderr));
    let (d, b) = (dir.to_str().unwrap(), db.to_str().unwrap());
    let share = troupekit(&["ingest", d, "--db", b, "--embedder", "hashing:256", "--max-remove-share", "100"]);
    assert_eq!(share.status.code(), Some(0), "{}", text(&share.stderr));
    assert!(text(&share.stdout).contains(&format!("{} removed", n - 1)), "{}", text(&share.stdout));
    std::fs::remove_file(&names[0]).unwrap();
    let refused = ingest(&dir, &db);
    assert_eq!(refused.status.code(), Some(1), "{}", text(&refused.stdout));
    let all = troupekit(&["ingest", d, "--db", b, "--embedder", "hashing:256", "--allow-remove-all"]);
    assert_eq!(all.status.code(), Some(0), "{}", text(&all.stderr));
    assert!(text(&all.stdout).contains("1 removed"), "{}", text(&all.stdout));
}
