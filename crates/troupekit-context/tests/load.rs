use std::fs;
use std::path::{Path, PathBuf};
use troupekit_context::load::{walk, LoadSettings, Walked};
use troupekit_context::text::stable_hash;
use troupekit_context::ContextError;

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/load")
}

fn write(root: &Path, relative: &str, bytes: &[u8]) {
    let path = root.join(relative);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, bytes).unwrap();
}

fn sources(walked: &Walked) -> Vec<&str> {
    walked.files.iter().map(|f| f.source.as_str()).collect()
}

fn invalid_message(result: Result<Walked, ContextError>) -> String {
    match result {
        Err(ContextError::Invalid(message)) => message,
        other => panic!("expected Invalid, got {other:?}"),
    }
}

#[test]
fn walk_reads_txt_md_html_sorted() {
    let walked = walk(&fixture(), &LoadSettings::default()).unwrap();
    assert_eq!(sources(&walked), ["guides/shipping.md", "guides/warranty.markdown", "notes.txt", "returns.html"]);
    assert_eq!(walked.unsupported, 1);
    assert_eq!(walked.symlinks, 0);
    let notes = &walked.files[2];
    assert_eq!(notes.text, "Refunds are paid within 14 days.\n");
    for file in &walked.files {
        assert_eq!(file.content_hash, format!("{:016x}", stable_hash(&[&file.text])));
        assert_eq!(file.content_hash.len(), 16);
    }
    assert_eq!(walk(&fixture(), &LoadSettings::default()).unwrap(), walked);
}

#[test]
fn html_scripts_and_tags_are_removed() {
    let walked = walk(&fixture(), &LoadSettings::default()).unwrap();
    let html = walked.files.iter().find(|f| f.source == "returns.html").unwrap();
    assert!(html.text.contains("Returns"), "{}", html.text);
    assert!(html.text.contains("Send items back with the label."), "{}", html.text);
    for gone in ["<", ">", "steal", "color"] {
        assert!(!html.text.contains(gone), "{gone} left in {}", html.text);
    }
}

#[test]
fn front_matter_is_dropped() {
    let walked = walk(&fixture(), &LoadSettings::default()).unwrap();
    let md = walked.files.iter().find(|f| f.source == "guides/shipping.md").unwrap();
    assert_eq!(md.text, "# Shipping\n\nOrders ship in 2 days.\n");
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "open.md", b"---\nnever closed\n");
    write(dir.path(), "plain.txt", b"---\nkept: yes\n---\nbody\n");
    let walked = walk(dir.path(), &LoadSettings::default()).unwrap();
    assert_eq!(walked.files[0].text, "---\nnever closed\n");
    assert_eq!(walked.files[1].text, "---\nkept: yes\n---\nbody\n");
}

#[cfg(unix)]
#[test]
fn symlink_outside_dir_is_not_followed() {
    let outside = tempfile::tempdir().unwrap();
    write(outside.path(), "secret.txt", b"private key material");
    write(outside.path(), "more/also.txt", b"private too");
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "inside.txt", b"public");
    std::os::unix::fs::symlink(outside.path().join("secret.txt"), dir.path().join("link.txt")).unwrap();
    std::os::unix::fs::symlink(outside.path().join("more"), dir.path().join("linked_dir")).unwrap();
    let walked = walk(dir.path(), &LoadSettings::default()).unwrap();
    assert_eq!(sources(&walked), ["inside.txt"]);
    assert_eq!(walked.symlinks, 2);
    assert!(walked.files.iter().all(|f| !f.text.contains("private")));
}

#[cfg(unix)]
#[test]
fn unsupported_and_symlinks_are_counted() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.txt", b"a");
    write(dir.path(), "b.pdf", b"%PDF");
    write(dir.path(), "sub/c.png", b"png");
    write(dir.path(), "sub/no_extension", b"x");
    std::os::unix::fs::symlink(dir.path().join("a.txt"), dir.path().join("again.txt")).unwrap();
    let walked = walk(dir.path(), &LoadSettings::default()).unwrap();
    assert_eq!(sources(&walked), ["a.txt"]);
    assert_eq!(walked.unsupported, 3);
    assert_eq!(walked.symlinks, 1);
    let only_md = LoadSettings::default().with_extensions(["MD"]);
    let walked = walk(dir.path(), &only_md).unwrap();
    assert!(walked.files.is_empty());
    assert_eq!(walked.unsupported, 4);
}

#[test]
fn hidden_files_are_skipped() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "visible.txt", b"seen");
    write(dir.path(), ".env.txt", b"TOKEN=1");
    write(dir.path(), ".git/notes.txt", b"history");
    write(dir.path(), ".git/objects.bin", b"bin");
    let walked = walk(dir.path(), &LoadSettings::default()).unwrap();
    assert_eq!(sources(&walked), ["visible.txt"]);
    assert_eq!(walked.unsupported, 0);
    assert_eq!(walked.symlinks, 0);
    let shown = walk(dir.path(), &LoadSettings::default().with_skip_hidden(false)).unwrap();
    assert_eq!(sources(&shown), [".env.txt", ".git/notes.txt", "visible.txt"]);
}

#[test]
fn deep_hidden_tree_does_not_trip_depth() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "top.txt", b"top");
    write(dir.path(), ".git/a/b/c/d/e/deep.txt", b"deep");
    let settings = LoadSettings::default().with_max_depth(1);
    let walked = walk(dir.path(), &settings).unwrap();
    assert_eq!(sources(&walked), ["top.txt"]);
}

#[test]
fn file_count_depth_and_size_caps() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.txt", b"12345");
    write(dir.path(), "b.txt", b"1234");
    let message = invalid_message(walk(dir.path(), &LoadSettings::default().with_max_file_bytes(4)));
    assert!(message.contains("a.txt") && message.contains("max_file_bytes 4"), "{message}");
    let walked = walk(dir.path(), &LoadSettings::default().with_max_file_bytes(5).with_max_files(2)).unwrap();
    assert_eq!(walked.files.len(), 2);
    let message = invalid_message(walk(dir.path(), &LoadSettings::default().with_max_files(1)));
    assert!(message.contains("max_files 1"), "{message}");
    write(dir.path(), "x/y/z.txt", b"deep");
    let message = invalid_message(walk(dir.path(), &LoadSettings::default().with_max_depth(1)));
    assert!(message.contains("max_depth 1"), "{message}");
    assert_eq!(walk(dir.path(), &LoadSettings::default().with_max_depth(2)).unwrap().files.len(), 3);
}

#[test]
fn over_max_files_is_an_error_naming_the_cap() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["a.txt", "b.md", "c.html"] {
        write(dir.path(), name, b"text");
    }
    write(dir.path(), "d.csv", b"not counted");
    let at_cap = walk(dir.path(), &LoadSettings::default().with_max_files(3)).unwrap();
    assert_eq!(at_cap.files.len(), 3);
    let message = invalid_message(walk(dir.path(), &LoadSettings::default().with_max_files(2)));
    assert!(message.contains("max_files 2"), "{message}");
    assert!(!message.contains(&dir.path().to_string_lossy().into_owned()), "{message}");
}

#[test]
fn over_max_depth_is_an_error_naming_the_path() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a/b/c.txt", b"deep");
    let at_cap = walk(dir.path(), &LoadSettings::default().with_max_depth(2)).unwrap();
    assert_eq!(sources(&at_cap), ["a/b/c.txt"]);
    let message = invalid_message(walk(dir.path(), &LoadSettings::default().with_max_depth(1)));
    assert!(message.starts_with("a/b:"), "{message}");
    assert!(message.contains("max_depth 1"), "{message}");
}

#[test]
fn over_max_entries_is_an_error() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..5 {
        write(dir.path(), &format!("f{i}.bin"), b"x");
    }
    let at_cap = walk(dir.path(), &LoadSettings::default().with_max_entries(5)).unwrap();
    assert_eq!(at_cap.unsupported, 5);
    let message = invalid_message(walk(dir.path(), &LoadSettings::default().with_max_entries(4)));
    assert!(message.contains("max_entries 4"), "{message}");
}

#[test]
fn invalid_utf8_is_an_error_naming_the_source() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "docs/bad.txt", &[0x66, 0xff, 0xfe]);
    let message = invalid_message(walk(dir.path(), &LoadSettings::default()));
    assert!(message.contains("docs/bad.txt") && message.contains("UTF-8"), "{message}");
}

#[test]
fn source_is_relative_with_slashes() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a/b/c.txt", b"nested");
    let walked = walk(dir.path(), &LoadSettings::default()).unwrap();
    assert_eq!(sources(&walked), ["a/b/c.txt"]);
    let root = dir.path().to_string_lossy().into_owned();
    assert!(!walked.files[0].source.contains(&root));
    assert!(!walked.files[0].source.starts_with('/'));
}

#[test]
fn not_a_folder_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "file.txt", b"x");
    let message = invalid_message(walk(&dir.path().join("file.txt"), &LoadSettings::default()));
    assert!(message.contains("not a folder"), "{message}");
    assert!(matches!(walk(&dir.path().join("missing"), &LoadSettings::default()), Err(ContextError::Invalid(_))));
}

#[test]
fn defaults_are_the_safe_values() {
    let s = LoadSettings::default();
    assert_eq!(s.extensions(), ["txt", "md", "markdown", "html", "htm"]);
    assert_eq!(s.max_files(), 10_000);
    assert_eq!(s.max_file_bytes(), 8 * 1024 * 1024);
    assert_eq!(s.max_depth(), 16);
    assert_eq!(s.max_entries(), 100_000);
    assert!(!s.follow_symlinks());
    assert!(s.skip_hidden());
}

#[cfg(unix)]
#[test]
fn non_regular_files_are_skipped_not_opened() {
    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "a.txt", b"a");
    let status = std::process::Command::new("mkfifo").arg(dir.path().join("waiting.txt")).status().unwrap();
    assert!(status.success());
    let root = dir.path().to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(walk(&root, &LoadSettings::default())).unwrap());
    let walked = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("walk blocked on a FIFO").unwrap();
    assert_eq!(sources(&walked), ["a.txt"]);
    assert_eq!(walked.unsupported, 1);
    let follow = LoadSettings::default().with_follow_symlinks(true);
    std::os::unix::fs::symlink(dir.path().join("waiting.txt"), dir.path().join("link.md")).unwrap();
    let root = dir.path().to_path_buf();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || tx.send(walk(&root, &follow)).unwrap());
    let walked = rx.recv_timeout(std::time::Duration::from_secs(10)).expect("walk blocked on a linked FIFO").unwrap();
    assert_eq!(sources(&walked), ["a.txt"]);
    assert_eq!(walked.unsupported, 2);
}
