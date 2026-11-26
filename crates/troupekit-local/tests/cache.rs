use std::sync::Arc;
use troupekit_local::sha256::{digest, to_hex};
use troupekit_local::{LocalError, LocalSpec, ModelCache};
use troupekit_providers::transport::{Scripted, ScriptedTransport};

const BODY: &[u8] = b"GGUF pretend model bytes for the cache tests";

fn spec_for(body: &[u8]) -> LocalSpec {
    LocalSpec::new(
        "tiny",
        "https://example.org/tiny.gguf",
        &to_hex(&digest(body)),
        body.len() as u64,
    )
    .expect("row")
}

#[tokio::test]
async fn fetch_writes_the_file_and_path_finds_it() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(BODY);
    assert!(cache.path(&spec).is_none());
    let transport = ScriptedTransport::new(vec![Scripted::chunked(
        std::str::from_utf8(BODY).expect("ascii"),
        7,
    )]);
    let path = cache.fetch(&spec, &transport).await.expect("fetches");
    assert_eq!(std::fs::read(&path).expect("read"), BODY);
    assert_eq!(cache.path(&spec), Some(path.clone()));
    assert_eq!(transport.sent().len(), 1);
    assert_eq!(transport.sent()[0].url, "https://example.org/tiny.gguf");

    let empty = ScriptedTransport::new(vec![]);
    assert_eq!(cache.fetch(&spec, &empty).await.expect("cached"), path);
    assert!(empty.sent().is_empty(), "a cached file needs no network");
}

#[tokio::test]
async fn fetch_with_wrong_hash_is_deleted_and_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(b"different bytes of the same length as the body!!");
    let wrong: Vec<u8> = b"different bytes of the same length as the body??".to_vec();
    assert_eq!(wrong.len() as u64, spec.bytes());
    let transport = ScriptedTransport::new(vec![Scripted::body(200, wrong)]);
    let err = cache.fetch(&spec, &transport).await.expect_err("bad hash");
    assert!(matches!(err, LocalError::Cache(_)), "{err:?}");
    assert!(cache.path(&spec).is_none());
    assert_eq!(
        std::fs::read_dir(dir.path()).expect("dir").count(),
        0,
        "part file deleted"
    );
}

#[tokio::test]
async fn fetch_over_declared_size_is_refused() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(BODY);
    let mut long = BODY.to_vec();
    long.extend_from_slice(b"extra");
    let transport = ScriptedTransport::new(vec![Scripted::body(200, long)]);
    let err = cache.fetch(&spec, &transport).await.expect_err("too long");
    assert!(matches!(err, LocalError::Cache(_)), "{err:?}");
    assert_eq!(std::fs::read_dir(dir.path()).expect("dir").count(), 0);
}

#[tokio::test]
async fn fetch_refuses_a_non_200_reply() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let transport = ScriptedTransport::new(vec![Scripted::body(404, "missing")]);
    let err = cache
        .fetch(&spec_for(BODY), &transport)
        .await
        .expect_err("404");
    assert!(matches!(err, LocalError::Cache(_)), "{err:?}");
}

#[test]
fn path_is_none_for_partial_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(BODY);
    std::fs::write(dir.path().join("tiny.gguf"), &BODY[..10]).expect("write");
    assert!(cache.path(&spec).is_none());
    std::fs::write(dir.path().join("tiny.gguf"), BODY).expect("write");
    assert!(cache.path(&spec).is_some());
}

#[test]
fn model_dir_variable_sets_the_default_folder() {
    let dir = tempfile::tempdir().expect("tempdir");
    std::env::set_var("TROUPEKIT_MODEL_DIR", dir.path());
    let cache = ModelCache::default_dir().expect("default");
    std::env::remove_var("TROUPEKIT_MODEL_DIR");
    assert_eq!(cache.dir(), dir.path());
    let _shared: Arc<ModelCache> = Arc::new(cache);
}

#[cfg(unix)]
#[tokio::test]
async fn fetched_file_is_private_to_the_user() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path().join("models"));
    let transport = ScriptedTransport::new(vec![Scripted::body(200, BODY)]);
    let path = cache
        .fetch(&spec_for(BODY), &transport)
        .await
        .expect("fetches");
    let mode = std::fs::metadata(&path).expect("meta").permissions().mode() & 0o777;
    assert_eq!(mode, 0o600);
    let dir_mode = std::fs::metadata(dir.path().join("models"))
        .expect("meta")
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(dir_mode, 0o700);
}

#[tokio::test]
async fn fetch_leaves_another_fetchs_part_file_alone() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(BODY);
    let other = dir.path().join("tiny.gguf.part");
    std::fs::write(&other, b"another fetch in progress").expect("write");
    let transport = ScriptedTransport::new(vec![Scripted::chunked(
        std::str::from_utf8(BODY).expect("ascii"),
        7,
    )]);
    let path = cache.fetch(&spec, &transport).await.expect("fetches");
    assert_eq!(std::fs::read(&path).expect("read"), BODY);
    assert_eq!(
        std::fs::read(&other).expect("other part kept"),
        b"another fetch in progress"
    );
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .expect("list")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n != "tiny.gguf" && n != "tiny.gguf.part")
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

#[tokio::test]
async fn concurrent_fetches_of_one_row_both_publish_a_checked_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(BODY);
    let text = std::str::from_utf8(BODY).expect("ascii");
    let a = ScriptedTransport::new(vec![Scripted::chunked(text, 3)]);
    let b = ScriptedTransport::new(vec![Scripted::chunked(text, 5)]);
    let (ra, rb) = tokio::join!(cache.fetch(&spec, &a), cache.fetch(&spec, &b));
    let (pa, pb) = (ra.expect("first"), rb.expect("second"));
    assert_eq!(pa, pb);
    assert_eq!(std::fs::read(&pa).expect("read"), BODY);
}
