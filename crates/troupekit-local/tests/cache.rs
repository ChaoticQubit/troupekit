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

fn file_name(spec: &LocalSpec) -> String {
    format!("tiny-{}.gguf", spec.sha256())
}

#[tokio::test]
async fn a_row_with_another_digest_is_not_a_cache_hit() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let (first, second) = (spec_for(b"AAAA"), spec_for(b"BBBB"));
    assert_eq!((first.id(), first.bytes()), (second.id(), second.bytes()));
    let a = ScriptedTransport::new(vec![Scripted::chunked("AAAA", 2)]);
    let pa = cache.fetch(&first, &a).await.expect("first fetches");
    assert!(cache.path(&second).is_none(), "another digest misses");
    let b = ScriptedTransport::new(vec![Scripted::chunked("BBBB", 2)]);
    let pb = cache.fetch(&second, &b).await.expect("second fetches");
    assert_eq!(b.sent().len(), 1, "the second row downloads");
    assert_ne!(pa, pb);
    assert_eq!(std::fs::read(&pa).expect("read"), b"AAAA");
    assert_eq!(std::fs::read(&pb).expect("read"), b"BBBB");
    assert_eq!(cache.path(&first), Some(pa));
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
    std::fs::write(dir.path().join(file_name(&spec)), &BODY[..10]).expect("write");
    assert!(cache.path(&spec).is_none());
    std::fs::write(dir.path().join("tiny.gguf"), BODY).expect("write");
    assert!(
        cache.path(&spec).is_none(),
        "a file without the digest is not used"
    );
    std::fs::write(dir.path().join(file_name(&spec)), BODY).expect("write");
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
        .filter(|n| *n != file_name(&spec) && n != "tiny.gguf.part")
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

struct Stall;

#[async_trait::async_trait]
impl troupekit_providers::transport::Body for Stall {
    async fn chunk(
        &mut self,
    ) -> Result<Option<Vec<u8>>, troupekit_providers::transport::HttpError> {
        std::future::pending().await
    }
}

struct StallingTransport;

#[async_trait::async_trait]
impl troupekit_providers::transport::Transport for StallingTransport {
    async fn send(
        &self,
        req: troupekit_providers::transport::HttpRequest,
    ) -> Result<
        troupekit_providers::transport::HttpResponse,
        troupekit_providers::transport::HttpError,
    > {
        Ok(troupekit_providers::transport::HttpResponse {
            status: 200,
            url: req.url,
            headers: Vec::new(),
            body: Box::new(Stall),
            truncated: false,
        })
    }
}

#[tokio::test]
async fn cancelled_fetch_removes_its_part_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let cache = ModelCache::new(dir.path());
    let spec = spec_for(BODY);
    let timed = tokio::time::timeout(
        std::time::Duration::from_millis(50),
        cache.fetch(&spec, &StallingTransport),
    )
    .await;
    assert!(timed.is_err(), "the fetch was still downloading");
    let left: Vec<_> = std::fs::read_dir(dir.path())
        .expect("list")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    assert!(left.is_empty(), "part file left behind: {left:?}");
}
