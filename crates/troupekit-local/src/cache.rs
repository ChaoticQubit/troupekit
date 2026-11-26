use crate::catalog::LocalSpec;
use crate::error::LocalError;
use crate::sha256::{to_hex, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const PART_TRIES: usize = 64;
use troupekit_core::http::{HttpRequest, Transport};

/// A folder of downloaded model files, one `<id>.gguf` per catalog row.
/// A file is used only when its size matches the row; `fetch` checks
/// the SHA-256 before a file gets its final name. No call downloads alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCache {
    dir: PathBuf,
}

impl ModelCache {
    /// Uses `dir` as the model folder; it is created on the first fetch.
    /// Mobile apps pass their own app folder here.
    /// Output: the cache.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// Picks the default folder: `$TROUPEKIT_MODEL_DIR`, else
    /// `$HOME/.cache/troupekit/models`, else `%LOCALAPPDATA%\troupekit\models`.
    /// Output: the cache, or `Cache` asking for `ModelCache::new(dir)`.
    pub fn default_dir() -> Result<Self, LocalError> {
        let non_empty = |name: &str| {
            std::env::var_os(name)
                .filter(|v| !v.is_empty())
                .map(PathBuf::from)
        };
        if let Some(dir) = non_empty("TROUPEKIT_MODEL_DIR") {
            return Ok(Self::new(dir));
        }
        if let Some(home) = non_empty("HOME") {
            return Ok(Self::new(
                home.join(".cache").join("troupekit").join("models"),
            ));
        }
        if let Some(local) = non_empty("LOCALAPPDATA") {
            return Ok(Self::new(local.join("troupekit").join("models")));
        }
        Err(LocalError::Cache(
            "no model folder: set TROUPEKIT_MODEL_DIR or build the cache with ModelCache::new(dir)"
                .into(),
        ))
    }

    /// Returns the model folder.
    /// Input: none.
    /// Output: the folder path.
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Returns the file of a row when it exists as a regular file of
    /// exactly the declared size; a partial or other file gives `None`.
    /// Never touches the network.
    pub fn path(&self, spec: &LocalSpec) -> Option<PathBuf> {
        let path = self.file_of(spec);
        let meta = std::fs::symlink_metadata(&path).ok()?;
        (meta.is_file() && meta.len() == spec.bytes()).then_some(path)
    }

    /// Downloads a row through `transport` unless `path` already finds
    /// it: into a part file of its own (created exclusively, 0600), refused
    /// past the declared size or on a hash mismatch, then renamed. Output: the path.
    pub async fn fetch(
        &self,
        spec: &LocalSpec,
        transport: &dyn Transport,
    ) -> Result<PathBuf, LocalError> {
        if let Some(path) = self.path(spec) {
            return Ok(path);
        }
        if !spec.url().starts_with("https://") {
            return Err(LocalError::Cache(format!(
                "model {}: refusing a non-https URL",
                spec.id()
            )));
        }
        create_private_dir(&self.dir)?;
        let (part, file) = create_part_file(&self.dir, spec.id())?;
        if let Err(e) = self.download(spec, transport, file).await {
            let _ = std::fs::remove_file(&part);
            return Err(e);
        }
        let path = self.file_of(spec);
        std::fs::rename(&part, &path).map_err(|e| {
            let _ = std::fs::remove_file(&part);
            LocalError::Cache(format!("model {}: rename failed: {e}", spec.id()))
        })?;
        if self.path(spec).is_none() {
            return Err(LocalError::Cache(format!(
                "model {}: file changed after it was published",
                spec.id()
            )));
        }
        Ok(path)
    }

    async fn download(
        &self,
        spec: &LocalSpec,
        transport: &dyn Transport,
        mut file: File,
    ) -> Result<(), LocalError> {
        let id = spec.id();
        let mut response = transport
            .send(HttpRequest::get(spec.url()))
            .await
            .map_err(|e| LocalError::Cache(format!("model {id}: download failed: {e}")))?;
        if response.status != 200 {
            return Err(LocalError::Cache(format!(
                "model {id}: server answered {}",
                response.status
            )));
        }
        let mut hasher = Sha256::new();
        let mut total: u64 = 0;
        while let Some(chunk) = response
            .body
            .chunk()
            .await
            .map_err(|e| LocalError::Cache(format!("model {id}: download failed: {e}")))?
        {
            total = total.saturating_add(chunk.len() as u64);
            if total > spec.bytes() {
                return Err(LocalError::Cache(format!(
                    "model {id}: download is over the declared {} bytes",
                    spec.bytes()
                )));
            }
            hasher.update(&chunk);
            file.write_all(&chunk)
                .map_err(|e| LocalError::Cache(format!("model {id}: write failed: {e}")))?;
        }
        if response.truncated || total != spec.bytes() {
            return Err(LocalError::Cache(format!(
                "model {id}: got {total} bytes, expected {}",
                spec.bytes()
            )));
        }
        file.sync_all()
            .map_err(|e| LocalError::Cache(format!("model {id}: sync failed: {e}")))?;
        let got = to_hex(&hasher.finish());
        if got != spec.sha256() {
            return Err(LocalError::Cache(format!(
                "model {id}: SHA-256 {got} does not match the catalog's {}",
                spec.sha256()
            )));
        }
        Ok(())
    }

    fn file_of(&self, spec: &LocalSpec) -> PathBuf {
        self.dir.join(format!("{}.gguf", spec.id()))
    }
}

fn create_private_dir(dir: &Path) -> Result<(), LocalError> {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        builder.mode(0o700);
    }
    builder
        .create(dir)
        .map_err(|e| LocalError::Cache(format!("cannot create {}: {e}", dir.display())))
}

fn create_part_file(dir: &Path, id: &str) -> Result<(PathBuf, File), LocalError> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    for _ in 0..PART_TRIES {
        let n = NEXT.fetch_add(1, Ordering::Relaxed);
        let part = dir.join(format!("{id}.gguf.{}-{n}.part", std::process::id()));
        match options.open(&part) {
            Ok(file) => return Ok((part, file)),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(LocalError::Cache(format!(
                    "cannot create {}: {e}",
                    part.display()
                )))
            }
        }
    }
    Err(LocalError::Cache(format!(
        "model {id}: no free part file name after {PART_TRIES} tries"
    )))
}
