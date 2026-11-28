use crate::error::KitError;
use std::path::{Component, Path, PathBuf};
use troupekit_trace::{trace, Trace};

const MAX_LAW_FILE_BYTES: u64 = 1024 * 1024;
const MAX_INLINE_LAW_BYTES: usize = 64 * 1024;
const MAX_LAWS: usize = 64;

/// The laws attached to an agent or graph, checked at `build`.
/// 0.2 stores and validates them; no run proves them yet. `auto` asks
/// the verifier to also derive laws from the code, and defaults to `true`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Laws {
    paths: Vec<PathBuf>,
    inline: Vec<String>,
    auto: bool,
}

impl Laws {
    /// Returns the law files, as absolute paths resolved at `build`.
    /// Each is an existing `.bend` file of at most 1 MiB of UTF-8.
    /// Order is the order they were added.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Returns the inline laws in the order they were added.
    /// Each is non-blank, at most 64 KiB and free of NUL bytes.
    /// The text is kept exactly as given.
    pub fn inline(&self) -> &[String] {
        &self.inline
    }

    /// Reports whether laws derived from the code are wanted.
    /// Set with `.auto_laws(..)`; `true` unless turned off.
    /// 0.2 records the choice only.
    pub fn auto(&self) -> bool {
        self.auto
    }

    pub(crate) fn check(
        owner: &str,
        paths: &[PathBuf],
        inline: &[String],
        auto: bool,
    ) -> Result<Laws, KitError> {
        let fail = |m: String| KitError::config(owner, m);
        let total = paths.len() + inline.len();
        if total > MAX_LAWS {
            return Err(fail(format!("takes at most {MAX_LAWS} laws, got {total}")));
        }
        let mut resolved = Vec::with_capacity(paths.len());
        for path in paths {
            resolved.push(
                read_checked(path, "bend", MAX_LAW_FILE_BYTES)
                    .map(|(absolute, _)| absolute)
                    .map_err(|m| fail(format!("law file `{}` {m}", path.display())))?,
            );
        }
        for (i, law) in inline.iter().enumerate() {
            let n = i + 1;
            if law.trim().is_empty() {
                return Err(fail(format!("inline law {n} is blank")));
            }
            if law.len() > MAX_INLINE_LAW_BYTES {
                return Err(fail(format!("inline law {n} is over 64 KiB")));
            }
            if law.contains('\0') {
                return Err(fail(format!("inline law {n} contains a NUL byte")));
            }
        }
        assert_eq!(resolved.len(), paths.len());
        for path in &resolved {
            let shown = path.display().to_string();
            trace!(
                Trace::Laws,
                "law.load",
                owner = owner,
                source = "file",
                path = shown,
                auto = auto,
            );
        }
        for law in inline {
            trace!(
                Trace::Laws,
                "law.load",
                owner = owner,
                source = "inline",
                bytes = law.len(),
                auto = auto,
            );
        }
        Ok(Laws {
            paths: resolved,
            inline: inline.to_vec(),
            auto,
        })
    }
}

impl Default for Laws {
    fn default() -> Self {
        Laws {
            paths: Vec::new(),
            inline: Vec::new(),
            auto: true,
        }
    }
}

/// Checks and reads one user-named file: non-empty path, no `..`, the
/// given extension, a regular file of at most `max_bytes`, valid UTF-8.
/// Returns the absolute path and the text, or the reason it was refused.
/// `max_bytes` must be a whole number of MiB.
pub(crate) fn read_checked(
    path: &Path,
    extension: &str,
    max_bytes: u64,
) -> Result<(PathBuf, String), String> {
    assert!(max_bytes > 0 && max_bytes % (1024 * 1024) == 0);
    let too_big = || format!("is over {} MiB", max_bytes / (1024 * 1024));
    if path.as_os_str().is_empty() {
        return Err("is empty".into());
    }
    if path.components().any(|c| c == Component::ParentDir) {
        return Err("must not contain `..`".into());
    }
    if path.extension().is_none_or(|e| e != extension) {
        return Err(format!("must be a .{extension} file"));
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|e| format!("cannot be resolved: {e}"))?
            .join(path)
    };
    let meta = match std::fs::metadata(&absolute) {
        Ok(meta) => meta,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err("does not exist".into()),
        Err(e) => return Err(format!("cannot be read: {e}")),
    };
    if !meta.is_file() {
        return Err("is not a regular file".into());
    }
    if meta.len() > max_bytes {
        return Err(too_big());
    }
    let bytes = std::fs::read(&absolute).map_err(|e| format!("cannot be read: {e}"))?;
    if bytes.len() as u64 > max_bytes {
        return Err(too_big());
    }
    let text = String::from_utf8(bytes).map_err(|_| "is not valid UTF-8".to_string())?;
    Ok((absolute, text))
}
