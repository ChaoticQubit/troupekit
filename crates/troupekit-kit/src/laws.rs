use crate::error::KitError;
use std::path::{Component, PathBuf};
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
                check_path(path).map_err(|m| fail(format!("law file `{}` {m}", path.display())))?,
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

fn check_path(path: &PathBuf) -> Result<PathBuf, String> {
    if path.as_os_str().is_empty() {
        return Err("is empty".into());
    }
    if path.components().any(|c| c == Component::ParentDir) {
        return Err("must not contain `..`".into());
    }
    if path.extension().is_none_or(|e| e != "bend") {
        return Err("must be a .bend file".into());
    }
    let absolute = if path.is_absolute() {
        path.clone()
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
    if meta.len() > MAX_LAW_FILE_BYTES {
        return Err("is over 1 MiB".into());
    }
    let bytes = std::fs::read(&absolute).map_err(|e| format!("cannot be read: {e}"))?;
    if bytes.len() as u64 > MAX_LAW_FILE_BYTES {
        return Err("is over 1 MiB".into());
    }
    if std::str::from_utf8(&bytes).is_err() {
        return Err("is not valid UTF-8".into());
    }
    Ok(absolute)
}
