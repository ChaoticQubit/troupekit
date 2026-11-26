//! Loads text, markdown and HTML files from a folder for ingest.
//! The walk is bounded in file count, file size and depth, skips hidden
//! entries and symlinks by default, and returns files sorted by source.

use crate::error::ContextError;
use crate::text::stable_hash;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use troupekit_tools::text::html_to_text;

const MAX_FILES_DEFAULT: usize = 10_000;
const MAX_FILE_BYTES_DEFAULT: u64 = 8 * 1024 * 1024;
const MAX_DEPTH_DEFAULT: usize = 16;
const MAX_ENTRIES_DEFAULT: usize = 100_000;
const EXTENSIONS_DEFAULT: [&str; 5] = ["txt", "md", "markdown", "html", "htm"];
const HTML_EXTENSIONS: [&str; 2] = ["html", "htm"];
const MARKDOWN_EXTENSIONS: [&str; 2] = ["md", "markdown"];

/// Bounds and filters of a folder walk. Every field has a safe default:
/// the five text extensions, 10 000 files, 8 MiB per file, depth 16,
/// 100 000 directory entries, symlinks not followed, hidden entries
/// skipped. Built with `Default` and the `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadSettings {
    extensions: Vec<String>,
    max_files: usize,
    max_file_bytes: u64,
    max_depth: usize,
    max_entries: usize,
    follow_symlinks: bool,
    skip_hidden: bool,
}

impl Default for LoadSettings {
    fn default() -> Self {
        Self {
            extensions: EXTENSIONS_DEFAULT.iter().map(|e| (*e).to_string()).collect(),
            max_files: MAX_FILES_DEFAULT,
            max_file_bytes: MAX_FILE_BYTES_DEFAULT,
            max_depth: MAX_DEPTH_DEFAULT,
            max_entries: MAX_ENTRIES_DEFAULT,
            follow_symlinks: false,
            skip_hidden: true,
        }
    }
}

impl LoadSettings {
    /// Sets the file extensions to load, compared without case.
    /// Only `txt`, `md`, `markdown`, `html` and `htm` can be read; any
    /// other extension in the list is still skipped.
    pub fn with_extensions<I, S>(mut self, extensions: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.extensions = extensions.into_iter().map(|e| e.into().to_ascii_lowercase()).collect();
        self
    }

    /// Sets the most supported files one walk loads.
    /// A folder with more is an error, so a walk never stops halfway
    /// in silence. Default 10 000.
    pub fn with_max_files(mut self, max_files: usize) -> Self {
        self.max_files = max_files;
        self
    }

    /// Sets the largest file in bytes.
    /// A larger file is an error that names the source, checked from
    /// metadata before any read. Default 8 MiB.
    pub fn with_max_file_bytes(mut self, max_file_bytes: u64) -> Self {
        self.max_file_bytes = max_file_bytes;
        self
    }

    /// Sets the deepest folder level below the root, which is level 0.
    /// A folder deeper than this is an error that names it. Hidden
    /// folders are never entered, so they never trip it. Default 16.
    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Sets the most directory entries one walk visits, of any kind.
    /// It bounds a walk through big folders of unsupported files; more
    /// entries is an error. Default 100 000.
    pub fn with_max_entries(mut self, max_entries: usize) -> Self {
        self.max_entries = max_entries;
        self
    }

    /// Sets whether symlinks are followed.
    /// Off by default, because a link can point outside the folder the
    /// user chose. A link not followed is counted in `Walked.symlinks`.
    pub fn with_follow_symlinks(mut self, follow: bool) -> Self {
        self.follow_symlinks = follow;
        self
    }

    /// Sets whether entries whose name starts with a dot are skipped.
    /// On by default, so `.git` and editor files stay out. A skipped
    /// hidden entry is neither entered nor counted.
    pub fn with_skip_hidden(mut self, skip: bool) -> Self {
        self.skip_hidden = skip;
        self
    }

    /// Returns the extensions loaded, lowercase.
    /// Only the five readable text extensions can match a file.
    /// Default `txt`, `md`, `markdown`, `html`, `htm`.
    pub fn extensions(&self) -> &[String] {
        &self.extensions
    }

    /// Returns the most supported files one walk loads.
    /// One more is an error naming the cap.
    /// Default 10 000.
    pub fn max_files(&self) -> usize {
        self.max_files
    }

    /// Returns the largest file in bytes.
    /// A larger file is an error naming the source.
    /// Default 8 MiB.
    pub fn max_file_bytes(&self) -> u64 {
        self.max_file_bytes
    }

    /// Returns the deepest folder level walked.
    /// The root is level 0.
    /// Default 16.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// Returns the most directory entries one walk visits.
    /// Every entry counts, hidden or not.
    /// Default 100 000.
    pub fn max_entries(&self) -> usize {
        self.max_entries
    }

    /// Returns whether symlinks are followed.
    /// A followed link is loaded like the file it points to.
    /// Default false.
    pub fn follow_symlinks(&self) -> bool {
        self.follow_symlinks
    }

    /// Returns whether hidden entries are skipped.
    /// A hidden entry starts with a dot.
    /// Default true.
    pub fn skip_hidden(&self) -> bool {
        self.skip_hidden
    }
}

/// One loaded file. `source` is the path relative to the walked folder
/// with `/` separators, so no absolute path reaches a prompt. `text` is
/// plain text, and `content_hash` is its stable hash in 16 hex chars.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoadedFile {
    pub source: String,
    pub text: String,
    pub content_hash: String,
}

/// The result of one walk: the loaded files sorted by `source`, the
/// count of entries skipped as unsupported (extension, or not a regular
/// file such as a FIFO) and the count of symlinks not followed.
/// Hidden entries are neither entered nor counted.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Walked {
    pub files: Vec<LoadedFile>,
    pub unsupported: usize,
    pub symlinks: usize,
}

/// Walks `dir` and loads every file the settings allow.
/// Input: a folder and the walk bounds. Output: the files sorted by
/// source and the skip counts, or `ContextError::Invalid` naming the
/// cap and relative path, or the source on invalid UTF-8 or a read error.
pub fn walk(dir: &Path, settings: &LoadSettings) -> Result<Walked, ContextError> {
    let root_meta = fs::metadata(dir).map_err(|e| ContextError::Invalid(format!("cannot read folder: {e}")))?;
    if !root_meta.is_dir() {
        return Err(ContextError::Invalid("not a folder".into()));
    }
    let mut walked = Walked { files: Vec::new(), unsupported: 0, symlinks: 0 };
    let mut pending: Vec<(PathBuf, usize)> = vec![(dir.to_path_buf(), 0)];
    let mut entries_seen: usize = 0;
    while let Some((folder, depth)) = pending.pop() {
        assert!(depth <= settings.max_depth, "folder queued past max_depth");
        let entries = fs::read_dir(&folder).map_err(|e| invalid(dir, &folder, &e.to_string()))?;
        for entry in entries {
            let entry = entry.map_err(|e| invalid(dir, &folder, &e.to_string()))?;
            let path = entry.path();
            entries_seen += 1;
            if entries_seen > settings.max_entries {
                return Err(invalid(dir, &path, &format!("more than max_entries {} entries", settings.max_entries)));
            }
            if settings.skip_hidden && entry.file_name().to_string_lossy().starts_with('.') {
                continue;
            }
            let file_type = entry.file_type().map_err(|e| invalid(dir, &path, &e.to_string()))?;
            let (is_dir, is_file) = if file_type.is_symlink() {
                if !settings.follow_symlinks {
                    walked.symlinks += 1;
                    continue;
                }
                let target = fs::metadata(&path).map_err(|e| invalid(dir, &path, &e.to_string()))?;
                (target.is_dir(), target.is_file())
            } else {
                (file_type.is_dir(), file_type.is_file())
            };
            if is_dir {
                if depth + 1 > settings.max_depth {
                    return Err(invalid(dir, &path, &format!("folder deeper than max_depth {}", settings.max_depth)));
                }
                pending.push((path, depth + 1));
                continue;
            }
            let Some(ext) = extension(&path, settings).filter(|_| is_file) else {
                walked.unsupported += 1;
                continue;
            };
            if walked.files.len() == settings.max_files {
                return Err(invalid(dir, &path, &format!("more than max_files {} files", settings.max_files)));
            }
            walked.files.push(load_file(dir, &path, &ext, settings)?);
        }
    }
    walked.files.sort_by(|a, b| a.source.cmp(&b.source));
    assert!(walked.files.len() <= settings.max_files, "loaded past max_files");
    assert!(entries_seen <= settings.max_entries, "visited past max_entries");
    assert!(walked.files.windows(2).all(|w| w[0].source < w[1].source), "sources not unique and sorted");
    Ok(walked)
}

/// Returns the lowercase extension of `path` when the settings list it
/// and the loader can read it. Input: a file path and the settings.
/// Output: the extension, or `None` to skip the file.
fn extension(path: &Path, settings: &LoadSettings) -> Option<String> {
    let ext = path.extension()?.to_str()?.to_ascii_lowercase();
    let readable = EXTENSIONS_DEFAULT.contains(&ext.as_str());
    (readable && settings.extensions.contains(&ext)).then_some(ext)
}

/// Reads one file under the byte cap and turns it into plain text.
/// Input: the walk root, the file, its extension and the settings.
/// Output: the loaded file, or `Invalid` naming the source.
fn load_file(root: &Path, path: &Path, ext: &str, settings: &LoadSettings) -> Result<LoadedFile, ContextError> {
    let source = source_of(root, path)?;
    let fail = |why: String| ContextError::Invalid(format!("{source}: {why}"));
    let checked = if settings.follow_symlinks { fs::metadata(path) } else { fs::symlink_metadata(path) }.map_err(|e| fail(e.to_string()))?;
    if !checked.is_file() {
        return Err(fail("not a regular file".into()));
    }
    let file = fs::File::open(path).map_err(|e| fail(e.to_string()))?;
    let opened = file.metadata().map_err(|e| fail(e.to_string()))?;
    if !opened.is_file() || !same_file(&checked, &opened) {
        return Err(fail("changed between check and open".into()));
    }
    let size = opened.len();
    if size > settings.max_file_bytes {
        return Err(fail(format!("{size} bytes, over max_file_bytes {}", settings.max_file_bytes)));
    }
    let mut bytes = Vec::new();
    file.take(settings.max_file_bytes.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| fail(e.to_string()))?;
    if bytes.len() as u64 > settings.max_file_bytes {
        return Err(fail(format!("grew over max_file_bytes {} while read", settings.max_file_bytes)));
    }
    let raw = String::from_utf8(bytes).map_err(|_| fail("not valid UTF-8".into()))?;
    let text = if HTML_EXTENSIONS.contains(&ext) {
        html_to_text(&raw)
    } else if MARKDOWN_EXTENSIONS.contains(&ext) {
        strip_front_matter(&raw).to_string()
    } else {
        raw
    };
    let content_hash = format!("{:016x}", stable_hash(&[&text]));
    assert_eq!(content_hash.len(), 16, "hash is 16 hex chars");
    Ok(LoadedFile { source, text, content_hash })
}

/// Tells whether two metadata records name the same file: same device
/// and inode on Unix, so a path swapped for a symlink after the check is
/// refused. Input: the checked and the opened metadata. Output: the answer.
#[cfg(unix)]
fn same_file(checked: &fs::Metadata, opened: &fs::Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    checked.dev() == opened.dev() && checked.ino() == opened.ino()
}

/// Tells whether two metadata records name the same file. Without inode
/// numbers it compares length and modification time, the best std offers.
/// Input: the checked and the opened metadata. Output: the answer.
#[cfg(not(unix))]
fn same_file(checked: &fs::Metadata, opened: &fs::Metadata) -> bool {
    checked.len() == opened.len() && checked.modified().ok() == opened.modified().ok()
}

/// Drops markdown front matter: a first line `---` up to the next line
/// that is `---`. Input: the file text. Output: the text after the
/// closing line, or the whole text when no closed block opens it.
fn strip_front_matter(text: &str) -> &str {
    let Some(rest) = text.strip_prefix("---\n").or_else(|| text.strip_prefix("---\r\n")) else {
        return text;
    };
    let mut offset = 0;
    for line in rest.split_inclusive('\n') {
        offset += line.len();
        if line.trim_end_matches(['\r', '\n']) == "---" {
            return &rest[offset..];
        }
    }
    text
}

/// Path of `path` relative to `root`, joined with `/`.
/// Input: the walk root and a path inside it. Output: the relative
/// source, or `Invalid` for a name that is not UTF-8.
fn source_of(root: &Path, path: &Path) -> Result<String, ContextError> {
    let relative = path.strip_prefix(root).expect("walked path is under the root");
    let mut parts = Vec::new();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            unreachable!("read_dir yields plain names only");
        };
        let name = name.to_str().ok_or_else(|| ContextError::Invalid(format!("{}: name is not UTF-8", relative.to_string_lossy())))?;
        parts.push(name);
    }
    assert!(!parts.is_empty(), "a file has a name");
    Ok(parts.join("/"))
}

/// Error for an entry the walk cannot read, naming its relative path.
/// Input: the root, the entry and the reason.
/// Output: `Invalid` with the path and the reason.
fn invalid(root: &Path, path: &Path, why: &str) -> ContextError {
    let shown = path.strip_prefix(root).map_or_else(|_| String::from("."), |p| p.to_string_lossy().into_owned());
    let shown = if shown.is_empty() { String::from(".") } else { shown };
    ContextError::Invalid(format!("{shown}: {why}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_needs_a_closing_line() {
        assert_eq!(strip_front_matter("---\ntitle: a\n---\nbody"), "body");
        assert_eq!(strip_front_matter("---\r\ntitle: a\r\n---\r\nbody"), "body");
        assert_eq!(strip_front_matter("---\nno close"), "---\nno close");
        assert_eq!(strip_front_matter("body\n---\nx\n---\n"), "body\n---\nx\n---\n");
    }
}
