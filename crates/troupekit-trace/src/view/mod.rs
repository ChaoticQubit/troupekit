//! Reads trace files back and renders them for people.
//! `read_jsonl` builds a `SpanTree` from a possibly truncated, hostile or
//! huge JSON lines file; `render_terminal` and `render_html` show it.

mod html;
mod parse;
mod terminal;
mod tree;

pub use html::render_html;
pub use terminal::{render_terminal, TermOptions};
pub use tree::{FieldValue, Node, NodeKind, SpanTree};

use parse::{parse, ParseError};
use std::error::Error;
use std::fmt;
use std::io::{self, BufRead, Read};

const LINE_BYTES_MAX_DEFAULT: usize = 1024 * 1024;
const RECORDS_MAX_DEFAULT: u64 = 1_000_000;
const DEPTH_MAX_DEFAULT: usize = 32;

/// Bounds for reading a trace file. Defaults: lines up to 1 MiB, at most
/// 1 000 000 records, JSON nesting at most 32 levels deep.
/// Input: `with_*` setters. Output: passed to `read_jsonl`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ViewLimits {
    line_bytes_max: usize,
    records_max: u64,
    depth_max: usize,
}

impl Default for ViewLimits {
    fn default() -> ViewLimits {
        ViewLimits {
            line_bytes_max: LINE_BYTES_MAX_DEFAULT,
            records_max: RECORDS_MAX_DEFAULT,
            depth_max: DEPTH_MAX_DEFAULT,
        }
    }
}

impl ViewLimits {
    /// Sets the longest accepted line in bytes, at least 1.
    pub fn with_line_bytes_max(mut self, bytes: usize) -> ViewLimits {
        self.line_bytes_max = bytes.max(1);
        self
    }

    /// Sets the most records read before stopping.
    pub fn with_records_max(mut self, records: u64) -> ViewLimits {
        self.records_max = records;
        self
    }

    /// Sets the deepest accepted JSON nesting, at least 1.
    pub fn with_depth_max(mut self, depth: usize) -> ViewLimits {
        self.depth_max = depth.max(1);
        self
    }
}

/// Errors from reading a trace file. Bad lines are counted in the tree,
/// not returned; only a failing reader is an error.
#[derive(Debug)]
#[non_exhaustive]
pub enum ViewError {
    Io(io::Error),
}

impl fmt::Display for ViewError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ViewError::Io(err) => write!(f, "reading the trace failed: {err}"),
        }
    }
}

impl Error for ViewError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            ViewError::Io(err) => Some(err),
        }
    }
}

impl From<io::Error> for ViewError {
    fn from(err: io::Error) -> ViewError {
        ViewError::Io(err)
    }
}

/// Reads JSON trace lines into a `SpanTree`, within `limits`.
/// A bad, oversized or too-deep line is skipped and counted; an `end`
/// with no `start` is counted; a span with no `end` is marked open.
pub fn read_jsonl<R: BufRead>(mut reader: R, limits: ViewLimits) -> Result<SpanTree, ViewError> {
    let mut tree = SpanTree::default();
    let mut line = Vec::new();
    let mut records = 0u64;
    loop {
        line.clear();
        let cap = (limits.line_bytes_max as u64).saturating_add(1);
        if (&mut reader).take(cap).read_until(b'\n', &mut line)? == 0 {
            break;
        }
        let complete = line.last() == Some(&b'\n');
        if complete {
            line.pop();
        }
        if line.len() > limits.line_bytes_max {
            tree.oversized_lines += 1;
            if !complete {
                skip_line(&mut reader)?;
            }
            continue;
        }
        if line.iter().all(u8::is_ascii_whitespace) {
            continue;
        }
        if records == limits.records_max {
            tree.record_limit_hit = true;
            break;
        }
        records += 1;
        read_line(&mut tree, &line, &limits);
    }
    assert!(records <= limits.records_max);
    Ok(tree)
}

fn read_line(tree: &mut SpanTree, line: &[u8], limits: &ViewLimits) {
    let Ok(text) = std::str::from_utf8(line) else {
        tree.bad_lines += 1;
        return;
    };
    match parse(text, limits.depth_max) {
        Ok(record) => {
            if tree.apply(record).is_err() {
                tree.bad_lines += 1;
            }
        }
        Err(ParseError::Depth) => tree.deep_lines += 1,
        Err(ParseError::Syntax) => tree.bad_lines += 1,
    }
}

fn skip_line<R: BufRead>(reader: &mut R) -> io::Result<()> {
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            return Ok(());
        }
        if let Some(at) = buffer.iter().position(|&b| b == b'\n') {
            reader.consume(at + 1);
            return Ok(());
        }
        let len = buffer.len();
        reader.consume(len);
    }
}
