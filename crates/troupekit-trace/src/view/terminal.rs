use super::tree::{FieldValue, Node, NodeKind, SpanTree};
use crate::category::TraceSet;
use std::fmt::Write;

const LABEL_CHARS: usize = 24;
const BAR_CHARS_MIN: usize = 10;
const WIDTH_MIN: usize = LABEL_CHARS + 3 + BAR_CHARS_MIN;
const INDENT_LEVELS_MAX: usize = 32;
const VALUE_CHARS_MAX: usize = 80;
const GREEN: &str = "\u{1b}[32m";
const RED: &str = "\u{1b}[31m";
const YELLOW: &str = "\u{1b}[33m";
const RESET: &str = "\u{1b}[0m";

/// Settings for `render_terminal`. Defaults: 100 columns, no color, every
/// category, at most 2 000 rows per section. The CLI turns color on when
/// its output is a terminal.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct TermOptions {
    width: usize,
    color: bool,
    only: TraceSet,
    max_rows: usize,
}

impl Default for TermOptions {
    fn default() -> TermOptions {
        TermOptions {
            width: 100,
            color: false,
            only: TraceSet::all(),
            max_rows: 2_000,
        }
    }
}

impl TermOptions {
    /// Sets the line width, at least 37 columns.
    pub fn with_width(mut self, width: usize) -> TermOptions {
        self.width = width;
        self
    }

    /// Turns plain ANSI color codes on or off.
    pub fn with_color(mut self, color: bool) -> TermOptions {
        self.color = color;
        self
    }

    /// Shows only nodes of these categories.
    pub fn with_only(mut self, only: TraceSet) -> TermOptions {
        self.only = only;
        self
    }

    /// Sets the most rows per section, at least 1.
    pub fn with_max_rows(mut self, max_rows: usize) -> TermOptions {
        self.max_rows = max_rows.max(1);
        self
    }
}

/// Renders a span tree as text: a header line, the tree with category,
/// status, duration and fields, a blank line, then one timeline row per
/// span. Control characters in traced text print as `\xNN`.
pub fn render_terminal(tree: &SpanTree, options: &TermOptions) -> String {
    let rows = visible_rows(tree, options.only);
    let mut out = String::new();
    let name = escape(tree.name().unwrap_or("-"));
    let _ = writeln!(
        out,
        "trace {name} {}",
        escape(tree.trace_id().unwrap_or("-"))
    );
    let tree_rows = rows
        .iter()
        .map(|&(index, depth)| tree_row(tree.node(index), depth, options.color));
    push_capped(&mut out, tree_rows, options.max_rows);
    out.push('\n');
    let spans = rows
        .iter()
        .filter(|&&(index, _)| tree.node(index).kind() == NodeKind::Span);
    let bars = spans.map(|&(index, depth)| timeline_row(tree, tree.node(index), depth, options));
    push_capped(&mut out, bars, options.max_rows);
    assert!(out.bytes().all(|b| b >= 0x20 || b == b'\n' || b == 0x1b));
    out
}

fn push_capped(out: &mut String, rows: impl Iterator<Item = String>, max_rows: usize) {
    let mut hidden = 0usize;
    for (count, row) in rows.enumerate() {
        if count < max_rows {
            out.push_str(&row);
            out.push('\n');
        } else {
            hidden += 1;
        }
    }
    if hidden > 0 {
        let _ = writeln!(out, "... {hidden} more");
    }
}

fn visible_rows(tree: &SpanTree, only: TraceSet) -> Vec<(usize, usize)> {
    let mut rows = Vec::new();
    let mut stack: Vec<(usize, usize)> = tree.roots().iter().rev().map(|&i| (i, 0)).collect();
    while let Some((index, depth)) = stack.pop() {
        let node = tree.node(index);
        let shown = only.contains(node.category());
        if shown {
            rows.push((index, depth));
        }
        let child_depth = if shown { depth + 1 } else { depth };
        for &child in node.children().iter().rev() {
            assert!(child > index, "a child is always read after its parent");
            stack.push((child, child_depth));
        }
    }
    assert!(rows.len() <= tree.nodes().len());
    rows
}

fn tree_row(node: &Node, depth: usize, color: bool) -> String {
    let mut row = indent(depth);
    let _ = write!(row, "{} [{}] ", escape(node.name()), node.category());
    let (word, code) = status(node);
    paint(&mut row, word, code, color);
    if let Some(dur_us) = node.dur_us() {
        let _ = write!(row, " {}", duration(dur_us));
    }
    if let Some(err) = node.err() {
        let _ = write!(row, " err={}", escape(cut(err)));
    }
    for (key, value) in node.fields() {
        let _ = write!(row, " {}={}", escape(cut(key)), field_text(value));
    }
    row
}

fn timeline_row(tree: &SpanTree, node: &Node, depth: usize, options: &TermOptions) -> String {
    let bar_chars = options.width.max(WIDTH_MIN) - LABEL_CHARS - 3;
    let (first, last) = tree.time_range_us().unwrap_or((0, 0));
    let total = (last - first).max(1);
    let column = |ts: u64| {
        let offset = u128::from(ts.clamp(first, last) - first);
        usize::try_from(offset * bar_chars as u128 / u128::from(total)).unwrap_or(bar_chars)
    };
    let begin = column(node.start_us()).min(bar_chars - 1);
    let end = column(node.end_us().unwrap_or(last)).clamp(begin + 1, bar_chars);
    let label: String = (indent(depth) + &escape(node.name()))
        .chars()
        .take(LABEL_CHARS)
        .collect();
    let mut row = format!("{label:<LABEL_CHARS$} |{}", " ".repeat(begin));
    paint(
        &mut row,
        &"#".repeat(end - begin),
        status(node).1,
        options.color,
    );
    let _ = write!(row, "{}|", " ".repeat(bar_chars - end));
    row
}

fn status(node: &Node) -> (&'static str, &'static str) {
    match (node.kind(), node.ok()) {
        (NodeKind::Event, _) => ("event", GREEN),
        (NodeKind::Span, None) => ("open", YELLOW),
        (NodeKind::Span, Some(true)) => ("ok", GREEN),
        (NodeKind::Span, Some(false)) => ("err", RED),
    }
}

fn paint(out: &mut String, text: &str, code: &str, color: bool) {
    if color {
        let _ = write!(out, "{code}{text}{RESET}");
    } else {
        out.push_str(text);
    }
}

fn indent(depth: usize) -> String {
    "  ".repeat(depth.min(INDENT_LEVELS_MAX))
}

fn duration(dur_us: u64) -> String {
    match dur_us {
        0..=999 => format!("{dur_us}us"),
        1_000..=999_999 => format!("{}.{:03}ms", dur_us / 1_000, dur_us % 1_000),
        _ => format!("{}.{:03}s", dur_us / 1_000_000, dur_us / 1_000 % 1_000),
    }
}

fn field_text(value: &FieldValue) -> String {
    match value {
        FieldValue::Str(text) => escape(cut(text)),
        FieldValue::Int(n) => n.to_string(),
        FieldValue::UInt(n) => n.to_string(),
        FieldValue::Float(n) => n.to_string(),
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Null => "null".to_owned(),
        FieldValue::Withheld(bytes) => format!("<{bytes} bytes>"),
        _ => "[..]".to_owned(),
    }
}

fn cut(text: &str) -> &str {
    match text.char_indices().nth(VALUE_CHARS_MAX) {
        Some((at, _)) => &text[..at],
        None => text,
    }
}

/// Returns `text` with every control character written out: C0 and DEL
/// as `\xNN`, C1 as `\u{NN}`, so traced text cannot move the cursor.
/// Input: any text. Output: text with no control characters.
pub(crate) fn escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.chars() {
        let code = u32::from(ch);
        if code < 0x20 || code == 0x7f {
            let _ = write!(out, "\\x{code:02x}");
        } else if (0x80..0xa0).contains(&code) {
            let _ = write!(out, "\\u{{{code:x}}}");
        } else {
            out.push(ch);
        }
    }
    assert!(!out.chars().any(char::is_control));
    out
}
