use super::tree::{FieldValue, Node, NodeKind, SpanTree};
use crate::json::{push_key, push_str, push_u64};

const PAGE: &str = include_str!("page.html");
const SLOT: &str = "__TROUPEKIT_TRACE_DATA__";

/// Renders a span tree as one self-contained HTML page with an
/// interactive timeline. The trace sits as JSON in a data script with
/// `<`, `>` and `&` escaped, and the page script builds text nodes only.
pub fn render_html(tree: &SpanTree) -> String {
    assert_eq!(PAGE.matches(SLOT).count(), 1, "the page has one data slot");
    let data = data_json(tree);
    let safe = data
        .replace('<', "\\u003c")
        .replace('>', "\\u003e")
        .replace('&', "\\u0026");
    assert!(!safe.contains(['<', '>', '&']));
    let page = PAGE.replacen(SLOT, &safe, 1);
    assert!(page.len() >= PAGE.len() - SLOT.len());
    page
}

fn data_json(tree: &SpanTree) -> String {
    let mut out = String::with_capacity(256 + tree.nodes().len() * 160);
    out.push('{');
    push_key(&mut out, "name", true);
    push_str(&mut out, tree.name().unwrap_or("-"));
    push_key(&mut out, "trace", false);
    push_str(&mut out, tree.trace_id().unwrap_or("-"));
    let (first, last) = tree.time_range_us().unwrap_or((0, 0));
    push_key(&mut out, "range", false);
    out.push('[');
    push_u64(&mut out, first);
    out.push(',');
    push_u64(&mut out, last);
    out.push(']');
    push_key(&mut out, "open", false);
    push_u64(&mut out, tree.open_spans());
    push_key(&mut out, "bad", false);
    push_u64(
        &mut out,
        tree.bad_lines() + tree.oversized_lines() + tree.deep_lines(),
    );
    push_key(&mut out, "truncated", false);
    out.push_str(if tree.truncated() || tree.record_limit_hit() {
        "true"
    } else {
        "false"
    });
    push_key(&mut out, "roots", false);
    push_indexes(&mut out, tree.roots());
    push_key(&mut out, "nodes", false);
    out.push('[');
    for (i, node) in tree.nodes().iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_node(&mut out, node);
    }
    out.push_str("]}");
    out
}

fn push_node(out: &mut String, node: &Node) {
    out.push('{');
    push_key(out, "k", true);
    push_str(
        out,
        if node.kind() == NodeKind::Span {
            "s"
        } else {
            "e"
        },
    );
    push_key(out, "c", false);
    push_str(out, node.category().as_str());
    push_key(out, "n", false);
    push_str(out, node.name());
    push_key(out, "s", false);
    push_u64(out, node.start_us());
    push_key(out, "e", false);
    push_opt(out, node.end_us());
    push_key(out, "d", false);
    push_opt(out, node.dur_us());
    push_key(out, "ok", false);
    out.push_str(match node.ok() {
        Some(true) => "true",
        Some(false) => "false",
        None => "null",
    });
    push_key(out, "err", false);
    match node.err() {
        Some(err) => push_str(out, err),
        None => out.push_str("null"),
    }
    push_key(out, "f", false);
    out.push('[');
    for (i, (key, value)) in node.fields().iter().enumerate() {
        out.push_str(if i > 0 { ",[" } else { "[" });
        push_str(out, key);
        out.push(',');
        push_str(out, &field_text(value));
        out.push(']');
    }
    out.push(']');
    push_key(out, "ch", false);
    push_indexes(out, node.children());
    out.push('}');
}

fn push_opt(out: &mut String, value: Option<u64>) {
    match value {
        Some(n) => push_u64(out, n),
        None => out.push_str("null"),
    }
}

fn push_indexes(out: &mut String, indexes: &[usize]) {
    out.push('[');
    for (i, &index) in indexes.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        push_u64(out, index as u64);
    }
    out.push(']');
}

fn field_text(value: &FieldValue) -> String {
    match value {
        FieldValue::Str(text) => text.clone(),
        FieldValue::Int(n) => n.to_string(),
        FieldValue::UInt(n) => n.to_string(),
        FieldValue::Float(n) => n.to_string(),
        FieldValue::Bool(b) => b.to_string(),
        FieldValue::Null => "null".to_owned(),
        FieldValue::Withheld(bytes) => format!("withheld, {bytes} bytes"),
        _ => "[..]".to_owned(),
    }
}
