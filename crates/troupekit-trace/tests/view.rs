use crate::support::lock;
use std::io::Cursor;
use std::sync::Arc;
use troupekit_trace::view::{
    read_jsonl, render_terminal, FieldValue, NodeKind, SpanTree, TermOptions, ViewLimits,
};
use troupekit_trace::{content, trace, trace_span, MemorySink, Trace, TraceSet, Tracer};

fn read(text: &str, limits: ViewLimits) -> SpanTree {
    read_jsonl(Cursor::new(text.as_bytes()), limits).unwrap()
}

fn start(span: u64, parent: u64, name: &str, ts: u64) -> String {
    format!(
        "{{\"v\":1,\"t\":\"start\",\"trace\":\"00000000000000ab\",\"span\":{span},\"parent\":{parent},\"cat\":\"agents\",\"name\":\"{name}\",\"ts_us\":{ts}}}\n"
    )
}

fn end(span: u64, ts: u64, dur: u64) -> String {
    format!(
        "{{\"v\":1,\"t\":\"end\",\"trace\":\"00000000000000ab\",\"span\":{span},\"ts_us\":{ts},\"dur_us\":{dur},\"ok\":true}}\n"
    )
}

#[test]
fn reads_its_own_writer_output() {
    let _guard = lock();
    let sink = Arc::new(MemorySink::default());
    let t = Tracer::builder()
        .categories(TraceSet::all())
        .sink(sink.clone())
        .name("demo")
        .build()
        .unwrap();
    t.enter(|| {
        let run = trace_span!(Trace::Agents, "agent.run", run = "r1");
        run.enter(|| {
            let mut step = trace_span!(Trace::Agents, "agent.step", step = 1u64);
            step.enter(|| {
                trace!(
                    Trace::Laws,
                    "law.check",
                    held = false,
                    prompt = content("hi")
                )
            });
            step.record("score", -2i64);
            step.fail("bad");
        });
    });
    drop(t);
    let text = sink.lines().join("\n");
    let tree = read(&text, ViewLimits::default());
    assert_eq!(tree.name(), Some("demo"));
    assert_eq!(tree.bad_lines(), 0);
    assert_eq!(tree.roots().len(), 1);
    let run = tree.node(tree.roots()[0]);
    assert_eq!(run.name(), "agent.run");
    assert_eq!(run.field("run"), Some(&FieldValue::Str("r1".to_owned())));
    assert_eq!(run.ok(), Some(true));
    let step = tree.node(run.children()[0]);
    assert_eq!(step.name(), "agent.step");
    assert_eq!(step.ok(), Some(false));
    assert_eq!(step.err(), Some("bad"));
    assert_eq!(step.field("score"), Some(&FieldValue::Int(-2)));
    let law = tree.node(step.children()[0]);
    assert_eq!(law.kind(), NodeKind::Event);
    assert_eq!(law.category(), Trace::Laws);
    assert_eq!(law.field("held"), Some(&FieldValue::Bool(false)));
    assert_eq!(law.field("prompt"), Some(&FieldValue::Withheld(2)));
    assert_eq!(tree.count(Trace::Agents), 2);
    assert_eq!(tree.count(Trace::Laws), 1);
    let (first, last) = tree.time_range_us().unwrap();
    assert!(first <= last);
}

#[test]
fn random_bytes_never_panic() {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let alphabet = b"{}[]\":,0123456789.-+eEtrufalsn\\u \t{\"v\":1,\"t\":\"start\"span}parent";
    let mut text = Vec::new();
    for line in 0..10_000u64 {
        let len = (next() % 200) as usize;
        for _ in 0..len {
            let r = next();
            let byte = if line % 2 == 0 {
                (r & 0xff) as u8
            } else {
                alphabet[(r % alphabet.len() as u64) as usize]
            };
            text.push(if byte == b'\n' { b' ' } else { byte });
        }
        text.push(b'\n');
    }
    let tree = read_jsonl(Cursor::new(text), ViewLimits::default()).unwrap();
    assert!(tree.bad_lines() > 0);
}

#[test]
fn oversized_line_is_refused_and_counted() {
    let big = format!("{{\"v\":1,\"pad\":\"{}\"}}\n", "x".repeat(1_000));
    let text = start(1, 0, "a", 10) + &big + &end(1, 20, 10);
    let tree = read(&text, ViewLimits::default().with_line_bytes_max(200));
    assert_eq!(tree.oversized_lines(), 1);
    assert_eq!(tree.roots().len(), 1);
    assert_eq!(tree.node(tree.roots()[0]).ok(), Some(true));
}

#[test]
fn deep_nesting_is_refused() {
    let nested = |depth: usize| {
        format!(
            "{{\"v\":1,\"t\":\"event\",\"trace\":\"00000000000000ab\",\"span\":0,\"parent\":0,\"cat\":\"laws\",\"name\":\"e{depth}\",\"ts_us\":1,\"fields\":{{\"x\":{}1{}}}}}\n",
            "[".repeat(depth),
            "]".repeat(depth)
        )
    };
    let tree = read(&(nested(29) + &nested(40)), ViewLimits::default());
    assert_eq!(tree.deep_lines(), 1);
    assert_eq!(tree.roots().len(), 1);
    assert_eq!(tree.node(tree.roots()[0]).name(), "e29");
}

#[test]
fn unclosed_span_is_marked_open() {
    let text = start(1, 0, "root", 10) + &start(2, 1, "child", 11) + &end(2, 15, 4);
    let tree = read(&text, ViewLimits::default());
    let root = tree.node(tree.roots()[0]);
    assert!(root.is_open());
    assert_eq!(root.ok(), None);
    assert!(!tree.node(root.children()[0]).is_open());
    assert_eq!(tree.open_spans(), 1);
}

#[test]
fn record_limit_is_enforced() {
    let text: String = (1..=10).map(|i| start(i, 0, "s", i)).collect();
    let tree = read(&text, ViewLimits::default().with_records_max(3));
    assert_eq!(tree.roots().len(), 3);
    assert!(tree.record_limit_hit());
    let all = read(&text, ViewLimits::default());
    assert_eq!(all.roots().len(), 10);
    assert!(!all.record_limit_hit());
}

const SAMPLE: &str = include_str!("data/sample.jsonl");
const SAMPLE_TERMINAL: &str = include_str!("data/sample_terminal.txt");

fn plain(width: usize) -> TermOptions {
    TermOptions::default().with_width(width).with_color(false)
}

#[test]
fn tree_lists_children_in_start_order() {
    let tree = read(SAMPLE, ViewLimits::default());
    assert_eq!(render_terminal(&tree, &plain(40)), SAMPLE_TERMINAL);
}

#[test]
fn timeline_bars_scale_to_width() {
    let tree = read(SAMPLE, ViewLimits::default());
    let out = render_terminal(&tree, &plain(60));
    let rows: Vec<&str> = out.lines().filter(|l| l.ends_with('|')).collect();
    assert_eq!(rows.len(), 3);
    for row in &rows {
        assert_eq!(row.chars().count(), 60, "{row:?}");
    }
    assert!(rows[0].ends_with(&format!("|{}|", "#".repeat(60 - 24 - 3))));
    let narrow = render_terminal(&tree, &plain(5));
    let row = narrow.lines().find(|l| l.ends_with('|')).unwrap();
    assert_eq!(row.chars().count(), 24 + 3 + 10);
}

#[test]
fn filter_keeps_only_selected_categories() {
    let tree = read(SAMPLE, ViewLimits::default());
    let only = plain(40).with_only(TraceSet::none().with(Trace::Tools));
    let out = render_terminal(&tree, &only);
    assert!(out.contains("\ntool.call [tools] open\n"), "{out}");
    assert!(!out.contains("agent.run"));
    assert!(!out.contains("law.check"));
}

#[test]
fn control_characters_are_escaped() {
    let hostile = format!(
        "{{\"v\":1,\"t\":\"start\",\"trace\":\"00000000000000ab\",\"span\":1,\"parent\":0,\"cat\":\"tools\",\"name\":\"x\\u001b[2Jy\",\"ts_us\":1,\"fields\":{{\"k\\u0007\":\"a\\rb\\u009bc\"}}}}\n"
    );
    let tree = read(&hostile, ViewLimits::default());
    let out = render_terminal(&tree, &plain(40));
    assert!(out.bytes().all(|b| b >= 0x20 || b == b'\n'), "{out:?}");
    assert!(out.contains("x\\x1b[2Jy"));
    assert!(out.contains("k\\x07=a\\x0db\\u{9b}c"));
    assert!(!out.contains('\u{9b}'));
    let colored = render_terminal(&tree, &plain(40).with_color(true));
    for (at, _) in colored.match_indices('\u{1b}') {
        let code = &colored[at..];
        assert!(["\u{1b}[0m", "\u{1b}[32m", "\u{1b}[31m", "\u{1b}[33m"]
            .iter()
            .any(|c| code.starts_with(c)));
    }
}

#[test]
fn row_cap_prints_more_line() {
    let text: String = (1..=10).map(|i| start(i, 0, "s", i)).collect();
    let tree = read(&text, ViewLimits::default());
    let out = render_terminal(&tree, &plain(40).with_max_rows(3));
    assert_eq!(out.matches("... 7 more\n").count(), 2, "{out}");
    assert_eq!(
        out.lines().filter(|l| l.starts_with("s [agents]")).count(),
        3
    );
}

#[test]
fn huge_line_limit_does_not_overflow() {
    let text = start(1, 0, "a", 1) + &end(1, 2, 1);
    let tree = read(&text, ViewLimits::default().with_line_bytes_max(usize::MAX));
    assert_eq!(tree.roots().len(), 1);
}

#[test]
fn appended_sessions_keep_both_trees_and_sum_footers() {
    let _guard = lock();
    let sink = Arc::new(MemorySink::new(1_000));
    for session in 0..2u64 {
        let tracer = Tracer::builder()
            .categories(TraceSet::all())
            .sink(sink.clone())
            .build()
            .unwrap();
        tracer.enter(|| {
            let mut span = trace_span!(Trace::Agents, "agent.step", session = session);
            for i in 0..40u64 {
                span.record("extra", i);
            }
            drop(span);
        });
    }
    let text = sink.lines().join("\n") + "\n";
    let tree = read(&text, ViewLimits::default());
    let steps: Vec<_> = tree.nodes().iter().filter(|n| n.name() == "agent.step").collect();
    assert_eq!(steps.len(), 2, "same span id in two traces stays two spans");
    assert!(steps.iter().all(|n| n.kind() == NodeKind::Span && n.end_us().is_some()));
    let (dropped, _) = tree.footer_counts().expect("footers read");
    assert_eq!(dropped, 2 * 8, "each session dropped 8 end fields past the cap");
}
