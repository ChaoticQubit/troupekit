use troupekit_core::error::ProviderErrorKind;
use troupekit_providers::sse::{SseEvent, SseParser, MAX_LINE_BYTES};

fn ev(event: Option<&str>, data: &str) -> SseEvent {
    SseEvent {
        event: event.map(String::from),
        data: data.into(),
    }
}

fn parse_in_chunks(bytes: &[u8], size: usize) -> Vec<SseEvent> {
    let mut parser = SseParser::new();
    let mut out = Vec::new();
    for chunk in bytes.chunks(size) {
        out.extend(parser.push(chunk).unwrap());
    }
    out.extend(parser.finish().unwrap());
    out
}

#[test]
fn same_events_for_every_chunk_size() {
    let text = "event: a\ndata: {\"t\":\"h\u{e9}llo \u{1f600}\"}\n\n: comment\n\ndata: x\r\ndata: y\r\n\r\nevent: ping\n\ndata:tight\n\n";
    let want = vec![
        ev(Some("a"), "{\"t\":\"h\u{e9}llo \u{1f600}\"}"),
        ev(None, "x\ny"),
        ev(None, "tight"),
    ];
    for size in 1..=text.len() {
        assert_eq!(
            parse_in_chunks(text.as_bytes(), size),
            want,
            "chunk size {size}"
        );
    }
}

#[test]
fn finish_flushes_an_unterminated_event() {
    assert_eq!(parse_in_chunks(b"data: last", 3), vec![ev(None, "last")]);
}

#[test]
fn rejects_invalid_utf8_and_huge_lines() {
    let mut parser = SseParser::new();
    assert_eq!(
        parser.push(b"data: \xff\n").unwrap_err().kind,
        ProviderErrorKind::Decode
    );
    let mut parser = SseParser::new();
    assert_eq!(
        parser
            .push(&vec![b'a'; MAX_LINE_BYTES + 1])
            .unwrap_err()
            .kind,
        ProviderErrorKind::Decode
    );
}
