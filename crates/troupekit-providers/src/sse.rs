use crate::status::decode_error;
use troupekit_core::error::ProviderError;

/// Longest line, in bytes, the parser buffers before failing with `Decode`.
/// Bounds memory when a server never sends a newline.
/// Applies to the bytes buffered for one line, not the whole stream.
pub const MAX_LINE_BYTES: usize = 16 * 1024 * 1024;

/// One server-sent event: its optional `event:` name and its `data:`
/// lines joined with newlines. Events without data are never produced.
/// `SseParser` yields these; `Decoder::sse` turns them into deltas.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    pub event: Option<String>,
    pub data: String,
}

/// Splits a server-sent-events byte stream into events, for every
/// streaming protocol. Chunks may break anywhere, even inside a line or
/// a UTF-8 character; comments and unknown fields are skipped.
#[derive(Debug, Default)]
pub struct SseParser {
    buf: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
}

impl SseParser {
    /// Starts a parser for one stream.
    /// Returns the parser.
    /// Feed it bytes with `push`, then call `finish` once.
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds the next `chunk` of the stream.
    /// Returns the events it completed, or `Decode` for a line that is
    /// not UTF-8 or grows past `MAX_LINE_BYTES`.
    pub fn push(&mut self, chunk: &[u8]) -> Result<Vec<SseEvent>, ProviderError> {
        let mut out = Vec::new();
        let mut search = self.buf.len();
        self.buf.extend_from_slice(chunk);
        let mut start = 0;
        while let Some(offset) = self.buf[search..].iter().position(|b| *b == b'\n') {
            let end = search + offset;
            let line = line_text(&self.buf[start..end])?;
            self.line(&line, &mut out);
            start = end + 1;
            search = start;
        }
        self.buf.drain(..start);
        if self.buf.len() > MAX_LINE_BYTES {
            return Err(decode_error("event stream line is too long"));
        }
        Ok(out)
    }

    /// Ends the stream, completing a last event that had no blank line.
    /// Returns the remaining events, or `Decode` for a bad last line.
    /// Consumes the parser, so no bytes can follow.
    pub fn finish(mut self) -> Result<Vec<SseEvent>, ProviderError> {
        let mut out = Vec::new();
        if !self.buf.is_empty() {
            let rest = std::mem::take(&mut self.buf);
            let line = line_text(&rest)?;
            self.line(&line, &mut out);
        }
        self.line("", &mut out);
        Ok(out)
    }

    fn line(&mut self, line: &str, out: &mut Vec<SseEvent>) {
        if line.is_empty() {
            if !self.data.is_empty() {
                out.push(SseEvent {
                    event: self.event.take(),
                    data: self.data.join("\n"),
                });
                self.data.clear();
            }
            self.event = None;
            return;
        }
        if line.starts_with(':') {
            return;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => self.data.push(value.to_owned()),
            "event" => self.event = Some(value.to_owned()),
            _ => {}
        }
    }
}

fn line_text(raw: &[u8]) -> Result<String, ProviderError> {
    let raw = raw.strip_suffix(b"\r").unwrap_or(raw);
    std::str::from_utf8(raw)
        .map(str::to_owned)
        .map_err(|_| decode_error("event stream line is not UTF-8"))
}
