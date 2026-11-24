use crate::error::ContextError;
use crate::text::CHARS_PER_TOKEN;

const PARAGRAPH_SEP: &str = "\n\n";
const SPACE_SEP: &str = " ";
const NO_SEP: &str = "";

/// Size limits for `chunk`, in estimated tokens of four chars each.
/// `max_tokens` bounds every chunk; `overlap_tokens` is how much of a chunk's
/// end is repeated at the start of the next one, and must be below `max_tokens`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkConfig {
    pub max_tokens: usize,
    pub overlap_tokens: usize,
}

/// One packable piece of text and the separator that joins it to the text before.
/// `sep` is a paragraph break, a space, or nothing for a cut inside a word
/// or between sentences that had no whitespace in the source.
struct Unit {
    sep: &'static str,
    text: String,
}

/// Splits `text` into overlapping chunks of at most `cfg.max_tokens` tokens.
/// Prefers paragraph breaks, then sentence ends, then spaces, then a char cut.
/// Returns `ContextError::Invalid` for a zero limit or an overlap not below it,
/// and no chunks for blank text. The same input always gives the same chunks.
pub fn chunk(text: &str, cfg: &ChunkConfig) -> Result<Vec<String>, ContextError> {
    if cfg.max_tokens == 0 {
        return Err(ContextError::Invalid("max_tokens must be positive".into()));
    }
    if cfg.overlap_tokens >= cfg.max_tokens {
        return Err(ContextError::Invalid("overlap_tokens must be below max_tokens".into()));
    }
    let max = cfg.max_tokens.checked_mul(CHARS_PER_TOKEN).ok_or_else(|| ContextError::Invalid("max_tokens too large".into()))?;
    let overlap = cfg.overlap_tokens * CHARS_PER_TOKEN;
    let piece = max - overlap - 1;
    let mut chunks = Vec::new();
    let mut cur = String::new();
    for unit in units(text, max, piece) {
        let unit_len = char_len(&unit.text);
        if cur.is_empty() {
            cur = unit.text;
            continue;
        }
        if char_len(&cur) + unit.sep.len() + unit_len <= max {
            cur.push_str(unit.sep);
            cur.push_str(&unit.text);
            continue;
        }
        let tail = overlap_tail(&cur, overlap);
        chunks.push(std::mem::take(&mut cur));
        cur = if !tail.is_empty() && char_len(&tail) + unit.sep.len() + unit_len <= max {
            format!("{tail}{}{}", unit.sep, unit.text)
        } else {
            unit.text
        };
    }
    if !cur.is_empty() {
        chunks.push(cur);
    }
    for c in &chunks {
        assert!(!c.is_empty() && char_len(c) <= max, "chunk outside its limit");
    }
    Ok(chunks)
}

/// Counts the Unicode scalar values in `s`.
/// Used for every length check so a cut never lands inside a code point.
/// Returns the char count.
fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Breaks `text` into units no longer than `max` chars.
/// Paragraphs that fit stay whole; longer ones split into sentences, then words,
/// and a word longer than `max` is cut into windows of `piece` chars.
fn units(text: &str, max: usize, piece: usize) -> Vec<Unit> {
    let mut out = Vec::new();
    for para in paragraphs(text) {
        if char_len(&para) <= max {
            out.push(Unit { sep: PARAGRAPH_SEP, text: para });
            continue;
        }
        for (i, (sentence, sep)) in sentences(&para).into_iter().enumerate() {
            let sep = if i == 0 { PARAGRAPH_SEP } else { sep };
            if char_len(&sentence) <= max {
                out.push(Unit { sep, text: sentence });
                continue;
            }
            for (j, word) in sentence.split_whitespace().enumerate() {
                let sep = if j == 0 { sep } else { SPACE_SEP };
                if char_len(word) <= max {
                    out.push(Unit { sep, text: word.to_string() });
                    continue;
                }
                let chars: Vec<char> = word.chars().collect();
                for (k, window) in chars.chunks(piece).enumerate() {
                    out.push(Unit { sep: if k == 0 { sep } else { NO_SEP }, text: window.iter().collect() });
                }
            }
        }
    }
    out
}

/// Groups the lines of `text` into paragraphs separated by blank lines.
/// Lines inside a paragraph keep their newlines; edges are trimmed.
/// Returns the non-empty paragraphs in order.
fn paragraphs(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut lines: Vec<&str> = Vec::new();
    for line in text.lines().chain(std::iter::once("")) {
        if line.trim().is_empty() {
            let para = lines.join("\n").trim().to_string();
            if !para.is_empty() {
                out.push(para);
            }
            lines.clear();
        } else {
            lines.push(line.trim_end());
        }
    }
    out
}

/// Splits a paragraph into sentences with the separator that preceded each.
/// A sentence ends at `.`, `!` or `?` before whitespace or the end, and always
/// at `。`, `！` or `？`. The separator is a space when whitespace followed the end.
fn sentences(para: &str) -> Vec<(String, &'static str)> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut sep = NO_SEP;
    let mut chars = para.chars().peekable();
    while let Some(c) = chars.next() {
        buf.push(c);
        let next = chars.peek().copied();
        let ends = matches!(c, '。' | '！' | '？') || (matches!(c, '.' | '!' | '?') && next.is_none_or(char::is_whitespace));
        if ends {
            let sentence = buf.trim().to_string();
            if !sentence.is_empty() {
                out.push((sentence, sep));
            }
            buf.clear();
            sep = if next.is_some_and(char::is_whitespace) { SPACE_SEP } else { NO_SEP };
        }
    }
    let rest = buf.trim().to_string();
    if !rest.is_empty() {
        out.push((rest, sep));
    }
    out
}

/// Takes the last `overlap` chars of `chunk` to repeat at the start of the next.
/// When the cut lands inside the chunk and the tail holds whitespace, the tail
/// starts after its first whitespace so it begins on a word. Returns "" for 0.
fn overlap_tail(chunk: &str, overlap: usize) -> String {
    if overlap == 0 {
        return String::new();
    }
    let start = char_len(chunk).saturating_sub(overlap);
    let tail: String = chunk.chars().skip(start).collect();
    match tail.find(char::is_whitespace) {
        Some(i) if start > 0 => tail[i..].trim_start().to_string(),
        _ => tail,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(max_tokens: usize, overlap_tokens: usize) -> ChunkConfig {
        ChunkConfig { max_tokens, overlap_tokens }
    }

    fn len(s: &str) -> usize {
        s.chars().count()
    }

    fn shares_overlap(prev: &str, next: &str) -> bool {
        let next: Vec<char> = next.chars().collect();
        (1..=next.len()).any(|k| prev.ends_with(&next[..k].iter().collect::<String>()))
    }

    #[test]
    fn empty_and_blank_give_no_chunks() {
        assert!(chunk("", &cfg(10, 2)).unwrap().is_empty());
        assert!(chunk(" \n\n \t", &cfg(10, 2)).unwrap().is_empty());
    }

    #[test]
    fn rejects_bad_config() {
        assert!(matches!(chunk("a", &cfg(0, 0)), Err(ContextError::Invalid(_))));
        assert!(matches!(chunk("a", &cfg(5, 5)), Err(ContextError::Invalid(_))));
        assert!(matches!(chunk("a", &cfg(5, 9)), Err(ContextError::Invalid(_))));
    }

    #[test]
    fn small_paragraphs_pack_into_one_chunk() {
        assert_eq!(chunk("a b.\n\n  \nc d.", &cfg(100, 10)).unwrap(), vec!["a b.\n\nc d."]);
    }

    #[test]
    fn splits_long_paragraph_on_sentences_with_overlap() {
        let sentences: Vec<String> = (0..10).map(|i| format!("Sentence number {i} talks about protein intake.")).collect();
        let text = sentences.join(" ");
        let chunks = chunk(&text, &cfg(25, 5)).unwrap();
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert!(len(c) <= 100, "chunk too long: {c}");
        }
        for pair in chunks.windows(2) {
            assert!(shares_overlap(&pair[0], &pair[1]), "no overlap between {:?} and {:?}", pair[0], pair[1]);
        }
        for s in &sentences {
            assert!(chunks.iter().any(|c| c.contains(s.as_str())), "lost sentence {s}");
        }
    }

    #[test]
    fn huge_paragraph_without_boundaries_is_hard_split() {
        let text: String = (0..5000).map(|i| char::from(b'0' + (i % 7) as u8)).collect();
        let chunks = chunk(&text, &cfg(50, 10)).unwrap();
        assert!(chunks.len() > 1);
        let mut rebuilt = chunks[0].clone();
        for c in &chunks {
            assert!(len(c) <= 200);
        }
        for pair in chunks.windows(2) {
            let tail: String = pair[0].chars().skip(len(&pair[0]) - 40).collect();
            assert!(pair[1].starts_with(&tail));
            rebuilt.push_str(&pair[1][tail.len()..]);
        }
        assert_eq!(rebuilt, text);
    }

    #[test]
    fn long_sentence_with_spaces_splits_on_words() {
        let text = vec!["word"; 300].join(" ");
        let chunks = chunk(&text, &cfg(20, 4)).unwrap();
        for c in &chunks {
            assert!(len(c) <= 80);
            assert!(c.split(' ').all(|w| w == "word"), "cut inside a word: {c}");
        }
    }

    #[test]
    fn unicode_never_splits_code_points() {
        let text = format!("{}\n\n{}", "日本語の文章です。".repeat(300), "😀".repeat(500));
        let chunks = chunk(&text, &cfg(30, 5)).unwrap();
        for c in &chunks {
            assert!(len(c) <= 120);
        }
        let cjk: Vec<&String> = chunks.iter().filter(|c| c.contains('日') && !c.contains('😀')).collect();
        assert!(cjk.len() > 1);
        for c in &cjk {
            assert!(c.ends_with('。'), "sentence cut: {c}");
            assert!(!c.contains(' '));
        }
        let emoji: usize = chunks.iter().map(|c| c.matches('😀').count()).sum();
        assert!(emoji >= 500);
    }

    #[test]
    fn deterministic() {
        let text = "One. Two! Three?\n\nFour five six. ".repeat(40);
        assert_eq!(chunk(&text, &cfg(12, 3)).unwrap(), chunk(&text, &cfg(12, 3)).unwrap());
    }
}
