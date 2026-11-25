/// Reads the value of the Rust string literal token `lit`.
/// Accepts `"..."` with every escape form and line continuation, and raw
/// `r"..."` or `r#"..."#` with any hash count. Byte strings, C strings,
/// suffixes and non-strings are refused with a message.
pub fn unescape(lit: &str) -> Result<String, String> {
    if lit.starts_with('b') || lit.starts_with('c') {
        return Err("byte strings and C strings are not supported".into());
    }
    if let Some(rest) = lit.strip_prefix('r') {
        return raw(rest);
    }
    let Some(body) = lit.strip_prefix('"').and_then(|s| s.strip_suffix('"')) else {
        return Err("expected a string literal".into());
    };
    escaped(body)
}

/// Writes `text` as a Rust string literal that reads back as `text`.
/// Uses the standard `Debug` escaping, so quotes, backslashes, control
/// characters and `"#` never end the literal early.
/// Returns the literal source text, quotes included.
pub fn quote_str(text: &str) -> String {
    format!("{text:?}")
}

fn raw(rest: &str) -> Result<String, String> {
    let hashes = rest.bytes().take_while(|&b| b == b'#').count();
    let fence = "#".repeat(hashes);
    let body = rest[hashes..]
        .strip_prefix('"')
        .and_then(|s| s.strip_suffix(fence.as_str()))
        .and_then(|s| s.strip_suffix('"'));
    match body {
        Some(body) => Ok(body.to_string()),
        None => Err("expected a string literal".into()),
    }
}

fn escaped(body: &str) -> Result<String, String> {
    let mut out = String::with_capacity(body.len());
    let mut chars = body.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '"' {
            return Err("expected a string literal".into());
        }
        if c != '\\' {
            out.push(c);
            continue;
        }
        let Some(e) = chars.next() else {
            return Err("string literal ends in a backslash".into());
        };
        match e {
            'n' => out.push('\n'),
            'r' => out.push('\r'),
            't' => out.push('\t'),
            '\\' => out.push('\\'),
            '0' => out.push('\0'),
            '\'' => out.push('\''),
            '"' => out.push('"'),
            'x' => {
                let hex: String = chars.by_ref().take(2).collect();
                let value = u8::from_str_radix(&hex, 16)
                    .ok()
                    .filter(|v| hex.len() == 2 && *v <= 0x7f)
                    .ok_or_else(|| format!("invalid escape \\x{hex}"))?;
                out.push(char::from(value));
            }
            'u' => {
                if chars.next() != Some('{') {
                    return Err("invalid unicode escape".into());
                }
                let mut hex = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some('_') => {}
                        Some(h) if h.is_ascii_hexdigit() && hex.len() < 6 => hex.push(h),
                        _ => return Err("invalid unicode escape".into()),
                    }
                }
                let value = u32::from_str_radix(&hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .ok_or_else(|| format!("invalid unicode escape \\u{{{hex}}}"))?;
                out.push(value);
            }
            '\n' | '\r' => while chars.next_if(|c| c.is_ascii_whitespace()).is_some() {},
            other => return Err(format!("unknown escape \\{other}")),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unescape_every_escape_form() {
        let lit = r#""a\nb\rc\td\\e\0f\'g\"h\x41\x7f\u{48}\u{1F600}\u{0}""#;
        assert_eq!(
            unescape(lit).unwrap(),
            "a\nb\rc\td\\e\0f'g\"hA\x7fH\u{1F600}\0"
        );
        assert_eq!(unescape("\"one \\\n      two\"").unwrap(), "one two");
        assert_eq!(unescape("\"plain é\"").unwrap(), "plain é");
        assert!(unescape(r#""\x80""#).is_err());
        assert!(unescape(r#""\u{D800}""#).is_err());
        assert!(unescape(r#""\q""#).is_err());
        assert!(unescape("\"x\"suffix").is_err());
        assert!(unescape("42").is_err());
    }

    #[test]
    fn unescape_raw_strings_any_hash_count() {
        assert_eq!(unescape(r#"r"a\nb""#).unwrap(), r"a\nb");
        assert_eq!(unescape(r##"r#"say "hi""#"##).unwrap(), r#"say "hi""#);
        assert_eq!(unescape(r###"r##"a "# b"##"###).unwrap(), r##"a "# b"##);
        assert!(unescape(r##"r#"open""##).is_err());
    }

    #[test]
    fn unescape_refuses_byte_and_c_strings() {
        for lit in [r#"b"x""#, r##"br#"x"#"##, r#"c"x""#, r#"cr"x""#] {
            assert_eq!(
                unescape(lit).unwrap_err(),
                "byte strings and C strings are not supported"
            );
        }
    }

    #[test]
    fn quote_str_round_trips_hostile_text() {
        for text in [
            "",
            "say \"hi\"",
            "back\\slash",
            "raw \"# end",
            "\"\"\" triple",
            "tab\tnew\nline\r",
            "non-ASCII é 日本 \u{1F600}",
            "nul \u{0} and \u{7f}",
            "'''",
        ] {
            assert_eq!(unescape(&quote_str(text)).unwrap(), text, "{text:?}");
        }
    }
}
