use std::fmt::Write;

/// Appends `text` to `out` as a JSON string literal with quotes.
/// Escapes `"`, `\`, every control character as `\u00XX`, and U+2028 and
/// U+2029, so a line is safe in JSON, JSON lines and inline script.
pub(crate) fn push_str(out: &mut String, text: &str) {
    let len_before = out.len();
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{2028}' => out.push_str("\\u2028"),
            '\u{2029}' => out.push_str("\\u2029"),
            c if u32::from(c) < 0x20 || c == '\u{7f}' => {
                let _ = write!(out, "\\u{:04x}", u32::from(c));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    assert!(out.len() >= len_before + text.len() + 2);
}

/// Appends `value` as a JSON number, or `null` when it is not finite.
/// Input: the output buffer and value. Output: none.
pub(crate) fn push_f64(out: &mut String, value: f64) {
    if value.is_finite() {
        let _ = write!(out, "{value}");
    } else {
        out.push_str("null");
    }
}

/// Appends `value` as a JSON number. Input: buffer and value. Output: none.
pub(crate) fn push_u64(out: &mut String, value: u64) {
    let _ = write!(out, "{value}");
}

/// Appends `value` as a JSON number. Input: buffer and value. Output: none.
pub(crate) fn push_i64(out: &mut String, value: i64) {
    let _ = write!(out, "{value}");
}

/// Appends `"key":` to `out`, with a leading comma unless `first`.
/// Input: buffer, key and position flag. Output: none.
pub(crate) fn push_key(out: &mut String, key: &str, first: bool) {
    if !first {
        out.push(',');
    }
    push_str(out, key);
    out.push(':');
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quoted(text: &str) -> String {
        let mut out = String::new();
        push_str(&mut out, text);
        out
    }

    #[test]
    fn escapes_every_special_character() {
        assert_eq!(quoted("a\"b\\c"), r#""a\"b\\c""#);
        assert_eq!(
            quoted("\n\t\u{0}\u{1f}\u{7f}"),
            r#""\u000a\u0009\u0000\u001f\u007f""#
        );
        assert_eq!(quoted("\u{2028}\u{2029}é"), "\"\\u2028\\u2029é\"");
    }

    #[test]
    fn non_finite_floats_are_null() {
        let mut out = String::new();
        push_f64(&mut out, f64::NAN);
        push_f64(&mut out, 1.5);
        assert_eq!(out, "null1.5");
    }
}
