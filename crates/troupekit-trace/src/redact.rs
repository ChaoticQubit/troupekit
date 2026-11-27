use crate::value::{cut_to, Value};

pub(crate) const FIELD_BYTES_MAX: usize = 16 * 1024;
pub(crate) const FIELDS_MAX: usize = 32;
pub(crate) const NAME_BYTES_MAX: usize = 64;
const REDACTED: &str = "[redacted]";
const DENIED_NAMES: [&str; 7] = [
    "authorization",
    "api_key",
    "apikey",
    "token",
    "secret",
    "password",
    "cookie",
];

/// A field value after redaction, ready for the JSON writer.
/// `Withheld` is user content written only as its byte count.
pub(crate) enum Clean {
    Str(String),
    Int(i64),
    UInt(u64),
    Float(f64),
    Bool(bool),
    Withheld(u64),
}

/// What redaction needs from the scope: secrets and the content flag.
/// Input: built by the record writer per record.
pub(crate) struct Policy<'a> {
    pub(crate) secrets: &'a [String],
    pub(crate) content: bool,
}

/// Replaces every occurrence of every secret in `text` with `[redacted]`.
/// Matches are found in the original text and their union is replaced,
/// so overlapping secrets and the marker itself never leave a fragment.
/// Input: the text and non-empty secrets. Output: the scrubbed text.
pub(crate) fn scrub(text: &str, secrets: &[String]) -> String {
    let mut ranges: Vec<(usize, usize)> = Vec::new();
    for secret in secrets {
        assert!(!secret.is_empty(), "an empty secret would match everywhere");
        let mut from = 0;
        while let Some(at) = text[from..].find(secret.as_str()) {
            let start = from + at;
            ranges.push((start, start + secret.len()));
            from = start + text[start..].chars().next().map_or(1, char::len_utf8);
        }
    }
    if ranges.is_empty() {
        return text.to_owned();
    }
    ranges.sort_unstable();
    let mut out = String::with_capacity(text.len());
    let mut copied = 0;
    let mut i = 0;
    while i < ranges.len() {
        let (start, mut end) = ranges[i];
        i += 1;
        while i < ranges.len() && ranges[i].0 <= end {
            end = end.max(ranges[i].1);
            i += 1;
        }
        assert!(copied <= start && end <= text.len());
        out.push_str(&text[copied..start]);
        out.push_str(REDACTED);
        copied = end;
    }
    out.push_str(&text[copied..]);
    out
}

/// Returns whether a field named `name` must never show a string value.
/// ASCII case-insensitive, and `-` counts as `_`, so `X-Api-Key` matches.
/// Input: the field name. Output: true when denied.
pub(crate) fn is_denied(name: &str) -> bool {
    let folded: String = name
        .chars()
        .map(|c| {
            if c == '-' {
                '_'
            } else {
                c.to_ascii_lowercase()
            }
        })
        .collect();
    assert_eq!(folded.len(), name.len());
    DENIED_NAMES.iter().any(|denied| folded.contains(denied))
}

/// Cuts `text` past 16 KiB at a char boundary and appends `...[cut N bytes]`.
/// Input: the text. Output: the text and whether it was cut.
pub(crate) fn cut(text: String) -> (String, bool) {
    if text.len() <= FIELD_BYTES_MAX {
        return (text, false);
    }
    let kept = cut_to(&text, FIELD_BYTES_MAX);
    let removed = text.len() - kept.len();
    assert!(removed > 0);
    (format!("{kept}...[cut {removed} bytes]"), true)
}

/// Returns a field name scrubbed and cut to 64 bytes.
/// Input: the raw name and secrets. Output: the clean name.
pub(crate) fn name(raw: &str, secrets: &[String]) -> String {
    let scrubbed = scrub(raw, secrets);
    let clean = cut_to(&scrubbed, NAME_BYTES_MAX).to_owned();
    assert!(clean.len() <= NAME_BYTES_MAX);
    clean
}

/// Redacts one field value in the fixed order: scrub secrets, deny by
/// field name, withhold content, then cut long strings.
/// Input: field name, value and policy. Output: the clean value and cut flag.
pub(crate) fn value(name: &str, value: Value, policy: &Policy<'_>) -> (Clean, bool) {
    let text = match value {
        Value::Int(n) => return (Clean::Int(n), false),
        Value::UInt(n) => return (Clean::UInt(n), false),
        Value::Float(n) => return (Clean::Float(n), false),
        Value::Bool(b) => return (Clean::Bool(b), false),
        Value::Str(text) => text,
        Value::Content(text) if policy.content || is_denied(name) => text,
        Value::Content(text) => return (Clean::Withheld(text.len() as u64), false),
    };
    if is_denied(name) {
        return (Clean::Str(REDACTED.to_owned()), false);
    }
    let (text, was_cut) = cut(scrub(&text, policy.secrets));
    (Clean::Str(text), was_cut)
}

/// Returns `url` with only scheme, host, port and path: user info, query
/// and fragment are dropped. Like URL parsers, any run of `/` or `\\`
/// after `scheme:` leads to the authority, which ends at `/` or `\\`.
/// Use it for every URL written to a trace. Output: the reduced URL.
pub fn url_for_trace(url: &str) -> String {
    let end = url.find(['?', '#']).unwrap_or(url.len());
    let url = &url[..end];
    let (scheme, rest) = match split_scheme(url) {
        Some((scheme, rest)) => (scheme, rest.trim_start_matches(['/', '\\'])),
        None => ("", url),
    };
    let authority_end = rest.find(['/', '\\']).unwrap_or(rest.len());
    let (authority, path) = rest.split_at(authority_end);
    let host = authority
        .rsplit_once('@')
        .map_or(authority, |(_, host)| host);
    let out = match scheme {
        "" => format!("{host}{path}"),
        scheme => format!("{scheme}//{host}{path}"),
    };
    assert!(!host.contains('@'));
    assert!(!out.contains(['?', '#']));
    out
}

/// Splits `url` into `scheme:` and the rest when it starts with a scheme
/// (a letter, then letters, digits, `+`, `-` or `.`) followed by `:` and
/// a `/` or `\\`. Output: the two parts, or `None` for any other text.
fn split_scheme(url: &str) -> Option<(&str, &str)> {
    let colon = url.find(':')?;
    let name = &url[..colon];
    let valid = name.chars().next().is_some_and(|c| c.is_ascii_alphabetic())
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    let rest = &url[colon + 1..];
    let slashed = rest.starts_with(['/', '\\']);
    (valid && slashed).then(|| url.split_at(colon + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlapping_secrets_are_redacted_against_the_original_text() {
        let secrets = ["abc".to_owned(), "abc-private".to_owned(), "c-pri".to_owned()];
        assert_eq!(scrub("x abc-private y", &secrets), "x [redacted] y");
        let shifted = ["bcd".to_owned(), "abc".to_owned()];
        assert_eq!(scrub("abcde", &shifted), "[redacted]e");
        let marker = ["key1".to_owned(), "dact".to_owned()];
        assert_eq!(scrub("key1 key1", &marker), "[redacted] [redacted]");
        assert_eq!(scrub("no match", &secrets), "no match");
    }

    #[test]
    #[should_panic(expected = "empty secret")]
    fn empty_secret_is_a_bug() {
        scrub("text", &[String::new()]);
    }

    #[test]
    fn numbers_under_denied_names_stay() {
        let policy = Policy {
            secrets: &[],
            content: false,
        };
        assert!(matches!(
            value("tokens", Value::UInt(5), &policy).0,
            Clean::UInt(5)
        ));
        assert!(
            matches!(value("tokens", Value::from("5"), &policy).0, Clean::Str(s) if s == REDACTED)
        );
    }
}
