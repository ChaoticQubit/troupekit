use crate::lit::unescape;
use crate::tokens::{Error, Tok};

const MAX_DOC_BYTES: usize = 16 * 1024;

/// Joins the `#[doc = "..."]` attributes among `attrs` into one text.
/// Each attr is the token list inside `#[...]`. Strips one leading space
/// per line and blank lines at both ends; ignores other `doc` forms.
/// Returns `None` without doc text, an error over 16 KiB.
pub fn doc_text(attrs: &[Vec<Tok>], what: &str) -> Result<Option<String>, Error> {
    let mut lines: Vec<String> = Vec::new();
    let mut first = None;
    for attr in attrs {
        let [Tok::Ident(name, span), eq, Tok::Lit(lit, _)] = attr.as_slice() else {
            continue;
        };
        if name != "doc" || !eq.is_punct('=') {
            continue;
        }
        first.get_or_insert(*span);
        let text = unescape(lit).map_err(|e| Error::new(*span, e))?;
        for line in text.split('\n') {
            lines.push(
                line.strip_prefix(' ')
                    .unwrap_or(line)
                    .trim_end_matches('\r')
                    .to_string(),
            );
        }
    }
    let start = lines.iter().position(|l| !l.trim().is_empty());
    let end = lines.iter().rposition(|l| !l.trim().is_empty());
    let (Some(start), Some(end)) = (start, end) else {
        return Ok(None);
    };
    let text = lines[start..=end].join("\n");
    if text.len() > MAX_DOC_BYTES {
        let span = first.expect("doc lines come from a doc attribute");
        return Err(Error::new(span, format!("the {what} is over 16 KiB")));
    }
    Ok(Some(text))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::lex;

    fn attrs(src: &str) -> Vec<Vec<Tok>> {
        lex(src)
            .into_iter()
            .filter_map(|t| match t {
                Tok::Group(_, inner, _) => Some(inner),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn doc_text_joins_lines_and_strips_one_space() {
        let a = attrs("///\n/// Adds two numbers.\n///   indented\n#[doc = \"last\"]\n///\n");
        assert_eq!(
            doc_text(&a, "x").unwrap().unwrap(),
            "Adds two numbers.\n  indented\nlast"
        );
        let block = attrs("#[doc = \" one\\n two\\n\"]");
        assert_eq!(doc_text(&block, "x").unwrap().unwrap(), "one\ntwo");
    }

    #[test]
    fn doc_text_ignores_doc_hidden() {
        let a = attrs("#[doc(hidden)]\n#[inline]\n/// Kept.");
        assert_eq!(doc_text(&a, "x").unwrap().unwrap(), "Kept.");
    }

    #[test]
    fn doc_text_none_without_doc() {
        assert_eq!(doc_text(&attrs("#[inline]"), "x").unwrap(), None);
        assert_eq!(doc_text(&attrs("///\n///   "), "x").unwrap(), None);
        assert_eq!(doc_text(&[], "x").unwrap(), None);
    }

    #[test]
    fn doc_text_over_16k_is_an_error() {
        let long = format!("/// {}\n/// tail", "x".repeat(16 * 1024));
        let a = attrs(&long);
        let err = doc_text(&a, "#[tool] description").unwrap_err();
        assert_eq!(err.message, "the #[tool] description is over 16 KiB");
        assert_eq!(err.span, a[0][0].span());
        let exact = format!("/// {}", "x".repeat(16 * 1024));
        assert!(doc_text(&attrs(&exact), "x").unwrap().is_some());
    }
}
