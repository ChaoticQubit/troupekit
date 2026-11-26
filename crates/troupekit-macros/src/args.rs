use crate::lit::unescape;
use crate::tokens::{Error, SpanId, Tok};

/// One `key = value` entry of a macro attribute.
/// `value` holds every token up to the next top-level comma.
/// `key_span` aims errors about the key itself.
#[derive(Clone, Debug)]
pub struct Arg {
    pub key: String,
    pub key_span: SpanId,
    pub value: Vec<Tok>,
}

impl Arg {
    /// Returns the span errors about the value point at: its first
    /// token, or the key when the value is empty.
    /// Never fails.
    pub fn value_span(&self) -> SpanId {
        self.value.first().map_or(self.key_span, Tok::span)
    }

    /// Reads the value as one string literal for macro `what`.
    /// Returns the text, or `<what> key `k` takes a string literal`.
    /// Escapes are resolved.
    pub fn string(&self, what: &str) -> Result<String, Error> {
        match self.value.as_slice() {
            [Tok::Lit(lit, span)] => unescape(lit).map_err(|_| self.not_string(what, *span)),
            _ => Err(self.not_string(what, self.value_span())),
        }
    }

    fn not_string(&self, what: &str, span: SpanId) -> Error {
        Error::new(
            span,
            format!("{what} key `{}` takes a string literal", self.key),
        )
    }
}

/// Parses attribute tokens `toks` of macro `what` as `key = value, ...`
/// with an optional trailing comma. Refuses a missing `=`, an empty value
/// and a key set twice. Returns the entries in order.
pub fn parse_args(toks: &[Tok], what: &str) -> Result<Vec<Arg>, Error> {
    let mut args: Vec<Arg> = Vec::new();
    for entry in split_top_level(toks) {
        let Some(first) = entry.first() else {
            continue;
        };
        let Tok::Ident(key, key_span) = first else {
            return Err(Error::new(
                first.span(),
                format!("{what} takes key = value entries"),
            ));
        };
        let key = key.strip_prefix("r#").unwrap_or(key).to_string();
        if !entry.get(1).is_some_and(|t| t.is_punct('=')) {
            return Err(Error::new(
                *key_span,
                format!("{what} key `{key}` needs = and a value"),
            ));
        }
        if entry.len() == 2 {
            return Err(Error::new(
                *key_span,
                format!("{what} key `{key}` needs a value"),
            ));
        }
        if args.iter().any(|a| a.key == key) {
            return Err(Error::new(
                *key_span,
                format!("{what} key `{key}` is set twice"),
            ));
        }
        args.push(Arg {
            key,
            key_span: *key_span,
            value: entry[2..].to_vec(),
        });
    }
    Ok(args)
}

/// Splits `toks` at top-level commas. Commas inside groups never split;
/// `<` and `>` are not tracked, so a generic with two types needs
/// parentheses in an expression value. Returns the pieces, empty included.
pub fn split_top_level(toks: &[Tok]) -> Vec<Vec<Tok>> {
    let mut out = vec![Vec::new()];
    for tok in toks {
        if tok.is_punct(',') {
            out.push(Vec::new());
        } else {
            out.last_mut()
                .expect("out starts non-empty")
                .push(tok.clone());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::lex;

    #[test]
    fn parses_entries_and_trailing_comma() {
        let args = parse_args(&lex("name = \"add\", crate = \"::kit\","), "#[tool]").unwrap();
        assert_eq!(args.len(), 2);
        assert_eq!(args[0].key, "name");
        assert_eq!(args[0].string("#[tool]").unwrap(), "add");
        assert_eq!(args[1].key, "crate");
        assert!(parse_args(&[], "#[tool]").unwrap().is_empty());
    }

    #[test]
    fn rejects_missing_equals_empty_value_and_duplicates() {
        let what = "#[tool]";
        let e = parse_args(&lex("name \"x\""), what).unwrap_err();
        assert_eq!(e.message, "#[tool] key `name` needs = and a value");
        let e = parse_args(&lex("name ="), what).unwrap_err();
        assert_eq!(e.message, "#[tool] key `name` needs a value");
        let toks = lex("name = \"a\", name = \"b\"");
        let e = parse_args(&toks, what).unwrap_err();
        assert_eq!(e.message, "#[tool] key `name` is set twice");
        assert_eq!(e.span, toks[4].span());
        let e = parse_args(&lex("\"x\" = 1"), what).unwrap_err();
        assert_eq!(e.message, "#[tool] takes key = value entries");
    }

    #[test]
    fn string_values_refuse_other_tokens() {
        let args = parse_args(&lex("name = add"), "#[tool]").unwrap();
        let e = args[0].string("#[tool]").unwrap_err();
        assert_eq!(e.message, "#[tool] key `name` takes a string literal");
        assert_eq!(e.span, args[0].value[0].span());
        let args = parse_args(&lex("name = b\"x\""), "#[tool]").unwrap();
        assert!(args[0].string("#[tool]").is_err());
    }
}
