use crate::tokens::{Delim, Error, SpanId, Tok};

/// One outer attribute `#[...]`: the spans of `#` and of the bracket
/// group, for re-emitting it, and the tokens inside the brackets.
/// `name` is the first ident inside, such as `doc` or `cfg`.
#[derive(Clone, Debug)]
pub struct Attr {
    pub hash: SpanId,
    pub group: SpanId,
    pub inner: Vec<Tok>,
}

impl Attr {
    /// Reports whether the attribute's path is the single ident `name`.
    /// `#[doc = ".."]` and `#[doc(hidden)]` both have the name `doc`.
    /// Returns false for an empty attribute.
    pub fn is(&self, name: &str) -> bool {
        self.inner.first().is_some_and(|t| t.is_ident(name))
    }
}

/// A fn item split into its parts, before macro-specific checks.
/// Qualifiers are kept with their spans so each macro can refuse its
/// own; `generics` and `where_span` point at `<` and `where`.
#[derive(Clone, Debug)]
pub struct FnItem {
    pub attrs: Vec<Attr>,
    pub vis: Vec<SpanId>,
    pub quals: Vec<(String, SpanId)>,
    pub ident: String,
    pub ident_span: SpanId,
    pub generics: Option<SpanId>,
    pub params: Vec<Tok>,
    pub ret: Vec<Tok>,
    pub where_span: Option<SpanId>,
    pub body_span: SpanId,
}

/// Splits `item` into a `FnItem`, or fails with `<what> goes on a fn` at
/// the first token after the attributes when it is not a fn with a body.
/// Generic parameters are recorded, not refused.
pub fn parse_fn(item: &[Tok], what: &str) -> Result<FnItem, Error> {
    let not_fn = |t: Option<&Tok>| {
        let span = t.or(item.first()).map_or(SpanId(0), Tok::span);
        Error::new(span, format!("{what} goes on a fn"))
    };
    let mut pos = 0;
    let mut attrs = Vec::new();
    while let (Some(hash), Some(Tok::Group(Delim::Bracket, inner, group))) =
        (item.get(pos), item.get(pos + 1))
    {
        if !hash.is_punct('#') {
            break;
        }
        attrs.push(Attr {
            hash: hash.span(),
            group: *group,
            inner: inner.clone(),
        });
        pos += 2;
    }
    let mut vis = Vec::new();
    if item.get(pos).is_some_and(|t| t.is_ident("pub")) {
        vis.push(item[pos].span());
        pos += 1;
        if let Some(Tok::Group(Delim::Paren, _, span)) = item.get(pos) {
            vis.push(*span);
            pos += 1;
        }
    }
    let mut quals = Vec::new();
    while let Some(Tok::Ident(word, span)) = item.get(pos) {
        if !matches!(
            word.as_str(),
            "const" | "async" | "unsafe" | "extern" | "default"
        ) {
            break;
        }
        quals.push((word.clone(), *span));
        pos += 1;
        if word == "extern" && matches!(item.get(pos), Some(Tok::Lit(..))) {
            pos += 1;
        }
    }
    if !item.get(pos).is_some_and(|t| t.is_ident("fn")) {
        return Err(not_fn(item.get(pos)));
    }
    pos += 1;
    let Some(Tok::Ident(ident, ident_span)) = item.get(pos) else {
        return Err(not_fn(item.get(pos)));
    };
    pos += 1;
    let mut generics = None;
    if item.get(pos).is_some_and(|t| t.is_punct('<')) {
        generics = Some(item[pos].span());
        let mut depth = 0i32;
        while let Some(t) = item.get(pos) {
            pos += 1;
            if t.is_punct('<') {
                depth += 1;
            } else if t.is_punct('>') {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            }
        }
    }
    let Some(Tok::Group(Delim::Paren, params, _)) = item.get(pos) else {
        return Err(not_fn(item.get(pos)));
    };
    pos += 1;
    let Some(Tok::Group(Delim::Brace, _, body_span)) = item.last() else {
        return Err(not_fn(item.last()));
    };
    let rest = &item[pos..item.len() - 1];
    let where_at = rest.iter().position(|t| t.is_ident("where"));
    let head = &rest[..where_at.unwrap_or(rest.len())];
    let ret = match head {
        [] => Vec::new(),
        [dash, gt, ret @ ..] if dash.is_punct('-') && gt.is_punct('>') && !ret.is_empty() => {
            ret.to_vec()
        }
        [first, ..] => return Err(not_fn(Some(first))),
    };
    Ok(FnItem {
        attrs,
        vis,
        quals,
        ident: ident.clone(),
        ident_span: *ident_span,
        generics,
        params: params.clone(),
        ret,
        where_span: where_at.map(|i| rest[i].span()),
        body_span: *body_span,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::lex;

    #[test]
    fn splits_a_full_fn() {
        let toks = lex(
            "/// Doc.\n#[cfg(test)]\npub(crate) async fn add(a: i64) -> Vec<i64> where X: Y { a }",
        );
        let f = parse_fn(&toks, "#[tool]").unwrap();
        assert_eq!(f.attrs.len(), 2);
        assert!(f.attrs[0].is("doc") && f.attrs[1].is("cfg"));
        assert_eq!(f.vis.len(), 2);
        assert_eq!(f.quals[0].0, "async");
        assert_eq!(f.ident, "add");
        assert_eq!(f.ret.len(), 4);
        assert!(f.where_span.is_some());
        assert!(f.generics.is_none());
    }

    #[test]
    fn refuses_non_fns() {
        for src in [
            "struct S;",
            "const X: i32 = 1;",
            "fn f();",
            "pub fn f() -> {}",
        ] {
            let e = parse_fn(&lex(src), "#[tool]").unwrap_err();
            assert_eq!(e.message, "#[tool] goes on a fn", "{src}");
        }
        let toks = lex("#[inline] struct S;");
        let e = parse_fn(&toks, "#[tool]").unwrap_err();
        assert_eq!(e.span, toks[2].span());
    }
}
