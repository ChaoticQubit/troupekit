use crate::args::{parse_args, Arg};
use crate::doc::doc_text;
use crate::item::{parse_fn, Attr};
use crate::lit::quote_str;
use crate::tokens::{Delim, Error, Piece, SpanId, Tok};

const WHAT: &str = "#[tool]";
const MAX_ARGS: usize = 32;
const MAX_NAME_BYTES: usize = 64;
const MAX_DESCRIPTION_BYTES: usize = 16 * 1024;
const DEFAULT_CRATE: &str = "::troupekit";

/// How a `#[tool]` fn parameter is filled at call time.
/// `Arg` comes from the model's JSON arguments and is in the schema;
/// `Context` receives the call's `ToolContext`, by value or by reference.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParamKind {
    Arg,
    Context { by_ref: bool },
}

/// One parameter of a `#[tool]` fn.
/// `ident` may be raw (`r#type`); serde reads it as `type`. `ty` keeps the user's tokens
/// so rustc errors point at them. `optional` marks an `Option<..>` type.
#[derive(Clone, Debug)]
pub struct Param {
    pub ident: String,
    pub ident_span: SpanId,
    pub is_mut: bool,
    pub ty: Vec<Tok>,
    pub kind: ParamKind,
    pub optional: bool,
}

/// A validated `#[tool]` fn, ready for code generation.
/// `name` and `description` are what the model sees; `krate` is the
/// path the expansion starts from. Spans re-emit the user's code.
#[derive(Clone, Debug)]
pub struct ToolFn {
    pub cfg_attrs: Vec<Attr>,
    pub other_attrs: Vec<Attr>,
    pub vis: Vec<SpanId>,
    pub is_async: bool,
    pub ident: String,
    pub ident_span: SpanId,
    pub name: String,
    pub description: String,
    pub params: Vec<Param>,
    pub ret: Vec<Tok>,
    pub body_span: SpanId,
    pub krate: String,
}

/// Parses `#[tool(attr)]` on `item` into a `ToolFn`.
/// Every unsupported shape fails with its documented message at the
/// token that causes it. Argument types are checked later by rustc.
pub fn parse_tool(attr: &[Tok], item: &[Tok]) -> Result<ToolFn, Error> {
    let args = parse_args(attr, WHAT)?;
    for arg in &args {
        if !matches!(arg.key.as_str(), "name" | "description" | "crate") {
            return Err(Error::new(
                arg.key_span,
                format!(
                    "unknown {WHAT} key `{}`; known keys: name, description, crate",
                    arg.key
                ),
            ));
        }
    }
    let f = parse_fn(item, WHAT)?;
    let mut is_async = false;
    for (word, span) in &f.quals {
        if word == "async" {
            is_async = true;
        } else {
            return Err(Error::new(
                *span,
                format!("{WHAT} fns cannot be const, unsafe or extern"),
            ));
        }
    }
    if let Some(span) = f.generics.or(f.where_span) {
        return Err(Error::new(
            span,
            format!("{WHAT} fns cannot be generic; use concrete argument types"),
        ));
    }
    let params = parse_params(&f.params)?;
    let find = |key: &str| args.iter().find(|a| a.key == key);
    let plain_ident = f.ident.strip_prefix("r#").unwrap_or(&f.ident).to_string();
    let (name, name_span) = match find("name") {
        Some(arg) => (arg.string(WHAT)?, arg.value_span()),
        None => (plain_ident, f.ident_span),
    };
    if !valid_tool_name(&name) {
        return Err(Error::new(
            name_span,
            "tool name must be 1 to 64 of [A-Za-z0-9_-]",
        ));
    }
    let description = match find("description") {
        Some(arg) => {
            let text = arg.string(WHAT)?;
            if text.len() > MAX_DESCRIPTION_BYTES {
                return Err(Error::new(
                    arg.value_span(),
                    format!("the {WHAT} description is over 16 KiB"),
                ));
            }
            text
        }
        None => doc_text(
            &f.attrs.iter().map(|a| a.inner.clone()).collect::<Vec<_>>(),
            "#[tool] description",
        )?
        .ok_or_else(|| {
            Error::new(
                f.ident_span,
                format!("{WHAT} needs a /// doc comment or description = \"...\"; it becomes the tool description"),
            )
        })?,
    };
    let krate = match find("crate") {
        Some(arg) => crate_path(arg, WHAT)?,
        None => DEFAULT_CRATE.to_string(),
    };
    let (cfg_attrs, other_attrs) = f
        .attrs
        .into_iter()
        .filter(|a| !a.is("doc"))
        .partition(|a| a.is("cfg"));
    Ok(ToolFn {
        cfg_attrs,
        other_attrs,
        vis: f.vis,
        is_async,
        ident: f.ident,
        ident_span: f.ident_span,
        name,
        description,
        params,
        ret: f.ret,
        body_span: f.body_span,
        krate,
    })
}

/// Reads the `crate = ".."` value of macro `what` as a Rust path such as
/// `::troupekit` or `my_kit`. Refuses anything else, so the path is safe
/// to splice into generated code. Returns the path text.
pub fn crate_path(arg: &Arg, what: &str) -> Result<String, Error> {
    let text = arg.string(what)?;
    let body = text.strip_prefix("::").unwrap_or(&text);
    let valid = !body.is_empty()
        && body.split("::").all(|seg| {
            let mut chars = seg.chars();
            chars
                .next()
                .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
                && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
                && seg != "_"
        });
    if !valid {
        return Err(Error::new(
            arg.value_span(),
            format!("{what} key `crate` takes a path like \"::troupekit\""),
        ));
    }
    Ok(text)
}

/// Generates the code for a parsed `#[tool]` fn: a unit struct named
/// like the fn, an `invoke` fn holding the body, a private argument
/// struct deriving the schema, and the `Tool` impl calling `invoke`.
/// Returns the pieces in output order.
pub fn expand_tool(t: &ToolFn) -> Vec<Piece> {
    let p = format!("{}::__private", t.krate);
    let plain = t.ident.strip_prefix("r#").unwrap_or(&t.ident);
    let args_ty = format!("__troupekit_args_{plain}");
    let ident = Piece::Orig(t.ident_span);
    let vis: Vec<Piece> = t.vis.iter().map(|s| Piece::Orig(*s)).collect();
    let cfg = || attrs(&t.cfg_attrs);
    let code = |s: &str| Piece::Code(s.to_string());
    let mut out = Vec::new();

    out.extend(cfg());
    out.push(Piece::Code(format!(
        "#[doc = {}] #[allow(non_camel_case_types)] #[derive(Clone, Copy, Debug, Default)]",
        quote_str(&t.description)
    )));
    out.extend(vis.clone());
    out.push(code("struct"));
    out.push(ident.clone());
    out.push(code(";"));

    out.extend(cfg());
    out.push(code("impl"));
    out.push(ident.clone());
    let mut invoke = attrs(&t.other_attrs);
    invoke.extend(vis.clone());
    invoke.push(code(if t.is_async {
        "async fn invoke"
    } else {
        "fn invoke"
    }));
    let mut outer = Vec::new();
    let mut inner = Vec::new();
    for (i, param) in t.params.iter().enumerate() {
        outer.push(code(&format!("__troupekit_a{i}:")));
        outer.extend(param.ty.iter().map(|tok| Piece::Orig(tok.span())));
        outer.push(code(","));
    }
    let forwarded: Vec<String> = (0..t.params.len())
        .map(|i| format!("__troupekit_a{i}"))
        .collect();
    inner.push(code("#[allow(dead_code)]"));
    inner.push(code(if t.is_async { "async fn" } else { "fn" }));
    inner.push(ident.clone());
    inner.push(Piece::Group(Delim::Paren, outer.clone()));
    if !t.ret.is_empty() {
        inner.push(code("->"));
        inner.extend(t.ret.iter().map(|tok| Piece::Orig(tok.span())));
    }
    let mut forward = vec![
        ident.clone(),
        code(&format!("::invoke({})", forwarded.join(", "))),
    ];
    if t.is_async {
        forward.push(code(".await"));
    }
    inner.push(Piece::Group(Delim::Brace, forward));
    for (i, param) in t.params.iter().enumerate() {
        inner.push(code(if param.is_mut { "let mut" } else { "let" }));
        inner.push(Piece::Orig(param.ident_span));
        inner.push(code(":"));
        inner.extend(param.ty.iter().map(|tok| Piece::Orig(tok.span())));
        inner.push(code(&format!("= __troupekit_a{i};")));
    }
    invoke.push(Piece::Group(Delim::Paren, outer));
    if !t.ret.is_empty() {
        invoke.push(code("->"));
        invoke.extend(t.ret.iter().map(|tok| Piece::Orig(tok.span())));
    }
    inner.push(Piece::Orig(t.body_span));
    invoke.push(Piece::Group(Delim::Brace, inner));
    out.push(Piece::Group(Delim::Brace, invoke));

    out.extend(cfg());
    out.push(Piece::Code(format!(
        "#[doc(hidden)] #[allow(non_camel_case_types)] \
         #[derive({p}::serde::Deserialize, {p}::schemars::JsonSchema)] \
         #[serde(crate = \"{p}::serde\", deny_unknown_fields)] \
         #[schemars(crate = \"{p}::schemars\")] struct {args_ty}"
    )));
    let mut fields = Vec::new();
    for param in t.params.iter().filter(|p| p.kind == ParamKind::Arg) {
        if param.optional {
            fields.push(code("#[serde(default)]"));
        }
        fields.push(Piece::Orig(param.ident_span));
        fields.push(code(":"));
        fields.extend(param.ty.iter().map(|tok| Piece::Orig(tok.span())));
        fields.push(code(","));
    }
    out.push(Piece::Group(Delim::Brace, fields));

    let call_args: Vec<String> = t
        .params
        .iter()
        .map(|p| match p.kind {
            ParamKind::Arg => format!("__troupekit_args.{}", p.ident),
            ParamKind::Context { by_ref: true } => "__troupekit_ctx".to_string(),
            ParamKind::Context { by_ref: false } => {
                "::core::clone::Clone::clone(__troupekit_ctx)".to_string()
            }
        })
        .collect();
    let await_suffix = if t.is_async { ".await" } else { "" };
    out.extend(cfg());
    out.push(Piece::Code(format!(
        "#[{p}::async_trait] impl {p}::Tool for"
    )));
    out.push(ident);
    out.push(Piece::Code(format!(
        "{{ \
         fn spec(&self) -> &{p}::ToolSpec {{ \
             static SPEC: ::std::sync::OnceLock<{p}::ToolSpec> = ::std::sync::OnceLock::new(); \
             SPEC.get_or_init(|| {p}::spec::<{args_ty}>({name}, {description})) \
         }} \
         async fn call(&self, __troupekit_value: {p}::serde_json::Value, __troupekit_ctx: &{p}::ToolContext) \
             -> ::core::result::Result<{p}::ToolOutput, {p}::ToolError> {{ \
             let __troupekit_args: {args_ty} = {p}::serde_json::from_value(__troupekit_value) \
                 .map_err(|e| {p}::ToolError::InvalidArgs(::std::string::ToString::to_string(&e)))?; \
             ::core::result::Result::Ok({p}::IntoToolOutput::into_output(Self::invoke({call}){await_suffix})) \
         }} \
         }}",
        name = quote_str(&t.name),
        description = quote_str(&t.description),
        call = call_args.join(", "),
    )));
    out
}

fn attrs(list: &[Attr]) -> Vec<Piece> {
    list.iter()
        .flat_map(|a| [Piece::Orig(a.hash), Piece::Orig(a.group)])
        .collect()
}

fn valid_tool_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= MAX_NAME_BYTES
        && name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

fn parse_params(toks: &[Tok]) -> Result<Vec<Param>, Error> {
    let mut params: Vec<Param> = Vec::new();
    for (i, entry) in split_params(toks).into_iter().enumerate() {
        let Some(first) = entry.first() else {
            continue;
        };
        if i >= MAX_ARGS {
            return Err(Error::new(
                first.span(),
                format!("{WHAT} takes at most {MAX_ARGS} arguments"),
            ));
        }
        let param = parse_param(&entry)?;
        if matches!(param.kind, ParamKind::Context { .. })
            && params
                .iter()
                .any(|p| matches!(p.kind, ParamKind::Context { .. }))
        {
            return Err(Error::new(
                first.span(),
                format!("{WHAT} takes at most one ToolContext argument"),
            ));
        }
        params.push(param);
    }
    assert!(params.len() <= MAX_ARGS);
    Ok(params)
}

fn parse_param(entry: &[Tok]) -> Result<Param, Error> {
    let first = &entry[0];
    if first.is_punct('#') {
        return Err(Error::new(
            first.span(),
            format!("attributes on {WHAT} arguments are not supported"),
        ));
    }
    let colon = entry
        .iter()
        .position(|t| matches!(t, Tok::Punct(':', crate::tokens::Spacing::Alone, _)));
    let head = &entry[..colon.unwrap_or(entry.len())];
    if let Some(s) = head.iter().find(|t| t.is_ident("self")) {
        return Err(Error::new(
            s.span(),
            format!("{WHAT} goes on a free fn, not a method"),
        ));
    }
    let pattern = || {
        Error::new(
            first.span(),
            format!("{WHAT} arguments must be plain names like a: i64"),
        )
    };
    let (is_mut, name) = match head {
        [Tok::Ident(m, _), name] if m == "mut" => (true, name),
        [name] => (false, name),
        _ => return Err(pattern()),
    };
    let Tok::Ident(ident, ident_span) = name else {
        return Err(pattern());
    };
    if matches!(ident.as_str(), "_" | "mut" | "ref") {
        return Err(pattern());
    }
    let ty = entry[colon.map_or(entry.len(), |c| c + 1)..].to_vec();
    let Some(ty_first) = ty.first() else {
        return Err(pattern());
    };
    if ty_first.is_ident("impl") {
        return Err(Error::new(
            ty_first.span(),
            format!("{WHAT} arguments cannot be impl Trait"),
        ));
    }
    let by_ref = ty_first.is_punct('&');
    let is_context = last_segment(&ty) == Some("ToolContext");
    if by_ref && !is_context {
        return Err(Error::new(
            ty_first.span(),
            format!("{WHAT} arguments must be owned types; use String instead of &str"),
        ));
    }
    let kind = if is_context {
        ParamKind::Context { by_ref }
    } else {
        ParamKind::Arg
    };
    let optional = !is_context && type_head(&ty) == Some("Option");
    Ok(Param {
        ident: ident.clone(),
        ident_span: *ident_span,
        is_mut,
        ty,
        kind,
        optional,
    })
}

fn split_params(toks: &[Tok]) -> Vec<Vec<Tok>> {
    let mut out = vec![Vec::new()];
    let mut depth = 0usize;
    let mut prev_dash = false;
    for tok in toks {
        if tok.is_punct('<') {
            depth += 1;
        } else if tok.is_punct('>') && !prev_dash {
            depth = depth.saturating_sub(1);
        }
        prev_dash = matches!(tok, Tok::Punct('-', crate::tokens::Spacing::Joint, _));
        if depth == 0 && tok.is_punct(',') {
            out.push(Vec::new());
        } else {
            out.last_mut()
                .expect("out starts non-empty")
                .push(tok.clone());
        }
    }
    out
}

fn last_segment(ty: &[Tok]) -> Option<&str> {
    if ty.iter().any(|t| t.is_punct('<')) {
        return None;
    }
    match ty.last() {
        Some(Tok::Ident(name, _)) => Some(name.as_str()),
        _ => None,
    }
}

fn type_head(ty: &[Tok]) -> Option<&str> {
    let end = ty.iter().position(|t| t.is_punct('<'))?;
    match ty[..end].last() {
        Some(Tok::Ident(name, _)) => Some(name.as_str()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::lex;
    use crate::tokens::Delim;

    fn parse(attr: &str, item: &str) -> Result<ToolFn, Error> {
        parse_tool(&lex(attr), &lex(item))
    }

    fn find(toks: &[Tok], pred: &dyn Fn(&Tok) -> bool) -> Option<SpanId> {
        for t in toks {
            if pred(t) {
                return Some(t.span());
            }
            if let Tok::Group(_, inner, _) = t {
                if let Some(s) = find(inner, pred) {
                    return Some(s);
                }
            }
        }
        None
    }

    fn rejects(item: &str, message: &str, at: &dyn Fn(&Tok) -> bool) {
        let toks = lex(item);
        let err = parse_tool(&[], &toks).unwrap_err();
        assert_eq!(err.message, message, "{item}");
        assert_eq!(Some(err.span), find(&toks, at), "{item}");
    }

    const DOC: &str = "/// Adds.\n";

    #[test]
    fn parses_async_and_sync_fns() {
        let a = parse(
            "",
            &format!("{DOC}pub async fn add(a: i64, b: i64) -> i64 {{ a + b }}"),
        )
        .unwrap();
        assert!(a.is_async);
        assert_eq!(a.name, "add");
        assert_eq!(a.description, "Adds.");
        assert_eq!(a.params.len(), 2);
        assert_eq!(a.ret.len(), 1);
        assert_eq!(a.krate, "::troupekit");
        let s = parse("", &format!("{DOC}fn ping() {{}}")).unwrap();
        assert!(!s.is_async);
        assert!(s.params.is_empty() && s.ret.is_empty() && s.vis.is_empty());
    }

    #[test]
    fn parses_pub_and_pub_crate() {
        assert_eq!(
            parse("", &format!("{DOC}pub fn f() {{}}"))
                .unwrap()
                .vis
                .len(),
            1
        );
        assert_eq!(
            parse("", &format!("{DOC}pub(crate) fn f() {{}}"))
                .unwrap()
                .vis
                .len(),
            2
        );
        assert_eq!(
            parse("", &format!("{DOC}pub(in crate::a) fn f() {{}}"))
                .unwrap()
                .vis
                .len(),
            2
        );
    }

    #[test]
    fn parses_trailing_comma_and_mut_params() {
        let t = parse(
            "",
            &format!("{DOC}fn f(mut a: Vec<String>, b: HashMap<String, (i64, u8)>,) {{}}"),
        )
        .unwrap();
        assert_eq!(t.params.len(), 2);
        assert!(t.params[0].is_mut && !t.params[1].is_mut);
        assert_eq!(t.params[1].ident, "b");
        assert_eq!(t.params[1].ty.len(), 6);
        let f = parse(
            "",
            &format!("{DOC}fn f(g: Box<dyn Fn(i64) -> Vec<u8>>, h: i64) {{}}"),
        )
        .unwrap();
        assert_eq!(f.params.len(), 2);
    }

    #[test]
    fn context_param_detected_by_last_segment() {
        let t = parse(
            "",
            &format!("{DOC}fn f(a: i64, ctx: &troupekit::kit::ToolContext) {{}}"),
        )
        .unwrap();
        assert_eq!(t.params[1].kind, ParamKind::Context { by_ref: true });
        let t = parse(
            "",
            &format!("{DOC}fn f(c: ToolContext, o: Option<String>) {{}}"),
        )
        .unwrap();
        assert_eq!(t.params[0].kind, ParamKind::Context { by_ref: false });
        assert!(t.params[1].optional && !t.params[0].optional);
        assert_eq!(t.params[1].kind, ParamKind::Arg);
    }

    #[test]
    fn raw_ident_param_maps_to_plain_key() {
        let t = parse("", &format!("{DOC}fn r#match(r#type: String) {{}}")).unwrap();
        assert_eq!(t.params[0].ident, "r#type");
        assert_eq!(t.name, "match");
    }

    #[test]
    fn name_and_description_overrides() {
        let t = parse("name = \"sum-up\", description = \"Sums.\"", "fn add() {}").unwrap();
        assert_eq!(t.name, "sum-up");
        assert_eq!(t.description, "Sums.");
    }

    #[test]
    fn crate_override() {
        let t = parse("crate = \"::my_kit::inner\"", &format!("{DOC}fn f() {{}}")).unwrap();
        assert_eq!(t.krate, "::my_kit::inner");
        let e = parse("crate = \"a b\"", &format!("{DOC}fn f() {{}}")).unwrap_err();
        assert_eq!(
            e.message,
            "#[tool] key `crate` takes a path like \"::troupekit\""
        );
    }

    #[test]
    fn cfg_and_other_attributes_are_sorted() {
        let t = parse("", &format!("{DOC}#[cfg(test)]\n#[inline]\nfn f() {{}}")).unwrap();
        assert_eq!(t.cfg_attrs.len(), 1);
        assert_eq!(t.other_attrs.len(), 1);
    }

    #[test]
    fn rejects_missing_doc() {
        rejects("fn add() {}", "#[tool] needs a /// doc comment or description = \"...\"; it becomes the tool description", &|t| t.is_ident("add"));
    }

    #[test]
    fn rejects_non_fn() {
        rejects("/// D.\nstruct S;", "#[tool] goes on a fn", &|t| {
            t.is_ident("struct")
        });
    }

    #[test]
    fn rejects_const_unsafe_extern() {
        for kw in ["const", "unsafe", "extern"] {
            rejects(
                &format!("{DOC}{kw} fn f() {{}}"),
                "#[tool] fns cannot be const, unsafe or extern",
                &|t| t.is_ident(kw),
            );
        }
    }

    #[test]
    fn rejects_generics_and_where() {
        rejects(
            &format!("{DOC}fn f<T>(a: T) {{}}"),
            "#[tool] fns cannot be generic; use concrete argument types",
            &|t| t.is_punct('<'),
        );
        rejects(
            &format!("{DOC}fn f(a: i64) where i64: Copy {{}}"),
            "#[tool] fns cannot be generic; use concrete argument types",
            &|t| t.is_ident("where"),
        );
    }

    #[test]
    fn rejects_self_receiver() {
        for recv in ["self", "&self", "&mut self", "mut self"] {
            rejects(
                &format!("{DOC}fn f({recv}) {{}}"),
                "#[tool] goes on a free fn, not a method",
                &|t| t.is_ident("self"),
            );
        }
    }

    #[test]
    fn rejects_pattern_argument() {
        rejects(
            &format!("{DOC}fn f((a, b): (i64, i64)) {{}}"),
            "#[tool] arguments must be plain names like a: i64",
            &|t| matches!(t, Tok::Group(Delim::Paren, inner, _) if inner.first().is_some_and(|x| x.is_ident("a"))),
        );
        rejects(
            &format!("{DOC}fn f(_: i64) {{}}"),
            "#[tool] arguments must be plain names like a: i64",
            &|t| t.is_ident("_"),
        );
    }

    #[test]
    fn rejects_reference_argument() {
        rejects(
            &format!("{DOC}fn f(a: &str) {{}}"),
            "#[tool] arguments must be owned types; use String instead of &str",
            &|t| t.is_punct('&'),
        );
    }

    #[test]
    fn rejects_impl_trait_argument() {
        rejects(
            &format!("{DOC}fn f(a: impl Into<String>) {{}}"),
            "#[tool] arguments cannot be impl Trait",
            &|t| t.is_ident("impl"),
        );
    }

    #[test]
    fn rejects_argument_attribute() {
        rejects(
            &format!("{DOC}fn f(#[x] a: i64) {{}}"),
            "attributes on #[tool] arguments are not supported",
            &|t| matches!(t, Tok::Punct('#', _, s) if s.0 > 5),
        );
    }

    #[test]
    fn rejects_two_context_arguments() {
        rejects(
            &format!("{DOC}fn f(a: ToolContext, b: &ToolContext) {{}}"),
            "#[tool] takes at most one ToolContext argument",
            &|t| t.is_ident("b"),
        );
    }

    #[test]
    fn rejects_more_than_32_arguments() {
        let params: Vec<String> = (0..33).map(|i| format!("a{i}: i64")).collect();
        rejects(
            &format!("{DOC}fn f({}) {{}}", params.join(", ")),
            "#[tool] takes at most 32 arguments",
            &|t| t.is_ident("a32"),
        );
        let ok: Vec<String> = (0..32).map(|i| format!("a{i}: i64")).collect();
        assert!(parse("", &format!("{DOC}fn f({}) {{}}", ok.join(", "))).is_ok());
    }

    #[test]
    fn rejects_unknown_key() {
        let e = parse("x = \"y\"", &format!("{DOC}fn f() {{}}")).unwrap_err();
        assert_eq!(
            e.message,
            "unknown #[tool] key `x`; known keys: name, description, crate"
        );
        assert_eq!(e.span, lex("x = \"y\"")[0].span());
    }

    #[test]
    fn rejects_duplicate_key() {
        let attr = lex("name = \"a\", name = \"b\"");
        let e = parse_tool(&attr, &lex(&format!("{DOC}fn f() {{}}"))).unwrap_err();
        assert_eq!(e.message, "#[tool] key `name` is set twice");
        assert_eq!(e.span, attr[4].span());
    }

    #[test]
    fn rejects_bad_name() {
        for bad in ["", "has space", &"x".repeat(65)] {
            let attr = lex(&format!("name = \"{bad}\""));
            let e = parse_tool(&attr, &lex(&format!("{DOC}fn f() {{}}"))).unwrap_err();
            assert_eq!(e.message, "tool name must be 1 to 64 of [A-Za-z0-9_-]");
            assert_eq!(e.span, attr[2].span());
        }
    }

    #[test]
    fn rejects_non_string_value() {
        let attr = lex("name = add");
        let e = parse_tool(&attr, &lex(&format!("{DOC}fn f() {{}}"))).unwrap_err();
        assert_eq!(e.message, "#[tool] key `name` takes a string literal");
        assert_eq!(e.span, attr[2].span());
    }

    #[test]
    fn rejects_doc_over_16k() {
        let item = lex(&format!("/// {}\nfn f() {{}}", "x".repeat(16 * 1024 + 1)));
        let e = parse_tool(&[], &item).unwrap_err();
        assert_eq!(e.message, "the #[tool] description is over 16 KiB");
        assert_eq!(e.span, find(&item, &|t| t.is_ident("doc")).unwrap());
    }

    fn text(tok: &Tok) -> String {
        match tok {
            Tok::Ident(s, _) | Tok::Lit(s, _) => s.clone(),
            Tok::Punct(c, _, _) => c.to_string(),
            Tok::Group(d, inner, _) => {
                let body: Vec<String> = inner.iter().map(text).collect();
                let (o, c) = match d {
                    Delim::Paren => ("(", ")"),
                    Delim::Bracket => ("[", "]"),
                    Delim::Brace => ("{", "}"),
                    Delim::None => ("", ""),
                };
                format!("{o}{}{c}", body.join(" "))
            }
        }
    }

    fn render(pieces: &[Piece], toks: &[Tok]) -> String {
        let all = |id: SpanId| {
            let pred = |t: &Tok| t.span() == id;
            let mut stack: Vec<&Tok> = toks.iter().collect();
            while let Some(t) = stack.pop() {
                if pred(t) {
                    return text(t);
                }
                if let Tok::Group(_, inner, _) = t {
                    stack.extend(inner.iter());
                }
            }
            panic!("span {id:?} not in source");
        };
        pieces
            .iter()
            .map(|p| match p {
                Piece::Code(s) => s.clone(),
                Piece::Orig(id) => all(*id),
                Piece::Group(Delim::Paren, inner) => format!("({})", render(inner, toks)),
                Piece::Group(_, inner) => format!("{{{}}}", render(inner, toks)),
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn expansion_shape_for_add() {
        let item = lex("/// Adds two numbers.\npub async fn add(a: i64, o: Option<u8>, ctx: ToolContext) -> i64 { a }");
        let t = parse_tool(&[], &item).unwrap();
        let out = render(&expand_tool(&t), &item);
        for part in [
            "#[doc = \"Adds two numbers.\"]",
            "pub struct add ;",
            "impl add {pub async fn invoke (__troupekit_a0: i64 , __troupekit_a1: Option < u8 > , __troupekit_a2: ToolContext ,) -> i64 {#[allow(dead_code)] = __troupekit_a0; let o : Option < u8 > = __troupekit_a1; let ctx : ToolContext = __troupekit_a2; {a}}}",
            "struct __troupekit_args_add {a : i64 , #[serde(default)] o : Option < u8 > ,}",
            "#[serde(crate = \"::troupekit::__private::serde\", deny_unknown_fields)]",
            "let __troupekit_args: __troupekit_args_add =",
            "Self::invoke(__troupekit_args.a, __troupekit_args.o, ::core::clone::Clone::clone(__troupekit_ctx)).await",
            "spec::<__troupekit_args_add>(\"add\", \"Adds two numbers.\")",
            "{#[allow(dead_code)] async fn add (__troupekit_a0: i64 , __troupekit_a1: Option < u8 > , __troupekit_a2: ToolContext ,) -> i64 {add ::invoke(__troupekit_a0, __troupekit_a1, __troupekit_a2) .await} let a",
        ] {
            assert!(out.contains(part), "missing {part:?} in {out}");
        }
        let clash = lex("/// Q.\nfn query(query: String) -> String { query }");
        let out = render(&expand_tool(&parse_tool(&[], &clash).unwrap()), &clash);
        assert!(
            out.contains("{#[allow(dead_code)] fn query (__troupekit_a0: String ,) -> String {query ::invoke(__troupekit_a0)} let query"),
            "{out}"
        );
        let sync = lex("/// S.\nfn s(ctx: &ToolContext) {}");
        let out = render(&expand_tool(&parse_tool(&[], &sync).unwrap()), &sync);
        assert!(out.contains("Self::invoke(__troupekit_ctx))"), "{out}");
        assert!(!out.contains(".await"), "{out}");
    }
}
