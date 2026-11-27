use crate::args::{parse_args, split_top_level, Arg};
use crate::doc::doc_text;
use crate::item::{parse_fn, Attr};
use crate::lit::quote_str;
use crate::tokens::{Delim, Error, Piece, SpanId, Tok};
use crate::tool::crate_path;

const WHAT: &str = "#[agent]";
const KEYS: [&str; 25] = [
    "model",
    "tools",
    "evals",
    "limits",
    "checkpoints",
    "max_turns",
    "max_tokens",
    "retries",
    "role",
    "goal",
    "display_name",
    "instructions",
    "laws",
    "auto_laws",
    "memory",
    "knowledge",
    "name",
    "crate",
    "trace",
    "trace_content",
    "local",
    "download",
    "route",
    "private",
    "tier",
];
const TRACE_MESSAGE: &str = "trace = takes all or a list like [Tools, Laws]";
const DEFAULT_CRATE: &str = "::troupekit";

/// One builder call the `#[agent]` expansion makes: the method name and
/// the value pieces passed to it, original tokens keeping their spans.
/// Calls run in attribute order, after `new` and `instructions`.
#[derive(Clone, Debug)]
pub struct Call {
    pub method: &'static str,
    pub value: Vec<Piece>,
}

/// A validated `#[agent]` fn, ready for code generation.
/// `name` is the persona name, `instructions` the doc comment unless an
/// `instructions` key replaces it. `calls` holds every other key.
#[derive(Clone, Debug)]
pub struct AgentFn {
    pub attrs: Vec<Attr>,
    pub vis: Vec<SpanId>,
    pub ident_span: SpanId,
    pub name: String,
    pub instructions: Option<String>,
    pub calls: Vec<Call>,
    pub krate: String,
}

/// Parses `#[agent(attr)]` on `item` into an `AgentFn`.
/// The fn must be `fn name() {}`; every key is checked for its value
/// shape and every unsupported input fails with its documented message.
pub fn parse_agent(attr: &[Tok], item: &[Tok]) -> Result<AgentFn, Error> {
    let args = parse_args(attr, WHAT)?;
    for arg in &args {
        if !KEYS.contains(&arg.key.as_str()) {
            return Err(Error::new(
                arg.key_span,
                format!(
                    "unknown {WHAT} key `{}`; known keys: {}",
                    arg.key,
                    KEYS.join(", ")
                ),
            ));
        }
    }
    let f = parse_fn(item, WHAT)?;
    if let Some((word, span)) = f.quals.first() {
        let message = if word == "async" {
            format!("{WHAT} fn cannot be async")
        } else {
            format!("{WHAT} fn cannot be const, unsafe or extern")
        };
        return Err(Error::new(*span, message));
    }
    if let Some(span) = f.generics.or(f.where_span) {
        return Err(Error::new(span, format!("{WHAT} fn cannot be generic")));
    }
    let extra = f
        .params
        .first()
        .or(f.ret.first())
        .map(Tok::span)
        .or_else(|| (!f.body.is_empty()).then_some(f.body_span));
    if let Some(span) = extra {
        return Err(Error::new(
            span,
            format!("{WHAT} fn must take no arguments, return nothing and have an empty body {{}}; configure it with attribute keys or use Agent::new"),
        ));
    }
    let ident = f.ident.strip_prefix("r#").unwrap_or(&f.ident);
    let valid_ident = ident
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c.is_ascii_digit())
        && ident
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_');
    let mut name = ident.replace('_', "-");
    let mut instructions = None;
    let mut krate = DEFAULT_CRATE.to_string();
    let mut name_set = false;
    let mut calls = Vec::new();
    let mut trace_at = None;
    for arg in &args {
        let spans = || orig(&arg.value);
        match arg.key.as_str() {
            "name" => {
                name = arg.string(WHAT)?;
                name_set = true;
            }
            "crate" => krate = crate_path(arg, WHAT)?,
            "instructions" => instructions = Some(arg.string(WHAT)?),
            "model" | "role" | "goal" | "display_name" => {
                arg.string(WHAT)?;
                calls.push(Call {
                    method: method(&arg.key),
                    value: spans(),
                });
            }
            "tools" => {
                for path in list(arg, "tools = takes a list like [add, math::mul]")? {
                    if !is_path(&path) {
                        return Err(Error::new(
                            path[0].span(),
                            "tools = takes a list like [add, math::mul]",
                        ));
                    }
                    calls.push(Call {
                        method: "tool",
                        value: orig(&path),
                    });
                }
            }
            "evals" => {
                let message = "evals = takes a list of string literals";
                for entry in list(arg, message)? {
                    if !matches!(entry.as_slice(), [Tok::Lit(l, _)] if l.starts_with('"') || l.starts_with('r'))
                    {
                        return Err(Error::new(entry[0].span(), message));
                    }
                    calls.push(Call {
                        method: "eval",
                        value: orig(&entry[..1]),
                    });
                }
            }
            "laws" => {
                let entries = match arg.value.as_slice() {
                    [Tok::Group(Delim::Bracket, ..)] => list(arg, "")?,
                    _ => vec![arg.value.clone()],
                };
                for entry in entries {
                    let one = Arg {
                        key: arg.key.clone(),
                        key_span: arg.key_span,
                        value: entry,
                    };
                    one.string(WHAT)?;
                    calls.push(Call {
                        method: "laws",
                        value: orig(&one.value),
                    });
                }
            }
            "auto_laws" => {
                if !matches!(arg.value.as_slice(), [t] if t.is_ident("true") || t.is_ident("false"))
                {
                    return Err(Error::new(
                        arg.value_span(),
                        "auto_laws = takes true or false",
                    ));
                }
                calls.push(Call {
                    method: "auto_laws",
                    value: spans(),
                });
            }
            "trace" => {
                trace_at = Some((calls.len(), arg));
                calls.push(Call {
                    method: "trace",
                    value: Vec::new(),
                });
            }
            "trace_content" | "private" | "download" => {
                if !is_bool(&arg.value) {
                    return Err(Error::new(
                        arg.value_span(),
                        format!("{} = takes true or false", arg.key),
                    ));
                }
                calls.push(Call {
                    method: method(&arg.key),
                    value: spans(),
                });
            }
            "route" => calls.push(Call {
                method: if is_string(&arg.value) { "route_file" } else { "route" },
                value: spans(),
            }),
            "tier" => {
                arg.string(WHAT)?;
                calls.push(Call {
                    method: "tier",
                    value: spans(),
                });
            }
            _ => calls.push(Call {
                method: method(&arg.key),
                value: spans(),
            }),
        }
    }
    if let Some((at, arg)) = trace_at {
        assert_eq!(calls[at].method, "trace", "the placeholder is in place");
        calls[at] = trace_call(arg, &krate)?;
    }
    if !name_set && !valid_ident {
        return Err(Error::new(
            f.ident_span,
            "agent name must be lowercase ASCII letters, digits and _",
        ));
    }
    if instructions.is_none() {
        let attrs: Vec<Vec<Tok>> = f.attrs.iter().map(|a| a.inner.clone()).collect();
        instructions = doc_text(&attrs, "#[agent] instructions")?;
    }
    if instructions.is_none() {
        return Err(Error::new(
            f.ident_span,
            format!("{WHAT} needs a /// doc comment or instructions = \"...\"; it becomes the instructions"),
        ));
    }
    Ok(AgentFn {
        attrs: f.attrs,
        vis: f.vis,
        ident_span: f.ident_span,
        name,
        instructions,
        calls,
        krate,
    })
}

/// Generates `vis fn name() -> Agent { Agent::new(..).instructions(..)... }`
/// for a parsed `#[agent]` fn. Attributes, doc comments included, stay
/// on the fn. Returns the pieces in output order.
pub fn expand_agent(a: &AgentFn) -> Vec<Piece> {
    let p = format!("{}::__private", a.krate);
    let mut out: Vec<Piece> = a
        .attrs
        .iter()
        .flat_map(|at| [Piece::Orig(at.hash), Piece::Orig(at.group)])
        .collect();
    out.extend(a.vis.iter().map(|s| Piece::Orig(*s)));
    out.push(Piece::Code("fn".into()));
    out.push(Piece::Orig(a.ident_span));
    out.push(Piece::Code(format!("() -> {p}::Agent")));
    let instructions = a
        .instructions
        .as_deref()
        .expect("parse_agent sets instructions");
    let mut body = vec![Piece::Code(format!(
        "{p}::Agent::new({}).instructions({})",
        quote_str(&a.name),
        quote_str(instructions)
    ))];
    for call in &a.calls {
        body.push(Piece::Code(format!(".{}", call.method)));
        body.push(Piece::Group(Delim::Paren, call.value.clone()));
    }
    out.push(Piece::Group(Delim::Brace, body));
    out
}

fn orig(toks: &[Tok]) -> Vec<Piece> {
    toks.iter().map(|t| Piece::Orig(t.span())).collect()
}

fn is_bool(toks: &[Tok]) -> bool {
    matches!(toks, [t] if t.is_ident("true") || t.is_ident("false"))
}

fn is_string(toks: &[Tok]) -> bool {
    matches!(toks, [Tok::Lit(l, _)] if l.starts_with('"') || l.starts_with('r'))
}

fn trace_call(arg: &Arg, krate: &str) -> Result<Call, Error> {
    if matches!(arg.value.as_slice(), [t] if t.is_ident("all")) {
        return Ok(Call {
            method: "tracing",
            value: vec![Piece::Code("true".into())],
        });
    }
    let mut inner = Vec::new();
    for (i, entry) in list(arg, TRACE_MESSAGE)?.iter().enumerate() {
        if !is_path(entry) {
            return Err(Error::new(entry[0].span(), TRACE_MESSAGE));
        }
        if i > 0 {
            inner.push(Piece::Code(",".into()));
        }
        if entry.len() == 1 {
            inner.push(Piece::Code(format!("{krate}::__private::trace::Trace::")));
        }
        inner.extend(orig(entry));
    }
    assert!(inner.iter().all(|p| !matches!(p, Piece::Group(..))));
    Ok(Call {
        method: "trace",
        value: vec![Piece::Group(Delim::Bracket, inner)],
    })
}

fn method(key: &str) -> &'static str {
    KEYS.iter()
        .copied()
        .find(|k| *k == key)
        .expect("keys are checked before methods are named")
}

fn list(arg: &Arg, message: &str) -> Result<Vec<Vec<Tok>>, Error> {
    let message = if message.is_empty() {
        format!("{WHAT} key `{}` takes a string literal", arg.key)
    } else {
        message.to_string()
    };
    match arg.value.as_slice() {
        [Tok::Group(Delim::Bracket, inner, _)] => Ok(split_top_level(inner)
            .into_iter()
            .filter(|e| !e.is_empty())
            .collect()),
        _ => Err(Error::new(arg.value_span(), message)),
    }
}

fn is_path(toks: &[Tok]) -> bool {
    let mut expect_ident = true;
    let mut i = 0usize;
    if toks.first().is_some_and(|t| t.is_punct(':')) {
        expect_ident = false;
    }
    while i < toks.len() {
        if expect_ident {
            if !matches!(toks[i], Tok::Ident(..)) {
                return false;
            }
            i += 1;
        } else {
            if !(toks[i].is_punct(':') && toks.get(i + 1).is_some_and(|t| t.is_punct(':'))) {
                return false;
            }
            i += 2;
        }
        expect_ident = !expect_ident;
    }
    !toks.is_empty() && !expect_ident
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lex::lex;

    const DOC: &str = "/// Adds numbers.\n";

    fn parse(attr: &str, item: &str) -> Result<AgentFn, Error> {
        parse_agent(&lex(attr), &lex(item))
    }

    fn methods(a: &AgentFn) -> Vec<&'static str> {
        a.calls.iter().map(|c| c.method).collect()
    }

    #[test]
    fn memory_and_knowledge_keys_become_builder_calls() {
        let a = parse(
            "model = \"m/x\", memory = mem(1), knowledge = \"docs\"",
            &format!("{DOC}fn calc() {{}}"),
        )
        .unwrap();
        assert_eq!(methods(&a), ["model", "memory", "knowledge"]);
        assert_eq!(a.calls[1].value.len(), 2);
    }

    fn rejects(attr: &str, item: &str, message: &str) -> Error {
        let err = parse(attr, item).unwrap_err();
        assert_eq!(err.message, message, "{attr} {item}");
        err
    }

    #[test]
    fn parses_every_key() {
        let attr = "model = \"openai/gpt-6-luna\", tools = [add], evals = [\"min_items(x, 1)\"], \
            limits = Limits::default(), checkpoints = Storage::memory(), max_turns = 4, \
            max_tokens = 100, retries = 1, role = \"R\", goal = \"G\", display_name = \"D\", \
            instructions = \"I\", laws = [\"a.bend\", \"b.bend\"], auto_laws = false, \
            name = \"adder\", crate = \"::kit\", trace = all, trace_content = true, \
            local = \"qwen\", download = true, route = \"route.toml\", private = true, \
            tier = \"small\",";
        let a = parse(attr, "fn calc() {}").unwrap();
        assert_eq!(a.name, "adder");
        assert_eq!(a.instructions.as_deref(), Some("I"));
        assert_eq!(a.krate, "::kit");
        assert_eq!(
            methods(&a),
            [
                "model",
                "tool",
                "eval",
                "limits",
                "checkpoints",
                "max_turns",
                "max_tokens",
                "retries",
                "role",
                "goal",
                "display_name",
                "laws",
                "laws",
                "auto_laws",
                "tracing",
                "trace_content",
                "local",
                "download",
                "route_file",
                "private",
                "tier"
            ]
        );
        assert_eq!(a.calls[3].value.len(), 5);
    }

    #[test]
    fn tools_list_keeps_paths() {
        let a = parse(
            "tools = [add, math::mul, ::ext::div,]",
            &format!("{DOC}fn calc() {{}}"),
        )
        .unwrap();
        let lens: Vec<usize> = a.calls.iter().map(|c| c.value.len()).collect();
        assert_eq!(lens, [1, 4, 6]);
        rejects(
            "tools = add",
            &format!("{DOC}fn calc() {{}}"),
            "tools = takes a list like [add, math::mul]",
        );
        rejects(
            "tools = [add()]",
            &format!("{DOC}fn calc() {{}}"),
            "tools = takes a list like [add, math::mul]",
        );
        rejects(
            "tools = [\"add\"]",
            &format!("{DOC}fn calc() {{}}"),
            "tools = takes a list like [add, math::mul]",
        );
    }

    #[test]
    fn evals_must_be_string_literals() {
        let a = parse(
            "evals = [\"a(1)\", r\"b(2)\"]",
            &format!("{DOC}fn calc() {{}}"),
        )
        .unwrap();
        assert_eq!(methods(&a), ["eval", "eval"]);
        rejects(
            "evals = [a]",
            &format!("{DOC}fn calc() {{}}"),
            "evals = takes a list of string literals",
        );
        rejects(
            "evals = \"a\"",
            &format!("{DOC}fn calc() {{}}"),
            "evals = takes a list of string literals",
        );
    }

    #[test]
    fn fn_name_maps_underscore_to_dash() {
        let a = parse("", &format!("{DOC}pub fn web_researcher_2() {{}}")).unwrap();
        assert_eq!(a.name, "web-researcher-2");
        assert_eq!(a.instructions.as_deref(), Some("Adds numbers."));
        assert_eq!(a.krate, "::troupekit");
    }

    #[test]
    fn expansion_builds_on_agent_new() {
        let item = lex(&format!("{DOC}pub fn calc() {{}}"));
        let attr = lex("model = \"m/x\", tools = [add]");
        let a = parse_agent(&attr, &item).unwrap();
        let pieces = expand_agent(&a);
        let code: Vec<String> = pieces
            .iter()
            .filter_map(|p| match p {
                Piece::Code(s) => Some(s.clone()),
                Piece::Group(_, inner) => Some(
                    inner
                        .iter()
                        .filter_map(|q| match q {
                            Piece::Code(s) => Some(s.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(""),
                ),
                Piece::Orig(_) => None,
            })
            .collect();
        assert_eq!(code[0], "fn");
        assert_eq!(code[1], "() -> ::troupekit::__private::Agent");
        assert_eq!(
            code[2],
            "::troupekit::__private::Agent::new(\"calc\").instructions(\"Adds numbers.\").model.tool"
        );
    }

    #[test]
    fn rejects_non_fn() {
        rejects("", "struct S;", "#[agent] goes on a fn");
    }

    #[test]
    fn rejects_arguments_return_and_body() {
        let message = "#[agent] fn must take no arguments, return nothing and have an empty body {}; configure it with attribute keys or use Agent::new";
        let item = lex(&format!("{DOC}fn calc(a: i64) {{}}"));
        let err = parse_agent(&[], &item).unwrap_err();
        assert_eq!(err.message, message);
        rejects("", &format!("{DOC}fn calc() -> Agent {{}}"), message);
        rejects("", &format!("{DOC}fn calc() {{ todo!() }}"), message);
    }

    #[test]
    fn rejects_async() {
        rejects(
            "",
            &format!("{DOC}async fn calc() {{}}"),
            "#[agent] fn cannot be async",
        );
    }

    #[test]
    fn rejects_generic() {
        rejects(
            "",
            &format!("{DOC}fn calc<T>() {{}}"),
            "#[agent] fn cannot be generic",
        );
        rejects(
            "",
            &format!("{DOC}fn calc() where u8: Copy {{}}"),
            "#[agent] fn cannot be generic",
        );
    }

    #[test]
    fn rejects_missing_doc() {
        let item = lex("fn calc() {}");
        let err = parse_agent(&[], &item).unwrap_err();
        assert_eq!(err.message, "#[agent] needs a /// doc comment or instructions = \"...\"; it becomes the instructions");
        assert_eq!(err.span, item[1].span());
    }

    #[test]
    fn rejects_bad_name() {
        let item = lex(&format!("{DOC}fn Calc() {{}}"));
        let err = parse_agent(&[], &item).unwrap_err();
        assert_eq!(
            err.message,
            "agent name must be lowercase ASCII letters, digits and _"
        );
        assert_eq!(err.span, item[3].span());
        rejects(
            "",
            &format!("{DOC}fn _calc() {{}}"),
            "agent name must be lowercase ASCII letters, digits and _",
        );
        assert!(parse("name = \"calc\"", &format!("{DOC}fn Calc() {{}}")).is_ok());
    }

    #[test]
    fn rejects_unknown_key() {
        rejects("x = 1", &format!("{DOC}fn calc() {{}}"), "unknown #[agent] key `x`; known keys: model, tools, evals, limits, checkpoints, max_turns, max_tokens, retries, role, goal, display_name, instructions, laws, auto_laws, memory, knowledge, name, crate, trace, trace_content, local, download, route, private, tier");
    }

    #[test]
    fn rejects_duplicate_key() {
        rejects(
            "model = \"a/b\", model = \"a/c\"",
            &format!("{DOC}fn calc() {{}}"),
            "#[agent] key `model` is set twice",
        );
    }

    #[test]
    fn rejects_non_string_value() {
        rejects(
            "model = gpt",
            &format!("{DOC}fn calc() {{}}"),
            "#[agent] key `model` takes a string literal",
        );
        rejects(
            "laws = [rules]",
            &format!("{DOC}fn calc() {{}}"),
            "#[agent] key `laws` takes a string literal",
        );
    }

    fn code_of(pieces: &[Piece]) -> String {
        pieces
            .iter()
            .map(|p| match p {
                Piece::Code(s) => s.clone(),
                Piece::Orig(_) => "#".into(),
                Piece::Group(_, inner) => format!("[{}]", code_of(inner)),
            })
            .collect()
    }

    #[test]
    fn trace_all_and_list_become_builder_calls() {
        let item = format!("{DOC}fn calc() {{}}");
        let all = parse("trace = all", &item).unwrap();
        assert_eq!(methods(&all), ["tracing"]);
        assert_eq!(code_of(&all.calls[0].value), "true");
        let list = parse(
            "trace = [Tools, Trace::Laws, ::x::Trace::Agents], crate = \"::kit\"",
            &item,
        )
        .unwrap();
        assert_eq!(methods(&list), ["trace"]);
        assert_eq!(
            code_of(&list.calls[0].value),
            "[::kit::__private::trace::Trace::#,####,#########]"
        );
        let empty = parse("trace = []", &item).unwrap();
        assert_eq!(code_of(&empty.calls[0].value), "[]");
        for bad in ["trace = true", "trace = \"all\"", "trace = [1]", "trace = none"] {
            rejects(bad, &item, TRACE_MESSAGE);
        }
    }

    #[test]
    fn local_and_route_accept_literal_or_expression() {
        let item = format!("{DOC}fn calc() {{}}");
        let a = parse(
            "local = \"qwen\", route = \"route.toml\", tier = \"small\"",
            &item,
        )
        .unwrap();
        assert_eq!(methods(&a), ["local", "route_file", "tier"]);
        let b = parse("local = my_model(), route = office_policy()", &item).unwrap();
        assert_eq!(methods(&b), ["local", "route"]);
        assert_eq!(b.calls[0].value.len(), 2);
        rejects("tier = small", &item, "#[agent] key `tier` takes a string literal");
        for key in ["private", "trace_content", "download"] {
            rejects(
                &format!("{key} = maybe"),
                &item,
                &format!("{key} = takes true or false"),
            );
        }
        let flags = parse("private = false, download = false", &item).unwrap();
        assert_eq!(methods(&flags), ["private", "download"]);
    }

    #[test]
    fn rejects_bad_auto_laws() {
        rejects(
            "auto_laws = yes",
            &format!("{DOC}fn calc() {{}}"),
            "auto_laws = takes true or false",
        );
    }
}
