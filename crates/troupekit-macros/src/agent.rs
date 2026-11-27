use crate::args::{parse_args, split_top_level, Arg};
use crate::doc::doc_text;
use crate::item::{parse_fn, Attr};
use crate::lit::quote_str;
use crate::tokens::{Delim, Error, Piece, SpanId, Tok};
use crate::tool::crate_path;

const WHAT: &str = "#[agent]";
const KEYS: [&str; 24] = [
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
    "agents",
    "delegates",
    "handoffs",
    "manages",
    "members",
    "team",
    "name",
    "crate",
];
const DEFAULT_CRATE: &str = "::troupekit";

/// One builder call the `#[agent]` expansion makes: the method name, the
/// original value tokens passed to it, re-emitted with their spans, and
/// the shape they take. Calls run in attribute order, after `new` and
/// `instructions`.
#[derive(Clone, Debug)]
pub struct Call {
    pub method: &'static str,
    pub value: Vec<SpanId>,
    pub shape: Shape,
}

/// How a call passes its value: `Plain` as written, `Invoke` as the call
/// `path()`, `InvokeList` as the one-element array `[path()]`.
/// `Members(mode)` as `[path()]` followed by the `TeamMode` variant.
/// `agents`, `delegates`, `handoffs`, `manages` and `members` name fns
/// that return an Agent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Shape {
    Plain,
    Invoke,
    InvokeList,
    Members(&'static str),
}

const TEAM_MODES: [(&str, &str); 5] = [
    ("route", "Route"),
    ("coordinate", "Coordinate"),
    ("collaborate", "Collaborate"),
    ("broadcast", "Broadcast"),
    ("tasks", "Tasks"),
];
const TEAM_MESSAGE: &str = "team = takes one of route, coordinate, collaborate, broadcast, tasks";

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
    let mut team: Option<(&'static str, SpanId)> = None;
    for arg in &args {
        let spans = || arg.value.iter().map(Tok::span).collect::<Vec<_>>();
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
                    shape: Shape::Plain,
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
                        value: path.iter().map(Tok::span).collect(),
                        shape: Shape::Plain,
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
                        value: vec![entry[0].span()],
                        shape: Shape::Plain,
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
                        value: one.value.iter().map(Tok::span).collect(),
                        shape: Shape::Plain,
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
                    shape: Shape::Plain,
                });
            }
            "members" => {
                let message = "members = takes a list like [billing, tech]";
                let paths = list(arg, message)?;
                if paths.is_empty() {
                    return Err(Error::new(arg.value_span(), message));
                }
                for path in paths {
                    if !is_path(&path) {
                        return Err(Error::new(path[0].span(), message));
                    }
                    calls.push(Call {
                        method: "members",
                        value: path.iter().map(Tok::span).collect(),
                        shape: Shape::Members("Coordinate"),
                    });
                }
            }
            "team" => team = Some((team_mode(arg)?, arg.key_span)),
            "agents" | "delegates" | "handoffs" | "manages" => {
                let (method, shape) = match arg.key.as_str() {
                    "agents" => ("agent_tool", Shape::Invoke),
                    "delegates" => ("delegates", Shape::InvokeList),
                    "handoffs" => ("handoffs", Shape::InvokeList),
                    _ => ("manages", Shape::InvokeList),
                };
                calls.extend(agent_paths(arg, method, shape)?);
            }
            _ => calls.push(Call {
                method: method(&arg.key),
                value: spans(),
                shape: Shape::Plain,
            }),
        }
    }
    set_team_mode(&mut calls, team)?;
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
        let mut value: Vec<Piece> = call.value.iter().map(|s| Piece::Orig(*s)).collect();
        if call.shape != Shape::Plain {
            value.push(Piece::Code("()".into()));
        }
        if matches!(call.shape, Shape::InvokeList | Shape::Members(_)) {
            value = vec![Piece::Group(Delim::Bracket, value)];
        }
        if let Shape::Members(mode) = call.shape {
            value.push(Piece::Code(format!(", {p}::TeamMode::{mode}")));
        }
        body.push(Piece::Group(Delim::Paren, value));
    }
    out.push(Piece::Group(Delim::Brace, body));
    out
}

/// Parses an `agents`, `delegates`, `handoffs` or `manages` list of fn
/// paths into one `method` call per path, in list order. Returns an error
/// on a value that is not a list of paths, or on an empty list other than
/// `agents`.
fn agent_paths(arg: &Arg, method: &'static str, shape: Shape) -> Result<Vec<Call>, Error> {
    let message = format!(
        "{} = takes a list like [researcher, team::checker]",
        arg.key
    );
    let paths = list(arg, &message)?;
    if paths.is_empty() && arg.key != "agents" {
        return Err(Error::new(arg.value_span(), message));
    }
    let mut calls = Vec::with_capacity(paths.len());
    for path in paths {
        if !is_path(&path) {
            return Err(Error::new(path[0].span(), message));
        }
        calls.push(Call {
            method,
            value: path.iter().map(Tok::span).collect(),
            shape,
        });
    }
    assert!(calls
        .iter()
        .all(|c| !c.value.is_empty() && c.shape != Shape::Plain));
    Ok(calls)
}

/// Reads the `team` value, a string literal naming one team mode, and
/// returns its `TeamMode` variant name. Returns the team error for any
/// other value.
fn team_mode(arg: &Arg) -> Result<&'static str, Error> {
    let text = match arg.value.as_slice() {
        [Tok::Lit(l, _)] if l.starts_with('"') && l.ends_with('"') && l.len() >= 2 => {
            &l[1..l.len() - 1]
        }
        _ => return Err(Error::new(arg.value_span(), TEAM_MESSAGE)),
    };
    let Some((_, variant)) = TEAM_MODES.iter().find(|(name, _)| *name == text) else {
        return Err(Error::new(arg.value_span(), TEAM_MESSAGE));
    };
    assert!(variant
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_uppercase()));
    Ok(variant)
}

/// Gives every `members` call the mode of the `team` key, or keeps the
/// default `Coordinate`. Returns an error for `team` without `members`.
/// Calls of other keys are left as they are.
fn set_team_mode(calls: &mut [Call], team: Option<(&'static str, SpanId)>) -> Result<(), Error> {
    let has_members = calls.iter().any(|c| matches!(c.shape, Shape::Members(_)));
    let Some((mode, span)) = team else {
        return Ok(());
    };
    if !has_members {
        return Err(Error::new(
            span,
            format!("{WHAT} key `team` needs members = [..]"),
        ));
    }
    for call in calls
        .iter_mut()
        .filter(|c| matches!(c.shape, Shape::Members(_)))
    {
        call.shape = Shape::Members(mode);
    }
    assert!(calls.iter().any(|c| c.shape == Shape::Members(mode)));
    Ok(())
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
            name = \"adder\", crate = \"::kit\",";
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
                "auto_laws"
            ]
        );
        assert_eq!(a.calls[3].value.len(), 5);
    }

    #[test]
    fn agents_and_delegates_call_each_path() {
        let a = parse(
            "agents = [researcher, team::checker], delegates = [writer, ::x::editor]",
            &format!("{DOC}fn lead() {{}}"),
        )
        .unwrap();
        assert_eq!(
            methods(&a),
            ["agent_tool", "agent_tool", "delegates", "delegates"]
        );
        let shapes: Vec<Shape> = a.calls.iter().map(|c| c.shape).collect();
        assert_eq!(
            shapes,
            [
                Shape::Invoke,
                Shape::Invoke,
                Shape::InvokeList,
                Shape::InvokeList
            ]
        );
        let lens: Vec<usize> = a.calls.iter().map(|c| c.value.len()).collect();
        assert_eq!(lens, [1, 4, 1, 6]);
        let item = lex(&format!("{DOC}fn lead() {{}}"));
        let a = parse_agent(&lex("agents = [researcher], delegates = [writer]"), &item).unwrap();
        let pieces = expand_agent(&a);
        let Some(Piece::Group(Delim::Brace, body)) = pieces.last() else {
            panic!("the body is the last piece");
        };
        let invoke = |p: &Piece| match p {
            Piece::Group(Delim::Paren, inner) => {
                matches!(inner.as_slice(), [Piece::Orig(_), Piece::Code(c)] if c == "()")
            }
            _ => false,
        };
        assert!(matches!(&body[1], Piece::Code(c) if c == ".agent_tool"));
        assert!(invoke(&body[2]));
        assert!(matches!(&body[3], Piece::Code(c) if c == ".delegates"));
        let Piece::Group(Delim::Paren, outer) = &body[4] else {
            panic!("delegates takes one argument");
        };
        let [Piece::Group(Delim::Bracket, list)] = outer.as_slice() else {
            panic!("delegates takes an array");
        };
        assert!(invoke(&Piece::Group(Delim::Paren, list.clone())));
    }

    #[test]
    fn handoffs_call_once_with_every_path() {
        let item = format!("{DOC}fn desk() {{}}");
        let a = parse_agent(&lex("handoffs = [billing, ::x::tech]"), &lex(&item)).unwrap();
        assert_eq!(methods(&a), ["handoffs", "handoffs"]);
        assert!(a.calls.iter().all(|c| c.shape == Shape::InvokeList));
        assert_eq!(a.calls[1].value.len(), 6);
        let message = "handoffs = takes a list like [researcher, team::checker]";
        rejects("handoffs = []", &item, message);
        rejects("handoffs = billing", &item, message);
    }

    #[test]
    fn manages_calls_once_with_every_path() {
        let item = lex(&format!("{DOC}fn boss() {{}}"));
        let a = parse_agent(&lex("manages = [researcher, ::x::editor]"), &item).unwrap();
        assert_eq!(methods(&a), ["manages", "manages"]);
        assert!(a.calls.iter().all(|c| c.shape == Shape::InvokeList));
        let message = "manages = takes a list like [researcher, team::checker]";
        rejects("manages = []", &format!("{DOC}fn boss() {{}}"), message);
        rejects("manages = boss", &format!("{DOC}fn boss() {{}}"), message);
    }

    #[test]
    fn members_take_the_team_mode() {
        let item = format!("{DOC}fn desk() {{}}");
        let a = parse("members = [billing, ::x::tech], team = \"route\"", &item).unwrap();
        assert_eq!(methods(&a), ["members", "members"]);
        assert!(a.calls.iter().all(|c| c.shape == Shape::Members("Route")));
        let a = parse("team = \"broadcast\", members = [billing]", &item).unwrap();
        assert_eq!(a.calls[0].shape, Shape::Members("Broadcast"));
        let a = parse("members = [billing]", &item).unwrap();
        assert_eq!(a.calls[0].shape, Shape::Members("Coordinate"));
        let pieces = expand_agent(&a);
        let Some(Piece::Group(Delim::Brace, body)) = pieces.last() else {
            panic!("the body is the last piece");
        };
        let Piece::Group(Delim::Paren, args) = &body[2] else {
            panic!("members takes arguments");
        };
        assert!(matches!(&args[0], Piece::Group(Delim::Bracket, _)));
        assert!(
            matches!(&args[1], Piece::Code(c) if c == ", ::troupekit::__private::TeamMode::Coordinate")
        );
        let mode = "team = takes one of route, coordinate, collaborate, broadcast, tasks";
        rejects("members = [a], team = \"solo\"", &item, mode);
        rejects("members = [a], team = route", &item, mode);
        rejects(
            "team = \"route\"",
            &item,
            "#[agent] key `team` needs members = [..]",
        );
        let list = "members = takes a list like [billing, tech]";
        rejects("members = billing", &item, list);
        rejects("members = []", &item, list);
        rejects("members = [\"billing\"]", &item, list);
    }

    #[test]
    fn agents_and_delegates_take_paths() {
        let item = format!("{DOC}fn lead() {{}}");
        let agents = "agents = takes a list like [researcher, team::checker]";
        let delegates = "delegates = takes a list like [researcher, team::checker]";
        rejects("agents = researcher", &item, agents);
        rejects("agents = [researcher()]", &item, agents);
        rejects("delegates = writer", &item, delegates);
        rejects("delegates = [\"writer\"]", &item, delegates);
        rejects("delegates = []", &item, delegates);
        assert!(parse("agents = []", &item).unwrap().calls.is_empty());
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
        rejects("x = 1", &format!("{DOC}fn calc() {{}}"), "unknown #[agent] key `x`; known keys: model, tools, evals, limits, checkpoints, max_turns, max_tokens, retries, role, goal, display_name, instructions, laws, auto_laws, memory, knowledge, agents, delegates, handoffs, manages, members, team, name, crate");
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

    #[test]
    fn rejects_bad_auto_laws() {
        rejects(
            "auto_laws = yes",
            &format!("{DOC}fn calc() {{}}"),
            "auto_laws = takes true or false",
        );
    }
}
