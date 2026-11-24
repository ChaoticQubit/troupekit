use serde::{Deserialize, Serialize};
use serde_json::Value;

/// One function call written in a persona, such as `min_items(actions, 3)`
/// or `judge('Plan is safe', 7)`. Quoted strings and bare words become JSON
/// strings, numbers become JSON numbers, in the order written.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Call {
    pub name: String,
    pub args: Vec<Value>,
}

/// Parses `src` as `name(arg, ...)`: a name of `[a-z_][a-z0-9_]*`, then
/// arguments that are single- or double-quoted strings, numbers or bare
/// words of `[A-Za-z0-9_./-]`. Returns the call, or why it does not parse.
pub fn parse_call(src: &str) -> Result<Call, String> {
    let s = src.trim();
    let open = s
        .find('(')
        .ok_or_else(|| format!("`{s}` is not a call like name(arg, ...)"))?;
    let name = s[..open].trim();
    if !is_ident(name) {
        return Err(format!("call name `{name}` must match [a-z_][a-z0-9_]*"));
    }
    let inner = s[open + 1..]
        .strip_suffix(')')
        .ok_or_else(|| format!("`{s}` must end with `)`"))?;
    let mut args = Vec::new();
    let mut rest = inner.trim_start();
    while !rest.is_empty() {
        let (arg, after) = arg(rest)?;
        args.push(arg);
        rest = after.trim_start();
        if let Some(next) = rest.strip_prefix(',') {
            rest = next.trim_start();
            if rest.is_empty() {
                return Err("trailing comma".into());
            }
        } else if !rest.is_empty() {
            return Err(format!("expected `,` before `{rest}`"));
        }
    }
    Ok(Call {
        name: name.to_string(),
        args,
    })
}

/// Reports whether `s` is an identifier of `[a-z_][a-z0-9_]*`.
/// Call names, eval names and output field names all follow it.
/// Returns false for an empty string.
pub fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    chars
        .next()
        .is_some_and(|c| c.is_ascii_lowercase() || c == '_')
        && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn arg(s: &str) -> Result<(Value, &str), String> {
    let first = s.chars().next().expect("caller passes a non-empty string");
    if first == '\'' || first == '"' {
        let body = &s[1..];
        let end = body
            .find(first)
            .ok_or_else(|| format!("unclosed string `{s}`"))?;
        return Ok((Value::String(body[..end].to_string()), &body[end + 1..]));
    }
    let end = s
        .find(|c: char| !(c.is_ascii_alphanumeric() || "_./-".contains(c)))
        .unwrap_or(s.len());
    if end == 0 {
        return Err(format!("unexpected `{s}`"));
    }
    let word = &s[..end];
    let value = if let Ok(n) = word.parse::<i64>() {
        Value::from(n)
    } else if let Some(n) = word.parse::<f64>().ok().filter(|n| n.is_finite()) {
        Value::from(n)
    } else {
        Value::String(word.to_string())
    };
    Ok((value, &s[end..]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_calls() {
        assert_eq!(
            parse_call("min_items(actions, 3)").unwrap(),
            Call {
                name: "min_items".into(),
                args: vec![json!("actions"), json!(3)]
            }
        );
        assert_eq!(
            parse_call(" judge('Plan is safe, short', 7.5) ")
                .unwrap()
                .args,
            vec![json!("Plan is safe, short"), json!(7.5)]
        );
        assert_eq!(
            parse_call("no_overlap()").unwrap().args,
            Vec::<Value>::new()
        );
        assert_eq!(
            parse_call("within(actions.start, \"a'b\")").unwrap().args,
            vec![json!("actions.start"), json!("a'b")]
        );
    }

    #[test]
    fn rejects_bad_calls() {
        for bad in [
            "plain text",
            "Bad(x)",
            "f(x",
            "f('x)",
            "f(x,)",
            "f(x y)",
            "f(@)",
            "(x)",
        ] {
            assert!(parse_call(bad).is_err(), "{bad}");
        }
    }
}
