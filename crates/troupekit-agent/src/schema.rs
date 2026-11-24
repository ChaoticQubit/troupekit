use serde_json::{json, Map, Value};

/// Compiles a persona `[output]` table into a strict JSON Schema object.
/// Each key is a top-level field and each value a type: `string`, `int`,
/// `number`, `bool`, `list`, `list<T>`, `{a, b: T}`, `enum(x, y)`, with a
/// trailing `?` for nullable. Every object forbids extra keys and requires all.
pub fn compile_output(fields: &toml::Table) -> Result<Value, String> {
    if fields.is_empty() {
        return Err("output has no fields".into());
    }
    let mut props = Vec::with_capacity(fields.len());
    for (name, ty) in fields {
        check_field(name).map_err(|e| format!("{name}: {e}"))?;
        let src = ty
            .as_str()
            .ok_or_else(|| format!("{name}: type must be a string"))?;
        props.push((
            name.clone(),
            parse_type(src).map_err(|e| format!("{name}: {e}"))?,
        ));
    }
    Ok(object(props))
}

/// Checks `value` against a compiled output `schema`.
/// Returns the problems as `"<pointer>: <message>"` lines, at most five,
/// or an empty list when the value is valid.
pub fn validate(schema: &Value, value: &Value) -> Vec<String> {
    match troupekit_tools::compile_schema(schema) {
        Ok(validator) => troupekit_tools::schema_errors(&validator, value),
        Err(e) => vec![format!("output schema does not compile: {e}")],
    }
}

/// Parses a model's answer as JSON. Surrounding whitespace and one code
/// fence (with or without a `json` tag) are removed first.
/// Returns the value, or why the text is not JSON.
pub fn parse_json(text: &str) -> Result<Value, String> {
    let t = text.trim();
    let inner = t
        .strip_prefix("```")
        .and_then(|rest| rest.strip_suffix("```"))
        .map(|body| body.strip_prefix("json").unwrap_or(body))
        .unwrap_or(t);
    serde_json::from_str(inner.trim()).map_err(|e| format!("answer is not JSON: {e}"))
}

fn parse_type(src: &str) -> Result<Value, String> {
    let mut p = Parser {
        s: src.as_bytes(),
        i: 0,
    };
    let ty = p.ty()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("unexpected `{}` in type `{src}`", &src[p.i..]));
    }
    Ok(ty)
}

fn check_field(name: &str) -> Result<(), String> {
    if !crate::dsl::is_ident(name) {
        return Err(format!("field name `{name}` must match [a-z_][a-z0-9_]*"));
    }
    Ok(())
}

fn object(props: Vec<(String, Value)>) -> Value {
    let required: Vec<Value> = props
        .iter()
        .map(|(k, _)| Value::String(k.clone()))
        .collect();
    let properties: Map<String, Value> = props.into_iter().collect();
    json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false})
}

fn nullable(mut ty: Value) -> Value {
    let base = ty["type"].clone();
    ty["type"] = json!([base, "null"]);
    if let Some(values) = ty.get_mut("enum").and_then(Value::as_array_mut) {
        values.push(Value::Null);
    }
    ty
}

struct Parser<'a> {
    s: &'a [u8],
    i: usize,
}

impl Parser<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> bool {
        self.ws();
        if self.s.get(self.i) == Some(&c) {
            self.i += 1;
            return true;
        }
        false
    }

    fn expect(&mut self, c: u8) -> Result<(), String> {
        if self.eat(c) {
            return Ok(());
        }
        Err(format!("expected `{}` at column {}", c as char, self.i + 1))
    }

    fn word(&mut self) -> Result<String, String> {
        self.ws();
        let start = self.i;
        while self.i < self.s.len()
            && (self.s[self.i].is_ascii_alphanumeric()
                || self.s[self.i] == b'_'
                || self.s[self.i] == b'-')
        {
            self.i += 1;
        }
        if start == self.i {
            return Err(format!("expected a name at column {}", self.i + 1));
        }
        Ok(String::from_utf8_lossy(&self.s[start..self.i]).into_owned())
    }

    fn ty(&mut self) -> Result<Value, String> {
        let base = if self.eat(b'{') {
            self.object()?
        } else {
            self.named()?
        };
        Ok(if self.eat(b'?') { nullable(base) } else { base })
    }

    fn named(&mut self) -> Result<Value, String> {
        let name = self.word()?;
        Ok(match name.as_str() {
            "string" => json!({"type": "string"}),
            "int" => json!({"type": "integer"}),
            "number" => json!({"type": "number"}),
            "bool" => json!({"type": "boolean"}),
            "list" if self.eat(b'<') => {
                let items = self.ty()?;
                self.expect(b'>')?;
                json!({"type": "array", "items": items})
            }
            "list" => json!({"type": "array", "items": {"type": "string"}}),
            "enum" => {
                self.expect(b'(')?;
                let mut values = vec![Value::String(self.word()?)];
                while self.eat(b',') {
                    values.push(Value::String(self.word()?));
                }
                self.expect(b')')?;
                json!({"type": "string", "enum": values})
            }
            other => return Err(format!("unknown type `{other}`")),
        })
    }

    fn object(&mut self) -> Result<Value, String> {
        let mut props: Vec<(String, Value)> = Vec::new();
        loop {
            let name = self.word()?;
            check_field(&name)?;
            if props.iter().any(|(k, _)| *k == name) {
                return Err(format!("field `{name}` repeats"));
            }
            let ty = if self.eat(b':') {
                self.ty()?
            } else {
                json!({"type": "string"})
            };
            props.push((name, ty));
            if !self.eat(b',') {
                break;
            }
        }
        self.expect(b'}')?;
        Ok(object(props))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(src: &str) -> toml::Table {
        src.parse().unwrap()
    }

    #[test]
    fn compiles_testy_shorthand() {
        let s = compile_output(&table(
            "actions = \"list<{name, minutes: int}>\"\nnotes = \"string\"",
        ))
        .unwrap();
        assert_eq!(
            s,
            json!({
                "type": "object",
                "properties": {
                    "actions": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "minutes": {"type": "integer"}},
                        "required": ["name", "minutes"],
                        "additionalProperties": false
                    }},
                    "notes": {"type": "string"}
                },
                "required": ["actions", "notes"],
                "additionalProperties": false
            })
        );
    }

    #[test]
    fn nullable_and_enum() {
        let s = compile_output(&table(
            "level = \"enum(low, high)?\"\ndone = \"bool\"\ntags = \"list\"\nscore = \"number?\"",
        ))
        .unwrap();
        assert_eq!(
            s["properties"]["level"],
            json!({"type": ["string", "null"], "enum": ["low", "high", null]})
        );
        assert_eq!(s["properties"]["done"], json!({"type": "boolean"}));
        assert_eq!(
            s["properties"]["tags"],
            json!({"type": "array", "items": {"type": "string"}})
        );
        assert_eq!(
            s["properties"]["score"],
            json!({"type": ["number", "null"]})
        );
    }

    #[test]
    fn rejects_bad_types() {
        for bad in [
            "a = \"lst\"",
            "a = \"list<\"",
            "a = \"enum()\"",
            "Bad = \"string\"",
            "a = \"{x, x}\"",
            "a = \"string extra\"",
            "a = 3",
            "",
        ] {
            assert!(compile_output(&table(bad)).is_err(), "{bad}");
        }
    }

    #[test]
    fn parse_json_strips_one_fence() {
        assert_eq!(parse_json(" {\"a\":1} ").unwrap(), json!({"a": 1}));
        assert_eq!(
            parse_json("```json\n{\"a\":1}\n```").unwrap(),
            json!({"a": 1})
        );
        assert_eq!(parse_json("```\n[1]\n```").unwrap(), json!([1]));
        assert!(parse_json("Sure! {\"a\":1}")
            .unwrap_err()
            .starts_with("answer is not JSON"));
    }

    #[test]
    fn validate_lists_errors() {
        let s = compile_output(&table("actions = \"list<{name, minutes: int}>\"")).unwrap();
        assert!(validate(&s, &json!({"actions": [{"name": "run", "minutes": 20}]})).is_empty());
        let errs = validate(
            &s,
            &json!({"actions": [{"name": "run", "minutes": "x"}], "extra": 1}),
        );
        assert!(
            errs.iter().any(|e| e.starts_with("/actions/0/minutes")),
            "{errs:?}"
        );
        assert!(errs.iter().any(|e| e.contains("extra")), "{errs:?}");
    }
}
