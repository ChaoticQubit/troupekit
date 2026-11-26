use crate::{Grammar, LocalError};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fmt::Write as _;

const ROOT: &str = "root";
const MAX_DEPTH: usize = 64;
const MAX_REF_DEPTH: usize = 16;
const MAX_ITEMS: u64 = 64;
const MAX_LENGTH: u64 = 1024;
const MAX_ALTERNATIVES: usize = 64;
const MAX_DIGITS: usize = 16;
const SEP: &str = " ws \",\" ws ";
const SUPPORTED: [&str; 16] = [
    "type",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "minItems",
    "maxItems",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "$ref",
    "maxLength",
    "minimum",
    "maximum",
    "$defs",
];
const ANNOTATIONS: [&str; 8] = [
    "title",
    "description",
    "default",
    "examples",
    "$schema",
    "$id",
    "$comment",
    "format",
];
const BASE: &str = r#"ws ::= ( " " | "\n" [ \t]{0,20} )?
null ::= "null"
boolean ::= "true" | "false"
char ::= [^"\\\x7F\x00-\x1F] | [\\] (["\\bfnrt] | "u" [0-9a-fA-F]{4})
string ::= "\"" char* "\""
integer ::= "-"? ( "0" | [1-9] [0-9]{0,15} )
number ::= integer ( "." [0-9]{1,16} )? ( [eE] [-+]? [0-9]{1,3} )?
"#;

/// Converts a JSON Schema into a GBNF grammar whose start rule is `root`.
/// Input: the schema; the supported keywords are listed in the 0.3 plan
/// (R2). Output: the grammar, or `LocalError::Grammar` naming the first
/// unsupported keyword, a broken bound, a cyclic `$ref` or the size cap.
pub fn schema_to_gbnf(schema: &Value) -> Result<Grammar, LocalError> {
    let mut builder = GbnfBuilder::new();
    let root = builder.schema(schema)?;
    assert!(!root.is_empty(), "every schema has an expression");
    builder.finish(&root)
}

fn fail(message: impl Into<String>) -> LocalError {
    LocalError::Grammar(message.into())
}

fn too_big(bytes: usize) -> LocalError {
    assert!(bytes > Grammar::MAX_BYTES);
    fail(format!(
        "grammar is {bytes} bytes, limit {}",
        Grammar::MAX_BYTES
    ))
}

/// Quotes `text` as one GBNF literal. Quotes, backslashes and control
/// characters are escaped, so no input can end the literal early.
/// Output: the literal with its surrounding double quotes.
pub(crate) fn literal(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if u32::from(c) < 0x20 || u32::from(c) == 0x7F => {
                write!(out, "\\x{:02X}", u32::from(c)).expect("writing to a String");
            }
            c => out.push(c),
        }
    }
    out.push('"');
    assert!(
        out.len() >= 2 && !out.contains('\n'),
        "a literal is one line"
    );
    out
}

fn json_literal(value: &Value) -> String {
    literal(&serde_json::to_string(value).expect("a JSON value serializes"))
}

fn digits(bound: f64) -> usize {
    assert!(bound.is_finite());
    let whole = bound.abs().trunc();
    let count = if whole >= 1e16 {
        MAX_DIGITS
    } else {
        whole.to_string().len().min(MAX_DIGITS)
    };
    assert!((1..=MAX_DIGITS).contains(&count));
    count
}

/// Collects GBNF rules for one or more schemas into one grammar, sharing
/// the base rules. Rule names are generated (`r0`, `r1`, ...), never
/// taken from a schema. `finish` checks the 256 KiB cap.
pub(crate) struct GbnfBuilder {
    rules: Vec<Option<String>>,
    bytes: usize,
    defs: Option<Map<String, Value>>,
    refs: BTreeMap<String, String>,
    stack: Vec<String>,
}

impl GbnfBuilder {
    pub(crate) fn new() -> Self {
        Self {
            rules: Vec::new(),
            bytes: BASE.len(),
            defs: None,
            refs: BTreeMap::new(),
            stack: Vec::new(),
        }
    }

    /// Adds the rules for `schema`, whose `$ref`s resolve against its own
    /// top-level `$defs`. Output: the GBNF expression that matches a
    /// value of the schema, to use in `rule` or `finish`.
    pub(crate) fn schema(&mut self, schema: &Value) -> Result<String, LocalError> {
        assert!(self.stack.is_empty(), "no $ref is open between schemas");
        self.defs = match schema.get("$defs") {
            None => None,
            Some(Value::Object(defs)) => Some(defs.clone()),
            Some(_) => return Err(fail("`$defs` must be an object")),
        };
        self.refs.clear();
        let expr = self.value(schema, 0)?;
        assert!(self.stack.is_empty(), "every $ref is closed");
        Ok(expr)
    }

    /// Adds one rule with GBNF body `body` under a generated name.
    /// Output: the rule name, or the size-cap error.
    pub(crate) fn rule(&mut self, body: String) -> Result<String, LocalError> {
        let id = self.alloc();
        self.set(id, body)
    }

    /// Builds the grammar with `root ::= <root>`, every rule, then the
    /// base rules. Output: the checked `Grammar`, or `LocalError::Grammar`.
    pub(crate) fn finish(self, root: &str) -> Result<Grammar, LocalError> {
        let mut text = format!("{ROOT} ::= {root}\n");
        for rule in self.rules {
            text.push_str(&rule.expect("every allocated rule is set"));
        }
        text.push_str(BASE);
        if text.len() > Grammar::MAX_BYTES {
            return Err(too_big(text.len()));
        }
        Grammar::new(&text, ROOT)
    }

    fn alloc(&mut self) -> usize {
        self.rules.push(None);
        self.rules.len() - 1
    }

    fn set(&mut self, id: usize, body: String) -> Result<String, LocalError> {
        assert!(self.rules[id].is_none(), "a rule is set once");
        assert!(!body.is_empty());
        let line = format!("r{id} ::= {body}\n");
        self.bytes += line.len();
        if self.bytes > Grammar::MAX_BYTES {
            return Err(too_big(self.bytes));
        }
        self.rules[id] = Some(line);
        Ok(format!("r{id}"))
    }

    fn value(&mut self, node: &Value, depth: usize) -> Result<String, LocalError> {
        if depth > MAX_DEPTH {
            return Err(fail(format!("schema nests deeper than {MAX_DEPTH} levels")));
        }
        let obj = match node {
            Value::Object(obj) => obj,
            Value::Bool(true) => {
                return Err(fail("schema `true` allows any value; give it a `type`"))
            }
            Value::Bool(false) => return Err(fail("schema `false` allows no value")),
            _ => return Err(fail("a schema must be a JSON object")),
        };
        let known = |k: &str| SUPPORTED.contains(&k) || ANNOTATIONS.contains(&k);
        if let Some(key) = obj.keys().find(|k| !known(k)) {
            return Err(fail(format!(
                "JSON Schema keyword `{key}` is not supported"
            )));
        }
        if let Some(target) = obj.get("$ref") {
            return self.reference(target, depth);
        }
        if let Some(value) = obj.get("const") {
            return Ok(json_literal(value));
        }
        if let Some(values) = obj.get("enum") {
            return self.enumeration(values);
        }
        match (obj.get("anyOf"), obj.get("oneOf")) {
            (Some(_), Some(_)) => {
                return Err(fail("a schema cannot hold both `anyOf` and `oneOf`"))
            }
            (Some(list), None) | (None, Some(list)) => return self.alternatives(list, depth),
            (None, None) => {}
        }
        match obj.get("type") {
            Some(Value::String(name)) => self.typed(obj, name, depth),
            Some(Value::Array(names)) => self.type_list(obj, names, depth),
            Some(_) => Err(fail("`type` must be a string or a list of strings")),
            None => Err(fail(
                "a schema needs `type`, `enum`, `const`, `anyOf`, `oneOf` or `$ref`",
            )),
        }
    }

    fn typed(
        &mut self,
        obj: &Map<String, Value>,
        name: &str,
        depth: usize,
    ) -> Result<String, LocalError> {
        match name {
            "object" => self.object(obj, depth),
            "array" => self.array(obj, depth),
            "string" => self.string(obj),
            "integer" => self.numeric(obj, true),
            "number" => self.numeric(obj, false),
            "boolean" | "null" => Ok(name.to_string()),
            other => Err(fail(format!("`type` `{other}` is not a JSON Schema type"))),
        }
    }

    fn type_list(
        &mut self,
        obj: &Map<String, Value>,
        names: &[Value],
        depth: usize,
    ) -> Result<String, LocalError> {
        if names.is_empty() || names.len() > MAX_ALTERNATIVES {
            return Err(fail(format!(
                "`type` lists 1 to {MAX_ALTERNATIVES} types, not {}",
                names.len()
            )));
        }
        let mut alts = Vec::with_capacity(names.len());
        for name in names {
            let name = name
                .as_str()
                .ok_or_else(|| fail("`type` must be a string or a list of strings"))?;
            alts.push(self.typed(obj, name, depth)?);
        }
        assert_eq!(alts.len(), names.len());
        self.rule(alts.join(" | "))
    }

    fn enumeration(&mut self, values: &Value) -> Result<String, LocalError> {
        let values = values
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| fail("`enum` must be a non-empty list"))?;
        let alts: Vec<String> = values.iter().map(json_literal).collect();
        assert_eq!(alts.len(), values.len());
        match alts.as_slice() {
            [one] => Ok(one.clone()),
            _ => self.rule(alts.join(" | ")),
        }
    }

    fn alternatives(&mut self, list: &Value, depth: usize) -> Result<String, LocalError> {
        let list = list
            .as_array()
            .filter(|l| !l.is_empty() && l.len() <= MAX_ALTERNATIVES)
            .ok_or_else(|| {
                fail(format!(
                    "`anyOf` and `oneOf` take a list of 1 to {MAX_ALTERNATIVES} schemas"
                ))
            })?;
        let mut alts = Vec::with_capacity(list.len());
        for sub in list {
            alts.push(self.value(sub, depth + 1)?);
        }
        assert_eq!(alts.len(), list.len());
        self.rule(alts.join(" | "))
    }

    fn reference(&mut self, target: &Value, depth: usize) -> Result<String, LocalError> {
        let path = target
            .as_str()
            .ok_or_else(|| fail("`$ref` must be a string"))?;
        let Some(name) = path.strip_prefix("#/$defs/") else {
            return Err(fail(format!(
                "`$ref` `{path}` is not supported; use #/$defs/<name>"
            )));
        };
        let name = name.replace("~1", "/").replace("~0", "~");
        if let Some(done) = self.refs.get(&name) {
            return Ok(done.clone());
        }
        if self.stack.contains(&name) {
            return Err(fail(format!("`$ref` `{path}` is cyclic")));
        }
        if self.stack.len() >= MAX_REF_DEPTH {
            return Err(fail(format!("`$ref` chain is deeper than {MAX_REF_DEPTH}")));
        }
        let def = self
            .defs
            .as_ref()
            .and_then(|defs| defs.get(&name))
            .cloned()
            .ok_or_else(|| fail(format!("`$ref` `{path}` names no entry in `$defs`")))?;
        self.stack.push(name.clone());
        let expr = self.value(&def, depth + 1);
        let closed = self.stack.pop();
        assert_eq!(
            closed.as_deref(),
            Some(name.as_str()),
            "refs close in order"
        );
        let expr = expr?;
        self.refs.insert(name, expr.clone());
        Ok(expr)
    }

    fn string(&mut self, obj: &Map<String, Value>) -> Result<String, LocalError> {
        let Some(limit) = obj.get("maxLength") else {
            return Ok("string".to_string());
        };
        let limit = limit.as_u64().filter(|n| *n <= MAX_LENGTH).ok_or_else(|| {
            fail(format!(
                "`maxLength` must be a whole number at most {MAX_LENGTH}"
            ))
        })?;
        assert!(limit <= MAX_LENGTH);
        self.rule(format!(r#""\"" char{{0,{limit}}} "\"""#))
    }

    fn numeric(&mut self, obj: &Map<String, Value>, integer: bool) -> Result<String, LocalError> {
        let bound = |key: &str| match obj.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_f64()
                .filter(|f| f.is_finite())
                .map(Some)
                .ok_or_else(|| fail(format!("`{key}` must be a number"))),
        };
        let (min, max) = (bound("minimum")?, bound("maximum")?);
        if min.is_none() && max.is_none() {
            return Ok(if integer { "integer" } else { "number" }.to_string());
        }
        if let (Some(low), Some(high)) = (min, max) {
            if low > high {
                return Err(fail("`minimum` is above `maximum`"));
            }
        }
        let mut alts = Vec::new();
        if max.is_none_or(|m| m >= 0.0) {
            let count = max.filter(|m| *m >= 0.0).map_or(MAX_DIGITS, digits);
            alts.push(match count {
                1 => "[0-9]".to_string(),
                n => format!(r#""0" | [1-9] [0-9]{{0,{}}}"#, n - 1),
            });
        }
        if min.is_none_or(|m| m < 0.0) {
            let count = min.filter(|m| *m < 0.0).map_or(MAX_DIGITS, digits);
            alts.push(match count {
                1 => r#""-" [1-9]"#.to_string(),
                n => format!(r#""-" [1-9] [0-9]{{0,{}}}"#, n - 1),
            });
        }
        assert!(!alts.is_empty(), "minimum <= maximum leaves a sign");
        let body = alts.join(" | ");
        self.rule(if integer {
            body
        } else {
            format!(r#"( {body} ) ( "." [0-9]{{1,16}} )?"#)
        })
    }

    fn object(&mut self, obj: &Map<String, Value>, depth: usize) -> Result<String, LocalError> {
        match obj.get("additionalProperties") {
            None | Some(Value::Bool(false)) => {}
            Some(_) => {
                return Err(fail(
                    "`additionalProperties` other than false is not supported",
                ))
            }
        }
        let empty = Map::new();
        let props = match obj.get("properties") {
            None => &empty,
            Some(Value::Object(props)) => props,
            Some(_) => return Err(fail("`properties` must be an object")),
        };
        let required: Vec<&str> = match obj.get("required") {
            None => Vec::new(),
            Some(Value::Array(names)) => names
                .iter()
                .map(|n| {
                    n.as_str()
                        .ok_or_else(|| fail("`required` must list strings"))
                })
                .collect::<Result<_, _>>()?,
            Some(_) => return Err(fail("`required` must be a list of strings")),
        };
        if let Some(missing) = required.iter().find(|r| !props.contains_key(**r)) {
            return Err(fail(format!(
                "`required` names `{missing}`, which is not in `properties`"
            )));
        }
        let id = self.alloc();
        let (mut req, mut opt) = (Vec::new(), Vec::new());
        for (key, sub) in props {
            let name = json_literal(&Value::String(key.clone()));
            let pair = format!("{name} ws \":\" ws {}", self.value(sub, depth + 1)?);
            if required.contains(&key.as_str()) {
                req.push(pair);
            } else {
                opt.push(pair);
            }
        }
        assert_eq!(req.len() + opt.len(), props.len());
        let tail = self.optional_tail(&opt)?;
        let body = match (req.is_empty(), tail) {
            (true, None) => String::new(),
            (true, Some(tail)) => format!("{tail}? "),
            (false, None) => format!("{} ", req.join(SEP)),
            (false, Some(tail)) => format!("{} ( ws \",\" ws {tail} )? ", req.join(SEP)),
        };
        if body.is_empty() {
            return self.set(id, r#""{" ws "}""#.to_string());
        }
        self.set(id, format!(r#""{{" ws {body}ws "}}""#))
    }

    fn optional_tail(&mut self, pairs: &[String]) -> Result<Option<String>, LocalError> {
        let mut next: Option<String> = None;
        for pair in pairs.iter().rev() {
            let body = match &next {
                None => pair.clone(),
                Some(rest) => format!("{pair} ( ws \",\" ws {rest} )? | {rest}"),
            };
            next = Some(self.rule(body)?);
        }
        assert_eq!(next.is_some(), !pairs.is_empty());
        Ok(next)
    }

    fn array(&mut self, obj: &Map<String, Value>, depth: usize) -> Result<String, LocalError> {
        let items = obj
            .get("items")
            .ok_or_else(|| fail("an array schema needs `items`"))?;
        let count = |key: &str| match obj.get(key) {
            None => Ok(None),
            Some(v) => v
                .as_u64()
                .filter(|n| *n <= MAX_ITEMS)
                .map(Some)
                .ok_or_else(|| {
                    fail(format!(
                        "`{key}` must be a whole number at most {MAX_ITEMS}"
                    ))
                }),
        };
        let (min, max) = (count("minItems")?.unwrap_or(0), count("maxItems")?);
        if max.is_some_and(|m| min > m) {
            return Err(fail("`minItems` is above `maxItems`"));
        }
        let id = self.alloc();
        let item = self.value(items, depth + 1)?;
        if max == Some(0) {
            return self.set(id, r#""[" ws "]""#.to_string());
        }
        let more = format!("( ws \",\" ws {item} )");
        let seq = match (min.saturating_sub(1), max.map(|m| m - 1)) {
            (_, Some(0)) => item,
            (0, None) => format!("{item} {more}*"),
            (low, None) => format!("{item} {more}{{{low},}}"),
            (low, Some(high)) => format!("{item} {more}{{{low},{high}}}"),
        };
        assert!(max.is_none_or(|m| m >= 1));
        if min == 0 {
            return self.set(id, format!(r#""[" ws ( {seq} )? ws "]""#));
        }
        self.set(id, format!(r#""[" ws {seq} ws "]""#))
    }
}
