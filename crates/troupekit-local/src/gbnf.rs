use crate::{Grammar, LocalError};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

const ROOT: &str = "root";
const MAX_DEPTH: usize = 64;
const MAX_REF_DEPTH: usize = 16;
const MAX_ITEMS: u64 = 64;
const MAX_LENGTH: u64 = 1024;
const MAX_ALTERNATIVES: usize = 64;
const MAX_DIGITS: u32 = 16;
const MAX_MAGNITUDE: u64 = 10u64.pow(MAX_DIGITS) - 1;
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

fn kind_of(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn type_allows(name: &str, value: &Value) -> bool {
    match name {
        "integer" => value.as_f64().is_some_and(|f| f.fract() == 0.0),
        "number" => value.is_number(),
        other => kind_of(value) == other,
    }
}

/// JSON Schema equality: numbers by mathematical value (exact for
/// integers past 2^53), arrays element by element, objects by key set
/// and values. Everything else compares as JSON.
fn same_value(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => same_number(x, y),
        (Value::Array(x), Value::Array(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| same_value(p, q))
        }
        (Value::Object(x), Value::Object(y)) => {
            x.len() == y.len()
                && x.iter()
                    .all(|(k, v)| y.get(k).is_some_and(|w| same_value(v, w)))
        }
        _ => a == b,
    }
}

fn whole_number(n: &serde_json::Number) -> Option<i128> {
    n.as_i64()
        .map(i128::from)
        .or_else(|| n.as_u64().map(i128::from))
}

fn float_is(f: f64, whole: i128) -> bool {
    f.is_finite() && f.fract() == 0.0 && f.abs() < 1.7e38 && f as i128 == whole
}

fn same_number(x: &serde_json::Number, y: &serde_json::Number) -> bool {
    match (whole_number(x), whole_number(y)) {
        (Some(p), Some(q)) => p == q,
        (Some(p), None) => y.as_f64().is_some_and(|f| float_is(f, p)),
        (None, Some(q)) => x.as_f64().is_some_and(|f| float_is(f, q)),
        (None, None) => x.as_f64() == y.as_f64(),
    }
}

/// Reads bound `key` as an exact integer: integer JSON numbers as they
/// are, fractions rounded up (`round_up`) or down. Output: the bound,
/// `None` when absent, or an error past 16 digits or for a non-number.
fn integer_bound(
    obj: &Map<String, Value>,
    key: &str,
    round_up: bool,
) -> Result<Option<i64>, LocalError> {
    let Some(value) = obj.get(key) else {
        return Ok(None);
    };
    let bad = || {
        fail(format!(
            "`{key}` must be a number of at most {MAX_DIGITS} digits"
        ))
    };
    let whole = match (value.as_i64(), value.as_f64()) {
        (Some(exact), _) => exact,
        (None, _) if value.is_u64() => return Err(bad()),
        (None, Some(f)) if f.is_finite() && f.abs() < 1e17 => {
            let rounded = if round_up { f.ceil() } else { f.floor() };
            rounded as i64
        }
        _ => return Err(bad()),
    };
    if whole.unsigned_abs() > MAX_MAGNITUDE {
        return Err(bad());
    }
    assert!(whole.unsigned_abs() <= MAX_MAGNITUDE);
    Ok(Some(whole))
}

fn kind_of_name(name: &str) -> &'static str {
    match name {
        "null" => "null",
        "boolean" => "boolean",
        "string" => "string",
        "array" => "array",
        "object" => "object",
        _ => "number",
    }
}

fn def_name(path: &str) -> Option<String> {
    let name = path.strip_prefix("#/$defs/")?;
    Some(name.replace("~1", "/").replace("~0", "~"))
}

/// Refuses a keyword next to `form` that the grammar cannot apply as
/// well: every supported constraint except `$defs` and `allowed`.
/// Annotations are always accepted.
fn only_with(obj: &Map<String, Value>, form: &str, allowed: &[&str]) -> Result<(), LocalError> {
    assert!(SUPPORTED.contains(&form));
    let extra = obj
        .keys()
        .map(String::as_str)
        .find(|k| SUPPORTED.contains(k) && *k != "$defs" && *k != form && !allowed.contains(k));
    match extra {
        Some(key) => Err(fail(format!(
            "`{key}` next to `{form}` is not supported: the grammar cannot apply both"
        ))),
        None => Ok(()),
    }
}

/// Checks that literal `value` (from `const` or `enum`) has a type the
/// sibling `type` keyword allows. Output: `Ok`, or an error naming
/// `type`.
fn check_literal(obj: &Map<String, Value>, form: &str, value: &Value) -> Result<(), LocalError> {
    let names: Vec<&str> = match obj.get("type") {
        None => return Ok(()),
        Some(Value::String(name)) => vec![name.as_str()],
        Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).collect(),
        Some(_) => return Err(fail("`type` must be a string or a list of strings")),
    };
    if names.iter().any(|name| type_allows(name, value)) {
        return Ok(());
    }
    Err(fail(format!(
        "`{form}` value {value} does not match `type`"
    )))
}

/// Writes the fixed-width digit strings from `low` to `high` (ASCII
/// digits, same length, `low <= high`) as one GBNF expression that
/// matches exactly the strings in that range, leading zeros included.
fn fixed(low: &[u8], high: &[u8]) -> String {
    assert_eq!(low.len(), high.len());
    assert!(!low.is_empty() && low <= high);
    let (a, b) = (low[0], high[0]);
    if low.len() == 1 {
        return class(a, b);
    }
    let (low_rest, high_rest) = (&low[1..], &high[1..]);
    if a == b {
        return format!("{} {}", class(a, a), fixed(low_rest, high_rest));
    }
    let mut alts = Vec::new();
    let mut first = a;
    if low_rest.iter().any(|d| *d != b'0') {
        let nines = vec![b'9'; low_rest.len()];
        alts.push(format!("{} {}", class(a, a), fixed(low_rest, &nines)));
        first = a + 1;
    }
    let mut last = b;
    let mut tail = None;
    if high_rest.iter().any(|d| *d != b'9') {
        let zeros = vec![b'0'; high_rest.len()];
        tail = Some(format!("{} {}", class(b, b), fixed(&zeros, high_rest)));
        last = b - 1;
    }
    if first <= last {
        alts.push(format!(
            "{} {}",
            class(first, last),
            any_digits(low_rest.len())
        ));
    }
    alts.extend(tail);
    group(alts)
}

fn class(a: u8, b: u8) -> String {
    assert!(a.is_ascii_digit() && b.is_ascii_digit() && a <= b);
    if a == b {
        format!("\"{}\"", char::from(a))
    } else {
        format!("[{}-{}]", char::from(a), char::from(b))
    }
}

fn any_digits(count: usize) -> String {
    assert!(count >= 1);
    if count == 1 {
        "[0-9]".to_string()
    } else {
        format!("[0-9]{{{count}}}")
    }
}

fn group(mut parts: Vec<String>) -> String {
    assert!(!parts.is_empty());
    if parts.len() == 1 {
        return parts.pop().expect("one part");
    }
    format!("( {} )", parts.join(" | "))
}

fn digit_count(value: u64) -> usize {
    value.to_string().len()
}

/// Alternatives matching exactly the integers `low..=high` written without
/// leading zeros; `high == MAX_MAGNITUDE` stands for "no upper bound" and
/// folds every longer length into one alternative.
fn non_negative(low: u64, high: u64) -> Vec<String> {
    assert!(low <= high && high <= MAX_MAGNITUDE);
    let (low_len, high_len) = (digit_count(low), digit_count(high));
    let mut parts = Vec::new();
    for len in low_len..=high_len {
        if high == MAX_MAGNITUDE && len > low_len {
            parts.push(format!("[1-9] [0-9]{{{},{}}}", len - 1, MAX_DIGITS - 1));
            break;
        }
        let from = if len == low_len {
            low
        } else {
            10u64.pow(len as u32 - 1)
        };
        let to = if len == high_len {
            high
        } else {
            10u64.pow(len as u32) - 1
        };
        parts.push(fixed(
            from.to_string().as_bytes(),
            to.to_string().as_bytes(),
        ));
    }
    assert!(!parts.is_empty());
    parts
}

/// What a `oneOf` branch can match, read without building rules: JSON
/// kinds, the literal values of a `const` or `enum`, and required
/// properties fixed by `const` (tags). Unknown shapes match anything.
struct Shape {
    kinds: BTreeSet<&'static str>,
    literals: Option<Vec<Value>>,
    tags: BTreeMap<String, Value>,
}

impl Shape {
    fn any() -> Self {
        let kinds = ["null", "boolean", "number", "string", "array", "object"];
        Shape {
            kinds: kinds.into_iter().collect(),
            literals: None,
            tags: BTreeMap::new(),
        }
    }

    fn disjoint(&self, other: &Shape) -> bool {
        let common: Vec<&str> = self.kinds.intersection(&other.kinds).copied().collect();
        if common.is_empty() {
            return true;
        }
        if let (Some(a), Some(b)) = (&self.literals, &other.literals) {
            return a.iter().all(|x| !b.iter().any(|y| same_value(x, y)));
        }
        common == ["object"]
            && self
                .tags
                .iter()
                .any(|(key, value)| other.tags.get(key).is_some_and(|w| !same_value(value, w)))
    }
}

fn tags_of(obj: &Map<String, Value>) -> BTreeMap<String, Value> {
    let required: Vec<&str> = obj
        .get("required")
        .and_then(Value::as_array)
        .map(|r| r.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();
    let mut tags = BTreeMap::new();
    if let Some(props) = obj.get("properties").and_then(Value::as_object) {
        for (key, sub) in props.iter().filter(|(k, _)| required.contains(&k.as_str())) {
            if let Some(value) = sub.get("const") {
                tags.insert(key.clone(), value.clone());
            }
        }
    }
    assert!(tags.keys().all(|k| required.contains(&k.as_str())));
    tags
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
            only_with(obj, "$ref", &[])?;
            return self.reference(target, depth);
        }
        match (obj.get("const"), obj.get("enum")) {
            (Some(_), Some(_)) => return Err(fail("a schema cannot hold both `const` and `enum`")),
            (Some(value), None) => {
                only_with(obj, "const", &["type"])?;
                check_literal(obj, "const", value)?;
                return Ok(json_literal(value));
            }
            (None, Some(values)) => {
                only_with(obj, "enum", &["type"])?;
                return self.enumeration(obj, values);
            }
            (None, None) => {}
        }
        match (obj.get("anyOf"), obj.get("oneOf")) {
            (Some(_), Some(_)) => {
                return Err(fail("a schema cannot hold both `anyOf` and `oneOf`"))
            }
            (Some(list), None) => {
                only_with(obj, "anyOf", &[])?;
                return self.alternatives(list, depth);
            }
            (None, Some(list)) => {
                only_with(obj, "oneOf", &[])?;
                self.check_exclusive(list)?;
                return self.alternatives(list, depth);
            }
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

    fn enumeration(
        &mut self,
        obj: &Map<String, Value>,
        values: &Value,
    ) -> Result<String, LocalError> {
        let values = values
            .as_array()
            .filter(|v| !v.is_empty())
            .ok_or_else(|| fail("`enum` must be a non-empty list"))?;
        for value in values {
            check_literal(obj, "enum", value)?;
        }
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
        let Some(name) = def_name(path) else {
            return Err(fail(format!(
                "`$ref` `{path}` is not supported; use #/$defs/<name>"
            )));
        };
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
        if !obj.contains_key("minimum") && !obj.contains_key("maximum") {
            return Ok(if integer { "integer" } else { "number" }.to_string());
        }
        if !integer {
            return Err(fail(
                "`minimum` and `maximum` work only on type integer: \
                 the grammar cannot bound a fraction exactly",
            ));
        }
        let cap = MAX_MAGNITUDE as i64;
        let low = integer_bound(obj, "minimum", true)?.unwrap_or(-cap);
        let high = integer_bound(obj, "maximum", false)?.unwrap_or(cap);
        if low > high {
            return Err(fail("`minimum` is above `maximum`: no integer fits"));
        }
        assert!(
            low >= -cap && high <= cap,
            "bounds stay within {MAX_DIGITS} digits"
        );
        let mut alts = Vec::new();
        if high >= 0 {
            alts.extend(non_negative(low.max(0) as u64, high as u64));
        }
        if low < 0 {
            let from = if high < 0 { high.unsigned_abs() } else { 1 };
            alts.push(format!(
                "\"-\" {}",
                group(non_negative(from, low.unsigned_abs()))
            ));
        }
        assert!(!alts.is_empty(), "low <= high leaves a value");
        self.rule(alts.join(" | "))
    }

    fn check_exclusive(&self, list: &Value) -> Result<(), LocalError> {
        let Some(branches) = list.as_array() else {
            return Ok(());
        };
        let shapes: Vec<Shape> = branches.iter().map(|b| self.shape(b, 0)).collect();
        assert_eq!(shapes.len(), branches.len());
        for (i, first) in shapes.iter().enumerate() {
            for (j, second) in shapes.iter().enumerate().skip(i + 1) {
                if !first.disjoint(second) {
                    return Err(fail(format!(
                        "`oneOf` branches {i} and {j} can match the same value, which the \
                         grammar cannot exclude; use `anyOf` if either may match"
                    )));
                }
            }
        }
        Ok(())
    }

    fn shape(&self, node: &Value, hops: usize) -> Shape {
        let Some(obj) = node.as_object() else {
            return Shape::any();
        };
        if let Some(path) = obj.get("$ref").and_then(Value::as_str) {
            let def = def_name(path)
                .filter(|_| hops < MAX_REF_DEPTH)
                .and_then(|name| self.defs.as_ref()?.get(&name).cloned());
            return def.map_or_else(Shape::any, |d| self.shape(&d, hops + 1));
        }
        let literals = match (obj.get("const"), obj.get("enum").and_then(Value::as_array)) {
            (Some(value), _) => Some(vec![value.clone()]),
            (None, Some(values)) => Some(values.clone()),
            (None, None) => None,
        };
        if let Some(values) = literals {
            return Shape {
                kinds: values.iter().map(kind_of).collect(),
                literals: Some(values),
                tags: BTreeMap::new(),
            };
        }
        let names: Vec<&str> = match obj.get("type") {
            Some(Value::String(name)) => vec![name.as_str()],
            Some(Value::Array(names)) => names.iter().filter_map(Value::as_str).collect(),
            _ => return Shape::any(),
        };
        let kinds: BTreeSet<&'static str> = names
            .iter()
            .map(|n| {
                if *n == "integer" {
                    "number"
                } else {
                    kind_of_name(n)
                }
            })
            .collect();
        let tags = if kinds.contains("object") {
            tags_of(obj)
        } else {
            BTreeMap::new()
        };
        Shape {
            kinds,
            literals: None,
            tags,
        }
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
