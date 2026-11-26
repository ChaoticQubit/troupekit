use serde_json::{json, Value};
use std::path::PathBuf;
use troupekit_local::{schema_to_gbnf, Grammar, LocalError};

const BASE_RULES: [&str; 7] = [
    "ws", "null", "boolean", "char", "string", "integer", "number",
];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gbnf")
}

fn check_fixture(name: &str) -> Grammar {
    let dir = fixture_dir();
    let schema: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap())
            .unwrap();
    let expected = std::fs::read_to_string(dir.join(format!("{name}.gbnf"))).unwrap();
    let grammar = schema_to_gbnf(&schema).unwrap();
    assert_eq!(grammar.gbnf(), expected, "fixture {name}");
    assert_eq!(grammar.root(), "root");
    grammar
}

fn refused(schema: Value) -> String {
    match schema_to_gbnf(&schema) {
        Err(LocalError::Grammar(message)) => message,
        Err(other) => panic!("expected a Grammar error, got {other}"),
        Ok(g) => panic!("expected a refusal, got {}", g.gbnf()),
    }
}

fn rule_names(gbnf: &str) -> Vec<String> {
    gbnf.lines()
        .map(|line| {
            line.split_once(" ::= ")
                .expect("one rule per line")
                .0
                .to_owned()
        })
        .collect()
}

#[test]
fn simple_object_grammar_matches_fixture() {
    check_fixture("simple_object");
}

#[test]
fn every_fixture_matches() {
    let mut names: Vec<String> = std::fs::read_dir(fixture_dir())
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".json").map(str::to_owned)
        })
        .collect();
    names.sort();
    assert_eq!(names.len(), 8, "{names:?}");
    for name in names {
        check_fixture(&name);
    }
}

#[test]
fn optional_properties_follow_required() {
    let grammar = check_fixture("optional");
    let first = grammar.gbnf().lines().nth(1).unwrap();
    assert!(
        first.starts_with("r0 ::= \"{\" ws \"\\\"c\\\"\""),
        "{first}"
    );
    assert!(
        !first.contains("\\\"a\\\""),
        "optional a is outside the required run"
    );
    let none = schema_to_gbnf(&json!({"type": "object"})).unwrap();
    assert!(none.gbnf().contains("r0 ::= \"{\" ws \"}\"\n"));
}

#[test]
fn enum_and_const_are_literals() {
    let grammar = check_fixture("enum_const");
    assert!(grammar
        .gbnf()
        .contains("r1 ::= \"\\\"low\\\"\" | \"\\\"high\\\"\" | \"3\" | \"null\"\n"));
    let one = schema_to_gbnf(&json!({"const": {"b": [1, true]}})).unwrap();
    assert!(
        one.gbnf()
            .starts_with("root ::= \"{\\\"b\\\":[1,true]}\"\n"),
        "{}",
        one.gbnf()
    );
    assert!(refused(json!({"enum": []})).contains("enum"));
}

#[test]
fn gbnf_escapes_hostile_property_names() {
    let hostile = "a\"b\\c\nd ::= x";
    let schema = json!({
        "type": "object",
        "properties": {hostile: {"type": "string"}, "root": {"type": "boolean"}, "ws": {"type": "null"}},
        "required": [hostile, "root", "ws"],
    });
    let grammar = schema_to_gbnf(&schema).unwrap();
    let text = grammar.gbnf();
    assert!(text.contains(r#""\"a\\\"b\\\\c\\nd ::= x\"""#), "{text}");
    assert!(text.contains(r#""\"root\"" ws ":" ws boolean"#), "{text}");
    let names = rule_names(text);
    assert_eq!(names.len(), 2 + BASE_RULES.len(), "{names:?}");
    assert_eq!(names[0], "root");
    assert_eq!(names[1], "r0");
    assert_eq!(&names[2..], BASE_RULES);
    let control = schema_to_gbnf(&json!({"const": "tab\there\u{7f}"})).unwrap();
    assert!(
        control
            .gbnf()
            .starts_with("root ::= \"\\\"tab\\\\there\\x7F\\\"\"\n"),
        "{}",
        control.gbnf()
    );
}

#[test]
fn unsupported_keyword_is_named() {
    assert!(refused(json!({"type": "string", "pattern": "^a"})).contains("`pattern`"));
    let nested = json!({"type": "object", "properties": {"x": {"type": "string", "minLength": 1}}});
    assert!(refused(nested).contains("`minLength`"));
    let open = json!({"type": "object", "additionalProperties": true});
    assert!(refused(open).contains("`additionalProperties`"));
    let old = json!({"$ref": "#/definitions/x", "definitions": {"x": {"type": "null"}}});
    assert!(refused(old).contains("definitions"));
    assert!(refused(json!(true)).contains("any value"));
    assert!(refused(json!({"description": "no type"})).contains("`type`"));
    assert!(refused(json!({"type": "date"})).contains("date"));
    let both = json!({"anyOf": [{"type": "null"}], "oneOf": [{"type": "null"}]});
    assert!(refused(both).contains("`oneOf`"));
    let missing = json!({"type": "object", "properties": {}, "required": ["gone"]});
    assert!(refused(missing).contains("gone"));
}

#[test]
fn cyclic_ref_is_refused() {
    let own = json!({
        "$defs": {"a": {"type": "array", "items": {"$ref": "#/$defs/a"}}},
        "$ref": "#/$defs/a",
    });
    assert!(refused(own).contains("cyclic"));
    let mutual = json!({
        "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"type": "array", "items": {"$ref": "#/$defs/a"}}},
        "$ref": "#/$defs/a",
    });
    assert!(refused(mutual).contains("cyclic"));
    let mut defs = serde_json::Map::new();
    for i in 0..17 {
        defs.insert(
            format!("d{i}"),
            json!({"$ref": format!("#/$defs/d{}", i + 1)}),
        );
    }
    defs.insert("d17".into(), json!({"type": "null"}));
    let chain = json!({"$defs": defs, "$ref": "#/$defs/d0"});
    assert!(refused(chain).contains("16"));
    assert!(refused(json!({"$ref": "#/$defs/nope", "$defs": {}})).contains("nope"));
}

#[test]
fn grammar_size_cap_is_enforced() {
    let long: Vec<Value> = (0..300)
        .map(|i| json!(format!("{i}{}", "x".repeat(1000))))
        .collect();
    let message = refused(json!({"enum": long}));
    assert!(message.contains("limit"), "{message}");
    assert!(
        message.contains(&Grammar::MAX_BYTES.to_string()),
        "{message}"
    );
    let fits: Vec<Value> = (0..100)
        .map(|i| json!(format!("{i}{}", "x".repeat(1000))))
        .collect();
    let grammar = schema_to_gbnf(&json!({"enum": fits})).unwrap();
    assert!(grammar.gbnf().len() <= Grammar::MAX_BYTES);
}

#[test]
fn counts_lengths_and_depth_are_bounded() {
    let items = |extra: Value| {
        let mut schema = json!({"type": "array", "items": {"type": "null"}});
        schema
            .as_object_mut()
            .unwrap()
            .extend(extra.as_object().unwrap().clone());
        schema
    };
    assert!(refused(items(json!({"maxItems": 65}))).contains("`maxItems`"));
    assert!(refused(items(json!({"minItems": 65}))).contains("`minItems`"));
    assert!(refused(items(json!({"minItems": 3, "maxItems": 2}))).contains("`minItems`"));
    let none = schema_to_gbnf(&items(json!({"maxItems": 0}))).unwrap();
    assert!(none.gbnf().contains("r0 ::= \"[\" ws \"]\"\n"));
    let one = schema_to_gbnf(&items(json!({"minItems": 1, "maxItems": 1}))).unwrap();
    assert!(
        one.gbnf().contains("r0 ::= \"[\" ws null ws \"]\"\n"),
        "{}",
        one.gbnf()
    );
    let many = schema_to_gbnf(&items(json!({"minItems": 3}))).unwrap();
    assert!(
        many.gbnf().contains("( ws \",\" ws null ){2,}"),
        "{}",
        many.gbnf()
    );
    assert!(refused(json!({"type": "string", "maxLength": 1025})).contains("`maxLength`"));
    let reversed = json!({"type": "integer", "minimum": 5, "maximum": 1});
    assert!(refused(reversed).contains("`minimum`"));
    let negative = schema_to_gbnf(&json!({"type": "integer", "maximum": -1})).unwrap();
    assert!(
        negative
            .gbnf()
            .contains("r0 ::= \"-\" ( [1-9] | [1-9] [0-9]{1,15} )\n"),
        "{}",
        negative.gbnf()
    );
    let mut deep = json!({"type": "null"});
    for _ in 0..100 {
        deep = json!({"type": "array", "items": deep});
    }
    assert!(refused(deep).contains("64"));
}

#[test]
fn enum_and_const_take_sibling_constraints() {
    let long = json!({"type": "string", "enum": ["a", "long"], "maxLength": 1});
    assert!(refused(long).contains("`maxLength`"));
    let mixed = json!({"type": "string", "enum": ["a", 1]});
    assert!(refused(mixed).contains("`type`"));
    let whole = json!({"type": "integer", "const": 1.5});
    assert!(refused(whole).contains("`type`"));
    let typed = schema_to_gbnf(&json!({"type": "string", "enum": ["a", "b"]})).unwrap();
    assert!(
        typed
            .gbnf()
            .starts_with("root ::= r0\nr0 ::= \"\\\"a\\\"\" | \"\\\"b\\\"\"\n"),
        "{}",
        typed.gbnf()
    );
    let reference = json!({"$defs": {"x": {"type": "array", "items": {"type": "null"}}}, "$ref": "#/$defs/x", "maxItems": 1});
    assert!(refused(reference).contains("`maxItems`"));
    let union = json!({"anyOf": [{"type": "null"}], "type": "null"});
    assert!(refused(union).contains("`type`"));
    let noted = json!({"$defs": {"x": {"type": "null"}}, "$ref": "#/$defs/x", "description": "ok"});
    assert!(schema_to_gbnf(&noted).is_ok());
}

#[test]
fn integer_bounds_are_exact() {
    let exact = schema_to_gbnf(&json!({"type": "integer", "minimum": 5, "maximum": 5})).unwrap();
    assert!(exact.gbnf().contains("r0 ::= \"5\"\n"), "{}", exact.gbnf());
    let byte = schema_to_gbnf(&json!({"type": "integer", "minimum": 0, "maximum": 255})).unwrap();
    assert!(
        byte.gbnf().contains(
            "r0 ::= [0-9] | [1-9] [0-9] | ( \"1\" [0-9]{2} | \"2\" ( [0-4] [0-9] | \"5\" [0-5] ) )\n"
        ),
        "{}",
        byte.gbnf()
    );
    let floor = schema_to_gbnf(&json!({"type": "integer", "minimum": 0.0})).unwrap();
    assert!(
        floor.gbnf().contains("r0 ::= [0-9] | [1-9] [0-9]{1,15}\n"),
        "{}",
        floor.gbnf()
    );
    let teens =
        schema_to_gbnf(&json!({"type": "integer", "minimum": 12.5, "maximum": 19})).unwrap();
    assert!(
        teens.gbnf().contains("r0 ::= \"1\" [3-9]\n"),
        "{}",
        teens.gbnf()
    );
    let between = json!({"type": "integer", "minimum": 1.2, "maximum": 1.8});
    assert!(refused(between).contains("`minimum`"));
    let float = json!({"type": "number", "minimum": 0, "maximum": 1});
    assert!(refused(float).contains("integer"));
}

#[test]
fn one_of_needs_disjoint_branches() {
    let numbers = json!({"oneOf": [{"type": "integer"}, {"type": "number"}]});
    assert!(refused(numbers).contains("`anyOf`"));
    let objects = json!({"oneOf": [
        {"type": "object", "properties": {"a": {"type": "null"}}},
        {"type": "object", "properties": {"b": {"type": "null"}}},
    ]});
    assert!(refused(objects).contains("`anyOf`"));
    let same = json!({"oneOf": [{"const": "x"}, {"enum": ["y", "x"]}]});
    assert!(refused(same).contains("`anyOf`"));
    let names = json!({"oneOf": [
        {"type": "string", "const": "red", "description": "Red."},
        {"type": "string", "const": "blue"},
    ]});
    assert!(schema_to_gbnf(&names).is_ok());
    let tagged = json!({
        "$defs": {"cat": {"type": "object", "properties": {"kind": {"const": "cat"}}, "required": ["kind"]}},
        "oneOf": [
            {"$ref": "#/$defs/cat"},
            {"type": "object", "properties": {"kind": {"const": "dog"}, "bark": {"type": "boolean"}}, "required": ["kind"]},
        ],
    });
    assert!(schema_to_gbnf(&tagged).is_ok());
    let union = json!({"anyOf": [{"type": "integer"}, {"type": "number"}]});
    assert!(schema_to_gbnf(&union).is_ok());
}

#[test]
fn integer_bounds_past_sixteen_digits_are_refused() {
    let huge = json!({"type": "integer", "maximum": 10_000_000_000_000_000u64});
    assert!(refused(huge).contains("`maximum`"));
    let edge = schema_to_gbnf(&json!({"type": "integer", "minimum": -9_999_999_999_999_998i64}));
    assert!(edge.is_ok());
}
