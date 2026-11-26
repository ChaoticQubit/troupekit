use std::path::Path;

fn sources() -> Vec<String> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut texts = Vec::new();
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if path.extension().is_some_and(|e| e == "rs") && !name.ends_with("_tests.rs") {
            texts.push(std::fs::read_to_string(path).unwrap());
        }
    }
    assert!(texts.len() >= 8);
    texts
}

fn declares(texts: &[String], name: &str) -> bool {
    let forms = [
        format!("pub struct {name} "),
        format!("pub struct {name};"),
        format!("pub enum {name} "),
        format!("pub trait {name}:"),
        format!("pub trait {name} "),
        format!("pub fn {name}("),
    ];
    texts
        .iter()
        .any(|t| forms.iter().any(|f| t.contains(f.as_str())))
}

fn blocks<'a>(text: &'a str, owner: &str) -> Vec<&'a str> {
    let starts = [format!("impl {owner} {{"), format!("pub trait {owner}")];
    let mut found = Vec::new();
    for start in &starts {
        let mut rest = text;
        while let Some(at) = rest.find(start.as_str()) {
            let block = &rest[at..];
            let end = block.find("\n}\n").map_or(block.len(), |e| e + 3);
            found.push(&block[..end]);
            rest = &block[end..];
        }
    }
    found
}

fn has_method(texts: &[String], owner: &str, method: &str) -> bool {
    let forms = [
        format!("pub fn {method}("),
        format!("pub async fn {method}("),
        format!("    fn {method}("),
    ];
    texts
        .iter()
        .flat_map(|t| blocks(t, owner))
        .any(|b| forms.iter().any(|f| b.contains(f.as_str())))
}

#[test]
fn freeze_names_match_adapters() {
    let texts = sources();
    let lib =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs")).unwrap();
    let freeze = include_str!("../contract/freeze.txt");
    let names: Vec<&str> = freeze
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    assert!(names.len() >= 80, "{}", names.len());
    let mut missing = Vec::new();
    for name in &names {
        let found = match name.split_once("::") {
            Some((owner, method)) => declares(&texts, owner) && has_method(&texts, owner, method),
            None => declares(&texts, name) && lib.contains(name),
        };
        if !found {
            missing.push(*name);
        }
    }
    assert!(
        missing.is_empty(),
        "freeze names with no pub item: {missing:?}"
    );
}

#[test]
fn freeze_check_catches_a_missing_name() {
    let texts = sources();
    assert!(!declares(&texts, "NoSuchHandle"));
    assert!(!has_method(&texts, "AgentBuilder", "no_such_setter"));
    assert!(has_method(&texts, "AgentBuilder", "build"));
}
