use troupekit_local::{DeviceClass, LocalCatalog, LocalError, LocalSpec};

const HASH: &str = "741ad12b64088fedc17c33aacb22e48be1972ef36a39f03666dd68bd15614fb9";

fn spec(id: &str, ram: u64, class: DeviceClass) -> LocalSpec {
    LocalSpec::new(id, "https://example.org/m.gguf", HASH, 1_000)
        .expect("valid row")
        .with_ram_bytes(ram)
        .with_class(class)
}

#[test]
fn catalog_rejects_http_url_and_bad_hash() {
    let err = LocalSpec::new("m", "http://example.org/m.gguf", HASH, 1).expect_err("http");
    assert!(matches!(err, LocalError::Catalog(_)), "{err:?}");
    for bad in [
        "abc",
        &HASH.to_uppercase(),
        &format!("{}z", &HASH[..63]),
        &format!("{HASH}0"),
    ] {
        let err = LocalSpec::new("m", "https://example.org/m.gguf", bad, 1).expect_err("bad hash");
        assert!(matches!(err, LocalError::Catalog(_)), "{bad}: {err:?}");
    }
    for bad_id in ["", "../m", "a/b", "m m", "M"] {
        assert!(
            LocalSpec::new(bad_id, "https://example.org/m.gguf", HASH, 1).is_err(),
            "{bad_id:?}"
        );
    }
    assert!(LocalSpec::new("m", "https://example.org/m.gguf", HASH, 0).is_err());
}

#[test]
fn catalog_rejects_duplicate_ids_and_two_recommended_per_class() {
    let a = spec("a", 1, DeviceClass::Phone);
    let err = LocalCatalog::new(vec![a.clone(), a.clone()]).expect_err("duplicate");
    assert!(matches!(err, LocalError::Catalog(_)), "{err:?}");
    let r1 = spec("r1", 1, DeviceClass::Phone).with_recommended(true);
    let r2 = spec("r2", 2, DeviceClass::Phone).with_recommended(true);
    assert!(LocalCatalog::new(vec![r1.clone(), r2]).is_err());
    let r3 = spec("r3", 2, DeviceClass::Desktop).with_recommended(true);
    assert!(LocalCatalog::new(vec![r1, r3]).is_ok());
}

#[test]
fn pick_returns_largest_fitting_row() {
    let catalog = LocalCatalog::new(vec![
        spec("small", 100, DeviceClass::Phone),
        spec("mid", 500, DeviceClass::Phone),
        spec("big", 900, DeviceClass::Phone),
        spec("desk", 50, DeviceClass::Desktop),
    ])
    .expect("catalog");
    assert_eq!(
        catalog.pick(DeviceClass::Phone, 600).map(|s| s.id()),
        Some("mid")
    );
    assert_eq!(catalog.pick(DeviceClass::Phone, 99), None);
    assert_eq!(catalog.pick(DeviceClass::Pi, 10_000), None);
    assert_eq!(
        catalog.get("desk").map(|s| s.class()),
        Some(DeviceClass::Desktop)
    );
    assert!(catalog.get("nope").is_none());
}

#[test]
fn pick_prefers_the_recommended_row_when_it_fits() {
    let catalog = LocalCatalog::new(vec![
        spec("small", 100, DeviceClass::Phone).with_recommended(true),
        spec("big", 900, DeviceClass::Phone),
    ])
    .expect("catalog");
    assert_eq!(
        catalog.pick(DeviceClass::Phone, 1_000).map(|s| s.id()),
        Some("small")
    );
    assert_eq!(catalog.pick(DeviceClass::Phone, 50), None);
}

#[test]
fn defaults_are_the_safe_values() {
    let s = LocalSpec::new("m", "https://example.org/m.gguf", HASH, 7).expect("row");
    assert_eq!(s.bytes(), 7);
    assert_eq!(s.class(), DeviceClass::Desktop);
    assert!(!s.recommended());
    assert!(!s.embeddings());
    assert_eq!(s.licence(), "");
}

const ROW: &str = r#"
[[model]]
id = "m"
url = "https://example.org/m.gguf"
sha256 = "741ad12b64088fedc17c33aacb22e48be1972ef36a39f03666dd68bd15614fb9"
bytes = 10
ram_bytes = 20
context = 2048
class = "pi"
licence = "mit"
embeddings = true
recommended = true
"#;

#[test]
fn load_reads_every_field() {
    let catalog = LocalCatalog::load(ROW).expect("loads");
    let m = catalog.get("m").expect("row");
    assert_eq!(m.bytes(), 10);
    assert_eq!(m.ram_bytes(), 20);
    assert_eq!(m.context(), 2_048);
    assert_eq!(m.class(), DeviceClass::Pi);
    assert_eq!(m.licence(), "mit");
    assert!(m.embeddings() && m.recommended());
}

#[test]
fn load_applies_the_row_checks() {
    let http = ROW.replace("https://", "http://");
    assert!(matches!(
        LocalCatalog::load(&http),
        Err(LocalError::Catalog(_))
    ));
    let bad_hash = ROW.replace("741ad12b", "741AD12B");
    assert!(matches!(
        LocalCatalog::load(&bad_hash),
        Err(LocalError::Catalog(_))
    ));
    let unknown = ROW.replace("licence = \"mit\"", "licence = \"mit\"\nmirror = \"x\"");
    assert!(matches!(
        LocalCatalog::load(&unknown),
        Err(LocalError::Catalog(_))
    ));
    let missing = ROW.replace("bytes = 10\n", "");
    assert!(matches!(
        LocalCatalog::load(&missing),
        Err(LocalError::Catalog(_))
    ));
    let big = format!("{ROW}{}", " ".repeat(LocalCatalog::MAX_BYTES));
    assert!(matches!(
        LocalCatalog::load(&big),
        Err(LocalError::Catalog(_))
    ));
}

#[test]
fn builtin_catalog_holds_the_test_model() {
    let catalog = LocalCatalog::builtin();
    let m = catalog
        .get("smollm2-135m-instruct-q2k")
        .expect("test model row");
    assert_eq!(m.bytes(), 88_202_080);
    assert_eq!(
        m.sha256(),
        "741ad12b64088fedc17c33aacb22e48be1972ef36a39f03666dd68bd15614fb9"
    );
    assert_eq!(m.licence(), "apache-2.0");
}

#[test]
fn builtin_catalog_ships_inside_the_crate() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("catalog/local.toml");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{} is not in the crate: {e}", path.display()));
    let file = LocalCatalog::load(&text).expect("the shipped catalog parses");
    let ids = |c: &LocalCatalog| {
        c.rows()
            .iter()
            .map(|s| s.id().to_string())
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&file), ids(LocalCatalog::builtin()));
    assert!(!file.rows().is_empty());
}
