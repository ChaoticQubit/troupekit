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
