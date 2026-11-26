use troupekit_providers::route::{NetworkMode, RouteError, RoutePolicy, Thermal};

const FULL: &str = r#"
name = "office"
default = "cloud"
order = ["local", "cloud"]
[[target]]
id = "local"
tiers = ["small"]
context_tokens = 4096
[[target]]
id = "cloud"
[network]
mode = "auto"
offline_after_failures = 1
retry_online_after_ms = 30000
[budget]
cloud_usd_per_day = 1.0
[device]
min_battery_percent = 20
max_thermal = "fair"
[fallback]
on_error = true
"#;

fn refused(text: &str) -> RouteError {
    RoutePolicy::parse("route.toml", text).unwrap_err()
}

fn assert_names(err: &RouteError, key: &str) {
    let text = err.to_string();
    assert!(text.contains("route.toml"), "no file in {text}");
    assert!(text.contains(key), "no key {key} in {text}");
}

#[test]
fn parses_full_policy() {
    let p = RoutePolicy::parse("route.toml", FULL).unwrap();
    assert_eq!(p.name(), "office");
    assert_eq!(p.default_target(), Some("cloud"));
    assert_eq!(p.order(), ["local", "cloud"]);
    assert_eq!(p.targets().len(), 2);
    let local = p.target("local").unwrap();
    assert_eq!(local.id(), "local");
    assert_eq!(local.tiers(), ["small"]);
    assert_eq!(local.context_tokens(), Some(4096));
    let cloud = p.target("cloud").unwrap();
    assert!(cloud.tiers().is_empty());
    assert_eq!(cloud.context_tokens(), None);
    assert!(p.target("other").is_none());
    assert_eq!(p.network().mode(), NetworkMode::Auto);
    assert_eq!(p.network().offline_after_failures(), 1);
    assert_eq!(p.network().retry_online_after_ms(), 30_000);
    assert_eq!(p.budget().cloud_usd_micros_per_day(), Some(1_000_000));
    assert_eq!(p.device().min_battery_percent(), Some(20));
    assert_eq!(p.device().max_thermal(), Some(Thermal::Fair));
    assert!(p.fallback().on_error());
    assert_eq!(p.tiers(), ["small"]);
}

#[test]
fn defaults_are_the_safe_values() {
    let p = RoutePolicy::parse(
        "route.toml",
        "name = \"min\"\n[[target]]\nid = \"b\"\n[[target]]\nid = \"a\"\n",
    )
    .unwrap();
    assert_eq!(p.default_target(), None);
    assert_eq!(p.order(), ["b", "a"]);
    assert_eq!(p.network().mode(), NetworkMode::Auto);
    assert_eq!(p.network().offline_after_failures(), 1);
    assert_eq!(p.network().retry_online_after_ms(), 30_000);
    assert_eq!(p.budget().cloud_usd_micros_per_day(), None);
    assert_eq!(p.device().min_battery_percent(), None);
    assert_eq!(p.device().max_thermal(), None);
    assert!(p.fallback().on_error());
    assert_eq!(RoutePolicy::MAX_TARGETS, 16);
    assert_eq!(RoutePolicy::MAX_BYTES, 64 * 1024);

    let partial = RoutePolicy::parse(
        "route.toml",
        "name = \"p\"\norder = [\"c\"]\n[[target]]\nid = \"a\"\n[[target]]\nid = \"c\"\n",
    )
    .unwrap();
    assert_eq!(partial.order(), ["c", "a"]);
}

#[test]
fn unknown_key_and_unknown_target_are_refused() {
    let unknown_top = refused("name = \"x\"\ncolour = 1\n[[target]]\nid = \"a\"\n");
    assert_names(&unknown_top, "colour");
    let unknown_nested = refused("name = \"x\"\n[[target]]\nid = \"a\"\n[network]\nspeed = 3\n");
    assert_names(&unknown_nested, "speed");
    let on_device_key = refused("name = \"x\"\n[[target]]\nid = \"a\"\non_device = true\n");
    assert_names(&on_device_key, "on_device");
    let bad_default = refused("name = \"x\"\ndefault = \"ghost\"\n[[target]]\nid = \"a\"\n");
    assert_names(&bad_default, "default");
    assert!(bad_default.to_string().contains("ghost"));
    let bad_order = refused("name = \"x\"\norder = [\"a\", \"ghost\"]\n[[target]]\nid = \"a\"\n");
    assert_names(&bad_order, "order");
    let repeated_order = refused("name = \"x\"\norder = [\"a\", \"a\"]\n[[target]]\nid = \"a\"\n");
    assert_names(&repeated_order, "order");
    let duplicate = refused("name = \"x\"\n[[target]]\nid = \"a\"\n[[target]]\nid = \"a\"\n");
    assert_names(&duplicate, "target");
    let bad_id = refused("name = \"x\"\n[[target]]\nid = \"A b\"\n");
    assert_names(&bad_id, "target.id");
    let bad_tier = refused("name = \"x\"\n[[target]]\nid = \"a\"\ntiers = [\"Big!\"]\n");
    assert_names(&bad_tier, "tiers");
    let bad_name = refused("name = \"\"\n[[target]]\nid = \"a\"\n");
    assert_names(&bad_name, "name");
    let no_target = refused("name = \"x\"\n");
    assert_names(&no_target, "target");
    let bad_thermal =
        refused("name = \"x\"\n[[target]]\nid = \"a\"\n[device]\nmax_thermal = \"hot\"\n");
    assert_names(&bad_thermal, "hot");
    let bad_mode = refused("name = \"x\"\n[[target]]\nid = \"a\"\n[network]\nmode = \"maybe\"\n");
    assert_names(&bad_mode, "maybe");
    let not_toml = refused("name = ");
    assert!(matches!(not_toml, RouteError::Parse { .. }));
    assert!(not_toml.to_string().contains("route.toml"));
}

#[test]
fn bounds_are_enforced() {
    let mut many = String::from("name = \"x\"\n");
    for i in 0..=RoutePolicy::MAX_TARGETS {
        many.push_str(&format!("[[target]]\nid = \"t{i}\"\n"));
    }
    assert_names(&refused(&many), "target");

    let mut huge = String::from("name = \"x\"\n[[target]]\nid = \"a\"\n");
    while huge.len() <= RoutePolicy::MAX_BYTES {
        huge.push_str("                                \n");
    }
    assert!(matches!(refused(&huge), RouteError::Invalid { .. }));
    assert!(refused(&huge).to_string().contains("route.toml"));

    let one = "name = \"x\"\n[[target]]\nid = \"a\"\n";
    let cases = [
        (
            "[[target]]\nid = \"b\"\ncontext_tokens = 0\n",
            "context_tokens",
        ),
        (
            "[network]\noffline_after_failures = 0\n",
            "offline_after_failures",
        ),
        (
            "[network]\noffline_after_failures = 1001\n",
            "offline_after_failures",
        ),
        (
            "[network]\nretry_online_after_ms = 0\n",
            "retry_online_after_ms",
        ),
        (
            "[network]\nretry_online_after_ms = 86400001\n",
            "retry_online_after_ms",
        ),
        ("[budget]\ncloud_usd_per_day = -1.0\n", "cloud_usd_per_day"),
        (
            "[budget]\ncloud_usd_per_day = 1000001.0\n",
            "cloud_usd_per_day",
        ),
        ("[budget]\ncloud_usd_per_day = nan\n", "cloud_usd_per_day"),
        (
            "[device]\nmin_battery_percent = 101\n",
            "min_battery_percent",
        ),
    ];
    for (extra, key) in cases {
        assert_names(&refused(&format!("{one}{extra}")), key);
    }

    let long_id = "a".repeat(RoutePolicy::MAX_ID_BYTES + 1);
    assert_names(
        &refused(&format!("name = \"x\"\n[[target]]\nid = \"{long_id}\"\n")),
        "target.id",
    );
    let ok_zero_budget = RoutePolicy::parse(
        "route.toml",
        &format!("{one}[budget]\ncloud_usd_per_day = 0\n"),
    )
    .unwrap();
    assert_eq!(ok_zero_budget.budget().cloud_usd_micros_per_day(), Some(0));
}
