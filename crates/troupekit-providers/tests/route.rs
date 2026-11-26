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
    let reserved = refused("name = \"x\"\n[[target]]\nid = \"a\"\ntiers = [\"private\"]\n");
    assert_names(&reserved, "tiers");
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

use std::sync::Arc;
use troupekit_core::clock::FixedClock;
use troupekit_providers::route::{
    decide, BoundTarget, DeviceState, NetworkProbe, NetworkState, PassiveProbe, RoutePlan,
    RouteTags, Signals,
};

const OFFICE: &str = r#"
name = "office"
default = "cloud"
order = ["local", "cloud", "phone"]
[[target]]
id = "local"
tiers = ["small"]
context_tokens = 4096
[[target]]
id = "cloud"
tiers = ["large"]
[[target]]
id = "phone"
tiers = ["small"]
context_tokens = 100000
[budget]
cloud_usd_per_day = 1.0
[device]
min_battery_percent = 20
max_thermal = "fair"
"#;

fn office() -> RoutePolicy {
    RoutePolicy::parse("office.toml", OFFICE).unwrap()
}

fn bound() -> Vec<BoundTarget> {
    vec![
        BoundTarget::on_device("local"),
        BoundTarget::remote("cloud"),
        BoundTarget::on_device("phone"),
    ]
}

fn ids(plan: &RoutePlan) -> Vec<&str> {
    plan.targets().iter().map(|t| t.id()).collect()
}

#[test]
fn offline_plans_only_local() {
    let signals = Signals::default().with_online(Some(false));
    let plan = decide(&office(), &bound(), &RouteTags::default(), &signals);
    assert_eq!(ids(&plan), ["local", "phone"]);
    assert!(plan.targets().iter().all(|t| t.is_on_device()));
    let online = decide(
        &office(),
        &bound(),
        &RouteTags::default(),
        &Signals::default().with_online(Some(true)),
    );
    assert_eq!(ids(&online), ["cloud", "local", "phone"]);
    let unknown = decide(
        &office(),
        &bound(),
        &RouteTags::default(),
        &Signals::default(),
    );
    assert_eq!(ids(&unknown), ["cloud", "local", "phone"]);
}

#[test]
fn private_plans_only_on_device_even_when_online() {
    let tags = RouteTags::default().with_private(true);
    let signals = Signals::default().with_online(Some(true));
    let plan = decide(&office(), &bound(), &tags, &signals);
    assert_eq!(ids(&plan), ["local", "phone"]);
    plan.assert_lawful(&tags);
}

#[test]
fn tier_selects_listed_targets() {
    let small = decide(
        &office(),
        &bound(),
        &RouteTags::default().with_tier("small"),
        &Signals::default(),
    );
    assert_eq!(ids(&small), ["local", "phone"]);
    let large = decide(
        &office(),
        &bound(),
        &RouteTags::default().with_tier("large"),
        &Signals::default(),
    );
    assert_eq!(ids(&large), ["cloud"]);
    let none = decide(
        &office(),
        &bound(),
        &RouteTags::default().with_tier("huge"),
        &Signals::default(),
    );
    assert!(none.is_empty());
    assert!(none.reason().contains("tier"));
}

#[test]
fn oversized_prompt_skips_local() {
    let signals = Signals::default()
        .with_prompt_tokens(4000)
        .with_max_tokens(200);
    let plan = decide(&office(), &bound(), &RouteTags::default(), &signals);
    assert_eq!(ids(&plan), ["cloud", "phone"]);
    let fits = Signals::default()
        .with_prompt_tokens(3896)
        .with_max_tokens(200);
    let plan = decide(&office(), &bound(), &RouteTags::default(), &fits);
    assert_eq!(ids(&plan), ["cloud", "local", "phone"]);
}

#[test]
fn low_battery_skips_local() {
    let low = Signals::default().with_battery_percent(Some(19));
    let plan = decide(&office(), &bound(), &RouteTags::default(), &low);
    assert_eq!(ids(&plan), ["cloud"]);
    let ok = Signals::default().with_battery_percent(Some(20));
    assert_eq!(
        ids(&decide(&office(), &bound(), &RouteTags::default(), &ok)),
        ["cloud", "local", "phone"]
    );
    let hot = Signals::default().with_thermal(Some(Thermal::Serious));
    assert_eq!(
        ids(&decide(&office(), &bound(), &RouteTags::default(), &hot)),
        ["cloud"]
    );
    let fair = Signals::default().with_thermal(Some(Thermal::Fair));
    assert_eq!(
        ids(&decide(&office(), &bound(), &RouteTags::default(), &fair)),
        ["cloud", "local", "phone"]
    );

    let device = DeviceState::new();
    assert_eq!(device.battery_percent(), None);
    assert_eq!(device.thermal(), None);
    device.set_battery_percent(Some(55));
    device.set_thermal(Some(Thermal::Critical));
    let shared = device.clone();
    assert_eq!(shared.battery_percent(), Some(55));
    assert_eq!(shared.thermal(), Some(Thermal::Critical));
    device.set_battery_percent(None);
    assert_eq!(shared.battery_percent(), None);
}

#[test]
fn exhausted_budget_skips_cloud() {
    let spent = Signals::default().with_cloud_spent_today_usd_micros(1_000_000);
    let plan = decide(&office(), &bound(), &RouteTags::default(), &spent);
    assert_eq!(ids(&plan), ["local", "phone"]);
    let under = Signals::default().with_cloud_spent_today_usd_micros(999_999);
    assert_eq!(
        ids(&decide(&office(), &bound(), &RouteTags::default(), &under)),
        ["cloud", "local", "phone"]
    );
}

#[test]
fn empty_plan_when_private_and_no_local_target() {
    let remote_only = vec![
        BoundTarget::remote("local"),
        BoundTarget::remote("cloud"),
        BoundTarget::remote("phone"),
    ];
    let tags = RouteTags::default().with_private(true);
    let plan = decide(&office(), &remote_only, &tags, &Signals::default());
    assert!(plan.is_empty());
    assert!(plan.reason().contains("private"));
    plan.assert_lawful(&tags);

    let unbound = decide(
        &office(),
        &[BoundTarget::remote("cloud")],
        &RouteTags::default().with_tier("small"),
        &Signals::default(),
    );
    assert!(unbound.is_empty());
}

#[test]
fn passive_probe_marks_offline_then_retries() {
    let policy = RoutePolicy::parse(
        "p.toml",
        "name = \"p\"\n[[target]]\nid = \"a\"\n[network]\noffline_after_failures = 2\nretry_online_after_ms = 1000\n",
    )
    .unwrap();
    let clock = Arc::new(FixedClock::new(5_000));
    let probe = PassiveProbe::new(policy.network()).with_clock(clock.clone());
    assert_eq!(probe.online(), None);
    probe.record_remote(true);
    assert_eq!(probe.online(), None);
    probe.record_remote(true);
    assert_eq!(probe.online(), Some(false));
    clock.advance(999);
    assert_eq!(probe.online(), Some(false));
    clock.advance(1);
    assert_eq!(probe.online(), None);
    probe.record_remote(true);
    assert_eq!(probe.online(), Some(false));
    clock.advance(1000);
    assert_eq!(probe.online(), None);
    probe.record_remote(false);
    assert_eq!(probe.online(), None);
    probe.record_remote(true);
    assert_eq!(probe.online(), None);

    let state = NetworkState::new();
    assert_eq!(state.online(), None);
    state.set(Some(false));
    let shared = state.clone();
    assert_eq!(shared.online(), Some(false));
    state.set(Some(true));
    assert_eq!(shared.online(), Some(true));
}

#[test]
#[should_panic(expected = "private plan names a remote target")]
fn private_assert_fires_on_forged_plan() {
    let forged = RoutePlan::forged(vec![
        BoundTarget::on_device("local"),
        BoundTarget::remote("cloud"),
    ]);
    forged.assert_lawful(&RouteTags::default().with_private(true));
}

use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::message::{Message, Role};
use troupekit_providers::catalog::Price;
use troupekit_providers::model::Model;
use troupekit_providers::request::{CompletionRequest, CompletionResponse, Delta, StopReason};
use troupekit_providers::route::RoutedModel;

enum Step {
    Reply(u64),
    Fail(ProviderErrorKind),
    DeltaThenFail,
}

struct Scripted {
    id: String,
    script: Mutex<VecDeque<Step>>,
    calls: Arc<AtomicUsize>,
}

fn scripted(id: &str, steps: Vec<Step>) -> (Arc<dyn Model>, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let model = Scripted {
        id: id.into(),
        script: Mutex::new(steps.into()),
        calls: calls.clone(),
    };
    (Arc::new(model), calls)
}

fn answer(text: &str, cost_usd_micros: u64) -> CompletionResponse {
    CompletionResponse {
        message: Message::text(Role::Assistant, text),
        usage: Usage {
            input_tokens: 10,
            output_tokens: 10,
            cost_usd_micros,
        },
        stop: StopReason::EndTurn,
    }
}

impl Scripted {
    fn next(&self) -> Step {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.script
            .lock()
            .unwrap()
            .pop_front()
            .expect("script ran out")
    }
}

#[async_trait]
impl Model for Scripted {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, _: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        match self.next() {
            Step::Reply(cost) => Ok(answer(&self.id, cost)),
            Step::Fail(kind) => Err(ProviderError::new(kind, "scripted failure")),
            Step::DeltaThenFail => Err(ProviderError::new(ProviderErrorKind::Server, "x")),
        }
    }

    async fn stream(
        &self,
        _: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let text = Delta::TextDelta {
            text: self.id.clone(),
        };
        match self.next() {
            Step::Reply(cost) => {
                on_delta(&text);
                Ok(answer(&self.id, cost))
            }
            Step::Fail(kind) => Err(ProviderError::new(kind, "scripted failure")),
            Step::DeltaThenFail => {
                on_delta(&text);
                Err(ProviderError::new(ProviderErrorKind::Server, "mid-stream"))
            }
        }
    }
}

const PAIR: &str = r#"
name = "office"
default = "cloud"
order = ["local", "cloud"]
[[target]]
id = "local"
tiers = ["small"]
[[target]]
id = "cloud"
tiers = ["large"]
[budget]
cloud_usd_per_day = 1.0
"#;

fn pair_policy(extra: &str) -> RoutePolicy {
    RoutePolicy::parse("office.toml", &format!("{PAIR}{extra}")).unwrap()
}

fn price(input: u64, output: u64) -> Price {
    Price {
        input_usd_micros_per_mtok: input,
        output_usd_micros_per_mtok: output,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    }
}

fn routed(policy: RoutePolicy, local: Arc<dyn Model>, cloud: Arc<dyn Model>) -> RoutedModel {
    RoutedModel::builder(policy)
        .on_device("local", local, Price::FREE)
        .remote("cloud", cloud, price(1, 2))
        .clock(Arc::new(FixedClock::new(1_800_000_000_000)))
        .build()
        .unwrap()
}

fn text(reply: &CompletionResponse) -> String {
    reply.message.text_content()
}

fn req() -> CompletionRequest {
    CompletionRequest::user("hi", 16)
}

#[tokio::test]
async fn routes_by_plan_order() {
    let (local, local_calls) = scripted("local", vec![Step::Reply(0)]);
    let (cloud, cloud_calls) = scripted("cloud", vec![Step::Reply(0)]);
    let model = routed(pair_policy(""), local, cloud);
    assert_eq!(model.id(), "route/office");
    assert_eq!(text(&model.complete(&req()).await.unwrap()), "cloud");
    assert_eq!(cloud_calls.load(Ordering::SeqCst), 1);
    assert_eq!(local_calls.load(Ordering::SeqCst), 0);
    let small = model.view(RouteTags::default().with_tier("small"));
    assert_eq!(text(&small.complete(&req()).await.unwrap()), "local");
    assert_eq!(local_calls.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn falls_back_on_error_when_allowed() {
    let (local, local_calls) = scripted("local", vec![Step::Reply(0)]);
    let (cloud, _) = scripted("cloud", vec![Step::Fail(ProviderErrorKind::Server)]);
    let model = routed(pair_policy(""), local, cloud);
    assert_eq!(text(&model.complete(&req()).await.unwrap()), "local");
    assert_eq!(local_calls.load(Ordering::SeqCst), 1);

    let (local, local_calls) = scripted("local", vec![]);
    let (cloud, _) = scripted("cloud", vec![Step::Fail(ProviderErrorKind::Server)]);
    let strict = routed(pair_policy("[fallback]\non_error = false\n"), local, cloud);
    let err = strict.complete(&req()).await.unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::Server);
    assert_eq!(local_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn private_never_falls_back_to_remote() {
    let (local, local_calls) = scripted(
        "local",
        vec![
            Step::Fail(ProviderErrorKind::Server),
            Step::Fail(ProviderErrorKind::Network),
        ],
    );
    let (cloud, cloud_calls) = scripted("cloud", vec![]);
    let model = routed(pair_policy(""), local, cloud);
    let private = model.view(RouteTags::default().with_private(true));
    assert_eq!(private.id(), "route/office:private");
    assert!(private.complete(&req()).await.is_err());
    let relaxed = private.view(RouteTags::default());
    let err = relaxed
        .stream(&req(), &mut |_: &Delta| {})
        .await
        .unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::Network);
    assert_eq!(local_calls.load(Ordering::SeqCst), 2);
    assert_eq!(cloud_calls.load(Ordering::SeqCst), 0);

    let (cloud, cloud_calls) = scripted("cloud", vec![]);
    let remote_only = RoutedModel::builder(
        RoutePolicy::parse("r.toml", "name = \"r\"\n[[target]]\nid = \"cloud\"\n").unwrap(),
    )
    .remote("cloud", cloud, Price::FREE)
    .build()
    .unwrap()
    .view(RouteTags::default().with_private(true));
    let err = remote_only.complete(&req()).await.unwrap_err();
    assert_eq!(err.kind, ProviderErrorKind::Network);
    assert!(err.message.starts_with("no route: "));
    assert!(err.message.contains("private"));
    assert_eq!(cloud_calls.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn no_switch_after_first_delta() {
    let (local, local_calls) = scripted("local", vec![]);
    let (cloud, _) = scripted("cloud", vec![Step::DeltaThenFail]);
    let model = routed(pair_policy(""), local, cloud);
    let mut seen = Vec::new();
    let err = model
        .stream(&req(), &mut |d: &Delta| seen.push(d.clone()))
        .await
        .unwrap_err();
    assert_eq!(err.message, "mid-stream");
    assert_eq!(seen.len(), 1);
    assert_eq!(local_calls.load(Ordering::SeqCst), 0);

    let (local, _) = scripted("local", vec![Step::Reply(0)]);
    let (cloud, _) = scripted("cloud", vec![Step::Fail(ProviderErrorKind::Server)]);
    let model = routed(pair_policy(""), local, cloud);
    let mut seen = Vec::new();
    let reply = model
        .stream(&req(), &mut |d: &Delta| seen.push(d.clone()))
        .await
        .unwrap();
    assert_eq!(text(&reply), "local");
    assert_eq!(
        seen,
        [Delta::TextDelta {
            text: "local".into()
        }]
    );
}

#[tokio::test]
async fn budget_counts_remote_spend() {
    let (local, local_calls) = scripted("local", vec![Step::Reply(0)]);
    let (cloud, cloud_calls) = scripted("cloud", vec![Step::Reply(600_000), Step::Reply(400_000)]);
    let model = routed(pair_policy(""), local, cloud);
    assert_eq!(text(&model.complete(&req()).await.unwrap()), "cloud");
    assert_eq!(text(&model.complete(&req()).await.unwrap()), "cloud");
    assert_eq!(text(&model.complete(&req()).await.unwrap()), "local");
    assert_eq!(cloud_calls.load(Ordering::SeqCst), 2);
    assert_eq!(local_calls.load(Ordering::SeqCst), 1);
}

#[test]
fn views_cover_private_and_tiers() {
    let (local, _) = scripted("local", vec![]);
    let (cloud, _) = scripted("cloud", vec![]);
    let model = routed(pair_policy(""), local, cloud);
    let views = model.views("route/office");
    let names: Vec<&str> = views.iter().map(|(n, _, _)| n.as_str()).collect();
    assert_eq!(
        names,
        [
            "route/office",
            "route/office:private",
            "route/office:small",
            "route/office:large"
        ]
    );
    for (name, view, _) in &views {
        assert_eq!(view.id(), name.as_str());
    }
    assert_eq!(views[0].2, price(1, 2));
    assert_eq!(views[1].2, Price::FREE);
    assert_eq!(views[2].2, Price::FREE);
    assert_eq!(views[3].2, price(1, 2));
}

#[test]
fn unbound_target_is_a_build_error() {
    let (local, _) = scripted("local", vec![]);
    let (cloud, _) = scripted("cloud", vec![]);
    let err = RoutedModel::builder(pair_policy(""))
        .on_device("local", local.clone(), Price::FREE)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, RouteError::Bind { ref target, .. } if target == "cloud"));
    let err = RoutedModel::builder(pair_policy(""))
        .on_device("local", local.clone(), Price::FREE)
        .remote("cloud", cloud.clone(), Price::FREE)
        .remote("ghost", cloud.clone(), Price::FREE)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, RouteError::Bind { ref target, .. } if target == "ghost"));
    let err = RoutedModel::builder(pair_policy(""))
        .on_device("local", local.clone(), Price::FREE)
        .remote("local", cloud.clone(), Price::FREE)
        .remote("cloud", cloud, Price::FREE)
        .build()
        .err()
        .unwrap();
    assert!(matches!(err, RouteError::Bind { ref target, .. } if target == "local"));
}

use crate::common::{golden_text, scripted_model};
use troupekit_providers::transport::{Scripted as Http, ScriptedTransport};
use troupekit_providers::wire::Protocol;

fn http_pair(
    local_replies: usize,
    cloud: Vec<Http>,
) -> (
    RoutedModel,
    Arc<ScriptedTransport>,
    Arc<ScriptedTransport>,
    Arc<FixedClock>,
    NetworkState,
) {
    let ok = || Http::body(200, golden_text("openai_reply.json"));
    let (local, local_wire) =
        scripted_model(Protocol::OpenAi, (0..local_replies).map(|_| ok()).collect());
    let (cloud, cloud_wire) = scripted_model(Protocol::OpenAi, cloud);
    let clock = Arc::new(FixedClock::new(1_800_000_000_000));
    let state = NetworkState::new();
    let model = RoutedModel::builder(pair_policy(""))
        .on_device("local", Arc::new(local), Price::FREE)
        .remote("cloud", Arc::new(cloud), price(1, 2))
        .network(Arc::new(state.clone()))
        .clock(clock.clone())
        .build()
        .unwrap();
    (model, local_wire, cloud_wire, clock, state)
}

#[tokio::test]
async fn private_never_routes_remote() {
    let ok = Http::body(200, golden_text("openai_reply.json"));
    let (model, local_wire, cloud_wire, _, state) = http_pair(1, vec![ok]);
    state.set(Some(true));
    let private = model.view(RouteTags::default().with_private(true));
    private.complete(&req()).await.unwrap();
    assert_eq!(local_wire.sent().len(), 1);
    assert!(cloud_wire.sent().is_empty());
}

#[tokio::test]
async fn offline_state_uses_local_transport_only() {
    let ok = Http::body(200, golden_text("openai_reply.json"));
    let (model, local_wire, cloud_wire, _, state) = http_pair(1, vec![ok]);
    state.set(Some(false));
    model.complete(&req()).await.unwrap();
    assert_eq!(local_wire.sent().len(), 1);
    assert!(cloud_wire.sent().is_empty());
}

#[tokio::test]
async fn remote_failure_falls_back_to_local() {
    let (model, local_wire, cloud_wire, _, _) =
        http_pair(1, vec![Http::body(503, "upstream down")]);
    model.complete(&req()).await.unwrap();
    assert_eq!(cloud_wire.sent().len(), 1);
    assert_eq!(local_wire.sent().len(), 1);
}

#[tokio::test]
async fn network_errors_flip_passive_probe_offline() {
    let (model, local_wire, cloud_wire, clock, _) = http_pair(3, vec![]);
    model.complete(&req()).await.unwrap();
    assert_eq!(cloud_wire.sent().len(), 1);
    assert_eq!(local_wire.sent().len(), 1);
    model.complete(&req()).await.unwrap();
    assert_eq!(cloud_wire.sent().len(), 1);
    assert_eq!(local_wire.sent().len(), 2);
    clock.advance(30_000);
    model.complete(&req()).await.unwrap();
    assert_eq!(cloud_wire.sent().len(), 2);
    assert_eq!(local_wire.sent().len(), 3);
}

fn traced_records(sink: &troupekit_trace::MemorySink) -> Vec<serde_json::Value> {
    sink.lines()
        .iter()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

fn tracer_into(sink: &Arc<troupekit_trace::MemorySink>) -> troupekit_trace::Tracer {
    troupekit_trace::Tracer::builder()
        .sink(sink.clone())
        .build()
        .unwrap()
}

#[tokio::test]
async fn route_decide_emits_routing_span() {
    let (local, _) = scripted("local", vec![Step::Reply(0)]);
    let (cloud, _) = scripted("cloud", vec![Step::Fail(ProviderErrorKind::Server)]);
    let model = routed(pair_policy(""), local, cloud);
    let sink = Arc::new(troupekit_trace::MemorySink::default());
    let tracer = tracer_into(&sink);
    tracer.in_scope(model.complete(&req())).await.unwrap();
    drop(tracer);
    let records = traced_records(&sink);
    let start = records
        .iter()
        .find(|r| r["t"] == "start" && r["name"] == "route.decide")
        .expect("a route.decide span");
    assert_eq!(start["cat"], "routing");
    assert_eq!(start["fields"]["private"], false);
    assert_eq!(start["fields"]["plan"], "cloud,local");
    let end = records
        .iter()
        .find(|r| r["t"] == "end" && r["span"] == start["span"])
        .expect("the span ends");
    assert_eq!(end["ok"], true);
    assert_eq!(end["fields"]["chosen"], "local");
    let fallback = records
        .iter()
        .find(|r| r["t"] == "event" && r["name"] == "route.fallback")
        .expect("a route.fallback event");
    assert_eq!(fallback["cat"], "routing");
    assert_eq!(fallback["fields"]["from"], "cloud");
    assert_eq!(fallback["fields"]["kind"], "Server");
}

#[tokio::test]
async fn private_law_event_is_emitted() {
    let (local, _) = scripted("local", vec![Step::Reply(0)]);
    let (cloud, cloud_calls) = scripted("cloud", vec![]);
    let model = routed(pair_policy(""), local, cloud).view(RouteTags::default().with_private(true));
    let sink = Arc::new(troupekit_trace::MemorySink::default());
    let tracer = tracer_into(&sink);
    tracer.in_scope(model.complete(&req())).await.unwrap();
    drop(tracer);
    let records = traced_records(&sink);
    let law = records
        .iter()
        .find(|r| r["t"] == "event" && r["name"] == "law.check")
        .expect("a law.check event");
    assert_eq!(law["cat"], "laws");
    assert_eq!(
        law["fields"]["law"],
        "private_plan_has_only_on_device_targets"
    );
    assert_eq!(law["fields"]["held"], true);
    assert_eq!(law["fields"]["private"], true);
    assert_eq!(cloud_calls.load(Ordering::SeqCst), 0);
}
