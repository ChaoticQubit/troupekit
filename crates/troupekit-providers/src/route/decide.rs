use super::policy::RoutePolicy;
use super::signals::{RouteTags, Signals};

/// A policy target bound in code: its id and whether its model runs on
/// the device. Only code knows whether a model touches the network, so
/// locality is declared here and never read from TOML.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundTarget {
    id: String,
    on_device: bool,
}

impl BoundTarget {
    /// Declares target `id` as running on the device, with no network.
    /// Private requests may reach it.
    /// Returns the binding.
    pub fn on_device(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            on_device: true,
        }
    }

    /// Declares target `id` as remote: its model may use the network.
    /// Private and offline requests never reach it.
    /// Returns the binding.
    pub fn remote(id: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            on_device: false,
        }
    }

    /// The policy target id this binding names.
    /// Matches a `[[target]] id` exactly.
    /// Plans and errors name it.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Whether the bound model runs on the device.
    /// Set only by the constructor used.
    /// The private rule reads it.
    pub fn is_on_device(&self) -> bool {
        self.on_device
    }
}

/// The ordered targets one request may try, from `decide`. Empty only
/// when no target is eligible; `reason` then names the rules that removed
/// targets. A private plan never names a remote target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePlan {
    targets: Vec<BoundTarget>,
    reason: String,
}

impl RoutePlan {
    /// The targets in try order.
    /// Each id appears at most once.
    /// Empty when nothing is eligible.
    pub fn targets(&self) -> &[BoundTarget] {
        &self.targets
    }

    /// Whether no target is eligible.
    /// A routed model then fails with a `Network` error.
    /// Its message carries `reason`.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// The rules that removed targets, comma separated, in rule order.
    /// Empty when no rule removed any.
    /// Holds names only, never request text.
    pub fn reason(&self) -> &str {
        &self.reason
    }

    /// Panics unless this plan obeys the law for `tags`: a private plan
    /// names only on-device targets. `decide` and the routed model both
    /// call it, so a broken rule crashes instead of sending data out.
    pub fn assert_lawful(&self, tags: &RouteTags) {
        if tags.private {
            for target in &self.targets {
                assert!(
                    target.on_device,
                    "private plan names a remote target: {}",
                    target.id
                );
            }
        }
    }

    /// Builds a plan without running `decide`, for the test that proves
    /// `assert_lawful` fires on a forged plan. Not for production code.
    /// Returns a plan with `targets` and no reason.
    #[doc(hidden)]
    pub fn forged(targets: Vec<BoundTarget>) -> Self {
        Self {
            targets,
            reason: String::new(),
        }
    }
}

struct Pass<'a> {
    kept: Vec<&'a BoundTarget>,
    reasons: Vec<&'static str>,
}

impl<'a> Pass<'a> {
    fn drop_if(&mut self, rule: &'static str, mut removes: impl FnMut(&BoundTarget) -> bool) {
        let before = self.kept.len();
        self.kept.retain(|t| !removes(t));
        assert!(self.kept.len() <= before, "a rule never adds targets");
        if self.kept.len() < before {
            self.reasons.push(rule);
        }
    }
}

/// Plans the targets one request may try, as a pure function of the
/// policy, the bound targets, the request tags and the call signals.
/// Rules: offline, private, tier, context, battery and thermal, budget,
/// then `order` with `default` first. Unbound policy targets never plan.
pub fn decide(
    policy: &RoutePolicy,
    targets: &[BoundTarget],
    tags: &RouteTags,
    signals: &Signals,
) -> RoutePlan {
    let bound = |id: &str| targets.iter().find(|t| t.id == id);
    let mut pass = Pass {
        kept: policy.order().iter().filter_map(|id| bound(id)).collect(),
        reasons: Vec::new(),
    };
    assert!(
        pass.kept.len() <= policy.order().len(),
        "one binding per id"
    );
    if pass.kept.len() < policy.order().len() {
        pass.reasons.push("unbound");
    }
    if signals.online == Some(false) {
        pass.drop_if("offline", |t| !t.on_device);
    }
    if tags.private {
        pass.drop_if("private", |t| !t.on_device);
    }
    if let Some(tier) = &tags.tier {
        pass.drop_if("tier", |t| {
            policy
                .target(&t.id)
                .is_none_or(|p| !p.tiers().contains(tier))
        });
    }
    let needed = signals.prompt_tokens.saturating_add(signals.max_tokens);
    pass.drop_if("context", |t| {
        policy
            .target(&t.id)
            .and_then(|p| p.context_tokens())
            .is_some_and(|limit| limit < needed)
    });
    let device = policy.device();
    if let (Some(min), Some(level)) = (device.min_battery_percent(), signals.battery_percent) {
        if level < min {
            pass.drop_if("battery", |t| t.on_device);
        }
    }
    if let (Some(max), Some(now)) = (device.max_thermal(), signals.thermal) {
        if now > max {
            pass.drop_if("thermal", |t| t.on_device);
        }
    }
    if let Some(cap) = policy.budget().cloud_usd_micros_per_day() {
        if signals.cloud_spent_today_usd_micros >= cap {
            pass.drop_if("budget", |t| !t.on_device);
        }
    }
    if let Some(first) = policy.default_target() {
        if let Some(at) = pass.kept.iter().position(|t| t.id == first) {
            let target = pass.kept.remove(at);
            pass.kept.insert(0, target);
        }
    }
    let plan = RoutePlan {
        targets: pass.kept.into_iter().cloned().collect(),
        reason: pass.reasons.join(", "),
    };
    assert!(
        plan.targets.len() <= policy.targets().len(),
        "plan within policy"
    );
    plan.assert_lawful(tags);
    plan
}
