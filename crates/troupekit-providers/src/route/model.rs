use super::decide::{decide, BoundTarget, RoutePlan};
use super::policy::{NetworkMode, RoutePolicy};
use super::signals::{DeviceState, NetworkProbe, PassiveProbe, RouteTags, Signals};
use super::RouteError;
use crate::catalog::Price;
use crate::model::Model;
use crate::request::{CompletionRequest, CompletionResponse, Delta};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use troupekit_core::clock::{Clock, SystemClock};
use troupekit_core::error::{CoreError, ProviderError, ProviderErrorKind};
use troupekit_core::event::Usage;
use troupekit_core::store::UsageLedger;
use troupekit_trace::{trace, trace_span, Span, Trace};

const DAY_MS: i64 = 86_400_000;

#[derive(Default)]
struct DayLedger(Mutex<HashMap<String, u64>>);

#[async_trait]
impl UsageLedger for DayLedger {
    async fn period_total(&self, key: &str) -> Result<u64, CoreError> {
        let totals = self.0.lock().unwrap_or_else(|p| p.into_inner());
        Ok(totals.get(key).copied().unwrap_or(0))
    }

    async fn record(&self, key: &str, usage: Usage) -> Result<(), CoreError> {
        let mut totals = self.0.lock().unwrap_or_else(|p| p.into_inner());
        let total = totals.entry(key.to_owned()).or_insert(0);
        *total = total.saturating_add(usage.cost_usd_micros);
        Ok(())
    }
}

struct Bound {
    target: BoundTarget,
    model: Arc<dyn Model>,
    price: Price,
}

struct Shared {
    policy: RoutePolicy,
    bound: Vec<Bound>,
    targets: Vec<BoundTarget>,
    app_probe: Option<Arc<dyn NetworkProbe>>,
    passive: PassiveProbe,
    device: Option<DeviceState>,
    ledger: Arc<dyn UsageLedger>,
    clock: Arc<dyn Clock>,
}

/// Builds a `RoutedModel`: bind every policy target to a model, on the
/// device or remote, then `build`. Defaults: a passive network probe, no
/// device state, an in-memory spend ledger per UTC day, the wall clock.
pub struct RoutedModelBuilder {
    policy: RoutePolicy,
    bound: Vec<Bound>,
    app_probe: Option<Arc<dyn NetworkProbe>>,
    device: Option<DeviceState>,
    ledger: Option<Arc<dyn UsageLedger>>,
    clock: Arc<dyn Clock>,
}

impl RoutedModelBuilder {
    /// Binds target `id` to `model`, declared as running on the device.
    /// `price` prices its replies for views and the spend ledger.
    /// Private and offline requests may reach it.
    pub fn on_device(mut self, id: &str, model: Arc<dyn Model>, price: Price) -> Self {
        self.bound.push(Bound {
            target: BoundTarget::on_device(id),
            model,
            price,
        });
        self
    }

    /// Binds target `id` to `model`, declared as remote.
    /// Its spend counts against the daily cap.
    /// Private and offline requests never reach it.
    pub fn remote(mut self, id: &str, model: Arc<dyn Model>, price: Price) -> Self {
        self.bound.push(Bound {
            target: BoundTarget::remote(id),
            model,
            price,
        });
        self
    }

    /// Reads the network state from `probe`, for example the app's
    /// `NetworkState`. When it reports unknown, the passive probe decides.
    /// Ignored when the policy pins `[network] mode`.
    pub fn network(mut self, probe: Arc<dyn NetworkProbe>) -> Self {
        self.app_probe = Some(probe);
        self
    }

    /// Reads battery and thermal state from `device` at each call.
    /// Without it both are unknown and their rules are skipped.
    /// Keep a clone to update it.
    pub fn device(mut self, device: DeviceState) -> Self {
        self.device = Some(device);
        self
    }

    /// Keeps remote spend in `ledger`, one key per policy and UTC day.
    /// Default: in memory, lost on restart. A failed read counts as the
    /// cap reached; a failed write is dropped and the reply still returned.
    pub fn ledger(mut self, ledger: Arc<dyn UsageLedger>) -> Self {
        self.ledger = Some(ledger);
        self
    }

    /// Reads time from `clock` for the UTC day and the passive probe.
    /// Default: the wall clock.
    /// Tests inject a `FixedClock`.
    pub fn clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    /// Checks the bindings and builds the routed model. Returns
    /// `RouteError::Bind` for an id the policy does not name, an id bound
    /// twice, or a policy target left unbound.
    pub fn build(self) -> Result<RoutedModel, RouteError> {
        let bind_error = |target: &str, message: &str| RouteError::Bind {
            target: target.to_owned(),
            message: message.to_owned(),
        };
        for (at, b) in self.bound.iter().enumerate() {
            if self.policy.target(b.target.id()).is_none() {
                return Err(bind_error(b.target.id(), "the policy names no such target"));
            }
            if self.bound[..at]
                .iter()
                .any(|o| o.target.id() == b.target.id())
            {
                return Err(bind_error(b.target.id(), "bound twice"));
            }
        }
        for t in self.policy.targets() {
            if !self.bound.iter().any(|b| b.target.id() == t.id()) {
                return Err(bind_error(t.id(), "the policy target is not bound"));
            }
        }
        assert_eq!(
            self.bound.len(),
            self.policy.targets().len(),
            "one binding per target"
        );
        let passive = PassiveProbe::new(self.policy.network()).with_clock(self.clock.clone());
        let id = format!("route/{}", self.policy.name());
        let shared = Shared {
            targets: self.bound.iter().map(|b| b.target.clone()).collect(),
            policy: self.policy,
            bound: self.bound,
            app_probe: self.app_probe,
            passive,
            device: self.device,
            ledger: self
                .ledger
                .unwrap_or_else(|| Arc::new(DayLedger::default())),
            clock: self.clock,
        };
        Ok(RoutedModel {
            shared: Arc::new(shared),
            tags: RouteTags::default(),
            id,
        })
    }
}

/// A `Model` that routes each request by a `RoutePolicy`: it plans the
/// eligible targets with `decide` and tries them in order. A private
/// request never reaches a remote target, even when every local one fails.
#[derive(Clone)]
pub struct RoutedModel {
    shared: Arc<Shared>,
    tags: RouteTags,
    id: String,
}

fn max_price(prices: impl Iterator<Item = Price>) -> Price {
    let mut out = Price::FREE;
    for p in prices {
        let lc = p.long_context;
        let input = lc.map_or(0, |l| l.input_usd_micros_per_mtok);
        let output = lc.map_or(0, |l| l.output_usd_micros_per_mtok);
        out.input_usd_micros_per_mtok = out
            .input_usd_micros_per_mtok
            .max(p.input_usd_micros_per_mtok)
            .max(input);
        out.output_usd_micros_per_mtok = out
            .output_usd_micros_per_mtok
            .max(p.output_usd_micros_per_mtok)
            .max(output);
        out.search_usd_micros = out.search_usd_micros.max(p.search_usd_micros);
        out.verified = out.verified && p.verified;
    }
    assert!(
        out.long_context.is_none(),
        "a view price folds long-context rates in"
    );
    out
}

fn estimate_prompt_tokens(req: &CompletionRequest) -> u64 {
    let bytes = req.system.as_ref().map_or(0, String::len)
        + serde_json::to_string(&req.messages).map_or(0, |s| s.len())
        + serde_json::to_string(&req.tools).map_or(0, |s| s.len());
    u64::try_from(bytes.div_ceil(4)).unwrap_or(u64::MAX)
}

impl RoutedModel {
    /// Starts a builder for `policy`. Every policy target must be bound
    /// with `on_device` or `remote` before `build`.
    /// Returns the builder.
    pub fn builder(policy: RoutePolicy) -> RoutedModelBuilder {
        RoutedModelBuilder {
            policy,
            bound: Vec::new(),
            app_probe: None,
            device: None,
            ledger: None,
            clock: Arc::new(SystemClock),
        }
    }

    /// The policy this model routes by.
    /// Never changes after `build`.
    /// Shared by every view.
    pub fn policy(&self) -> &RoutePolicy {
        &self.shared.policy
    }

    /// Returns a view of this model with `tags` added: a cheap clone that
    /// shares targets, probe and ledger. `private` is sticky, so a view of
    /// a private view stays private; a missing tier keeps the current one.
    pub fn view(&self, tags: RouteTags) -> RoutedModel {
        let private = self.tags.private || tags.private;
        let tier = tags.tier.or_else(|| self.tags.tier.clone());
        let mut id = format!("route/{}", self.shared.policy.name());
        if private {
            id.push_str(":private");
        }
        if let Some(t) = &tier {
            id.push(':');
            id.push_str(t);
        }
        let tags = RouteTags { private, tier };
        assert!(
            !self.tags.private || tags.private,
            "a view never clears private"
        );
        RoutedModel {
            shared: self.shared.clone(),
            tags,
            id,
        }
    }

    /// Lists the views to register under `prefix`: `prefix`,
    /// `prefix:private` and `prefix:<tier>` per policy tier, each with the
    /// highest price among the targets it can reach, so budgets stay safe.
    pub fn views(&self, prefix: &str) -> Vec<(String, Arc<dyn Model>, Price)> {
        let base = self.view(RouteTags::default());
        let mut tagged = vec![
            (String::new(), base.clone()),
            (
                ":private".to_owned(),
                base.view(RouteTags::default().with_private(true)),
            ),
        ];
        for tier in self.shared.policy.tiers() {
            let view = base.view(RouteTags::default().with_tier(tier.clone()));
            tagged.push((format!(":{tier}"), view));
        }
        let out: Vec<(String, Arc<dyn Model>, Price)> = tagged
            .into_iter()
            .map(|(suffix, view)| {
                let name = format!("{prefix}{suffix}");
                let price = view.price();
                let view = RoutedModel {
                    id: name.clone(),
                    ..view
                };
                (name, Arc::new(view) as Arc<dyn Model>, price)
            })
            .collect();
        assert_eq!(
            out.len(),
            2 + self.shared.policy.tiers().len(),
            "one view per name"
        );
        out
    }

    /// The highest price among the targets this view can reach: on-device
    /// only when private, only targets serving the tier when one is set.
    /// Budgets charge it, so they never under-count.
    pub fn price(&self) -> Price {
        let shared = &self.shared;
        let reaches = |b: &&Bound| {
            let serves = self.tags.tier.as_ref().is_none_or(|tier| {
                shared
                    .policy
                    .target(b.target.id())
                    .is_some_and(|t| t.tiers().contains(tier))
            });
            serves && (!self.tags.private || b.target.is_on_device())
        };
        max_price(shared.bound.iter().filter(reaches).map(|b| b.price))
    }

    fn day_key(&self) -> String {
        let day = self.shared.clock.now_ms().div_euclid(DAY_MS);
        format!("route/{}/remote/{day}", self.shared.policy.name())
    }

    fn online(&self) -> Option<bool> {
        let shared = &self.shared;
        match shared.policy.network().mode() {
            NetworkMode::Online => Some(true),
            NetworkMode::Offline => Some(false),
            _ => shared
                .app_probe
                .as_ref()
                .and_then(|p| p.online())
                .or_else(|| shared.passive.online()),
        }
    }

    async fn plan(
        &self,
        req: &CompletionRequest,
        stats: &mut RouteStats,
    ) -> Result<RoutePlan, ProviderError> {
        let shared = &self.shared;
        let spent = match shared.policy.budget().cloud_usd_micros_per_day() {
            None => 0,
            Some(_) => shared
                .ledger
                .period_total(&self.day_key())
                .await
                .unwrap_or(u64::MAX),
        };
        let mut signals = Signals::default()
            .with_online(self.online())
            .with_cloud_spent_today_usd_micros(spent)
            .with_prompt_tokens(estimate_prompt_tokens(req))
            .with_max_tokens(u64::from(req.max_tokens));
        if let Some(device) = &shared.device {
            signals = signals
                .with_battery_percent(device.battery_percent())
                .with_thermal(device.thermal());
        }
        let plan = decide(&shared.policy, &shared.targets, &self.tags, &signals);
        let held = !self.tags.private || plan.targets().iter().all(BoundTarget::is_on_device);
        trace!(
            Trace::Laws,
            "law.check",
            law = "private_plan_has_only_on_device_targets",
            held = held,
            private = self.tags.private,
            targets = plan.targets().len(),
        );
        plan.assert_lawful(&self.tags);
        stats.planned(&plan, signals);
        if plan.is_empty() {
            let message = format!("no route: {}", plan.reason());
            return Err(ProviderError::new(ProviderErrorKind::Network, message));
        }
        Ok(plan)
    }

    fn decide_span(&self) -> Span {
        let mut span = trace_span!(
            Trace::Routing,
            "route.decide",
            model = self.id.as_str(),
            private = self.tags.private,
        );
        if let Some(tier) = &self.tags.tier {
            span.record("tier", tier.as_str());
        }
        span
    }

    async fn complete_routed(
        &self,
        req: &CompletionRequest,
        stats: &mut RouteStats,
    ) -> Result<CompletionResponse, ProviderError> {
        let plan = self.plan(req, stats).await?;
        let fallback = self.shared.policy.fallback().on_error();
        let mut last = None;
        for target in plan.targets() {
            let b = self.bound(target);
            let result = b.model.complete(req).await;
            self.settle(b, &result).await;
            match result {
                Ok(reply) => {
                    stats.chosen = Some(target.id().to_owned());
                    return Ok(reply);
                }
                Err(err) if !fallback => return Err(err),
                Err(err) => {
                    trace_fallback(target, &err);
                    last = Some(err);
                }
            }
        }
        Err(last.expect("a non-empty plan tried a target"))
    }

    async fn stream_routed(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
        stats: &mut RouteStats,
    ) -> Result<CompletionResponse, ProviderError> {
        let plan = self.plan(req, stats).await?;
        let fallback = self.shared.policy.fallback().on_error();
        let mut last = None;
        for target in plan.targets() {
            let b = self.bound(target);
            let mut emitted = false;
            let result = b
                .model
                .stream(req, &mut |d: &Delta| {
                    emitted = true;
                    on_delta(d);
                })
                .await;
            self.settle(b, &result).await;
            match result {
                Ok(reply) => {
                    stats.chosen = Some(target.id().to_owned());
                    return Ok(reply);
                }
                Err(err) if emitted || !fallback => return Err(err),
                Err(err) => {
                    trace_fallback(target, &err);
                    last = Some(err);
                }
            }
        }
        Err(last.expect("a non-empty plan tried a target"))
    }

    fn bound(&self, target: &BoundTarget) -> &Bound {
        assert!(
            !self.tags.private || target.is_on_device(),
            "private request about to call a remote target"
        );
        let b = self
            .shared
            .bound
            .iter()
            .find(|b| b.target.id() == target.id());
        let b = b.expect("a planned target is bound");
        assert_eq!(
            b.target.is_on_device(),
            target.is_on_device(),
            "plan keeps locality"
        );
        b
    }

    async fn settle(&self, b: &Bound, result: &Result<CompletionResponse, ProviderError>) {
        if b.target.is_on_device() {
            return;
        }
        let shared = &self.shared;
        match result {
            Ok(reply) => {
                shared.passive.record_remote(false);
                if let Some(p) = &shared.app_probe {
                    p.record_remote(false);
                }
                let cost = reply
                    .usage
                    .cost_usd_micros
                    .max(b.price.reply_cost_usd_micros(reply));
                let usage = Usage {
                    cost_usd_micros: cost,
                    ..reply.usage
                };
                let _recorded = shared.ledger.record(&self.day_key(), usage).await;
            }
            Err(err) if err.kind == ProviderErrorKind::Network => {
                shared.passive.record_remote(true);
                if let Some(p) = &shared.app_probe {
                    p.record_remote(true);
                }
            }
            Err(_) => {}
        }
    }
}

fn trace_fallback(target: &BoundTarget, err: &ProviderError) {
    trace!(
        Trace::Routing,
        "route.fallback",
        from = target.id(),
        on_device = target.is_on_device(),
        kind = format!("{:?}", err.kind),
    );
}

#[derive(Default)]
struct RouteStats {
    plan: String,
    reason: String,
    signals: Option<Signals>,
    chosen: Option<String>,
}

impl RouteStats {
    fn planned(&mut self, plan: &RoutePlan, signals: Signals) {
        let ids: Vec<&str> = plan.targets().iter().map(BoundTarget::id).collect();
        self.plan = ids.join(",");
        self.reason = plan.reason().to_owned();
        self.signals = Some(signals);
    }

    fn finish(&self, span: &mut Span, result: &Result<CompletionResponse, ProviderError>) {
        if !span.is_enabled() {
            return;
        }
        span.record("plan", self.plan.as_str());
        span.record("reason", self.reason.as_str());
        if let Some(signals) = &self.signals {
            if let Some(online) = signals.online {
                span.record("online", online);
            }
            if let Some(battery) = signals.battery_percent {
                span.record("battery_percent", battery);
            }
            if let Some(thermal) = signals.thermal {
                span.record("thermal", format!("{thermal:?}").to_ascii_lowercase());
            }
            span.record(
                "spent_today_usd_micros",
                signals.cloud_spent_today_usd_micros,
            );
            span.record("prompt_tokens", signals.prompt_tokens);
            span.record("max_tokens", signals.max_tokens);
        }
        if let Some(chosen) = &self.chosen {
            span.record("chosen", chosen.as_str());
        }
        if let Err(e) = result {
            span.fail("no route target answered");
            span.record("kind", format!("{:?}", e.kind));
        }
    }
}

#[async_trait]
impl Model for RoutedModel {
    fn id(&self) -> &str {
        &self.id
    }

    async fn complete(&self, req: &CompletionRequest) -> Result<CompletionResponse, ProviderError> {
        let mut span = self.decide_span();
        let mut stats = RouteStats::default();
        let result = span.instrument(self.complete_routed(req, &mut stats)).await;
        stats.finish(&mut span, &result);
        result
    }

    async fn stream(
        &self,
        req: &CompletionRequest,
        on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        let mut span = self.decide_span();
        let mut stats = RouteStats::default();
        let result = span
            .instrument(self.stream_routed(req, on_delta, &mut stats))
            .await;
        stats.finish(&mut span, &result);
        result
    }
}
