use super::policy::{NetworkPolicy, Thermal};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use troupekit_core::clock::{Clock, SystemClock};

const UNKNOWN: u8 = u8::MAX;
const OFFLINE: u8 = 0;
const ONLINE: u8 = 1;

/// The tags of one routed request: whether it is private and which task
/// tier it asks for. A private request only reaches on-device targets.
/// Defaults: not private, no tier.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct RouteTags {
    pub private: bool,
    pub tier: Option<String>,
}

impl RouteTags {
    /// Returns these tags with `private` set.
    /// A private request never leaves the device.
    /// Once set by a view it is never cleared by a later view.
    pub fn with_private(mut self, private: bool) -> Self {
        self.private = private;
        self
    }

    /// Returns these tags asking for `tier`.
    /// Only targets listing the tier stay in the plan.
    /// The tier is compared exactly.
    pub fn with_tier(mut self, tier: impl Into<String>) -> Self {
        self.tier = Some(tier.into());
        self
    }
}

/// What the routed model knows at call time: network and device state,
/// today's remote spend and the request size in tokens. `None` means
/// unknown, and an unknown state never drops a target.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Signals {
    pub online: Option<bool>,
    pub battery_percent: Option<u8>,
    pub thermal: Option<Thermal>,
    pub cloud_spent_today_usd_micros: u64,
    pub prompt_tokens: u64,
    pub max_tokens: u64,
}

impl Signals {
    /// Returns these signals with the network state set.
    /// `Some(false)` drops remote targets.
    /// `None` is unknown and drops nothing.
    pub fn with_online(mut self, online: Option<bool>) -> Self {
        self.online = online;
        self
    }

    /// Returns these signals with the battery percent set; a value above
    /// 100 counts as 100. Below the policy minimum, on-device targets are
    /// dropped. `None` is unknown and drops nothing.
    pub fn with_battery_percent(mut self, percent: Option<u8>) -> Self {
        self.battery_percent = percent.map(|p| p.min(100));
        assert!(self.battery_percent.is_none_or(|p| p <= 100));
        self
    }

    /// Returns these signals with the thermal state set.
    /// Hotter than the policy maximum, on-device targets are dropped.
    /// `None` is unknown and drops nothing.
    pub fn with_thermal(mut self, thermal: Option<Thermal>) -> Self {
        self.thermal = thermal;
        self
    }

    /// Returns these signals with today's remote spend set, in millionths
    /// of a dollar. At or above the policy cap, remote targets are
    /// dropped.
    pub fn with_cloud_spent_today_usd_micros(mut self, spent: u64) -> Self {
        self.cloud_spent_today_usd_micros = spent;
        self
    }

    /// Returns these signals with the estimated prompt size in tokens.
    /// With `max_tokens` it decides whether a target's context fits.
    /// The sum saturates in that check.
    pub fn with_prompt_tokens(mut self, tokens: u64) -> Self {
        self.prompt_tokens = tokens;
        self
    }

    /// Returns these signals with the reply cap in tokens.
    /// With `prompt_tokens` it decides whether a target's context fits.
    /// The sum saturates in that check.
    pub fn with_max_tokens(mut self, tokens: u64) -> Self {
        self.max_tokens = tokens;
        self
    }
}

/// Tells a routed model whether the network is up. `online` returns
/// `None` when unknown. `record_remote` hears each remote call's outcome:
/// true for a network-kind failure, false for a success.
pub trait NetworkProbe: Send + Sync {
    fn online(&self) -> Option<bool>;

    /// Hears the outcome of one remote call: `network_failed` is true for
    /// a network-kind error, false for a success. Other errors are not
    /// reported. The default ignores it.
    fn record_remote(&self, network_failed: bool) {
        let _ = network_failed;
    }
}

/// A shared network state the app sets from its own reachability API:
/// unknown, online or offline. Clones share one state, so the app keeps
/// a clone and the routed model reads another.
#[derive(Debug, Clone)]
pub struct NetworkState(Arc<AtomicU8>);

impl Default for NetworkState {
    fn default() -> Self {
        Self::new()
    }
}

impl NetworkState {
    /// Builds a state that reads unknown until `set` is called.
    /// Returns the handle.
    /// Clone it to share it.
    pub fn new() -> Self {
        Self(Arc::new(AtomicU8::new(UNKNOWN)))
    }

    /// Sets the state: `Some(true)` online, `Some(false)` offline, `None`
    /// unknown. Every clone sees the change on its next read.
    /// Never blocks.
    pub fn set(&self, online: Option<bool>) {
        let raw = match online {
            None => UNKNOWN,
            Some(false) => OFFLINE,
            Some(true) => ONLINE,
        };
        self.0.store(raw, Ordering::SeqCst);
    }
}

impl NetworkProbe for NetworkState {
    fn online(&self) -> Option<bool> {
        match self.0.load(Ordering::SeqCst) {
            UNKNOWN => None,
            OFFLINE => Some(false),
            ONLINE => Some(true),
            raw => unreachable!("network state holds unknown value {raw}"),
        }
    }
}

#[derive(Debug, Default)]
struct ProbeState {
    failures: u32,
    offline_since_ms: Option<i64>,
}

/// Infers the network state from remote outcomes: offline after
/// `offline_after_failures` network errors in a row, then unknown again
/// once `retry_online_after_ms` has passed, so one call can try online.
pub struct PassiveProbe {
    failures_max: u32,
    retry_after_ms: i64,
    clock: Arc<dyn Clock>,
    state: Mutex<ProbeState>,
}

impl PassiveProbe {
    /// Builds a probe from the policy's `[network]` rules, reading the
    /// wall clock. Starts unknown with no failures.
    /// Returns the probe.
    pub fn new(network: &NetworkPolicy) -> Self {
        let failures_max = network.offline_after_failures();
        assert!(failures_max >= 1, "policy allows at least one failure");
        let retry_after_ms =
            i64::try_from(network.retry_online_after_ms()).expect("policy bounds the retry delay");
        assert!(retry_after_ms >= 1, "policy retry delay is positive");
        Self {
            failures_max,
            retry_after_ms,
            clock: Arc::new(SystemClock),
            state: Mutex::new(ProbeState::default()),
        }
    }

    /// Returns this probe reading `clock` instead of the wall clock.
    /// Tests inject a `FixedClock`.
    /// The state is kept.
    pub fn with_clock(mut self, clock: Arc<dyn Clock>) -> Self {
        self.clock = clock;
        self
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, ProbeState> {
        self.state.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl NetworkProbe for PassiveProbe {
    fn online(&self) -> Option<bool> {
        let state = self.lock();
        let since = state.offline_since_ms?;
        assert!(
            state.failures >= self.failures_max,
            "offline needs the failures"
        );
        let waited = self.clock.now_ms().saturating_sub(since);
        if waited >= self.retry_after_ms {
            None
        } else {
            Some(false)
        }
    }

    fn record_remote(&self, network_failed: bool) {
        let now = self.clock.now_ms();
        let mut state = self.lock();
        if !network_failed {
            *state = ProbeState::default();
            return;
        }
        state.failures = state.failures.saturating_add(1);
        if state.failures >= self.failures_max {
            state.offline_since_ms = Some(now);
        }
        assert!(
            state.offline_since_ms.is_some() || state.failures < self.failures_max,
            "enough failures always mark offline"
        );
    }
}

/// A shared device state the app sets from its battery and thermal
/// APIs. Both start unknown, and an unknown value skips its rule. Clones
/// share one state.
#[derive(Debug, Clone)]
pub struct DeviceState(Arc<(AtomicU8, AtomicU8)>);

impl Default for DeviceState {
    fn default() -> Self {
        Self::new()
    }
}

fn thermal_code(thermal: Thermal) -> u8 {
    match thermal {
        Thermal::Nominal => 0,
        Thermal::Fair => 1,
        Thermal::Serious => 2,
        Thermal::Critical => 3,
    }
}

impl DeviceState {
    /// Builds a state with battery and thermal unknown.
    /// Returns the handle.
    /// Clone it to share it.
    pub fn new() -> Self {
        Self(Arc::new((AtomicU8::new(UNKNOWN), AtomicU8::new(UNKNOWN))))
    }

    /// Sets the battery percent, or unknown with `None`; a value above
    /// 100 counts as 100. Every clone sees the change.
    /// Never blocks.
    pub fn set_battery_percent(&self, percent: Option<u8>) {
        let stored = percent.map_or(UNKNOWN, |p| p.min(100));
        assert!(stored <= 100 || stored == UNKNOWN);
        self.0 .0.store(stored, Ordering::SeqCst);
    }

    /// Sets the thermal state, or unknown with `None`.
    /// Every clone sees the change.
    /// Never blocks.
    pub fn set_thermal(&self, thermal: Option<Thermal>) {
        self.0
             .1
            .store(thermal.map_or(UNKNOWN, thermal_code), Ordering::SeqCst);
    }

    /// The battery percent, or `None` when unknown.
    /// Always 0 to 100 when known.
    /// Read at each routed call.
    pub fn battery_percent(&self) -> Option<u8> {
        let raw = self.0 .0.load(Ordering::SeqCst);
        assert!(raw <= 100 || raw == UNKNOWN, "battery holds {raw}");
        (raw != UNKNOWN).then_some(raw)
    }

    /// The thermal state, or `None` when unknown.
    /// Read at each routed call.
    /// Never blocks.
    pub fn thermal(&self) -> Option<Thermal> {
        match self.0 .1.load(Ordering::SeqCst) {
            UNKNOWN => None,
            0 => Some(Thermal::Nominal),
            1 => Some(Thermal::Fair),
            2 => Some(Thermal::Serious),
            3 => Some(Thermal::Critical),
            raw => unreachable!("thermal state holds unknown value {raw}"),
        }
    }
}
