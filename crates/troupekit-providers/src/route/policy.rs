use super::RouteError;
use serde::Deserialize;
use std::collections::HashSet;

const MAX_TIERS_PER_TARGET: usize = 16;
const MAX_CONTEXT_TOKENS: i64 = 100_000_000;
const MAX_OFFLINE_AFTER_FAILURES: i64 = 1_000;
const MAX_RETRY_ONLINE_AFTER_MS: i64 = 86_400_000;
const MAX_CLOUD_USD_PER_DAY: f64 = 1_000_000.0;
const DEFAULT_OFFLINE_AFTER_FAILURES: u32 = 1;
const DEFAULT_RETRY_ONLINE_AFTER_MS: u64 = 30_000;

/// How a routed model learns whether the network is up.
/// `Auto` asks the app's network handle, else a passive probe that counts
/// network errors. `Online` and `Offline` pin the state.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum NetworkMode {
    #[default]
    Auto,
    Online,
    Offline,
}

/// A device's thermal state, coolest first, as phones report it.
/// The `[device] max_thermal` key names the hottest state at which
/// on-device targets still run.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Thermal {
    Nominal,
    Fair,
    Serious,
    Critical,
}

/// One named `[[target]]` of a policy: its id, the task tiers it serves
/// and its context window. Whether it runs on the device is never read
/// from TOML: code says so when it binds a model to the id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RouteTarget {
    id: String,
    tiers: Vec<String>,
    context_tokens: Option<u64>,
}

impl RouteTarget {
    /// The target id, unique in its policy, of `[a-z0-9_-]`.
    /// Models are bound to targets by this id.
    /// Views and errors name it.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// The task tiers this target serves, in file order.
    /// Empty when the target serves no tier.
    /// A request tagged with a tier only reaches targets listing it.
    pub fn tiers(&self) -> &[String] {
        &self.tiers
    }

    /// The context window in tokens, when the file sets one.
    /// A request whose prompt plus reply cannot fit skips this target.
    /// `None` means no limit is checked.
    pub fn context_tokens(&self) -> Option<u64> {
        self.context_tokens
    }
}

/// The `[network]` table: how the network state is read and when a
/// passive probe flips to offline and back. Defaults: `auto`, offline
/// after 1 network error, retry online after 30 000 ms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkPolicy {
    mode: NetworkMode,
    offline_after_failures: u32,
    retry_online_after_ms: u64,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self {
            mode: NetworkMode::Auto,
            offline_after_failures: DEFAULT_OFFLINE_AFTER_FAILURES,
            retry_online_after_ms: DEFAULT_RETRY_ONLINE_AFTER_MS,
        }
    }
}

impl NetworkPolicy {
    /// How the network state is read.
    /// Default `Auto`.
    /// See `NetworkMode`.
    pub fn mode(&self) -> NetworkMode {
        self.mode
    }

    /// Network errors in a row before the passive probe reports offline.
    /// Default 1, range 1 to 1 000.
    /// Only network-kind errors count.
    pub fn offline_after_failures(&self) -> u32 {
        self.offline_after_failures
    }

    /// Milliseconds after which an offline passive probe tries online.
    /// Default 30 000, range 1 to 86 400 000.
    /// One success resets the count.
    pub fn retry_online_after_ms(&self) -> u64 {
        self.retry_online_after_ms
    }
}

/// The `[budget]` table: a daily cap on remote spend, kept in millionths
/// of a dollar. Written in dollars as `cloud_usd_per_day`.
/// Default: no cap.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BudgetPolicy {
    cloud_usd_micros_per_day: Option<u64>,
}

impl BudgetPolicy {
    /// The daily cap on remote spend in millionths of a dollar.
    /// `None` when the policy sets no cap.
    /// Once the day's spend reaches it, remote targets are skipped.
    pub fn cloud_usd_micros_per_day(&self) -> Option<u64> {
        self.cloud_usd_micros_per_day
    }
}

/// The `[device]` table: battery and thermal limits for on-device
/// targets. Default: no limit. An unknown device state skips the rule,
/// so a limit never blocks a device that cannot report its state.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DevicePolicy {
    min_battery_percent: Option<u8>,
    max_thermal: Option<Thermal>,
}

impl DevicePolicy {
    /// The lowest battery percent at which on-device targets still run.
    /// `None` when unset; range 0 to 100.
    /// Below it, on-device targets are skipped.
    pub fn min_battery_percent(&self) -> Option<u8> {
        self.min_battery_percent
    }

    /// The hottest thermal state at which on-device targets still run.
    /// `None` when unset.
    /// Hotter than it, on-device targets are skipped.
    pub fn max_thermal(&self) -> Option<Thermal> {
        self.max_thermal
    }
}

/// The `[fallback]` table: whether a failed target hands the request to
/// the next one in the plan. Default true. Never switches after the first
/// streamed delta, and never leaves the device for a private request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FallbackPolicy {
    on_error: bool,
}

impl Default for FallbackPolicy {
    fn default() -> Self {
        Self { on_error: true }
    }
}

impl FallbackPolicy {
    /// Whether an error moves the request to the next planned target.
    /// Default true.
    /// When false, the first target's error is returned.
    pub fn on_error(&self) -> bool {
        self.on_error
    }
}

/// A checked routing policy loaded from TOML: named targets, their order
/// and the network, budget, device and fallback rules. Built only by
/// `parse`, so every value is in range and every target name resolves.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutePolicy {
    name: String,
    default_target: Option<String>,
    order: Vec<String>,
    targets: Vec<RouteTarget>,
    tiers: Vec<String>,
    network: NetworkPolicy,
    budget: BudgetPolicy,
    device: DevicePolicy,
    fallback: FallbackPolicy,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawPolicy {
    name: String,
    #[serde(default)]
    default: Option<String>,
    #[serde(default)]
    order: Vec<String>,
    #[serde(default, rename = "target")]
    targets: Vec<RawTarget>,
    #[serde(default)]
    network: RawNetwork,
    #[serde(default)]
    budget: RawBudget,
    #[serde(default)]
    device: RawDevice,
    #[serde(default)]
    fallback: RawFallback,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawTarget {
    id: String,
    #[serde(default)]
    tiers: Vec<String>,
    #[serde(default)]
    context_tokens: Option<i64>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawNetwork {
    #[serde(default)]
    mode: NetworkMode,
    #[serde(default)]
    offline_after_failures: Option<i64>,
    #[serde(default)]
    retry_online_after_ms: Option<i64>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawBudget {
    #[serde(default)]
    cloud_usd_per_day: Option<f64>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawDevice {
    #[serde(default)]
    min_battery_percent: Option<i64>,
    #[serde(default)]
    max_thermal: Option<Thermal>,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFallback {
    #[serde(default)]
    on_error: Option<bool>,
}

fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= RoutePolicy::MAX_ID_BYTES
        && id
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'_' || b == b'-')
}

fn in_range(value: i64, low: i64, high: i64) -> bool {
    (low..=high).contains(&value)
}

impl RoutePolicy {
    /// The most targets one policy may name.
    /// Setting S34 of the 0.3 plan.
    /// `parse` refuses a file with more.
    pub const MAX_TARGETS: usize = 16;

    /// The largest policy file accepted, in bytes.
    /// Setting S34 of the 0.3 plan: 64 KiB.
    /// `parse` refuses a longer text before reading it.
    pub const MAX_BYTES: usize = 64 * 1024;

    /// The longest policy name, target id or tier name, in bytes.
    /// Each is `[a-z0-9_-]` and not empty.
    /// `parse` refuses a longer one.
    pub const MAX_ID_BYTES: usize = 64;

    /// Parses and checks a policy from TOML `text`; `file` names it in
    /// errors. Returns `RouteError::Parse` for bad TOML or an unknown key
    /// and `RouteError::Invalid` naming the key for a value out of rule.
    pub fn parse(file: &str, text: &str) -> Result<Self, RouteError> {
        let invalid = |key: &str, message: String| RouteError::Invalid {
            file: file.to_owned(),
            key: key.to_owned(),
            message,
        };
        if text.len() > Self::MAX_BYTES {
            return Err(invalid(
                "file",
                format!("{} bytes, limit {}", text.len(), Self::MAX_BYTES),
            ));
        }
        let raw: RawPolicy = toml::from_str(text).map_err(|e| RouteError::Parse {
            file: file.to_owned(),
            message: e.to_string(),
        })?;
        if !valid_id(&raw.name) {
            return Err(invalid(
                "name",
                "must be 1 to 64 bytes of a-z, 0-9, _ or -".into(),
            ));
        }
        if raw.targets.is_empty() {
            return Err(invalid(
                "target",
                "a policy needs at least one target".into(),
            ));
        }
        if raw.targets.len() > Self::MAX_TARGETS {
            return Err(invalid(
                "target",
                format!("{} targets, limit {}", raw.targets.len(), Self::MAX_TARGETS),
            ));
        }
        let mut targets = Vec::with_capacity(raw.targets.len());
        let mut ids = HashSet::new();
        let mut tiers: Vec<String> = Vec::new();
        for t in raw.targets {
            if !valid_id(&t.id) {
                return Err(invalid(
                    "target.id",
                    "must be 1 to 64 bytes of a-z, 0-9, _ or -".into(),
                ));
            }
            if !ids.insert(t.id.clone()) {
                return Err(invalid("target", format!("id {} is repeated", t.id)));
            }
            let key = format!("target.{}.tiers", t.id);
            if t.tiers.len() > MAX_TIERS_PER_TARGET {
                return Err(invalid(
                    &key,
                    format!("{} tiers, limit {MAX_TIERS_PER_TARGET}", t.tiers.len()),
                ));
            }
            let mut seen = HashSet::new();
            for tier in &t.tiers {
                if !valid_id(tier) {
                    return Err(invalid(
                        &key,
                        "a tier must be 1 to 64 bytes of a-z, 0-9, _ or -".into(),
                    ));
                }
                if tier == "private" {
                    return Err(invalid(&key, "tier name private is reserved".into()));
                }
                if !seen.insert(tier.as_str()) {
                    return Err(invalid(&key, format!("tier {tier} is repeated")));
                }
                if !tiers.contains(tier) {
                    tiers.push(tier.clone());
                }
            }
            let context_tokens = match t.context_tokens {
                None => None,
                Some(n) if in_range(n, 1, MAX_CONTEXT_TOKENS) => Some(n.unsigned_abs()),
                Some(n) => {
                    return Err(invalid(
                        &format!("target.{}.context_tokens", t.id),
                        format!("{n} is out of range 1 to {MAX_CONTEXT_TOKENS}"),
                    ))
                }
            };
            targets.push(RouteTarget {
                id: t.id,
                tiers: t.tiers,
                context_tokens,
            });
        }
        if let Some(d) = &raw.default {
            if !ids.contains(d) {
                return Err(invalid("default", format!("names no target: {d}")));
            }
        }
        let mut order = Vec::with_capacity(targets.len());
        for id in raw.order {
            if !ids.contains(&id) {
                return Err(invalid("order", format!("names no target: {id}")));
            }
            if order.contains(&id) {
                return Err(invalid("order", format!("names {id} twice")));
            }
            order.push(id);
        }
        for t in &targets {
            if !order.contains(&t.id) {
                order.push(t.id.clone());
            }
        }
        assert_eq!(order.len(), targets.len(), "order covers every target once");
        let network = Self::parse_network(&raw.network, &invalid)?;
        let budget = match raw.budget.cloud_usd_per_day {
            None => BudgetPolicy::default(),
            Some(usd) if usd.is_finite() && (0.0..=MAX_CLOUD_USD_PER_DAY).contains(&usd) => {
                BudgetPolicy {
                    cloud_usd_micros_per_day: Some((usd * 1_000_000.0).round() as u64),
                }
            }
            Some(usd) => {
                return Err(invalid(
                    "budget.cloud_usd_per_day",
                    format!("{usd} is out of range 0 to {MAX_CLOUD_USD_PER_DAY}"),
                ))
            }
        };
        let min_battery_percent = match raw.device.min_battery_percent {
            None => None,
            Some(p) => Some(u8::try_from(p).ok().filter(|p| *p <= 100).ok_or_else(|| {
                invalid(
                    "device.min_battery_percent",
                    format!("{p} is out of range 0 to 100"),
                )
            })?),
        };
        Ok(Self {
            name: raw.name,
            default_target: raw.default,
            order,
            targets,
            tiers,
            network,
            budget,
            device: DevicePolicy {
                min_battery_percent,
                max_thermal: raw.device.max_thermal,
            },
            fallback: FallbackPolicy {
                on_error: raw.fallback.on_error.unwrap_or(true),
            },
        })
    }

    fn parse_network(
        raw: &RawNetwork,
        invalid: &dyn Fn(&str, String) -> RouteError,
    ) -> Result<NetworkPolicy, RouteError> {
        let offline_after_failures = match raw.offline_after_failures {
            None => DEFAULT_OFFLINE_AFTER_FAILURES,
            Some(n) => u32::try_from(n)
                .ok()
                .filter(|_| in_range(n, 1, MAX_OFFLINE_AFTER_FAILURES))
                .ok_or_else(|| {
                    invalid(
                        "network.offline_after_failures",
                        format!("{n} is out of range 1 to {MAX_OFFLINE_AFTER_FAILURES}"),
                    )
                })?,
        };
        let retry_online_after_ms = match raw.retry_online_after_ms {
            None => DEFAULT_RETRY_ONLINE_AFTER_MS,
            Some(n) if in_range(n, 1, MAX_RETRY_ONLINE_AFTER_MS) => n.unsigned_abs(),
            Some(n) => {
                return Err(invalid(
                    "network.retry_online_after_ms",
                    format!("{n} is out of range 1 to {MAX_RETRY_ONLINE_AFTER_MS}"),
                ))
            }
        };
        Ok(NetworkPolicy {
            mode: raw.mode,
            offline_after_failures,
            retry_online_after_ms,
        })
    }

    /// The policy name, used as the routed model's display name.
    /// Of `[a-z0-9_-]`, 1 to 64 bytes.
    /// Views are named from it by the caller.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The target tried first whenever it is eligible.
    /// `None` when the file sets no `default`.
    /// Always names a target of this policy.
    pub fn default_target(&self) -> Option<&str> {
        self.default_target.as_deref()
    }

    /// Every target id in try order: the file's `order` first, then any
    /// target it leaves out, in file order.
    /// Each id appears exactly once.
    pub fn order(&self) -> &[String] {
        &self.order
    }

    /// Every target in file order.
    /// At least one, at most `MAX_TARGETS`.
    /// Ids are unique.
    pub fn targets(&self) -> &[RouteTarget] {
        &self.targets
    }

    /// Looks up the target with `id`.
    /// Returns it, or `None` when the policy names no such target.
    /// Ids match exactly.
    pub fn target(&self, id: &str) -> Option<&RouteTarget> {
        self.targets.iter().find(|t| t.id == id)
    }

    /// Every tier any target serves, in first-seen file order.
    /// Empty when no target lists a tier.
    /// A routed model offers one view per tier.
    pub fn tiers(&self) -> &[String] {
        &self.tiers
    }

    /// The `[network]` rules, or their defaults.
    /// See `NetworkPolicy`.
    /// Never absent.
    pub fn network(&self) -> &NetworkPolicy {
        &self.network
    }

    /// The `[budget]` rules, or their defaults.
    /// See `BudgetPolicy`.
    /// Never absent.
    pub fn budget(&self) -> &BudgetPolicy {
        &self.budget
    }

    /// The `[device]` rules, or their defaults.
    /// See `DevicePolicy`.
    /// Never absent.
    pub fn device(&self) -> &DevicePolicy {
        &self.device
    }

    /// The `[fallback]` rules, or their defaults.
    /// See `FallbackPolicy`.
    /// Never absent.
    pub fn fallback(&self) -> &FallbackPolicy {
        &self.fallback
    }
}
