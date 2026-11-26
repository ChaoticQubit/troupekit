//! Model routing: a TOML policy, a pure route decision over bound targets
//! and network or device signals, and `RoutedModel`, a `Model` that tries
//! the planned targets in order. A private request never leaves the device.

mod decide;
mod model;
mod policy;
mod signals;

pub use decide::{decide, BoundTarget, RoutePlan};
pub use model::{RoutedModel, RoutedModelBuilder};

pub use policy::{
    BudgetPolicy, DevicePolicy, FallbackPolicy, NetworkMode, NetworkPolicy, RoutePolicy,
    RouteTarget, Thermal,
};
pub use signals::{DeviceState, NetworkProbe, NetworkState, PassiveProbe, RouteTags, Signals};

/// Why a route policy could not be loaded or a routed model built.
/// `Parse` is bad TOML or an unknown key, `Invalid` a value that breaks a
/// rule, both naming the policy file; `Bind` is a target binding error.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RouteError {
    #[error("{file}: route policy is not valid TOML: {message}")]
    Parse { file: String, message: String },
    #[error("{file}: {key}: {message}")]
    Invalid {
        file: String,
        key: String,
        message: String,
    },
    #[error("route target {target}: {message}")]
    Bind { target: String, message: String },
}
