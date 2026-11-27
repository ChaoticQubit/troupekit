use crate::agent::Agent;
use crate::error::KitError;
use crate::graph::Graph;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use troupekit_agent::{ModelHandle, Persona};
use troupekit_core::error::{ProviderError, ProviderErrorKind};
use troupekit_providers::request::{CompletionRequest, CompletionResponse, Delta};
use troupekit_providers::route::{
    DeviceState, NetworkState, RouteError, RoutePolicy, RouteTags, RoutedModel,
};
use troupekit_providers::{Model, Price};

const LOCAL: &str = "local";
const CLOUD: &str = "cloud";
const NO_ON_DEVICE: &str = "a private agent needs an on-device model: call .local(..), \
     or .route(..) with a `local` target and .local(..)";

#[derive(Clone, Default)]
pub(crate) struct RouteConfig {
    policy: Option<RoutePolicy>,
    file: Option<PathBuf>,
    pub(crate) private: bool,
    pub(crate) tier: Option<String>,
    network: Option<Arc<NetworkState>>,
    device: Option<Arc<DeviceState>>,
    pub(crate) on_device: Option<(Arc<dyn Model>, Price)>,
    pub(crate) local_only: bool,
    #[cfg(feature = "local")]
    pub(crate) local_model: Option<Arc<dyn troupekit_local::LocalModel>>,
}

fn same_arc<T: ?Sized>(a: &Option<Arc<T>>, b: &Option<Arc<T>>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => Arc::ptr_eq(a, b),
        _ => false,
    }
}

fn read_policy(owner: &str, path: &Path) -> Result<RoutePolicy, KitError> {
    let shown = path.display().to_string();
    let fail = |m: String| KitError::config(owner, format!("route file `{shown}` {m}"));
    let file = std::fs::File::open(path).map_err(|e| fail(format!("cannot be read: {e}")))?;
    let limit = u64::try_from(RoutePolicy::MAX_BYTES).expect("64 KiB fits in u64");
    let mut text = String::new();
    file.take(limit + 1)
        .read_to_string(&mut text)
        .map_err(|e| fail(format!("cannot be read as UTF-8: {e}")))?;
    if text.len() > RoutePolicy::MAX_BYTES {
        return Err(fail(format!("is over {} bytes", RoutePolicy::MAX_BYTES)));
    }
    assert!(text.len() <= RoutePolicy::MAX_BYTES, "the read is bounded");
    Ok(RoutePolicy::parse(&shown, &text)?)
}

impl RouteConfig {
    pub(crate) fn same(&self, other: &RouteConfig) -> bool {
        let same_local = match (&self.on_device, &other.on_device) {
            (None, None) => true,
            (Some((a, pa)), Some((b, pb))) => {
                (Arc::ptr_eq(a, b) || self.same_loaded(other)) && pa == pb
            }
            _ => false,
        };
        self.policy == other.policy
            && self.file == other.file
            && self.private == other.private
            && self.tier == other.tier
            && same_arc(&self.network, &other.network)
            && same_arc(&self.device, &other.device)
            && same_local
            && self.local_only == other.local_only
    }

    #[cfg(feature = "local")]
    fn same_loaded(&self, other: &RouteConfig) -> bool {
        match (&self.local_model, &other.local_model) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }

    #[cfg(not(feature = "local"))]
    fn same_loaded(&self, _other: &RouteConfig) -> bool {
        false
    }

    pub(crate) fn has_route(&self) -> bool {
        self.policy.is_some() || self.file.is_some()
    }

    pub(crate) fn check_judges(&self, owner: &str, persona: &Persona) -> Result<(), KitError> {
        if !self.private {
            return Ok(());
        }
        for eval in persona.evals.iter().filter(|e| e.name == "judge") {
            let named = eval.args.get(2).and_then(serde_json::Value::as_str);
            if let Some(model) = named.filter(|m| !m.starts_with("local/")) {
                return Err(KitError::config(
                    owner,
                    format!(
                        "a private agent cannot send answers to judge model `{model}`: \
                         name a local/.. model or leave the judge on the agent's model"
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Picks the model handle the agent runs on. `cloud` resolves the
    /// `.model(..)` selector and is called only when the result can use
    /// it, so a private or local-only agent needs no cloud key.
    /// Output: the handle, or the first configuration error.
    pub(crate) fn handle(
        &self,
        owner: &str,
        cloud: impl FnOnce() -> Result<ModelHandle, KitError>,
    ) -> Result<ModelHandle, KitError> {
        let policy = match (&self.policy, &self.file) {
            (Some(policy), _) => Some(policy.clone()),
            (None, Some(path)) => Some(read_policy(owner, path)?),
            (None, None) => None,
        };
        let Some(policy) = policy else {
            return self.unrouted(owner, cloud);
        };
        if let Some(tier) = self.tier.as_ref().filter(|t| !policy.tiers().contains(t)) {
            return Err(KitError::config(
                owner,
                format!(
                    "tier `{tier}` is served by no target of route `{}`",
                    policy.name()
                ),
            ));
        }
        let has_local = policy.target(LOCAL).is_some();
        let mut builder = RoutedModel::builder(policy.clone());
        let mut cloud = Some(cloud);
        for target in policy.targets() {
            builder = match target.id() {
                LOCAL => match &self.on_device {
                    Some((model, price)) => builder.on_device(LOCAL, model.clone(), *price),
                    None => {
                        return Err(KitError::config(
                            owner,
                            "route target `local` needs an on-device model: call .local(..)",
                        ))
                    }
                },
                CLOUD if self.local_only => {
                    return Err(KitError::config(
                        owner,
                        "route target `cloud` needs a cloud model: call .model(\"vendor/model\")",
                    ))
                }
                CLOUD if self.private => builder.remote(CLOUD, Arc::new(Unreachable), Price::FREE),
                CLOUD => {
                    let resolve = cloud.take().expect("route target ids are unique");
                    let handle = resolve()?;
                    builder.remote(CLOUD, handle.model, handle.price)
                }
                other => {
                    return Err(KitError::config(
                        owner,
                        format!(
                        "route target `{other}` is not a kit target: name them `local` and `cloud`"
                    ),
                    ))
                }
            };
        }
        if let Some(network) = &self.network {
            builder = builder.network(Arc::new(NetworkState::clone(network)));
        }
        if let Some(device) = &self.device {
            builder = builder.device(DeviceState::clone(device));
        }
        if self.private && !has_local {
            return Err(KitError::config(owner, NO_ON_DEVICE));
        }
        let mut tags = RouteTags::default().with_private(self.private);
        tags.tier = self.tier.clone();
        let view = builder.build()?.view(tags);
        let price = view.price();
        Ok(ModelHandle {
            model: Arc::new(view),
            price,
        })
    }

    fn unrouted(
        &self,
        owner: &str,
        cloud: impl FnOnce() -> Result<ModelHandle, KitError>,
    ) -> Result<ModelHandle, KitError> {
        if self.tier.is_some() {
            return Err(KitError::config(
                owner,
                "a tier needs a policy: call .route(..)",
            ));
        }
        if !self.private && !self.local_only {
            return cloud();
        }
        match &self.on_device {
            Some((model, price)) => Ok(ModelHandle {
                model: model.clone(),
                price: *price,
            }),
            None => Err(KitError::config(owner, NO_ON_DEVICE)),
        }
    }
}

/// Stand-in for route target `cloud` in a private agent's route. The
/// private view never plans a remote target, so no call reaches it;
/// binding it this way needs no cloud key. Every call returns an error.
struct Unreachable;

#[async_trait::async_trait]
impl Model for Unreachable {
    fn id(&self) -> &str {
        "route/unreachable-cloud"
    }

    async fn complete(
        &self,
        _req: &CompletionRequest,
    ) -> Result<CompletionResponse, ProviderError> {
        Err(unreachable_call())
    }

    async fn stream(
        &self,
        _req: &CompletionRequest,
        _on_delta: &mut (dyn for<'d> FnMut(&'d Delta) + Send),
    ) -> Result<CompletionResponse, ProviderError> {
        Err(unreachable_call())
    }
}

fn unreachable_call() -> ProviderError {
    ProviderError::new(
        ProviderErrorKind::Network,
        "a private agent never calls route target `cloud`",
    )
}

impl From<RouteError> for KitError {
    fn from(err: RouteError) -> Self {
        KitError::Route(err)
    }
}

impl Agent {
    /// Routes each model call by `policy`: target `cloud` is the
    /// `.model(..)` selector, target `local` the `.local(..)` model; any
    /// other target id fails `build`. Replaces `route_file`.
    pub fn route(mut self, policy: RoutePolicy) -> Self {
        self.route.policy = Some(policy);
        self.route.file = None;
        self
    }

    /// Routes by the policy in TOML file `path`, read and checked at
    /// `build` (at most 64 KiB). Same targets as `route`.
    /// Replaces `route`.
    pub fn route_file(mut self, path: impl AsRef<Path>) -> Self {
        self.route.file = Some(path.as_ref().to_path_buf());
        self.route.policy = None;
        self
    }

    /// Keeps every request on the device. Default false. `build` refuses
    /// a private agent with no on-device model, a cloud knowledge
    /// embedder, or a judge eval naming a cloud model.
    pub fn private(mut self, on: bool) -> Self {
        self.route.private = on;
        self
    }

    /// Sends requests only to route targets that list task tier `name`.
    /// Needs `.route(..)`; a tier no target serves fails `build`.
    /// Default: no tier.
    pub fn tier(mut self, name: impl Into<String>) -> Self {
        self.route.tier = Some(name.into());
        self
    }

    /// Reads the network state for routing from `state`, which the app
    /// keeps a clone of and sets. Without it a passive probe infers the
    /// state from network errors.
    pub fn network(mut self, state: NetworkState) -> Self {
        self.route.network = Some(Arc::new(state));
        self
    }

    /// Reads battery and thermal state for routing from `state`, which the
    /// app keeps a clone of and sets. Without it both are unknown and the
    /// policy's device rules are skipped.
    pub fn device(mut self, state: DeviceState) -> Self {
        self.route.device = Some(Arc::new(state));
        self
    }
}

impl Graph {
    /// Makes every agent node private, as `Agent::private(true)` does.
    /// Default false. Routes, local models and tiers stay per agent,
    /// since a graph has no model of its own.
    pub fn private(mut self, on: bool) -> Self {
        self.private = on;
        self
    }
}
