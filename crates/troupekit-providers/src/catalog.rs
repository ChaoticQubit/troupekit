use crate::middleware::MeteredModel;
use crate::model::{valid_model_id, Endpoint, HttpModel, Model};
use crate::request::CompletionResponse;
use crate::transport::Transport;
use crate::wire::{AuthStyle, Protocol, Quirks};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use troupekit_core::event::Usage;
use troupekit_core::secret::Secret;
use troupekit_core::tool::WEB_SEARCH;

/// Why a catalog could not be loaded or a model could not be connected.
/// Parse and Invalid come from `load`; the rest from selecting and
/// connecting a `vendor/model`.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CatalogError {
    #[error("catalog is not valid TOML: {0}")]
    Parse(String),
    #[error("invalid catalog: {0}")]
    Invalid(String),
    #[error("model selector must look like vendor/model: {0}")]
    BadSelector(String),
    #[error("unknown vendor: {0}")]
    UnknownVendor(String),
    #[error("unknown model: {0}")]
    UnknownModel(String),
    #[error("vendor {0} needs an API key")]
    MissingKey(String),
    #[error("transport: {0}")]
    Transport(String),
}

/// Higher rates a model charges for a whole call once its input passes
/// `above_input_tokens`, in millionths of a dollar per million tokens.
/// Written inline as `long_context = { ... }` on a catalog model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LongContextPrice {
    pub above_input_tokens: u64,
    pub input_usd_micros_per_mtok: u64,
    pub output_usd_micros_per_mtok: u64,
}

/// A model's prices in millionths of a US dollar per million tokens.
/// `verified` is false when nobody checked them against the vendor's
/// price page: budgets must not read an unverified zero as free.
/// `search_usd_micros` is the vendor's fee per hosted web search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Price {
    pub input_usd_micros_per_mtok: u64,
    pub output_usd_micros_per_mtok: u64,
    pub verified: bool,
    pub long_context: Option<LongContextPrice>,
    #[serde(default)]
    pub search_usd_micros: u64,
}

impl Price {
    /// A verified zero price, for local models that cost nothing to call.
    /// Open-model vendors such as Ollama use it for unlisted models.
    /// Every usage costs 0 under it.
    pub const FREE: Price = Price {
        input_usd_micros_per_mtok: 0,
        output_usd_micros_per_mtok: 0,
        verified: true,
        long_context: None,
        search_usd_micros: 0,
    };

    /// Cost of one finished `reply`: its tokens plus the fee of every
    /// hosted web search it ran. A failed search is counted too, so the
    /// figure never under-counts. Saturates at `u64::MAX`.
    pub fn reply_cost_usd_micros(&self, reply: &CompletionResponse) -> u64 {
        let searches = reply.message.hosted_calls(WEB_SEARCH);
        self.cost_usd_micros(&reply.usage)
            .saturating_add(searches.saturating_mul(self.search_usd_micros))
    }

    /// Cost of `usage` in millionths of a dollar, rounded up so budgets
    /// never under-count. Long-context rates apply to the whole call once
    /// input passes their threshold; the result saturates at `u64::MAX`.
    pub fn cost_usd_micros(&self, usage: &Usage) -> u64 {
        let (input, output) = match self.long_context {
            Some(lc) if usage.input_tokens > lc.above_input_tokens => {
                (lc.input_usd_micros_per_mtok, lc.output_usd_micros_per_mtok)
            }
            _ => (
                self.input_usd_micros_per_mtok,
                self.output_usd_micros_per_mtok,
            ),
        };
        let total = (u128::from(usage.input_tokens) * u128::from(input))
            .saturating_add(u128::from(usage.output_tokens) * u128::from(output));
        u64::try_from(total.div_ceil(1_000_000)).unwrap_or(u64::MAX)
    }
}

/// One model a vendor serves: its wire id, its context window in tokens
/// and its prices. A `[[vendor.model]]` row in the catalog.
/// `price` turns the price fields into a `Price`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub id: String,
    pub context_window: u64,
    pub input_usd_micros_per_mtok: u64,
    pub output_usd_micros_per_mtok: u64,
    pub price_verified: bool,
    #[serde(default)]
    pub long_context: Option<LongContextPrice>,
}

impl ModelSpec {
    /// Collects this model's price fields into one value.
    /// Returns the price, verified or not as the catalog says.
    /// Used by the cost meter and budget checks.
    pub fn price(&self) -> Price {
        Price {
            input_usd_micros_per_mtok: self.input_usd_micros_per_mtok,
            output_usd_micros_per_mtok: self.output_usd_micros_per_mtok,
            verified: self.price_verified,
            long_context: self.long_context,
            search_usd_micros: 0,
        }
    }
}

/// One vendor in the catalog: how to reach it and which models it serves.
/// `open_models` vendors (local servers) also accept model ids the
/// catalog does not list, at a verified zero price. `price_source` is the
/// vendor's price page that verified paid prices were checked against; `search_usd_micros` its fee per hosted web search.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VendorSpec {
    pub id: String,
    pub name: String,
    pub protocol: Protocol,
    pub base_url: String,
    pub auth: AuthStyle,
    #[serde(default)]
    pub price_source: Option<String>,
    #[serde(default)]
    pub open_models: bool,
    #[serde(default)]
    pub quirks: Quirks,
    #[serde(default)]
    pub search_usd_micros: u64,
    #[serde(default, rename = "model")]
    pub models: Vec<ModelSpec>,
}

impl VendorSpec {
    /// Builds the endpoint that reaches this vendor.
    /// Returns its protocol, base URL, auth style and quirks.
    /// Pure; `HttpModel` sends through it.
    pub fn endpoint(&self) -> Endpoint {
        Endpoint {
            protocol: self.protocol,
            base_url: self.base_url.clone(),
            auth: self.auth,
            quirks: self.quirks,
        }
    }
}

/// Default inputs per embeddings request when a row names none.
/// Below every supported vendor's own limit.
/// Setting S29.
const DEFAULT_EMBEDDING_MAX_BATCH: usize = 64;

/// Largest embedding dimension or batch a catalog row may name.
/// Matches the index bound on vector dimensions.
/// Rows above it are `Invalid`.
const MAX_EMBEDDING_DIMS: usize = 65_536;

const MAX_EMBEDDING_BATCH: usize = 2_048;

fn default_embedding_max_batch() -> usize {
    DEFAULT_EMBEDDING_MAX_BATCH
}

/// One embedding model a vendor serves: a top-level `[[embedding]]` row.
/// `dims` is the vector length the endpoint returns, `max_batch` the
/// inputs per request, the price is per million input tokens.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EmbeddingSpec {
    pub vendor: String,
    pub id: String,
    pub dims: usize,
    #[serde(default = "default_embedding_max_batch")]
    pub max_batch: usize,
    pub input_usd_micros_per_mtok: u64,
    pub price_verified: bool,
}

/// The provider catalog: vendors as data, loaded from TOML `[[vendor]]`
/// tables with `[[vendor.model]]` rows and top-level `[[embedding]]` rows.
/// `connect` turns a `vendor/model` choice into a ready `Model`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    #[serde(default, rename = "vendor")]
    vendors: Vec<VendorSpec>,
    #[serde(default, rename = "embedding", skip_serializing_if = "Vec::is_empty")]
    embeddings: Vec<EmbeddingSpec>,
}

impl Catalog {
    /// Parses and checks a catalog from TOML `text`.
    /// Returns `Parse` for bad TOML or unknown fields, `UnknownVendor` for
    /// an embedding row naming no vendor, and `Invalid` for a bad id, URL,
    /// bound, repeated entry, or verified price without a price source.
    pub fn load(text: &str) -> Result<Self, CatalogError> {
        let catalog: Catalog =
            toml::from_str(text).map_err(|e| CatalogError::Parse(e.to_string()))?;
        catalog.validate()?;
        Ok(catalog)
    }

    /// Lists every vendor in catalog order, for a provider dropdown.
    /// Returns the vendors as a slice.
    /// Each carries its models and prices.
    pub fn vendors(&self) -> &[VendorSpec] {
        &self.vendors
    }

    /// Lists every embedding model in catalog order.
    /// Returns the rows as a slice; empty for a catalog without any.
    /// Each names its vendor, dimension, batch size and price.
    pub fn embedding_specs(&self) -> &[EmbeddingSpec] {
        &self.embeddings
    }

    /// Looks up the vendor with `id`.
    /// Returns it, or `None` when the catalog has no such vendor.
    /// Ids match exactly.
    pub fn vendor(&self, id: &str) -> Option<&VendorSpec> {
        self.vendors.iter().find(|v| v.id == id)
    }

    /// Prices `selector`, written `vendor/model`, for budgets and the
    /// cost meter. Returns `BadSelector`, `UnknownVendor` or
    /// `UnknownModel` when the selector names nothing in the catalog.
    pub fn price(&self, selector: &str) -> Result<Price, CatalogError> {
        self.resolve(selector).map(|(_, _, price)| price)
    }

    /// Connects `selector` (`vendor/model`) through `transport`, using
    /// `key` unless the vendor takes none. Returns a metered `Model`, or
    /// a selector error, or `MissingKey` when a needed key is absent.
    pub fn connect_with(
        &self,
        selector: &str,
        key: Option<Secret>,
        transport: Arc<dyn Transport>,
    ) -> Result<Box<dyn Model>, CatalogError> {
        let (vendor, model, price) = self.resolve(selector)?;
        let key = match vendor.auth {
            AuthStyle::NoKey => None,
            _ => Some(key.ok_or_else(|| CatalogError::MissingKey(vendor.id.clone()))?),
        };
        let http = HttpModel::new(selector, model, vendor.endpoint(), key, transport)
            .map_err(|e| CatalogError::BadSelector(e.message))?;
        Ok(Box::new(MeteredModel::new(Box::new(http), price)))
    }

    /// Connects `selector` like `connect_with`, over a new default
    /// `ReqwestTransport`. Returns `Transport` when the HTTP client
    /// cannot start, else what `connect_with` returns.
    #[cfg(feature = "reqwest")]
    pub fn connect(
        &self,
        selector: &str,
        key: Option<Secret>,
    ) -> Result<Box<dyn Model>, CatalogError> {
        let transport = crate::reqwest_transport::ReqwestTransport::new()
            .map_err(|e| CatalogError::Transport(e.message))?;
        self.connect_with(selector, key, Arc::new(transport))
    }

    /// Builds an embedder for `selector` (`vendor/model`, an embedding
    /// row) through `transport`, using `key` unless the vendor takes none.
    /// Returns a selector error, or `MissingKey` when a needed key is absent.
    #[cfg(feature = "embeddings")]
    pub fn embedder_with(
        &self,
        selector: &str,
        key: Option<Secret>,
        transport: Arc<dyn Transport>,
    ) -> Result<crate::embed::HttpEmbedder, CatalogError> {
        let (vendor_id, model_id) = selector
            .split_once('/')
            .filter(|(v, m)| !v.is_empty() && !m.is_empty())
            .ok_or_else(|| CatalogError::BadSelector(selector.to_owned()))?;
        let vendor = self
            .vendor(vendor_id)
            .ok_or_else(|| CatalogError::UnknownVendor(vendor_id.to_owned()))?;
        if vendor.protocol == Protocol::Anthropic {
            return Err(CatalogError::BadSelector(format!(
                "{selector}: vendor has no embeddings endpoint"
            )));
        }
        let spec = self
            .embeddings
            .iter()
            .find(|e| e.vendor == vendor_id && e.id == model_id)
            .ok_or_else(|| CatalogError::UnknownModel(selector.to_owned()))?;
        let key = match vendor.auth {
            AuthStyle::NoKey => None,
            _ => Some(key.ok_or_else(|| CatalogError::MissingKey(vendor.id.clone()))?),
        };
        Ok(crate::embed::HttpEmbedder::new(
            selector.to_owned(),
            spec.id.clone(),
            vendor.endpoint(),
            key,
            transport,
            spec.dims,
            spec.max_batch,
        ))
    }

    /// Builds an embedder like `embedder_with`, over a new default
    /// `ReqwestTransport`. Returns `Transport` when the HTTP client
    /// cannot start, else what `embedder_with` returns.
    #[cfg(all(feature = "embeddings", feature = "reqwest"))]
    pub fn embedder(
        &self,
        selector: &str,
        key: Option<Secret>,
    ) -> Result<crate::embed::HttpEmbedder, CatalogError> {
        let transport = crate::reqwest_transport::ReqwestTransport::new()
            .map_err(|e| CatalogError::Transport(e.message))?;
        self.embedder_with(selector, key, Arc::new(transport))
    }

    pub(crate) fn resolve(
        &self,
        selector: &str,
    ) -> Result<(&VendorSpec, String, Price), CatalogError> {
        let (vendor_id, model_id) = selector
            .split_once('/')
            .filter(|(v, m)| !v.is_empty() && !m.is_empty())
            .ok_or_else(|| CatalogError::BadSelector(selector.to_owned()))?;
        let vendor = self
            .vendor(vendor_id)
            .ok_or_else(|| CatalogError::UnknownVendor(vendor_id.to_owned()))?;
        if let Some(model) = vendor.models.iter().find(|m| m.id == model_id) {
            let price = Price {
                search_usd_micros: vendor.search_usd_micros,
                ..model.price()
            };
            return Ok((vendor, model.id.clone(), price));
        }
        match (vendor.open_models, valid_model_id(model_id)) {
            (true, true) => Ok((vendor, model_id.to_owned(), Price::FREE)),
            (true, false) => Err(CatalogError::BadSelector(selector.to_owned())),
            (false, _) => Err(CatalogError::UnknownModel(selector.to_owned())),
        }
    }

    fn validate(&self) -> Result<(), CatalogError> {
        let mut vendor_ids = HashSet::new();
        for v in &self.vendors {
            if v.id.is_empty()
                || !v
                    .id
                    .bytes()
                    .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-')
            {
                return invalid(format!(
                    "vendor id {:?} must be lowercase letters, digits and dashes",
                    v.id
                ));
            }
            if !vendor_ids.insert(v.id.as_str()) {
                return invalid(format!("vendor {} is listed twice", v.id));
            }
            if v.name.trim().is_empty() {
                return invalid(format!("vendor {} has a blank name", v.id));
            }
            let plain = v.base_url.starts_with("http://");
            if !plain && !v.base_url.starts_with("https://") {
                return invalid(format!("vendor {} base_url must start with https://", v.id));
            }
            if plain && v.auth != AuthStyle::NoKey {
                return invalid(format!(
                    "vendor {} would send its key over plain http",
                    v.id
                ));
            }
            if v.price_source
                .as_deref()
                .is_some_and(|s| !s.starts_with("https://"))
            {
                return invalid(format!(
                    "vendor {} price_source must start with https://",
                    v.id
                ));
            }
            let paid =
                |m: &ModelSpec| m.input_usd_micros_per_mtok > 0 || m.output_usd_micros_per_mtok > 0;
            if v.price_source.is_none() && v.models.iter().any(|m| m.price_verified && paid(m)) {
                return invalid(format!(
                    "vendor {} has verified prices but no price_source",
                    v.id
                ));
            }
            let mut model_ids = HashSet::new();
            for m in &v.models {
                if !valid_model_id(&m.id) {
                    return invalid(format!(
                        "vendor {} model id {:?} is not allowed",
                        v.id, m.id
                    ));
                }
                if !model_ids.insert(m.id.as_str()) {
                    return invalid(format!("vendor {} lists model {} twice", v.id, m.id));
                }
                if m.long_context.is_some_and(|lc| lc.above_input_tokens == 0) {
                    return invalid(format!(
                        "vendor {} model {} has a zero long-context threshold",
                        v.id, m.id
                    ));
                }
            }
        }
        self.validate_embeddings()
    }

    fn validate_embeddings(&self) -> Result<(), CatalogError> {
        let mut seen = HashSet::new();
        for e in &self.embeddings {
            let vendor = self
                .vendor(&e.vendor)
                .ok_or_else(|| CatalogError::UnknownVendor(e.vendor.clone()))?;
            if vendor.protocol == Protocol::Anthropic {
                return invalid(format!(
                    "embedding {}/{}: vendor has no embeddings endpoint",
                    e.vendor, e.id
                ));
            }
            if !valid_model_id(&e.id) {
                return invalid(format!("embedding id {:?} is not allowed", e.id));
            }
            if !seen.insert((e.vendor.as_str(), e.id.as_str())) {
                return invalid(format!("embedding {}/{} is listed twice", e.vendor, e.id));
            }
            if e.dims == 0 || e.dims > MAX_EMBEDDING_DIMS {
                return invalid(format!(
                    "embedding {}/{} dims must be in 1..={MAX_EMBEDDING_DIMS}",
                    e.vendor, e.id
                ));
            }
            if e.max_batch == 0 || e.max_batch > MAX_EMBEDDING_BATCH {
                return invalid(format!(
                    "embedding {}/{} max_batch must be in 1..={MAX_EMBEDDING_BATCH}",
                    e.vendor, e.id
                ));
            }
            if e.price_verified && e.input_usd_micros_per_mtok > 0 && vendor.price_source.is_none() {
                return invalid(format!(
                    "embedding {}/{} has a verified price but its vendor has no price_source",
                    e.vendor, e.id
                ));
            }
        }
        Ok(())
    }
}

fn invalid(message: String) -> Result<(), CatalogError> {
    Err(CatalogError::Invalid(message))
}
