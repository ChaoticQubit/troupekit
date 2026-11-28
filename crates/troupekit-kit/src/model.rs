use crate::error::KitError;
use std::sync::{Arc, OnceLock};
use troupekit_agent::ModelHandle;
use troupekit_core::secret::Secret;
use troupekit_providers::wire::AuthStyle;
use troupekit_providers::{Catalog, CatalogError, Model, Price, Transport};

const BUILTIN_CATALOG: &str = include_str!("../../../catalog/providers.toml");

pub(crate) fn builtin_catalog() -> &'static Catalog {
    static CATALOG: OnceLock<Catalog> = OnceLock::new();
    CATALOG.get_or_init(|| match Catalog::load(BUILTIN_CATALOG) {
        Ok(catalog) => catalog,
        Err(e) => panic!("the built-in catalog is valid: {e}"),
    })
}

pub(crate) struct ModelChoice<'a> {
    pub agent: &'a str,
    pub selector: &'a str,
    pub catalog: &'a Catalog,
    pub client: Option<Arc<dyn Model>>,
    pub price: Option<Price>,
    pub api_key: Option<Secret>,
    pub api_key_env: Option<&'a str>,
    pub transport: Option<Arc<dyn Transport>>,
}

pub(crate) fn resolve(choice: ModelChoice<'_>) -> Result<ModelHandle, KitError> {
    let agent = choice.agent;
    let selector = choice.selector;
    if let Some(model) = choice.client {
        let price = choice
            .price
            .or_else(|| choice.catalog.price(selector).ok())
            .unwrap_or(Price::FREE);
        return Ok(ModelHandle { model, price });
    }
    if !selector.contains('/') {
        return Err(KitError::config(
            agent,
            format!("model alias `{selector}` needs .model_client(..)"),
        ));
    }
    let catalog_price = choice
        .catalog
        .price(selector)
        .map_err(|source| catalog_error(agent, choice.catalog, source))?;
    let price = choice.price.unwrap_or(catalog_price);
    let vendor_id = selector.split_once('/').map_or(selector, |(v, _)| v);
    let vendor = choice.catalog.vendor(vendor_id);
    assert!(vendor.is_some(), "a priced selector names a catalog vendor");
    let key = match vendor.map(|v| v.auth) {
        Some(AuthStyle::NoKey) => None,
        _ => Some(key_for(vendor_id, choice.api_key, choice.api_key_env)?),
    };
    let transport = match choice.transport {
        Some(t) => t,
        None => default_transport(agent)?,
    };
    let model = choice
        .catalog
        .connect_with(selector, key, transport)
        .map_err(|source| catalog_error(agent, choice.catalog, source))?;
    Ok(ModelHandle {
        model: Arc::from(model),
        price,
    })
}

#[cfg(feature = "knowledge")]
pub(crate) fn default_embedder(
    choice: ModelChoice<'_>,
) -> Result<Arc<dyn troupekit_context::embed::Embedder>, KitError> {
    let agent = choice.agent;
    let none = |why: String| {
        KitError::config(
            agent,
            format!("knowledge needs an embedder: {why}; call Knowledge::embedder(..)"),
        )
    };
    let Some((vendor_id, _)) = choice.selector.split_once('/') else {
        return Err(none(format!(
            "model alias `{}` names no vendor",
            choice.selector
        )));
    };
    let Some(spec) = choice
        .catalog
        .embedding_specs()
        .iter()
        .find(|e| e.vendor == vendor_id)
    else {
        return Err(none(format!(
            "vendor `{vendor_id}` has no embedding model in the catalog"
        )));
    };
    let vendor = choice.catalog.vendor(vendor_id);
    assert!(
        vendor.is_some(),
        "a catalog embedding row names a catalog vendor"
    );
    let key = match vendor.map(|v| v.auth) {
        Some(AuthStyle::NoKey) => None,
        _ => Some(key_for(vendor_id, choice.api_key, choice.api_key_env)?),
    };
    let transport = match choice.transport {
        Some(t) => t,
        None => default_transport(agent)?,
    };
    let selector = format!("{}/{}", spec.vendor, spec.id);
    let embedder = choice
        .catalog
        .embedder_with(&selector, key, transport)
        .map_err(|source| catalog_error(agent, choice.catalog, source))?;
    Ok(Arc::new(embedder))
}

pub(crate) fn trace_secret(
    selector: &str,
    key: Option<Secret>,
    env: Option<&str>,
) -> Option<Secret> {
    match selector.split_once('/') {
        Some((vendor, _)) => key_for(vendor, key, env).ok(),
        None => key,
    }
}

fn key_for(vendor: &str, key: Option<Secret>, env: Option<&str>) -> Result<Secret, KitError> {
    if let Some(key) = key {
        return Ok(key);
    }
    let env = env.map_or_else(|| default_key_env(vendor), str::to_string);
    let value = std::env::var(&env).ok().and_then(|v| Secret::new(v).ok());
    value.ok_or_else(|| KitError::MissingKey {
        vendor: vendor.to_string(),
        env,
    })
}

fn default_key_env(vendor: &str) -> String {
    format!("{}_API_KEY", vendor.to_ascii_uppercase().replace('-', "_"))
}

#[cfg(feature = "reqwest")]
pub(crate) fn default_transport(agent: &str) -> Result<Arc<dyn Transport>, KitError> {
    let transport = troupekit_providers::reqwest_transport::ReqwestTransport::new()
        .map_err(|e| KitError::config(agent, format!("HTTP client: {}", e.message)))?;
    Ok(Arc::new(transport))
}

#[cfg(not(feature = "reqwest"))]
pub(crate) fn default_transport(agent: &str) -> Result<Arc<dyn Transport>, KitError> {
    Err(KitError::config(
        agent,
        "no HTTP transport: enable the `reqwest` feature or call .transport(..)",
    ))
}

fn catalog_error(agent: &str, catalog: &Catalog, source: CatalogError) -> KitError {
    let hint = match &source {
        CatalogError::UnknownModel(selector) => {
            let vendor = selector
                .split_once('/')
                .map_or(selector.as_str(), |(v, _)| v);
            let models = catalog
                .vendor(vendor)
                .map(|v| {
                    v.models
                        .iter()
                        .map(|m| m.id.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .unwrap_or_default();
            format!("known models for {vendor}: {models}")
        }
        CatalogError::UnknownVendor(_) => {
            let vendors: Vec<&str> = catalog.vendors().iter().map(|v| v.id.as_str()).collect();
            format!("known vendors: {}", vendors.join(", "))
        }
        _ => "write the model as vendor/model, for example openai/gpt-6-luna".to_string(),
    };
    KitError::Catalog {
        agent: agent.to_string(),
        source,
        hint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_catalog_loads() {
        let catalog = builtin_catalog();
        assert!(catalog.vendor("openai").is_some());
        assert!(catalog.price("openai/gpt-6-luna").is_ok());
    }

    #[test]
    fn default_key_env_upper_cases_and_maps_dashes() {
        assert_eq!(default_key_env("openai"), "OPENAI_API_KEY");
        assert_eq!(default_key_env("lm-studio"), "LM_STUDIO_API_KEY");
    }
}
