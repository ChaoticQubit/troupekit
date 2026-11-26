pub use crate::agent::Agent;
pub use crate::output::IntoToolOutput;
pub use crate::tool::{spec, AsTool};
pub use async_trait::async_trait;
pub use schemars;
pub use serde;
pub use serde_json;
pub use troupekit_tools::{Tool, ToolContext, ToolError, ToolOutput, ToolSpec};

/// Returns the catalog vendor `agent`'s model selector names, for the
/// neutral layer's host transport allow-list. Not public API.
/// Output: `None` for an alias or an unlisted vendor.
pub fn vendor_of(agent: &Agent) -> Option<troupekit_providers::catalog::VendorSpec> {
    agent.vendor_spec()
}

/// Returns `agent`'s model selector, from `.model(..)` or a TOML
/// persona's `model`, for neutral-layer error messages. Not public API.
/// Output: `None` when no model is set.
pub fn selector_of(agent: &Agent) -> Option<String> {
    agent.selector()
}
