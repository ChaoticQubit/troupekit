//! On-device models for Troupekit: one sync `LocalModel` trait,
//! a checked model catalog and cache, and llama.cpp behind the `llama`
//! feature. The default build compiles no C or C++ code.

mod cache;
mod catalog;
#[cfg(feature = "embed")]
mod embed;
mod error;
mod model;
mod reply;
pub mod sha256;
mod tools;

#[cfg(feature = "llama")]
pub mod llama;

pub use cache::ModelCache;
pub use catalog::{DeviceClass, LocalCatalog, LocalSpec};
#[cfg(feature = "embed")]
pub use embed::LocalEmbedder;
pub use error::LocalError;
pub use model::{
    ChatMessage, ChatRole, Flow, GenParams, GenStop, Generation, Grammar, LocalInfo, LocalMetrics,
    LocalModel,
};
pub use reply::{parse_reply, REPLY_MAX_BYTES};
pub use tools::{tool_prompt, ToolMode};

use troupekit_providers as _;
use troupekit_trace as _;
