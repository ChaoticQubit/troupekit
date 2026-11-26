//! On-device models for Troupekit: one sync `LocalModel` trait,
//! a checked model catalog and cache, and llama.cpp behind the `llama`
//! feature. The default build compiles no C or C++ code.

mod cache;
mod catalog;
mod chat;
#[cfg(feature = "embed")]
mod embed;
mod error;
mod gbnf;
mod model;
mod reply;
pub mod sha256;
mod tools;

#[cfg(feature = "llama")]
pub mod llama;

pub use cache::ModelCache;
pub use catalog::{DeviceClass, LocalCatalog, LocalSpec};
pub use chat::{LocalChat, LocalChatSettings};
#[cfg(feature = "embed")]
pub use embed::LocalEmbedder;
pub use error::LocalError;
pub use gbnf::schema_to_gbnf;
pub use model::{
    ChatMessage, ChatRole, Flow, GenParams, GenStop, Generation, Grammar, LocalInfo, LocalMetrics,
    LocalModel,
};
pub use reply::{parse_reply, REPLY_MAX_BYTES};
pub use tools::{output_grammar, tool_grammar, tool_prompt, ToolMode};
