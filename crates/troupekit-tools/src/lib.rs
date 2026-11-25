//! Tools for Troupekit agents: one `Tool` trait, a registry that checks
//! arguments against each tool's JSON Schema, typed tools from serde
//! structs, an SSRF-guarded HTTP client, web fetch, web search and
//! declarative HTTP tools. Third-party text is always fenced as untrusted.

pub mod declarative;
pub mod fetch;
pub mod http;
pub mod registry;
pub mod search;
pub mod secrets;
pub mod text;
pub mod tool;
pub mod typed;

pub use declarative::http_toolkit_from_toml;
pub use fetch::web_fetch;
pub use http::{
    is_public_ip, receive, HttpError, HttpErrorKind, HttpLimits, HttpRequest, HttpResponse, Method,
    Received, Transport,
};
#[cfg(feature = "reqwest")]
pub use http::GuardedHttp;
pub use registry::{compile_schema, schema_errors, valid_tool_name, ToolRegistry, Toolkit};
pub use search::{BraveSearch, SearchHit, SearchProvider, SearchToolkit, SearxngSearch};
pub use secrets::{EnvSecrets, SecretSource};
pub use text::{
    cap_chars, fence_capped, fence_untrusted, html_to_text, response_text, MAX_TEXT_CHARS,
};
pub use tool::{hosted, Tool, ToolContext, ToolError, ToolOutput};
pub use troupekit_core::tool::HostedTool;
pub use troupekit_core::tool::ToolSpec;
pub use typed::{typed_tool, TypedTool};
