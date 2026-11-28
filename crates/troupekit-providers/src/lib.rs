//! Model providers for Troupekit: four wire protocols, a provider
//! catalog, injected transports and middleware behind one `Model` trait.
//! Start with `Catalog::load` and `Catalog::connect("vendor/model", key)`.

pub mod catalog;
#[cfg(feature = "embeddings")]
pub mod embed;
pub mod fold;
pub mod middleware;
pub mod model;
pub mod request;
pub mod route;
#[cfg(feature = "reqwest")]
pub mod reqwest_transport;
pub mod sse;
pub mod status;
pub mod transport;
pub mod wire;

pub use catalog::{Catalog, CatalogError, EmbeddingSpec, Price};
#[cfg(feature = "embeddings")]
pub use embed::HttpEmbedder;
pub use middleware::{FallbackModel, MeteredModel, RetryModel, RetryPolicy};
pub use model::{Endpoint, HttpModel, Model};
pub use request::{CompletionRequest, CompletionResponse, Delta, StopReason};
pub use transport::Transport;

use troupekit_trace as _;
