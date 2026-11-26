//! Memory and retrieval for Troupekit agents, built on the troupekit-core stores.
//! Covers run history, scoped knowledge, profile facts, chunking, embeddings,
//! hybrid retrieval and a bounded context block to prepend to a prompt.

pub mod block;
pub mod chunk;
pub mod embed;
pub mod error;
pub mod facts;
pub mod hybrid;
pub mod ingest;
pub mod load;
pub mod memory;
#[cfg(feature = "onnx")]
pub mod onnx;
pub mod retrieve;
pub mod summary;
pub mod text;
pub mod tools;

pub use error::ContextError;

use troupekit_trace as _;
