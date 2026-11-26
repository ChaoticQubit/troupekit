//! Core types and storage traits shared by every Troupekit crate.
//! No I/O lives here: providers, tools and hosts build on these types.
//! Storage is a set of async traits with in-memory versions for tests.

pub mod clock;
pub mod error;
pub mod event;
pub mod http;
pub mod ids;
pub mod inmem;
pub mod message;
pub mod secret;
pub mod store;
pub mod tool;

use troupekit_trace as _;
