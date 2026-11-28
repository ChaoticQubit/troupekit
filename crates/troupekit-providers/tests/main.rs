mod anthropic;
mod catalog;
mod common;
mod connect;
#[cfg(feature = "embeddings")]
mod embed;
mod fold;
mod gemini;
mod http_model;
mod middleware;
mod openai;
mod request;
mod route;
mod reqwest_transport;
mod sse;
mod status;
mod transport;
