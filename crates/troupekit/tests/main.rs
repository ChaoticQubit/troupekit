mod builder;
mod common;
#[cfg(feature = "kit")]
mod facade;
#[cfg(feature = "knowledge")]
mod knowledge;
#[cfg(feature = "knowledge")]
mod knowledge_smoke;
#[cfg(feature = "sqlite")]
mod kill_resume;
mod lines;
#[cfg(all(unix, feature = "llama", feature = "sqlite"))]
mod kill_model_call;
#[cfg(feature = "knowledge")]
mod offline;
#[cfg(feature = "knowledge")]
mod support;
