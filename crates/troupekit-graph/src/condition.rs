use crate::error::GraphError;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::sync::Arc;
use troupekit_agent::dsl::is_ident;

/// A router condition: reads the run state and the call's arguments and
/// returns the label of the edge to take, or why it cannot decide.
/// Must be deterministic, so a replayed step routes the same way.
pub type ConditionFn =
    Arc<dyn Fn(&Map<String, Value>, &[Value]) -> Result<String, String> + Send + Sync>;

/// Router conditions by name, as written in `condition = "name(args)"`.
/// `new` starts with the built-in `flag(key)`, which returns `"true"` or
/// `"false"` for a bool state key.
#[derive(Clone)]
pub struct Conditions {
    map: HashMap<String, ConditionFn>,
}

impl Default for Conditions {
    fn default() -> Self {
        Self::new()
    }
}

impl Conditions {
    /// Builds the set holding only the built-in `flag`.
    /// Register more with `register`.
    /// Returns the set.
    pub fn new() -> Self {
        let mut map: HashMap<String, ConditionFn> = HashMap::new();
        map.insert("flag".into(), Arc::new(flag));
        Self { map }
    }

    /// Adds condition `f` under `name`, which must match
    /// `[a-z_][a-z0-9_]*` and not be taken yet.
    /// Returns `Setup` otherwise.
    pub fn register(&mut self, name: &str, f: ConditionFn) -> Result<(), GraphError> {
        if !is_ident(name) {
            return Err(GraphError::Setup(format!(
                "condition name `{name}` must match [a-z_][a-z0-9_]*"
            )));
        }
        if self.map.contains_key(name) {
            return Err(GraphError::Setup(format!(
                "condition `{name}` is already registered"
            )));
        }
        self.map.insert(name.to_string(), f);
        Ok(())
    }

    /// Returns the condition registered under `name`.
    /// `None` when there is none.
    /// Graph runs check every router's condition at setup.
    pub fn get(&self, name: &str) -> Option<&ConditionFn> {
        self.map.get(name)
    }
}

fn flag(state: &Map<String, Value>, args: &[Value]) -> Result<String, String> {
    let [Value::String(key)] = args else {
        return Err("flag takes one state key".into());
    };
    match state.get(key) {
        Some(Value::Bool(b)) => Ok(b.to_string()),
        Some(_) => Err(format!("flag: `{key}` is not a bool")),
        None => Err(format!("flag: `{key}` is not set")),
    }
}
