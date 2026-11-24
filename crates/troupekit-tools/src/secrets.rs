use std::collections::HashMap;
use troupekit_core::secret::Secret;

/// Looks up secrets by name when a toolkit is built.
/// Hosts back it with their own key store; `EnvSecrets` reads the process
/// environment. Returns `None` when the name is unknown or blank.
pub trait SecretSource: Send + Sync {
    /// Returns the secret stored under `name`, if any.
    fn get(&self, name: &str) -> Option<Secret>;
}

/// Reads secrets from environment variables of the same name.
/// A missing, non-UTF-8 or blank variable counts as absent.
#[derive(Debug, Clone, Copy, Default)]
pub struct EnvSecrets;

impl SecretSource for EnvSecrets {
    fn get(&self, name: &str) -> Option<Secret> {
        std::env::var(name).ok().and_then(|v| Secret::new(v).ok())
    }
}

impl SecretSource for HashMap<String, Secret> {
    fn get(&self, name: &str) -> Option<Secret> {
        HashMap::get(self, name).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_source_returns_named_secret() {
        let mut m = HashMap::new();
        m.insert("KEY".to_string(), Secret::new("v-1".into()).unwrap());
        let src: &dyn SecretSource = &m;
        assert_eq!(src.get("KEY").unwrap().expose(), "v-1");
        assert!(src.get("OTHER").is_none());
    }

    #[test]
    fn env_source_treats_unset_as_absent() {
        assert!(EnvSecrets.get("TROUPEKIT_TEST_SURELY_UNSET_VAR").is_none());
    }
}
