use crate::error::CoreError;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const REDACTED: &str = "[redacted]";

/// An API key or token that never leaks through formatting or serialization.
/// Debug and Serialize both print "[redacted]" and there is no Display;
/// only `expose` returns the value, at the point of use.
#[derive(Clone, PartialEq, Eq)]
pub struct Secret(String);

impl Secret {
    /// Wraps a secret `value`.
    /// Returns `CoreError::InvalidMessage` when the value is empty or
    /// whitespace only, otherwise the wrapped secret.
    pub fn new(value: String) -> Result<Self, CoreError> {
        if value.trim().is_empty() {
            return Err(CoreError::InvalidMessage("secret is blank".into()));
        }
        Ok(Self(value))
    }

    /// Returns the raw value as a string slice.
    /// Call it only where the value is sent to its provider,
    /// never to log, store or display it.
    pub fn expose(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Debug for Secret {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Secret({REDACTED})")
    }
}

impl Serialize for Secret {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(REDACTED)
    }
}

impl<'de> Deserialize<'de> for Secret {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        Secret::new(raw).map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn never_prints_or_serializes_the_value() {
        let s = Secret::new("sk-live-123".into()).unwrap();
        assert_eq!(format!("{s:?}"), "Secret([redacted])");
        assert_eq!(serde_json::to_string(&s).unwrap(), "\"[redacted]\"");
        assert_eq!(s.expose(), "sk-live-123");
    }

    #[test]
    fn rejects_empty_and_whitespace() {
        assert!(Secret::new(String::new()).is_err());
        assert!(Secret::new("  ".into()).is_err());
    }

    #[test]
    fn deserializes_from_a_plain_string() {
        let s: Secret = serde_json::from_str("\"abc\"").unwrap();
        assert_eq!(s.expose(), "abc");
    }
}
