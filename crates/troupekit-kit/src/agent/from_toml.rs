use super::Agent;
use crate::error::KitError;
use crate::laws::read_checked;
use std::path::{Path, PathBuf};
use toml::Value as Toml;
use troupekit_agent::Persona;

/// Largest persona TOML that `Agent::from_toml` and `from_toml_file`
/// accept, in bytes. Larger text or files fail at `build` with
/// `KitError::Config`. Fixed at 1 MiB.
pub const MAX_PERSONA_BYTES: usize = 1024 * 1024;

const TOML_NAME: &str = "toml";
const TEXT_SOURCE: &str = "agent:toml";

#[derive(Clone, Debug)]
pub(crate) enum PersonaSource {
    Text(String),
    File(PathBuf),
}

impl Agent {
    /// Starts a builder from persona TOML `text`, parsed at `build`.
    /// Builder settings shared with the persona (model, instructions,
    /// display name, role, goal, evals, turn and token caps) beat the TOML.
    /// Tools the TOML names must be added with `.tool(..)`.
    pub fn from_toml(text: impl Into<String>) -> Agent {
        let mut agent = Agent::new(TOML_NAME);
        agent.persona_source = Some(PersonaSource::Text(text.into()));
        agent
    }

    /// Starts a builder from the persona file at `path`, read at `build`.
    /// The path must have no `..`, end in `.toml` and name a regular
    /// UTF-8 file of at most `MAX_PERSONA_BYTES`. Otherwise as `from_toml`.
    pub fn from_toml_file(path: impl AsRef<Path>) -> Agent {
        let mut agent = Agent::new(TOML_NAME);
        agent.persona_source = Some(PersonaSource::File(path.as_ref().to_path_buf()));
        agent
    }

    pub(crate) fn toml_source(&self) -> Result<Option<(String, toml::Table)>, KitError> {
        let Some(source) = &self.persona_source else {
            return Ok(None);
        };
        let (name, text) = match source {
            PersonaSource::Text(text) => {
                if text.len() > MAX_PERSONA_BYTES {
                    return Err(KitError::config(TEXT_SOURCE, "persona TOML is over 1 MiB"));
                }
                (TEXT_SOURCE.to_string(), text.clone())
            }
            PersonaSource::File(path) => {
                let shown = path.display().to_string();
                let (_, text) = read_checked(path, "toml", MAX_PERSONA_BYTES as u64)
                    .map_err(|m| KitError::config(&shown, format!("persona file {m}")))?;
                (shown, text)
            }
        };
        assert!(text.len() <= MAX_PERSONA_BYTES);
        match text.parse::<toml::Table>() {
            Ok(table) => Ok(Some((name, table))),
            Err(_) => Err(Persona::parse(&name, &text)
                .expect_err("text that is not a TOML table is not a persona")
                .into()),
        }
    }

    #[cfg(feature = "knowledge")]
    pub(crate) fn toml_selector(&self) -> Option<String> {
        let (_, table) = self.toml_source().ok()??;
        table.get("model")?.as_str().map(str::to_string)
    }

    pub(crate) fn toml_persona(
        &self,
        source: &str,
        mut table: toml::Table,
        model_key: Option<&str>,
        tools: &[String],
    ) -> Result<(Persona, String, String), KitError> {
        let text = |v: &str| Toml::String(v.to_string());
        for (key, value) in [
            ("model", &self.model),
            ("display_name", &self.display_name),
            ("role", &self.role),
            ("goal", &self.goal),
        ] {
            if let Some(value) = value {
                table.insert(key.into(), text(value));
            }
        }
        if let Some(instructions) = &self.instructions {
            table.remove("instructions_file");
            table.insert("instructions".into(), text(instructions));
        }
        if !self.evals.is_empty() {
            let evals = self.evals.iter().map(|e| text(e)).collect();
            table.insert("evals".into(), Toml::Array(evals));
        }
        if let Some(turns) = self.max_turns {
            table.insert("max_turns".into(), Toml::Integer(i64::from(turns)));
        }
        if let Some(tokens) = self.max_tokens {
            table.insert("max_tokens".into(), Toml::Integer(i64::from(tokens)));
        }
        if let Some(Toml::Array(named)) = table.get_mut("tools") {
            for tool in tools {
                if !named.iter().any(|n| n.as_str() == Some(tool)) {
                    named.push(text(tool));
                }
            }
        } else if !table.contains_key("tools") && !tools.is_empty() {
            let named = tools.iter().map(|t| text(t)).collect();
            table.insert("tools".into(), Toml::Array(named));
        }
        let selector = table
            .get("model")
            .and_then(Toml::as_str)
            .map(str::to_string);
        if let Some(key) = model_key {
            table.insert("model".into(), text(key));
        }
        let persona_text = toml::to_string(&table)
            .map_err(|e| KitError::config(source, format!("persona does not serialize: {e}")))?;
        let persona = Persona::parse(source, &persona_text)?;
        let Some(selector) = selector else {
            return Err(KitError::config(
                source,
                "needs a model: set model in the TOML or call .model(\"vendor/model\")",
            ));
        };
        Ok((persona, persona_text, selector))
    }
}
