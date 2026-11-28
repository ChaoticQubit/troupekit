use super::Agent;
use crate::error::KitError;
use crate::laws::read_checked;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use toml::Value as Toml;
use troupekit_agent::Persona;

/// Largest persona TOML that `Agent::from_toml` and `from_toml_file`
/// accept, in bytes. Larger text or files fail at `build` with
/// `KitError::Config`. Fixed at 1 MiB.
pub const MAX_PERSONA_BYTES: usize = 1024 * 1024;

const TOML_NAME: &str = "toml";
const TEXT_SOURCE: &str = "agent:toml";

#[derive(Clone, Debug, PartialEq, Eq)]
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
    /// UTF-8 file of at most `MAX_PERSONA_BYTES`. An `instructions_file`
    /// is read from the persona's own directory, no symlinks, no escape.
    /// Otherwise as `from_toml`, where `instructions_file` is refused.
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

    /// Returns what a graph needs to know of this agent before `build`:
    /// the persona name (the TOML `name`, else the builder name) and the
    /// TOML `[output]` table of a typed persona. Fails as `build` would.
    pub(crate) fn persona_identity(&self) -> Result<(String, Option<toml::Table>), KitError> {
        let Some((source, table)) = self.toml_source()? else {
            return Ok((self.name.clone(), None));
        };
        let output = table.get("output").and_then(Toml::as_table).cloned();
        match table.get("name").and_then(Toml::as_str) {
            Some(name) if !name.trim().is_empty() => Ok((name.to_string(), output)),
            _ => Err(KitError::config(&source, "persona needs name = \"...\"")),
        }
    }

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
        if table.contains_key("instructions_file") && !table.contains_key("instructions") {
            let body = self.instructions_file(source, &table)?;
            table.remove("instructions_file");
            table.insert("instructions".into(), text(&body));
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
        assert!(persona.instructions_file.is_none());
        assert!(!persona.instructions.trim().is_empty());
        let Some(selector) = selector else {
            return Err(KitError::config(
                source,
                "needs a model: set model in the TOML or call .model(\"vendor/model\")",
            ));
        };
        Ok((persona, persona_text, selector))
    }

    fn instructions_file(&self, source: &str, table: &toml::Table) -> Result<String, KitError> {
        let Some(PersonaSource::File(persona)) = &self.persona_source else {
            return Err(KitError::config(
                source,
                "instructions_file needs from_toml_file, or call .instructions(..)",
            ));
        };
        let Some(relative) = table.get("instructions_file").and_then(Toml::as_str) else {
            return Err(KitError::config(
                source,
                "instructions_file must be a string",
            ));
        };
        let body = instructions_body(persona, relative)
            .map_err(|m| KitError::config(source, format!("instructions_file {m}")))?;
        assert!(body.len() <= MAX_PERSONA_BYTES);
        assert!(!table.contains_key("instructions"));
        Ok(body)
    }
}

/// Reads `relative`, an `instructions_file` value, from the directory of
/// the persona file at `persona`. Refuses `..`, absolute paths, symlinks,
/// and any path whose canonical form leaves that directory. Returns the
/// text of a .md or .txt file of at most `MAX_PERSONA_BYTES`.
fn instructions_body(persona: &Path, relative: &str) -> Result<String, String> {
    let relative = Path::new(relative);
    if relative.is_absolute() || relative.components().any(|c| c == Component::ParentDir) {
        return Err("must be a relative path without `..`".into());
    }
    let extension = match relative.extension().and_then(|e| e.to_str()) {
        Some("md") => "md",
        Some("txt") => "txt",
        _ => return Err("must be a .md or .txt file".into()),
    };
    let persona = persona
        .canonicalize()
        .map_err(|e| format!("cannot be resolved: {e}"))?;
    let directory = persona
        .parent()
        .expect("a canonical file path has a parent");
    let joined = directory.join(relative);
    match std::fs::symlink_metadata(&joined) {
        Ok(meta) if meta.file_type().is_symlink() => return Err("must not be a symlink".into()),
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => return Err("does not exist".into()),
        Err(e) => return Err(format!("cannot be read: {e}")),
    }
    let canonical = joined
        .canonicalize()
        .map_err(|e| format!("cannot be resolved: {e}"))?;
    if !canonical.starts_with(directory) {
        return Err("must stay under the persona file's directory".into());
    }
    let (read, body) = read_checked(&canonical, extension, MAX_PERSONA_BYTES as u64)?;
    assert_eq!(read, canonical);
    assert!(body.len() <= MAX_PERSONA_BYTES);
    Ok(body)
}
