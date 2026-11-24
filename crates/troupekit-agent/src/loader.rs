use crate::error::{AgentError, PersonaError};
use crate::persona::{valid_name, valid_relative_path, Persona};
use async_trait::async_trait;
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

/// A place persona files live: a folder, a database-backed virtual folder
/// or the built-in set. `read` takes a relative path (`<name>.toml` or an
/// `instructions_file`) and returns `None` when there is no such file.
#[async_trait]
pub trait PersonaSource: Send + Sync {
    fn label(&self) -> &str;
    async fn read(&self, path: &str) -> Result<Option<String>, AgentError>;
}

/// Persona files in a directory on disk.
/// Paths are checked with `valid_relative_path` before the join, so a
/// persona can never read outside its directory.
#[derive(Debug, Clone)]
pub struct DirSource {
    label: String,
    dir: PathBuf,
}

impl DirSource {
    /// Builds a source over `dir`, labelled with the directory path.
    /// The directory is read only on `read`, with blocking file I/O.
    /// Returns the source.
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        let dir = dir.into();
        Self {
            label: dir.display().to_string(),
            dir,
        }
    }
}

#[async_trait]
impl PersonaSource for DirSource {
    fn label(&self) -> &str {
        &self.label
    }

    async fn read(&self, path: &str) -> Result<Option<String>, AgentError> {
        if !valid_relative_path(path) {
            return Err(AgentError::Source(format!(
                "{}: unsafe path `{path}`",
                self.label
            )));
        }
        match std::fs::read_to_string(self.dir.join(path)) {
            Ok(text) => Ok(Some(text)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(AgentError::Source(format!("{}/{path}: {e}", self.label))),
        }
    }
}

/// Persona files held in memory, keyed by relative path.
/// Hosts use it for files loaded from a database; `builtin` uses it for
/// the personas shipped with this crate.
#[derive(Debug, Clone, Default)]
pub struct MemorySource {
    label: String,
    files: HashMap<String, String>,
}

impl MemorySource {
    /// Builds a source labelled `label` holding `files` as (path, text).
    /// Paths are matched exactly.
    /// Returns the source.
    pub fn new(
        label: impl Into<String>,
        files: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        Self {
            label: label.into(),
            files: files.into_iter().collect(),
        }
    }
}

#[async_trait]
impl PersonaSource for MemorySource {
    fn label(&self) -> &str {
        &self.label
    }

    async fn read(&self, path: &str) -> Result<Option<String>, AgentError> {
        Ok(self.files.get(path).cloned())
    }
}

/// The personas shipped with this crate, labelled `builtin`:
/// `assistant` (no tools) and `researcher` (`web_search`, `web_fetch`).
/// Put it last in a loader so every other source can override it.
pub fn builtin() -> MemorySource {
    MemorySource::new(
        "builtin",
        [
            (
                "assistant.toml".to_string(),
                include_str!("../personas/assistant.toml").to_string(),
            ),
            (
                "researcher.toml".to_string(),
                include_str!("../personas/researcher.toml").to_string(),
            ),
        ],
    )
}

/// Finds personas by name across sources in lookup order, for example the
/// goal's scoped folder, then the user's library, then `builtin()`.
/// The first source holding `<name>.toml` wins.
#[derive(Clone)]
pub struct PersonaLoader {
    sources: Vec<Arc<dyn PersonaSource>>,
}

impl PersonaLoader {
    /// Builds a loader that tries `sources` first to last.
    /// Put the most specific source first.
    /// Returns the loader.
    pub fn new(sources: Vec<Arc<dyn PersonaSource>>) -> Self {
        Self { sources }
    }

    /// Loads and validates persona `name` from the first source that has it,
    /// reading its `instructions_file` from the same source. Returns
    /// `NotFound` when no source has it, or the validation error.
    pub async fn load(&self, name: &str) -> Result<Persona, AgentError> {
        if !valid_name(name) {
            return Err(PersonaError::new(
                name,
                "name",
                "must be 1 to 64 of [a-z0-9-], not starting with `-`",
            )
            .into());
        }
        let path = format!("{name}.toml");
        for source in &self.sources {
            let Some(text) = source.read(&path).await? else {
                continue;
            };
            let file = format!("{}/{path}", source.label());
            let persona = Persona::parse(&file, &text)?;
            if persona.name != name {
                return Err(PersonaError::new(
                    &file,
                    "name",
                    format!("`{}` must equal the file name `{name}`", persona.name),
                )
                .into());
            }
            let Some(rel) = persona.instructions_file.clone() else {
                return Ok(persona);
            };
            let text = source.read(&rel).await?.ok_or_else(|| {
                PersonaError::new(&file, "instructions_file", format!("`{rel}` not found"))
            })?;
            return Ok(persona.with_instructions(text)?);
        }
        Err(AgentError::NotFound(name.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn persona(name: &str, role: &str) -> String {
        format!("name = '{name}'\ndisplay_name = 'P'\nrole = '{role}'\ngoal = 'g'\ninstructions = 'i'\nmodel = 'fast'\n")
    }

    fn mem(label: &str, files: &[(&str, String)]) -> Arc<dyn PersonaSource> {
        Arc::new(MemorySource::new(
            label,
            files.iter().map(|(p, t)| (p.to_string(), t.clone())),
        ))
    }

    #[tokio::test]
    async fn lookup_order_scoped_user_builtin() {
        let scoped = mem("goal", &[("coach.toml", persona("coach", "scoped"))]);
        let user = mem(
            "user",
            &[
                ("coach.toml", persona("coach", "user")),
                ("assistant.toml", persona("assistant", "mine")),
            ],
        );
        let loader = PersonaLoader::new(vec![scoped, user, Arc::new(builtin())]);
        let coach = loader.load("coach").await.unwrap();
        assert_eq!(
            (coach.role.as_str(), coach.source.as_str()),
            ("scoped", "goal/coach.toml")
        );
        assert_eq!(loader.load("assistant").await.unwrap().role, "mine");
        assert_eq!(
            loader.load("researcher").await.unwrap().source,
            "builtin/researcher.toml"
        );
        assert_eq!(
            loader.load("nobody").await.unwrap_err(),
            AgentError::NotFound("nobody".into())
        );
    }

    #[tokio::test]
    async fn rejects_unsafe_names_before_io() {
        let loader = PersonaLoader::new(vec![Arc::new(DirSource::new("/nonexistent"))]);
        for bad in ["../etc/passwd", "a/b", "", "Coach", "-x"] {
            assert!(
                matches!(loader.load(bad).await, Err(AgentError::Persona(ref e)) if e.field == "name"),
                "{bad}"
            );
        }
        let src = DirSource::new("/nonexistent");
        assert!(src.read("../x.toml").await.is_err());
        assert_eq!(src.read("x.toml").await.unwrap(), None);
    }

    #[tokio::test]
    async fn name_must_equal_file_stem() {
        let loader = PersonaLoader::new(vec![mem("lib", &[("coach.toml", persona("other", "r"))])]);
        let err = loader.load("coach").await.unwrap_err();
        assert_eq!(
            err.to_string(),
            "lib/coach.toml: name: `other` must equal the file name `coach`"
        );
    }

    #[tokio::test]
    async fn reads_instructions_file_from_same_source() {
        let text = persona("coach", "r").replace(
            "instructions = 'i'",
            "instructions_file = 'prompts/coach.md'",
        );
        let ok = PersonaLoader::new(vec![mem(
            "lib",
            &[
                ("coach.toml", text.clone()),
                ("prompts/coach.md", "Plan well.".into()),
            ],
        )]);
        assert_eq!(ok.load("coach").await.unwrap().instructions, "Plan well.");
        let missing = PersonaLoader::new(vec![mem("lib", &[("coach.toml", text)])]);
        assert!(
            matches!(missing.load("coach").await, Err(AgentError::Persona(ref e)) if e.field == "instructions_file")
        );
    }

    #[tokio::test]
    async fn dir_source_reads_files() {
        let dir = std::env::temp_dir().join(format!("tk-personas-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("coach.toml"), persona("coach", "disk")).unwrap();
        let loader = PersonaLoader::new(vec![Arc::new(DirSource::new(&dir))]);
        assert_eq!(loader.load("coach").await.unwrap().role, "disk");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[tokio::test]
    async fn builtin_personas_load() {
        let loader = PersonaLoader::new(vec![Arc::new(builtin())]);
        let assistant = loader.load("assistant").await.unwrap();
        assert!(assistant.tools.is_empty());
        let researcher = loader.load("researcher").await.unwrap();
        assert_eq!(researcher.tools, ["web_search", "web_fetch"]);
        assert!(researcher.output.is_some());
    }
}
