use crate::agent::Agent;
use crate::error::KitError;
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use troupekit_local::{LocalCatalog, LocalChat, LocalModel, ModelCache};
use troupekit_providers::{Price, Transport};

/// Where an on-device model comes from: a row id of the built-in local
/// catalog (`&str`), a GGUF file (`PathBuf`), or a loaded model
/// (`Arc<dyn LocalModel>`). Converts with `From`; read at `build`.
#[non_exhaustive]
#[derive(Clone)]
pub enum LocalChoice {
    Catalog(String),
    File(PathBuf),
    Model(Arc<dyn LocalModel>),
}

impl fmt::Debug for LocalChoice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LocalChoice::Catalog(id) => f.debug_tuple("Catalog").field(id).finish(),
            LocalChoice::File(path) => f.debug_tuple("File").field(path).finish(),
            LocalChoice::Model(_) => f.write_str("Model(..)"),
        }
    }
}

impl From<&str> for LocalChoice {
    fn from(id: &str) -> Self {
        LocalChoice::Catalog(id.to_string())
    }
}

impl From<String> for LocalChoice {
    fn from(id: String) -> Self {
        LocalChoice::Catalog(id)
    }
}

impl From<PathBuf> for LocalChoice {
    fn from(path: PathBuf) -> Self {
        LocalChoice::File(path)
    }
}

impl From<&Path> for LocalChoice {
    fn from(path: &Path) -> Self {
        LocalChoice::File(path.to_path_buf())
    }
}

impl From<Arc<dyn LocalModel>> for LocalChoice {
    fn from(model: Arc<dyn LocalModel>) -> Self {
        LocalChoice::Model(model)
    }
}

impl LocalChoice {
    fn key(&self) -> Option<String> {
        match self {
            LocalChoice::Catalog(id) => Some(format!("catalog:{id}")),
            LocalChoice::File(path) => Some(format!("file:{}", path.display())),
            LocalChoice::Model(_) => None,
        }
    }

    fn chat_id(&self) -> String {
        match self {
            LocalChoice::Catalog(id) => format!("local/{id}"),
            LocalChoice::File(path) => {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("model");
                format!("local/{stem}")
            }
            LocalChoice::Model(_) => "local/model".to_string(),
        }
    }
}

/// Models loaded during one `build`, by catalog id or file path, so two
/// graph nodes naming the same model share one load.
pub(crate) type Loads = BTreeMap<String, Arc<dyn LocalModel>>;

/// Settings `load` reads from the agent: whether `build` may download a
/// catalog model, and the transport to download it with.
pub(crate) struct Fetch<'a> {
    pub owner: &'a str,
    pub download: bool,
    pub transport: Option<Arc<dyn Transport>>,
}

/// Loads `choice` for agent `fetch.owner`, reusing a model in `loads`;
/// a file or catalog row loads with embeddings on when `embeddings` is
/// set. A catalog row is downloaded only when `fetch.download` is set.
/// Output: the model, or `KitError`.
pub(crate) async fn load(
    choice: &LocalChoice,
    fetch: &Fetch<'_>,
    loads: &mut Loads,
    embeddings: bool,
) -> Result<Arc<dyn LocalModel>, KitError> {
    let key = match (choice, choice.key()) {
        (LocalChoice::Model(model), _) => return Ok(model.clone()),
        (_, key) => key.expect("file and catalog choices have a key"),
    };
    let key = if embeddings {
        format!("{key}:embed")
    } else {
        key
    };
    if let Some(model) = loads.get(&key) {
        return Ok(model.clone());
    }
    let path = match choice {
        LocalChoice::File(path) => path.clone(),
        LocalChoice::Catalog(id) => cached(id, fetch).await?,
        LocalChoice::Model(_) => unreachable!("a loaded model returns above"),
    };
    let model = load_file(fetch.owner, &path, embeddings)?;
    loads.insert(key, model.clone());
    assert!(!loads.is_empty());
    Ok(model)
}

async fn cached(id: &str, fetch: &Fetch<'_>) -> Result<PathBuf, KitError> {
    let Some(spec) = LocalCatalog::builtin().get(id) else {
        return Err(KitError::config(
            fetch.owner,
            format!("unknown local model `{id}`: pick an id from LocalCatalog::builtin()"),
        ));
    };
    let cache = ModelCache::default_dir()?;
    if let Some(path) = cache.path(spec) {
        return Ok(path);
    }
    if !fetch.download {
        return Err(KitError::config(
            fetch.owner,
            format!(
                "local model `{id}` is not in {}: call .download(true) to let build \
                 fetch it with ModelCache::fetch, or pass a file to .local(..)",
                cache.dir().display()
            ),
        ));
    }
    let transport = match &fetch.transport {
        Some(t) => t.clone(),
        None => crate::model::default_transport(fetch.owner)?,
    };
    let path = cache.fetch(spec, transport.as_ref()).await?;
    assert!(
        cache.path(spec).is_some(),
        "a fetched model is in the cache"
    );
    Ok(path)
}

#[cfg(feature = "llama")]
fn load_file(owner: &str, path: &Path, embeddings: bool) -> Result<Arc<dyn LocalModel>, KitError> {
    use troupekit_local::llama::{LlamaModel, LlamaSettings};
    assert!(!owner.is_empty());
    let settings = LlamaSettings::default().with_embeddings(embeddings);
    let model = LlamaModel::load(path, settings)?;
    assert!(
        !embeddings || model.info().embedding_dims() > 0,
        "an embedding load advertises its dimensions"
    );
    Ok(Arc::new(model))
}

#[cfg(not(feature = "llama"))]
fn load_file(owner: &str, path: &Path, _embeddings: bool) -> Result<Arc<dyn LocalModel>, KitError> {
    Err(KitError::config(
        owner,
        format!(
            "loading `{}` needs the `llama` feature; or pass a loaded model to .local(..)",
            path.display()
        ),
    ))
}

impl Agent {
    /// Sets the on-device model: a local catalog id, a GGUF file path or
    /// a loaded `Arc<dyn LocalModel>`. Without `.model(..)` the agent is
    /// local only. Default: none. Needs feature `llama` to load a file.
    pub fn local(mut self, choice: impl Into<LocalChoice>) -> Self {
        self.local = Some(choice.into());
        self
    }

    /// Lets `build` download a catalog model that is not in the model
    /// cache yet. Default false: a missing model fails `build` instead.
    /// Has no effect on a file or a loaded model.
    pub fn download(mut self, on: bool) -> Self {
        self.download = on;
        self
    }

    pub(crate) fn fetch(&self) -> Fetch<'_> {
        Fetch {
            owner: self.name(),
            download: self.download,
            transport: self.transport_handle(),
        }
    }

    /// Loads the on-device model named by `.local(..)` or a
    /// `local/<id>` selector into the route's on-device slot; without a
    /// cloud model the agent becomes local only. Output: the agent.
    pub(crate) async fn load_local(mut self, loads: &mut Loads) -> Result<Self, KitError> {
        if self.route.local_model.is_some() {
            return Ok(self);
        }
        let from_selector = self
            .explicit_selector()
            .and_then(|s| s.strip_prefix("local/"))
            .filter(|_| !self.has_client())
            .map(str::to_string);
        let choice = match (from_selector.clone(), self.local.clone()) {
            (Some(_), Some(_)) => {
                return Err(KitError::config(
                    self.name(),
                    "name the on-device model once: .model(\"local/<id>\") or .local(..)",
                ))
            }
            (Some(id), None) => LocalChoice::Catalog(id),
            (None, Some(choice)) => choice,
            (None, None) => return Ok(self),
        };
        let local_only =
            !self.has_client() && (self.explicit_selector().is_none() || from_selector.is_some());
        let model = load(&choice, &self.fetch(), loads, false).await?;
        if self.explicit_selector().is_none() {
            self.set_selector(choice.chat_id());
        }
        let chat = LocalChat::new(model.clone(), choice.chat_id());
        self.route.on_device = Some((Arc::new(chat), Price::FREE));
        self.route.local_model = Some(model);
        self.route.local_only = local_only;
        assert!(
            self.explicit_selector().is_some(),
            "a loaded agent has a selector"
        );
        Ok(self)
    }

    /// Loads `choice` and wraps it as an on-device knowledge embedder.
    /// Output: the embedder, or a `Config` error when the model cannot
    /// embed.
    #[cfg(feature = "knowledge")]
    pub(crate) async fn local_embedder(
        &self,
        choice: &LocalChoice,
    ) -> Result<Arc<dyn troupekit_context::embed::Embedder>, KitError> {
        let model = load(choice, &self.fetch(), &mut Loads::new(), true).await?;
        embedder_of(self.name(), &model)
    }
}

/// Wraps `model` as an on-device embedder for agent `owner`.
/// Output: the embedder, or a `Config` error naming
/// `Knowledge::local_embedder(..)` when the model cannot embed.
#[cfg(feature = "knowledge")]
pub(crate) fn embedder_of(
    owner: &str,
    model: &Arc<dyn LocalModel>,
) -> Result<Arc<dyn troupekit_context::embed::Embedder>, KitError> {
    match troupekit_local::LocalEmbedder::new(model.clone()) {
        Ok(embedder) => Ok(Arc::new(embedder)),
        Err(troupekit_local::LocalError::Unsupported(_)) => Err(KitError::config(
            owner,
            "the local model cannot embed: call Knowledge::dir(..).local_embedder(..) \
             with a model that can, or .on_device_embedder(..)",
        )),
        Err(other) => Err(other.into()),
    }
}
