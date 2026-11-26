use crate::error::LocalError;
use std::collections::HashSet;

const ID_MAX_BYTES: usize = 64;
const URL_MAX_BYTES: usize = 2_048;
const ROWS_MAX: usize = 256;

/// The device a catalog row targets: `Phone`, `Pi` (single-board
/// computers) or `Desktop`. `pick` filters by it.
/// Default `Desktop`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum DeviceClass {
    Phone,
    Pi,
    #[default]
    Desktop,
}

/// One downloadable model: id, https URL, SHA-256, size in bytes,
/// resident RAM estimate, context, device class, licence, embedding
/// support and the recommended flag. Checked by `new`; `with_*` setters.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalSpec {
    id: String,
    url: String,
    sha256: String,
    bytes: u64,
    ram_bytes: u64,
    context: usize,
    class: DeviceClass,
    licence: String,
    embeddings: bool,
    recommended: bool,
}

impl LocalSpec {
    /// Checks and builds a row. Rules: id of lowercase letters, digits,
    /// `-`, `_`, `.` (1 to 64 bytes, no leading dot); `https://` URL; 64
    /// lowercase hex chars; size above 0. Output: the row, or `Catalog`.
    pub fn new(id: &str, url: &str, sha256: &str, bytes: u64) -> Result<Self, LocalError> {
        let id_ok = !id.is_empty()
            && id.len() <= ID_MAX_BYTES
            && !id.starts_with('.')
            && id.bytes().all(|b| {
                b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.')
            });
        if !id_ok {
            return Err(LocalError::Catalog(format!(
                "model id {id:?} is not a valid id"
            )));
        }
        let url_ok = url.len() <= URL_MAX_BYTES
            && url.len() > "https://".len()
            && url.starts_with("https://")
            && !url.bytes().any(|b| b.is_ascii_control() || b == b' ');
        if !url_ok {
            return Err(LocalError::Catalog(format!(
                "model {id}: url must be an https URL"
            )));
        }
        let hash_ok = sha256.len() == 64
            && sha256
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
        if !hash_ok {
            return Err(LocalError::Catalog(format!(
                "model {id}: sha256 must be 64 lowercase hex characters"
            )));
        }
        if bytes == 0 {
            return Err(LocalError::Catalog(format!(
                "model {id}: bytes must be above 0"
            )));
        }
        Ok(Self {
            id: id.to_string(),
            url: url.to_string(),
            sha256: sha256.to_string(),
            bytes,
            ram_bytes: bytes,
            context: 4_096,
            class: DeviceClass::default(),
            licence: String::new(),
            embeddings: false,
            recommended: false,
        })
    }

    /// Sets the resident memory estimate at the default context.
    /// `pick` compares it with the app's RAM budget.
    /// Default the file size.
    pub fn with_ram_bytes(mut self, ram_bytes: u64) -> Self {
        self.ram_bytes = ram_bytes;
        self
    }

    /// Sets the context length the row is meant to run with.
    /// Informational; `LlamaSettings` sets the real one.
    /// Default 4 096.
    pub fn with_context(mut self, context: usize) -> Self {
        self.context = context;
        self
    }

    /// Sets the device class.
    /// `pick` only returns rows of the asked class.
    /// Default `Desktop`.
    pub fn with_class(mut self, class: DeviceClass) -> Self {
        self.class = class;
        self
    }

    /// Sets the licence name, such as `apache-2.0`.
    /// The owner confirms licences before a release.
    /// Default empty.
    pub fn with_licence(mut self, licence: impl Into<String>) -> Self {
        self.licence = licence.into();
        self
    }

    /// Marks the model as able to embed.
    /// Load it with `LlamaSettings::with_embeddings(true)` to use it.
    /// Default false.
    pub fn with_embeddings(mut self, embeddings: bool) -> Self {
        self.embeddings = embeddings;
        self
    }

    /// Marks the row as the recommended one for its class.
    /// `LocalCatalog::new` allows at most one per class.
    /// Default false.
    pub fn with_recommended(mut self, recommended: bool) -> Self {
        self.recommended = recommended;
        self
    }

    /// Returns the model id, also its cache file name stem.
    /// Input: none.
    /// Output: the id.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Returns the https download URL.
    /// Input: none.
    /// Output: the URL.
    pub fn url(&self) -> &str {
        &self.url
    }

    /// Returns the expected SHA-256 in lowercase hex.
    /// Input: none.
    /// Output: 64 hex characters.
    pub fn sha256(&self) -> &str {
        &self.sha256
    }

    /// Returns the exact file size in bytes.
    /// Input: none.
    /// Output: above 0.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Returns the resident memory estimate in bytes.
    /// Input: none.
    /// Default the file size.
    pub fn ram_bytes(&self) -> u64 {
        self.ram_bytes
    }

    /// Returns the intended context length.
    /// Input: none.
    /// Default 4 096.
    pub fn context(&self) -> usize {
        self.context
    }

    /// Returns the device class.
    /// Input: none.
    /// Default `Desktop`.
    pub fn class(&self) -> DeviceClass {
        self.class
    }

    /// Returns the licence name.
    /// Input: none.
    /// Default empty.
    pub fn licence(&self) -> &str {
        &self.licence
    }

    /// Returns whether the model can embed.
    /// Input: none.
    /// Default false.
    pub fn embeddings(&self) -> bool {
        self.embeddings
    }

    /// Returns whether the row is recommended for its class.
    /// Input: none.
    /// Default false.
    pub fn recommended(&self) -> bool {
        self.recommended
    }
}

/// A checked list of downloadable models: unique ids, at most one
/// recommended row per class, at most 256 rows. `get` finds a row and
/// `pick` chooses one for a device class and RAM budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LocalCatalog {
    rows: Vec<LocalSpec>,
}

impl LocalCatalog {
    /// Builds a catalog from rows.
    /// Input: rows from `LocalSpec::new`. Output: the catalog, or
    /// `Catalog` on a duplicate id, two recommended rows or over 256 rows.
    pub fn new(rows: Vec<LocalSpec>) -> Result<Self, LocalError> {
        if rows.len() > ROWS_MAX {
            return Err(LocalError::Catalog(format!(
                "{} rows, limit {ROWS_MAX}",
                rows.len()
            )));
        }
        let mut ids = HashSet::new();
        let mut recommended = HashSet::new();
        for row in &rows {
            if !ids.insert(row.id.as_str()) {
                return Err(LocalError::Catalog(format!(
                    "model id {} appears twice",
                    row.id
                )));
            }
            if row.recommended && !recommended.insert(row.class) {
                return Err(LocalError::Catalog(format!(
                    "two recommended rows for class {:?}",
                    row.class
                )));
            }
        }
        Ok(Self { rows })
    }

    /// Finds a row by id.
    /// Input: the id.
    /// Output: the row, or `None`.
    pub fn get(&self, id: &str) -> Option<&LocalSpec> {
        self.rows.iter().find(|r| r.id == id)
    }

    /// Returns every row in file order.
    /// Input: none.
    /// Output: the rows.
    pub fn rows(&self) -> &[LocalSpec] {
        &self.rows
    }

    /// Chooses a model for a device: the recommended row of `class` if
    /// its RAM estimate fits `ram_budget_bytes`, else the fitting row with
    /// the largest estimate. Output: the row, or `None` when none fits.
    pub fn pick(&self, class: DeviceClass, ram_budget_bytes: u64) -> Option<&LocalSpec> {
        let fitting: Vec<&LocalSpec> = self
            .rows
            .iter()
            .filter(|r| r.class == class && r.ram_bytes <= ram_budget_bytes)
            .collect();
        fitting
            .iter()
            .find(|r| r.recommended)
            .or_else(|| fitting.iter().max_by_key(|r| (r.ram_bytes, r.bytes)))
            .copied()
    }
}
