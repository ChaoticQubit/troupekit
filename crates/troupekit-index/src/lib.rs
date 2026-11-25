//! Vector index contract for Troupekit: exact brute force and HNSW.
//! Sync and pure: no I/O, no async. A store persists the vectors and the
//! opaque pages an index emits, and hands both back on open.

mod brute;
mod quant;

use std::fmt;

pub use brute::BruteForce;

/// Largest vector dimension an index accepts.
/// Larger values are refused by `new_index` and by inserts.
/// Bounds memory per vector and the cost of one score.
pub const MAX_DIMS: usize = 65_536;

/// Largest `k` a search accepts.
/// Larger values are refused with `IndexError::Invalid`.
/// Bounds the result heap and the HNSW beam width.
pub const MAX_K: usize = 1_000;

/// One search result: an id and its cosine similarity to the query.
/// Higher scores are closer. Results come sorted by score descending,
/// ties broken by the lower id.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    pub id: u64,
    pub score: f32,
}

/// Key of one persisted page.
/// `Header` holds graph-wide state, `Node` the links of one id.
/// Brute force emits no pages.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PageKey {
    Header,
    Node(u64),
}

/// One page change since the last `take_changes` call.
/// `Put` writes or replaces the bytes of a key, `Delete` removes it.
/// The bytes are opaque to the store and untrusted on restore.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PageChange {
    Put(PageKey, Vec<u8>),
    Delete(PageKey),
}

/// In-memory format of stored vectors. Queries stay f32.
/// `F32` is exact, `F16` halves memory, `I8` quarters it with one
/// f32 scale per vector. Default `F32`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quantization {
    #[default]
    F32,
    F16,
    I8,
}

/// Build and search settings of an HNSW index.
/// Fields are private; start from `Default` and use the `with_*` setters.
/// `new_index` validates the values.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct HnswParams {
    m: usize,
    ef_construction: usize,
    ef_search: usize,
    seed: u64,
    max_deleted_fraction: f32,
}

impl Default for HnswParams {
    fn default() -> Self {
        Self { m: 16, ef_construction: 200, ef_search: 64, seed: 0x5452_4B54, max_deleted_fraction: 0.25 }
    }
}

impl HnswParams {
    /// Sets the link count per node on upper layers; layer 0 keeps `2*m`.
    /// Input: `m`, valid in `2..=128`. Output: the updated params.
    /// Checked by `new_index`.
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// Sets the beam width used while inserting.
    /// Input: `ef`, valid in `1..=MAX_K * 4`. Output: the updated params.
    /// Checked by `new_index`.
    pub fn with_ef_construction(mut self, ef: usize) -> Self {
        self.ef_construction = ef;
        self
    }

    /// Sets the beam width used while searching; raised to `k` per query.
    /// Input: `ef`, valid in `1..=MAX_K * 4`. Output: the updated params.
    /// Checked by `new_index`.
    pub fn with_ef_search(mut self, ef: usize) -> Self {
        self.ef_search = ef;
        self
    }

    /// Sets the seed of the level generator, so builds are reproducible.
    /// Input: any `u64`. Output: the updated params.
    /// A zero seed is replaced by a fixed non-zero one internally.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Sets the tombstone share of nodes that triggers a rebuild.
    /// Input: a fraction in `(0.0, 1.0]`. Output: the updated params.
    /// Checked by `new_index`.
    pub fn with_max_deleted_fraction(mut self, fraction: f32) -> Self {
        self.max_deleted_fraction = fraction;
        self
    }

    /// Returns the link count per node on upper layers.
    /// Layer 0 allows twice this many.
    /// Default 16.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Returns the beam width used while inserting.
    /// Higher builds a better graph more slowly.
    /// Default 200.
    pub fn ef_construction(&self) -> usize {
        self.ef_construction
    }

    /// Returns the beam width used while searching.
    /// A query with a larger `k` widens it to `k`.
    /// Default 64.
    pub fn ef_search(&self) -> usize {
        self.ef_search
    }

    /// Returns the seed of the level generator.
    /// The same seed and insert order give the same graph.
    /// Default `0x5452_4B54`.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Returns the tombstone share of nodes that triggers a rebuild.
    /// A rebuild re-inserts live vectors in insert order.
    /// Default 0.25.
    pub fn max_deleted_fraction(&self) -> f32 {
        self.max_deleted_fraction
    }
}

/// Which index to build. Default `BruteForce`, which is exact.
/// `Hnsw` is approximate and faster on large sets.
/// Passed to `new_index`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum IndexKind {
    #[default]
    BruteForce,
    Hnsw(HnswParams),
}

/// Failure of an index call. Caller input gets an error, never a panic.
/// `CorruptPage` means persisted pages were refused; the store then
/// rebuilds from vectors.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum IndexError {
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("corrupt page: {0}")]
    CorruptPage(String),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

/// A vector index over `u64` ids with one fixed dimension.
/// Search uses cosine similarity, higher is closer, ties by lower id,
/// at most `k` results, only ids where `allow` returns true.
pub trait VectorIndex: Send + Sync {
    /// Returns the fixed dimension of every vector and query.
    /// Set once by `new_index` and never changed.
    /// Inserts and searches of another length are refused.
    fn dims(&self) -> usize;

    /// Returns the number of live ids.
    /// Tombstones of an HNSW index are not counted.
    /// A replaced id counts once.
    fn len(&self) -> usize;

    /// Returns true when no live id is stored.
    /// Same as `len() == 0`.
    /// Tombstones alone leave an index empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the bytes held by stored vectors and graph links.
    /// Vectors count in their in-memory format: `4*dims`, `2*dims` or
    /// `dims + 4` bytes each for f32, f16 and i8.
    fn heap_bytes(&self) -> usize;

    /// Inserts a vector under `id`, replacing any vector already there.
    /// Input: an id and a vector of `dims` finite values.
    /// Errors on a wrong dimension or a non-finite component.
    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError>;

    /// Removes `id` from search results.
    /// Input: any id. Output: true when the id was live.
    /// An unknown or already removed id returns false.
    fn remove(&mut self, id: u64) -> bool;

    /// Returns at most `k` nearest live ids allowed by `allow`.
    /// Sorted by score descending, ties by lower id.
    /// Errors on a wrong dimension, a non-finite query or `k > MAX_K`.
    fn search(&self, query: &[f32], k: usize, allow: Option<&dyn Fn(u64) -> bool>) -> Result<Vec<Neighbor>, IndexError>;

    /// Drains the page changes made since the last call.
    /// The store writes them in the same transaction as the vectors.
    /// Brute force always returns an empty list.
    fn take_changes(&mut self) -> Vec<PageChange>;

    /// Loads vectors and persisted pages into an empty index.
    /// Pages are untrusted: any inconsistency is `CorruptPage` and
    /// leaves the index empty, so the store rebuilds from vectors.
    fn restore(&mut self, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError>;
}

impl fmt::Debug for dyn VectorIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("VectorIndex").field("dims", &self.dims()).field("len", &self.len()).finish()
    }
}

/// Builds an empty index of the given kind, dimension and format.
/// Input: `dims` in `1..=MAX_DIMS`. Output: the boxed index, or
/// `Invalid` for bad settings and `Unsupported` for a kind not built yet.
pub fn new_index(kind: &IndexKind, dims: usize, quant: Quantization) -> Result<Box<dyn VectorIndex>, IndexError> {
    if dims == 0 || dims > MAX_DIMS {
        return Err(IndexError::Invalid(format!("dims must be in 1..={MAX_DIMS}, got {dims}")));
    }
    match kind {
        IndexKind::BruteForce => Ok(Box::new(BruteForce::new(dims, quant))),
        IndexKind::Hnsw(_) => Err(IndexError::Unsupported("hnsw index")),
    }
}

/// Checks a vector or query against the index dimension.
/// Input: the index `dims` and the caller's slice. Output: `Ok`, or
/// `DimensionMismatch` for a wrong length and `Invalid` for NaN or inf.
pub(crate) fn check_vector(dims: usize, v: &[f32]) -> Result<(), IndexError> {
    if v.len() != dims {
        return Err(IndexError::DimensionMismatch { expected: dims, actual: v.len() });
    }
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(IndexError::Invalid(format!("component {i} is not finite")));
    }
    Ok(())
}

/// Checks a search `k` against `MAX_K`.
/// Input: the caller's `k`. Output: `Ok`, or `Invalid` above the bound.
/// A `k` of zero is valid and returns no results.
pub(crate) fn check_k(k: usize) -> Result<(), IndexError> {
    if k > MAX_K {
        return Err(IndexError::Invalid(format!("k must be at most {MAX_K}, got {k}")));
    }
    Ok(())
}
