use crate::hnsw::Hnsw;
use crate::{IndexError, PageChange, PageKey};

/// Drains the change sets of an HNSW index. T1.4 replaces this with
/// the page encoder; until then no pages are emitted.
/// Input: the index. Output: an empty list.
pub(crate) fn take_changes(index: &mut Hnsw) -> Vec<PageChange> {
    index.dirty.clear();
    index.dropped.clear();
    index.header_dirty = false;
    Vec::new()
}

/// Restores an HNSW index. T1.4 replaces this with the page decoder.
/// Input: vectors and pages. Output: `Unsupported` until then.
/// The index is left unchanged.
pub(crate) fn restore(_index: &mut Hnsw, _vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, _pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError> {
    Err(IndexError::Unsupported("hnsw restore"))
}
