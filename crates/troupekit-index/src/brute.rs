use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};

use crate::quant::Slots;
use crate::{check_k, check_vector, IndexError, Neighbor, PageChange, PageKey, Quantization, VectorIndex};

/// A search candidate ordered so that greater means closer.
/// Higher score wins; on equal scores the lower id wins.
/// Used by every index to keep a bounded best-k heap.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Ranked(pub(crate) Neighbor);

impl PartialEq for Ranked {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ranked {}

impl PartialOrd for Ranked {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ranked {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.score.total_cmp(&other.0.score).then_with(|| other.0.id.cmp(&self.0.id))
    }
}

/// Keeps the best `k` candidates pushed into it.
/// Input: candidates in any order. Output: `into_sorted` gives them
/// by score descending, ties by lower id, at most `k` long.
pub(crate) struct TopK {
    k: usize,
    heap: BinaryHeap<Reverse<Ranked>>,
}

impl TopK {
    pub(crate) fn new(k: usize) -> Self {
        Self { k, heap: BinaryHeap::with_capacity(k.saturating_add(1)) }
    }

    pub(crate) fn push(&mut self, n: Neighbor) {
        if self.k == 0 {
            return;
        }
        let r = Ranked(n);
        if self.heap.len() < self.k {
            self.heap.push(Reverse(r));
        } else if let Some(Reverse(worst)) = self.heap.peek() {
            if r > *worst {
                self.heap.pop();
                self.heap.push(Reverse(r));
            }
        }
        assert!(self.heap.len() <= self.k, "top-k heap exceeds k");
    }

    pub(crate) fn into_sorted(self) -> Vec<Neighbor> {
        self.heap.into_sorted_vec().into_iter().map(|Reverse(r)| r.0).collect()
    }
}

/// Asserts the search law: `want` entries, sorted by score descending
/// and ties by lower id. Input: the result and the expected length.
/// Panics when an index breaks the contract, which is a bug, not input.
pub(crate) fn assert_result_law(hits: &[Neighbor], want: usize) {
    assert_eq!(hits.len(), want, "search returned {} hits, law requires {want}", hits.len());
    assert!(hits.windows(2).all(|w| Ranked(w[0]) > Ranked(w[1])), "search result is not sorted by score then id");
}

/// Cosine of `query` with norm `query_norm` against `v`, in one pass.
/// Sums run in f64 so large finite components cannot overflow; only the
/// final score is cast to f32. Output: -1.0..=1.0, 0.0 for a zero vector.
pub(crate) fn cosine_with_norm(query: &[f32], query_norm: f64, v: &[f32]) -> f32 {
    assert_eq!(query.len(), v.len(), "cosine of unequal lengths");
    let mut dot = 0.0f64;
    let mut nv = 0.0f64;
    for (&x, &y) in query.iter().zip(v) {
        dot += f64::from(x) * f64::from(y);
        nv += f64::from(y) * f64::from(y);
    }
    let nv = nv.sqrt();
    if query_norm == 0.0 || nv == 0.0 {
        return 0.0;
    }
    let score = (dot / (query_norm * nv)) as f32;
    assert!(score.is_finite(), "cosine of finite vectors is finite in f64");
    score
}

/// Euclidean norm of `v`, summed in f64 in index order.
/// Input: any finite slice. Output: a non-negative finite value.
/// Matches the arithmetic of `cosine_with_norm`.
pub(crate) fn norm(v: &[f32]) -> f64 {
    v.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt()
}

/// Exact index: scores every allowed vector on each search.
/// Ids and vectors sit in parallel slots with an id-to-slot map;
/// removal swaps the last slot in. Emits no pages.
#[derive(Debug)]
pub struct BruteForce {
    dims: usize,
    quant: Quantization,
    ids: Vec<u64>,
    data: Slots,
    slot_of: HashMap<u64, usize>,
}

impl BruteForce {
    /// Builds an empty exact index of dimension `dims` and format `quant`.
    /// Input: `dims` in `1..=MAX_DIMS`, already checked by `new_index`,
    /// the only public way in. Output: the empty index.
    pub(crate) fn new(dims: usize, quant: Quantization) -> Self {
        assert!(dims > 0 && dims <= crate::MAX_DIMS, "dims out of range");
        Self { dims, quant, ids: Vec::new(), data: Slots::new(quant), slot_of: HashMap::new() }
    }

    fn check_shape(&self) {
        assert_eq!(self.ids.len(), self.slot_of.len(), "slot map out of step with ids");
        assert_eq!(self.data.count(self.dims), self.ids.len(), "vector data out of step with ids");
    }
}

impl VectorIndex for BruteForce {
    fn dims(&self) -> usize {
        self.dims
    }

    fn len(&self) -> usize {
        self.ids.len()
    }

    fn heap_bytes(&self) -> usize {
        self.ids.len() * self.data.bytes_per_vector(self.dims)
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        check_vector(self.dims, vector)?;
        match self.slot_of.get(&id) {
            Some(&i) => self.data.set(i, vector),
            None => {
                self.slot_of.insert(id, self.ids.len());
                self.ids.push(id);
                self.data.push(vector);
            }
        }
        self.check_shape();
        Ok(())
    }

    fn remove(&mut self, id: u64) -> bool {
        let Some(i) = self.slot_of.remove(&id) else {
            return false;
        };
        let last = self.ids.len() - 1;
        if i != last {
            let moved = self.ids[last];
            self.ids[i] = moved;
            self.data.move_slot(last, i, self.dims);
            self.slot_of.insert(moved, i);
        }
        self.ids.pop();
        self.data.truncate(last, self.dims);
        self.check_shape();
        true
    }

    fn search(&self, query: &[f32], k: usize, allow: Option<&dyn Fn(u64) -> bool>) -> Result<Vec<Neighbor>, IndexError> {
        check_vector(self.dims, query)?;
        check_k(k)?;
        let qn = norm(query);
        let mut top = TopK::new(k);
        let mut allowed = 0usize;
        let mut buf = Vec::with_capacity(self.dims);
        for (i, &id) in self.ids.iter().enumerate() {
            if allow.is_some_and(|f| !f(id)) {
                continue;
            }
            allowed += 1;
            let v = self.data.decode(i, self.dims, &mut buf);
            top.push(Neighbor { id, score: cosine_with_norm(query, qn, v) });
        }
        let hits = top.into_sorted();
        assert_result_law(&hits, k.min(allowed));
        Ok(hits)
    }

    fn take_changes(&mut self) -> Vec<PageChange> {
        Vec::new()
    }

    fn restore(&mut self, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError> {
        if !self.ids.is_empty() {
            return Err(IndexError::Invalid("restore needs an empty index".into()));
        }
        if pages.next().is_some() {
            return Err(IndexError::CorruptPage("brute force index has no pages".into()));
        }
        for (id, v) in vectors {
            if let Err(e) = self.insert(id, &v) {
                *self = Self::new(self.dims, self.quant);
                return Err(e);
            }
        }
        Ok(())
    }
}
