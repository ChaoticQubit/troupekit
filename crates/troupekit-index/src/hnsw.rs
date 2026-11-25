use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

use crate::brute::{cosine_with_norm, norm, Ranked, TopK};
use crate::quant::Slots;
use crate::rng::Rng;
use crate::{check_k, check_vector, HnswParams, IndexError, Neighbor, PageChange, PageKey, Quantization, VectorIndex};

/// A graph node on a search frontier: its rank and its slot.
/// Ordered by rank only, so ties go to the lower id and never depend
/// on slot numbering.
#[derive(Debug, Clone, Copy)]
struct Cand {
    rank: Ranked,
    slot: u32,
}

impl PartialEq for Cand {
    fn eq(&self, other: &Self) -> bool {
        self.rank == other.rank
    }
}

impl Eq for Cand {}

impl PartialOrd for Cand {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cand {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.rank.cmp(&other.rank)
    }
}

/// Approximate index after Malkov and Yashunin: a layered graph with
/// greedy descent and a beam search of width `ef` on layer 0.
/// Removal leaves a traversable tombstone; past `max_deleted_fraction`
/// the graph rebuilds from live vectors in insert order.
#[derive(Debug)]
pub struct Hnsw {
    pub(crate) dims: usize,
    pub(crate) quant: Quantization,
    pub(crate) params: HnswParams,
    pub(crate) ids: Vec<u64>,
    pub(crate) levels: Vec<u8>,
    pub(crate) deleted: Vec<bool>,
    pub(crate) links: Vec<Vec<Vec<u32>>>,
    pub(crate) data: Slots,
    pub(crate) slot_of: HashMap<u64, u32>,
    pub(crate) entry: Option<u32>,
    pub(crate) top: u8,
    pub(crate) live: usize,
    pub(crate) rng: Rng,
    pub(crate) dirty: BTreeSet<u64>,
    pub(crate) dropped: BTreeSet<u64>,
    pub(crate) header_dirty: bool,
}

impl Hnsw {
    /// Builds an empty HNSW index with validated params.
    /// Input: `dims` checked by `new_index`, params checked by
    /// `validate_params`. Output: the empty index.
    pub(crate) fn new(dims: usize, quant: Quantization, params: HnswParams) -> Self {
        assert!(dims > 0 && dims <= crate::MAX_DIMS, "dims out of range");
        assert!(validate_params(&params).is_ok(), "hnsw params not validated");
        let rng = Rng::new(params.seed());
        Self {
            dims,
            quant,
            params,
            ids: Vec::new(),
            levels: Vec::new(),
            deleted: Vec::new(),
            links: Vec::new(),
            data: Slots::new(quant),
            slot_of: HashMap::new(),
            entry: None,
            top: 0,
            live: 0,
            rng,
            dirty: BTreeSet::new(),
            dropped: BTreeSet::new(),
            header_dirty: false,
        }
    }

    pub(crate) fn cap(&self, layer: usize) -> usize {
        if layer == 0 {
            2 * self.params.m()
        } else {
            self.params.m()
        }
    }

    fn node_count(&self) -> usize {
        self.ids.len()
    }

    fn tombstones(&self) -> usize {
        self.node_count() - self.live
    }

    fn sim_to(&self, query: &[f32], qn: f64, slot: u32, buf: &mut Vec<f32>) -> f32 {
        let v = self.data.decode(slot as usize, self.dims, buf);
        cosine_with_norm(query, qn, v)
    }

    fn rank_of(&self, query: &[f32], qn: f64, slot: u32, buf: &mut Vec<f32>) -> Cand {
        let score = self.sim_to(query, qn, slot, buf);
        Cand { rank: Ranked(Neighbor { id: self.ids[slot as usize], score }), slot }
    }

    /// Beam search of width `ef` on one layer from the entry slots.
    /// Every reachable node may be traversed; only `returnable` ones
    /// enter the result. Output: best first, at most `ef` long.
    fn search_layer(&self, query: &[f32], qn: f64, entries: &[u32], ef: usize, layer: usize, returnable: &dyn Fn(u32) -> bool) -> Vec<Cand> {
        assert!(ef > 0, "beam width zero");
        let mut buf = Vec::with_capacity(self.dims);
        let mut visited: HashSet<u32> = HashSet::with_capacity(ef * 4);
        let mut frontier: BinaryHeap<Cand> = BinaryHeap::new();
        let mut best: BinaryHeap<Reverse<Cand>> = BinaryHeap::new();
        for &e in entries {
            if !visited.insert(e) {
                continue;
            }
            let c = self.rank_of(query, qn, e, &mut buf);
            frontier.push(c);
            if returnable(e) {
                best.push(Reverse(c));
                if best.len() > ef {
                    best.pop();
                }
            }
        }
        let mut pops = 0usize;
        while let Some(c) = frontier.pop() {
            pops += 1;
            assert!(pops <= self.node_count(), "beam search popped more than the node count");
            if best.len() >= ef {
                if let Some(Reverse(worst)) = best.peek() {
                    if c < *worst {
                        break;
                    }
                }
            }
            let Some(neighbors) = self.links[c.slot as usize].get(layer) else {
                continue;
            };
            for &n in neighbors {
                if !visited.insert(n) {
                    continue;
                }
                let nc = self.rank_of(query, qn, n, &mut buf);
                let full = best.len() >= ef;
                let beats_worst = best.peek().is_none_or(|Reverse(w)| nc > *w);
                if !full || beats_worst {
                    frontier.push(nc);
                    if returnable(n) {
                        best.push(Reverse(nc));
                        if best.len() > ef {
                            best.pop();
                        }
                    }
                }
            }
        }
        let mut out: Vec<Cand> = best.into_iter().map(|Reverse(c)| c).collect();
        out.sort_by(|a, b| b.cmp(a));
        assert!(out.len() <= ef, "beam result longer than ef");
        out
    }

    /// Picks at most `cap` links for a node from candidates sorted best
    /// first, by the paper's heuristic: keep a candidate only when it is
    /// closer to the base than to every link already kept.
    fn select(&self, sorted: &[Cand], cap: usize) -> Vec<u32> {
        let mut kept: Vec<u32> = Vec::with_capacity(cap);
        let mut a = Vec::with_capacity(self.dims);
        let mut b = Vec::with_capacity(self.dims);
        for c in sorted {
            if kept.len() >= cap {
                break;
            }
            let cv = self.data.decode(c.slot as usize, self.dims, &mut a).to_vec();
            let cn = norm(&cv);
            let diverse = kept.iter().all(|&k| {
                let kv = self.data.decode(k as usize, self.dims, &mut b);
                cosine_with_norm(&cv, cn, kv) < c.rank.0.score
            });
            if diverse {
                kept.push(c.slot);
            }
        }
        assert!(kept.len() <= cap, "selected more links than the cap");
        kept
    }

    fn set_links(&mut self, slot: u32, layer: usize, links: Vec<u32>) {
        assert!(links.len() <= self.cap(layer), "links exceed m or 2m");
        assert!(!links.contains(&slot), "node links to itself");
        self.links[slot as usize][layer] = links;
        self.dirty.insert(self.ids[slot as usize]);
    }

    fn add_back_link(&mut self, from: u32, to: u32, layer: usize) {
        if self.links[from as usize][layer].contains(&to) {
            return;
        }
        let cap = self.cap(layer);
        if self.links[from as usize][layer].len() < cap {
            self.links[from as usize][layer].push(to);
            assert!(self.links[from as usize][layer].len() <= cap, "links exceed m or 2m");
            self.dirty.insert(self.ids[from as usize]);
            return;
        }
        let mut buf = Vec::with_capacity(self.dims);
        let base = self.data.decode(from as usize, self.dims, &mut buf).to_vec();
        let bn = norm(&base);
        let mut scratch = Vec::with_capacity(self.dims);
        let mut cands: Vec<Cand> = self.links[from as usize][layer].iter().chain(std::iter::once(&to)).map(|&s| self.rank_of(&base, bn, s, &mut scratch)).collect();
        cands.sort_by(|a, b| b.cmp(a));
        let kept = self.select(&cands, cap);
        self.set_links(from, layer, kept);
    }

    /// Links `slot` into every layer up to its level, from the current
    /// entry point. `slot` is traversed but never chosen as its own link.
    /// Used by a fresh insert and by replace or revive of an id.
    fn link(&mut self, slot: u32, vector: &[f32]) {
        let Some(mut ep) = self.entry else {
            return;
        };
        let level = self.levels[slot as usize];
        let qn = norm(vector);
        let not_self = |s: u32| s != slot;
        let mut layer = self.top;
        while layer > level {
            let found = self.search_layer(vector, qn, &[ep], 1, layer as usize, &not_self);
            if let Some(c) = found.first() {
                ep = c.slot;
            }
            layer -= 1;
        }
        let mut entries = vec![ep];
        for layer in (0..=level.min(self.top) as usize).rev() {
            let (found, chosen) = {
                let deleted = &self.deleted;
                let live_not_self = |s: u32| s != slot && !deleted[s as usize];
                let found = self.search_layer(vector, qn, &entries, self.params.ef_construction(), layer, &live_not_self);
                let chosen = self.select(&found, self.params.m());
                (found, chosen)
            };
            self.set_links(slot, layer, chosen.clone());
            for n in chosen {
                self.add_back_link(n, slot, layer);
            }
            if !found.is_empty() {
                entries = found.iter().map(|c| c.slot).collect();
            }
        }
    }

    fn push_node(&mut self, id: u64, vector: &[f32], level: u8) -> u32 {
        let slot = u32::try_from(self.ids.len()).expect("node count checked below u32::MAX");
        self.ids.push(id);
        self.levels.push(level);
        self.deleted.push(false);
        self.links.push(vec![Vec::new(); level as usize + 1]);
        self.data.push(vector);
        self.slot_of.insert(id, slot);
        slot
    }

    fn insert_new(&mut self, id: u64, vector: &[f32]) {
        let level = self.rng.level(self.params.m());
        let slot = self.push_node(id, vector, level);
        self.live += 1;
        self.dirty.insert(id);
        self.dropped.remove(&id);
        self.link(slot, vector);
        if self.entry.is_none() || level > self.top {
            self.entry = Some(slot);
            self.top = level;
        }
        self.header_dirty = true;
    }

    fn rebuild(&mut self) {
        let mut buf = Vec::with_capacity(self.dims);
        let live: Vec<(u64, Vec<f32>)> = (0..self.node_count()).filter(|&s| !self.deleted[s]).map(|s| (self.ids[s], self.data.decode(s, self.dims, &mut buf).to_vec())).collect();
        let dropped: Vec<u64> = (0..self.node_count()).filter(|&s| self.deleted[s]).map(|s| self.ids[s]).collect();
        let mut fresh = Hnsw::new(self.dims, self.quant, self.params.clone());
        for (id, v) in &live {
            fresh.insert_new(*id, v);
        }
        fresh.dropped = std::mem::take(&mut self.dropped);
        fresh.dropped.extend(dropped);
        for id in &fresh.ids {
            fresh.dropped.remove(id);
        }
        fresh.header_dirty = true;
        assert_eq!(fresh.live, live.len(), "rebuild lost live vectors");
        *self = fresh;
    }

    fn assert_shape(&self) {
        let n = self.node_count();
        assert!(self.levels.len() == n && self.deleted.len() == n && self.links.len() == n, "node arrays out of step");
        assert_eq!(self.slot_of.len(), n, "slot map out of step with nodes");
        assert_eq!(self.data.count(self.dims), n, "vector data out of step with nodes");
        assert!(self.live <= n, "more live ids than nodes");
        assert_eq!(self.entry.is_none(), n == 0, "entry point out of step with nodes");
    }
}

/// Checks HNSW params before an index is built from them.
/// Input: the params. Output: `Ok`, or `Invalid` naming the first value
/// outside its range: `m` 2..=128, both `ef` 1..=4000, fraction (0, 1].
pub(crate) fn validate_params(p: &HnswParams) -> Result<(), IndexError> {
    if !(2..=128).contains(&p.m()) {
        return Err(IndexError::Invalid(format!("hnsw m must be in 2..=128, got {}", p.m())));
    }
    if !(1..=crate::MAX_K * 4).contains(&p.ef_construction()) {
        return Err(IndexError::Invalid(format!("hnsw ef_construction must be in 1..=4000, got {}", p.ef_construction())));
    }
    if !(1..=crate::MAX_K * 4).contains(&p.ef_search()) {
        return Err(IndexError::Invalid(format!("hnsw ef_search must be in 1..=4000, got {}", p.ef_search())));
    }
    let f = p.max_deleted_fraction();
    if !(f.is_finite() && f > 0.0 && f <= 1.0) {
        return Err(IndexError::Invalid(format!("hnsw max_deleted_fraction must be in (0, 1], got {f}")));
    }
    Ok(())
}

impl VectorIndex for Hnsw {
    fn dims(&self) -> usize {
        self.dims
    }

    fn len(&self) -> usize {
        self.live
    }

    fn heap_bytes(&self) -> usize {
        let vectors = self.node_count() * self.data.bytes_per_vector(self.dims);
        let links: usize = self.links.iter().flatten().map(|l| l.len() * std::mem::size_of::<u32>()).sum();
        vectors + links
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        check_vector(self.dims, vector)?;
        match self.slot_of.get(&id).copied() {
            Some(slot) => {
                self.data.set(slot as usize, vector);
                if self.deleted[slot as usize] {
                    self.deleted[slot as usize] = false;
                    self.live += 1;
                }
                self.dirty.insert(id);
                self.link(slot, vector);
                self.header_dirty = true;
            }
            None => {
                if self.ids.len() >= u32::MAX as usize {
                    return Err(IndexError::Invalid("hnsw node count is at u32::MAX".into()));
                }
                self.insert_new(id, vector);
            }
        }
        self.assert_shape();
        Ok(())
    }

    fn remove(&mut self, id: u64) -> bool {
        let Some(&slot) = self.slot_of.get(&id) else {
            return false;
        };
        if self.deleted[slot as usize] {
            return false;
        }
        self.deleted[slot as usize] = true;
        self.live -= 1;
        self.dirty.insert(id);
        self.header_dirty = true;
        let limit = self.params.max_deleted_fraction() as f64 * self.node_count() as f64;
        if self.tombstones() as f64 > limit {
            self.rebuild();
        }
        self.assert_shape();
        true
    }

    fn search(&self, query: &[f32], k: usize, allow: Option<&dyn Fn(u64) -> bool>) -> Result<Vec<Neighbor>, IndexError> {
        check_vector(self.dims, query)?;
        check_k(k)?;
        let Some(mut ep) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 || self.live == 0 {
            return Ok(Vec::new());
        }
        let qn = norm(query);
        let any = |_: u32| true;
        let mut layer = self.top as usize;
        while layer > 0 {
            if let Some(c) = self.search_layer(query, qn, &[ep], 1, layer, &any).first() {
                ep = c.slot;
            }
            layer -= 1;
        }
        let returnable = |s: u32| !self.deleted[s as usize] && allow.is_none_or(|f| f(self.ids[s as usize]));
        let ef = self.params.ef_search().max(k);
        let found = self.search_layer(query, qn, &[ep], ef, 0, &returnable);
        let mut top = TopK::new(k);
        for c in found {
            top.push(c.rank.0);
        }
        let hits = top.into_sorted();
        assert!(hits.len() <= k, "hnsw returned more than k");
        assert!(hits.windows(2).all(|w| Ranked(w[0]) > Ranked(w[1])), "hnsw result is not sorted by score then id");
        assert!(hits.iter().all(|n| self.slot_of.get(&n.id).is_some_and(|&s| !self.deleted[s as usize])), "hnsw returned a tombstone");
        Ok(hits)
    }

    fn take_changes(&mut self) -> Vec<PageChange> {
        crate::page::take_changes(self)
    }

    fn restore(&mut self, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError> {
        crate::page::restore(self, vectors, pages)
    }
}
