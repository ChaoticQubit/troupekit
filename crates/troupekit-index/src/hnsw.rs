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
/// Removal unlinks the node at once and repairs its neighbours, so no
/// removed vector stays in memory or in a page; past
/// `max_deleted_fraction` removals the graph rebuilds in id order.
#[derive(Debug)]
pub struct Hnsw {
    pub(crate) dims: usize,
    pub(crate) quant: Quantization,
    pub(crate) params: HnswParams,
    pub(crate) ids: Vec<u64>,
    pub(crate) levels: Vec<u8>,
    pub(crate) links: Vec<Vec<Vec<u32>>>,
    pub(crate) inbound: Vec<Vec<Vec<u32>>>,
    pub(crate) data: Slots,
    pub(crate) slot_of: HashMap<u64, u32>,
    pub(crate) entry: Option<u32>,
    pub(crate) top: u8,
    pub(crate) removed_since_build: usize,
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
            links: Vec::new(),
            inbound: Vec::new(),
            data: Slots::new(quant),
            slot_of: HashMap::new(),
            entry: None,
            top: 0,
            removed_since_build: 0,
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

    /// Picks at most `cap` links for a node by the paper's heuristic: a
    /// candidate is kept when it is closer to the base than to every kept
    /// link, then pruned ones fill the free slots, so equal vectors stay linked.
    fn select(&self, candidates: &[Cand], cap: usize) -> Vec<u32> {
        let mut sorted: Vec<Cand> = candidates.to_vec();
        sorted.sort_by(|x, y| y.rank.0.score.total_cmp(&x.rank.0.score).then_with(|| y.rank.0.id.cmp(&x.rank.0.id)));
        let mut kept: Vec<u32> = Vec::with_capacity(cap);
        let mut pruned: Vec<u32> = Vec::new();
        let mut a = Vec::with_capacity(self.dims);
        let mut b = Vec::with_capacity(self.dims);
        for c in &sorted {
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
            } else {
                pruned.push(c.slot);
            }
        }
        for p in pruned {
            if kept.len() >= cap {
                break;
            }
            kept.push(p);
        }
        assert!(kept.len() <= cap, "selected more links than the cap");
        kept
    }

    /// Replaces the links of `slot` on one layer and keeps the inbound
    /// lists in step: dropped targets forget `slot`, new targets learn it.
    /// Every link write goes through here or `add_back_link`.
    fn set_links(&mut self, slot: u32, layer: usize, links: Vec<u32>) {
        assert!(links.len() <= self.cap(layer), "links exceed m or 2m");
        assert!(!links.contains(&slot), "node links to itself");
        let old = std::mem::replace(&mut self.links[slot as usize][layer], links);
        for &o in &old {
            if !self.links[slot as usize][layer].contains(&o) {
                let back = &mut self.inbound[o as usize][layer];
                let at = back.iter().position(|&x| x == slot).expect("inbound mirrors links");
                back.swap_remove(at);
            }
        }
        for i in 0..self.links[slot as usize][layer].len() {
            let n = self.links[slot as usize][layer][i];
            if !old.contains(&n) {
                self.inbound[n as usize][layer].push(slot);
            }
        }
        self.dirty.insert(self.ids[slot as usize]);
    }

    fn add_back_link(&mut self, from: u32, to: u32, layer: usize) {
        if self.links[from as usize][layer].contains(&to) {
            return;
        }
        let cap = self.cap(layer);
        if self.links[from as usize][layer].len() < cap {
            self.links[from as usize][layer].push(to);
            self.inbound[to as usize][layer].push(from);
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
            let found = self.search_layer(vector, qn, &entries, self.params.ef_construction(), layer, &not_self);
            let chosen = self.select(&found, self.params.m());
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
        self.links.push(vec![Vec::new(); level as usize + 1]);
        self.inbound.push(vec![Vec::new(); level as usize + 1]);
        self.data.push(vector);
        self.slot_of.insert(id, slot);
        slot
    }

    fn insert_new(&mut self, id: u64, vector: &[f32]) {
        let level = self.rng.level(self.params.m());
        let slot = self.push_node(id, vector, level);
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
        let mut nodes: Vec<(u64, Vec<f32>)> = (0..self.node_count()).map(|s| (self.ids[s], self.data.decode(s, self.dims, &mut buf).to_vec())).collect();
        nodes.sort_by_key(|(id, _)| *id);
        let mut fresh = Hnsw::new(self.dims, self.quant, self.params.clone());
        for (id, v) in &nodes {
            fresh.insert_new(*id, v);
        }
        fresh.dropped = std::mem::take(&mut self.dropped);
        fresh.header_dirty = true;
        assert_eq!(fresh.node_count(), nodes.len(), "rebuild lost vectors");
        *self = fresh;
    }

    /// Removes every link to and from `slot` and repairs each node that
    /// pointed at it, found through the inbound lists: its remaining links
    /// plus the removed node's links are re-selected under the cap.
    fn unlink(&mut self, slot: u32) {
        let level = self.levels[slot as usize] as usize;
        let mut buf = Vec::with_capacity(self.dims);
        let mut scratch = Vec::with_capacity(self.dims);
        for layer in 0..=level {
            let gone: Vec<u32> = self.links[slot as usize][layer].clone();
            self.set_links(slot, layer, Vec::new());
            let mut sources: Vec<u32> = self.inbound[slot as usize][layer].clone();
            sources.sort_by_key(|&p| self.ids[p as usize]);
            for p in sources {
                let mut pool: Vec<u32> = Vec::new();
                for &c in self.links[p as usize][layer].iter().chain(gone.iter()) {
                    if c != slot && c != p && !pool.contains(&c) {
                        pool.push(c);
                    }
                }
                let base = self.data.decode(p as usize, self.dims, &mut buf).to_vec();
                let bn = norm(&base);
                let mut cands: Vec<Cand> = pool.iter().map(|&c| self.rank_of(&base, bn, c, &mut scratch)).collect();
                cands.sort_by(|a, b| b.cmp(a));
                let kept = self.select(&cands, self.cap(layer));
                self.set_links(p, layer, kept);
            }
            assert!(self.inbound[slot as usize][layer].is_empty(), "a link to the removed node survived");
            assert!(self.links[slot as usize][layer].is_empty(), "the removed node kept a link");
        }
    }

    /// Drops an unlinked `slot` by moving the last slot into its place.
    /// Only the moved node's neighbours are renumbered, through its links
    /// and inbound lists. A removed entry point passes to the highest
    /// remaining node, lowest id first.
    fn drop_slot(&mut self, slot: u32) {
        let last = (self.node_count() - 1) as u32;
        assert!(self.inbound[slot as usize].iter().all(Vec::is_empty) && self.links[slot as usize].iter().all(Vec::is_empty), "drop_slot needs an unlinked node");
        let was_entry = self.entry == Some(slot);
        let entry_was_last = self.entry == Some(last);
        let id = self.ids[slot as usize];
        self.slot_of.remove(&id);
        if slot != last {
            for layer in 0..=self.levels[last as usize] as usize {
                for i in 0..self.inbound[last as usize][layer].len() {
                    let p = self.inbound[last as usize][layer][i];
                    for c in self.links[p as usize][layer].iter_mut() {
                        if *c == last {
                            *c = slot;
                        }
                    }
                }
                for i in 0..self.links[last as usize][layer].len() {
                    let t = self.links[last as usize][layer][i];
                    for c in self.inbound[t as usize][layer].iter_mut() {
                        if *c == last {
                            *c = slot;
                        }
                    }
                }
            }
            self.ids.swap(slot as usize, last as usize);
            self.levels.swap(slot as usize, last as usize);
            self.links.swap(slot as usize, last as usize);
            self.inbound.swap(slot as usize, last as usize);
            self.data.move_slot(last as usize, slot as usize, self.dims);
            self.slot_of.insert(self.ids[slot as usize], slot);
        }
        self.ids.pop();
        self.levels.pop();
        self.links.pop();
        self.inbound.pop();
        self.data.truncate(last as usize, self.dims);
        if self.ids.is_empty() {
            self.entry = None;
            self.top = 0;
        } else if was_entry {
            let mut best = 0usize;
            for s in 1..self.node_count() {
                if (self.levels[s], std::cmp::Reverse(self.ids[s])) > (self.levels[best], std::cmp::Reverse(self.ids[best])) {
                    best = s;
                }
            }
            self.entry = Some(best as u32);
            self.top = self.levels[best];
        } else if entry_was_last {
            self.entry = Some(slot);
        }
    }

    fn assert_shape(&self) {
        let n = self.node_count();
        assert!(self.levels.len() == n && self.links.len() == n && self.inbound.len() == n, "node arrays out of step");
        assert_eq!(self.slot_of.len(), n, "slot map out of step with nodes");
        assert_eq!(self.data.count(self.dims), n, "vector data out of step with nodes");
        assert_eq!(self.entry.is_none(), n == 0, "entry point out of step with nodes");
        assert!(self.entry.is_none_or(|e| self.levels[e as usize] == self.top), "entry point is not on the top level");
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
        self.node_count()
    }

    fn heap_bytes(&self) -> usize {
        let vectors = self.node_count() * self.data.bytes_per_vector(self.dims);
        let links: usize = self.links.iter().chain(self.inbound.iter()).flatten().map(|l| l.len() * std::mem::size_of::<u32>()).sum();
        vectors + links
    }

    fn insert(&mut self, id: u64, vector: &[f32]) -> Result<(), IndexError> {
        check_vector(self.dims, vector)?;
        match self.slot_of.get(&id).copied() {
            Some(slot) => {
                self.data.set(slot as usize, vector);
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
        self.unlink(slot);
        self.drop_slot(slot);
        self.dirty.remove(&id);
        self.dropped.insert(id);
        self.header_dirty = true;
        self.removed_since_build += 1;
        let limit = f64::from(self.params.max_deleted_fraction()) * self.node_count() as f64;
        if self.node_count() == 0 {
            self.removed_since_build = 0;
        } else if self.removed_since_build as f64 > limit {
            self.rebuild();
        }
        assert!(self.removed_since_build as f64 <= f64::from(self.params.max_deleted_fraction()) * self.node_count() as f64, "removal count past the rebuild limit");
        self.assert_shape();
        true
    }

    fn search(&self, query: &[f32], k: usize, allow: Option<&dyn Fn(u64) -> bool>) -> Result<Vec<Neighbor>, IndexError> {
        check_vector(self.dims, query)?;
        check_k(k)?;
        let Some(mut ep) = self.entry else {
            return Ok(Vec::new());
        };
        if k == 0 {
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
        let returnable = |s: u32| allow.is_none_or(|f| f(self.ids[s as usize]));
        let ef = self.params.ef_search().max(k);
        let found = self.search_layer(query, qn, &[ep], ef, 0, &returnable);
        let mut top = TopK::new(k);
        for c in found {
            top.push(c.rank.0);
        }
        let hits = top.into_sorted();
        assert!(hits.len() <= k, "hnsw returned more than k");
        assert!(hits.windows(2).all(|w| Ranked(w[0]) > Ranked(w[1])), "hnsw result is not sorted by score then id");
        assert!(hits.iter().all(|n| self.slot_of.contains_key(&n.id)), "hnsw returned a removed id");
        Ok(hits)
    }

    fn take_changes(&mut self) -> Vec<PageChange> {
        crate::page::take_changes(self)
    }

    fn restore(&mut self, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError> {
        crate::page::restore(self, vectors, pages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_inbound_mirrors_links(h: &Hnsw) {
        let n = h.node_count();
        let mut want: Vec<Vec<Vec<u32>>> = h.levels.iter().map(|&l| vec![Vec::new(); l as usize + 1]).collect();
        for p in 0..n {
            for (layer, l) in h.links[p].iter().enumerate() {
                for &t in l {
                    assert!(h.levels[t as usize] as usize >= layer, "link to a node below the layer");
                    want[t as usize][layer].push(p as u32);
                }
            }
        }
        for (s, layers) in want.iter_mut().enumerate() {
            for (layer, w) in layers.iter_mut().enumerate() {
                let mut got = h.inbound[s][layer].clone();
                got.sort_unstable();
                w.sort_unstable();
                assert_eq!(&got, w, "inbound of slot {s} layer {layer}");
            }
        }
    }

    #[test]
    fn inbound_mirrors_links_through_inserts_replaces_and_removals() {
        let params = HnswParams::default().with_m(4).with_ef_construction(32).with_max_deleted_fraction(0.3);
        let mut h = Hnsw::new(8, Quantization::F32, params);
        let mut rng = Rng::new(7);
        let vector = |rng: &mut Rng| -> Vec<f32> { (0..8).map(|_| rng.unit_open() as f32 - 0.5).collect() };
        for id in 0..400u64 {
            h.insert(id, &vector(&mut rng)).unwrap();
        }
        assert_inbound_mirrors_links(&h);
        for id in (0..400u64).step_by(3) {
            assert!(h.remove(id));
            assert_inbound_mirrors_links(&h);
            assert!(!h.slot_of.contains_key(&id));
        }
        for id in (0..400u64).step_by(5) {
            h.insert(id, &vector(&mut rng)).unwrap();
            assert_inbound_mirrors_links(&h);
        }
        for id in 0..400u64 {
            h.remove(id);
        }
        assert_inbound_mirrors_links(&h);
        assert!(h.is_empty() && h.entry.is_none());
    }
}
