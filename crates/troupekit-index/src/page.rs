use std::collections::{BTreeMap, HashMap};

use crate::hnsw::Hnsw;
use crate::rng::{Rng, MAX_LEVEL};
use crate::{check_vector, IndexError, PageChange, PageKey};

const FORMAT: u8 = 1;
const HEADER_LEN: usize = 1 + 1 + 8 + 1 + 8 + 8 + 8 + 8 + 8 + 8 + 4 + 4 + 8;

/// Little-endian reader over untrusted page bytes.
/// Every read checks the bytes left and fails with `CorruptPage`.
/// Never panics, whatever the input.
struct Reader<'a> {
    bytes: &'a [u8],
    what: String,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8], what: String) -> Self {
        Self { bytes, what }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], IndexError> {
        if self.bytes.len() < n {
            return Err(corrupt(&self.what, "truncated"));
        }
        let (head, rest) = self.bytes.split_at(n);
        self.bytes = rest;
        Ok(head)
    }

    fn u8(&mut self) -> Result<u8, IndexError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, IndexError> {
        let b = self.take(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, IndexError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, IndexError> {
        let b = self.take(8)?;
        let mut a = [0u8; 8];
        a.copy_from_slice(b);
        Ok(u64::from_le_bytes(a))
    }

    fn left(&self) -> usize {
        self.bytes.len()
    }
}

fn corrupt(what: &str, why: &str) -> IndexError {
    IndexError::CorruptPage(format!("{what}: {why}"))
}

/// Graph-wide state read back from the header page.
/// Checked against the index params and dims before any node is read.
/// `entry` is `None` only for an index with no nodes.
struct Header {
    entry: Option<u64>,
    top: u8,
    nodes: u64,
    removed_since_build: u64,
    rng: Rng,
}

fn encode_header(index: &Hnsw) -> Vec<u8> {
    let p = &index.params;
    let mut out = Vec::with_capacity(HEADER_LEN);
    out.push(FORMAT);
    match index.entry {
        Some(slot) => {
            out.push(1);
            out.extend_from_slice(&index.ids[slot as usize].to_le_bytes());
        }
        None => {
            out.push(0);
            out.extend_from_slice(&0u64.to_le_bytes());
        }
    }
    out.push(index.top);
    out.extend_from_slice(&(index.ids.len() as u64).to_le_bytes());
    out.extend_from_slice(&(index.removed_since_build as u64).to_le_bytes());
    out.extend_from_slice(&(p.m() as u64).to_le_bytes());
    out.extend_from_slice(&(p.ef_construction() as u64).to_le_bytes());
    out.extend_from_slice(&(p.ef_search() as u64).to_le_bytes());
    out.extend_from_slice(&p.seed().to_le_bytes());
    out.extend_from_slice(&p.max_deleted_fraction().to_bits().to_le_bytes());
    out.extend_from_slice(&(index.dims as u32).to_le_bytes());
    out.extend_from_slice(&index.rng.state().to_le_bytes());
    assert_eq!(out.len(), HEADER_LEN, "header length drifted from HEADER_LEN");
    out
}

fn decode_header(index: &Hnsw, bytes: &[u8]) -> Result<Header, IndexError> {
    let what = "header".to_string();
    if bytes.len() != HEADER_LEN {
        return Err(corrupt(&what, "wrong length"));
    }
    let mut r = Reader::new(bytes, what.clone());
    if r.u8()? != FORMAT {
        return Err(corrupt(&what, "unknown format byte"));
    }
    let has_entry = r.u8()?;
    let entry_id = r.u64()?;
    let entry = match has_entry {
        0 => None,
        1 => Some(entry_id),
        _ => return Err(corrupt(&what, "bad entry flag")),
    };
    let top = r.u8()?;
    let nodes = r.u64()?;
    let removed_since_build = r.u64()?;
    let p = &index.params;
    let params_match = r.u64()? == p.m() as u64 && r.u64()? == p.ef_construction() as u64 && r.u64()? == p.ef_search() as u64 && r.u64()? == p.seed() && r.u32()? == p.max_deleted_fraction().to_bits();
    if !params_match {
        return Err(corrupt(&what, "params differ from the index params"));
    }
    if r.u32()? as usize != index.dims {
        return Err(corrupt(&what, "dims differ from the index dims"));
    }
    let rng = Rng::from_state(r.u64()?).ok_or_else(|| corrupt(&what, "zero generator state"))?;
    assert_eq!(r.left(), 0, "header length was checked up front");
    if top > MAX_LEVEL || entry.is_none() != (nodes == 0) {
        return Err(corrupt(&what, "inconsistent counts"));
    }
    if removed_since_build as f64 > f64::from(p.max_deleted_fraction()) * nodes as f64 {
        return Err(corrupt(&what, "removal count past the rebuild limit"));
    }
    Ok(Header { entry, top, nodes, removed_since_build, rng })
}

fn encode_node(index: &Hnsw, slot: usize) -> Vec<u8> {
    let level = index.levels[slot];
    let mut out = vec![FORMAT, level, 0u8];
    assert_eq!(index.links[slot].len(), level as usize + 1, "layer count out of step with level");
    for layer in &index.links[slot] {
        let count = u16::try_from(layer.len()).expect("link count bounded by 2m <= 256");
        out.extend_from_slice(&count.to_le_bytes());
        for &n in layer {
            out.extend_from_slice(&index.ids[n as usize].to_le_bytes());
        }
    }
    out
}

/// One node page decoded but not yet linked to slots.
/// Neighbour ids are checked against the node set afterwards.
/// Pages never carry a vector: vectors come from the store.
struct NodePage {
    level: u8,
    layers: Vec<Vec<u64>>,
}

fn decode_node(index: &Hnsw, id: u64, top: u8, bytes: &[u8]) -> Result<NodePage, IndexError> {
    let what = format!("node {id}");
    let mut r = Reader::new(bytes, what.clone());
    if r.u8()? != FORMAT {
        return Err(corrupt(&what, "unknown format byte"));
    }
    let level = r.u8()?;
    if level > top {
        return Err(corrupt(&what, "level above the header top level"));
    }
    if r.u8()? != 0 {
        return Err(corrupt(&what, "tombstone flag set; removed nodes have no page"));
    }
    let mut layers = Vec::with_capacity(level as usize + 1);
    for layer in 0..=level as usize {
        let count = r.u16()? as usize;
        if count > index.cap(layer) {
            return Err(corrupt(&what, "link count above m or 2m"));
        }
        if r.left() < count * 8 {
            return Err(corrupt(&what, "truncated"));
        }
        let mut links = Vec::with_capacity(count);
        for _ in 0..count {
            let n = r.u64()?;
            if n == id || links.contains(&n) {
                return Err(corrupt(&what, "self or duplicate link"));
            }
            links.push(n);
        }
        layers.push(links);
    }
    if r.left() != 0 {
        return Err(corrupt(&what, "page carries trailing bytes"));
    }
    Ok(NodePage { level, layers })
}

/// Drains the change sets of an HNSW index into pages.
/// Input: the index. Output: the header first when it changed, then
/// `Put` per changed node and `Delete` per dropped node, by id order.
pub(crate) fn take_changes(index: &mut Hnsw) -> Vec<PageChange> {
    let mut out = Vec::new();
    if index.header_dirty {
        out.push(PageChange::Put(PageKey::Header, encode_header(index)));
    }
    let dirty = std::mem::take(&mut index.dirty);
    for id in dirty {
        if let Some(&slot) = index.slot_of.get(&id) {
            out.push(PageChange::Put(PageKey::Node(id), encode_node(index, slot as usize)));
        }
    }
    let dropped = std::mem::take(&mut index.dropped);
    for id in dropped {
        assert!(!index.slot_of.contains_key(&id), "dropped id is still a node");
        out.push(PageChange::Delete(PageKey::Node(id)));
    }
    index.header_dirty = false;
    out
}

/// Restores an HNSW index from live vectors and persisted pages.
/// Input: an empty index, the store's live vectors and its pages.
/// Output: `Ok`, or `CorruptPage` with the index left empty.
pub(crate) fn restore(index: &mut Hnsw, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError> {
    if !index.ids.is_empty() {
        return Err(IndexError::Invalid("restore needs an empty index".into()));
    }
    let result = restore_into(index, vectors, pages);
    if result.is_err() {
        *index = Hnsw::new(index.dims, index.quant, index.params.clone());
    }
    result
}

fn restore_into(index: &mut Hnsw, vectors: &mut dyn Iterator<Item = (u64, Vec<f32>)>, pages: &mut dyn Iterator<Item = (PageKey, Vec<u8>)>) -> Result<(), IndexError> {
    let mut header_bytes: Option<Vec<u8>> = None;
    let mut node_bytes: BTreeMap<u64, Vec<u8>> = BTreeMap::new();
    for (key, bytes) in pages {
        let duplicate = match key {
            PageKey::Header => header_bytes.replace(bytes).is_some(),
            PageKey::Node(id) => node_bytes.insert(id, bytes).is_some(),
        };
        if duplicate {
            return Err(corrupt("pages", "duplicate page key"));
        }
    }
    let mut live: HashMap<u64, Vec<f32>> = HashMap::new();
    let mut live_order: Vec<u64> = Vec::new();
    for (id, v) in vectors {
        check_vector(index.dims, &v)?;
        if live.insert(id, v).is_some() {
            return Err(IndexError::Invalid(format!("vector id {id} given twice")));
        }
        live_order.push(id);
    }
    let Some(header_bytes) = header_bytes else {
        if !node_bytes.is_empty() {
            return Err(corrupt("pages", "node pages without a header"));
        }
        for id in live_order {
            let v = live.remove(&id).expect("id taken from the live map");
            crate::VectorIndex::insert(index, id, &v)?;
        }
        return Ok(());
    };
    let header = decode_header(index, &header_bytes)?;
    if node_bytes.len() as u64 != header.nodes || live.len() as u64 != header.nodes {
        return Err(corrupt("header", "node or live count differs from the pages"));
    }
    if header.nodes > u64::from(u32::MAX) {
        return Err(corrupt("header", "node count above u32::MAX"));
    }
    let mut decoded: BTreeMap<u64, NodePage> = BTreeMap::new();
    for (&id, bytes) in &node_bytes {
        let page = decode_node(index, id, header.top, bytes)?;
        if !live.contains_key(&id) {
            return Err(corrupt(&format!("node {id}"), "node page without a live vector"));
        }
        decoded.insert(id, page);
    }
    for (&id, page) in &decoded {
        for (layer, links) in page.layers.iter().enumerate() {
            for n in links {
                match decoded.get(n) {
                    Some(other) if other.level as usize >= layer => {}
                    _ => return Err(corrupt(&format!("node {id}"), "link to an unknown node or a node below the layer")),
                }
            }
        }
    }
    let entry_slot = match header.entry {
        Some(e) => match decoded.get(&e) {
            Some(p) if p.level == header.top => decoded.keys().position(|&k| k == e),
            _ => return Err(corrupt("header", "entry point is not a node at the top level")),
        },
        None => None,
    };
    if let Some(id) = live.keys().find(|id| !decoded.contains_key(id)) {
        return Err(corrupt(&format!("vector {id}"), "live vector without a node page"));
    }
    let slot_of: HashMap<u64, u32> = decoded.keys().enumerate().map(|(i, &id)| (id, i as u32)).collect();
    for (id, page) in decoded {
        let v = live.remove(&id).expect("node page checked against the live map");
        index.ids.push(id);
        index.levels.push(page.level);
        index.links.push(page.layers.iter().map(|l| l.iter().map(|n| slot_of[n]).collect()).collect());
        index.inbound.push(vec![Vec::new(); page.level as usize + 1]);
        index.data.push(&v);
    }
    for p in 0..index.links.len() {
        for layer in 0..index.links[p].len() {
            for i in 0..index.links[p][layer].len() {
                let t = index.links[p][layer][i] as usize;
                index.inbound[t][layer].push(p as u32);
            }
        }
    }
    assert!(live.is_empty(), "every live vector was checked to have a node page");
    index.slot_of = slot_of;
    index.entry = entry_slot.map(|s| s as u32);
    index.top = header.top;
    index.rng = header.rng;
    index.removed_since_build = usize::try_from(header.removed_since_build).expect("bounded by the node count");
    index.dirty.clear();
    index.dropped.clear();
    index.header_dirty = false;
    assert_eq!(index.ids.len() as u64, header.nodes, "node count restored");
    Ok(())
}
