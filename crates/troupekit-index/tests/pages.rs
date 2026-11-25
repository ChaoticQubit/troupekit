use std::collections::BTreeMap;

use troupekit_index::{new_index, HnswParams, IndexError, IndexKind, PageChange, PageKey, Quantization, VectorIndex};

use crate::contract::{clustered, Rng};

const DIMS: usize = 16;

fn params() -> HnswParams {
    HnswParams::default().with_m(8).with_ef_construction(64)
}

fn fresh(p: HnswParams) -> Box<dyn VectorIndex> {
    new_index(&IndexKind::Hnsw(p), DIMS, Quantization::F32).expect("hnsw index")
}

#[derive(Default, Clone)]
struct Disk {
    vectors: BTreeMap<u64, Vec<f32>>,
    pages: BTreeMap<PageKey, Vec<u8>>,
}

impl Disk {
    fn apply(&mut self, changes: Vec<PageChange>) {
        for c in changes {
            match c {
                PageChange::Put(k, b) => {
                    self.pages.insert(k, b);
                }
                PageChange::Delete(k) => {
                    self.pages.remove(&k);
                }
                _ => unreachable!("only Put and Delete exist"),
            }
        }
    }

    fn insert(&mut self, index: &mut dyn VectorIndex, id: u64, v: &[f32]) {
        index.insert(id, v).unwrap();
        self.vectors.insert(id, v.to_vec());
        self.apply(index.take_changes());
    }

    fn remove(&mut self, index: &mut dyn VectorIndex, id: u64) {
        assert!(index.remove(id));
        self.vectors.remove(&id);
        self.apply(index.take_changes());
    }

    fn restore(&self, p: HnswParams) -> (Box<dyn VectorIndex>, Result<(), IndexError>) {
        let mut index = fresh(p);
        let mut vectors = self.vectors.clone().into_iter();
        let mut pages = self.pages.clone().into_iter();
        let r = index.restore(&mut vectors, &mut pages);
        (index, r)
    }
}

fn built(n: usize, seed: u64) -> (Box<dyn VectorIndex>, Disk, Vec<Vec<f32>>) {
    let data = clustered(seed, n, DIMS, 10, 0.4);
    let mut index = fresh(params());
    let mut disk = Disk::default();
    for (i, v) in data.iter().enumerate() {
        disk.insert(index.as_mut(), i as u64 + 1, v);
    }
    (index, disk, data)
}

fn assert_same_results(a: &dyn VectorIndex, b: &dyn VectorIndex, queries: &[Vec<f32>]) {
    for q in queries {
        assert_eq!(a.search(q, 10, None).unwrap(), b.search(q, 10, None).unwrap());
    }
}

fn assert_corrupt(disk: &Disk) {
    let (index, r) = disk.restore(params());
    assert!(matches!(r, Err(IndexError::CorruptPage(_))), "{r:?}");
    assert_eq!(index.len(), 0);
    assert_eq!(index.heap_bytes(), 0);
}

fn first_node_with_links(disk: &Disk) -> PageKey {
    *disk.pages.iter().find(|(k, b)| matches!(k, PageKey::Node(_)) && b.len() > 5 && u16::from_le_bytes([b[3], b[4]]) > 0).expect("a node with links").0
}

#[test]
fn restore_round_trip_gives_identical_results() {
    let (index, disk, _) = built(2_000, 71);
    let (restored, r) = disk.restore(params());
    r.unwrap();
    assert_eq!(restored.len(), 2_000);
    let queries = clustered(72, 500, DIMS, 10, 0.4);
    assert_same_results(index.as_ref(), restored.as_ref(), &queries);
}

#[test]
fn one_insert_touches_bounded_pages() {
    let (mut index, _, _) = built(10_000, 73);
    let mut rng = Rng::new(9);
    index.insert(20_000, &rng.vector(DIMS)).unwrap();
    let changes = index.take_changes();
    assert!(changes.len() <= 2 * params().m() + 2, "{} pages", changes.len());
    assert!(changes.iter().any(|c| matches!(c, PageChange::Put(PageKey::Node(20_000), _))), "new node page missing");
    assert!(changes.iter().any(|c| matches!(c, PageChange::Put(PageKey::Header, _))), "header page missing");
}

#[test]
fn truncated_page_is_corrupt() {
    let (_, mut disk, _) = built(200, 74);
    let key = first_node_with_links(&disk);
    disk.pages.get_mut(&key).unwrap().pop();
    assert_corrupt(&disk);
    let (_, mut disk, _) = built(200, 74);
    disk.pages.get_mut(&PageKey::Header).unwrap().pop();
    assert_corrupt(&disk);
}

#[test]
fn count_above_bound_is_corrupt() {
    let (_, mut disk, _) = built(200, 75);
    let key = first_node_with_links(&disk);
    let page = disk.pages.get_mut(&key).unwrap();
    let over = (2 * params().m() + 1) as u16;
    page[3..5].copy_from_slice(&over.to_le_bytes());
    page.extend(std::iter::repeat_n(0u8, 8 * over as usize));
    assert_corrupt(&disk);
}

#[test]
fn unknown_neighbour_id_is_corrupt() {
    let (_, mut disk, _) = built(200, 76);
    let key = first_node_with_links(&disk);
    disk.pages.get_mut(&key).unwrap()[5..13].copy_from_slice(&999_999u64.to_le_bytes());
    assert_corrupt(&disk);
}

#[test]
fn wrong_format_byte_is_corrupt() {
    let (_, mut disk, _) = built(200, 77);
    let key = first_node_with_links(&disk);
    disk.pages.get_mut(&key).unwrap()[0] = 2;
    assert_corrupt(&disk);
    let (_, mut disk, _) = built(200, 77);
    disk.pages.get_mut(&PageKey::Header).unwrap()[0] = 0;
    assert_corrupt(&disk);
}

#[test]
fn params_mismatch_is_corrupt() {
    let (_, disk, _) = built(200, 78);
    for p in [params().with_m(9), params().with_ef_search(65), params().with_seed(1), params().with_max_deleted_fraction(0.5)] {
        let (index, r) = disk.restore(p);
        assert!(matches!(r, Err(IndexError::CorruptPage(_))), "{r:?}");
        assert_eq!(index.len(), 0);
    }
    let mut wrong_dims = new_index(&IndexKind::Hnsw(params()), DIMS + 1, Quantization::F32).unwrap();
    let r = wrong_dims.restore(&mut std::iter::empty(), &mut disk.pages.clone().into_iter());
    assert!(matches!(r, Err(IndexError::CorruptPage(_))), "{r:?}");
}

#[test]
fn random_bytes_never_panic() {
    let (_, disk, _) = built(50, 79);
    let keys: Vec<PageKey> = disk.pages.keys().copied().collect();
    let mut rng = Rng::new(80);
    for round in 0..10_000 {
        let mut bad = disk.clone();
        let key = keys[(rng.next_u64() % keys.len() as u64) as usize];
        let len = (rng.next_u64() % 160) as usize;
        let mut bytes: Vec<u8> = (0..len).map(|_| rng.next_u64() as u8).collect();
        if round % 2 == 0 && !bytes.is_empty() {
            bytes[0] = 1;
        }
        bad.pages.insert(key, bytes);
        let (index, r) = bad.restore(params());
        if r.is_err() {
            assert_eq!(index.len(), 0);
        }
    }
}

#[test]
fn restore_after_remove_without_live_vector_keeps_results() {
    let (mut index, mut disk, _) = built(1_000, 81);
    for id in (1..=1_000).step_by(9) {
        disk.remove(index.as_mut(), id);
    }
    assert_eq!(index.len(), 1_000 - 112);
    let (restored, r) = disk.restore(params());
    r.unwrap();
    assert_eq!(restored.len(), index.len());
    let queries = clustered(82, 200, DIMS, 10, 0.4);
    assert_same_results(index.as_ref(), restored.as_ref(), &queries);
}

fn removed() -> (Disk, u64) {
    let (mut index, mut disk, _) = built(200, 83);
    disk.remove(index.as_mut(), 17);
    (disk, 17)
}

#[test]
fn tombstone_flag_is_corrupt() {
    let (mut disk, _) = removed();
    let page = disk.pages.get_mut(&PageKey::Node(5)).unwrap();
    page[2] = 1;
    assert_corrupt(&disk);
    let (mut disk, _) = removed();
    let page = disk.pages.get_mut(&PageKey::Node(5)).unwrap();
    page[2] = 1;
    page.extend(std::iter::repeat_n(0u8, 4 * DIMS));
    assert_corrupt(&disk);
    let (disk, id) = removed();
    assert!(!disk.pages.contains_key(&PageKey::Node(id)));
    disk.restore(params()).1.unwrap();
}

#[test]
fn live_page_with_vector_is_corrupt() {
    let (_, mut disk, _) = built(200, 84);
    disk.pages.get_mut(&PageKey::Node(5)).unwrap().extend(std::iter::repeat_n(0u8, 4 * DIMS));
    assert_corrupt(&disk);
}

#[test]
fn node_page_without_live_vector_is_corrupt() {
    let (_, mut disk, _) = built(200, 87);
    disk.vectors.remove(&17);
    assert_corrupt(&disk);
}

#[test]
fn remove_deletes_the_node_page_at_once() {
    let data = clustered(85, 100, DIMS, 5, 0.4);
    let mut index = fresh(params());
    let mut disk = Disk::default();
    for (i, v) in data.iter().enumerate() {
        disk.insert(index.as_mut(), i as u64 + 1, v);
    }
    for id in 1..=26u64 {
        assert!(index.remove(id));
        disk.vectors.remove(&id);
        let changes = index.take_changes();
        assert!(changes.contains(&PageChange::Delete(PageKey::Node(id))), "id {id}: {changes:?}");
        assert!(!changes.iter().any(|c| matches!(c, PageChange::Put(PageKey::Node(n), _) if *n == id)));
        disk.apply(changes);
        assert!(!disk.pages.contains_key(&PageKey::Node(id)));
    }
    assert_eq!(index.len(), 74);
    assert_eq!(disk.pages.len(), 75);
    let (restored, r) = disk.restore(params());
    r.unwrap();
    assert_same_results(index.as_ref(), restored.as_ref(), &data);
}

#[test]
fn removed_vector_leaves_no_bytes_in_any_page() {
    let (mut index, mut disk, _) = built(300, 88);
    let marker: Vec<f32> = (0..DIMS).map(|i| 1234.5 + i as f32).collect();
    disk.insert(index.as_mut(), 999, &marker);
    disk.remove(index.as_mut(), 999);
    let needle: Vec<u8> = marker[..4].iter().flat_map(|x| x.to_le_bytes()).collect();
    for (key, bytes) in &disk.pages {
        assert!(!bytes.windows(needle.len()).any(|w| w == needle.as_slice()), "{key:?} holds the removed vector");
        if let PageKey::Node(n) = key {
            assert_ne!(*n, 999);
        }
        assert!(!matches!(key, PageKey::Node(_)) || bytes.windows(8).all(|w| w != 999u64.to_le_bytes()), "{key:?} still links to the removed id");
    }
    assert!(index.search(&marker, 300, None).unwrap().iter().all(|n| n.id != 999));
}

#[test]
fn reinsert_after_rebuild_clears_the_pending_delete() {
    let mut index = fresh(params().with_max_deleted_fraction(0.25));
    let mut rng = Rng::new(86);
    for id in 5..8u64 {
        index.insert(id, &rng.vector(DIMS)).unwrap();
    }
    index.take_changes();
    assert!(index.remove(7));
    index.insert(7, &rng.vector(DIMS)).unwrap();
    let changes = index.take_changes();
    assert!(changes.iter().any(|c| matches!(c, PageChange::Put(PageKey::Node(7), _))), "{changes:?}");
    assert!(!changes.iter().any(|c| matches!(c, PageChange::Delete(PageKey::Node(7)))), "{changes:?}");
    assert_eq!(index.len(), 3);
}
