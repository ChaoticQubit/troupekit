use troupekit_index::{new_index, HnswParams, IndexKind, Quantization, VectorIndex};

use crate::contract::{brute, clustered, contract, recall_at_10, Rng};

fn hnsw(dims: usize) -> Box<dyn VectorIndex> {
    new_index(&IndexKind::Hnsw(HnswParams::default()), dims, Quantization::F32).expect("hnsw index")
}

fn hnsw_with(dims: usize, params: HnswParams) -> Box<dyn VectorIndex> {
    new_index(&IndexKind::Hnsw(params), dims, Quantization::F32).expect("hnsw index")
}

fn build(index: &mut dyn VectorIndex, oracle: &mut dyn VectorIndex, data: &[Vec<f32>]) {
    for (i, v) in data.iter().enumerate() {
        index.insert(i as u64, v).unwrap();
        oracle.insert(i as u64, v).unwrap();
    }
}

#[test]
fn hnsw_contract() {
    contract(hnsw);
}

#[test]
fn recall_at_10_on_10k_is_at_least_0_95() {
    let data = clustered(41, 10_100, 64, 50, 0.12);
    let (items, queries) = data.split_at(10_000);
    let mut index = hnsw(64);
    let mut oracle = brute(64);
    build(index.as_mut(), oracle.as_mut(), items);
    assert_eq!(index.len(), 10_000);
    let recall = recall_at_10(index.as_ref(), oracle.as_ref(), queries);
    assert!(recall >= 0.95, "recall@10 {recall}");
}

#[test]
fn same_seed_same_results() {
    let data = clustered(43, 2_050, 32, 20, 0.4);
    let (items, queries) = data.split_at(2_000);
    let params = HnswParams::default().with_seed(99).with_m(8);
    let mut a = hnsw_with(32, params.clone());
    let mut b = hnsw_with(32, params);
    for (i, v) in items.iter().enumerate() {
        a.insert(i as u64, v).unwrap();
        b.insert(i as u64, v).unwrap();
    }
    for q in queries {
        assert_eq!(a.search(q, 10, None).unwrap(), b.search(q, 10, None).unwrap());
    }
}

#[test]
fn links_never_exceed_m_and_2m() {
    let m = 4;
    let data = clustered(47, 3_000, 16, 10, 0.3);
    let mut index = hnsw_with(16, HnswParams::default().with_m(m).with_ef_construction(32));
    let mut rng = Rng::new(5);
    for (i, v) in data.iter().enumerate() {
        index.insert(i as u64, v).unwrap();
        if i % 7 == 0 {
            index.insert(rng.next_u64() % (i as u64 + 1), &data[(rng.next_u64() % 3_000) as usize]).unwrap();
        }
        if i % 11 == 0 {
            index.remove(rng.next_u64() % (i as u64 + 1));
        }
    }
    let vectors = index.len() * 16 * 4;
    let link_bytes = index.heap_bytes() - vectors;
    assert!(link_bytes <= 3_000 * (2 * m + 16 * m) * 4, "link bytes {link_bytes}");
}

#[test]
fn filtered_search_returns_only_allowed() {
    let data = clustered(53, 3_000, 32, 20, 0.4);
    let mut index = hnsw(32);
    let mut oracle = brute(32);
    build(index.as_mut(), oracle.as_mut(), &data);
    let allow = |id: u64| id % 5 == 0;
    let mut found = 0usize;
    for q in data.iter().step_by(97) {
        let got = index.search(q, 10, Some(&allow)).unwrap();
        assert!(got.iter().all(|n| n.id % 5 == 0));
        assert_eq!(got.len(), 10);
        let want = oracle.search(q, 10, Some(&allow)).unwrap();
        found += got.iter().filter(|n| want.iter().any(|w| w.id == n.id)).count();
    }
    let queries = data.iter().step_by(97).count();
    assert!(found as f64 / (queries * 10) as f64 >= 0.9, "filtered recall {found}");
}

#[test]
fn filter_matching_one_node_still_finds_it() {
    let data = clustered(59, 2_000, 32, 20, 0.4);
    let mut index = hnsw(32);
    for (i, v) in data.iter().enumerate() {
        index.insert(i as u64, v).unwrap();
    }
    for target in [0u64, 777, 1_999] {
        let only = move |id: u64| id == target;
        let got = index.search(&data[((target + 1000) % 2000) as usize], 10, Some(&only)).unwrap();
        assert_eq!(got.len(), 1, "target {target}");
        assert_eq!(got[0].id, target);
    }
}

#[test]
fn removed_ids_are_never_returned() {
    let data = clustered(61, 1_000, 16, 10, 0.4);
    let mut index = hnsw_with(16, HnswParams::default().with_max_deleted_fraction(1.0));
    for (i, v) in data.iter().enumerate() {
        index.insert(i as u64, v).unwrap();
    }
    for id in (0..1_000).step_by(3) {
        assert!(index.remove(id));
    }
    assert_eq!(index.len(), 666);
    for q in data.iter().step_by(13) {
        let got = index.search(q, 50, None).unwrap();
        assert_eq!(got.len(), 50);
        assert!(got.iter().all(|n| n.id % 3 != 0));
    }
}

#[test]
fn rebuild_after_deleted_fraction_keeps_recall() {
    let data = clustered(67, 4_100, 32, 30, 0.35);
    let (items, queries) = data.split_at(4_000);
    let mut index = hnsw(32);
    let mut oracle = brute(32);
    build(index.as_mut(), oracle.as_mut(), items);
    let bytes_before = index.heap_bytes();
    for id in (0..4_000).filter(|id| id % 10 < 4) {
        assert!(index.remove(id));
        assert!(oracle.remove(id));
    }
    assert_eq!(index.len(), 2_400);
    assert!(index.heap_bytes() < bytes_before, "removal must free vector memory");
    let recall = recall_at_10(index.as_ref(), oracle.as_ref(), queries);
    assert!(recall >= 0.95, "recall@10 after rebuild {recall}");
}

#[test]
fn single_node_and_empty_index() {
    let mut index = hnsw(3);
    assert!(index.is_empty());
    assert!(index.search(&[1.0, 0.0, 0.0], 5, None).unwrap().is_empty());
    index.insert(9, &[0.0, 1.0, 0.0]).unwrap();
    let got = index.search(&[1.0, 0.0, 0.0], 5, None).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, 9);
    index.insert(9, &[1.0, 0.0, 0.0]).unwrap();
    assert_eq!(index.len(), 1);
    assert!((index.search(&[1.0, 0.0, 0.0], 1, None).unwrap()[0].score - 1.0).abs() < 1e-6);
    assert!(index.remove(9));
    assert!(index.is_empty());
    assert!(index.search(&[1.0, 0.0, 0.0], 5, None).unwrap().is_empty());
    index.insert(10, &[0.0, 0.0, 1.0]).unwrap();
    assert_eq!(index.search(&[0.0, 0.0, 1.0], 5, None).unwrap()[0].id, 10);
}

#[test]
fn hnsw_params_out_of_range_are_invalid() {
    for p in [
        HnswParams::default().with_m(1),
        HnswParams::default().with_m(129),
        HnswParams::default().with_ef_construction(0),
        HnswParams::default().with_ef_search(4_001),
        HnswParams::default().with_max_deleted_fraction(0.0),
        HnswParams::default().with_max_deleted_fraction(f32::NAN),
    ] {
        assert!(matches!(new_index(&IndexKind::Hnsw(p), 4, Quantization::F32), Err(troupekit_index::IndexError::Invalid(_))));
    }
}

#[test]
fn equal_vectors_stay_reachable_after_pruning() {
    for n in [6u64, 40] {
        let mut index = hnsw_with(2, HnswParams::default().with_m(2));
        for id in 0..n {
            index.insert(id, &[1.0, 0.0]).unwrap();
        }
        for id in 0..n {
            let only = move |x: u64| x == id;
            let got = index.search(&[1.0, 0.0], 1, Some(&only)).unwrap();
            assert_eq!(got.len(), 1, "n {n}: id {id} unreachable");
            assert_eq!(got[0].id, id);
        }
    }
}

#[test]
fn removal_repair_keeps_recall_without_rebuild() {
    let data = clustered(91, 3_100, 32, 30, 0.35);
    let (items, queries) = data.split_at(3_000);
    let mut index = hnsw_with(32, HnswParams::default().with_max_deleted_fraction(1.0));
    let mut oracle = brute(32);
    build(index.as_mut(), oracle.as_mut(), items);
    for id in (0..3_000).filter(|id| id % 10 < 3) {
        assert!(index.remove(id));
        assert!(oracle.remove(id));
        assert!(!index.remove(id));
    }
    assert_eq!(index.len(), 2_100);
    let recall = recall_at_10(index.as_ref(), oracle.as_ref(), queries);
    assert!(recall >= 0.95, "recall@10 after local repair {recall}");
}

#[test]
fn hard_data_recall_rises_with_ef_search() {
    let data = clustered(41, 10_100, 64, 50, 0.35);
    let (items, queries) = data.split_at(10_000);
    let mut oracle = brute(64);
    let mut recalls = Vec::new();
    for ef in [64usize, 400] {
        let mut index = hnsw_with(64, HnswParams::default().with_ef_search(ef));
        let mut scratch = brute(64);
        build(index.as_mut(), scratch.as_mut(), items);
        if oracle.is_empty() {
            oracle = scratch;
        }
        recalls.push(recall_at_10(index.as_ref(), oracle.as_ref(), queries));
    }
    assert!(recalls[1] > recalls[0], "recall must rise with ef_search: {recalls:?}");
    assert!(recalls[1] >= 0.95, "a connected graph reaches 0.95 with a wide beam: {recalls:?}");
}
