use std::collections::HashMap;
use std::collections::HashSet;

use troupekit_index::{new_index, IndexKind, Quantization, VectorIndex};

use crate::contract::{cosine, Rng};

fn index(dims: usize, quant: Quantization) -> Box<dyn VectorIndex> {
    new_index(&IndexKind::BruteForce, dims, quant).expect("brute force index")
}

fn unit(rng: &mut Rng, dims: usize) -> Vec<f32> {
    let v = rng.vector(dims);
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.into_iter().map(|x| x / n).collect()
}

#[test]
fn i8_cosine_error_is_bounded() {
    let mut rng = Rng::new(21);
    let mut q8 = index(384, Quantization::I8);
    let items: Vec<Vec<f32>> = (0..1_000).map(|_| unit(&mut rng, 384)).collect();
    for (i, v) in items.iter().enumerate() {
        q8.insert(i as u64, v).unwrap();
    }
    for _ in 0..5 {
        let q = unit(&mut rng, 384);
        let got: HashMap<u64, f32> = q8.search(&q, 1_000, None).unwrap().into_iter().map(|n| (n.id, n.score)).collect();
        assert_eq!(got.len(), 1_000);
        for (i, v) in items.iter().enumerate() {
            let err = (got[&(i as u64)] - cosine(&q, v)).abs();
            assert!(err <= 0.02, "id {i}: error {err}");
        }
    }
}

#[test]
fn i8_zero_vector_stays_zero() {
    let mut q8 = index(4, Quantization::I8);
    q8.insert(1, &[0.0; 4]).unwrap();
    q8.insert(2, &[1.0, -1.0, 0.5, 0.0]).unwrap();
    let got = q8.search(&[1.0, 0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(got[1].id, 1);
    assert_eq!(got[1].score, 0.0);
    q8.insert(2, &[0.0; 4]).unwrap();
    assert!(q8.search(&[1.0, 2.0, 3.0, 4.0], 2, None).unwrap().iter().all(|n| n.score == 0.0));
}

#[test]
fn heap_bytes_per_quantization() {
    let dims = 96;
    for (quant, per) in [(Quantization::F32, 4 * dims), (Quantization::F16, 2 * dims), (Quantization::I8, dims + 4)] {
        let mut rng = Rng::new(3);
        let mut idx = index(dims, quant);
        assert_eq!(idx.heap_bytes(), 0);
        for id in 0..50 {
            idx.insert(id, &rng.vector(dims)).unwrap();
        }
        assert_eq!(idx.heap_bytes(), 50 * per, "{quant:?}");
        idx.insert(7, &rng.vector(dims)).unwrap();
        assert!(idx.remove(3));
        assert_eq!(idx.heap_bytes(), 49 * per, "{quant:?}");
    }
}

#[test]
fn brute_force_contract_holds_for_f16_and_i8() {
    fn f16(dims: usize) -> Box<dyn VectorIndex> {
        index(dims, Quantization::F16)
    }
    fn i8q(dims: usize) -> Box<dyn VectorIndex> {
        index(dims, Quantization::I8)
    }
    for new in [f16 as crate::contract::Factory, i8q] {
        crate::contract::case_insert_same_id_replaces(new);
        crate::contract::case_remove_never_returned(new);
        crate::contract::case_dimension_mismatch(new);
        crate::contract::case_rejects_nan_and_big_k(new);
        crate::contract::case_zero_vector_scores_zero(new);
    }
    for quant in [Quantization::F16, Quantization::I8] {
        let mut rng = Rng::new(31);
        let mut exact = index(64, Quantization::F32);
        let mut q = index(64, quant);
        for id in 0..5_000 {
            let v = rng.vector(64);
            exact.insert(id, &v).unwrap();
            q.insert(id, &v).unwrap();
        }
        let mut found = 0usize;
        let queries = 200;
        for _ in 0..queries {
            let query = rng.vector(64);
            let want: HashSet<u64> = exact.search(&query, 10, None).unwrap().into_iter().map(|n| n.id).collect();
            found += q.search(&query, 10, None).unwrap().iter().filter(|n| want.contains(&n.id)).count();
        }
        let recall = found as f64 / (queries * 10) as f64;
        assert!(recall >= 0.99, "{quant:?} recall@10 {recall}");
    }
}

#[test]
fn i8_max_magnitude_stays_finite() {
    let mut q8 = index(3, Quantization::I8);
    q8.insert(1, &[f32::MAX, 0.0, -f32::MAX]).unwrap();
    q8.insert(2, &[0.0, 1.0, 0.0]).unwrap();
    let got = q8.search(&[1.0, 0.0, -1.0], 2, None).unwrap();
    assert!(got.iter().all(|n| n.score.is_finite()), "{got:?}");
    assert_eq!(got[0].id, 1);
    assert!((got[0].score - 1.0).abs() < 1e-3, "{:?}", got[0]);
}
