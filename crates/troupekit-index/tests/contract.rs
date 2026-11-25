use troupekit_index::{new_index, HnswParams, IndexError, IndexKind, Neighbor, Quantization, VectorIndex, MAX_DIMS, MAX_K};

pub type Factory = fn(usize) -> Box<dyn VectorIndex>;

pub fn brute(dims: usize) -> Box<dyn VectorIndex> {
    new_index(&IndexKind::BruteForce, dims, Quantization::F32).expect("brute force index")
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
    }

    pub fn vector(&mut self, dims: usize) -> Vec<f32> {
        (0..dims).map(|_| self.unit()).collect()
    }
}

pub fn gaussian(rng: &mut Rng) -> f32 {
    let u1 = ((rng.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

pub fn normalize(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n == 0.0 {
        return v;
    }
    v.into_iter().map(|x| x / n).collect()
}

pub fn clustered(seed: u64, n: usize, dims: usize, clusters: usize, spread: f32) -> Vec<Vec<f32>> {
    let mut rng = Rng::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| normalize((0..dims).map(|_| gaussian(&mut rng)).collect())).collect();
    (0..n)
        .map(|_| {
            let c = &centres[(rng.next_u64() % clusters as u64) as usize];
            normalize(c.iter().map(|x| x + spread * gaussian(&mut rng)).collect())
        })
        .collect()
}

pub fn recall_at_10(index: &dyn VectorIndex, oracle: &dyn VectorIndex, queries: &[Vec<f32>]) -> f64 {
    let mut found = 0usize;
    let mut total = 0usize;
    for q in queries {
        let want: Vec<u64> = ids(&oracle.search(q, 10, None).unwrap());
        let got: Vec<u64> = ids(&index.search(q, 10, None).unwrap());
        total += want.len();
        found += got.iter().filter(|id| want.contains(id)).count();
    }
    found as f64 / total.max(1) as f64
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| f64::from(x) * f64::from(y)).sum();
    let na = a.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    let nb = b.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    (dot / (na * nb)) as f32
}

pub fn exact(items: &[(u64, Vec<f32>)], query: &[f32], k: usize, allow: &dyn Fn(u64) -> bool) -> Vec<Neighbor> {
    let mut all: Vec<Neighbor> = items.iter().filter(|(id, _)| allow(*id)).map(|(id, v)| Neighbor { id: *id, score: cosine(query, v) }).collect();
    all.sort_by(|a, b| b.score.total_cmp(&a.score).then_with(|| a.id.cmp(&b.id)));
    all.truncate(k);
    all
}

pub fn ids(hits: &[Neighbor]) -> Vec<u64> {
    hits.iter().map(|n| n.id).collect()
}

pub fn case_min_k_and_n_sorted(new: Factory) {
    let mut rng = Rng::new(7);
    let mut index = new(16);
    let items: Vec<(u64, Vec<f32>)> = (0..40).map(|i| (i * 3 + 1, rng.vector(16))).collect();
    for (id, v) in &items {
        index.insert(*id, v).unwrap();
    }
    index.insert(1000, &items[0].1).unwrap();
    let all_items: Vec<(u64, Vec<f32>)> = items.iter().cloned().chain([(1000, items[0].1.clone())]).collect();
    assert_eq!(index.len(), 41);
    for k in [0, 1, 5, 41, 100] {
        let q = rng.vector(16);
        let got = index.search(&q, k, None).unwrap();
        let want = exact(&all_items, &q, k, &|_| true);
        assert_eq!(got.len(), k.min(41));
        assert_eq!(ids(&got), ids(&want), "k {k}");
    }
    let tie = index.search(&items[0].1, 2, None).unwrap();
    assert_eq!(ids(&tie), vec![items[0].0, 1000]);
}

pub fn case_allow_excludes(new: Factory) {
    let mut rng = Rng::new(11);
    let mut index = new(8);
    let items: Vec<(u64, Vec<f32>)> = (0..30).map(|i| (i, rng.vector(8))).collect();
    for (id, v) in &items {
        index.insert(*id, v).unwrap();
    }
    let even = |id: u64| id % 2 == 0;
    let q = rng.vector(8);
    let got = index.search(&q, 10, Some(&even)).unwrap();
    assert_eq!(ids(&got), ids(&exact(&items, &q, 10, &even)));
    assert!(got.iter().all(|n| n.id % 2 == 0));
    let none = |_: u64| false;
    assert!(index.search(&q, 10, Some(&none)).unwrap().is_empty());
}

pub fn case_insert_same_id_replaces(new: Factory) {
    let mut index = new(3);
    index.insert(5, &[1.0, 0.0, 0.0]).unwrap();
    index.insert(6, &[0.0, 1.0, 0.0]).unwrap();
    index.insert(5, &[0.0, 0.0, 1.0]).unwrap();
    assert_eq!(index.len(), 2);
    let got = index.search(&[0.0, 0.0, 1.0], 1, None).unwrap();
    assert_eq!(got[0].id, 5);
    assert!((got[0].score - 1.0).abs() < 1e-6);
    let got = index.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
    assert!(got.iter().all(|n| n.score < 0.5));
}

pub fn case_remove_never_returned(new: Factory) {
    let mut rng = Rng::new(13);
    let mut index = new(8);
    let items: Vec<(u64, Vec<f32>)> = (0..20).map(|i| (i, rng.vector(8))).collect();
    for (id, v) in &items {
        index.insert(*id, v).unwrap();
    }
    assert!(index.remove(4));
    assert!(!index.remove(4));
    assert!(!index.remove(999));
    assert_eq!(index.len(), 19);
    let got = index.search(&items[4].1, 20, None).unwrap();
    assert_eq!(got.len(), 19);
    assert!(got.iter().all(|n| n.id != 4));
    index.insert(4, &items[4].1).unwrap();
    assert_eq!(index.search(&items[4].1, 1, None).unwrap()[0].id, 4);
}

pub fn case_dimension_mismatch(new: Factory) {
    let mut index = new(4);
    assert_eq!(index.dims(), 4);
    assert_eq!(index.insert(1, &[1.0, 2.0]), Err(IndexError::DimensionMismatch { expected: 4, actual: 2 }));
    assert_eq!(index.search(&[1.0; 5], 1, None), Err(IndexError::DimensionMismatch { expected: 4, actual: 5 }));
    assert_eq!(index.len(), 0);
}

pub fn case_rejects_nan_and_big_k(new: Factory) {
    let mut index = new(2);
    assert!(matches!(index.insert(1, &[f32::NAN, 0.0]), Err(IndexError::Invalid(_))));
    assert!(matches!(index.insert(1, &[f32::INFINITY, 0.0]), Err(IndexError::Invalid(_))));
    assert_eq!(index.len(), 0);
    index.insert(1, &[1.0, 0.0]).unwrap();
    assert!(matches!(index.search(&[f32::NAN, 0.0], 1, None), Err(IndexError::Invalid(_))));
    assert!(matches!(index.search(&[1.0, 0.0], MAX_K + 1, None), Err(IndexError::Invalid(_))));
    assert_eq!(index.search(&[1.0, 0.0], MAX_K, None).unwrap().len(), 1);
}

pub fn case_zero_vector_scores_zero(new: Factory) {
    let mut index = new(3);
    index.insert(1, &[0.0, 0.0, 0.0]).unwrap();
    index.insert(2, &[1.0, 1.0, 0.0]).unwrap();
    let got = index.search(&[0.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(got, vec![Neighbor { id: 1, score: 0.0 }, Neighbor { id: 2, score: 0.0 }]);
    let got = index.search(&[1.0, 0.0, 0.0], 2, None).unwrap();
    assert_eq!(got[1], Neighbor { id: 1, score: 0.0 });
}

pub fn case_large_components_do_not_overflow(new: Factory) {
    let mut index = new(2);
    index.insert(1, &[1e20, 0.0]).unwrap();
    index.insert(2, &[1.0, 1.0]).unwrap();
    let got = index.search(&[1.0, 0.0], 1, None).unwrap();
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].id, 1);
    assert!((got[0].score - 1.0).abs() < 1e-6, "{:?}", got[0]);
    let got = index.search(&[3e38, 3e38], 2, None).unwrap();
    assert!(got.iter().all(|n| n.score.is_finite()), "{got:?}");
    assert_eq!(got[0].id, 2);
}

pub fn contract(new: Factory) {
    case_min_k_and_n_sorted(new);
    case_allow_excludes(new);
    case_insert_same_id_replaces(new);
    case_remove_never_returned(new);
    case_dimension_mismatch(new);
    case_rejects_nan_and_big_k(new);
    case_zero_vector_scores_zero(new);
    case_large_components_do_not_overflow(new);
}

#[test]
fn new_index_rejects_zero_and_oversized_dims() {
    for kind in [IndexKind::BruteForce, IndexKind::Hnsw(HnswParams::default())] {
        for dims in [0, MAX_DIMS + 1] {
            let err = new_index(&kind, dims, Quantization::F32).err().expect("dims out of range must fail");
            assert!(matches!(err, IndexError::Invalid(_)), "dims {dims}: {err:?}");
        }
    }
}

#[test]
fn trait_is_object_safe() {
    fn takes(_: Option<Box<dyn VectorIndex>>) {}
    takes(None);
}

#[test]
fn defaults_are_the_safe_values() {
    assert_eq!(IndexKind::default(), IndexKind::BruteForce);
    assert_eq!(Quantization::default(), Quantization::F32);
    let p = HnswParams::default();
    assert_eq!(p.m(), 16);
    assert_eq!(p.ef_construction(), 200);
    assert_eq!(p.ef_search(), 64);
    assert_eq!(p.seed(), 0x5452_4B54);
    assert_eq!(p.max_deleted_fraction(), 0.25);
}

#[test]
fn returns_min_k_and_n_sorted_by_score_then_id() {
    case_min_k_and_n_sorted(brute);
}

#[test]
fn allow_predicate_excludes_ids() {
    case_allow_excludes(brute);
}

#[test]
fn insert_same_id_replaces() {
    case_insert_same_id_replaces(brute);
}

#[test]
fn remove_then_search_never_returns_the_id() {
    case_remove_never_returned(brute);
}

#[test]
fn dimension_mismatch_on_insert_and_search() {
    case_dimension_mismatch(brute);
}

#[test]
fn rejects_nan_and_k_above_max() {
    case_rejects_nan_and_big_k(brute);
}

#[test]
fn zero_vector_scores_zero() {
    case_zero_vector_scores_zero(brute);
}

#[test]
fn brute_force_restore_takes_vectors_and_refuses_pages() {
    let mut index = brute(2);
    assert!(index.take_changes().is_empty());
    let mut vectors = vec![(1u64, vec![1.0, 0.0]), (2, vec![0.0, 1.0])].into_iter();
    index.restore(&mut vectors, &mut std::iter::empty()).unwrap();
    assert_eq!(index.len(), 2);
    let mut fresh = brute(2);
    let mut pages = vec![(troupekit_index::PageKey::Header, vec![1u8])].into_iter();
    let err = fresh.restore(&mut std::iter::empty(), &mut pages).unwrap_err();
    assert!(matches!(err, IndexError::CorruptPage(_)));
    assert_eq!(fresh.len(), 0);
}

#[test]
fn brute_force_contract() {
    contract(brute);
}

#[test]
fn large_finite_components_do_not_overflow() {
    case_large_components_do_not_overflow(brute);
}
