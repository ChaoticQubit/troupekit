use troupekit_index::{new_index, HnswParams, IndexError, IndexKind, Quantization, VectorIndex, MAX_DIMS};

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
