use crate::width_src::embedding_width;

#[test]
fn embedding_width_must_match_the_output_width() {
    assert_eq!(embedding_width(576, 576), Ok(576));
    let err = embedding_width(576, 1_024).unwrap_err();
    assert!(err.contains("576") && err.contains("1024"), "{err}");
    assert!(embedding_width(1_024, 576).is_err());
    assert!(embedding_width(0, 0).is_err());
    assert!(embedding_width(-1, -1).is_err());
}
