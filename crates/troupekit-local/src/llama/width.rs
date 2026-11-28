/// Checks the embedding width of a model loaded with embeddings on.
/// Each output row is read as `n_embd_out` floats, so a model whose
/// `n_embd` differs (an output projection) is refused, as is a width of 0.
/// Output: the width in floats, or the message naming both widths.
pub(crate) fn embedding_width(n_embd: i32, n_embd_out: i32) -> Result<usize, String> {
    if n_embd != n_embd_out || n_embd_out <= 0 {
        return Err(format!(
            "model embedding width {n_embd} differs from its output width {n_embd_out}: \
             troupekit embeds only with models where both match and are above 0"
        ));
    }
    let width = usize::try_from(n_embd_out).expect("checked above 0");
    assert!(width > 0);
    Ok(width)
}
