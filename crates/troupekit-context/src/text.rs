/// Chars counted as one model token by `estimate_tokens` and the chunker.
/// A rough average for English and most Latin-script text; it underestimates
/// CJK text, where one char is often a whole token.
pub const CHARS_PER_TOKEN: usize = 4;

const FNV_OFFSET: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Estimates the model tokens in `text` as one token per four chars.
/// Counts Unicode scalar values, not bytes, and rounds up.
/// Returns 0 for empty text.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(CHARS_PER_TOKEN)
}

/// Splits `text` into lowercase terms for keyword scoring and hashing.
/// A term is a maximal run of Unicode alphanumeric chars.
/// Returns the terms in text order, duplicates kept.
pub fn terms(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_alphanumeric()).filter(|t| !t.is_empty()).map(str::to_lowercase).collect()
}

/// FNV-1a 64 hash of `parts`, with a 0xff byte before every part after the first.
/// Stable across Rust releases and platforms, unlike `DefaultHasher`,
/// so it is safe for stored ids. Returns the 64-bit hash.
pub fn stable_hash(parts: &[&str]) -> u64 {
    let mut h = FNV_OFFSET;
    for (i, part) in parts.iter().enumerate() {
        let sep: &[u8] = if i == 0 { &[] } else { &[0xff] };
        for b in sep.iter().chain(part.as_bytes()) {
            h ^= u64::from(*b);
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimate_counts_chars_not_bytes() {
        assert_eq!(estimate_tokens(""), 0);
        assert_eq!(estimate_tokens("abcd"), 1);
        assert_eq!(estimate_tokens("abcde"), 2);
        assert_eq!(estimate_tokens("ééééé"), 2);
    }

    #[test]
    fn terms_lowercase_unicode_runs() {
        assert_eq!(terms("Protein, 120g! Café-au-lait"), vec!["protein", "120g", "café", "au", "lait"]);
        assert!(terms("  ...  ").is_empty());
    }

    #[test]
    fn stable_hash_is_fixed_and_separates_parts() {
        assert_eq!(stable_hash(&["a"]), stable_hash(&["a"]));
        assert_ne!(stable_hash(&["ab", "c"]), stable_hash(&["a", "bc"]));
        assert_eq!(stable_hash(&[]), 0xcbf29ce484222325);
    }
}
