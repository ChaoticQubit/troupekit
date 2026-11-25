/// Seeded xorshift64* generator, so HNSW builds are reproducible.
/// The same seed gives the same sequence on every platform.
/// A zero seed is replaced by a fixed non-zero constant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Rng(u64);

const ZERO_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Highest HNSW layer a node can get; fits the page's `u8` level.
/// Levels above it are clamped, which only flattens the rare tall node.
/// With `m = 2` the chance of passing it is `2^-16` per node.
pub(crate) const MAX_LEVEL: u8 = 16;

impl Rng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(if seed == 0 { ZERO_SEED } else { seed })
    }

    pub(crate) fn from_state(state: u64) -> Option<Self> {
        (state != 0).then_some(Self(state))
    }

    pub(crate) fn state(&self) -> u64 {
        self.0
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        assert!(self.0 != 0, "xorshift state reached zero");
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub(crate) fn unit_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }

    pub(crate) fn level(&mut self, m: usize) -> u8 {
        assert!(m >= 2, "hnsw m below 2");
        let u = self.unit_open();
        assert!(u > 0.0 && u <= 1.0, "unit draw out of (0, 1]");
        let level = (-u.ln() / (m as f64).ln()).floor();
        if level >= f64::from(MAX_LEVEL) {
            MAX_LEVEL
        } else {
            level as u8
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levels_are_bounded_and_mostly_zero() {
        let mut rng = Rng::new(1);
        let mut zero = 0;
        for _ in 0..10_000 {
            let l = rng.level(16);
            assert!(l <= MAX_LEVEL);
            if l == 0 {
                zero += 1;
            }
        }
        assert!(zero > 9_000, "{zero} of 10000 at level 0");
        assert_eq!(Rng::new(0).state(), ZERO_SEED);
        assert_eq!(Rng::from_state(0), None);
    }
}
