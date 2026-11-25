/// Seeded xorshift64* generator for reproducible example data.
/// The same seed gives the same vectors on every machine.
/// A zero seed is replaced by one.
pub struct Rng(u64);

impl Rng {
    /// Builds a generator from `seed`.
    /// Input: any `u64`; zero becomes one.
    /// Output: the generator.
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    /// Returns the next 64 random bits.
    /// Input: none. Output: a `u64`.
    /// The sequence repeats only after 2^64 - 1 draws.
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Returns one standard normal draw by the Box-Muller transform.
    /// Input: none. Output: an f32 with mean 0 and deviation 1.
    /// The first uniform is kept above zero so `ln` stays finite.
    pub fn gaussian(&mut self) -> f32 {
        let u1 = ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64;
        let u2 = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
    }
}

/// Scales `v` to unit length; a zero vector is returned unchanged.
/// Input: any vector. Output: the unit vector.
/// Cosine scores are then plain dot products.
pub fn normalize(v: Vec<f32>) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n == 0.0 {
        return v;
    }
    v.into_iter().map(|x| x / n).collect()
}

/// Draws `n` unit vectors around `clusters` Gaussian centres on the
/// unit sphere, with per-component noise of deviation `spread`.
/// Input: seed, count, dims, clusters, spread. Output: the vectors.
pub fn clustered(seed: u64, n: usize, dims: usize, clusters: usize, spread: f32) -> Vec<Vec<f32>> {
    assert!(clusters > 0 && dims > 0, "clusters and dims must be positive");
    let mut rng = Rng::new(seed);
    let centres: Vec<Vec<f32>> = (0..clusters).map(|_| normalize((0..dims).map(|_| rng.gaussian()).collect())).collect();
    (0..n)
        .map(|_| {
            let c = &centres[(rng.next_u64() % clusters as u64) as usize];
            normalize(c.iter().map(|x| x + spread * rng.gaussian()).collect())
        })
        .collect()
}

/// Reads the vector count from the first argument.
/// Input: the default when no argument is given. Output: the count,
/// or an exit with code 2 and a usage line for a bad argument.
pub fn count_arg(default: usize) -> usize {
    match std::env::args().nth(1) {
        None => default,
        Some(a) => match a.parse::<usize>() {
            Ok(n) if n > 0 => n,
            _ => {
                eprintln!("usage: <example> [vector count > 0], got {a:?}");
                std::process::exit(2);
            }
        },
    }
}
