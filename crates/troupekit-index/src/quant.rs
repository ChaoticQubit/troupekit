use crate::Quantization;

const F16_MAX_FINITE: u16 = 0x7bff;

/// Converts an f32 to IEEE 754 binary16 bits, round to nearest even.
/// Input: any f32; callers pass finite values. Output: the f16 bits,
/// subnormals kept, values past 65504 and infinities clamped to 65504.
pub(crate) fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | F16_MAX_FINITE;
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | F16_MAX_FINITE;
    }
    if e <= 0 {
        let shift = (14 - e) as u32;
        if shift > 24 {
            return sign;
        }
        let full = man | 0x0080_0000;
        let half = 1u32 << (shift - 1);
        let rem = full & ((1u32 << shift) - 1);
        let mut m = full >> shift;
        if rem > half || (rem == half && m & 1 == 1) {
            m += 1;
        }
        assert!(m <= 0x400, "subnormal f16 rounding overflow");
        return sign | m as u16;
    }
    let rem = man & 0x1fff;
    let mut h = ((e as u32) << 10) | (man >> 13);
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    if h >= 0x7c00 {
        return sign | F16_MAX_FINITE;
    }
    sign | h as u16
}

/// Converts IEEE 754 binary16 bits to the f32 of the same value.
/// Input: any f16 bits. Output: the exact f32; subnormals included,
/// infinities and NaNs map to their f32 counterparts.
pub(crate) fn f16_to_f32(h: u16) -> f32 {
    let negative = h & 0x8000 != 0;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x03ff);
    let sign = if negative { 0x8000_0000u32 } else { 0 };
    if exp == 0 {
        let v = man as f32 * (1.0 / 16_777_216.0);
        return if negative { -v } else { v };
    }
    if exp == 0x1f {
        return f32::from_bits(sign | 0x7f80_0000 | (man << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
}

/// Encodes one vector to symmetric i8 codes and returns its scale.
/// Input: a finite vector and the code buffer to append to.
/// Output: `scale = max_abs / 127`; a zero vector gets scale 0.0.
fn encode_i8(v: &[f32], codes: &mut Vec<i8>) -> f32 {
    let max_abs = v.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max_abs == 0.0 {
        codes.extend(std::iter::repeat_n(0i8, v.len()));
        return 0.0;
    }
    let scale = max_abs / 127.0;
    codes.extend(v.iter().map(|x| (x / scale).round().clamp(-127.0, 127.0) as i8));
    scale
}

/// Vector slots in one in-memory format, `dims` values per slot.
/// Slots are appended, overwritten, moved and truncated by index;
/// `decode` hands back a slot as f32 values.
#[derive(Debug, Clone)]
pub(crate) enum Slots {
    F32(Vec<f32>),
    F16(Vec<u16>),
    I8 { codes: Vec<i8>, scales: Vec<f32> },
}

impl Slots {
    pub(crate) fn new(quant: Quantization) -> Self {
        match quant {
            Quantization::F32 => Slots::F32(Vec::new()),
            Quantization::F16 => Slots::F16(Vec::new()),
            Quantization::I8 => Slots::I8 { codes: Vec::new(), scales: Vec::new() },
        }
    }

    pub(crate) fn bytes_per_vector(&self, dims: usize) -> usize {
        match self {
            Slots::F32(_) => 4 * dims,
            Slots::F16(_) => 2 * dims,
            Slots::I8 { .. } => dims + 4,
        }
    }

    pub(crate) fn count(&self, dims: usize) -> usize {
        match self {
            Slots::F32(d) => d.len() / dims,
            Slots::F16(d) => d.len() / dims,
            Slots::I8 { scales, .. } => scales.len(),
        }
    }

    pub(crate) fn push(&mut self, v: &[f32]) {
        match self {
            Slots::F32(d) => d.extend_from_slice(v),
            Slots::F16(d) => d.extend(v.iter().map(|&x| f32_to_f16(x))),
            Slots::I8 { codes, scales } => {
                let s = encode_i8(v, codes);
                scales.push(s);
            }
        }
    }

    pub(crate) fn set(&mut self, i: usize, v: &[f32]) {
        let dims = v.len();
        let range = i * dims..(i + 1) * dims;
        match self {
            Slots::F32(d) => d[range].copy_from_slice(v),
            Slots::F16(d) => {
                for (dst, &x) in d[range].iter_mut().zip(v) {
                    *dst = f32_to_f16(x);
                }
            }
            Slots::I8 { codes, scales } => {
                let mut tmp = Vec::with_capacity(dims);
                scales[i] = encode_i8(v, &mut tmp);
                codes[range].copy_from_slice(&tmp);
            }
        }
    }

    pub(crate) fn move_slot(&mut self, from: usize, to: usize, dims: usize) {
        let src = from * dims..(from + 1) * dims;
        let dst = to * dims;
        match self {
            Slots::F32(d) => d.copy_within(src, dst),
            Slots::F16(d) => d.copy_within(src, dst),
            Slots::I8 { codes, scales } => {
                codes.copy_within(src, dst);
                scales[to] = scales[from];
            }
        }
    }

    pub(crate) fn truncate(&mut self, n: usize, dims: usize) {
        match self {
            Slots::F32(d) => d.truncate(n * dims),
            Slots::F16(d) => d.truncate(n * dims),
            Slots::I8 { codes, scales } => {
                codes.truncate(n * dims);
                scales.truncate(n);
            }
        }
    }

    pub(crate) fn decode<'a>(&'a self, i: usize, dims: usize, buf: &'a mut Vec<f32>) -> &'a [f32] {
        let range = i * dims..(i + 1) * dims;
        match self {
            Slots::F32(d) => &d[range],
            Slots::F16(d) => {
                buf.clear();
                buf.extend(d[range].iter().map(|&h| f16_to_f32(h)));
                buf
            }
            Slots::I8 { codes, scales } => {
                let s = scales[i];
                buf.clear();
                buf.extend(codes[range].iter().map(|&c| f32::from(c) * s));
                buf
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f16_round_trip_matches_known_bit_patterns() {
        let table: [(f32, u16); 8] = [
            (0.0, 0x0000),
            (-0.0, 0x8000),
            (1.0, 0x3c00),
            (-2.0, 0xc000),
            (65504.0, 0x7bff),
            (5.960_464_5e-8, 0x0001),
            (6.103_515_6e-5, 0x0400),
            (0.333_251_95, 0x3555),
        ];
        for (x, bits) in table {
            assert_eq!(f32_to_f16(x), bits, "encode {x}");
            assert_eq!(f16_to_f32(bits).to_bits(), x.to_bits(), "decode {bits:#06x}");
        }
        assert_eq!(f32_to_f16(65520.0), 0x7bff);
        assert_eq!(f32_to_f16(1.0e9), 0x7bff);
        assert_eq!(f32_to_f16(-1.0e9), 0xfbff);
        assert_eq!(f32_to_f16(1.0e-9), 0x0000);
        assert_eq!(f32_to_f16(2.980_232_2e-8), 0x0000);
        assert_eq!(f32_to_f16(1.0 + 1.0 / 2048.0), 0x3c00);
        assert_eq!(f32_to_f16(1.0 + 3.0 / 2048.0), 0x3c02);
    }

    #[test]
    fn f16_error_is_within_half_ulp() {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        for _ in 0..100_000 {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            let exp = (r >> 59) as i32 % 30 - 14;
            let frac = 1.0 + (r & 0xff_ffff) as f32 / (1u32 << 24) as f32;
            let x = frac * 2f32.powi(exp) * if r & (1 << 30) == 0 { 1.0 } else { -1.0 };
            if x.abs() > 65504.0 {
                continue;
            }
            let back = f16_to_f32(f32_to_f16(x));
            assert!((back - x).abs() <= x.abs() * 2f32.powi(-11), "{x} came back as {back}");
        }
    }
}
