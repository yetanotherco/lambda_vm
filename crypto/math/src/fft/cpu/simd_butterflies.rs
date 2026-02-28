//! SIMD butterfly kernels for Goldilocks FFT.
//!
//! These operate on `&mut [u64]` slices (reinterpreted from `&mut [FieldElement<GoldilocksField>]`
//! via `#[repr(transparent)]`). Called from the TypeId dispatch in `bowers_fft.rs`.

use crate::field::fields::fft_friendly::u64_goldilocks::GoldilocksField;
use crate::field::traits::IsField;

// ==================== Scalar helpers ====================

#[inline(always)]
fn scalar_dif_butterfly(data: &mut [u64], j: usize, half_block: usize, tw: u64) {
    let a = data[j];
    let b = data[j + half_block];
    let sum = GoldilocksField::add(&a, &b);
    let diff = GoldilocksField::sub(&a, &b);
    data[j] = sum;
    data[j + half_block] = GoldilocksField::mul(&tw, &diff);
}

#[inline(always)]
fn scalar_dit_butterfly(data: &mut [u64], j: usize, half_block: usize, tw: u64) {
    let a = data[j];
    let b = data[j + half_block];
    let bw = GoldilocksField::mul(&tw, &b);
    data[j] = GoldilocksField::add(&a, &bw);
    data[j + half_block] = GoldilocksField::sub(&a, &bw);
}

// ==================== NEON (aarch64) ====================

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use crate::field::fields::fft_friendly::goldilocks_neon::PackedGoldilocks2;

    /// DIF single-layer butterfly on a block. `data` has length `2 * half_block`.
    /// `twiddles` has length `>= half_block`.
    pub fn dif_butterfly(data: &mut [u64], twiddles: &[u64], half_block: usize) {
        let simd_count = half_block / 2;
        let scalar_start = simd_count * 2;

        for i in 0..simd_count {
            let j = i * 2;
            let a = PackedGoldilocks2::load(&data[j..]);
            let b = PackedGoldilocks2::load(&data[j + half_block..]);
            let w = PackedGoldilocks2::load(&twiddles[j..]);

            let sum = a + b;
            let diff = a - b;
            let diff_w = w * diff;

            sum.store(&mut data[j..]);
            diff_w.store(&mut data[j + half_block..]);
        }

        for (j, &tw) in (scalar_start..).zip(&twiddles[scalar_start..half_block]) {
            scalar_dif_butterfly(data, j, half_block, tw);
        }
    }

    /// DIF fused 2-layer butterfly on a block of size `4 * quarter`.
    pub fn dif_fused_butterfly(block: &mut [u64], tw0: &[u64], tw1: &[u64]) {
        let block_size = block.len();
        let quarter = block_size / 4;
        let simd_count = quarter / 2;
        let scalar_start = simd_count * 2;

        for i in 0..simd_count {
            let j = i * 2;
            let i0 = j;
            let i1 = j + quarter;
            let i2 = j + 2 * quarter;
            let i3 = j + 3 * quarter;

            let a0 = PackedGoldilocks2::load(&block[i0..]);
            let a1 = PackedGoldilocks2::load(&block[i1..]);
            let a2 = PackedGoldilocks2::load(&block[i2..]);
            let a3 = PackedGoldilocks2::load(&block[i3..]);

            let w0 = PackedGoldilocks2::load(&tw0[j..]);
            let w1 = PackedGoldilocks2::load(&tw0[j + quarter..]);
            let w2 = PackedGoldilocks2::load(&tw1[j..]);

            // First layer
            let sum_02 = a0 + a2;
            let diff_02 = a0 - a2;
            let diff_02_w = w0 * diff_02;

            let sum_13 = a1 + a3;
            let diff_13 = a1 - a3;
            let diff_13_w = w1 * diff_13;

            // Second layer
            let final_0 = sum_02 + sum_13;
            let diff_sums = sum_02 - sum_13;
            let final_1 = w2 * diff_sums;

            let final_2 = diff_02_w + diff_13_w;
            let diff_diffs = diff_02_w - diff_13_w;
            let final_3 = w2 * diff_diffs;

            final_0.store(&mut block[i0..]);
            final_1.store(&mut block[i1..]);
            final_2.store(&mut block[i2..]);
            final_3.store(&mut block[i3..]);
        }

        // Scalar tail
        for j in scalar_start..quarter {
            let i0 = j;
            let i1 = j + quarter;
            let i2 = j + 2 * quarter;
            let i3 = j + 3 * quarter;

            let (w0, w1, w2) = (tw0[j], tw0[j + quarter], tw1[j]);

            let sum_02 = GoldilocksField::add(&block[i0], &block[i2]);
            let diff_02 = GoldilocksField::sub(&block[i0], &block[i2]);
            let diff_02_w = GoldilocksField::mul(&w0, &diff_02);

            let sum_13 = GoldilocksField::add(&block[i1], &block[i3]);
            let diff_13 = GoldilocksField::sub(&block[i1], &block[i3]);
            let diff_13_w = GoldilocksField::mul(&w1, &diff_13);

            let final_0 = GoldilocksField::add(&sum_02, &sum_13);
            let diff_sums = GoldilocksField::sub(&sum_02, &sum_13);
            let final_1 = GoldilocksField::mul(&w2, &diff_sums);

            let final_2 = GoldilocksField::add(&diff_02_w, &diff_13_w);
            let diff_diffs = GoldilocksField::sub(&diff_02_w, &diff_13_w);
            let final_3 = GoldilocksField::mul(&w2, &diff_diffs);

            block[i0] = final_0;
            block[i1] = final_1;
            block[i2] = final_2;
            block[i3] = final_3;
        }
    }

    /// DIT single-layer butterfly (inverse FFT).
    pub fn dit_butterfly(data: &mut [u64], twiddles: &[u64], half_block: usize) {
        let simd_count = half_block / 2;
        let scalar_start = simd_count * 2;

        for i in 0..simd_count {
            let j = i * 2;
            let a = PackedGoldilocks2::load(&data[j..]);
            let b = PackedGoldilocks2::load(&data[j + half_block..]);
            let w = PackedGoldilocks2::load(&twiddles[j..]);

            let bw = w * b;
            let sum = a + bw;
            let diff = a - bw;

            sum.store(&mut data[j..]);
            diff.store(&mut data[j + half_block..]);
        }

        for (j, &tw) in (scalar_start..).zip(&twiddles[scalar_start..half_block]) {
            scalar_dit_butterfly(data, j, half_block, tw);
        }
    }
}

// ==================== AVX2 (x86_64) ====================

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use crate::field::fields::fft_friendly::goldilocks_avx2::PackedGoldilocks4;

    /// DIF single-layer butterfly (AVX2).
    #[target_feature(enable = "avx2")]
    pub unsafe fn dif_butterfly(data: &mut [u64], twiddles: &[u64], half_block: usize) {
        let simd_count = half_block / 4;
        let scalar_start = simd_count * 4;

        for i in 0..simd_count {
            let j = i * 4;
            let a = PackedGoldilocks4::load(&data[j..]);
            let b = PackedGoldilocks4::load(&data[j + half_block..]);
            let w = PackedGoldilocks4::load(&twiddles[j..]);

            let sum = a + b;
            let diff = a - b;
            let diff_w = w * diff;

            sum.store(&mut data[j..]);
            diff_w.store(&mut data[j + half_block..]);
        }

        for (j, &tw) in (scalar_start..).zip(&twiddles[scalar_start..half_block]) {
            scalar_dif_butterfly(data, j, half_block, tw);
        }
    }

    /// DIF fused 2-layer butterfly (AVX2).
    #[target_feature(enable = "avx2")]
    pub unsafe fn dif_fused_butterfly(block: &mut [u64], tw0: &[u64], tw1: &[u64]) {
        let block_size = block.len();
        let quarter = block_size / 4;
        let simd_count = quarter / 4;
        let scalar_start = simd_count * 4;

        for i in 0..simd_count {
            let j = i * 4;
            let i0 = j;
            let i1 = j + quarter;
            let i2 = j + 2 * quarter;
            let i3 = j + 3 * quarter;

            let a0 = PackedGoldilocks4::load(&block[i0..]);
            let a1 = PackedGoldilocks4::load(&block[i1..]);
            let a2 = PackedGoldilocks4::load(&block[i2..]);
            let a3 = PackedGoldilocks4::load(&block[i3..]);

            let w0 = PackedGoldilocks4::load(&tw0[j..]);
            let w1 = PackedGoldilocks4::load(&tw0[j + quarter..]);
            let w2 = PackedGoldilocks4::load(&tw1[j..]);

            let sum_02 = a0 + a2;
            let diff_02 = a0 - a2;
            let diff_02_w = w0 * diff_02;

            let sum_13 = a1 + a3;
            let diff_13 = a1 - a3;
            let diff_13_w = w1 * diff_13;

            let final_0 = sum_02 + sum_13;
            let diff_sums = sum_02 - sum_13;
            let final_1 = w2 * diff_sums;

            let final_2 = diff_02_w + diff_13_w;
            let diff_diffs = diff_02_w - diff_13_w;
            let final_3 = w2 * diff_diffs;

            final_0.store(&mut block[i0..]);
            final_1.store(&mut block[i1..]);
            final_2.store(&mut block[i2..]);
            final_3.store(&mut block[i3..]);
        }

        // Scalar tail
        for j in scalar_start..quarter {
            let i0 = j;
            let i1 = j + quarter;
            let i2 = j + 2 * quarter;
            let i3 = j + 3 * quarter;

            let (w0, w1, w2) = (tw0[j], tw0[j + quarter], tw1[j]);

            let sum_02 = GoldilocksField::add(&block[i0], &block[i2]);
            let diff_02 = GoldilocksField::sub(&block[i0], &block[i2]);
            let diff_02_w = GoldilocksField::mul(&w0, &diff_02);

            let sum_13 = GoldilocksField::add(&block[i1], &block[i3]);
            let diff_13 = GoldilocksField::sub(&block[i1], &block[i3]);
            let diff_13_w = GoldilocksField::mul(&w1, &diff_13);

            let final_0 = GoldilocksField::add(&sum_02, &sum_13);
            let diff_sums = GoldilocksField::sub(&sum_02, &sum_13);
            let final_1 = GoldilocksField::mul(&w2, &diff_sums);

            let final_2 = GoldilocksField::add(&diff_02_w, &diff_13_w);
            let diff_diffs = GoldilocksField::sub(&diff_02_w, &diff_13_w);
            let final_3 = GoldilocksField::mul(&w2, &diff_diffs);

            block[i0] = final_0;
            block[i1] = final_1;
            block[i2] = final_2;
            block[i3] = final_3;
        }
    }

    /// DIT single-layer butterfly (inverse FFT, AVX2).
    #[target_feature(enable = "avx2")]
    pub unsafe fn dit_butterfly(data: &mut [u64], twiddles: &[u64], half_block: usize) {
        let simd_count = half_block / 4;
        let scalar_start = simd_count * 4;

        for i in 0..simd_count {
            let j = i * 4;
            let a = PackedGoldilocks4::load(&data[j..]);
            let b = PackedGoldilocks4::load(&data[j + half_block..]);
            let w = PackedGoldilocks4::load(&twiddles[j..]);

            let bw = w * b;
            let sum = a + bw;
            let diff = a - bw;

            sum.store(&mut data[j..]);
            diff.store(&mut data[j + half_block..]);
        }

        for (j, &tw) in (scalar_start..).zip(&twiddles[scalar_start..half_block]) {
            scalar_dit_butterfly(data, j, half_block, tw);
        }
    }
}

// ==================== Public dispatch ====================

/// Minimum `half_block` size for NEON dispatch (2 elements = 1 SIMD iteration).
#[cfg(target_arch = "aarch64")]
pub const NEON_MIN_HALF_BLOCK: usize = 2;

/// Minimum `half_block` size for AVX2 dispatch (4 elements = 1 SIMD iteration).
#[cfg(target_arch = "x86_64")]
pub const AVX2_MIN_HALF_BLOCK: usize = 4;

// --- NEON public API ---

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dif_butterfly_neon(data: &mut [u64], twiddles: &[u64], half_block: usize) {
    neon::dif_butterfly(data, twiddles, half_block);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dif_fused_butterfly_neon(block: &mut [u64], tw0: &[u64], tw1: &[u64]) {
    neon::dif_fused_butterfly(block, tw0, tw1);
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn dit_butterfly_neon(data: &mut [u64], twiddles: &[u64], half_block: usize) {
    neon::dit_butterfly(data, twiddles, half_block);
}

// --- AVX2 public API ---

#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dif_butterfly_avx2(data: &mut [u64], twiddles: &[u64], half_block: usize) {
    unsafe { avx2::dif_butterfly(data, twiddles, half_block) };
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dif_fused_butterfly_avx2(block: &mut [u64], tw0: &[u64], tw1: &[u64]) {
    unsafe { avx2::dif_fused_butterfly(block, tw0, tw1) };
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub unsafe fn dit_butterfly_avx2(data: &mut [u64], twiddles: &[u64], half_block: usize) {
    unsafe { avx2::dit_butterfly(data, twiddles, half_block) };
}
