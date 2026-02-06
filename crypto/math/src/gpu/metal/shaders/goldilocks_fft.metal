//! Metal shader for Goldilocks field FFT operations
//!
//! Implements field arithmetic and FFT butterflies for the Goldilocks prime field
//! p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001
//!
//! Key optimizations:
//! - Uses EPSILON = 2^32 - 1 for fast reduction (since 2^64 ≡ EPSILON mod p)
//! - Avoids branches where possible for better GPU utilization
//! - Threadgroup memory for intermediate values in Bowers fusion

#include <metal_stdlib>
using namespace metal;

// ============================================================================
// Goldilocks Field Constants
// ============================================================================

constant ulong GOLDILOCKS_PRIME = 0xFFFFFFFF00000001UL;
constant ulong EPSILON = 0xFFFFFFFFUL;  // 2^32 - 1 = -2^64 mod p

// ============================================================================
// Goldilocks Field Arithmetic
// ============================================================================

/// Canonicalize a field element to [0, p)
inline ulong goldilocks_canonicalize(ulong x) {
    // If x >= p, subtract p
    return (x >= GOLDILOCKS_PRIME) ? (x - GOLDILOCKS_PRIME) : x;
}

/// Addition with overflow handling
/// If a + b overflows, we add EPSILON (since 2^64 ≡ EPSILON mod p)
inline ulong goldilocks_add(ulong a, ulong b) {
    ulong sum = a + b;
    // Check for overflow: sum < a means overflow occurred
    bool overflow = sum < a;
    ulong carry = overflow ? EPSILON : 0UL;
    ulong result = sum + carry;
    // Second overflow is rare but possible
    bool overflow2 = result < sum;
    return overflow2 ? (result + EPSILON) : result;
}

/// Subtraction with underflow handling
inline ulong goldilocks_sub(ulong a, ulong b) {
    ulong diff = a - b;
    // Check for underflow: a < b means underflow occurred
    bool underflow = a < b;
    ulong borrow = underflow ? EPSILON : 0UL;
    ulong result = diff - borrow;
    // Second underflow is rare but possible
    bool underflow2 = diff < borrow;
    return underflow2 ? (result - EPSILON) : result;
}

/// Reduce a 128-bit value to 64-bit Goldilocks field element
/// Uses: 2^64 ≡ EPSILON (mod p) and 2^96 ≡ -1 (mod p)
inline ulong goldilocks_reduce128(ulong lo, ulong hi) {
    ulong hi_hi = hi >> 32;
    ulong hi_lo = hi & EPSILON;

    // Step 1: t0 = lo - hi_hi
    ulong t0 = lo - hi_hi;
    bool borrow = lo < hi_hi;
    t0 = borrow ? (t0 - EPSILON) : t0;

    // Step 2: t1 = hi_lo * EPSILON = (hi_lo << 32) - hi_lo
    ulong t1 = (hi_lo << 32) - hi_lo;

    // Step 3: result = t0 + t1
    ulong result = t0 + t1;
    bool carry = result < t0;
    return carry ? (result + EPSILON) : result;
}

/// Multiplication using 128-bit intermediate
inline ulong goldilocks_mul(ulong a, ulong b) {
    // Metal doesn't have native 128-bit multiply, so we split into 32-bit parts
    ulong a_lo = a & 0xFFFFFFFFUL;
    ulong a_hi = a >> 32;
    ulong b_lo = b & 0xFFFFFFFFUL;
    ulong b_hi = b >> 32;

    ulong p0 = a_lo * b_lo;  // Low 64 bits of low product
    ulong p1 = a_lo * b_hi;
    ulong p2 = a_hi * b_lo;
    ulong p3 = a_hi * b_hi;

    // Combine: result = p0 + (p1 + p2) << 32 + p3 << 64
    ulong mid = p1 + p2;
    ulong mid_carry = (mid < p1) ? 1UL : 0UL;

    ulong lo = p0 + (mid << 32);
    bool lo_carry = lo < p0;

    ulong hi = p3 + (mid >> 32) + (mid_carry << 32) + (lo_carry ? 1UL : 0UL);

    return goldilocks_reduce128(lo, hi);
}

/// Negation: -a = p - a (or 0 if a = 0)
inline ulong goldilocks_neg(ulong a) {
    ulong canonical = goldilocks_canonicalize(a);
    return (canonical == 0UL) ? 0UL : (GOLDILOCKS_PRIME - canonical);
}

// ============================================================================
// FFT Kernels
// ============================================================================

/// Radix-2 DIT butterfly kernel
/// Performs one stage of the Cooley-Tukey FFT
kernel void radix2_dit_butterfly(
    device ulong* input [[buffer(0)]],
    device const ulong* twiddles [[buffer(1)]],
    constant uint& stage [[buffer(2)]],
    constant uint& butterfly_count [[buffer(3)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    if (thread_pos >= butterfly_count) return;

    uint half_group_size = butterfly_count >> stage;
    uint group = thread_pos / half_group_size;
    uint pos_in_group = thread_pos & (half_group_size - 1);
    uint i = thread_pos * 2 - pos_in_group;

    ulong w = twiddles[group];
    ulong a = input[i];
    ulong b = input[i + half_group_size];

    // Butterfly: (a, b) -> (a + w*b, a - w*b)
    ulong wb = goldilocks_mul(w, b);
    ulong res1 = goldilocks_add(a, wb);
    ulong res2 = goldilocks_sub(a, wb);

    input[i] = res1;
    input[i + half_group_size] = res2;
}

/// Bowers FFT with 2-layer fusion
/// Processes two layers at once to reduce memory traffic
kernel void bowers_fft_fused_layer(
    device ulong* input [[buffer(0)]],
    device const ulong* twiddles_l0 [[buffer(1)]],
    device const ulong* twiddles_l1 [[buffer(2)]],
    constant uint& block_size [[buffer(3)]],
    constant uint& n [[buffer(4)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    uint quarter = block_size >> 2;
    if (thread_pos >= n / 4) return;

    uint block_idx = thread_pos / quarter;
    uint j = thread_pos % quarter;
    uint block_start = block_idx * block_size;

    uint i0 = block_start + j;
    uint i1 = block_start + j + quarter;
    uint i2 = block_start + j + 2 * quarter;
    uint i3 = block_start + j + 3 * quarter;

    // Load twiddle factors
    ulong w0 = twiddles_l0[j];
    ulong w1 = twiddles_l0[j + quarter];
    ulong w2 = twiddles_l1[j];

    // Load input values
    ulong v0 = input[i0];
    ulong v1 = input[i1];
    ulong v2 = input[i2];
    ulong v3 = input[i3];

    // First layer butterflies
    ulong sum_02 = goldilocks_add(v0, v2);
    ulong diff_02 = goldilocks_sub(v0, v2);
    ulong diff_02_w = goldilocks_mul(w0, diff_02);

    ulong sum_13 = goldilocks_add(v1, v3);
    ulong diff_13 = goldilocks_sub(v1, v3);
    ulong diff_13_w = goldilocks_mul(w1, diff_13);

    // Second layer butterflies
    ulong final_0 = goldilocks_add(sum_02, sum_13);
    ulong diff_sums = goldilocks_sub(sum_02, sum_13);
    ulong final_1 = goldilocks_mul(w2, diff_sums);

    ulong final_2 = goldilocks_add(diff_02_w, diff_13_w);
    ulong diff_diffs = goldilocks_sub(diff_02_w, diff_13_w);
    ulong final_3 = goldilocks_mul(w2, diff_diffs);

    // Store results
    input[i0] = final_0;
    input[i1] = final_1;
    input[i2] = final_2;
    input[i3] = final_3;
}

/// Single layer butterfly for odd layers (when layer count is odd)
kernel void bowers_fft_single_layer(
    device ulong* input [[buffer(0)]],
    device const ulong* twiddles [[buffer(1)]],
    constant uint& block_size [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    uint half_block = block_size >> 1;
    if (thread_pos >= n / 2) return;

    uint block_idx = thread_pos / half_block;
    uint j = thread_pos % half_block;
    uint block_start = block_idx * block_size;

    uint i0 = block_start + j;
    uint i1 = i0 + half_block;

    ulong w = twiddles[j];
    ulong a = input[i0];
    ulong b = input[i1];

    ulong sum = goldilocks_add(a, b);
    ulong diff = goldilocks_sub(a, b);
    ulong diff_w = goldilocks_mul(w, diff);

    input[i0] = sum;
    input[i1] = diff_w;
}

/// Bit-reversal permutation kernel
/// Reverses the bit order of indices to complete the FFT
kernel void bitrev_permutation(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint& len [[buffer(2)]],
    constant uint& log_len [[buffer(3)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    if (thread_pos >= len) return;

    // Compute bit-reversed index
    uint rev = 0;
    uint val = thread_pos;
    for (uint i = 0; i < log_len; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }

    output[rev] = input[thread_pos];
}

/// In-place bit-reversal permutation (only swap if rev > thread_pos)
kernel void bitrev_permutation_inplace(
    device ulong* data [[buffer(0)]],
    constant uint& len [[buffer(1)]],
    constant uint& log_len [[buffer(2)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    if (thread_pos >= len) return;

    // Compute bit-reversed index
    uint rev = 0;
    uint val = thread_pos;
    for (uint i = 0; i < log_len; i++) {
        rev = (rev << 1) | (val & 1);
        val >>= 1;
    }

    // Only swap if rev > thread_pos to avoid double-swapping
    if (rev > thread_pos) {
        ulong tmp = data[thread_pos];
        data[thread_pos] = data[rev];
        data[rev] = tmp;
    }
}

/// Calculate twiddle factors
/// omega should be the primitive n-th root of unity
kernel void calc_twiddles(
    device ulong* result [[buffer(0)]],
    constant ulong& omega [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    if (thread_pos >= count) return;

    // Compute omega^thread_pos by repeated squaring
    ulong base = omega;
    ulong exp = thread_pos;
    ulong acc = 1UL;

    while (exp > 0) {
        if (exp & 1) {
            acc = goldilocks_mul(acc, base);
        }
        base = goldilocks_mul(base, base);
        exp >>= 1;
    }

    result[thread_pos] = acc;
}

/// Calculate LayerTwiddles for Bowers FFT
/// Generates twiddles organized by layer for cache-friendly access
kernel void calc_layer_twiddles(
    device ulong* result [[buffer(0)]],
    constant ulong& root [[buffer(1)]],
    constant uint& layer [[buffer(2)]],
    constant uint& count [[buffer(3)]],
    uint thread_pos [[thread_position_in_grid]]
) {
    if (thread_pos >= count) return;

    // For layer k, twiddle j = root^(j * 2^k)
    uint stride = 1u << layer;
    ulong exp_val = (ulong)thread_pos * stride;

    // Compute root^exp_val
    ulong base = root;
    ulong exp = exp_val;
    ulong acc = 1UL;

    while (exp > 0) {
        if (exp & 1) {
            acc = goldilocks_mul(acc, base);
        }
        base = goldilocks_mul(base, base);
        exp >>= 1;
    }

    result[thread_pos] = acc;
}

// ============================================================================
// Batch FFT Kernels (for SoA layout)
// ============================================================================

/// Batch Bowers FFT fused layer for multiple polynomials (SoA layout)
kernel void batch_bowers_fft_fused_layer(
    device ulong* data [[buffer(0)]],
    device const ulong* twiddles_l0 [[buffer(1)]],
    device const ulong* twiddles_l1 [[buffer(2)]],
    constant uint& poly_len [[buffer(3)]],
    constant uint& num_polys [[buffer(4)]],
    constant uint& block_size [[buffer(5)]],
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint poly_idx = thread_pos.y;
    uint within_poly = thread_pos.x;

    if (poly_idx >= num_polys) return;

    uint quarter = block_size >> 2;
    uint elements_per_poly = poly_len / 4;
    if (within_poly >= elements_per_poly) return;

    uint block_idx = within_poly / quarter;
    uint j = within_poly % quarter;
    uint block_start = block_idx * block_size;

    // Offset into data for this polynomial (SoA layout)
    uint base_offset = poly_idx * poly_len;

    uint i0 = base_offset + block_start + j;
    uint i1 = base_offset + block_start + j + quarter;
    uint i2 = base_offset + block_start + j + 2 * quarter;
    uint i3 = base_offset + block_start + j + 3 * quarter;

    // Load twiddle factors (shared across all polynomials)
    ulong w0 = twiddles_l0[j];
    ulong w1 = twiddles_l0[j + quarter];
    ulong w2 = twiddles_l1[j];

    // Load input values
    ulong v0 = data[i0];
    ulong v1 = data[i1];
    ulong v2 = data[i2];
    ulong v3 = data[i3];

    // First layer butterflies
    ulong sum_02 = goldilocks_add(v0, v2);
    ulong diff_02 = goldilocks_sub(v0, v2);
    ulong diff_02_w = goldilocks_mul(w0, diff_02);

    ulong sum_13 = goldilocks_add(v1, v3);
    ulong diff_13 = goldilocks_sub(v1, v3);
    ulong diff_13_w = goldilocks_mul(w1, diff_13);

    // Second layer butterflies
    ulong final_0 = goldilocks_add(sum_02, sum_13);
    ulong diff_sums = goldilocks_sub(sum_02, sum_13);
    ulong final_1 = goldilocks_mul(w2, diff_sums);

    ulong final_2 = goldilocks_add(diff_02_w, diff_13_w);
    ulong diff_diffs = goldilocks_sub(diff_02_w, diff_13_w);
    ulong final_3 = goldilocks_mul(w2, diff_diffs);

    // Store results
    data[i0] = final_0;
    data[i1] = final_1;
    data[i2] = final_2;
    data[i3] = final_3;
}

/// Batch Bowers FFT single layer for multiple polynomials (SoA layout)
kernel void batch_bowers_fft_single_layer(
    device ulong* data [[buffer(0)]],
    device const ulong* twiddles [[buffer(1)]],
    constant uint& poly_len [[buffer(2)]],
    constant uint& num_polys [[buffer(3)]],
    constant uint& block_size [[buffer(4)]],
    uint2 thread_pos [[thread_position_in_grid]]
) {
    uint poly_idx = thread_pos.y;
    uint within_poly = thread_pos.x;

    if (poly_idx >= num_polys) return;

    uint half_block = block_size >> 1;
    uint elements_per_poly = poly_len / 2;
    if (within_poly >= elements_per_poly) return;

    uint block_idx = within_poly / half_block;
    uint j = within_poly % half_block;
    uint block_start = block_idx * block_size;

    uint base_offset = poly_idx * poly_len;

    uint i0 = base_offset + block_start + j;
    uint i1 = i0 + half_block;

    ulong w = twiddles[j];
    ulong a = data[i0];
    ulong b = data[i1];

    ulong sum = goldilocks_add(a, b);
    ulong diff = goldilocks_sub(a, b);
    ulong diff_w = goldilocks_mul(w, diff);

    data[i0] = sum;
    data[i1] = diff_w;
}
