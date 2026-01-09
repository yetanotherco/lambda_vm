//
// Goldilocks Field FFT Shaders for Metal
//
// Field: p = 2^64 - 2^32 + 1 (Goldilocks prime)
// Key properties:
//   - EPSILON = 2^32 - 1 = p - 2^64 (used for fast reduction)
//   - 2^64 ≡ EPSILON (mod p)
//   - 2^96 ≡ -1 (mod p)
//

#include <metal_stdlib>
using namespace metal;

// Goldilocks prime: p = 2^64 - 2^32 + 1
constant ulong GOLDILOCKS_PRIME = 0xFFFFFFFF00000001UL;

// EPSILON = 2^32 - 1 (key constant for fast reduction)
constant ulong EPSILON = 0xFFFFFFFFUL;

// =============================================================================
// Goldilocks Field Arithmetic
// =============================================================================

/// Reduce a value that might be >= p to canonical form [0, p)
inline ulong goldilocks_canonicalize(ulong x) {
    // Since values can be slightly above p after operations,
    // a single subtraction is usually sufficient
    return (x >= GOLDILOCKS_PRIME) ? (x - GOLDILOCKS_PRIME) : x;
}

/// Addition with overflow handling
/// If a + b overflows, we add EPSILON (since 2^64 ≡ EPSILON mod p)
inline ulong goldilocks_add(ulong a, ulong b) {
    ulong sum = a + b;
    // Check for overflow: if sum < a, overflow occurred
    bool overflow = (sum < a);

    // If overflow, add EPSILON
    if (overflow) {
        sum += EPSILON;
        // Second overflow is possible but rare
        if (sum < EPSILON) {
            sum += EPSILON;
        }
    }

    return sum;
}

/// Subtraction with underflow handling
/// If a - b underflows, we subtract EPSILON (since -2^64 ≡ -EPSILON mod p)
///
/// Mathematical justification:
/// - p = 2^64 - 2^32 + 1, so 2^64 = p + EPSILON where EPSILON = 2^32 - 1
/// - If a < b, then a - b wraps to (2^64 + a - b)
/// - We want (a - b) mod p = (2^64 + a - b) - 2^64 mod p
/// - Since 2^64 ≡ EPSILON (mod p), we subtract EPSILON from the wrapped result
inline ulong goldilocks_sub(ulong a, ulong b) {
    ulong diff = a - b;

    // Check for underflow: if b > a, underflow occurred
    if (b > a) {
        // First correction: subtract EPSILON
        ulong diff2 = diff - EPSILON;

        // Check if the subtraction itself underflowed
        // This happens when diff < EPSILON
        if (diff < EPSILON) {
            // Second underflow - subtract EPSILON again
            return diff2 - EPSILON;
        }
        return diff2;
    }

    return diff;
}

/// Multiplication using 128-bit intermediate and fast reduction
/// Uses the identity: 2^64 ≡ EPSILON (mod p) where EPSILON = 2^32 - 1
///
/// This is a direct port of the CPU reduce128 function.
inline ulong goldilocks_mul(ulong a, ulong b) {
    // Compute full 128-bit product
    ulong x_lo = a * b;
    ulong x_hi = mulhi(a, b);

    // Split x_hi into high and low 32-bit parts
    ulong x_hi_hi = x_hi >> 32;
    ulong x_hi_lo = x_hi & EPSILON;  // Note: EPSILON = 0xFFFFFFFF

    // Step 1: t0 = x_lo - x_hi_hi
    // (because 2^96 ≡ -1 mod p)
    ulong t0;
    bool borrow;
    if (x_hi_hi > x_lo) {
        // Underflow: need to wrap and subtract EPSILON
        t0 = x_lo - x_hi_hi;  // Wraps around
        t0 = t0 - EPSILON;    // Correct for the underflow (may wrap again)
        // Check if second subtraction also underflowed
        // This is extremely rare
    } else {
        t0 = x_lo - x_hi_hi;
    }

    // Step 2: t1 = x_hi_lo * EPSILON
    ulong t1 = x_hi_lo * EPSILON;

    // Step 3: result = t0 + t1
    ulong result = t0 + t1;
    if (result < t0) {
        // Overflow: add EPSILON
        result = result + EPSILON;
    }

    return result;
}

/// Negation: -a = p - a (or 0 if a ≡ 0)
inline ulong goldilocks_neg(ulong a) {
    ulong canonical = goldilocks_canonicalize(a);
    return (canonical == 0) ? 0 : (GOLDILOCKS_PRIME - canonical);
}

// =============================================================================
// FFT Butterfly Operations
// =============================================================================

/// Radix-2 butterfly: (a, b) -> (a + w*b, a - w*b)
/// This is the atomic operation of the Cooley-Tukey FFT
inline void butterfly_radix2(
    thread ulong& a,
    thread ulong& b,
    ulong twiddle
) {
    ulong wb = goldilocks_mul(twiddle, b);
    ulong new_a = goldilocks_add(a, wb);
    ulong new_b = goldilocks_sub(a, wb);
    a = new_a;
    b = new_b;
}

// =============================================================================
// FFT Kernels
// =============================================================================

/// Single stage of radix-2 NR DIT FFT
/// Each thread handles one butterfly operation
///
/// Parameters:
///   - data: Input/output buffer of field elements
///   - twiddles: Pre-computed twiddle factors (bit-reversed order)
///   - n: Total number of elements
///   - stage: Current FFT stage (0, 1, 2, ...)
///   - group_count: Number of groups in this stage (1, 2, 4, ...)
///   - group_size: Size of each group (n, n/2, n/4, ...)
kernel void fft_radix2_stage(
    device ulong* data [[buffer(0)]],
    device const ulong* twiddles [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& group_count [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one butterfly
    uint butterflies_per_group = group_size / 2;
    uint total_butterflies = group_count * butterflies_per_group;

    if (tid >= total_butterflies) {
        return;
    }

    // Determine which group and position within group
    uint group = tid / butterflies_per_group;
    uint j = tid % butterflies_per_group;

    // Calculate indices
    uint first_in_group = group * group_size;
    uint i = first_in_group + j;
    uint i_half = i + butterflies_per_group;

    // Get twiddle factor for this group
    ulong w = twiddles[group];

    // Load values
    ulong a = data[i];
    ulong b = data[i_half];

    // Perform butterfly
    butterfly_radix2(a, b, w);

    // Store results
    data[i] = a;
    data[i_half] = b;
}

/// Bit-reverse permutation kernel
/// Swaps elements at index i with element at bit_reverse(i)
kernel void bit_reverse_permute(
    device ulong* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    constant uint& log_n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }

    // Compute bit-reversed index
    uint rev = 0;
    uint temp = tid;
    for (uint i = 0; i < log_n; i++) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }

    // Only swap if tid < rev to avoid double-swapping
    if (tid < rev) {
        ulong tmp = data[tid];
        data[tid] = data[rev];
        data[rev] = tmp;
    }
}

/// Generate twiddle factors for FFT
/// Computes w^0, w^1, w^2, ..., w^(n/2-1) where w is primitive n-th root
kernel void generate_twiddles(
    device ulong* twiddles [[buffer(0)]],
    constant ulong& primitive_root [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= count) {
        return;
    }

    // Compute w^tid by repeated squaring
    // This is slower than sequential generation but parallelizable
    ulong result = 1;
    ulong base = primitive_root;
    uint exp = tid;

    while (exp > 0) {
        if (exp & 1) {
            result = goldilocks_mul(result, base);
        }
        base = goldilocks_mul(base, base);
        exp >>= 1;
    }

    twiddles[tid] = result;
}

/// Single-kernel FFT for small sizes that fit in threadgroup memory
/// More efficient for sizes up to ~2^13 due to reduced global memory traffic
kernel void fft_radix2_small(
    device ulong* data [[buffer(0)]],
    device const ulong* twiddles [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& log_n [[buffer(3)]],
    threadgroup ulong* shared_data [[threadgroup(0)]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    // Load data into shared memory
    for (uint i = tid; i < n; i += tg_size) {
        shared_data[i] = data[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform all FFT stages in shared memory
    uint group_count = 1;
    uint group_size = n;

    for (uint stage = 0; stage < log_n; stage++) {
        uint half_group = group_size / 2;

        // Each thread handles multiple butterflies if needed
        for (uint b = tid; b < n / 2; b += tg_size) {
            uint group = b / half_group;
            uint j = b % half_group;

            uint i = group * group_size + j;
            uint i_half = i + half_group;

            ulong w = twiddles[group];
            ulong a = shared_data[i];
            ulong b_val = shared_data[i_half];

            butterfly_radix2(a, b_val, w);

            shared_data[i] = a;
            shared_data[i_half] = b_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        group_count *= 2;
        group_size /= 2;
    }

    // Write results back to global memory
    for (uint i = tid; i < n; i += tg_size) {
        data[i] = shared_data[i];
    }
}

/// Canonicalize all elements in a buffer to [0, p)
kernel void canonicalize_buffer(
    device ulong* data [[buffer(0)]],
    constant uint& n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }
    data[tid] = goldilocks_canonicalize(data[tid]);
}

/// Test kernel: multiply each element by a constant
/// Used for debugging field arithmetic
kernel void test_multiply(
    device ulong* data [[buffer(0)]],
    constant ulong& multiplier [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }
    data[tid] = goldilocks_mul(data[tid], multiplier);
}

/// Test kernel: add each element with a constant
kernel void test_add(
    device ulong* data [[buffer(0)]],
    constant ulong& addend [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }
    data[tid] = goldilocks_add(data[tid], addend);
}

/// Test kernel: perform single butterfly for debugging
kernel void test_butterfly(
    device ulong* a_vals [[buffer(0)]],
    device ulong* b_vals [[buffer(1)]],
    device const ulong* twiddles [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) {
        return;
    }
    ulong a = a_vals[tid];
    ulong b = b_vals[tid];
    ulong w = twiddles[tid];

    butterfly_radix2(a, b, w);

    a_vals[tid] = a;
    b_vals[tid] = b;
}
