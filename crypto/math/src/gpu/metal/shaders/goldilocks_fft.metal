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

/// Radix-4 butterfly: processes 4 elements in one operation
/// Reduces number of stages by half compared to radix-2
///
/// This matches the CPU implementation in fft.rs:
/// - w1 = twiddles[group]
/// - w2 = twiddles[2 * group]
/// - w3 = twiddles[2 * group + 1]
///
/// Butterfly formula:
/// zw1 = w1 * z
/// tw1 = w1 * t
/// a = w2 * (y + tw1)
/// b = w3 * (y - tw1)
/// x' = x + zw1 + a
/// y' = x + zw1 - a
/// z' = x - zw1 + b
/// t' = x - zw1 - b
inline void butterfly_radix4(
    thread ulong& x,    // input[i]
    thread ulong& y,    // input[i + group_size/4]
    thread ulong& z,    // input[i + group_size/2]
    thread ulong& t,    // input[i + 3*group_size/4]
    ulong w1,           // twiddles[group]
    ulong w2,           // twiddles[2 * group]
    ulong w3            // twiddles[2 * group + 1]
) {
    // Compute intermediate values
    ulong zw1 = goldilocks_mul(w1, z);
    ulong tw1 = goldilocks_mul(w1, t);

    ulong y_plus_tw1 = goldilocks_add(y, tw1);
    ulong y_minus_tw1 = goldilocks_sub(y, tw1);

    ulong a = goldilocks_mul(w2, y_plus_tw1);
    ulong b = goldilocks_mul(w3, y_minus_tw1);

    // Compute x + zw1 and x - zw1 (reused in outputs)
    ulong x_plus_zw1 = goldilocks_add(x, zw1);
    ulong x_minus_zw1 = goldilocks_sub(x, zw1);

    // Compute outputs
    x = goldilocks_add(x_plus_zw1, a);   // x' = x + zw1 + a
    y = goldilocks_sub(x_plus_zw1, a);   // y' = x + zw1 - a
    z = goldilocks_add(x_minus_zw1, b);  // z' = x - zw1 + b
    t = goldilocks_sub(x_minus_zw1, b);  // t' = x - zw1 - b
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

/// Single stage of radix-4 FFT (equivalent to two radix-2 stages)
/// Each thread handles one radix-4 butterfly (4 elements)
///
/// This matches the CPU implementation twiddle indexing:
/// - w1 = twiddles[group]
/// - w2 = twiddles[2 * group]
/// - w3 = twiddles[2 * group + 1]
///
/// Parameters:
///   - data: Input/output buffer of field elements
///   - twiddles: Pre-computed twiddle factors (bit-reversed order)
///   - n: Total number of elements
///   - group_count: Number of radix-4 groups in this stage
///   - group_size: Size of each group (must be multiple of 4)
kernel void fft_radix4_stage(
    device ulong* data [[buffer(0)]],
    device const ulong* twiddles [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    constant uint& group_count [[buffer(3)]],
    constant uint& group_size [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    // Each thread handles one radix-4 butterfly (4 elements)
    uint butterflies_per_group = group_size / 4;
    uint total_butterflies = group_count * butterflies_per_group;

    if (tid >= total_butterflies) {
        return;
    }

    // Determine which group and position within group
    uint group = tid / butterflies_per_group;
    uint pos = tid % butterflies_per_group;

    // Calculate indices for the 4 elements (matching CPU indexing)
    uint first_in_group = group * group_size;
    uint i = first_in_group + pos;
    uint j = i + group_size / 4;
    uint k = i + group_size / 2;
    uint l = i + 3 * group_size / 4;

    // Get twiddle factors (matching CPU: twiddles[group], twiddles[2*group], twiddles[2*group+1])
    ulong w1 = twiddles[group];
    ulong w2 = twiddles[2 * group];
    ulong w3 = twiddles[2 * group + 1];

    // Load values
    ulong x = data[i];
    ulong y = data[j];
    ulong z = data[k];
    ulong t = data[l];

    // Perform radix-4 butterfly
    butterfly_radix4(x, y, z, t, w1, w2, w3);

    // Store results
    data[i] = x;
    data[j] = y;
    data[k] = z;
    data[l] = t;
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

// =============================================================================
// Poseidon2 Hash Function for Merkle Trees
// =============================================================================

// Poseidon2 parameters for Goldilocks width 8
constant uint POSEIDON2_WIDTH = 8;
constant uint POSEIDON2_EXTERNAL_ROUNDS_BEGIN = 4;
constant uint POSEIDON2_EXTERNAL_ROUNDS_END = 4;
constant uint POSEIDON2_INTERNAL_ROUNDS = 22;

// Diagonal elements for internal diffusion matrix
constant ulong MATRIX_DIAG_8[8] = {
    0xa98811a1fed4e3a5UL, 0x1cc48b54f377e2a0UL,
    0xe40cd4f6c5609a26UL, 0x11de79ebca97a4a3UL,
    0x9177c73d8b7e929cUL, 0x2a6fe8085797e791UL,
    0x3de6e93329f8d5adUL, 0x3f7af9125da962feUL
};

// External round constants - initial 4 rounds
constant ulong EXT_RC_INIT[4][8] = {
    { 0xdd5743e7f2a5a5d9UL, 0xcb3a864e58ada44bUL, 0xffa2449ed32f8cdcUL, 0x42025f65d6bd13eeUL,
      0x7889175e25506323UL, 0x34b98bb03d24b737UL, 0xbdcc535ecc4faa2aUL, 0x5b20ad869fc0d033UL },
    { 0xf1dda5b9259dfcb4UL, 0x27515210be112d59UL, 0x4227d1718c766c3fUL, 0x26d333161a5bd794UL,
      0x49b938957bf4b026UL, 0x4a56b5938b213669UL, 0x1120426b48c8353dUL, 0x6b323c3f10a56cadUL },
    { 0xce57d6245ddca6b2UL, 0xb1fc8d402bba1eb1UL, 0xb5c5096ca959bd04UL, 0x6db55cd306d31f7fUL,
      0xc49d293a81cb9641UL, 0x1ce55a4fe979719fUL, 0xa92e60a9d178a4d1UL, 0x002cc64973bcfd8cUL },
    { 0xcea721cce82fb11bUL, 0xe5b55eb8098ece81UL, 0x4e30525c6f1ddd66UL, 0x43c6702827070987UL,
      0xaca68430a7b5762aUL, 0x3674238634df9c93UL, 0x88cee1c825e33433UL, 0xde99ae8d74b57176UL }
};

// External round constants - terminal 4 rounds
constant ulong EXT_RC_TERM[4][8] = {
    { 0x014ef1197d341346UL, 0x9725e20825d07394UL, 0xfdb25aef2c5bae3bUL, 0xbe5402dc598c971eUL,
      0x93a5711f04cdca3dUL, 0xc45a9a5b2f8fb97bUL, 0xfe8946a924933545UL, 0x2af997a27369091cUL },
    { 0xaa62c88e0b294011UL, 0x058eb9d810ce9f74UL, 0xb3cb23eced349ae4UL, 0xa3648177a77b4a84UL,
      0x43153d905992d95dUL, 0xf4e2a97cda44aa4bUL, 0x5baa2702b908682fUL, 0x082923bdf4f750d1UL },
    { 0x98ae09a325893803UL, 0xf8a6475077968838UL, 0xceb0735bf00b2c5fUL, 0x0a1a5d953888e072UL,
      0x2fcb190489f94475UL, 0xb5be06270dec69fcUL, 0x739cb934b09acf8bUL, 0x537750b75ec7f25bUL },
    { 0xe9dd318bae1f3961UL, 0xf7462137299efe1aUL, 0xb1f6b8eee9adb940UL, 0xbdebcc8a809dfe6bUL,
      0x40fc1f791b178113UL, 0x3ac1c3362d014864UL, 0x9a016184bdb8aebaUL, 0x95f2394459fbc25eUL }
};

// Internal round constants (22 values)
constant ulong INT_RC[22] = {
    0x488897d85ff51f56UL, 0x1140737ccb162218UL, 0xa7eeb9215866ed35UL, 0x9bd2976fee49fcc9UL,
    0xc0c8f0de580a3fccUL, 0x4fb2dae6ee8fc793UL, 0x343a89f35f37395bUL, 0x223b525a77ca72c8UL,
    0x56ccb62574aaa918UL, 0xc4d507d8027af9edUL, 0xa080673cf0b7e95cUL, 0xf0184884eb70dcf8UL,
    0x044f10b0cb3d5c69UL, 0xe9e3f7993938f186UL, 0x1b761c80e772f459UL, 0x606cec607a1b5facUL,
    0x14a0c2e1d45f03cdUL, 0x4eace8855398574fUL, 0xf905ca7103eff3e6UL, 0xf8c8f8d20862c059UL,
    0xb524fe8bdd678e5aUL, 0xfbb7865901a1ec41UL
};

/// Poseidon2 S-box: x^7
inline ulong poseidon2_sbox(ulong x) {
    ulong x2 = goldilocks_mul(x, x);
    ulong x4 = goldilocks_mul(x2, x2);
    ulong x6 = goldilocks_mul(x4, x2);
    return goldilocks_mul(x6, x);
}

/// Apply Horizen Labs 4x4 MDS matrix to 4 elements
/// Matrix: [[5,7,1,3], [4,6,1,1], [1,3,5,7], [1,1,4,6]]
inline void apply_hl_mat4(thread ulong* x) {
    ulong t0 = goldilocks_add(x[0], x[1]);
    ulong t1 = goldilocks_add(x[2], x[3]);
    ulong x1_double = goldilocks_add(x[1], x[1]);
    ulong x3_double = goldilocks_add(x[3], x[3]);
    ulong t2 = goldilocks_add(x1_double, t1);
    ulong t3 = goldilocks_add(x3_double, t0);
    ulong t1_double = goldilocks_add(t1, t1);
    ulong t4 = goldilocks_add(goldilocks_add(t1_double, t1_double), t3);
    ulong t0_double = goldilocks_add(t0, t0);
    ulong t5 = goldilocks_add(goldilocks_add(t0_double, t0_double), t2);
    ulong t6 = goldilocks_add(t3, t5);
    ulong t7 = goldilocks_add(t2, t4);

    x[0] = t6;
    x[1] = t5;
    x[2] = t7;
    x[3] = t4;
}

/// External linear layer for width 8
inline void poseidon2_external_linear(thread ulong* state) {
    // Apply HL M4 to each half
    ulong first_half[4] = { state[0], state[1], state[2], state[3] };
    ulong second_half[4] = { state[4], state[5], state[6], state[7] };

    apply_hl_mat4(first_half);
    apply_hl_mat4(second_half);

    // Copy back
    for (uint i = 0; i < 4; i++) {
        state[i] = first_half[i];
        state[i + 4] = second_half[i];
    }

    // Diffuse across halves
    for (uint i = 0; i < 4; i++) {
        ulong sum = goldilocks_add(state[i], state[i + 4]);
        state[i] = goldilocks_add(state[i], sum);
        state[i + 4] = goldilocks_add(state[i + 4], sum);
    }
}

/// Internal linear layer: y[i] = diag[i] * x[i] + sum(x)
inline void poseidon2_internal_linear(thread ulong* state) {
    ulong sum = 0;
    for (uint i = 0; i < POSEIDON2_WIDTH; i++) {
        sum = goldilocks_add(sum, state[i]);
    }
    for (uint i = 0; i < POSEIDON2_WIDTH; i++) {
        ulong diag_x = goldilocks_mul(MATRIX_DIAG_8[i], state[i]);
        state[i] = goldilocks_add(diag_x, sum);
    }
}

/// External round: ARC + Sbox (all) + Linear
inline void poseidon2_external_round(thread ulong* state, const constant ulong* rc) {
    for (uint i = 0; i < POSEIDON2_WIDTH; i++) {
        state[i] = goldilocks_add(state[i], rc[i]);
    }
    for (uint i = 0; i < POSEIDON2_WIDTH; i++) {
        state[i] = poseidon2_sbox(state[i]);
    }
    poseidon2_external_linear(state);
}

/// Internal round: ARC[0] + Sbox[0] + Linear
inline void poseidon2_internal_round(thread ulong* state, ulong rc) {
    state[0] = goldilocks_add(state[0], rc);
    state[0] = poseidon2_sbox(state[0]);
    poseidon2_internal_linear(state);
}

/// Full Poseidon2 permutation
inline void poseidon2_permute(thread ulong* state) {
    // Initial linear layer
    poseidon2_external_linear(state);

    // Initial external rounds
    for (uint r = 0; r < POSEIDON2_EXTERNAL_ROUNDS_BEGIN; r++) {
        poseidon2_external_round(state, EXT_RC_INIT[r]);
    }

    // Internal rounds
    for (uint r = 0; r < POSEIDON2_INTERNAL_ROUNDS; r++) {
        poseidon2_internal_round(state, INT_RC[r]);
    }

    // Terminal external rounds
    for (uint r = 0; r < POSEIDON2_EXTERNAL_ROUNDS_END; r++) {
        poseidon2_external_round(state, EXT_RC_TERM[r]);
    }
}

/// Hash two field elements using Poseidon2 (for Merkle tree internal nodes)
inline ulong poseidon2_compress(ulong left, ulong right) {
    ulong state[8] = { left, right, 0, 0, 0, 0, 0, 2 }; // Domain separation = 2
    poseidon2_permute(state);
    return goldilocks_canonicalize(state[0]);
}

// =============================================================================
// Merkle Tree Kernels
// =============================================================================

/// Hash leaves kernel: each thread hashes one leaf (single field element)
kernel void merkle_hash_leaves(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint& n [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n) return;

    // Hash single element using Poseidon2
    ulong state[8] = { input[tid], 0, 0, 0, 0, 0, 0, 1 }; // Domain separation = 1
    poseidon2_permute(state);
    output[tid] = goldilocks_canonicalize(state[0]);
}

/// Build one level of Merkle tree: hash pairs of nodes
kernel void merkle_build_level(
    device const ulong* prev_level [[buffer(0)]],
    device ulong* next_level [[buffer(1)]],
    constant uint& num_pairs [[buffer(2)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_pairs) return;

    ulong left = prev_level[2 * tid];
    ulong right = prev_level[2 * tid + 1];
    next_level[tid] = poseidon2_compress(left, right);
}

/// Hash leaves in batches (multiple elements per leaf)
/// Each thread handles one leaf which is a vector of field elements
kernel void merkle_hash_leaf_batch(
    device const ulong* input [[buffer(0)]],
    device ulong* output [[buffer(1)]],
    constant uint& num_leaves [[buffer(2)]],
    constant uint& elements_per_leaf [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= num_leaves) return;

    uint start = tid * elements_per_leaf;

    // Initialize state with domain separation based on input length
    ulong state[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

    // Absorb elements (simple version: XOR into state)
    for (uint i = 0; i < elements_per_leaf && i < 7; i++) {
        state[i] = input[start + i];
    }
    state[7] = (ulong)elements_per_leaf; // Domain separation

    poseidon2_permute(state);
    output[tid] = goldilocks_canonicalize(state[0]);
}

// =============================================================================
// Stockham FFT - Auto-sort algorithm with better memory access patterns
// =============================================================================

/// Stockham radix-2 FFT stage (out-of-place, self-sorting)
/// This algorithm avoids the need for bit-reversal permutation and has
/// more coalesced memory access patterns suitable for GPUs.
///
/// Input is read from src, output is written to dst.
/// After all stages, result is in natural order.
kernel void fft_stockham_stage(
    device const ulong* src [[buffer(0)]],
    device ulong* dst [[buffer(1)]],
    device const ulong* twiddles [[buffer(2)]],
    constant uint& n [[buffer(3)]],
    constant uint& stage [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid >= n / 2) {
        return;
    }

    // Compute butterfly indices for Stockham FFT
    // At stage s (0-indexed), butterflies process elements with stride 2^s
    uint butterfly_size = 1u << (stage + 1);
    uint half_size = 1u << stage;

    // Thread tid handles the tid-th butterfly
    // Determine which butterfly and position within butterfly
    uint butterfly_idx = tid / half_size;
    uint pos_in_butterfly = tid % half_size;

    // Source indices (contiguous read for better coalescing)
    uint src_even = butterfly_idx * half_size + pos_in_butterfly;
    uint src_odd = src_even + (n / 2);

    // Destination indices (interleaved write)
    uint dst_base = butterfly_idx * butterfly_size + pos_in_butterfly;
    uint dst_lo = dst_base;
    uint dst_hi = dst_base + half_size;

    // Get twiddle factor
    // For Stockham FFT, twiddle index is pos_in_butterfly * (n / butterfly_size)
    uint twiddle_idx = pos_in_butterfly * (n / butterfly_size);
    ulong w = twiddles[twiddle_idx];

    // Load values
    ulong a = src[src_even];
    ulong b = src[src_odd];

    // Butterfly: (a, b) -> (a + w*b, a - w*b)
    ulong wb = goldilocks_mul(w, b);
    ulong sum = goldilocks_add(a, wb);
    ulong diff = goldilocks_sub(a, wb);

    // Store results
    dst[dst_lo] = sum;
    dst[dst_hi] = diff;
}
