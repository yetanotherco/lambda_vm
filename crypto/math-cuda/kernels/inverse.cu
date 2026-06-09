// Parallel Montgomery batch inverse over ext3 elements.
//
// Algorithm: given a[0..N-1] all non-zero, compute a^{-1}[0..N-1] using
//   prefix[i]  = a[0] * a[1] * ... * a[i]    (inclusive forward scan)
//   suffix[i]  = a[i] * a[i+1] * ... * a[N-1] (inclusive backward scan)
//   total      = prefix[N-1] = suffix[0]
//   inv_total  = 1 / total                    (one Fermat inversion on host)
//   a^{-1}[i]  = prefix[i-1] * inv_total * suffix[i+1]   (boundaries use identity)
//
// Each scan is a multi-block 3-phase Hillis-Steele scan in shared memory:
//   Phase 1: each block does an inclusive scan over its 256 elements and
//            writes its block sum to a per-block totals array.
//   Phase 2: recursively scan the block totals (host re-launches this same
//            kernel set; recursion depth = ceil(log_256(N))).
//   Phase 3: each block reads its offset (the inclusive prefix of all
//            preceding block sums) and multiplies it into every element.
//
// Forward and backward kernels are mirrors of each other (separate kernels
// for clarity; commutativity of Goldilocks would let them share code).
//
// Buffer layouts: all ext3 buffers are interleaved [a0,b0,c0, a1,b1,c1, ...]
// with one u64 per coordinate. `BLOCK_SIZE = 256` ext3 elements per block
// uses 6 KB of shared memory, well under the per-SM limit on Ada/Blackwell.

#include "goldilocks.cuh"
#include "ext3.cuh"

#define BLOCK_SIZE 256

// ---------------------------------------------------------------------------
// 1. compute_denoms_ext3
//
// If `subtract_x = 0` (R3 OOD convention): denoms[k * n + i] = z[k] - x[i].
//   Matches CPU `barycentric_inv_denoms(z, points)` = 1/(z - points[i]).
// If `subtract_x = 1` (R4 DEEP convention): denoms[k * n + i] = x[i] - z[k].
//   Matches CPU R4 `denoms.push(x_i - z_k)` convention.
//
// Output is ext3-interleaved of length 3 * k_scalars * n.
//
// Launched as grid = ceil(total / BLOCK_SIZE), where total = k_scalars * n.
// Each thread builds one denom.
// ---------------------------------------------------------------------------
extern "C" __global__ void compute_denoms_ext3(
    const uint64_t *x_base,    // n u64
    const uint64_t *z_scalars, // 3 * k_scalars u64
    uint64_t n,
    uint64_t k_scalars,
    uint64_t subtract_x,       // 0: z - x; 1: x - z
    uint64_t *denoms_out       // 3 * k_scalars * n u64
) {
    uint64_t flat = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint64_t total = k_scalars * n;
    if (flat >= total) return;

    uint64_t k = flat / n;
    uint64_t i = flat - k * n;

    uint64_t x_i = x_base[i];
    ext3::Fe3 z = {
        z_scalars[k * 3 + 0],
        z_scalars[k * 3 + 1],
        z_scalars[k * 3 + 2],
    };
    ext3::Fe3 d;
    if (subtract_x == 0) {
        // z - x: lift x to (x, 0, 0), subtract from z.
        d.a = goldilocks::sub(z.a, x_i);
        d.b = z.b;
        d.c = z.c;
    } else {
        // x - z: lift x to (x, 0, 0), subtract z.
        d.a = goldilocks::sub(x_i, z.a);
        d.b = goldilocks::neg(z.b);
        d.c = goldilocks::neg(z.c);
    }

    denoms_out[flat * 3 + 0] = d.a;
    denoms_out[flat * 3 + 1] = d.b;
    denoms_out[flat * 3 + 2] = d.c;
}

// ---------------------------------------------------------------------------
// 2. block_inclusive_scan_fwd_ext3
//
// Per-block forward Hillis-Steele inclusive scan with multiplication. Writes
// scan_out[gid] = product of input[block_start..=gid] and block_totals[bid] =
// the product over the entire block.
//
// Threads handle out-of-range positions by loading the identity element (1),
// so a partial last block still produces a correct scan.
// ---------------------------------------------------------------------------
extern "C" __global__ void block_inclusive_scan_fwd_ext3(
    const uint64_t *input,  // 3 * n u64
    uint64_t n,
    uint64_t *scan_out,     // 3 * n u64
    uint64_t *block_totals  // 3 * K u64, K = ceil(n / BLOCK_SIZE)
) {
    __shared__ ext3::Fe3 shmem[BLOCK_SIZE];
    uint64_t tid = threadIdx.x;
    uint64_t gid = (uint64_t)blockIdx.x * BLOCK_SIZE + tid;

    // Load input or identity.
    if (gid < n) {
        shmem[tid].a = input[gid * 3 + 0];
        shmem[tid].b = input[gid * 3 + 1];
        shmem[tid].c = input[gid * 3 + 2];
    } else {
        shmem[tid] = ext3::one();
    }
    __syncthreads();

    // Hillis-Steele inclusive scan: 8 doubling levels for BLOCK_SIZE = 256.
    for (uint32_t offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        ext3::Fe3 prev = (tid >= offset) ? shmem[tid - offset] : ext3::one();
        __syncthreads();
        if (tid >= offset) {
            shmem[tid] = ext3::mul(prev, shmem[tid]);
        }
        __syncthreads();
    }

    // Write per-element scan result.
    if (gid < n) {
        scan_out[gid * 3 + 0] = shmem[tid].a;
        scan_out[gid * 3 + 1] = shmem[tid].b;
        scan_out[gid * 3 + 2] = shmem[tid].c;
    }

    // Block total = scan value at the last VALID thread of this block.
    // Two cases: full block (tid == 255) or partial last block (gid == n-1).
    bool is_last_in_block = (tid == BLOCK_SIZE - 1) || (gid == n - 1);
    if (is_last_in_block) {
        block_totals[(uint64_t)blockIdx.x * 3 + 0] = shmem[tid].a;
        block_totals[(uint64_t)blockIdx.x * 3 + 1] = shmem[tid].b;
        block_totals[(uint64_t)blockIdx.x * 3 + 2] = shmem[tid].c;
    }
}

// ---------------------------------------------------------------------------
// 3. apply_block_offsets_fwd_ext3
//
// Phase 3 of the forward scan: each block b > 0 multiplies its per-block
// scan by `block_totals_scanned[b-1]` (the inclusive prefix of preceding
// block totals). Block 0 has no offset, so it returns early.
// ---------------------------------------------------------------------------
extern "C" __global__ void apply_block_offsets_fwd_ext3(
    uint64_t *scan_inout,                  // 3 * n u64 (modified in place)
    uint64_t n,
    const uint64_t *block_totals_scanned   // 3 * K u64, inclusive prefix of phase-1 totals
) {
    if (blockIdx.x == 0) return;
    uint64_t tid = threadIdx.x;
    uint64_t gid = (uint64_t)blockIdx.x * BLOCK_SIZE + tid;
    if (gid >= n) return;

    ext3::Fe3 offset = {
        block_totals_scanned[(blockIdx.x - 1) * 3 + 0],
        block_totals_scanned[(blockIdx.x - 1) * 3 + 1],
        block_totals_scanned[(blockIdx.x - 1) * 3 + 2],
    };
    ext3::Fe3 val = {
        scan_inout[gid * 3 + 0],
        scan_inout[gid * 3 + 1],
        scan_inout[gid * 3 + 2],
    };
    ext3::Fe3 res = ext3::mul(offset, val);
    scan_inout[gid * 3 + 0] = res.a;
    scan_inout[gid * 3 + 1] = res.b;
    scan_inout[gid * 3 + 2] = res.c;
}

// ---------------------------------------------------------------------------
// 4. block_inclusive_scan_rev_ext3
//
// Mirror of `block_inclusive_scan_fwd_ext3` for the suffix product:
//   suffix[i] = input[i] * input[i+1] * ... * input[n-1]
//
// Block b processes pos_from_end in [b*B, (b+1)*B), where gid = n-1-pos_from_end.
// Inside shmem the order is reversed so a forward Hillis-Steele scan over
// the loaded values produces the suffix scan in the original index space.
// ---------------------------------------------------------------------------
extern "C" __global__ void block_inclusive_scan_rev_ext3(
    const uint64_t *input,
    uint64_t n,
    uint64_t *scan_out,
    uint64_t *block_totals
) {
    __shared__ ext3::Fe3 shmem[BLOCK_SIZE];
    uint64_t tid = threadIdx.x;
    uint64_t pos_from_end = (uint64_t)blockIdx.x * BLOCK_SIZE + tid;
    bool valid = pos_from_end < n;
    uint64_t gid = valid ? (n - 1 - pos_from_end) : 0;

    if (valid) {
        shmem[tid].a = input[gid * 3 + 0];
        shmem[tid].b = input[gid * 3 + 1];
        shmem[tid].c = input[gid * 3 + 2];
    } else {
        shmem[tid] = ext3::one();
    }
    __syncthreads();

    for (uint32_t offset = 1; offset < BLOCK_SIZE; offset <<= 1) {
        ext3::Fe3 prev = (tid >= offset) ? shmem[tid - offset] : ext3::one();
        __syncthreads();
        if (tid >= offset) {
            shmem[tid] = ext3::mul(prev, shmem[tid]);
        }
        __syncthreads();
    }

    if (valid) {
        scan_out[gid * 3 + 0] = shmem[tid].a;
        scan_out[gid * 3 + 1] = shmem[tid].b;
        scan_out[gid * 3 + 2] = shmem[tid].c;
    }

    bool is_last_in_block = (tid == BLOCK_SIZE - 1) || (pos_from_end == n - 1);
    if (is_last_in_block) {
        block_totals[(uint64_t)blockIdx.x * 3 + 0] = shmem[tid].a;
        block_totals[(uint64_t)blockIdx.x * 3 + 1] = shmem[tid].b;
        block_totals[(uint64_t)blockIdx.x * 3 + 2] = shmem[tid].c;
    }
}

// ---------------------------------------------------------------------------
// 5. apply_block_offsets_rev_ext3
//
// Phase 3 of the suffix scan. Block b > 0 multiplies its per-block scan
// by the inclusive prefix of block totals from blocks [0..b-1] (which, in
// the reverse-block indexing, correspond to the indices LARGER than this
// block's gids).
// ---------------------------------------------------------------------------
extern "C" __global__ void apply_block_offsets_rev_ext3(
    uint64_t *scan_inout,
    uint64_t n,
    const uint64_t *block_totals_scanned
) {
    if (blockIdx.x == 0) return;
    uint64_t tid = threadIdx.x;
    uint64_t pos_from_end = (uint64_t)blockIdx.x * BLOCK_SIZE + tid;
    if (pos_from_end >= n) return;
    uint64_t gid = n - 1 - pos_from_end;

    ext3::Fe3 offset = {
        block_totals_scanned[(blockIdx.x - 1) * 3 + 0],
        block_totals_scanned[(blockIdx.x - 1) * 3 + 1],
        block_totals_scanned[(blockIdx.x - 1) * 3 + 2],
    };
    ext3::Fe3 val = {
        scan_inout[gid * 3 + 0],
        scan_inout[gid * 3 + 1],
        scan_inout[gid * 3 + 2],
    };
    ext3::Fe3 res = ext3::mul(offset, val);
    scan_inout[gid * 3 + 0] = res.a;
    scan_inout[gid * 3 + 1] = res.b;
    scan_inout[gid * 3 + 2] = res.c;
}

// ---------------------------------------------------------------------------
// 6. batch_inverse_combine_ext3
//
//   out[i] = prefix[i-1] * inv_total * suffix[i+1]
//
// Boundaries: prefix[-1] = identity, suffix[n] = identity.
// inv_total = 1 / (prefix[n-1]) = 1 / (suffix[0]); the caller computes it
// on host via Fermat's little theorem (one extension-field inverse per
// batch) and uploads as a 3 * u64 device buffer.
// ---------------------------------------------------------------------------
extern "C" __global__ void batch_inverse_combine_ext3(
    const uint64_t *prefix,      // 3 * n u64
    const uint64_t *suffix,      // 3 * n u64
    const uint64_t *inv_total,   // 3 u64
    uint64_t n,
    uint64_t *out                // 3 * n u64
) {
    uint64_t i = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;

    ext3::Fe3 inv_t = {inv_total[0], inv_total[1], inv_total[2]};

    ext3::Fe3 p;
    if (i == 0) {
        p = ext3::one();
    } else {
        p.a = prefix[(i - 1) * 3 + 0];
        p.b = prefix[(i - 1) * 3 + 1];
        p.c = prefix[(i - 1) * 3 + 2];
    }

    ext3::Fe3 s;
    if (i == n - 1) {
        s = ext3::one();
    } else {
        s.a = suffix[(i + 1) * 3 + 0];
        s.b = suffix[(i + 1) * 3 + 1];
        s.c = suffix[(i + 1) * 3 + 2];
    }

    ext3::Fe3 tmp = ext3::mul(p, inv_t);
    ext3::Fe3 res = ext3::mul(tmp, s);

    out[i * 3 + 0] = res.a;
    out[i * 3 + 1] = res.b;
    out[i * 3 + 2] = res.c;
}
