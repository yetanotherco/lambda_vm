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
// Forward and backward kernels are mirrors of each other.
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
// `denom_sign` matches `DenomSign` on the Rust side:
//   0 (DenomSign::ZMinusX): denoms[k * n + i] = z[k] - x[i].   (R3 OOD)
//   1 (DenomSign::XMinusZ): denoms[k * n + i] = x[i] - z[k].   (R4 DEEP)
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
    uint64_t denom_sign,       // 0: z - x; 1: x - z (mirrors `DenomSign`)
    uint64_t *denoms_out       // 3 * k_scalars * n u64
) {
    uint64_t flat = (uint64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    uint64_t total = k_scalars * n;
    if (flat >= total) return;

    uint64_t k = flat / n;
    uint64_t i = flat - k * n;

    // Hoist the per-thread index multiplications so the three indexed
    // loads/stores below are addition-only.
    const uint64_t *z_base = z_scalars + k * 3;
    uint64_t *out_base = denoms_out + flat * 3;

    uint64_t x_i = x_base[i];
    ext3::Fe3 z = { z_base[0], z_base[1], z_base[2] };
    ext3::Fe3 d;
    if (denom_sign == 0) {
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

    out_base[0] = d.a;
    out_base[1] = d.b;
    out_base[2] = d.c;
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

    // Load input or identity. Hoist the per-thread index multiplication
    // so the three loads/stores below are addition-only.
    if (gid < n) {
        const uint64_t *in_base = input + gid * 3;
        shmem[tid].a = in_base[0];
        shmem[tid].b = in_base[1];
        shmem[tid].c = in_base[2];
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
        uint64_t *out_base = scan_out + gid * 3;
        out_base[0] = shmem[tid].a;
        out_base[1] = shmem[tid].b;
        out_base[2] = shmem[tid].c;
    }

    // Block total = scan value at the last VALID thread of this block.
    // The last valid gid in this block is min(block_end - 1, n - 1).
    // Computing it explicitly (instead of `tid == 255 || gid == n - 1`)
    // ensures EXACTLY ONE thread writes per block — in a partial last
    // block the two conditions would otherwise both fire and race.
    uint64_t block_end = ((uint64_t)blockIdx.x + 1) * BLOCK_SIZE;
    uint64_t last_valid_gid = (block_end - 1 < n - 1) ? (block_end - 1) : (n - 1);
    if (gid == last_valid_gid) {
        uint64_t *bt_base = block_totals + (uint64_t)blockIdx.x * 3;
        bt_base[0] = shmem[tid].a;
        bt_base[1] = shmem[tid].b;
        bt_base[2] = shmem[tid].c;
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

    const uint64_t *off_base = block_totals_scanned + (blockIdx.x - 1) * 3;
    uint64_t *inout_base = scan_inout + gid * 3;
    ext3::Fe3 offset = { off_base[0], off_base[1], off_base[2] };
    ext3::Fe3 val    = { inout_base[0], inout_base[1], inout_base[2] };
    ext3::Fe3 res = ext3::mul(offset, val);
    inout_base[0] = res.a;
    inout_base[1] = res.b;
    inout_base[2] = res.c;
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
        const uint64_t *in_base = input + gid * 3;
        shmem[tid].a = in_base[0];
        shmem[tid].b = in_base[1];
        shmem[tid].c = in_base[2];
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
        uint64_t *out_base = scan_out + gid * 3;
        out_base[0] = shmem[tid].a;
        out_base[1] = shmem[tid].b;
        out_base[2] = shmem[tid].c;
    }

    // Mutually-exclusive last-thread mask (same idea as fwd): the last
    // valid pos_from_end in this block is min(block_end - 1, n - 1).
    uint64_t block_end_rev = ((uint64_t)blockIdx.x + 1) * BLOCK_SIZE;
    uint64_t last_valid_pos = (block_end_rev - 1 < n - 1) ? (block_end_rev - 1) : (n - 1);
    if (pos_from_end == last_valid_pos) {
        uint64_t *bt_base = block_totals + (uint64_t)blockIdx.x * 3;
        bt_base[0] = shmem[tid].a;
        bt_base[1] = shmem[tid].b;
        bt_base[2] = shmem[tid].c;
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

    const uint64_t *off_base = block_totals_scanned + (blockIdx.x - 1) * 3;
    uint64_t *inout_base = scan_inout + gid * 3;
    ext3::Fe3 offset = { off_base[0], off_base[1], off_base[2] };
    ext3::Fe3 val    = { inout_base[0], inout_base[1], inout_base[2] };
    ext3::Fe3 res = ext3::mul(offset, val);
    inout_base[0] = res.a;
    inout_base[1] = res.b;
    inout_base[2] = res.c;
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
        const uint64_t *p_base = prefix + (i - 1) * 3;
        p.a = p_base[0];
        p.b = p_base[1];
        p.c = p_base[2];
    }

    ext3::Fe3 s;
    if (i == n - 1) {
        s = ext3::one();
    } else {
        const uint64_t *s_base = suffix + (i + 1) * 3;
        s.a = s_base[0];
        s.b = s_base[1];
        s.c = s_base[2];
    }

    ext3::Fe3 tmp = ext3::mul(p, inv_t);
    ext3::Fe3 res = ext3::mul(tmp, s);

    uint64_t *out_base = out + i * 3;
    out_base[0] = res.a;
    out_base[1] = res.b;
    out_base[2] = res.c;
}

// ---------------------------------------------------------------------------
// 7. invert_total_ext3
//
// One-thread Fermat inversion of the scan total: out = src[n-1]^(p^3 - 2).
// Replaces the host round-trip (D2H + host Fermat + H2D + stream sync) so the
// whole batch inverse stays stream-ordered. The 192-bit exponent arrives as
// three little-endian u64 limbs.
// ---------------------------------------------------------------------------
extern "C" __global__ void invert_total_ext3(
    const uint64_t *src,   // 3 * n u64 (reads element n-1)
    uint64_t n,
    uint64_t e0,           // exponent limbs, little-endian
    uint64_t e1,
    uint64_t e2,
    uint64_t *out          // 3 u64
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const uint64_t *base = src + (n - 1) * 3;
    ext3::Fe3 a = {base[0], base[1], base[2]};
    ext3::Fe3 r = ext3::one();
    uint64_t limbs[3] = {e0, e1, e2};
    for (int li = 2; li >= 0; --li) {
        uint64_t bits = limbs[li];
        for (int b = 63; b >= 0; --b) {
            r = ext3::mul(r, r);
            if ((bits >> b) & 1) {
                r = ext3::mul(r, a);
            }
        }
    }
    out[0] = r.a;
    out[1] = r.b;
    out[2] = r.c;
}
