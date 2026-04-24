// Parallel Montgomery batch inverse over ext3, plus a compute-denoms
// helper for R3 OOD / R4 DEEP preludes.
//
// Batch inverse strategy (chunk-based parallel scan):
//
//   1. Chunk-local forward scan: each thread serially computes the
//      prefix product of its chunk of `C = ceil(N / K)` ext3 values;
//      writes the chunk output in place and posts its chunk total to
//      `chunk_totals[thread_id]`.
//   2. Single-block scan of `chunk_totals` (K ≤ 1024 for our shapes,
//      fits one block).
//   3. Chunk-local apply: each thread multiplies its chunk's local
//      prefix by the exclusive-scan offset from step 2, producing the
//      global forward prefix.
//   4. Mirror (1-3) in reverse for the suffix.
//   5. Single-thread kernel inverts total = prefix[N-1].
//   6. Pointwise combine: `inv[i] = prefix[i-1] * suffix[i+1] * inv_total`
//      (with prefix[-1] = suffix[N] = 1). One thread per element.
//
// Ext3 multiply is commutative in the field (it's a field, not just a
// ring), so prefix-product scans are well-defined. Layout is ext3
// INTERLEAVED: one u64 triple per element, 3*N u64s total.

#include "goldilocks.cuh"
#include "ext3.cuh"

#define INV_BLOCK 256

// ---------------------------------------------------------------------------
// B.1: compute denoms for R4 DEEP and R3 OOD.
//
//   denoms[k*n + i] = x[i * stride] - z[k]
// where `x` is a base-field coset (read at stride `stride`), `z` is an
// ext3 array of `k_scalars` entries (z^K and/or z·ω^k), and `n` is the
// trace-size count. Output is flat ext3 interleaved.
// ---------------------------------------------------------------------------
extern "C" __global__ void compute_denoms_ext3(
    const uint64_t *x_base,     // base-field LDE coset points
    uint64_t stride,            // read stride (blowup_factor for R4)
    const uint64_t *z_scalars,  // k_scalars * 3 u64 (ext3 interleaved)
    uint64_t k_scalars,
    uint64_t n,
    uint64_t *denoms_out) {     // k_scalars * n * 3 u64
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total = k_scalars * n;
    if (tid >= total) return;

    uint64_t k = tid / n;
    uint64_t i = tid - k * n;

    uint64_t x_i = x_base[i * stride];
    uint64_t z_a = z_scalars[k * 3 + 0];
    uint64_t z_b = z_scalars[k * 3 + 1];
    uint64_t z_c = z_scalars[k * 3 + 2];

    // base - ext3 = ext3 ( (x_i - z_a), -z_b, -z_c )
    uint64_t out_a = goldilocks::sub(x_i, z_a);
    uint64_t out_b = goldilocks::neg(z_b);
    uint64_t out_c = goldilocks::neg(z_c);

    uint64_t out_idx = tid * 3;
    denoms_out[out_idx + 0] = out_a;
    denoms_out[out_idx + 1] = out_b;
    denoms_out[out_idx + 2] = out_c;
}

// ---------------------------------------------------------------------------
// B.2 chunk-scan primitives for batch inverse.
//
// `a_in` is the input array of N ext3 elements (3*N u64, interleaved).
// `prefix_out` receives prefix[i] = prod(a[0..=i]) for all i.
// `chunk_totals` receives the per-chunk total (one ext3 per chunk).
//
// Each thread owns a contiguous chunk of C elements. With K=256 threads
// per block and a single block, we can handle up to 256*C elements.
// For N up to ~1M, C ≈ 4096, so one thread does ~4k ext3 multiplies
// serially in shmem-free fashion. Depth = O(C) + O(K) + O(C); with
// K=256 threads running in parallel, the `O(C)` phases parallelise
// perfectly across threads.
//
// For cleanliness, we launch as grid=1, block=K=256. For N up to 2^20
// that's fine; if we ever need N > 256 * C_max, we'd recurse.
// ---------------------------------------------------------------------------

// Phase 1 & 3 fused into one kernel would require shmem across phases.
// Splitting makes each kernel simpler.

// Phase 1: chunk-local forward scan. Also emits chunk_totals.
extern "C" __global__ void chunk_prefix_scan_ext3(
    const uint64_t *a_in,       // 3 * n u64 (ext3 interleaved)
    uint64_t n,
    uint64_t c_per_thread,      // C = ceil(n / K)
    uint64_t *prefix_out,       // 3 * n u64
    uint64_t *chunk_totals) {   // 3 * K u64
    uint32_t tid = threadIdx.x;
    uint64_t start = (uint64_t)tid * c_per_thread;
    uint64_t end = min(start + c_per_thread, n);

    ext3::Fe3 acc = ext3::one();
    for (uint64_t i = start; i < end; ++i) {
        ext3::Fe3 e = {a_in[i * 3 + 0], a_in[i * 3 + 1], a_in[i * 3 + 2]};
        acc = ext3::mul(acc, e);
        prefix_out[i * 3 + 0] = acc.a;
        prefix_out[i * 3 + 1] = acc.b;
        prefix_out[i * 3 + 2] = acc.c;
    }
    chunk_totals[tid * 3 + 0] = acc.a;
    chunk_totals[tid * 3 + 1] = acc.b;
    chunk_totals[tid * 3 + 2] = acc.c;
}

// Phase 2: exclusive prefix scan of chunk_totals, single-threaded.
// scan_out[0] = 1, scan_out[i] = prod(chunk_totals[0..i]).
extern "C" __global__ void exclusive_scan_of_totals_ext3(
    const uint64_t *chunk_totals,  // 3 * K u64
    uint64_t k,
    uint64_t *scan_out) {          // 3 * K u64
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    ext3::Fe3 acc = ext3::one();
    scan_out[0] = acc.a;
    scan_out[1] = acc.b;
    scan_out[2] = acc.c;
    for (uint64_t i = 1; i < k; ++i) {
        ext3::Fe3 ct = {
            chunk_totals[(i - 1) * 3 + 0],
            chunk_totals[(i - 1) * 3 + 1],
            chunk_totals[(i - 1) * 3 + 2],
        };
        acc = ext3::mul(acc, ct);
        scan_out[i * 3 + 0] = acc.a;
        scan_out[i * 3 + 1] = acc.b;
        scan_out[i * 3 + 2] = acc.c;
    }
}

// Phase 3: apply per-chunk offset to local scan result.
//   global_prefix[i] = offsets[thread] * local_prefix[i]
extern "C" __global__ void apply_scan_offsets_ext3(
    uint64_t *prefix_inout,  // 3 * n u64 (written in phase 1, rewritten here)
    uint64_t n,
    uint64_t c_per_thread,
    const uint64_t *offsets) {  // 3 * K u64
    uint32_t tid = threadIdx.x;
    uint64_t start = (uint64_t)tid * c_per_thread;
    uint64_t end = min(start + c_per_thread, n);

    ext3::Fe3 off = {
        offsets[tid * 3 + 0],
        offsets[tid * 3 + 1],
        offsets[tid * 3 + 2],
    };
    for (uint64_t i = start; i < end; ++i) {
        ext3::Fe3 local = {
            prefix_inout[i * 3 + 0],
            prefix_inout[i * 3 + 1],
            prefix_inout[i * 3 + 2],
        };
        ext3::Fe3 g = ext3::mul(off, local);
        prefix_inout[i * 3 + 0] = g.a;
        prefix_inout[i * 3 + 1] = g.b;
        prefix_inout[i * 3 + 2] = g.c;
    }
}

// Reverse-scan phase 1: chunk-local reverse prefix.
//   suffix_out[i] = prod(a[i..chunk_end])  (within chunk only)
//   chunk_totals[tid] = suffix_out[chunk_start]  (= full chunk product)
extern "C" __global__ void chunk_suffix_scan_ext3(
    const uint64_t *a_in,
    uint64_t n,
    uint64_t c_per_thread,
    uint64_t *suffix_out,
    uint64_t *chunk_totals) {
    uint32_t tid = threadIdx.x;
    uint64_t start = (uint64_t)tid * c_per_thread;
    // Walk backward; acc starts at 1 and accumulates a[end-1], a[end-2], ...
    // Empty chunks (start >= n) fall through with acc = 1 so that
    // chunk_totals receives the identity, matching the prefix-scan kernel.
    ext3::Fe3 acc = ext3::one();
    if (start < n) {
        uint64_t end = min(start + c_per_thread, n);
        for (uint64_t ri = end; ri > start; --ri) {
            uint64_t i = ri - 1;
            ext3::Fe3 e = {a_in[i * 3 + 0], a_in[i * 3 + 1], a_in[i * 3 + 2]};
            acc = ext3::mul(acc, e);
            suffix_out[i * 3 + 0] = acc.a;
            suffix_out[i * 3 + 1] = acc.b;
            suffix_out[i * 3 + 2] = acc.c;
        }
    }
    chunk_totals[tid * 3 + 0] = acc.a;
    chunk_totals[tid * 3 + 1] = acc.b;
    chunk_totals[tid * 3 + 2] = acc.c;
}

// Exclusive reverse scan of chunk totals.
//   scan_out[K-1] = 1
//   scan_out[k] = prod(chunk_totals[k+1..K])
extern "C" __global__ void exclusive_reverse_scan_of_totals_ext3(
    const uint64_t *chunk_totals,
    uint64_t k,
    uint64_t *scan_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    ext3::Fe3 acc = ext3::one();
    if (k == 0) return;
    scan_out[(k - 1) * 3 + 0] = acc.a;
    scan_out[(k - 1) * 3 + 1] = acc.b;
    scan_out[(k - 1) * 3 + 2] = acc.c;
    for (int64_t i = (int64_t)k - 2; i >= 0; --i) {
        ext3::Fe3 ct = {
            chunk_totals[(i + 1) * 3 + 0],
            chunk_totals[(i + 1) * 3 + 1],
            chunk_totals[(i + 1) * 3 + 2],
        };
        acc = ext3::mul(acc, ct);
        scan_out[i * 3 + 0] = acc.a;
        scan_out[i * 3 + 1] = acc.b;
        scan_out[i * 3 + 2] = acc.c;
    }
}

// Apply reverse offsets.
extern "C" __global__ void apply_reverse_scan_offsets_ext3(
    uint64_t *suffix_inout,
    uint64_t n,
    uint64_t c_per_thread,
    const uint64_t *offsets) {
    uint32_t tid = threadIdx.x;
    uint64_t start = (uint64_t)tid * c_per_thread;
    if (start >= n) return;
    uint64_t end = min(start + c_per_thread, n);

    ext3::Fe3 off = {
        offsets[tid * 3 + 0],
        offsets[tid * 3 + 1],
        offsets[tid * 3 + 2],
    };
    for (uint64_t i = start; i < end; ++i) {
        ext3::Fe3 local = {
            suffix_inout[i * 3 + 0],
            suffix_inout[i * 3 + 1],
            suffix_inout[i * 3 + 2],
        };
        ext3::Fe3 g = ext3::mul(off, local);
        suffix_inout[i * 3 + 0] = g.a;
        suffix_inout[i * 3 + 1] = g.b;
        suffix_inout[i * 3 + 2] = g.c;
    }
}

// Same fix for the forward apply_scan_offsets: threads whose chunks are
// empty must not write past end-of-array. (chunk_prefix_scan already
// behaves correctly because the start..end range is empty; apply just
// needs to handle start >= n gracefully — it already does by the same
// empty-range logic. No change needed there, just documenting.)

// Final combine: inv[i] = pre_excl[i] * suf_excl[i] * inv_total
// where pre_excl[i] = prefix[i-1] (with prefix[-1] = 1) and
//       suf_excl[i] = suffix[i+1] (with suffix[N]   = 1).
//
// Instead of creating separate pre_excl / suf_excl arrays, we pass the
// inclusive prefix / suffix arrays and shift the index here.
extern "C" __global__ void batch_inverse_combine_ext3(
    const uint64_t *prefix_incl,   // 3 * n u64; prefix_incl[i] = prod(a[0..=i])
    const uint64_t *suffix_incl,   // 3 * n u64; suffix_incl[i] = prod(a[i..n-1])
    const uint64_t *inv_total_ptr, // 3 u64
    uint64_t n,
    uint64_t *inv_out) {           // 3 * n u64
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    ext3::Fe3 pre;
    if (i == 0) {
        pre = ext3::one();
    } else {
        pre.a = prefix_incl[(i - 1) * 3 + 0];
        pre.b = prefix_incl[(i - 1) * 3 + 1];
        pre.c = prefix_incl[(i - 1) * 3 + 2];
    }
    ext3::Fe3 suf;
    if (i + 1 >= n) {
        suf = ext3::one();
    } else {
        suf.a = suffix_incl[(i + 1) * 3 + 0];
        suf.b = suffix_incl[(i + 1) * 3 + 1];
        suf.c = suffix_incl[(i + 1) * 3 + 2];
    }
    ext3::Fe3 inv_tot = {inv_total_ptr[0], inv_total_ptr[1], inv_total_ptr[2]};

    ext3::Fe3 r = ext3::mul(pre, suf);
    r = ext3::mul(r, inv_tot);

    inv_out[i * 3 + 0] = r.a;
    inv_out[i * 3 + 1] = r.b;
    inv_out[i * 3 + 2] = r.c;
}
