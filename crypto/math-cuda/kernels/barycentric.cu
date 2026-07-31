// Barycentric evaluation of a polynomial (given as evaluations on a coset) at
// a single out-of-domain point. Matches the CPU
// `math::polynomial::interpolate_coset_eval_*_with_g_n_inv` pair.
//
// Per column, the barycentric sum is
//     S = sum over i of point_i * eval_i * inv_denom_i
// where `point_i` is a base-field coset point, `eval_i` is the polynomial's
// value at that point (base for main-trace columns, ext3 for aux or composition
// columns), and `inv_denom_i = 1 / (z - point_i)` is an ext3 scalar (same for
// every column sharing the evaluation point `z`).
//
// These kernels compute only S. The full OOD value is S scaled by the ext3
// constant `vanishing * n_inv * g_n_inv`, which is constant across a column, so
// the caller applies it once per column (one ext3 mul per column, independent
// of n). Keeping it on the host means the kernel takes no extra ext3 constant
// argument.
//
// Launch: grid = (num_cols, 1, 1), block = (BARY_BLOCK_DIM, 1, 1).

#include "goldilocks.cuh"
#include "ext3.cuh"

// 256 threads/block. One ext3 accumulator per thread in shmem => 6 KiB.
#define BARY_BLOCK_DIM 256

__device__ __forceinline__ ext3::Fe3 block_reduce_ext3(ext3::Fe3 my) {
    __shared__ uint64_t shm_a[BARY_BLOCK_DIM];
    __shared__ uint64_t shm_b[BARY_BLOCK_DIM];
    __shared__ uint64_t shm_c[BARY_BLOCK_DIM];
    uint32_t tid = threadIdx.x;
    shm_a[tid] = my.a;
    shm_b[tid] = my.b;
    shm_c[tid] = my.c;
    __syncthreads();
    for (uint32_t s = BARY_BLOCK_DIM / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shm_a[tid] = goldilocks::add(shm_a[tid], shm_a[tid + s]);
            shm_b[tid] = goldilocks::add(shm_b[tid], shm_b[tid + s]);
            shm_c[tid] = goldilocks::add(shm_c[tid], shm_c[tid + s]);
        }
        __syncthreads();
    }
    return ext3::make(shm_a[0], shm_b[0], shm_c[0]);
}

/// Base-column variant: M base-field columns, each `col_stride` u64 apart.
/// `inv_denoms` is a flat 3N u64 buffer (ext3, interleaved `[a0,b0,c0,...]`).
/// Writes `out_ext3_int`: 3M u64, ext3 interleaved, one accumulator per column.
extern "C" __global__ void barycentric_base_batched(
    const uint64_t *columns,
    uint64_t col_stride,
    const uint64_t *coset_points,
    const uint64_t *inv_denoms,
    uint64_t n,
    uint64_t *out_ext3_int
) {
    uint64_t col = blockIdx.x;
    const uint64_t *col_data = columns + col * col_stride;

    ext3::Fe3 acc = ext3::zero();
    for (uint64_t i = threadIdx.x; i < n; i += BARY_BLOCK_DIM) {
        uint64_t eval  = col_data[i];
        uint64_t point = coset_points[i];
        uint64_t pe    = goldilocks::mul(point, eval);   // F * F -> F
        ext3::Fe3 inv_d = ext3::make(
            inv_denoms[i * 3 + 0],
            inv_denoms[i * 3 + 1],
            inv_denoms[i * 3 + 2]);
        ext3::Fe3 term = ext3::mul_base(inv_d, pe);      // E * F -> E
        acc = ext3::add(acc, term);
    }

    ext3::Fe3 sum = block_reduce_ext3(acc);
    if (threadIdx.x == 0) {
        out_ext3_int[col * 3 + 0] = sum.a;
        out_ext3_int[col * 3 + 1] = sum.b;
        out_ext3_int[col * 3 + 2] = sum.c;
    }
}

/// Same as `barycentric_base_batched` but reads rows at stride `row_stride`
/// within each column. Treats the column as an LDE of length `n * row_stride`
/// and sums over the trace-size coset (every `row_stride`-th row). Lets R3 OOD
/// run directly against the LDE device handle from R1 without copying the
/// strided rows into a separate trace-size buffer.
extern "C" __global__ void barycentric_base_batched_strided(
    const uint64_t *columns,
    uint64_t col_stride,
    uint64_t row_stride,
    const uint64_t *coset_points,
    const uint64_t *inv_denoms,
    uint64_t n,
    uint64_t *out_ext3_int
) {
    uint64_t col = blockIdx.x;
    const uint64_t *col_data = columns + col * col_stride;

    ext3::Fe3 acc = ext3::zero();
    for (uint64_t i = threadIdx.x; i < n; i += BARY_BLOCK_DIM) {
        uint64_t eval  = col_data[i * row_stride];
        uint64_t point = coset_points[i];
        uint64_t pe    = goldilocks::mul(point, eval);
        ext3::Fe3 inv_d = ext3::make(
            inv_denoms[i * 3 + 0],
            inv_denoms[i * 3 + 1],
            inv_denoms[i * 3 + 2]);
        ext3::Fe3 term = ext3::mul_base(inv_d, pe);
        acc = ext3::add(acc, term);
    }

    ext3::Fe3 sum = block_reduce_ext3(acc);
    if (threadIdx.x == 0) {
        out_ext3_int[col * 3 + 0] = sum.a;
        out_ext3_int[col * 3 + 1] = sum.b;
        out_ext3_int[col * 3 + 2] = sum.c;
    }
}

/// Ext3-column variant: M ext3 columns stored as 3M base slabs. Column `c`
/// lives at `columns[(c*3+k)*col_stride + i]` for component `k` in 0..3.
extern "C" __global__ void barycentric_ext3_batched(
    const uint64_t *columns,
    uint64_t col_stride,
    const uint64_t *coset_points,
    const uint64_t *inv_denoms,
    uint64_t n,
    uint64_t *out_ext3_int
) {
    uint64_t col = blockIdx.x;
    const uint64_t *slab_a = columns + (col * 3 + 0) * col_stride;
    const uint64_t *slab_b = columns + (col * 3 + 1) * col_stride;
    const uint64_t *slab_c = columns + (col * 3 + 2) * col_stride;

    ext3::Fe3 acc = ext3::zero();
    for (uint64_t i = threadIdx.x; i < n; i += BARY_BLOCK_DIM) {
        ext3::Fe3 eval = ext3::make(slab_a[i], slab_b[i], slab_c[i]);
        uint64_t point = coset_points[i];
        // F * E -> E. Point times eval, componentwise on the 3 base components.
        ext3::Fe3 pe = ext3::mul_base(eval, point);
        // E * E -> E
        ext3::Fe3 inv_d = ext3::make(
            inv_denoms[i * 3 + 0],
            inv_denoms[i * 3 + 1],
            inv_denoms[i * 3 + 2]);
        ext3::Fe3 term = ext3::mul(pe, inv_d);
        acc = ext3::add(acc, term);
    }

    ext3::Fe3 sum = block_reduce_ext3(acc);
    if (threadIdx.x == 0) {
        out_ext3_int[col * 3 + 0] = sum.a;
        out_ext3_int[col * 3 + 1] = sum.b;
        out_ext3_int[col * 3 + 2] = sum.c;
    }
}

/// Strided ext3 variant for R3 OOD of aux LDE.
extern "C" __global__ void barycentric_ext3_batched_strided(
    const uint64_t *columns,
    uint64_t col_stride,
    uint64_t row_stride,
    const uint64_t *coset_points,
    const uint64_t *inv_denoms,
    uint64_t n,
    uint64_t *out_ext3_int
) {
    uint64_t col = blockIdx.x;
    const uint64_t *slab_a = columns + (col * 3 + 0) * col_stride;
    const uint64_t *slab_b = columns + (col * 3 + 1) * col_stride;
    const uint64_t *slab_c = columns + (col * 3 + 2) * col_stride;

    ext3::Fe3 acc = ext3::zero();
    for (uint64_t i = threadIdx.x; i < n; i += BARY_BLOCK_DIM) {
        uint64_t lde_i = i * row_stride;
        ext3::Fe3 eval = ext3::make(slab_a[lde_i], slab_b[lde_i], slab_c[lde_i]);
        uint64_t point = coset_points[i];
        ext3::Fe3 pe = ext3::mul_base(eval, point);
        ext3::Fe3 inv_d = ext3::make(
            inv_denoms[i * 3 + 0],
            inv_denoms[i * 3 + 1],
            inv_denoms[i * 3 + 2]);
        ext3::Fe3 term = ext3::mul(pe, inv_d);
        acc = ext3::add(acc, term);
    }

    ext3::Fe3 sum = block_reduce_ext3(acc);
    if (threadIdx.x == 0) {
        out_ext3_int[col * 3 + 0] = sum.a;
        out_ext3_int[col * 3 + 1] = sum.b;
        out_ext3_int[col * 3 + 2] = sum.c;
    }
}

// Multi-eval-point + row-chunked barycentric. Two fixes over the *_strided
// kernels above: (1) the LDE column data is read ONCE for all K evaluation
// points (K inv_denom blocks, K accumulators) instead of once per point, and
// (2) each column is split into `num_chunks` row ranges so the grid is
// `num_cols * num_chunks` blocks instead of `num_cols` — the single-block-per-
// column grid left most SMs idle at typical column counts. Blocks emit partial
// sums; `barycentric_combine_partials` folds the chunk axis.
//
// `inv_denoms` holds K contiguous blocks of 3N u64 (ext3 interleaved), one per
// evaluation point — the layout `compute_and_invert_denoms_ext3_dev` already
// produces. Partials layout: `[(k*num_cols + col)*num_chunks + chunk]` ext3
// interleaved, so the combine pass reads each (k, col)'s chunks contiguously.
#define BARY_MAX_K 8

extern "C" __global__ void barycentric_base_strided_multi(
    const uint64_t *columns,
    uint64_t col_stride,
    uint64_t row_stride,
    const uint64_t *coset_points,
    const uint64_t *inv_denoms,
    uint64_t n,
    uint64_t k_points,
    uint64_t num_chunks,
    uint64_t *partials
) {
    uint64_t col = blockIdx.x;
    uint64_t chunk = blockIdx.y;
    const uint64_t *col_data = columns + col * col_stride;
    uint64_t chunk_len = (n + num_chunks - 1) / num_chunks;
    uint64_t start = chunk * chunk_len;
    uint64_t end = start + chunk_len < n ? start + chunk_len : n;

    ext3::Fe3 acc[BARY_MAX_K];
    for (uint32_t k = 0; k < k_points; ++k) acc[k] = ext3::zero();

    for (uint64_t i = start + threadIdx.x; i < end; i += BARY_BLOCK_DIM) {
        uint64_t eval  = col_data[i * row_stride];
        uint64_t point = coset_points[i];
        uint64_t pe    = goldilocks::mul(point, eval);
        for (uint32_t k = 0; k < k_points; ++k) {
            const uint64_t *inv = inv_denoms + (uint64_t)k * 3 * n + i * 3;
            ext3::Fe3 inv_d = ext3::make(inv[0], inv[1], inv[2]);
            acc[k] = ext3::add(acc[k], ext3::mul_base(inv_d, pe));
        }
    }

    for (uint32_t k = 0; k < k_points; ++k) {
        ext3::Fe3 sum = block_reduce_ext3(acc[k]);
        if (threadIdx.x == 0) {
            uint64_t o = ((k * gridDim.x + col) * num_chunks + chunk) * 3;
            partials[o + 0] = sum.a;
            partials[o + 1] = sum.b;
            partials[o + 2] = sum.c;
        }
        // block_reduce_ext3 reuses its shared buffers: every thread must be
        // done reading round k's result before round k+1 overwrites them.
        __syncthreads();
    }
}

extern "C" __global__ void barycentric_ext3_strided_multi(
    const uint64_t *columns,
    uint64_t col_stride,
    uint64_t row_stride,
    const uint64_t *coset_points,
    const uint64_t *inv_denoms,
    uint64_t n,
    uint64_t k_points,
    uint64_t num_chunks,
    uint64_t *partials
) {
    uint64_t col = blockIdx.x;
    uint64_t chunk = blockIdx.y;
    const uint64_t *slab_a = columns + (col * 3 + 0) * col_stride;
    const uint64_t *slab_b = columns + (col * 3 + 1) * col_stride;
    const uint64_t *slab_c = columns + (col * 3 + 2) * col_stride;
    uint64_t chunk_len = (n + num_chunks - 1) / num_chunks;
    uint64_t start = chunk * chunk_len;
    uint64_t end = start + chunk_len < n ? start + chunk_len : n;

    ext3::Fe3 acc[BARY_MAX_K];
    for (uint32_t k = 0; k < k_points; ++k) acc[k] = ext3::zero();

    for (uint64_t i = start + threadIdx.x; i < end; i += BARY_BLOCK_DIM) {
        uint64_t lde_i = i * row_stride;
        ext3::Fe3 eval = ext3::make(slab_a[lde_i], slab_b[lde_i], slab_c[lde_i]);
        uint64_t point = coset_points[i];
        ext3::Fe3 pe = ext3::mul_base(eval, point);
        for (uint32_t k = 0; k < k_points; ++k) {
            const uint64_t *inv = inv_denoms + (uint64_t)k * 3 * n + i * 3;
            ext3::Fe3 inv_d = ext3::make(inv[0], inv[1], inv[2]);
            acc[k] = ext3::add(acc[k], ext3::mul(pe, inv_d));
        }
    }

    for (uint32_t k = 0; k < k_points; ++k) {
        ext3::Fe3 sum = block_reduce_ext3(acc[k]);
        if (threadIdx.x == 0) {
            uint64_t o = ((k * gridDim.x + col) * num_chunks + chunk) * 3;
            partials[o + 0] = sum.a;
            partials[o + 1] = sum.b;
            partials[o + 2] = sum.c;
        }
        __syncthreads();
    }
}

// Fold the chunk axis of the multi kernels' partials: one thread per
// (k, col) pair sums its `num_chunks` ext3 partials sequentially (the whole
// buffer is tiny — K * cols * chunks). Output `out_ext3_int[k*num_cols+col]`,
// same per-column layout as the single-point kernels, K blocks concatenated.
extern "C" __global__ void barycentric_combine_partials(
    const uint64_t *partials,
    uint64_t num_chunks,
    uint64_t total,
    uint64_t *out_ext3_int
) {
    uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const uint64_t *row = partials + idx * num_chunks * 3;
    ext3::Fe3 acc = ext3::zero();
    for (uint64_t c = 0; c < num_chunks; ++c) {
        acc = ext3::add(acc, ext3::make(row[c * 3 + 0], row[c * 3 + 1], row[c * 3 + 2]));
    }
    out_ext3_int[idx * 3 + 0] = acc.a;
    out_ext3_int[idx * 3 + 1] = acc.b;
    out_ext3_int[idx * 3 + 2] = acc.c;
}

// Gather full rows from a device-resident base-field LDE (`buf[col*col_stride +
// row]`). One block per gathered row, threads stride over columns. Output is
// row-major `out[q*num_cols + col]` for gathered-row slot `q` — directly the
// concatenation of `gather_main_row(rows[q])` for each q. `rows` are the LDE row
// indices to gather (already the reversed query rows on the host side).
extern "C" __global__ void gather_rows_base(
    const uint64_t *__restrict__ columns,
    uint64_t col_stride,
    uint64_t num_cols,
    const uint32_t *__restrict__ rows,
    uint64_t num_rows,
    uint64_t *__restrict__ out
) {
    uint64_t q = blockIdx.x;
    if (q >= num_rows) return;
    uint64_t row = rows[q];
    for (uint64_t col = threadIdx.x; col < num_cols; col += blockDim.x) {
        out[q * num_cols + col] = columns[col * col_stride + row];
    }
}

// Ext3 variant: M ext3 columns as 3M base slabs, `columns[(col*3+k)*col_stride +
// row]`. Output interleaved ext3: `out[(q*num_cols + col)*3 + k]`.
extern "C" __global__ void gather_rows_ext3(
    const uint64_t *__restrict__ columns,
    uint64_t col_stride,
    uint64_t num_cols,
    const uint32_t *__restrict__ rows,
    uint64_t num_rows,
    uint64_t *__restrict__ out
) {
    uint64_t q = blockIdx.x;
    if (q >= num_rows) return;
    uint64_t row = rows[q];
    for (uint64_t col = threadIdx.x; col < num_cols; col += blockDim.x) {
        uint64_t o = (q * num_cols + col) * 3;
        out[o + 0] = columns[(col * 3 + 0) * col_stride + row];
        out[o + 1] = columns[(col * 3 + 1) * col_stride + row];
        out[o + 2] = columns[(col * 3 + 2) * col_stride + row];
    }
}
