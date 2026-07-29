// Radix-2 DIT NTT over Goldilocks: per-level, fused 8-level (shmem), and
// batched (multi-column) variants. The caller runs `bit_reverse_permute`
// once before the first butterfly level.
//
// Input layout: bit-reversed-order coefficients (after `bit_reverse_permute`).
// Output layout: natural-order evaluations — matches the CPU `evaluate_fft` contract.
//
// Twiddle table: `tw[i] = ω^i` for i in [0, n/2). Stride-indexed per level.

#include "goldilocks.cuh"

using goldilocks::add;
using goldilocks::sub;
using goldilocks::mul;

/// Reverse the low `log_n` bits of each index and swap x[i] ↔ x[rev(i)].
/// One thread per index; guarded by `tid < rev` to avoid double-swap.
extern "C" __global__ void bit_reverse_permute(uint64_t *x,
                                               uint64_t n,
                                               uint64_t log_n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;

    // __brevll reverses all 64 bits; shift right so result lives in [0, n).
    uint64_t rev = __brevll(tid) >> (64 - log_n);
    if (tid < rev) {
        uint64_t tmp = x[tid];
        x[tid] = x[rev];
        x[rev] = tmp;
    }
}

/// Pointwise multiply: x[i] *= w[i]. Used for coset scaling (w = g^i weights).
extern "C" __global__ void pointwise_mul(uint64_t *x,
                                         const uint64_t *w,
                                         uint64_t n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) x[tid] = mul(x[tid], w[tid]);
}

/// Broadcast scalar multiply: x[i] *= c. Used for the 1/n factor at the end of iNTT.
extern "C" __global__ void scalar_mul(uint64_t *x,
                                      uint64_t c,
                                      uint64_t n) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) x[tid] = mul(x[tid], c);
}

// ============================================================================
// BATCHED KERNELS
//
// One launch processes M columns at once. The device buffer holds M columns
// back-to-back; column `c` starts at `data + c * col_stride`. gridDim.y is
// the column index, so each block handles one (column, butterfly-window) pair.
//
// The same twiddle table is shared across all columns of a batch (they all
// NTT on the same domain). The coset weights are also shared.
// ============================================================================

extern "C" __global__ void bit_reverse_permute_batched(uint64_t *data,
                                                       uint64_t n,
                                                       uint64_t log_n,
                                                       uint64_t col_stride) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t *x = data + (uint64_t)blockIdx.y * col_stride;

    uint64_t rev = __brevll(tid) >> (64 - log_n);
    if (tid < rev) {
        uint64_t tmp = x[tid];
        x[tid] = x[rev];
        x[rev] = tmp;
    }
}

extern "C" __global__ void ntt_dit_level_batched(uint64_t *data,
                                                 const uint64_t *tw,
                                                 uint64_t n,
                                                 uint64_t log_n,
                                                 uint64_t level,
                                                 uint64_t col_stride) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_half = n >> 1;
    if (tid >= n_half) return;
    uint64_t *x = data + (uint64_t)blockIdx.y * col_stride;

    uint64_t half       = 1ULL << level;
    uint64_t block_size = half << 1;
    uint64_t block_idx  = tid >> level;
    uint64_t k          = tid & (half - 1);

    uint64_t i0 = block_idx * block_size + k;
    uint64_t i1 = i0 + half;

    uint64_t tw_index = k << (log_n - level - 1);
    uint64_t w = tw[tw_index];

    uint64_t u = x[i0];
    uint64_t v = mul(w, x[i1]);
    x[i0] = add(u, v);
    x[i1] = sub(u, v);
}

extern "C" __global__ void ntt_dit_8_levels_batched(uint64_t *data,
                                                    const uint64_t *tw,
                                                    uint64_t n,
                                                    uint64_t log_n,
                                                    uint64_t base_step,
                                                    uint64_t col_stride) {
    __shared__ uint64_t tile[256];
    uint64_t *x = data + (uint64_t)blockIdx.y * col_stride;

    uint32_t n_loc_steps = (uint32_t)min((uint64_t)8, log_n - base_step);

    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    uint64_t group_size = 1ULL << base_step;
    uint64_t n_groups   = n >> base_step;
    uint64_t low_bits   = tid / n_groups;
    uint64_t high_bits  = tid & (n_groups - 1);
    uint64_t row        = high_bits * group_size + low_bits;

    tile[threadIdx.x] = x[row];
    __syncthreads();

    uint32_t remaining_high_bits = (uint32_t)(log_n - base_step - 1);
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;

    for (uint32_t loc_step = 0; loc_step < n_loc_steps; ++loc_step) {
        if (threadIdx.x < 128) {
            uint32_t i      = threadIdx.x;
            uint32_t half   = 1u << loc_step;
            uint32_t grp    = i >> loc_step;
            uint32_t grp_pos = i & (half - 1);
            uint32_t idx1 = (grp << (loc_step + 1)) + grp_pos;
            uint32_t idx2 = idx1 + half;

            uint32_t gs  = (uint32_t)base_step + loc_step;
            uint32_t ggp = (blockIdx.x << 7) + i;
            ggp = ((ggp & high_mask) << (uint32_t)base_step) + (ggp >> remaining_high_bits);
            ggp = ggp & ((1u << gs) - 1u);
            uint64_t factor = tw[(uint64_t)ggp * (n >> (gs + 1))];

            uint64_t u = tile[idx1];
            uint64_t v = mul(tile[idx2], factor);
            tile[idx1] = add(u, v);
            tile[idx2] = sub(u, v);
        }
        __syncthreads();
    }

    x[row] = tile[threadIdx.x];
}


/// Batched pointwise multiply: first n elements of each column multiplied by
/// the SHARED weight vector `w` (size n). Used for coset scaling — every
/// column of a table sees the same `g^i / N` weights.
extern "C" __global__ void pointwise_mul_batched(uint64_t *data,
                                                 const uint64_t *w,
                                                 uint64_t n,
                                                 uint64_t col_stride) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t *x = data + (uint64_t)blockIdx.y * col_stride;
    x[tid] = mul(x[tid], w[tid]);
}

/// Batched broadcast scalar multiply — one scalar c applied to the first n
/// elements of every column.
extern "C" __global__ void scalar_mul_batched(uint64_t *data,
                                              uint64_t c,
                                              uint64_t n,
                                              uint64_t col_stride) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    uint64_t *x = data + (uint64_t)blockIdx.y * col_stride;
    x[tid] = mul(x[tid], c);
}

/// One DIT butterfly level. Thread `tid` (of n/2 total) owns exactly one
/// butterfly pair (i0, i1 = i0 + half). Twiddle picked from the shared full
/// `tw` table at stride `n / block_size`. Used for levels 0..7 when n < 256
/// (shmem fusion needs at least 256 elements), and for levels >= 8 of any
/// size (above the shmem-fusion window).
extern "C" __global__ void ntt_dit_level(uint64_t *x,
                                         const uint64_t *tw,
                                         uint64_t n,
                                         uint64_t log_n,
                                         uint64_t level) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_half = n >> 1;
    if (tid >= n_half) return;

    uint64_t half       = 1ULL << level;          // 2^ℓ
    uint64_t block_size = half << 1;              // 2^{ℓ+1}
    uint64_t block_idx  = tid >> level;           // floor(tid / half)
    uint64_t k          = tid & (half - 1);       // tid mod half

    uint64_t i0 = block_idx * block_size + k;
    uint64_t i1 = i0 + half;

    // Stride = n / block_size = n >> (level + 1).
    uint64_t tw_index = k << (log_n - level - 1);
    uint64_t w = tw[tw_index];

    uint64_t u = x[i0];
    uint64_t v = mul(w, x[i1]);
    x[i0] = add(u, v);
    x[i1] = sub(u, v);
}

/// Up to 8 DIT butterfly levels fused in one kernel using shared memory.
///
/// Ported from Zisk's `br_ntt_8_steps` (`pil2-stark/src/goldilocks/src/ntt_goldilocks.cu`),
/// simplified to single-column. Each block of 256 threads processes 256
/// elements in on-chip shared memory, running up to 8 butterfly levels
/// without writing to global memory between them — cuts DRAM traffic by up
/// to 8× vs the per-level kernel.
///
/// `base_step` selects which 8-level window this launch handles (0, 8, 16, ...).
/// For levels 0–7 the implicit DIT element layout already places all pair
/// mates inside the same 256-block; for higher base_step we remap the loaded
/// row so pair mates land in consecutive shared-memory slots.
///
/// Expects bit-reversed input (the caller runs `bit_reverse_permute` once
/// before the first kernel launch).
///
/// Assumes `n` is a multiple of 256, i.e. `log_n >= 8`.
extern "C" __global__ void ntt_dit_8_levels(uint64_t *x,
                                            const uint64_t *tw,
                                            uint64_t n,
                                            uint64_t log_n,
                                            uint64_t base_step) {
    __shared__ uint64_t tile[256];

    uint32_t n_loc_steps = (uint32_t)min((uint64_t)8, log_n - base_step);

    // tid is the *unpermuted* flat index the block/thread would own.
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Row remap: for base_step > 0, gather elements that pair at levels
    // `base_step..base_step+7` so they land consecutively in the block.
    uint64_t group_size = 1ULL << base_step;
    uint64_t n_groups   = n >> base_step;  // = n / group_size
    uint64_t low_bits   = tid / n_groups;
    uint64_t high_bits  = tid & (n_groups - 1);
    uint64_t row        = high_bits * group_size + low_bits;

    // Load one element per thread.
    tile[threadIdx.x] = x[row];
    __syncthreads();

    // Each butterfly level uses half the threads (128 butterflies per block).
    // The global butterfly index `ggp` is recovered from blockIdx + threadIdx
    // and reshaped by the same row-remap to find the right twiddle.
    uint32_t remaining_high_bits = (uint32_t)(log_n - base_step - 1);  // log2(n_groups / 2)
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;

    for (uint32_t loc_step = 0; loc_step < n_loc_steps; ++loc_step) {
        if (threadIdx.x < 128) {
            uint32_t i      = threadIdx.x;
            uint32_t half   = 1u << loc_step;
            uint32_t grp    = i >> loc_step;
            uint32_t grp_pos = i & (half - 1);
            uint32_t idx1 = (grp << (loc_step + 1)) + grp_pos;
            uint32_t idx2 = idx1 + half;

            // Global step and butterfly position for twiddle lookup.
            uint32_t gs  = (uint32_t)base_step + loc_step;
            uint32_t ggp = (blockIdx.x << 7) + i;  // blockIdx * 128 + i
            // Un-remap ggp to find its position in the natural ordering.
            ggp = ((ggp & high_mask) << (uint32_t)base_step) + (ggp >> remaining_high_bits);
            ggp = ggp & ((1u << gs) - 1u);
            uint64_t factor = tw[(uint64_t)ggp * (n >> (gs + 1))];

            uint64_t u = tile[idx1];
            uint64_t v = mul(tile[idx2], factor);
            tile[idx1] = add(u, v);
            tile[idx2] = sub(u, v);
        }
        __syncthreads();
    }

    // Store back to the remapped row.
    x[row] = tile[threadIdx.x];
}

// ============================================================================
// ROW-MAJOR BATCHED KERNELS
//
// Data layout: data[row * m + col] for n rows and m columns.
// threadIdx.x = column index → consecutive threads access consecutive columns
// of the same row → coalesced global memory access.
// Twiddle factors depend only on the butterfly position, not the column →
// one twiddle load is broadcast across the entire warp.
// ============================================================================

// Bit-reverse permute rows: swap row `row` with row `br(row)`.
// Grid: gridDim.x = ceil(m / 256), gridDim.y = min(n, 65535).
// Grid-stride loop over rows so a capped gridDim.y covers all n rows.
extern "C" __global__ void bit_reverse_row_major(uint64_t *data,
                                                  uint64_t n,
                                                  uint64_t log_n,
                                                  uint64_t m)
{
    uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= m) return;
    for (uint64_t row = blockIdx.y; row < n; row += gridDim.y) {
        uint64_t rev = __brevll(row) >> (64 - log_n);
        if (row < rev) {
            uint64_t tmp          = data[row * m + col];
            data[row * m + col]   = data[rev * m + col];
            data[rev * m + col]   = tmp;
        }
    }
}

// One DIT butterfly level on row-major data.
// Grid: gridDim.x = ceil(m / blockDim.x), gridDim.y = min(ceil(n/2 / blockDim.y), 65535).
// blockDim.x covers columns (coalescing), blockDim.y covers butterfly pairs.
// Grid-stride loop over butterfly-pair tiles so capped gridDim.y covers all n/2 pairs.
extern "C" __global__ void ntt_dit_level_row_major(uint64_t *data,
                                                    const uint64_t *tw,
                                                    uint64_t n,
                                                    uint64_t log_n,
                                                    uint64_t level,
                                                    uint64_t m)
{
    uint64_t col    = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n_half = n >> 1;
    if (col >= m) return;

    uint64_t half       = 1ULL << level;
    uint64_t block_size = half << 1;

    for (uint64_t bfly_base = blockIdx.y * blockDim.y;
         bfly_base < n_half;
         bfly_base += (uint64_t)gridDim.y * blockDim.y) {
        uint64_t butterfly = bfly_base + threadIdx.y;
        if (butterfly >= n_half) break;

        uint64_t block_idx = butterfly >> level;
        uint64_t k         = butterfly & (half - 1);
        uint64_t i0        = block_idx * block_size + k;
        uint64_t i1        = i0 + half;

        // Same twiddle for all columns at this butterfly position (broadcast).
        uint64_t w = tw[k << (log_n - level - 1)];

        uint64_t u = data[i0 * m + col];
        uint64_t v = mul(w, data[i1 * m + col]);
        data[i0 * m + col] = add(u, v);
        data[i1 * m + col] = sub(u, v);
    }
}

// Pointwise multiply row-major: data[row * m + col] *= weights[row].
// One weight per row, broadcast across all m columns.
// Grid: gridDim.x = ceil(m / 256), gridDim.y = min(n, 65535).
// Grid-stride loop over rows.
extern "C" __global__ void pointwise_mul_row_major(uint64_t *data,
                                                    const uint64_t *weights,
                                                    uint64_t n,
                                                    uint64_t m)
{
    uint64_t col = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= m) return;
    for (uint64_t row = blockIdx.y; row < n; row += gridDim.y)
        data[row * m + col] = mul(data[row * m + col], weights[row]);
}

// ── Row-major → column-major transpose (for GpuLdeBase handle) ───────────────
//
// Converts the row-major LDE output to the column-major layout that downstream
// GPU kernels (DEEP, barycentric) require for the device handle.
//
// src[r * cols + c]  →  dst[c * out_stride + r]
//
// Grid: gridDim.x = ceil(cols/32), gridDim.y = min(ceil(rows/32), 65535).
// Grid-strides over row tiles so all rows are covered when rows > 65535*32.

#define MTILE 32
#define MTILE_P (MTILE + 1)

extern "C" __global__ void matrix_transpose_strided(
    const uint64_t *__restrict__ src,
    uint64_t *__restrict__ dst,
    uint32_t rows,
    uint32_t cols,
    uint64_t out_stride)
{
    __shared__ uint64_t tile[MTILE][MTILE_P];

    for (uint32_t row_base = blockIdx.y * MTILE; row_base < rows;
         row_base += gridDim.y * MTILE) {
        uint32_t x = blockIdx.x * MTILE + threadIdx.x;
        uint32_t y = row_base + threadIdx.y;

        if (x < cols && y < rows)
            tile[threadIdx.y][threadIdx.x] = src[(uint64_t)y * cols + x];

        __syncthreads();

        uint32_t tx = row_base + threadIdx.x;
        uint32_t ty = blockIdx.x * MTILE + threadIdx.y;

        if (tx < rows && ty < cols)
            dst[(uint64_t)ty * out_stride + tx] = tile[threadIdx.x][threadIdx.y];

        __syncthreads();
    }
}

// First-8-levels fused DIT on row-major data: one block stages 256 consecutive
// rows x blockDim.x columns in shmem and runs levels 0..min(8,log_n) with
// __syncthreads between levels (row-major analog of ntt_dit_8_levels_batched
// with base_step == 0, whose twiddle math this reuses verbatim). Grid:
// x = column tiles, y = n/256 row blocks. Requires n >= 256. Shmem tile is
// padded (pitch = T+1) to break bank conflicts on the butterfly accesses.
extern "C" __global__ void ntt_dit_8_levels_row_major(uint64_t *data,
                                                      const uint64_t *tw,
                                                      uint64_t n,
                                                      uint64_t log_n,
                                                      uint64_t m)
{
    extern __shared__ uint64_t tile[];
    uint32_t T = blockDim.x;
    uint32_t pitch = T + 1;
    uint64_t col = (uint64_t)blockIdx.x * T + threadIdx.x;
    bool live = col < m;
    uint64_t row_base = (uint64_t)blockIdx.y * 256;

    for (uint32_t r = threadIdx.y; r < 256; r += blockDim.y) {
        if (live) tile[r * pitch + threadIdx.x] = data[(row_base + r) * m + col];
    }
    __syncthreads();

    uint32_t n_loc_steps = (uint32_t)min((uint64_t)8, log_n);
    uint32_t remaining_high_bits = (uint32_t)(log_n - 1);
    uint32_t high_mask = (1u << remaining_high_bits) - 1u;

    for (uint32_t loc_step = 0; loc_step < n_loc_steps; ++loc_step) {
        for (uint32_t i = threadIdx.y; i < 128; i += blockDim.y) {
            uint32_t half    = 1u << loc_step;
            uint32_t grp     = i >> loc_step;
            uint32_t grp_pos = i & (half - 1);
            uint32_t idx1 = (grp << (loc_step + 1)) + grp_pos;
            uint32_t idx2 = idx1 + half;

            uint32_t gs  = loc_step;
            uint32_t ggp = ((uint32_t)blockIdx.y << 7) + i;
            ggp = (ggp & high_mask) + (ggp >> remaining_high_bits);
            ggp = ggp & ((1u << gs) - 1u);
            uint64_t factor = tw[(uint64_t)ggp * (n >> (gs + 1))];

            if (live) {
                uint64_t u = tile[idx1 * pitch + threadIdx.x];
                uint64_t v = mul(tile[idx2 * pitch + threadIdx.x], factor);
                tile[idx1 * pitch + threadIdx.x] = add(u, v);
                tile[idx2 * pitch + threadIdx.x] = sub(u, v);
            }
        }
        __syncthreads();
    }

    for (uint32_t r = threadIdx.y; r < 256; r += blockDim.y) {
        if (live) data[(row_base + r) * m + col] = tile[r * pitch + threadIdx.x];
    }
}
