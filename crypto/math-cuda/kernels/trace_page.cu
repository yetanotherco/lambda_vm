// GPU PAGE-table generation (one table per touched page). Mirrors
// generate_page_trace in prover/src/tables/page.rs. 5 columns:
//   OFFSET(0)=row index, INIT(1)=initial byte, FINI(2)=final byte,
//   TIMESTAMP_LO(3), TIMESTAMP_HI(4).
// OFFSET + INIT are preprocessed; FINI + TIMESTAMP are per-proof (final memory
// state). Built as: dense fill (FINI=INIT, TS=0 for never-accessed bytes) then
// a sparse scatter of the page's touched cells (overwrite FINI + TIMESTAMP).
//
//   page_dense_kernel   — one thread per row (offset in 0..page_size).
//   page_scatter_kernel — one thread per touched cell.
//
// All values are bytes (<256) or <2^32 timestamp limbs, written raw.

#include <cstdint>

#define P_OFFSET 0
#define P_INIT 1
#define P_FINI 2
#define P_TS_LO 3
#define P_TS_HI 4

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

// Dense fill: OFFSET, INIT (from init_values[0..init_len], 0 beyond), and
// FINI = INIT / TS = 0 for bytes never accessed at runtime.
extern "C" __global__ void
page_dense_kernel(const uint8_t *__restrict__ init_values, uint64_t init_len,
                  uint64_t page_size, uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= page_size)
    return;
  uint64_t init = (r < init_len) ? (uint64_t)init_values[r] : 0ULL;
  sc(cols, page_size, P_OFFSET, r, r);
  sc(cols, page_size, P_INIT, r, init);
  sc(cols, page_size, P_FINI, r, init);
  sc(cols, page_size, P_TS_LO, r, 0);
  sc(cols, page_size, P_TS_HI, r, 0);
}

// Scatter: for each touched cell, overwrite FINI + TIMESTAMP at its offset.
extern "C" __global__ void
page_scatter_kernel(const uint32_t *__restrict__ offsets,
                    const uint8_t *__restrict__ values,
                    const uint64_t *__restrict__ timestamps, uint64_t n_cells,
                    uint64_t page_size, uint64_t *__restrict__ cols) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells)
    return;
  uint64_t r = offsets[i];
  uint64_t ts = timestamps[i];
  sc(cols, page_size, P_FINI, r, (uint64_t)values[i]);
  sc(cols, page_size, P_TS_LO, r, ts & 0xFFFFFFFF);
  sc(cols, page_size, P_TS_HI, r, ts >> 32);
}
