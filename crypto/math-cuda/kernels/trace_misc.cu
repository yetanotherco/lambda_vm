// GPU REGISTER + HALT trace generation (small fixed tables). Mirrors
// generate_register_trace / generate_halt_trace in prover/src/tables/.
//
// REGISTER (5 cols, like PAGE but TS defaults to 1 for never-accessed words):
//   register_dense_kernel   — one thread per row: OFFSET=addr_list[r],
//                             INIT=init[r], FINI=INIT, TS_LO=1; padding rows
//                             (r >= n_real) just set TS_LO=1.
//   register_scatter_kernel — overwrite FINI + TIMESTAMP for accessed words.
// HALT (4 cols, 1 row): halt_kernel writes timestamp + next_pc (DWordWL each).

#include <cstdint>

// REGISTER columns (prover/src/tables/register.rs `cols`).
#define R_OFFSET 0
#define R_INIT 1
#define R_FINI 2
#define R_TS_LO 3
#define R_TS_HI 4

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

extern "C" __global__ void
register_dense_kernel(const uint64_t *__restrict__ addr_list,
                      const uint64_t *__restrict__ init, uint64_t n_real,
                      uint64_t nrows, uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r < n_real) {
    uint64_t iv = init[r];
    sc(cols, nrows, R_OFFSET, r, addr_list[r]);
    sc(cols, nrows, R_INIT, r, iv);
    sc(cols, nrows, R_FINI, r, iv);
    sc(cols, nrows, R_TS_LO, r, 1); // never-accessed → ts=1, fini=init
    sc(cols, nrows, R_TS_HI, r, 0);
  } else {
    // Padding rows: TIMESTAMP_LO=1 (REG-C1 constant ts=1), rest zero.
    sc(cols, nrows, R_TS_LO, r, 1);
  }
}

extern "C" __global__ void
register_scatter_kernel(const uint32_t *__restrict__ rows,
                        const uint64_t *__restrict__ values,
                        const uint64_t *__restrict__ timestamps,
                        uint64_t n_cells, uint64_t nrows,
                        uint64_t *__restrict__ cols) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_cells)
    return;
  uint64_t r = rows[i];
  uint64_t ts = timestamps[i];
  sc(cols, nrows, R_FINI, r, values[i]);
  sc(cols, nrows, R_TS_LO, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, R_TS_HI, r, ts >> 32);
}

// HALT: single row, 4 cols [TIMESTAMP_0, TIMESTAMP_1, PC_0, PC_1].
extern "C" __global__ void
halt_kernel(uint64_t timestamp, uint64_t next_pc, uint64_t *__restrict__ cols) {
  if (blockIdx.x != 0 || threadIdx.x != 0)
    return;
  cols[0] = timestamp & 0xFFFFFFFF;
  cols[1] = timestamp >> 32;
  cols[2] = next_pc & 0xFFFFFFFF;
  cols[3] = next_pc >> 32;
}
