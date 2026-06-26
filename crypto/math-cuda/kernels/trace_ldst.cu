// GPU LOAD and STORE trace generation. One thread per row, no dedup (μ=1 each).
// Mirrors generate_load_trace / generate_store_trace in
// prover/src/tables/{load,store}.rs. Padding rows (r >= n) stay all-zero
// (MU=0), matching the CPU builders. Raw canonical u64 throughout (addresses
// and timestamps split into <2^32 limbs; result/value bytes are < 256).

#include "goldilocks.cuh"
#include <cstdint>

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}
__device__ __forceinline__ uint64_t from_u64(uint64_t x) {
  return x >= goldilocks::PRIME ? x - goldilocks::PRIME : x;
}

// ---------------------------------------------------------------------------
// LOAD: 18 columns. Per-op input stride LOAD_STRIDE:
//   [base_address, timestamp, width, signed, res0, res1, ..., res7]
// ---------------------------------------------------------------------------
#define L_BASE_ADDRESS_0 0
#define L_TIMESTAMP_0 2
#define L_READ2 4
#define L_READ4 5
#define L_READ8 6
#define L_SIGNED 7
#define L_RES 8 // res[0..8] at cols 8..16
#define L_SIGN_BIT 16
#define L_MU 17
#define LOAD_STRIDE 12

extern "C" __global__ void
trace_load_kernel(const uint64_t *__restrict__ in, uint64_t n, uint64_t nrows,
                  uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return; // padding rows already zeroed

  const uint64_t *op = in + r * LOAD_STRIDE;
  uint64_t base = op[0];
  uint64_t ts = op[1];
  uint64_t width = op[2];
  uint64_t is_signed = op[3];
  const uint64_t *res = op + 4;

  sc(cols, nrows, L_BASE_ADDRESS_0, r, base & 0xFFFFFFFF);
  sc(cols, nrows, L_BASE_ADDRESS_0 + 1, r, base >> 32);
  sc(cols, nrows, L_TIMESTAMP_0, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, L_TIMESTAMP_0 + 1, r, ts >> 32);

  // read flags: "exactly N" semantics.
  sc(cols, nrows, L_READ2, r, width == 2);
  sc(cols, nrows, L_READ4, r, width == 4);
  sc(cols, nrows, L_READ8, r, width == 8);
  sc(cols, nrows, L_SIGNED, r, is_signed);

  for (int i = 0; i < 8; i++)
    sc(cols, nrows, L_RES + i, r, from_u64(res[i]));

  // sign_bit = MSB (bit 7) of the highest byte read (idx by width).
  int byte_idx = width == 8 ? 7 : (width == 4 ? 3 : (width == 2 ? 1 : 0));
  sc(cols, nrows, L_SIGN_BIT, r, (res[byte_idx] >> 7) & 1);

  sc(cols, nrows, L_MU, r, 1);
}

// ---------------------------------------------------------------------------
// STORE: 16 columns. Per-op input stride STORE_STRIDE:
//   [base_address, timestamp, value, write_flags]
// write_flags packs write2|write4<<1|write8<<2.
// ---------------------------------------------------------------------------
#define S_BASE_ADDRESS_0 0
#define S_TIMESTAMP_0 2
#define S_WRITE2 4
#define S_WRITE4 5
#define S_WRITE8 6
#define S_VALUE 7 // value bytes at cols 7..15 (DWordBL)
#define S_MU 15
#define STORE_STRIDE 4

extern "C" __global__ void
trace_store_kernel(const uint64_t *__restrict__ in, uint64_t n, uint64_t nrows,
                   uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return; // padding rows already zeroed

  const uint64_t *op = in + r * STORE_STRIDE;
  uint64_t base = op[0];
  uint64_t ts = op[1];
  uint64_t value = op[2];
  uint64_t wflags = op[3];

  sc(cols, nrows, S_BASE_ADDRESS_0, r, base & 0xFFFFFFFF);
  sc(cols, nrows, S_BASE_ADDRESS_0 + 1, r, base >> 32);
  sc(cols, nrows, S_TIMESTAMP_0, r, ts & 0xFFFFFFFF);
  sc(cols, nrows, S_TIMESTAMP_0 + 1, r, ts >> 32);
  sc(cols, nrows, S_WRITE2, r, wflags & 1);
  sc(cols, nrows, S_WRITE4, r, (wflags >> 1) & 1);
  sc(cols, nrows, S_WRITE8, r, (wflags >> 2) & 1);

  // value as 8 little-endian bytes (DWordBL).
  for (int i = 0; i < 8; i++)
    sc(cols, nrows, S_VALUE + i, r, (value >> (i * 8)) & 0xFF);

  sc(cols, nrows, S_MU, r, 1);
}
