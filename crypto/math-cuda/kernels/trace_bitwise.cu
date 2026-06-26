// GPU BITWISE-table generation. The table is the fixed 2^20-row lookup
// (256 x 256 x 16, indexed row = x + 256*y + 65536*z) with 11 precomputed
// columns + 10 multiplicity columns. Mirrors generate_bitwise_trace +
// update_multiplicities in prover/src/tables/bitwise.rs.
//
//   bitwise_fixed_kernel  — one thread per row, fills precomputed cols 0..10.
//   bitwise_hist_kernel   — one thread per lookup, atomic scatter-add into the
//                           multiplicity column (11 + type) at the lookup's row.
//
// All values are < 2^16 (bytes / halfword shifts / bits), written raw.

#include <cstdint>

// Column indices (must match prover/src/tables/bitwise.rs `cols`).
#define B_X 0
#define B_Y 1
#define B_Z 2
#define B_AND 3
#define B_OR 4
#define B_XOR 5
#define B_MSB8 6
#define B_MSB16 7
#define B_ZERO 8
#define B_SLL 9
#define B_SLLC 10
// Multiplicity columns 11..20 = 11 + (BitwiseOperationType as u32), where the
// enum order is Msb8,Msb16,Zero,AreBytes,IsHalf,IsB20,Hwsl,ByteAluAnd,
// ByteAluOr,ByteAluXor.
#define B_MU_BASE 11

#define B_NROWS (1ULL << 20) // 256 * 256 * 16

__device__ __forceinline__ void sc(uint64_t *cols, int col, uint64_t r,
                                    uint64_t v) {
  cols[(uint64_t)col * B_NROWS + r] = v;
}

extern "C" __global__ void
bitwise_fixed_kernel(uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= B_NROWS)
    return;
  // r = x + 256*y + 65536*z, with z < 16.
  uint32_t x = (uint32_t)(r & 0xFF);
  uint32_t y = (uint32_t)((r >> 8) & 0xFF);
  uint32_t z = (uint32_t)((r >> 16) & 0xF);

  sc(cols, B_X, r, x);
  sc(cols, B_Y, r, y);
  sc(cols, B_Z, r, z);
  sc(cols, B_AND, r, x & y);
  sc(cols, B_OR, r, x | y);
  sc(cols, B_XOR, r, x ^ y);
  sc(cols, B_MSB8, r, (x >> 7) & 1);
  uint32_t halfword = x + y * 256;
  sc(cols, B_MSB16, r, (halfword >> 15) & 1);
  sc(cols, B_ZERO, r, (x == 0 && y == 0 && z == 0) ? 1 : 0);
  uint32_t sll = (z == 0) ? halfword : ((halfword << z) & 0xFFFF);
  uint32_t sllc = (z == 0) ? 0u : (halfword >> (16 - z));
  sc(cols, B_SLL, r, sll);
  sc(cols, B_SLLC, r, sllc);
}

// Each lookup is packed into a u32: x | y<<8 | z<<16 | type<<20.
extern "C" __global__ void
bitwise_hist_kernel(const uint32_t *__restrict__ ops, uint64_t n,
                    uint64_t *__restrict__ cols) {
  uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  uint32_t op = ops[i];
  uint64_t x = op & 0xFF;
  uint64_t y = (op >> 8) & 0xFF;
  uint64_t z = (op >> 16) & 0xF;
  uint64_t type = (op >> 20) & 0xF;
  uint64_t row = x + (y << 8) + (z << 16);
  uint64_t idx = (B_MU_BASE + type) * B_NROWS + row;
  atomicAdd(reinterpret_cast<unsigned long long *>(&cols[idx]), 1ULL);
}
