// GPU BRANCH-table fill. One thread per (host-deduped) op; dedup by
// (pc, offset, register, jalr) with summed multiplicity is done on the host
// (small table). Mirrors generate_branch_trace in prover/src/tables/branch.rs.
// Padding rows (r >= n) stay zero.
//
// Per-op input is interleaved, stride BR_STRIDE: [pc, offset, register, jalr,
// mult]. pc/offset/register are 64-bit (split into <2^32 limbs); next_pc is
// derived: base = jalr ? register : pc; unmasked = base + offset (wrapping);
// next_pc = unmasked & ~1.

#include <cstdint>

#define BR_PC_0 0
#define BR_OFFSET_0 2
#define BR_REGISTER_0 4
#define BR_JALR 6
#define BR_NEXT_PC_HIGH_0 7
#define BR_NEXT_PC_LOW_0 10
#define BR_UNMASKED_LOW_BYTE 12
#define BR_MU 13
#define BR_STRIDE 5

__device__ __forceinline__ void sc(uint64_t *cols, uint64_t nrows, int col,
                                    uint64_t r, uint64_t v) {
  cols[(uint64_t)col * nrows + r] = v;
}

extern "C" __global__ void
trace_branch_kernel(const uint64_t *__restrict__ in, uint64_t n, uint64_t nrows,
                    uint64_t *__restrict__ cols) {
  uint64_t r = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= nrows)
    return;
  if (r >= n)
    return; // padding rows already zeroed

  const uint64_t *op = in + r * BR_STRIDE;
  uint64_t pc = op[0];
  uint64_t offset = op[1];
  uint64_t reg = op[2];
  uint64_t jalr = op[3];
  uint64_t mult = op[4];

  uint64_t base = jalr ? reg : pc;
  uint64_t unmasked = base + offset; // u64 wraps naturally
  uint64_t next_pc = unmasked & ~1ULL;

  sc(cols, nrows, BR_PC_0, r, pc & 0xFFFFFFFF);
  sc(cols, nrows, BR_PC_0 + 1, r, pc >> 32);
  sc(cols, nrows, BR_OFFSET_0, r, offset & 0xFFFFFFFF);
  sc(cols, nrows, BR_OFFSET_0 + 1, r, offset >> 32);
  sc(cols, nrows, BR_REGISTER_0, r, reg & 0xFFFFFFFF);
  sc(cols, nrows, BR_REGISTER_0 + 1, r, reg >> 32);
  sc(cols, nrows, BR_JALR, r, jalr);
  sc(cols, nrows, BR_NEXT_PC_HIGH_0, r, (next_pc >> 16) & 0xFFFF);
  sc(cols, nrows, BR_NEXT_PC_HIGH_0 + 1, r, (next_pc >> 32) & 0xFFFF);
  sc(cols, nrows, BR_NEXT_PC_HIGH_0 + 2, r, (next_pc >> 48) & 0xFFFF);
  sc(cols, nrows, BR_NEXT_PC_LOW_0, r, next_pc & 0xFF);
  sc(cols, nrows, BR_NEXT_PC_LOW_0 + 1, r, (next_pc >> 8) & 0xFF);
  sc(cols, nrows, BR_UNMASKED_LOW_BYTE, r, unmasked & 0xFF);
  sc(cols, nrows, BR_MU, r, mult);
}
