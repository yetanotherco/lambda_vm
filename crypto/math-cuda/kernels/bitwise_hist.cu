// On-GPU BITWISE multiplicity histogram (the prover's `BitwiseHistogram`).
//
// The counter array is `hist[type_idx * num_rows + row_index(x, y, z)]` (u64), with
// `row_index(x, y, z) = x + y*256 + z*65536`. This kernel emits the per-CPU-op in-walk
// range-check lookups and atomic-adds them — the device analog of the host
// `add_ops(collect_bitwise_ops(cpu_ops))`, which alone is ~half of all BITWISE bumps.
//
// Mirrors `CpuOperation::collect_bitwise_ops` bit-for-bit: 3 ARE_BYTES (type 3) + 4
// IS_HALF (type 4). Word (`*W`) instruction rows zero rs1/rs2/rd/alu_flags/mem_flags
// and res (CPU32 emits their real range checks); half_instruction_length is never
// zeroed. `row_index(x, y, 0) = x + y*256`, and an IS_HALF halfword `h` maps to
// `(h & 0xFF) + (h >> 8)*256 == h`.

#include <cstdint>

// `num_copies` REPLICATED histograms defuse atomic contention: the ARE_BYTES lookups
// key on register indices (~1K hot bins), so naive single-copy global atomics serialize.
// Each block accumulates into copy `blockIdx.x % num_copies` (stride `copy_stride =
// num_rows * num_types`), spreading each hot address across `num_copies` addresses;
// `bitwise_hist_reduce` then sums the copies. Concurrent blocks land on distinct copies.
extern "C" __global__ void bitwise_hist_cpu_ops(
    uint64_t n, const uint8_t *rs1, const uint8_t *rs2, const uint8_t *rd,
    const uint8_t *hil, const uint8_t *alu_flags, const uint8_t *mem_flags,
    const uint64_t *res, const uint8_t *word, uint64_t num_rows, uint32_t num_copies,
    uint64_t copy_stride, unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    bool w = word[i] != 0;
    uint64_t z_rs1 = w ? 0u : rs1[i];
    uint64_t z_rs2 = w ? 0u : rs2[i];
    uint64_t z_rd = w ? 0u : rd[i];
    uint64_t z_alu = w ? 0u : alu_flags[i];
    uint64_t z_mem = w ? 0u : mem_flags[i];
    uint64_t h = hil[i]; // half_instruction_length — NOT zeroed on word rows
    uint64_t r = w ? 0ull : res[i];

    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    const uint64_t are = base + 3ull * num_rows; // AreBytes lane in this copy
    const uint64_t ish = base + 4ull * num_rows; // IsHalf lane in this copy

    atomicAdd(&hist[are + z_rs1 + z_rs2 * 256ull], 1ull);
    atomicAdd(&hist[are + z_rd + h * 256ull], 1ull);
    atomicAdd(&hist[are + z_alu + z_mem * 256ull], 1ull);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        uint64_t half = (r >> (k * 16)) & 0xFFFFull;
        atomicAdd(&hist[ish + half], 1ull);
    }
}

// MEMW_R source: one IS_HALF lookup per register row, keyed by the timestamp delta
// `diff = ts_lo - old_ts_lo - 1` (u16). Mirrors `memw_register_is_half_lookup`; IS_HALF
// is type 4 and `row_index(diff&0xff, diff>>8, 0) == diff`. Accumulates into the same
// replicated histogram as `bitwise_hist_cpu_ops`.
extern "C" __global__ void bitwise_hist_memw_reg(uint64_t n, const uint64_t *ts,
                                                 const uint64_t *old_ts, uint64_t num_rows,
                                                 uint32_t num_copies, uint64_t copy_stride,
                                                 unsigned long long *hist) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint32_t ts_lo = (uint32_t)(ts[i] & 0xFFFFFFFFull);
    uint32_t ot_lo = (uint32_t)(old_ts[i] & 0xFFFFFFFFull);
    uint64_t diff = (uint64_t)((ts_lo - ot_lo - 1u) & 0xFFFFu);
    uint64_t base = (uint64_t)(blockIdx.x % num_copies) * copy_stride;
    atomicAdd(&hist[base + 4ull * num_rows + diff], 1ull);
}

// Sum the `num_copies` replicated histograms into `out[i] = Σ_r hist[r*copy_stride + i]`.
// One thread per counter bin (`copy_stride` = num_rows * num_types total bins).
extern "C" __global__ void bitwise_hist_reduce(uint64_t copy_stride, uint32_t num_copies,
                                               const unsigned long long *hist,
                                               unsigned long long *out) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= copy_stride)
        return;
    unsigned long long s = 0ull;
    for (uint32_t r = 0; r < num_copies; ++r)
        s += hist[(uint64_t)r * copy_stride + i];
    out[i] = s;
}
