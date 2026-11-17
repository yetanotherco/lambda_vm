// A table's columns filled on the card from the records its rows come from,
// instead of filled packed on the host and widened here (`widen_narrow_col_major`
// in arith.cu): MEMW_R from its register rows, CPU from the executor's logs and
// the program's decode table, KECCAK_RND from its keccak operations.
//
// Each kernel writes exactly the words the prover's host fill writes for the
// same rows (`prover/src/tables/{memw_register,cpu,keccak_rnd}.rs`), every one
// canonical as `FE::from(u64)` makes it, column-major (`out[col * height + row]`):
// the layout a table's run of columns has in the multilinear prover's resident
// store, which is the layout the widen writes. A kernel covers every row of the
// table, the padding rows included, so its output needs no zeroing.
//
// The records are the `#[repr(C)]` structs of `src/fill.rs`, field for field;
// the static_asserts below and the const asserts there pin both sides.

#include <cstddef>
#include <cstdint>

#include "goldilocks.cuh"
#include "p1w16_constants.cuh"

// ---------------------------------------------------------------------------
// The records
// ---------------------------------------------------------------------------

// The executor's log of one cycle (`executor/src/vm/logs.rs`).
struct LogC {
    uint64_t current_pc;
    uint64_t next_pc;
    uint64_t src1_val;
    uint64_t src2_val;
    uint64_t dst_val;
};
static_assert(sizeof(LogC) == 40, "LogC is 40 bytes");
static_assert(offsetof(LogC, dst_val) == 32, "LogC field order");

// One CPU row's register slots under the register format that checks a CPU
// row's register accesses in the row itself (`cpu.rs`, `CpuSide`): the
// previous timestamps of rs1, rs2 and rd, then rd's old value (low, high).
struct CpuSideC {
    uint32_t prev_ts[3];
    uint32_t old_rd[2];
};
static_assert(sizeof(CpuSideC) == 20, "CpuSideC is 20 bytes");
static_assert(offsetof(CpuSideC, old_rd) == 12, "CpuSideC field order");

// One MEMW_R row (`memw_register.rs`, `RegRow`): `address` is the register
// index, `old_ts_lo` the old timestamp's low word (its high word is the
// timestamp's), `is_read` 0 or 1, `pad` 0.
struct RegRowC {
    uint64_t timestamp;
    uint32_t val0;
    uint32_t val1;
    uint32_t old0;
    uint32_t old1;
    uint32_t old_ts_lo;
    uint16_t address;
    uint8_t is_read;
    uint8_t pad;
};
static_assert(sizeof(RegRowC) == 32, "RegRowC is 32 bytes");
static_assert(offsetof(RegRowC, old_ts_lo) == 24, "RegRowC field order");
static_assert(offsetof(RegRowC, address) == 28, "RegRowC field order");
static_assert(offsetof(RegRowC, is_read) == 30, "RegRowC field order");

// One DECODE row (`types.rs`, `DecodeEntry`): the pc, the extended immediate,
// then `ShrunkDecode`'s fields as bytes (flags 0 or 1).
struct DecodeEntryC {
    uint64_t pc;
    uint64_t imm;
    uint8_t read_register1;
    uint8_t read_register2;
    uint8_t write_register;
    uint8_t word_instr;
    uint8_t alu;
    uint8_t add;
    uint8_t sub;
    uint8_t memory;
    uint8_t branch;
    uint8_t ecall;
    uint8_t rs1;
    uint8_t rs2;
    uint8_t rd;
    uint8_t half_instruction_length;
    uint8_t alu_flags;
    uint8_t mem_flags;
};
static_assert(sizeof(DecodeEntryC) == 32, "DecodeEntryC is 32 bytes");
static_assert(offsetof(DecodeEntryC, read_register1) == 16, "DecodeEntryC field order");
static_assert(offsetof(DecodeEntryC, mem_flags) == 31, "DecodeEntryC field order");

// A run of pcs 4 apart (`decode.rs`, `DecodeTable::runs`): pc `first_pc + 4k`
// is DECODE row `row + k` for `k < len`. The runs are ascending and disjoint.
struct DecodeRunC {
    uint64_t first_pc;
    uint32_t row;
    uint32_t len;
};
static_assert(sizeof(DecodeRunC) == 16, "DecodeRunC is 16 bytes");
static_assert(offsetof(DecodeRunC, len) == 12, "DecodeRunC field order");

// One keccak permutation call (`keccak_rnd.rs`, `KeccakRoundOperation`): its 24
// rows are the rounds from `input`. `output` is carried, not read.
struct KeccakOpC {
    uint64_t timestamp;
    uint64_t input[25];
    uint64_t output[25];
};
static_assert(sizeof(KeccakOpC) == 408, "KeccakOpC is 408 bytes");
static_assert(offsetof(KeccakOpC, output) == 208, "KeccakOpC field order");

// The same call as a table held as its records carries it
// (`multilinear::records::KeccakRndOp`): the fields its rows read, no output.
struct KeccakOpInC {
    uint64_t timestamp;
    uint64_t input[25];
};
static_assert(sizeof(KeccakOpInC) == 208, "KeccakOpInC is 208 bytes");
static_assert(offsetof(KeccakOpInC, input) == 8, "KeccakOpInC field order");

// One MEMW_A row (`memw_aligned.rs`, `AlignedRow`): the access's base address,
// timestamp and shared old timestamp, its value and old value packed in a
// word each (eight bytes for a memory access, the two 32-bit halves for a
// register access), its width, and flags 0 or 1 (`limbs`: the row's columns
// 25 and 26 are the step's 16-bit limbs, not the old timestamp).
struct AlignedRowC {
    uint64_t base_address;
    uint64_t timestamp;
    uint64_t old_timestamp;
    uint64_t value;
    uint64_t old;
    uint8_t width;
    uint8_t is_read;
    uint8_t is_register;
    uint8_t limbs;
};
static_assert(sizeof(AlignedRowC) == 48, "AlignedRowC is 48 bytes");
static_assert(offsetof(AlignedRowC, width) == 40, "AlignedRowC field order");
static_assert(offsetof(AlignedRowC, is_register) == 42, "AlignedRowC field order");
static_assert(offsetof(AlignedRowC, limbs) == 43, "AlignedRowC field order");

// One LOAD op (`load.rs`, `LoadOperation`): the base address, the timestamp,
// the width, whether it sign-extends (0 or 1) and the result's eight bytes.
struct LoadOpC {
    uint64_t base_address;
    uint64_t timestamp;
    uint8_t width;
    uint8_t is_signed;
    uint8_t res[8];
};
static_assert(sizeof(LoadOpC) == 32, "LoadOpC is 32 bytes");
static_assert(offsetof(LoadOpC, is_signed) == 17, "LoadOpC field order");
static_assert(offsetof(LoadOpC, res) == 18, "LoadOpC field order");

// One STORE op (`store.rs`, `StoreOperation`): the base address, the
// timestamp, the value, and the width flags (0 or 1; none for one byte).
struct StoreOpC {
    uint64_t base_address;
    uint64_t timestamp;
    uint64_t value;
    uint8_t write2;
    uint8_t write4;
    uint8_t write8;
};
static_assert(sizeof(StoreOpC) == 32, "StoreOpC is 32 bytes");
static_assert(offsetof(StoreOpC, write2) == 24, "StoreOpC field order");

// One fused aligned CPU load (the prover's `MemLoadRow`): the address, the
// timestamp, the step to its bytes' old timestamp less one (`ts − old − 1`,
// two 16-bit limbs), the result's eight bytes (the loaded bytes, then the
// sign or zero fill), the width and whether it sign-extends (0 or 1).
struct MemLoadRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t res[8];
    uint8_t width;
    uint8_t is_signed;
    uint8_t pad[2];
};
static_assert(sizeof(MemLoadRowC) == 32, "MemLoadRowC is 32 bytes");
static_assert(offsetof(MemLoadRowC, res) == 20, "MemLoadRowC field order");
static_assert(offsetof(MemLoadRowC, width) == 28, "MemLoadRowC field order");

// One fused aligned CPU store (the prover's `MemStoreRow`): the address, the
// timestamp, the step as a load's, the width, the eight bytes written and the
// eight bytes they replace, packed a word each.
struct MemStoreRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t width;
    uint8_t pad[3];
    uint64_t value;
    uint64_t old;
};
static_assert(sizeof(MemStoreRowC) == 40, "MemStoreRowC is 40 bytes");
static_assert(offsetof(MemStoreRowC, width) == 20, "MemStoreRowC field order");
static_assert(offsetof(MemStoreRowC, value) == 24, "MemStoreRowC field order");

// One SHIFT op (`shift.rs`, `ShiftOperation`): the full shift amount, the input
// as four 16-bit halves, the amount's low byte, and flags 0 or 1.
struct ShiftOpC {
    uint64_t shift_amount;
    uint16_t in_halves[4];
    uint8_t shift;
    uint8_t direction;
    uint8_t is_signed;
    uint8_t word_instr;
};
static_assert(sizeof(ShiftOpC) == 24, "ShiftOpC is 24 bytes");
static_assert(offsetof(ShiftOpC, in_halves) == 8, "ShiftOpC field order");
static_assert(offsetof(ShiftOpC, word_instr) == 19, "ShiftOpC field order");

// One MEMW op (`memw.rs`, `MemwOperation`): the base address, the timestamp,
// the eight old timestamps, the eight value and old elements (a memory byte
// or a 32-bit register half each), and the flags and width.
struct MemwOpC {
    uint64_t base_address;
    uint64_t timestamp;
    uint64_t old_timestamp[8];
    uint32_t value[8];
    uint32_t old[8];
    uint8_t is_register;
    uint8_t width;
    uint8_t is_read;
};
static_assert(sizeof(MemwOpC) == 152, "MemwOpC is 152 bytes");
static_assert(offsetof(MemwOpC, value) == 80, "MemwOpC field order");
static_assert(offsetof(MemwOpC, is_register) == 144, "MemwOpC field order");

// ---------------------------------------------------------------------------
// The setters (`types.rs`, `VmTable`)
// ---------------------------------------------------------------------------

namespace fill {

// One row's cells: column `col` at `out[col * height + row]`.
struct RowWriter {
    uint64_t *out;
    uint64_t height;
    uint64_t row;

    // `set_u64` (and `set_bool` / `set_byte` / `set_half` / `set_word`): one
    // canonical word.
    __device__ __forceinline__ void word(uint32_t col, uint64_t v) const {
        out[(uint64_t)col * height + row] = goldilocks::canonical(v);
    }
    // `set_dword_wl`: the low and high 32-bit words.
    __device__ __forceinline__ void dword_wl(uint32_t col, uint64_t v) const {
        word(col, v & 0xFFFFFFFFull);
        word(col + 1, v >> 32);
    }
    // `set_dword_hl`: four 16-bit halves, low first.
    __device__ __forceinline__ void dword_hl(uint32_t col, uint64_t v) const {
#pragma unroll
        for (uint32_t i = 0; i < 4; ++i) word(col + i, (v >> (16 * i)) & 0xFFFFull);
    }
    // `set_dword_bl`: eight bytes, low first.
    __device__ __forceinline__ void dword_bl(uint32_t col, uint64_t v) const {
#pragma unroll
        for (uint32_t i = 0; i < 8; ++i) word(col + i, (v >> (8 * i)) & 0xFFull);
    }
    // Columns `[first, end)` zero: a padding row's.
    __device__ __forceinline__ void zeros(uint32_t first, uint32_t end) const {
        for (uint32_t c = first; c < end; ++c) out[(uint64_t)c * height + row] = 0;
    }
};

}  // namespace fill

// ---------------------------------------------------------------------------
// MEMW_R (`memw_register.rs`, `generate_memw_register_trace_from_rows_as`)
// ---------------------------------------------------------------------------

namespace memw_r_cols {
constexpr uint32_t ADDRESS = 0;
constexpr uint32_t TIMESTAMP_0 = 1;
constexpr uint32_t VAL_0 = 3;
constexpr uint32_t VAL_1 = 4;
constexpr uint32_t OLD_0 = 5;
constexpr uint32_t OLD_1 = 6;
constexpr uint32_t OLD_TIMESTAMP_LO = 7;
constexpr uint32_t MU_READ = 8;
constexpr uint32_t MU_WRITE = 9;
constexpr uint32_t NUM_COLUMNS = 10;
}  // namespace memw_r_cols

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_memw_r(const RegRowC *rows, uint64_t n, uint64_t height,
                                       uint64_t *out) {
    namespace c = memw_r_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const RegRowC r = rows[row];
    w.word(c::ADDRESS, r.address);
    w.dword_wl(c::TIMESTAMP_0, r.timestamp);
    w.word(c::VAL_0, r.val0);
    w.word(c::VAL_1, r.val1);
    w.word(c::OLD_0, r.old0);
    w.word(c::OLD_1, r.old1);
    w.word(c::OLD_TIMESTAMP_LO, r.old_ts_lo);
    w.word(c::MU_READ, r.is_read != 0);
    w.word(c::MU_WRITE, r.is_read == 0);
}

// ---------------------------------------------------------------------------
// MEMW_A (`memw_aligned.rs`, `generate_memw_aligned_trace_segments_as`)
// ---------------------------------------------------------------------------

namespace memw_a_cols {
constexpr uint32_t IS_REGISTER = 0;
constexpr uint32_t BASE_ADDRESS = 1;  // DWordWHH: 3 columns
constexpr uint32_t VALUE = 4;         // 8 columns
constexpr uint32_t TIMESTAMP_0 = 12;
constexpr uint32_t WRITE2 = 14;
constexpr uint32_t WRITE4 = 15;
constexpr uint32_t WRITE8 = 16;
constexpr uint32_t OLD = 17;  // 8 columns
constexpr uint32_t OLD_TIMESTAMP_0 = 25;
constexpr uint32_t STEP_0 = 25;  // the limbs' form: the step's limbs in 25 and 26
constexpr uint32_t STEP_1 = 26;
constexpr uint32_t MU_READ = 27;
constexpr uint32_t MU_WRITE = 28;
constexpr uint32_t NUM_COLUMNS = 29;
}  // namespace memw_a_cols

namespace fill {

// `AlignedRow::unpack` into eight columns from `col`: a register access's
// two 32-bit halves (the rest zero), or a memory access's eight bytes.
__device__ __forceinline__ void aligned_elements(const RowWriter &w, uint32_t col, uint64_t packed,
                                                 bool is_register) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) {
        uint64_t element = is_register ? (i < 2 ? (packed >> (32 * i)) & 0xFFFFFFFFull : 0)
                                       : (packed >> (8 * i)) & 0xFFull;
        w.word(col + i, element);
    }
}

}  // namespace fill

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_memw_a(const AlignedRowC *rows, uint64_t n, uint64_t height,
                                       uint64_t *out) {
    namespace c = memw_a_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const AlignedRowC r = rows[row];
    const bool is_register = r.is_register != 0;
    w.word(c::IS_REGISTER, is_register);
    // `set_dword_whh`: the low and middle halves, then the high word.
    w.word(c::BASE_ADDRESS, r.base_address & 0xFFFFull);
    w.word(c::BASE_ADDRESS + 1, (r.base_address >> 16) & 0xFFFFull);
    w.word(c::BASE_ADDRESS + 2, r.base_address >> 32);
    fill::aligned_elements(w, c::VALUE, r.value, is_register);
    w.dword_wl(c::TIMESTAMP_0, r.timestamp);
    w.word(c::WRITE2, r.width == 2);
    w.word(c::WRITE4, r.width == 4);
    w.word(c::WRITE8, r.width == 8);
    fill::aligned_elements(w, c::OLD, r.old, is_register);
    if (r.limbs != 0) {
        // `step_limbs`: timestamp − old_timestamp − 1 = d0 + 2^16·d1. The walk's
        // routing gave this row a step in 1..2^32 on one high word.
        const uint64_t d = r.timestamp - r.old_timestamp - 1;
        w.word(c::STEP_0, d & 0xFFFFull);
        w.word(c::STEP_1, (d >> 16) & 0xFFFFull);
    } else {
        w.dword_wl(c::OLD_TIMESTAMP_0, r.old_timestamp);
    }
    w.word(c::MU_READ, r.is_read != 0);
    w.word(c::MU_WRITE, r.is_read == 0);
}

// ---------------------------------------------------------------------------
// LOAD (`load.rs`, `generate_load_trace_segments_as`)
// ---------------------------------------------------------------------------

namespace load_cols {
constexpr uint32_t BASE_ADDRESS_0 = 0;
constexpr uint32_t TIMESTAMP_0 = 2;
constexpr uint32_t READ2 = 4;
constexpr uint32_t READ4 = 5;
constexpr uint32_t READ8 = 6;
constexpr uint32_t SIGNED = 7;
constexpr uint32_t RES = 8;  // 8 columns
constexpr uint32_t SIGN_BIT = 16;
constexpr uint32_t MU = 17;
constexpr uint32_t NUM_COLUMNS = 18;
}  // namespace load_cols

// Rows `[0, n)` from `ops`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_load(const LoadOpC *ops, uint64_t n, uint64_t height,
                                     uint64_t *out) {
    namespace c = load_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const LoadOpC op = ops[row];
    w.dword_wl(c::BASE_ADDRESS_0, op.base_address);
    w.dword_wl(c::TIMESTAMP_0, op.timestamp);
    w.word(c::READ2, op.width == 2);
    w.word(c::READ4, op.width == 4);
    w.word(c::READ8, op.width == 8);
    w.word(c::SIGNED, op.is_signed != 0);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::RES + i, op.res[i]);
    // `compute_sign_bit`: the top bit of the highest byte read.
    const uint32_t top = op.width == 2 ? 1 : (op.width == 4 ? 3 : (op.width == 8 ? 7 : 0));
    w.word(c::SIGN_BIT, (op.res[top] >> 7) & 1);
    w.word(c::MU, 1);
}

// ---------------------------------------------------------------------------
// STORE (`store.rs`, `generate_store_trace_segments_as`)
// ---------------------------------------------------------------------------

namespace store_cols {
constexpr uint32_t BASE_ADDRESS_0 = 0;
constexpr uint32_t TIMESTAMP_0 = 2;
constexpr uint32_t WRITE2 = 4;
constexpr uint32_t WRITE4 = 5;
constexpr uint32_t WRITE8 = 6;
constexpr uint32_t VALUE = 7;  // 8 columns
constexpr uint32_t MU = 15;
constexpr uint32_t NUM_COLUMNS = 16;
}  // namespace store_cols

// Rows `[0, n)` from `ops`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_store(const StoreOpC *ops, uint64_t n, uint64_t height,
                                      uint64_t *out) {
    namespace c = store_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const StoreOpC op = ops[row];
    w.dword_wl(c::BASE_ADDRESS_0, op.base_address);
    w.dword_wl(c::TIMESTAMP_0, op.timestamp);
    w.word(c::WRITE2, op.write2 != 0);
    w.word(c::WRITE4, op.write4 != 0);
    w.word(c::WRITE8, op.write8 != 0);
    w.dword_bl(c::VALUE, op.value);
    w.word(c::MU, 1);
}

// ---------------------------------------------------------------------------
// MEM_LOAD and MEM_STORE: the aligned CPU loads and stores checked in one row
// with their memory tokens (the fused memory format). A row is LOAD's (STORE's)
// columns and MEMW_A's: the address as MEMW_A's (low half, middle half, high
// word), the timestamp as two words, the width flags, the bytes, the step's two
// 16-bit limbs, μ.
// ---------------------------------------------------------------------------

namespace mem_load_cols {
constexpr uint32_t BASE_ADDRESS = 0;  // 3 columns: half, half, word
constexpr uint32_t TIMESTAMP_0 = 3;   // 2 columns
constexpr uint32_t READ2 = 5;
constexpr uint32_t READ4 = 6;
constexpr uint32_t READ8 = 7;
constexpr uint32_t SIGNED = 8;
constexpr uint32_t RES = 9;  // 8 columns
constexpr uint32_t SIGN_BIT = 17;
constexpr uint32_t STEP_0 = 18;
constexpr uint32_t STEP_1 = 19;
constexpr uint32_t MU = 20;
constexpr uint32_t NUM_COLUMNS = 21;
}  // namespace mem_load_cols

namespace mem_store_cols {
constexpr uint32_t BASE_ADDRESS = 0;  // 3 columns
constexpr uint32_t TIMESTAMP_0 = 3;   // 2 columns
constexpr uint32_t WRITE2 = 5;
constexpr uint32_t WRITE4 = 6;
constexpr uint32_t WRITE8 = 7;
constexpr uint32_t VALUE = 8;  // 8 columns
constexpr uint32_t OLD = 16;   // 8 columns
constexpr uint32_t STEP_0 = 24;
constexpr uint32_t STEP_1 = 25;
constexpr uint32_t MU = 26;
constexpr uint32_t NUM_COLUMNS = 27;
}  // namespace mem_store_cols

namespace fill {

// `set_dword_whh`: the low and middle halves, then the high word (MEMW_A's
// address).
__device__ __forceinline__ void address_whh(const RowWriter &w, uint32_t col, uint64_t a) {
    w.word(col, a & 0xFFFFull);
    w.word(col + 1, (a >> 16) & 0xFFFFull);
    w.word(col + 2, a >> 32);
}

}  // namespace fill

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_load(const MemLoadRowC *rows, uint64_t n, uint64_t height,
                                         uint64_t *out) {
    namespace c = mem_load_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const MemLoadRowC r = rows[row];
    fill::address_whh(w, c::BASE_ADDRESS, r.address);
    w.dword_wl(c::TIMESTAMP_0, r.timestamp);
    w.word(c::READ2, r.width == 2);
    w.word(c::READ4, r.width == 4);
    w.word(c::READ8, r.width == 8);
    w.word(c::SIGNED, r.is_signed != 0);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::RES + i, r.res[i]);
    // LOAD's `compute_sign_bit`: the top bit of the highest byte read.
    const uint32_t top = r.width == 2 ? 1 : (r.width == 4 ? 3 : (r.width == 8 ? 7 : 0));
    w.word(c::SIGN_BIT, (r.res[top] >> 7) & 1);
    w.word(c::STEP_0, r.step & 0xFFFFu);
    w.word(c::STEP_1, r.step >> 16);
    w.word(c::MU, 1);
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_store(const MemStoreRowC *rows, uint64_t n, uint64_t height,
                                          uint64_t *out) {
    namespace c = mem_store_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const MemStoreRowC r = rows[row];
    fill::address_whh(w, c::BASE_ADDRESS, r.address);
    w.dword_wl(c::TIMESTAMP_0, r.timestamp);
    w.word(c::WRITE2, r.width == 2);
    w.word(c::WRITE4, r.width == 4);
    w.word(c::WRITE8, r.width == 8);
    w.dword_bl(c::VALUE, r.value);
    w.dword_bl(c::OLD, r.old);
    w.word(c::STEP_0, r.step & 0xFFFFu);
    w.word(c::STEP_1, r.step >> 16);
    w.word(c::MU, 1);
}

// ---------------------------------------------------------------------------
// SHIFT (`shift.rs`, `generate_shift_trace_segments_as` and
// `ShiftOperation::compute_aux`)
// ---------------------------------------------------------------------------

namespace shift_cols {
constexpr uint32_t IN_0 = 0;  // 4 columns
constexpr uint32_t SHIFT_AMOUNT = 4;
constexpr uint32_t DIRECTION = 5;
constexpr uint32_t SIGNED = 6;
constexpr uint32_t WORD_INSTR = 7;
constexpr uint32_t OUT_0 = 8;  // 2 columns
constexpr uint32_t IS_NEGATIVE = 10;
constexpr uint32_t BIT_SHIFT = 11;
constexpr uint32_t ZBS = 12;
constexpr uint32_t X_0 = 13;  // 5 columns
constexpr uint32_t Y_0 = 18;  // 4 columns
constexpr uint32_t LIMB_SHIFT_RAW_0 = 22;  // 3 columns
constexpr uint32_t MU = 25;
constexpr uint32_t SHIFT_B1 = 26;
constexpr uint32_t SHIFT_H1 = 27;
constexpr uint32_t SHIFT_HIGH = 28;
constexpr uint32_t NUM_COLUMNS = 29;
}  // namespace shift_cols

// Rows `[0, n)` from `ops`, rows `[n, height)` padding (ZBS 1, every other
// column zero). One thread a row.
extern "C" __global__ void fill_shift(const ShiftOpC *ops, uint64_t n, uint64_t height,
                                      uint64_t *out) {
    namespace c = shift_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        w.word(c::ZBS, 1);
        return;
    }
    const ShiftOpC op = ops[row];
    const bool left = op.direction == 0;
    const bool right = !left;
    const bool is_negative = op.is_signed != 0 && ((op.in_halves[3] >> 15) & 1) == 1;
    const uint16_t extension = is_negative ? 0xFFFF : 0;
    const uint8_t bit_shift =
        left ? (uint8_t)(op.shift & 15) : (uint8_t)((uint16_t)(256 - (uint16_t)op.shift) & 15);
    const bool zbs = bit_shift == 0;
    // `hwsl`: (half << z) & 0xFFFF; `hwslc`: half >> (16 - z), 0 at z = 0.
    uint16_t x[5] = {0, 0, 0, 0, 0};
    uint16_t y[4] = {0, 0, 0, 0};
    if (zbs) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (left) {
                x[i] = op.in_halves[i];
            } else {
                y[i] = op.in_halves[i];
            }
        }
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            x[i] = (uint16_t)((uint32_t)op.in_halves[i] << bit_shift);
            y[i] = (uint16_t)(op.in_halves[i] >> (16 - bit_shift));
        }
        x[4] = (uint16_t)((uint32_t)extension << bit_shift);
    }
    // `limb_shift`: one-hot of (shift >> 4) & mask.
    const int limb = op.word_instr != 0 ? ((op.shift >> 4) & 1) : ((op.shift >> 4) & 3);
    // `compute_shifted`, 16-bit wrapping sums.
    uint16_t shifted[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint16_t v = 0;
        if (left && limb <= i) {
            const int k = i - limb;
            v = (uint16_t)(v + (k == 0 ? x[0] : (uint16_t)(x[k] + y[k - 1])));
        }
        if (right) {
            if (limb <= 3 - i) {
                const int k = i + limb;
                v = (uint16_t)(v + (uint16_t)(y[k] + x[k + 1]));
            } else {
                v = (uint16_t)(v + extension);
            }
        }
        shifted[i] = v;
    }
    w.dword_hl(c::IN_0, (uint64_t)op.in_halves[0] | (uint64_t)op.in_halves[1] << 16 |
                            (uint64_t)op.in_halves[2] << 32 | (uint64_t)op.in_halves[3] << 48);
    w.word(c::SHIFT_AMOUNT, op.shift);
    w.word(c::SHIFT_B1, (op.shift_amount >> 8) & 0xFFull);
    w.word(c::SHIFT_H1, (op.shift_amount >> 16) & 0xFFFFull);
    w.word(c::SHIFT_HIGH, op.shift_amount >> 32);
    w.word(c::DIRECTION, op.direction != 0);
    w.word(c::SIGNED, op.is_signed != 0);
    w.word(c::WORD_INSTR, op.word_instr != 0);
    w.word(c::OUT_0, (uint64_t)shifted[0] | (uint64_t)shifted[1] << 16);
    w.word(c::OUT_0 + 1, (uint64_t)shifted[2] | (uint64_t)shifted[3] << 16);
    w.word(c::IS_NEGATIVE, is_negative);
    w.word(c::BIT_SHIFT, bit_shift);
    w.word(c::ZBS, zbs);
#pragma unroll
    for (int i = 0; i < 5; ++i) w.word(c::X_0 + i, x[i]);
#pragma unroll
    for (int i = 0; i < 4; ++i) w.word(c::Y_0 + i, y[i]);
#pragma unroll
    for (int i = 0; i < 3; ++i) w.word(c::LIMB_SHIFT_RAW_0 + i, limb == i);
    w.word(c::MU, 1);
}

// ---------------------------------------------------------------------------
// MEMW (`memw.rs`, `generate_memw_trace_segments_as`)
// ---------------------------------------------------------------------------

namespace memw_cols {
constexpr uint32_t IS_REGISTER = 0;
constexpr uint32_t BASE_ADDRESS_0 = 1;
constexpr uint32_t VALUE = 3;  // 8 columns
constexpr uint32_t TIMESTAMP_0 = 11;
constexpr uint32_t WRITE2 = 13;
constexpr uint32_t WRITE4 = 14;
constexpr uint32_t WRITE8 = 15;
constexpr uint32_t OLD = 16;    // 8 columns
constexpr uint32_t CARRY = 24;  // 7 columns
constexpr uint32_t OLD_TIMESTAMP = 31;  // 8 × 2 columns
constexpr uint32_t MU_READ = 47;
constexpr uint32_t MU_WRITE = 48;
constexpr uint32_t NUM_COLUMNS = 49;
}  // namespace memw_cols

// Rows `[0, n)` from `ops`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_memw(const MemwOpC *ops, uint64_t n, uint64_t height,
                                     uint64_t *out) {
    namespace c = memw_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const MemwOpC &op = ops[row];
    w.word(c::IS_REGISTER, op.is_register != 0);
    w.dword_wl(c::BASE_ADDRESS_0, op.base_address);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::VALUE + i, op.value[i]);
    w.dword_wl(c::TIMESTAMP_0, op.timestamp);
    w.word(c::WRITE2, op.width == 2);
    w.word(c::WRITE4, op.width == 4);
    w.word(c::WRITE8, op.width == 8);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::OLD + i, op.old[i]);
    // carry[i]: the low word plus i + 1 reaches 2^32.
    const uint64_t lo = op.base_address & 0xFFFFFFFFull;
#pragma unroll
    for (uint32_t i = 0; i < 7; ++i) w.word(c::CARRY + i, lo + i + 1 >= (1ull << 32));
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.dword_wl(c::OLD_TIMESTAMP + 2 * i, op.old_timestamp[i]);
    w.word(c::MU_READ, op.is_read != 0);
    w.word(c::MU_WRITE, op.is_read == 0);
}

// ---------------------------------------------------------------------------
// CPU (`cpu.rs`, `CpuOperation::from_log` then `fill_cpu_rows`)
// ---------------------------------------------------------------------------

namespace cpu_cols {
constexpr uint32_t TIMESTAMP = 0;
constexpr uint32_t PC_0 = 1;
constexpr uint32_t RS1 = 3;
constexpr uint32_t RS2 = 4;
constexpr uint32_t RD = 5;
constexpr uint32_t READ_REGISTER1 = 6;
constexpr uint32_t READ_REGISTER2 = 7;
constexpr uint32_t WRITE_REGISTER = 8;
constexpr uint32_t IMM_0 = 9;
constexpr uint32_t HALF_INSTRUCTION_LENGTH = 11;
constexpr uint32_t WORD_INSTR = 12;
constexpr uint32_t ALU = 13;
constexpr uint32_t ALU_FLAGS = 14;
constexpr uint32_t ADD = 15;
constexpr uint32_t SUB = 16;
constexpr uint32_t MEMORY = 17;
constexpr uint32_t MEM_FLAGS = 18;
constexpr uint32_t BRANCH = 19;
constexpr uint32_t ECALL = 20;
constexpr uint32_t NEXT_PC_0 = 21;
constexpr uint32_t RVD_0 = 23;
constexpr uint32_t PC_DOUBLE_READ = 25;
constexpr uint32_t RV1_0 = 26;
constexpr uint32_t RV2_0 = 28;
constexpr uint32_t ARG2_0 = 30;
constexpr uint32_t RES_0 = 32;
constexpr uint32_t BRANCH_COND = 36;
constexpr uint32_t NUM_COLUMNS = 37;
// The in-row register format's five (`cpu.rs`, `inrow_cols`): PREV_TS_1,
// PREV_TS_2, PREV_TS_D, OLD_RD_0, OLD_RD_1.
constexpr uint32_t PREV_TS_1 = 37;
constexpr uint32_t OLD_RD_0 = 40;
constexpr uint32_t NUM_COLUMNS_IN_ROW = 42;
}  // namespace cpu_cols

// `alu_op::EQ` / `alu_op::LT` (`types.rs`).
constexpr uint32_t FILL_ALU_EQ = 3;
constexpr uint32_t FILL_ALU_LT = 4;
// The CPU padding rows' pc (`cpu.rs`, `CPU_PADDING_PC`).
constexpr uint64_t FILL_CPU_PADDING_PC = 1;

namespace fill {

// `DecodeTable::row`: the row of the run holding `pc`, or -1. The runs are
// ascending and disjoint, so the only run that can hold `pc` is the last one
// starting at or below it.
__device__ __forceinline__ int64_t decode_row(const DecodeRunC *runs, uint32_t nruns,
                                              uint64_t pc) {
    uint32_t lo = 0, hi = nruns;
    while (lo < hi) {
        uint32_t mid = lo + (hi - lo) / 2;
        if (runs[mid].first_pc <= pc) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if (lo == 0) return -1;
    const DecodeRunC run = runs[lo - 1];
    uint64_t offset = pc - run.first_pc;
    if (offset % 4 != 0 || offset / 4 >= run.len) return -1;
    return (int64_t)run.row + (int64_t)(offset / 4);
}

// `CpuOperation::branch_taken`: `alu_flags = op + 32·signed + 64·invert`.
__device__ __forceinline__ bool branch_taken(uint8_t alu_flags, uint64_t rv1, uint64_t rv2) {
    uint32_t op = alu_flags & 0x1F;
    bool is_signed = (alu_flags >> 5) & 1;
    bool invert = (alu_flags >> 6) & 1;
    bool cmp = false;
    if (op == FILL_ALU_EQ) {
        cmp = rv1 == rv2;
    } else if (op == FILL_ALU_LT) {
        cmp = is_signed ? (int64_t)rv1 < (int64_t)rv2 : rv1 < rv2;
    }
    return cmp != invert;
}

}  // namespace fill

// Rows `[0, n)` from `logs`, each read with its DECODE entry; rows
// `[n, height)` the CPU padding rows. Every row's TIMESTAMP is the table's own
// preprocessed `4·(row + 1)` (`cpu.rs`, `local_timestamp`). A log whose pc has
// no DECODE row counts into `missing` and leaves its row zero: the caller
// refuses the table. With `sides` (the in-row register format's, one a log;
// null otherwise) the table has five more columns: row `r < n` takes side
// record `r` (`cpu.rs`, `generate_cpu_trace_inrow_segments_as`), a padding row
// zeros. One thread a row.
extern "C" __global__ void fill_cpu_from_logs(const LogC *logs, uint64_t n,
                                              const DecodeEntryC *entries,
                                              const DecodeRunC *runs, uint32_t nruns,
                                              uint64_t height, uint64_t *out,
                                              unsigned long long *missing,
                                              const CpuSideC *sides) {
    namespace c = cpu_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    const uint64_t timestamp = 4 * (row + 1);

    if (row >= n) {
        // pc = next_pc = 1, every flag 0.
        w.zeros(0, sides ? c::NUM_COLUMNS_IN_ROW : c::NUM_COLUMNS);
        w.word(c::TIMESTAMP, timestamp);
        w.word(c::PC_0, FILL_CPU_PADDING_PC);
        w.word(c::NEXT_PC_0, FILL_CPU_PADDING_PC);
        return;
    }

    if (sides) {
        const CpuSideC side = sides[row];
#pragma unroll
        for (uint32_t i = 0; i < 3; ++i) w.word(c::PREV_TS_1 + i, side.prev_ts[i]);
        w.word(c::OLD_RD_0, side.old_rd[0]);
        w.word(c::OLD_RD_0 + 1, side.old_rd[1]);
    }

    const LogC log = logs[row];
    int64_t at = fill::decode_row(runs, nruns, log.current_pc);
    if (at < 0) {
        atomicAdd(missing, 1ull);
        w.zeros(0, sides ? c::NUM_COLUMNS_IN_ROW : c::NUM_COLUMNS);
        return;
    }
    const DecodeEntryC e = entries[at];
    uint64_t length = 2 * (uint64_t)e.half_instruction_length;
    bool word_instr = e.word_instr != 0;

    // `from_log`. A word instruction's row is a CPU32 delegate: its operational
    // columns are zero and its pc advances by the length.
    uint64_t next_pc = e.pc + length;
    uint64_t rv1 = 0, rv2 = 0, arg2 = 0, res = 0, rvd = 0;
    bool branch_cond = false;
    if (!word_instr) {
        // x255 is the pc register (AUIPC / JAL read it through rs1).
        rv1 = e.rs1 == 255 ? log.current_pc : (e.read_register1 ? log.src1_val : 0);
        rv2 = e.read_register2 ? log.src2_val : 0;
        bool jalr = (e.mem_flags & 1) != 0;
        arg2 = e.memory ? e.imm : (e.branch ? rv2 : rv2 + e.imm);
        branch_cond = e.branch && (jalr || fill::branch_taken(e.alu_flags, rv1, rv2));
        if (e.add) {
            res = rv1 + arg2;
        } else if (e.sub) {
            res = rv1 - arg2;
        } else if (e.alu) {
            res = e.branch ? (uint64_t)branch_cond : log.dst_val;
        }
        // Under MEMORY, mem_flags bit 0 is the store bit.
        if (e.memory) {
            rvd = jalr ? 0 : log.dst_val;
        } else if (e.branch) {
            rvd = e.pc + length;
        } else {
            rvd = res;
        }
        // An ECALL keeps pc + length (the executor's 0 signals the halt).
        if (!e.ecall && branch_cond) next_pc = log.next_pc;
    }

    // `fill_cpu_rows`.
    bool live = !word_instr;
    w.word(c::TIMESTAMP, timestamp);
    w.dword_wl(c::PC_0, e.pc);
    w.word(c::RS1, live ? e.rs1 : 0);
    w.word(c::RS2, live ? e.rs2 : 0);
    w.word(c::RD, live ? e.rd : 0);
    // x0 is never read or written; x255 is read.
    w.word(c::READ_REGISTER1, live && e.read_register1 && e.rs1 != 0);
    w.word(c::READ_REGISTER2, live && e.read_register2 && e.rs2 != 0);
    w.word(c::WRITE_REGISTER, live && e.write_register && e.rd != 0);
    w.dword_wl(c::IMM_0, live ? e.imm : 0);
    w.word(c::HALF_INSTRUCTION_LENGTH, e.half_instruction_length);
    w.word(c::WORD_INSTR, word_instr);
    w.word(c::ALU, live && e.alu);
    w.word(c::ALU_FLAGS, live ? e.alu_flags : 0);
    w.word(c::ADD, live && e.add);
    w.word(c::SUB, live && e.sub);
    w.word(c::MEMORY, live && e.memory);
    w.word(c::MEM_FLAGS, live ? e.mem_flags : 0);
    w.word(c::BRANCH, live && e.branch);
    w.word(c::ECALL, live && e.ecall);
    w.dword_wl(c::NEXT_PC_0, next_pc);
    w.dword_wl(c::RVD_0, rvd);
    w.dword_wl(c::RV1_0, rv1);
    w.dword_wl(c::RV2_0, rv2);
    w.dword_wl(c::ARG2_0, arg2);
    w.dword_hl(c::RES_0, res);
    w.word(c::BRANCH_COND, branch_cond);
    // The inline pc read's column.
    w.word(c::PC_DOUBLE_READ, live && e.read_register1 && e.rs1 == 255);
}

// ---------------------------------------------------------------------------
// KECCAK_RND (`keccak_rnd.rs`, `generate_keccak_rnd_rows_as` and `fill_op`)
// ---------------------------------------------------------------------------

namespace keccak_cols {
constexpr uint32_t TIMESTAMP_0 = 0;
constexpr uint32_t ROUND = 2;
constexpr uint32_t START = 3;
constexpr uint32_t CXZ = START + 200;
constexpr uint32_t CXZ_LEFT = CXZ + 160;
constexpr uint32_t CXZ_RIGHT = CXZ_LEFT + 40;
constexpr uint32_t DXZ = CXZ_RIGHT + 20;
constexpr uint32_t THETA = DXZ + 40;
constexpr uint32_t ROT_LEFT = THETA + 200;
constexpr uint32_t ROT_RIGHT = ROT_LEFT + 200;
constexpr uint32_t CHI_ANDS = ROT_RIGHT + 200;
constexpr uint32_t CHI = CHI_ANDS + 200;
constexpr uint32_t RC = CHI + 200;
constexpr uint32_t IOTA = RC + 8;
constexpr uint32_t MU = IOTA + 8;
constexpr uint32_t NUM_COLUMNS = MU + 1;
static_assert(NUM_COLUMNS == 1480, "KECCAK_RND has 1,480 columns");
}  // namespace keccak_cols

// The kernel's grid.y: three runs of columns, each a function of the round's
// start state alone. Threads of different sections recompute the state.
//   0: timestamp, round, start, Cxz, Cxz_left, Cxz_right, Dxz   [0, 463)
//   1: theta, rot_left, rot_right                                [463, 1063)
//   2: chi_ands, chi, rc, iota, mu                                [1063, 1480)
constexpr uint32_t FILL_KECCAK_SECTIONS = 3;

__device__ __constant__ uint64_t FILL_KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL,
    0x8000000080008000ULL, 0x000000000000808bULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL, 0x000000000000008aULL,
    0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
    0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL,
    0x8000000000008003ULL, 0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800aULL, 0x800000008000000aULL, 0x8000000080008081ULL,
    0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL,
};

// `KECCAK_RHO[x][y]`, indexed by lane `x + 5y`.
__device__ __constant__ uint32_t FILL_KECCAK_RHO[25] = {
    0,  1,  62, 28, 27,  // y = 0: x = 0..4
    36, 44, 6,  55, 20,  // y = 1
    3,  10, 43, 25, 39,  // y = 2
    41, 45, 15, 21, 8,   // y = 3
    18, 2,  61, 56, 14,  // y = 4
};

namespace fill {

__device__ __forceinline__ uint64_t rotl(uint64_t v, uint32_t n) {
    return n == 0 ? v : (v << n) | (v >> (64 - n));
}

// θ's column parities `C[x]` and `D[x] = C[x−1] ⊕ rotl(C[x+1], 1)`.
__device__ __forceinline__ void theta_d(const uint64_t s[25], uint64_t cpar[5],
                                        uint64_t d[5]) {
#pragma unroll
    for (int x = 0; x < 5; ++x) cpar[x] = s[x] ^ s[x + 5] ^ s[x + 10] ^ s[x + 15] ^ s[x + 20];
#pragma unroll
    for (int x = 0; x < 5; ++x) d[x] = cpar[(x + 4) % 5] ^ rotl(cpar[(x + 1) % 5], 1);
}

// θ applied, then ρ and π: `b[y + 5·((2x + 3y) mod 5)] = rotl(θ[x + 5y], ρ[x][y])`.
__device__ __forceinline__ void rho_pi(uint64_t s[25], const uint64_t d[5], uint64_t b[25]) {
#pragma unroll
    for (int y = 0; y < 5; ++y) {
#pragma unroll
        for (int x = 0; x < 5; ++x) {
            b[y + 5 * ((2 * x + 3 * y) % 5)] = rotl(s[x + 5 * y] ^ d[x], FILL_KECCAK_RHO[x + 5 * y]);
        }
    }
}

// One round of Keccak-f[1600] on `s`.
__device__ __forceinline__ void keccak_round(uint64_t s[25], uint64_t rc) {
    uint64_t cpar[5], d[5], b[25];
    theta_d(s, cpar, d);
    rho_pi(s, d, b);
#pragma unroll
    for (int y = 0; y < 5; ++y) {
#pragma unroll
        for (int x = 0; x < 5; ++x) {
            s[x + 5 * y] = b[x + 5 * y] ^ (~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y]);
        }
    }
    s[0] ^= rc;
}

// `hwsl` on each halfword of `v` at once: the halfwords shifted left by `n`
// (their low part, `rot_left`) and the bits they carry out (`rot_right`).
__device__ __forceinline__ void halfword_shift(uint64_t v, uint32_t n, uint64_t &left,
                                               uint64_t &right) {
    if (n == 0) {
        left = v;
        right = 0;
        return;
    }
    const uint64_t keep_high = ((0xFFFFull << n) & 0xFFFFull) * 0x0001000100010001ull;
    const uint64_t keep_low = ((1ull << n) - 1) * 0x0001000100010001ull;
    left = (v << n) & keep_high;
    right = (v >> (16 - n)) & keep_low;
}

}  // namespace fill

// Rows `[skip, skip + rows)` of the table `ops` make (24 rows an op, op `k`'s
// round `r` at row `24k + r`), as a table of `rows` rows; rows past the last
// op are zero (μ = 0). An op the slice cuts writes only its rows inside. Grid:
// x a run of rows, one thread a row, so a column's stores coalesce; y the
// column section (`FILL_KECCAK_SECTIONS`). `Op` is either record: the rows
// read its timestamp and input alone.
template <typename Op>
__device__ __forceinline__ void fill_keccak_rnd_rows(const Op *ops, uint32_t nops, uint64_t skip,
                                                     uint64_t rows, uint64_t *out) {
    namespace c = keccak_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint32_t section = blockIdx.y;
    const fill::RowWriter w{out, rows, row};
    const uint32_t first_col = section == 0 ? 0 : (section == 1 ? c::THETA : c::CHI_ANDS);
    const uint32_t end_col = section == 0 ? c::THETA : (section == 1 ? c::CHI_ANDS : c::NUM_COLUMNS);

    const uint64_t at = skip + row;
    const uint64_t op = at / 24;
    const uint32_t round = (uint32_t)(at % 24);
    if (op >= nops) {
        w.zeros(first_col, end_col);
        return;
    }

    // The state at the start of this row's round.
    uint64_t s[25];
#pragma unroll
    for (int i = 0; i < 25; ++i) s[i] = ops[op].input[i];
    for (uint32_t r = 0; r < round; ++r) fill::keccak_round(s, FILL_KECCAK_RC[r]);

    uint64_t cpar[5], d[5];
    fill::theta_d(s, cpar, d);

    if (section == 0) {
        w.dword_wl(c::TIMESTAMP_0, ops[op].timestamp);
        w.word(c::ROUND, round);
#pragma unroll
        for (int lane = 0; lane < 25; ++lane) w.dword_bl(c::START + 8 * lane, s[lane]);
#pragma unroll
        for (int x = 0; x < 5; ++x) {
            // The parity chain: stage k is lanes y = 0..k+1 of column x.
            uint64_t chain = s[x] ^ s[x + 5];
            w.dword_bl(c::CXZ + (4 * x) * 8, chain);
#pragma unroll
            for (int stage = 1; stage < 4; ++stage) {
                chain ^= s[x + 5 * (stage + 1)];
                w.dword_bl(c::CXZ + (4 * x + stage) * 8, chain);
            }
            // HWSL(C[x], 1): the halfwords shifted by one and their carry bits.
            uint64_t left, right;
            fill::halfword_shift(cpar[x], 1, left, right);
            w.dword_bl(c::CXZ_LEFT + 8 * x, left);
#pragma unroll
            for (int hw = 0; hw < 4; ++hw) {
                w.word(c::CXZ_RIGHT + 4 * x + hw, (right >> (16 * hw)) & 1);
            }
            w.dword_bl(c::DXZ + 8 * x, d[x]);
        }
    } else if (section == 1) {
#pragma unroll
        for (int y = 0; y < 5; ++y) {
#pragma unroll
            for (int x = 0; x < 5; ++x) {
                const int lane = x + 5 * y;
                const uint64_t theta = s[lane] ^ d[x];
                w.dword_bl(c::THETA + 8 * lane, theta);
                uint64_t left, right;
                fill::halfword_shift(theta, FILL_KECCAK_RHO[lane] % 16, left, right);
                w.dword_bl(c::ROT_LEFT + 8 * lane, left);
                w.dword_bl(c::ROT_RIGHT + 8 * lane, right);
            }
        }
    } else {
        uint64_t b[25];
        fill::rho_pi(s, d, b);
        const uint64_t rc = FILL_KECCAK_RC[round];
#pragma unroll
        for (int y = 0; y < 5; ++y) {
#pragma unroll
            for (int x = 0; x < 5; ++x) {
                const int lane = x + 5 * y;
                const uint64_t ands = ~b[(x + 1) % 5 + 5 * y] & b[(x + 2) % 5 + 5 * y];
                const uint64_t chi = b[lane] ^ ands;
                w.dword_bl(c::CHI_ANDS + 8 * lane, ands);
                w.dword_bl(c::CHI + 8 * lane, chi);
                if (lane == 0) w.dword_bl(c::IOTA, chi ^ rc);
            }
        }
        w.dword_bl(c::RC, rc);
        w.word(c::MU, 1);
    }
}

extern "C" __global__ void fill_keccak_rnd(const KeccakOpC *ops, uint32_t nops, uint64_t skip,
                                           uint64_t rows, uint64_t *out) {
    fill_keccak_rnd_rows(ops, nops, skip, rows, out);
}

// The same, over the records a table held as its keccak operations uploads.
extern "C" __global__ void fill_keccak_rnd_in(const KeccakOpInC *ops, uint32_t nops,
                                              uint64_t skip, uint64_t rows, uint64_t *out) {
    fill_keccak_rnd_rows(ops, nops, skip, rows, out);
}

// ---------------------------------------------------------------------------
// The LFM width-16 Poseidon1 socket (`prover/src/lfm/p1w16_socket.rs`): its
// row-major trace widened from the columns the host uploads — the preprocessed
// prefix and the sixteen input lanes, `narrow_w = prep_w + 16` words a row —
// to the whole row (`width` words): the prefix and inputs copied, then the
// round witness, `x³` and `x⁷` of every S-boxed lane of every round, in
// `fill_row`'s column order from `rounds_off`. Rows at or past `real_rows` are
// padding: their witness words are zero, as the host's calloc'd trace leaves
// them. Row-major in and out (`out[row · width + col]`), the R1 main commit's
// input layout.
//
// BIT-IDENTICAL to the host fill, not only congruent: the trace words feed the
// LDE and the commitment as stored. So every operation is the host's own on
// raw `[0, 2^64)` words: `goldilocks::add`/`mul` port the host's `add`/`mul`
// one to one; `x³ = (a·a)·a` and `x⁷ = (x³·x³)·a` keep `fill_row`'s
// association; the MDS is `poseidon1_w16::mds` — the exact 128-bit lane sum
// `Σ s_j·M[i][j]` over raw words, then `Fp::from(lo) + Fp::from(hi·EPSILON)`,
// where `Fp::from` reduces a word ≥ p once (`canonical`) and `hi·EPSILON`
// (< 2^41) is already reduced; the round constants go through `Fp::from` too.
// ---------------------------------------------------------------------------
namespace lfm_socket {

constexpr int WIDTH = 16;
constexpr int ROUNDS = 30;
constexpr int HALF_FULL = 4;

__device__ __forceinline__ bool full_round(int r) { return r < HALF_FULL || r >= ROUNDS - HALF_FULL; }

// `poseidon1_w16::mds`, word for word.
__device__ __forceinline__ void mds(const uint64_t s[WIDTH], uint64_t out[WIDTH]) {
    uint32_t lo32[WIDTH], hi32[WIDTH];
#pragma unroll
    for (int j = 0; j < WIDTH; ++j) {
        lo32[j] = (uint32_t)s[j];
        hi32[j] = (uint32_t)(s[j] >> 32);
    }
#pragma unroll
    for (int i = 0; i < WIDTH; ++i) {
        // Σ_j c·lo32_j and Σ_j c·hi32_j, each < 371·2^32 < 2^41: the lane's
        // exact sum is acc_hi·2^32 + acc_lo < 2^73.
        uint64_t acc_lo = 0, acc_hi = 0;
#pragma unroll
        for (int j = 0; j < WIDTH; ++j) {
            const uint64_t c = p1w16::MDS_ROW[(j + WIDTH - i) % WIDTH];
            acc_lo += c * (uint64_t)lo32[j];
            acc_hi += c * (uint64_t)hi32[j];
        }
        const uint64_t lo = (acc_hi << 32) + acc_lo;
        const uint64_t carry = (lo < acc_lo) ? 1ull : 0ull;
        const uint64_t hi = (acc_hi >> 32) + carry;
        out[i] = goldilocks::add(goldilocks::canonical(lo), hi * goldilocks::EPSILON);
    }
}

}  // namespace lfm_socket

extern "C" __global__ void fill_lfm_p1w16_socket(const uint64_t *__restrict__ narrow, uint64_t rows,
                                                 uint64_t real_rows, uint64_t prep_w, uint64_t width,
                                                 uint64_t rounds_off, uint64_t *__restrict__ out) {
    using namespace lfm_socket;
    const uint64_t narrow_w = prep_w + WIDTH;
    for (uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; row < rows;
         row += (uint64_t)gridDim.x * blockDim.x) {
        const uint64_t *src = narrow + row * narrow_w;
        uint64_t *dst = out + row * width;
        for (uint64_t c = 0; c < narrow_w; ++c) dst[c] = src[c];
        if (row >= real_rows) {
            for (uint64_t c = narrow_w; c < width; ++c) dst[c] = 0;
            continue;
        }
        uint64_t s[WIDTH];
#pragma unroll
        for (int i = 0; i < WIDTH; ++i) s[i] = src[prep_w + i];
        uint64_t off = rounds_off;
        for (int r = 0; r < ROUNDS; ++r) {
            uint64_t f[WIDTH];
#pragma unroll
            for (int i = 0; i < WIDTH; ++i) f[i] = goldilocks::add(s[i], goldilocks::canonical(p1w16::RC[r][i]));
            const int lanes = full_round(r) ? WIDTH : 1;
            for (int lane = 0; lane < lanes; ++lane) {
                const uint64_t a = f[lane];
                const uint64_t x3 = goldilocks::mul(goldilocks::mul(a, a), a);
                const uint64_t x7 = goldilocks::mul(goldilocks::mul(x3, x3), a);
                dst[off + lane] = x3;
                dst[off + lanes + lane] = x7;
                f[lane] = x7;
            }
            off += 2 * (uint64_t)lanes;
            mds(f, s);
        }
    }
}

// ---------------------------------------------------------------------------
// The word memory format's tables (`mem_load8.rs`, `mem_store8.rs`, `mem_loadw.rs`,
// `mem_storew.rs`, `mem_load2w.rs`, `mem_store2w.rs`, `memw_w.rs`: each one's
// `generate_*_trace_segments_as`), filled from the walk's rows. Each row is the
// prover's struct, field for field; padding rows are zero in every column.
// ---------------------------------------------------------------------------

// MEM_LOAD8's row: an aligned LD of `word`, `step` = ts − old_ts − 1.
struct Load8RowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t pad[4];
    uint64_t word;
};
static_assert(sizeof(Load8RowC) == 32, "Load8RowC is 32 bytes");
static_assert(offsetof(Load8RowC, step) == 16, "Load8RowC field order");
static_assert(offsetof(Load8RowC, word) == 24, "Load8RowC field order");

// MEM_STORE8's row: an aligned SD of `new_word` over `old`.
struct Store8RowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t pad[4];
    uint64_t old;
    uint64_t new_word;
};
static_assert(sizeof(Store8RowC) == 40, "Store8RowC is 40 bytes");
static_assert(offsetof(Store8RowC, old) == 24, "Store8RowC field order");
static_assert(offsetof(Store8RowC, new_word) == 32, "Store8RowC field order");

// MEM_LOADW's row: a load of 1, 2 or 4 bytes inside one word.
struct LoadWRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t width;
    uint8_t is_signed;
    uint8_t pad[2];
    uint64_t word;
};
static_assert(sizeof(LoadWRowC) == 32, "LoadWRowC is 32 bytes");
static_assert(offsetof(LoadWRowC, width) == 20, "LoadWRowC field order");
static_assert(offsetof(LoadWRowC, is_signed) == 21, "LoadWRowC field order");
static_assert(offsetof(LoadWRowC, word) == 24, "LoadWRowC field order");

// MEM_STOREW's row: a store of 1, 2 or 4 bytes of `value` inside the word `old`.
struct StoreWRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t width;
    uint8_t pad[3];
    uint64_t value;
    uint64_t old;
};
static_assert(sizeof(StoreWRowC) == 40, "StoreWRowC is 40 bytes");
static_assert(offsetof(StoreWRowC, width) == 20, "StoreWRowC field order");
static_assert(offsetof(StoreWRowC, value) == 24, "StoreWRowC field order");
static_assert(offsetof(StoreWRowC, old) == 32, "StoreWRowC field order");

// MEM_LOAD2W's row: a load of 2, 4 or 8 bytes across two words, the one holding
// `address` first.
struct Load2WRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t steps[2];
    uint64_t words[2];
    uint8_t width;
    uint8_t is_signed;
    uint8_t pad[6];
};
static_assert(sizeof(Load2WRowC) == 48, "Load2WRowC is 48 bytes");
static_assert(offsetof(Load2WRowC, steps) == 16, "Load2WRowC field order");
static_assert(offsetof(Load2WRowC, words) == 24, "Load2WRowC field order");
static_assert(offsetof(Load2WRowC, width) == 40, "Load2WRowC field order");
static_assert(offsetof(Load2WRowC, is_signed) == 41, "Load2WRowC field order");

// MEM_STORE2W's row: a store of 2, 4 or 8 bytes of `value` across the two words `old`.
struct Store2WRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t steps[2];
    uint64_t value;
    uint64_t old[2];
    uint8_t width;
    uint8_t pad[7];
};
static_assert(sizeof(Store2WRowC) == 56, "Store2WRowC is 56 bytes");
static_assert(offsetof(Store2WRowC, value) == 24, "Store2WRowC field order");
static_assert(offsetof(Store2WRowC, old) == 32, "Store2WRowC field order");
static_assert(offsetof(Store2WRowC, width) == 48, "Store2WRowC field order");

// MEMW_W's row: a chip's word access, `kind` 0 (an access message) or 1 (a write).
struct MemwWordRowC {
    uint64_t address;
    uint64_t timestamp;
    uint32_t step;
    uint8_t kind;
    uint8_t pad[3];
    uint64_t old;
    uint64_t new_word;
};
static_assert(sizeof(MemwWordRowC) == 40, "MemwWordRowC is 40 bytes");
static_assert(offsetof(MemwWordRowC, kind) == 20, "MemwWordRowC field order");
static_assert(offsetof(MemwWordRowC, old) == 24, "MemwWordRowC field order");
static_assert(offsetof(MemwWordRowC, new_word) == 32, "MemwWordRowC field order");

namespace mem_load8_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t LO = 3;
constexpr uint32_t HI = 4;
constexpr uint32_t STEP_0 = 5;
constexpr uint32_t STEP_1 = 6;
constexpr uint32_t MU = 7;
constexpr uint32_t NUM_COLUMNS = 8;
}  // namespace mem_load8_cols

namespace mem_store8_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t OLD_LO = 3;
constexpr uint32_t OLD_HI = 4;
constexpr uint32_t NEW_0 = 5;  // 4 columns
constexpr uint32_t STEP_0 = 9;
constexpr uint32_t STEP_1 = 10;
constexpr uint32_t MU = 11;
constexpr uint32_t NUM_COLUMNS = 12;
}  // namespace mem_store8_cols

namespace mem_loadw_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t OFF_0 = 3;  // 8 columns
constexpr uint32_t READ2 = 11;
constexpr uint32_t READ4 = 12;
constexpr uint32_t SIGNED = 13;
constexpr uint32_t BYTES_0 = 14;  // 8 columns
constexpr uint32_t RES_0 = 22;    // 4 columns
constexpr uint32_t SIGN_BIT = 26;
constexpr uint32_t EXT = 27;
constexpr uint32_t STEP_0 = 28;
constexpr uint32_t STEP_1 = 29;
constexpr uint32_t MU = 30;
constexpr uint32_t NUM_COLUMNS = 31;
}  // namespace mem_loadw_cols

namespace mem_storew_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t OFF_0 = 3;  // 8 columns
constexpr uint32_t WRITE2 = 11;
constexpr uint32_t WRITE4 = 12;
constexpr uint32_t OLD_0 = 13;    // 8 columns
constexpr uint32_t VALUE_0 = 21;  // 4 columns
constexpr uint32_t VALUE_HI = 25;
constexpr uint32_t NEW_LO = 26;
constexpr uint32_t NEW_HI = 27;
constexpr uint32_t STEP_0 = 28;
constexpr uint32_t STEP_1 = 29;
constexpr uint32_t MU = 30;
constexpr uint32_t NUM_COLUMNS = 31;
}  // namespace mem_storew_cols

namespace mem_load2w_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t OFF_0 = 3;  // 8 columns
constexpr uint32_t READ2 = 11;
constexpr uint32_t READ4 = 12;
constexpr uint32_t READ8 = 13;
constexpr uint32_t SIGNED = 14;
constexpr uint32_t BYTES_0 = 15;  // 16 columns
constexpr uint32_t RES_0 = 31;    // 8 columns
constexpr uint32_t SIGN_BIT = 39;
constexpr uint32_t EXT = 40;
constexpr uint32_t STEP_0 = 41;
constexpr uint32_t STEP_1 = 42;
constexpr uint32_t STEP2_0 = 43;
constexpr uint32_t STEP2_1 = 44;
constexpr uint32_t MU = 45;
constexpr uint32_t CARRY = 46;
constexpr uint32_t NUM_COLUMNS = 47;
}  // namespace mem_load2w_cols

namespace mem_store2w_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t OFF_0 = 3;  // 8 columns
constexpr uint32_t WRITE2 = 11;
constexpr uint32_t WRITE4 = 12;
constexpr uint32_t WRITE8 = 13;
constexpr uint32_t OLD_0 = 14;    // 16 columns
constexpr uint32_t VALUE_0 = 30;  // 8 columns
constexpr uint32_t NEW1_LO = 38;
constexpr uint32_t NEW1_HI = 39;
constexpr uint32_t NEW2_LO = 40;
constexpr uint32_t NEW2_HI = 41;
constexpr uint32_t STEP_0 = 42;
constexpr uint32_t STEP_1 = 43;
constexpr uint32_t STEP2_0 = 44;
constexpr uint32_t STEP2_1 = 45;
constexpr uint32_t MU = 46;
constexpr uint32_t CARRY = 47;
constexpr uint32_t NUM_COLUMNS = 48;
}  // namespace mem_store2w_cols

namespace memw_w_cols {
constexpr uint32_t ADDRESS_LO = 0;
constexpr uint32_t ADDRESS_HI = 1;
constexpr uint32_t TIMESTAMP = 2;
constexpr uint32_t OLD_LO = 3;
constexpr uint32_t OLD_HI = 4;
constexpr uint32_t NEW_LO = 5;
constexpr uint32_t NEW_HI = 6;
constexpr uint32_t STEP_0 = 7;
constexpr uint32_t STEP_1 = 8;
constexpr uint32_t MU_WRITE = 9;
constexpr uint32_t MU_ACCESS = 10;
constexpr uint32_t NUM_COLUMNS = 11;
}  // namespace memw_w_cols

namespace fill {

// A word table's address, timestamp and steps: `ADDRESS_LO`, `ADDRESS_HI`
// (`halves`), `TIMESTAMP` (`set_u64`).
__device__ __forceinline__ void word_head(const RowWriter &w, uint64_t address, uint64_t timestamp) {
    w.word(0, address & 0xFFFFFFFFull);
    w.word(1, address >> 32);
    w.word(2, timestamp);
}

// `step_limbs`: the step's low and high 16 bits.
__device__ __forceinline__ void step_limbs(const RowWriter &w, uint32_t col0, uint32_t col1, uint32_t step) {
    w.word(col0, step & 0xFFFFu);
    w.word(col1, step >> 16);
}

// The one-hot offset columns `OFF[0..8]`: 1 at the address's offset in its word.
__device__ __forceinline__ void offset_one_hot(const RowWriter &w, uint32_t col, uint32_t offset) {
#pragma unroll
    for (uint32_t k = 0; k < 8; ++k) w.word(col + k, k == offset);
}

// `word_cpu::load_result` over `N` little-endian bytes: the result's eight bytes
// (the `width` bytes from `offset`, then the sign or zero fill) and the sign bit
// (the top bit of the highest byte read). Indices wrap at `N`, which a row the
// prover made (its offset and width inside the bytes) never reaches.
template <uint32_t N>
__device__ __forceinline__ bool load_result(const uint8_t (&bytes)[N], uint32_t offset, uint32_t width,
                                            bool is_signed, uint8_t (&res)[8]) {
    const bool sign_bit = (bytes[(offset + width - 1) & (N - 1)] >> 7) & 1;
    const uint8_t fill_byte = (is_signed && sign_bit) ? 0xFF : 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) res[i] = i < width ? bytes[(offset + i) & (N - 1)] : fill_byte;
    return sign_bit;
}

// `word_cpu::store_into`: the `width` low bytes of `value` written over `bytes`
// from `offset` (indices wrapping as above).
template <uint32_t N>
__device__ __forceinline__ void store_into(uint8_t (&bytes)[N], uint32_t offset, uint32_t width, uint64_t value) {
    for (uint32_t i = 0; i < width && i < 8; ++i) bytes[(offset + i) & (N - 1)] = (value >> (8 * i)) & 0xFF;
}

__device__ __forceinline__ void le_bytes(uint64_t v, uint8_t *out) {
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) out[i] = (v >> (8 * i)) & 0xFF;
}

__device__ __forceinline__ uint64_t le_word(const uint8_t *bytes) {
    uint64_t v = 0;
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) v |= (uint64_t)bytes[i] << (8 * i);
    return v;
}

// `word_cpu::carries`: the first word is the last of its low 2^32 range, so the
// second word's key carries into the high word.
__device__ __forceinline__ bool carries(uint64_t address) {
    return (address & 0xFFFFFFFFull) - (address & 7) == 0xFFFFFFF8ull;
}

}  // namespace fill

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_load8(const Load8RowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = mem_load8_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const Load8RowC r = rows[row];
    fill::word_head(w, r.address, r.timestamp);
    w.dword_wl(c::LO, r.word);
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.step);
    w.word(c::MU, 1);
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_store8(const Store8RowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = mem_store8_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const Store8RowC r = rows[row];
    fill::word_head(w, r.address, r.timestamp);
    w.dword_wl(c::OLD_LO, r.old);
    w.dword_hl(c::NEW_0, r.new_word);
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.step);
    w.word(c::MU, 1);
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_loadw(const LoadWRowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = mem_loadw_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const LoadWRowC r = rows[row];
    const uint32_t offset = (uint32_t)(r.address & 7);
    const bool is_signed = r.is_signed != 0;
    fill::word_head(w, r.address, r.timestamp);
    fill::offset_one_hot(w, c::OFF_0, offset);
    w.word(c::READ2, r.width == 2);
    w.word(c::READ4, r.width == 4);
    w.word(c::SIGNED, is_signed);
    uint8_t bytes[8];
    fill::le_bytes(r.word, bytes);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::BYTES_0 + i, bytes[i]);
    uint8_t res[8];
    const bool sign_bit = fill::load_result(bytes, offset, r.width, is_signed, res);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) w.word(c::RES_0 + i, res[i]);
    w.word(c::SIGN_BIT, sign_bit);
    w.word(c::EXT, is_signed && sign_bit);
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.step);
    w.word(c::MU, 1);
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_storew(const StoreWRowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = mem_storew_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const StoreWRowC r = rows[row];
    const uint32_t offset = (uint32_t)(r.address & 7);
    fill::word_head(w, r.address, r.timestamp);
    fill::offset_one_hot(w, c::OFF_0, offset);
    w.word(c::WRITE2, r.width == 2);
    w.word(c::WRITE4, r.width == 4);
    uint8_t bytes[8];
    fill::le_bytes(r.old, bytes);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::OLD_0 + i, bytes[i]);
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i) w.word(c::VALUE_0 + i, (r.value >> (8 * i)) & 0xFF);
    w.word(c::VALUE_HI, r.value >> 32);
    fill::store_into(bytes, offset, r.width, r.value);
    w.dword_wl(c::NEW_LO, fill::le_word(bytes));
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.step);
    w.word(c::MU, 1);
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_load2w(const Load2WRowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = mem_load2w_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const Load2WRowC r = rows[row];
    const uint32_t offset = (uint32_t)(r.address & 7);
    const bool is_signed = r.is_signed != 0;
    fill::word_head(w, r.address, r.timestamp);
    fill::offset_one_hot(w, c::OFF_0, offset);
    w.word(c::READ2, r.width == 2);
    w.word(c::READ4, r.width == 4);
    w.word(c::READ8, r.width == 8);
    w.word(c::SIGNED, is_signed);
    uint8_t bytes[16];
    fill::le_bytes(r.words[0], bytes);
    fill::le_bytes(r.words[1], bytes + 8);
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) w.word(c::BYTES_0 + i, bytes[i]);
    uint8_t res[8];
    const bool sign_bit = fill::load_result(bytes, offset, r.width, is_signed, res);
#pragma unroll
    for (uint32_t i = 0; i < 8; ++i) w.word(c::RES_0 + i, res[i]);
    w.word(c::SIGN_BIT, sign_bit);
    w.word(c::EXT, is_signed && sign_bit);
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.steps[0]);
    fill::step_limbs(w, c::STEP2_0, c::STEP2_1, r.steps[1]);
    w.word(c::MU, 1);
    w.word(c::CARRY, fill::carries(r.address));
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_mem_store2w(const Store2WRowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = mem_store2w_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const Store2WRowC r = rows[row];
    const uint32_t offset = (uint32_t)(r.address & 7);
    fill::word_head(w, r.address, r.timestamp);
    fill::offset_one_hot(w, c::OFF_0, offset);
    w.word(c::WRITE2, r.width == 2);
    w.word(c::WRITE4, r.width == 4);
    w.word(c::WRITE8, r.width == 8);
    uint8_t bytes[16];
    fill::le_bytes(r.old[0], bytes);
    fill::le_bytes(r.old[1], bytes + 8);
#pragma unroll
    for (uint32_t i = 0; i < 16; ++i) w.word(c::OLD_0 + i, bytes[i]);
    w.dword_bl(c::VALUE_0, r.value);
    fill::store_into(bytes, offset, r.width, r.value);
    w.dword_wl(c::NEW1_LO, fill::le_word(bytes));
    w.dword_wl(c::NEW2_LO, fill::le_word(bytes + 8));
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.steps[0]);
    fill::step_limbs(w, c::STEP2_0, c::STEP2_1, r.steps[1]);
    w.word(c::MU, 1);
    w.word(c::CARRY, fill::carries(r.address));
}

// Rows `[0, n)` from `rows`, rows `[n, height)` zero. One thread a row.
extern "C" __global__ void fill_memw_w(const MemwWordRowC *rows, uint64_t n, uint64_t height, uint64_t *out) {
    namespace c = memw_w_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    if (row >= n) {
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const MemwWordRowC r = rows[row];
    fill::word_head(w, r.address, r.timestamp);
    w.dword_wl(c::OLD_LO, r.old);
    w.dword_wl(c::NEW_LO, r.new_word);
    fill::step_limbs(w, c::STEP_0, c::STEP_1, r.step);
    w.word(c::MU_WRITE, r.kind == 1);
    w.word(c::MU_ACCESS, r.kind != 1);
}
