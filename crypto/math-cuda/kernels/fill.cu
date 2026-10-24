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
// register access), its width, and flags 0 or 1.
struct AlignedRowC {
    uint64_t base_address;
    uint64_t timestamp;
    uint64_t old_timestamp;
    uint64_t value;
    uint64_t old;
    uint8_t width;
    uint8_t is_read;
    uint8_t is_register;
};
static_assert(sizeof(AlignedRowC) == 48, "AlignedRowC is 48 bytes");
static_assert(offsetof(AlignedRowC, width) == 40, "AlignedRowC field order");
static_assert(offsetof(AlignedRowC, is_register) == 42, "AlignedRowC field order");

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
    w.dword_wl(c::OLD_TIMESTAMP_0, r.old_timestamp);
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
// refuses the table. One thread a row.
extern "C" __global__ void fill_cpu_from_logs(const LogC *logs, uint64_t n,
                                              const DecodeEntryC *entries,
                                              const DecodeRunC *runs, uint32_t nruns,
                                              uint64_t height, uint64_t *out,
                                              unsigned long long *missing) {
    namespace c = cpu_cols;
    uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= height) return;
    const fill::RowWriter w{out, height, row};
    const uint64_t timestamp = 4 * (row + 1);

    if (row >= n) {
        // pc = next_pc = 1, every flag 0.
        w.zeros(0, c::NUM_COLUMNS);
        w.word(c::TIMESTAMP, timestamp);
        w.word(c::PC_0, FILL_CPU_PADDING_PC);
        w.word(c::NEXT_PC_0, FILL_CPU_PADDING_PC);
        return;
    }

    const LogC log = logs[row];
    int64_t at = fill::decode_row(runs, nruns, log.current_pc);
    if (at < 0) {
        atomicAdd(missing, 1ull);
        w.zeros(0, c::NUM_COLUMNS);
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
