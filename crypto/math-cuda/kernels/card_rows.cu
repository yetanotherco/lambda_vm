// The card-built LOAD and STORE rows (`src/card_rows.rs`): a group's LOAD or
// STORE tables derived on the card from the CPU logs it already holds, instead of filled from records the host made. Loaded by
// `src/card_rows.rs` and launched only under `LAMBDA_VM_CARD_ROWS=derive`.
//
// Each row is written with the words `fill.cu`'s `fill_load` and
// `fill_store` write for the same op (the same writers and columns, included
// below), from the op's cycle's log and DECODE row, as the walk makes the op
// (`CpuOperation::from_log` for the register values). A probe held these
// rules to the walk on a real block's chunks: the card's rows and the host's
// fill of the walk's rows were the same words.

#include "fill.cu"

namespace derive {

// The memory width and signedness bits of `mem_flags` (`types.rs`).
__device__ __forceinline__ uint32_t mem_bytes(uint8_t mem_flags) {
    if ((mem_flags >> 4) & 1) return 8;
    if ((mem_flags >> 3) & 1) return 4;
    if ((mem_flags >> 2) & 1) return 2;
    return 1;
}
__device__ __forceinline__ bool mem_signed(uint8_t mem_flags) { return (mem_flags >> 1) & 1; }

// A cycle's register values and ALU result as `CpuOperation::from_log` makes
// them: a word instruction's carry the log's values (the CPU row zeroes them,
// its register rows do not).
struct Values {
    uint64_t rv1, rv2, rvd, res;
};

__device__ __forceinline__ Values values(const LogC &log, const DecodeEntryC &e) {
    Values v{0, 0, 0, 0};
    uint64_t length = 2 * (uint64_t)e.half_instruction_length;
    if (e.word_instr) {
        v.rv1 = log.src1_val;
        v.rv2 = e.read_register2 ? log.src2_val : 0;
        v.rvd = log.dst_val;
        return v;
    }
    v.rv1 = e.rs1 == 255 ? log.current_pc : (e.read_register1 ? log.src1_val : 0);
    v.rv2 = e.read_register2 ? log.src2_val : 0;
    bool jalr = (e.mem_flags & 1) != 0;
    uint64_t arg2 = e.memory ? e.imm : (e.branch ? v.rv2 : v.rv2 + e.imm);
    bool branch_cond = e.branch && (jalr || fill::branch_taken(e.alu_flags, v.rv1, v.rv2));
    if (e.add) {
        v.res = v.rv1 + arg2;
    } else if (e.sub) {
        v.res = v.rv1 - arg2;
    } else if (e.alu) {
        v.res = e.branch ? (uint64_t)branch_cond : log.dst_val;
    }
    if (e.memory) {
        v.rvd = jalr ? 0 : log.dst_val;
    } else if (e.branch) {
        v.rvd = e.pc + length;
    } else {
        v.rvd = v.res;
    }
    return v;
}

// LOAD's row `row` from cycle `i`'s log and decode (`fill_load`'s writes).
__device__ __forceinline__ void load_row(const LogC *logs, uint64_t first, uint64_t i,
                                         const DecodeEntryC *entries, const DecodeRunC *runs,
                                         uint32_t nruns, uint64_t row, uint64_t height,
                                         uint64_t *out, unsigned long long *check) {
    namespace c = load_cols;
    const LogC log = logs[i];
    const int64_t at = fill::decode_row(runs, nruns, log.current_pc);
    const fill::RowWriter w{out, height, row};
    if (at < 0 || !entries[at].memory || (entries[at].mem_flags & 1) != 0) {
        atomicAdd(check, 1ull);
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const DecodeEntryC d = entries[at];
    const Values v = values(log, d);
    const uint32_t width = mem_bytes(d.mem_flags);
    const bool is_signed = mem_signed(d.mem_flags);
    uint8_t res[8];
    for (uint32_t k = 0; k < 8; ++k) res[k] = (uint8_t)(v.rvd >> (8 * k));
    if (width < 8) {
        const uint8_t fill = (is_signed && (res[width - 1] >> 7)) ? 0xFF : 0;
        for (uint32_t k = width; k < 8; ++k) res[k] = fill;
    }
    w.dword_wl(c::BASE_ADDRESS_0, v.res);
    w.dword_wl(c::TIMESTAMP_0, 4 * (first + i) + 4);
    w.word(c::READ2, width == 2);
    w.word(c::READ4, width == 4);
    w.word(c::READ8, width == 8);
    w.word(c::SIGNED, is_signed);
    for (uint32_t k = 0; k < 8; ++k) w.word(c::RES + k, res[k]);
    w.word(c::SIGN_BIT, (res[width - 1] >> 7) & 1);
    w.word(c::MU, 1);
}

// STORE's row `row` from cycle `i`'s log and decode (`fill_store`'s writes).
__device__ __forceinline__ void store_row(const LogC *logs, uint64_t first, uint64_t i,
                                          const DecodeEntryC *entries, const DecodeRunC *runs,
                                          uint32_t nruns, uint64_t row, uint64_t height,
                                          uint64_t *out, unsigned long long *check) {
    namespace c = store_cols;
    const LogC log = logs[i];
    const int64_t at = fill::decode_row(runs, nruns, log.current_pc);
    const fill::RowWriter w{out, height, row};
    if (at < 0 || !entries[at].memory || (entries[at].mem_flags & 1) == 0) {
        atomicAdd(check, 1ull);
        w.zeros(0, c::NUM_COLUMNS);
        return;
    }
    const DecodeEntryC d = entries[at];
    const Values v = values(log, d);
    const uint32_t width = mem_bytes(d.mem_flags);
    w.dword_wl(c::BASE_ADDRESS_0, v.res);
    w.dword_wl(c::TIMESTAMP_0, 4 * (first + i) + 4);
    w.word(c::WRITE2, width == 2);
    w.word(c::WRITE4, width == 4);
    w.word(c::WRITE8, width == 8);
    w.dword_bl(c::VALUE, v.rv2);
    w.word(c::MU, 1);
}

}  // namespace derive

// ---------------------------------------------------------------------------
// `src/card_rows.rs`'s derived tables: a group's LOAD (`store` 0) or STORE tables
// derived from the CPU logs the card already holds. Each table names its
// rows' cycles (`cycles`, one a row, the global cycle); a row's log is found
// in the cached CPU chunks (`entries`, ascending by first cycle, disjoint).
// One thread a row over every table of the launch (`tables`, each with the
// launch's rows before it). A row whose cycle no entry holds, or whose log is
// not that family's op, is counted into `check` and written as zeros: the
// group is then refused, as a CPU log with no DECODE row refuses it.
// ---------------------------------------------------------------------------

// A CPU chunk's logs on the card: cycles [first, first + n).
struct CachedLogsC {
    uint64_t first;
    uint64_t n;
    const LogC *logs;
};
static_assert(sizeof(CachedLogsC) == 24, "CachedLogsC is 24 bytes");

// One derived table of a launch.
struct RowsTableC {
    const uint32_t *cycles;
    uint64_t rows;
    uint64_t height;
    uint64_t *out;
    // The launch's rows before this table's.
    uint64_t before;
};
static_assert(sizeof(RowsTableC) == 40, "RowsTableC is 40 bytes");

extern "C" __global__ void derive_rows_from_cache(const CachedLogsC *entries, uint32_t nentries,
                                                  const RowsTableC *tables, uint32_t ntables,
                                                  uint64_t total, uint32_t store,
                                                  const DecodeEntryC *dec, const DecodeRunC *runs,
                                                  uint32_t nruns, unsigned long long *check) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= total) return;
    // The table: the last whose rows start at or before `g`.
    uint32_t lo = 0, hi = ntables;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        if (tables[mid].before <= g) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    const RowsTableC t = tables[lo];
    const uint64_t r = g - t.before;
    if (r >= t.rows) return;
    const uint64_t c = t.cycles[r];
    // The entry holding cycle `c`: the last starting at or before it.
    uint32_t a = 0, b = nentries;
    while (b - a > 1) {
        const uint32_t mid = (a + b) / 2;
        if (entries[mid].first <= c) {
            a = mid;
        } else {
            b = mid;
        }
    }
    const CachedLogsC e = nentries ? entries[a] : CachedLogsC{0, 0, nullptr};
    if (c < e.first || c - e.first >= e.n) {
        atomicAdd(check, 1ull);
        const fill::RowWriter w{t.out, t.height, r};
        w.zeros(0, store ? store_cols::NUM_COLUMNS : load_cols::NUM_COLUMNS);
        return;
    }
    if (store) {
        derive::store_row(e.logs, e.first, c - e.first, dec, runs, nruns, r, t.height, t.out, check);
    } else {
        derive::load_row(e.logs, e.first, c - e.first, dec, runs, nruns, r, t.height, t.out, check);
    }
}
