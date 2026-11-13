// The card-built LOAD and STORE rows (`src/card_rows.rs`): a group's LOAD or
// STORE tables derived on the card from the CPU logs it already holds, instead of filled from records the host made. Loaded by
// `src/card_rows.rs` and launched only under `LAMBDA_VM_CARD_ROWS=derive` (or `registers`). Under
// `registers`, MEMW_R's register rows are built from the same logs (`derive_registers_from_cache`, below).
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

// ---------------------------------------------------------------------------
// `src/card_rows.rs`'s register rows: a group's MEMW_R tables built on the
// card from the CPU logs it holds (`LAMBDA_VM_CARD_ROWS=registers`). Each row
// comes as one word (`RegisterPlan::words`): its timestamp `ts` in the low
// half and its old timestamp `old` in the high half, or a low half of 0 (no
// timestamp is) and the index of the row in the table's whole rows, which
// cross as they are. For a row that comes as its timestamps, by the walk's
// rules (`trace_builder.rs`, the register accesses of a cycle; `history.rs`,
// `step_registers`):
//   - its cycle is `c = (ts - 4) / 4` and its slot `(ts - 4) % 4`: 0 `rs1`,
//     1 `rs2`, 2 `rd`;
//   - its register and value are the cycle's DECODE `rs1` / `rs2` / `rd` and
//     `rv1` / `rv2` / `rvd` (`values`, as for LOAD's and STORE's rows), and
//     it is a read for slots 0 and 1;
//   - a read's old value is its value; a write's is the register's value at
//     `old`: the log of `old`'s cycle by `old`'s slot (`src1_val`,
//     `src2_val`, `dst_val`), the access there being to the same register.
// The host sends a row this way only when that cycle is in the same cached
// CPU table, after its first cycle, and no ECALL is involved; the kernel
// checks each of these, and that the cycle makes such an access within
// MEMW_R's step, and counts a row that fails into `check` (written as zeros):
// the group is then refused.
// ---------------------------------------------------------------------------

// One MEMW_R table of a launch.
struct RegistersTableC {
    const uint64_t *words;
    const RegRowC *whole;
    uint64_t rows;
    uint64_t height;
    uint64_t *out;
    // The launch's rows before this table's.
    uint64_t before;
    // The run index of `whole`'s first row: a marker's high half less this
    // indexes `whole` (0 for a table the host planned).
    uint64_t whole_base;
};
static_assert(sizeof(RegistersTableC) == 56, "RegistersTableC is 56 bytes");

// A card test's mutants (`src/card_rows.rs`, `FAULT_REGISTERS`): `rs2`'s row
// read as `rs1`'s; a write's old value read as the written value of `old`'s
// cycle, whatever its slot.
constexpr uint32_t REGISTER_FAULT_SLOT = 1;
constexpr uint32_t REGISTER_FAULT_VALUE_AT = 2;

namespace derive {

// MEMW_R's row (`fill_memw_r`'s writes, field for field).
__device__ __forceinline__ void register_row(const fill::RowWriter &w, uint32_t reg, uint64_t ts,
                                             uint64_t value, uint64_t old, uint32_t old_ts_lo,
                                             bool is_read) {
    namespace c = memw_r_cols;
    w.word(c::ADDRESS, reg);
    w.dword_wl(c::TIMESTAMP_0, ts);
    w.word(c::VAL_0, value & 0xFFFFFFFFull);
    w.word(c::VAL_1, value >> 32);
    w.word(c::OLD_0, old & 0xFFFFFFFFull);
    w.word(c::OLD_1, old >> 32);
    w.word(c::OLD_TIMESTAMP_LO, old_ts_lo);
    w.word(c::MU_READ, is_read);
    w.word(c::MU_WRITE, !is_read);
}

// The cached entry holding cycle `c`, or none.
__device__ __forceinline__ const CachedLogsC *entry_of(const CachedLogsC *entries,
                                                      uint32_t nentries, uint64_t c) {
    if (nentries == 0) return nullptr;
    uint32_t a = 0, b = nentries;
    while (b - a > 1) {
        const uint32_t mid = (a + b) / 2;
        if (entries[mid].first <= c) {
            a = mid;
        } else {
            b = mid;
        }
    }
    const CachedLogsC *e = &entries[a];
    return (c >= e->first && c - e->first < e->n) ? e : nullptr;
}

}  // namespace derive

extern "C" __global__ void derive_registers_from_cache(
    const CachedLogsC *entries, uint32_t nentries, const RegistersTableC *tables, uint32_t ntables,
    uint64_t total, const DecodeEntryC *dec, const DecodeRunC *runs, uint32_t nruns,
    uint32_t fault, unsigned long long *check) {
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
    const RegistersTableC t = tables[lo];
    const uint64_t r = g - t.before;
    if (r >= t.rows) return;
    const fill::RowWriter w{t.out, t.height, r};
    const uint64_t word = t.words[r];
    if ((uint32_t)word == 0) {
        // A row that crossed whole.
        const RegRowC x = t.whole[(word >> 32) - t.whole_base];
        derive::register_row(w, x.address, x.timestamp,
                              (uint64_t)x.val0 | ((uint64_t)x.val1 << 32),
                              (uint64_t)x.old0 | ((uint64_t)x.old1 << 32), x.old_ts_lo,
                              x.is_read != 0);
        return;
    }
    const uint64_t ts = (uint32_t)word, old = word >> 32;
    // MEMW_R's step: 0 < ts - old <= 2^16.
    bool ok = ts >= 4 && old < ts && ts - old <= (1ull << 16);
    const uint64_t c = ok ? (ts - 4) / 4 : 0;
    const uint32_t slot = ok ? (uint32_t)((ts - 4) % 4) : 3;
    const CachedLogsC *e = derive::entry_of(entries, nentries, c);
    ok = ok && slot < 3 && e != nullptr;
    int64_t at = -1;
    LogC log{};
    if (ok) {
        log = e->logs[c - e->first];
        at = fill::decode_row(runs, nruns, log.current_pc);
    }
    // An ECALL cycle's rows are its handler's: never built here.
    ok = ok && at >= 0 && !dec[at].ecall;
    if (!ok) {
        atomicAdd(check, 1ull);
        w.zeros(0, memw_r_cols::NUM_COLUMNS);
        return;
    }
    const DecodeEntryC d = dec[at];
    const derive::Values v = derive::values(log, d);
    uint32_t k = slot;
    if ((fault & REGISTER_FAULT_SLOT) && k == 1) k = 0;
    const bool on = k == 0 ? d.read_register1 != 0
                           : (k == 1 ? d.read_register2 != 0 : d.write_register != 0);
    const uint32_t reg = k == 0 ? d.rs1 : (k == 1 ? d.rs2 : d.rd);
    const uint64_t value = k == 0 ? v.rv1 : (k == 1 ? v.rv2 : v.rvd);
    ok = on && reg != 0;
    uint64_t before = value;
    if (ok && slot == 2) {
        // The value at `old`: in this table, after its first cycle, and in no
        // ECALL cycle (the host sends any other write whole).
        ok = old >= 4 * e->first + 4;
        const uint64_t j = ok ? (old - 4) / 4 : c;
        const uint32_t sj = ok ? (uint32_t)((old - 4) % 4) : 3;
        ok = ok && j <= c && sj < 3;
        if (ok) {
            const LogC lj = e->logs[j - e->first];
            const int64_t aj = fill::decode_row(runs, nruns, lj.current_pc);
            ok = aj >= 0 && !dec[aj].ecall;
            if (fault & REGISTER_FAULT_VALUE_AT) {
                before = lj.dst_val;
            } else {
                before = sj == 0 ? lj.src1_val : (sj == 1 ? lj.src2_val : lj.dst_val);
            }
        }
    }
    if (!ok) {
        atomicAdd(check, 1ull);
        w.zeros(0, memw_r_cols::NUM_COLUMNS);
        return;
    }
    derive::register_row(w, reg, ts, value, before, (uint32_t)old, slot < 2);
}
