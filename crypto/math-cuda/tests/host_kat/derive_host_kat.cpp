// Known-answer replay of `kernels/card_rows.cu`'s `derive_registers_from_cache`
// (MEMW_R's register rows built on the card from the CPU logs it holds,
// `LAMBDA_VM_CARD_ROWS=registers`), run on the host.
//
// WHY THIS EXISTS. The card test checks the kernel on a GPU; this checks its
// arithmetic and indexing without one, before a GPU run is spent on it. The
// vectors come from the prover (the ignored test
// `register_rows_tests::write_the_register_rows_kernel_vectors`): record-mode
// runs' MEMW_R chunks as the host plans them (`card_rows::plan_registers`),
// the logs of the CPU tables taken as held, the program's DECODE rows, and the
// words the prover's host fill writes for each chunk. Nothing here computes an
// expected word.
//
// WHAT IT CHECKS, per table:
//   1. the replay writes the host fill's words, every cell, and counts nothing
//      into the check;
//   2. the kernel's mutants (rs2 read as rs1; a write's old value read as the
//      written value whatever the slot) and the plans made under the host's
//      mutants (a write before its CPU table sent as timestamps; ECALL rows
//      sent as timestamps) each change a word or count into the check on at
//      least one table.
// Every cell starts as a sentinel, so a cell no thread writes is a mismatch.
//
//   3. words mode (`LAMBDA_VM_CARD_ROWS=words`): the same runs walked as words; each
//      chunk's words as the walk wrote them, its markers' rows at their run index
//      less the table's `whole_base` (at least one table past the run's first
//      whole rows), against the host fill of its decoded rows; the two kernel
//      mutants on them.
// WHAT IT DOES NOT: nvcc, the grid, the launch over several tables, timing,
// the upload's choice to launch (a unit test in `src/card_rows.rs`).
// That is the card test's (`card_rows_card_tests`).
//
// Build and run (from the repository root):
//   REGISTER_ROWS_VECTORS=/tmp/rr.bin cargo test -p lambda-vm-prover --lib -- --ignored --exact \
//     tables::trace_builder::windowed::parallel::register_rows_tests::write_the_register_rows_kernel_vectors
//   c++ -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Icrypto/math-cuda/tests/host_kat -Icrypto/math-cuda/kernels \
//     -o target/host_kat/derive_host_kat crypto/math-cuda/tests/host_kat/derive_host_kat.cpp
//   target/host_kat/derive_host_kat /tmp/rr.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_host_shim.h"

#include "card_rows.cu"

namespace {

struct Reader {
    std::vector<uint8_t> bytes;
    size_t at = 0;
    uint64_t u64() {
        if (at + 8 > bytes.size()) {
            std::fprintf(stderr, "vectors end early at %zu\n", at);
            std::exit(2);
        }
        uint64_t v;
        std::memcpy(&v, bytes.data() + at, 8);
        at += 8;
        return v;
    }
    template <class T>
    std::vector<T> records(uint64_t n) {
        std::vector<T> v(n);
        const size_t len = n * sizeof(T);
        if (at + len > bytes.size()) {
            std::fprintf(stderr, "vectors end early at %zu\n", at);
            std::exit(2);
        }
        if (len) std::memcpy(v.data(), bytes.data() + at, len);
        at += len;
        return v;
    }
};

struct Plan {
    std::vector<uint64_t> words;
    std::vector<RegRowC> whole;
};

Plan read_plan(Reader &r) {
    Plan p;
    p.words = r.records<uint64_t>(r.u64());
    p.whole = r.records<RegRowC>(r.u64());
    return p;
}

struct Table {
    uint64_t height;
    Plan plan;
    std::vector<uint64_t> want;  // column-major, height × 10
    bool has[2];
    Plan faulted[2];  // the host mutants' plans: a write before its table, ECALL rows
};

constexpr uint64_t SENTINEL = 0xdeadbeefcafef00dull;

// One table replayed: thread `i` is row `i`. Returns (cells differing, check).
std::pair<uint64_t, unsigned long long> replay(const std::vector<CachedLogsC> &entries,
                                               const std::vector<DecodeEntryC> &dec,
                                               const std::vector<DecodeRunC> &runs,
                                               const Table &t, const Plan &plan, uint32_t fault,
                                               uint64_t whole_base = 0) {
    std::vector<uint64_t> out(t.height * memw_r_cols::NUM_COLUMNS, SENTINEL);
    const RegistersTableC table{plan.words.data(), plan.whole.data(), plan.words.size(), t.height,
                                out.data(), 0, whole_base};
    unsigned long long check = 0;
    for (uint64_t i = 0; i < plan.words.size(); ++i) {
        blockIdx.x = 0;
        blockDim.x = 0;
        threadIdx.x = (unsigned)i;
        derive_registers_from_cache(entries.data(), (uint32_t)entries.size(), &table, 1,
                                    plan.words.size(), dec.data(), runs.data(),
                                    (uint32_t)runs.size(), fault, &check);
    }
    uint64_t differ = 0;
    for (size_t k = 0; k < out.size(); ++k) differ += out[k] != t.want[k];
    return {differ, check};
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <vectors>\n", argv[0]);
        return 2;
    }
    Reader r;
    {
        FILE *f = std::fopen(argv[1], "rb");
        if (!f) {
            std::perror(argv[1]);
            return 2;
        }
        std::fseek(f, 0, SEEK_END);
        r.bytes.resize((size_t)std::ftell(f));
        std::fseek(f, 0, SEEK_SET);
        if (std::fread(r.bytes.data(), 1, r.bytes.size(), f) != r.bytes.size()) return 2;
        std::fclose(f);
    }
    if (r.bytes.size() < 8 || std::memcmp(r.bytes.data(), "CRREG002", 8) != 0) {
        std::fprintf(stderr, "not register-row vectors\n");
        return 2;
    }
    r.at = 8;
    const uint64_t cases = r.u64();
    uint64_t tables = 0, rows = 0, failures = 0, words_tables = 0, words_based = 0;
    uint64_t words_applies[2] = {0, 0}, words_moved[2] = {0, 0};
    // Per mutant: tables it applies to, tables it moved.
    uint64_t applies[4] = {0, 0, 0, 0}, moved[4] = {0, 0, 0, 0};
    const char *names[4] = {"rs2 read as rs1", "old value read whatever its slot",
                            "write before its table sent as timestamps",
                            "ECALL rows sent as timestamps"};
    for (uint64_t c = 0; c < cases; ++c) {
        const uint64_t nentries = r.u64();
        std::vector<std::vector<LogC>> logs(nentries);
        std::vector<CachedLogsC> entries(nentries);
        for (uint64_t e = 0; e < nentries; ++e) {
            const uint64_t first = r.u64(), n = r.u64();
            logs[e] = r.records<LogC>(n);
            entries[e] = CachedLogsC{first, n, logs[e].data()};
        }
        const auto dec = r.records<DecodeEntryC>(r.u64());
        const auto runs = r.records<DecodeRunC>(r.u64());
        const uint64_t ntables = r.u64();
        for (uint64_t k = 0; k < ntables; ++k) {
            Table t;
            const uint64_t n = r.u64(), cells = r.u64();
            t.height = n;
            t.plan = read_plan(r);
            t.want = r.records<uint64_t>(cells);
            for (int f = 0; f < 2; ++f) {
                t.has[f] = r.u64() != 0;
                if (t.has[f]) t.faulted[f] = read_plan(r);
            }
            if (t.plan.words.size() != n || cells != n * memw_r_cols::NUM_COLUMNS) {
                std::fprintf(stderr, "case %llu table %llu: malformed\n", (unsigned long long)c,
                             (unsigned long long)k);
                return 2;
            }
            ++tables;
            rows += n;
            const auto clean = replay(entries, dec, runs, t, t.plan, 0);
            if (clean.first != 0 || clean.second != 0) {
                ++failures;
                std::printf("FAIL case %llu table %llu: %llu cells differ, check %llu\n",
                            (unsigned long long)c, (unsigned long long)k,
                            (unsigned long long)clean.first, clean.second);
            }
            const uint32_t kernel_faults[2] = {REGISTER_FAULT_SLOT, REGISTER_FAULT_VALUE_AT};
            for (int f = 0; f < 2; ++f) {
                const auto m = replay(entries, dec, runs, t, t.plan, kernel_faults[f]);
                ++applies[f];
                moved[f] += m.first != 0 || m.second != 0;
            }
            for (int f = 0; f < 2; ++f) {
                if (!t.has[f]) continue;
                const auto m = replay(entries, dec, runs, t, t.faulted[f], 0);
                ++applies[2 + f];
                moved[2 + f] += m.first != 0 || m.second != 0;
            }
        }
        // Words mode: the tables as the walk wrote them, their markers' high
        // halves run indices from `whole_base`.
        const uint64_t nwords = r.u64();
        for (uint64_t k = 0; k < nwords; ++k) {
            Table t;
            const uint64_t n = r.u64(), cells = r.u64(), base = r.u64();
            t.height = n;
            t.plan.words = r.records<uint64_t>(n);
            t.plan.whole = r.records<RegRowC>(r.u64());
            t.want = r.records<uint64_t>(cells);
            if (cells != n * memw_r_cols::NUM_COLUMNS) {
                std::fprintf(stderr, "case %llu words table %llu: malformed\n", (unsigned long long)c,
                             (unsigned long long)k);
                return 2;
            }
            ++words_tables;
            words_based += base > 0;
            rows += n;
            const auto clean = replay(entries, dec, runs, t, t.plan, 0, base);
            if (clean.first != 0 || clean.second != 0) {
                ++failures;
                std::printf("FAIL case %llu words table %llu (base %llu): %llu cells differ, check %llu\n",
                            (unsigned long long)c, (unsigned long long)k, (unsigned long long)base,
                            (unsigned long long)clean.first, clean.second);
            }
            const uint32_t kernel_faults[2] = {REGISTER_FAULT_SLOT, REGISTER_FAULT_VALUE_AT};
            for (int f = 0; f < 2; ++f) {
                const auto m = replay(entries, dec, runs, t, t.plan, kernel_faults[f], base);
                ++words_applies[f];
                words_moved[f] += m.first != 0 || m.second != 0;
            }
        }
    }
    if (r.at != r.bytes.size()) {
        std::fprintf(stderr, "%zu bytes left over\n", r.bytes.size() - r.at);
        return 2;
    }
    bool bites = true;
    for (int f = 0; f < 4; ++f) {
        std::printf("MUTANT %s: moved %llu of %llu tables it changes\n", names[f],
                    (unsigned long long)moved[f], (unsigned long long)applies[f]);
        bites &= moved[f] > 0;
    }
    for (int f = 0; f < 2; ++f) {
        std::printf("MUTANT words, %s: moved %llu of %llu tables\n", names[f],
                    (unsigned long long)words_moved[f], (unsigned long long)words_applies[f]);
        bites &= words_moved[f] > 0;
    }
    const bool words_ok = words_tables >= 2 && words_based > 0;
    std::printf("REGISTER KAT: %llu tables, %llu as words (%llu with a run index past 0), %llu rows, %llu "
                "failed · mutants %s\n",
                (unsigned long long)tables, (unsigned long long)words_tables,
                (unsigned long long)words_based, (unsigned long long)rows,
                (unsigned long long)failures, bites ? "all bite" : "NOT all bite");
    return failures == 0 && bites && tables > 0 && words_ok ? 0 : 1;
}
