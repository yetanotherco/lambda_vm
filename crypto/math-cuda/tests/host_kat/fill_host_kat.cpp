// Known-answer tests for `kernels/fill.cu`, the table fills from records, run
// on the host.
//
// WHY THIS EXISTS. A fill is right when every column of every row gets the
// word the prover's host fill writes there: which record field, which setter,
// which column, which row. That is arithmetic and indexing, and this checks it
// without a GPU: the kernel source compiles through `cuda_host_shim.h`, and a
// launch is replayed one thread at a time (the kernels' threads are
// independent: one thread a row, disjoint cells). The expected words come from
// the prover's host fill (`fill_kat_vectors.h`, written and pinned by the
// prover's `tables::card_fill_inputs`); nothing here computes them.
//
// WHAT IT COVERS:
//   1. MEMW_R: thirteen register rows, then zero rows, every word.
//   2. CPU: 29 logs over 24 decode entries in three runs (every op class the
//      fill branches on, each entry in turn), then padding rows, every word;
//      and a log whose pc no run holds, counted as missing with its row left
//      zero.
//   3. KECCAK_RND: a table cut into its first op, holding every round of the
//      second and zero rows after it, and one cut at both ends: every row's
//      fifteen column groups, by digest; the same through the entry that reads
//      the records a table held as its keccak operations uploads.
//   Every cell of each table starts as a sentinel, so a cell no thread writes
//   is a mismatch, and threads past the table must leave a guard alone.
//
// WHAT IT DOES NOT COVER: whether nvcc accepts the file, and everything about
// execution: the grid, coalescing, the sections of a KECCAK_RND row running as
// separate blocks. That is the card test's (`prover/src/tables/card_fill_tests.rs`).
// Passing here is necessary, never sufficient.
//
// Build and run with `make test-fill-host-kat`.

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "cuda_host_shim.h"

#include "fill.cu"

#include "fill_kat_vectors.h"

namespace {

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        ++g_checks;                                          \
        if (!(cond)) {                                       \
            ++g_failures;                                    \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
        }                                                    \
    } while (0)

constexpr uint64_t UNWRITTEN = 0x5A5A5A5A5A5A5A5Aull;
constexpr uint64_t GUARD = 0xDEADBEEFCAFEF00Dull;
constexpr size_t GUARD_WORDS = 64;
// Threads launched past a table's rows.
constexpr unsigned EXTRA_THREADS = 7;

// A table of `words` cells, each UNWRITTEN, then a guard.
std::vector<uint64_t> table(size_t words) {
    std::vector<uint64_t> t(words + GUARD_WORDS, GUARD);
    for (size_t i = 0; i < words; ++i) t[i] = UNWRITTEN;
    return t;
}

bool guard_intact(const std::vector<uint64_t> &t, size_t words) {
    for (size_t i = words; i < t.size(); ++i) {
        if (t[i] != GUARD) return false;
    }
    return true;
}

// One launch of a row kernel, replayed: thread `i` of column section `y` is row
// `i`, for every row and EXTRA_THREADS more.
template <typename Launch>
void each_row(uint64_t rows, unsigned y, Launch launch) {
    for (unsigned i = 0; i < rows + EXTRA_THREADS; ++i) {
        blockIdx.x = 0;
        blockIdx.y = y;
        blockDim.x = 0;
        threadIdx.x = i;
        launch();
    }
}

// Every word of a column-major table against the expected; the first few
// mismatches named.
void check_words(const char *name, const std::vector<uint64_t> &got, const uint64_t *want,
                 uint64_t rows, uint64_t cols) {
    int named = 0;
    uint64_t bad = 0;
    for (uint64_t c = 0; c < cols; ++c) {
        for (uint64_t r = 0; r < rows; ++r) {
            uint64_t i = c * rows + r;
            if (got[i] != want[i]) {
                ++bad;
                if (named++ < 8) {
                    std::printf("  %s column %llu row %llu: got %llu, want %llu\n", name,
                                (unsigned long long)c, (unsigned long long)r,
                                (unsigned long long)got[i], (unsigned long long)want[i]);
                }
            }
        }
    }
    CHECK(bad == 0, "%s: %llu of %llu words differ", name, (unsigned long long)bad,
          (unsigned long long)(rows * cols));
    CHECK(guard_intact(got, rows * cols), "%s: a thread past the table wrote", name);
}

// FNV-1a over the words' little-endian bytes (the vectors' digest).
uint64_t fnv1a(const uint64_t *words, uint64_t n, uint64_t stride) {
    uint64_t h = 0xCBF29CE484222325ull;
    for (uint64_t k = 0; k < n; ++k) {
        uint64_t w = words[k * stride];
        for (int b = 0; b < 8; ++b) {
            h = (h ^ ((w >> (8 * b)) & 0xFF)) * 0x00000100000001B3ull;
        }
    }
    return h;
}

void test_memw_r() {
    const uint64_t height = KAT_MEMW_R_HEIGHT;
    const uint64_t cols = memw_r_cols::NUM_COLUMNS;
    auto out = table(height * cols);
    each_row(height, 0, [&] { fill_memw_r(KAT_MEMW_R_ROWS, KAT_MEMW_R_N, height, out.data()); });
    check_words("MEMW_R", out, KAT_MEMW_R_WORDS, height, cols);
}

void test_cpu() {
    const uint64_t height = KAT_CPU_HEIGHT;
    const uint64_t cols = cpu_cols::NUM_COLUMNS;
    auto out = table(height * cols);
    unsigned long long missing = 0;
    each_row(height, 0, [&] {
        fill_cpu_from_logs(KAT_CPU_LOGS, KAT_CPU_N, KAT_CPU_ENTRIES,
                           KAT_CPU_RUNS, KAT_CPU_RUNS_N, height, out.data(), &missing);
    });
    CHECK(missing == 0, "CPU: %llu logs missing their decode row", missing);
    check_words("CPU", out, KAT_CPU_WORDS, height, cols);

    // A log whose pc no run holds: counted, its row zero, every other row as before.
    std::vector<LogC> logs(KAT_CPU_LOGS, KAT_CPU_LOGS + KAT_CPU_N);
    const uint64_t absent_row = 5;
    logs[absent_row].current_pc = KAT_CPU_ABSENT_PC;
    std::vector<uint64_t> want(KAT_CPU_WORDS, KAT_CPU_WORDS + height * cols);
    for (uint64_t c = 0; c < cols; ++c) want[c * height + absent_row] = 0;
    auto refused = table(height * cols);
    missing = 0;
    each_row(height, 0, [&] {
        fill_cpu_from_logs(logs.data(), KAT_CPU_N, KAT_CPU_ENTRIES,
                           KAT_CPU_RUNS, KAT_CPU_RUNS_N, height, refused.data(), &missing);
    });
    CHECK(missing == 1, "CPU: an absent pc counted %llu times, not once", missing);
    check_words("CPU with an absent pc", refused, want.data(), height, cols);
}

// Every KECCAK_RND case through `fill(case, out)`, one of the kernel's
// entries, against the digests.
template <typename Fill>
void check_keccak_rnd(const char *name, Fill fill) {
    const uint64_t cols = keccak_cols::NUM_COLUMNS;
    CHECK(KAT_KECCAK_GROUP_END[KAT_KECCAK_GROUPS - 1] == cols, "the groups end at the last column");
    size_t digest = 0;
    for (int k = 0; k < KAT_KECCAK_CASES_N; ++k) {
        const KatKeccakCase cs = KAT_KECCAK_CASES[k];
        auto out = table(cs.rows * cols);
        for (unsigned section = 0; section < FILL_KECCAK_SECTIONS; ++section) {
            each_row(cs.rows, section, [&] { fill(cs, out.data()); });
        }
        CHECK(guard_intact(out, cs.rows * cols), "%s case %d: a thread past the table wrote", name,
              k);
        int bad = 0;
        for (uint64_t r = 0; r < cs.rows; ++r) {
            uint32_t start = 0;
            for (int g = 0; g < KAT_KECCAK_GROUPS; ++g) {
                const uint32_t end = KAT_KECCAK_GROUP_END[g];
                const uint64_t got = fnv1a(out.data() + start * cs.rows + r, end - start, cs.rows);
                if (got != KAT_KECCAK_DIGESTS[digest]) {
                    if (bad++ < 8) {
                        std::printf("  %s case %d row %llu columns [%u, %u) differ\n", name, k,
                                    (unsigned long long)r, start, end);
                    }
                }
                ++digest;
                start = end;
            }
        }
        CHECK(bad == 0, "%s case %d: %d row groups differ", name, k, bad);
    }
    CHECK(digest == sizeof(KAT_KECCAK_DIGESTS) / sizeof(KAT_KECCAK_DIGESTS[0]),
          "every digest read");
}

void test_keccak_rnd() {
    check_keccak_rnd("KECCAK_RND", [](const KatKeccakCase &cs, uint64_t *out) {
        fill_keccak_rnd(KAT_KECCAK_OPS + cs.first_op, cs.nops, cs.skip, cs.rows, out);
    });
    // The records a table held as its keccak operations uploads: the same
    // calls without their outputs, the same words.
    std::vector<KeccakOpInC> ops;
    for (const KeccakOpC &op : KAT_KECCAK_OPS) {
        KeccakOpInC in{};
        in.timestamp = op.timestamp;
        for (int i = 0; i < 25; ++i) in.input[i] = op.input[i];
        ops.push_back(in);
    }
    check_keccak_rnd("KECCAK_RND from records", [&](const KatKeccakCase &cs, uint64_t *out) {
        fill_keccak_rnd_in(ops.data() + cs.first_op, cs.nops, cs.skip, cs.rows, out);
    });
}

}  // namespace

int main() {
    test_memw_r();
    test_cpu();
    test_keccak_rnd();
    std::printf("fill host KAT: %d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
