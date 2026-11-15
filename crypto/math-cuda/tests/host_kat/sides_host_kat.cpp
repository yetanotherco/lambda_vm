// Known-answer replay of `kernels/fill.cu`'s `derive_cpu_sides` (the in-row
// register format's side records derived on the card from the executor's
// history) followed by `fill_cpu_from_logs` reading what it derived, run on the
// host.
//
// WHY THIS EXISTS. The card test checks the kernels on a GPU; this checks their
// arithmetic and indexing without one, before a GPU run is spent on it. The
// vectors come from the prover (the ignored test
// `history_sides_tests::write_the_side_records_kernel_vectors`): record-mode
// walks of asm programs, Rust guests and an ethrex block, each whole CPU
// chunk's logs, history records (`HistExtra`) and exceptions as the card gets
// them, the program's DECODE rows, and the words the prover's host fill writes
// for the chunk's CPU table with the WALK's side records. Nothing here computes
// an expected word.
//
// WHAT IT CHECKS, per chunk:
//   1. the two kernels write the host fill's words, every cell of the 42-column
//      table, and count nothing into the check;
//   2. each of the derivation's four mutants (rs2's stamp read as rs1's; the
//      keep-alive rule dropped; rd's old value read whatever the slot; the
//      exceptions ignored) changes a word or counts into the check on at
//      least one chunk.
// Every cell starts as a sentinel, so a cell no thread writes is a mismatch.
//
// WHAT IT DOES NOT: nvcc, the grid, timing, the upload's staging (the card
// test's, `card_records_card_tests`).
//
// Build and run (from the repository root):
//   SIDE_RECORDS_VECTORS=/tmp/sides.bin cargo test -p lambda-vm-prover --lib -- --ignored --exact \
//     tables::trace_builder::windowed::history_sides_tests::write_the_side_records_kernel_vectors
//   c++ -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Icrypto/math-cuda/tests/host_kat -Icrypto/math-cuda/kernels \
//     -o target/host_kat/sides_host_kat crypto/math-cuda/tests/host_kat/sides_host_kat.cpp
//   target/host_kat/sides_host_kat /tmp/sides.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda_host_shim.h"

#include "fill.cu"

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

struct Chunk {
    uint64_t first;
    std::vector<LogC> logs;
    std::vector<RegisterStampsC> stamps;
    std::vector<SideExceptionC> ex;
    std::vector<uint64_t> want;  // column-major, n × 42
};

constexpr uint64_t SENTINEL = 0xdeadbeefcafef00dull;

// One chunk replayed, one thread a row for each kernel. Returns (cells
// differing, check).
std::pair<uint64_t, unsigned long long> replay(const std::vector<DecodeEntryC> &dec,
                                               const std::vector<DecodeRunC> &runs,
                                               const Chunk &c, uint32_t fault) {
    const uint64_t n = c.logs.size();
    std::vector<CpuSideC> sides(n, CpuSideC{{0xAAAAAAAAu, 0xAAAAAAAAu, 0xAAAAAAAAu},
                                            {0xAAAAAAAAu, 0xAAAAAAAAu}});
    unsigned long long check = 0;
    blockIdx.x = 0;
    blockDim.x = 0;
    for (uint64_t i = 0; i < n; ++i) {
        threadIdx.x = (unsigned)i;
        derive_cpu_sides(c.logs.data(), n, c.first, c.stamps.data(), c.ex.data(),
                         (uint32_t)c.ex.size(), dec.data(), runs.data(), (uint32_t)runs.size(),
                         sides.data(), &check, fault);
    }
    std::vector<uint64_t> out(n * cpu_cols::NUM_COLUMNS_IN_ROW, SENTINEL);
    for (uint64_t i = 0; i < n; ++i) {
        threadIdx.x = (unsigned)i;
        fill_cpu_from_logs(c.logs.data(), n, dec.data(), runs.data(), (uint32_t)runs.size(), n,
                           out.data(), &check, sides.data());
    }
    uint64_t differ = 0;
    for (size_t k = 0; k < out.size(); ++k) differ += out[k] != c.want[k];
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
    if (r.bytes.size() < 8 || std::memcmp(r.bytes.data(), "CTSIDE01", 8) != 0) {
        std::fprintf(stderr, "not side-record vectors\n");
        return 2;
    }
    r.at = 8;
    const uint64_t cases = r.u64();
    uint64_t chunks = 0, rows = 0, failures = 0, listed = 0;
    const uint32_t faults[4] = {SIDES_FAULT_SLOT, SIDES_FAULT_KEEP_ALIVE, SIDES_FAULT_VALUE,
                                SIDES_FAULT_EXCEPTIONS};
    const char *names[4] = {"rs2 read from rs1's stamp", "no keep-alive", "old value whatever its slot",
                            "exceptions ignored"};
    uint64_t moved[4] = {0, 0, 0, 0};
    for (uint64_t k = 0; k < cases; ++k) {
        const auto dec = r.records<DecodeEntryC>(r.u64());
        const auto runs = r.records<DecodeRunC>(r.u64());
        const uint64_t nchunks = r.u64();
        for (uint64_t j = 0; j < nchunks; ++j) {
            Chunk c;
            c.first = r.u64();
            const uint64_t n = r.u64();
            c.logs = r.records<LogC>(n);
            c.stamps = r.records<RegisterStampsC>(n);
            c.ex = r.records<SideExceptionC>(r.u64());
            const uint64_t cells = r.u64();
            c.want = r.records<uint64_t>(cells);
            if (cells != n * cpu_cols::NUM_COLUMNS_IN_ROW) {
                std::fprintf(stderr, "case %llu chunk %llu: malformed\n", (unsigned long long)k,
                             (unsigned long long)j);
                return 2;
            }
            ++chunks;
            rows += n;
            listed += c.ex.size();
            const auto clean = replay(dec, runs, c, 0);
            if (clean.first != 0 || clean.second != 0) {
                ++failures;
                std::printf("FAIL case %llu chunk %llu (first %llu): %llu cells differ, check %llu\n",
                            (unsigned long long)k, (unsigned long long)j,
                            (unsigned long long)c.first, (unsigned long long)clean.first,
                            clean.second);
            }
            for (int f = 0; f < 4; ++f) {
                const auto m = replay(dec, runs, c, faults[f]);
                moved[f] += m.first != 0 || m.second != 0;
            }
        }
    }
    if (r.at != r.bytes.size()) {
        std::fprintf(stderr, "%zu bytes left over\n", r.bytes.size() - r.at);
        return 2;
    }
    bool bites = true;
    for (int f = 0; f < 4; ++f) {
        std::printf("MUTANT %s: moved %llu of %llu chunks\n", names[f],
                    (unsigned long long)moved[f], (unsigned long long)chunks);
        bites &= moved[f] > 0;
    }
    std::printf("SIDES KAT: %llu chunks, %llu rows, %llu exceptions, %llu failed · mutants %s\n",
                (unsigned long long)chunks, (unsigned long long)rows, (unsigned long long)listed,
                (unsigned long long)failures, bites ? "all bite" : "NOT all bite");
    return failures == 0 && bites && chunks >= 2 ? 0 : 1;
}
