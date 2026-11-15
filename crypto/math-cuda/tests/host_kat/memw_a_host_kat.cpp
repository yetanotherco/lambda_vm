// Known-answer replay of `kernels/card_rows.cu`'s `derive_memw_a_from_cache`
// (MEMW_A's rows derived on the card from the CPU logs it holds and the
// executor's history), run on the host.
//
// WHY THIS EXISTS. The card test checks the kernel on a GPU; this checks its
// arithmetic and indexing without one, before a GPU run is spent on it. The
// vectors come from the prover (the ignored test
// `history_rows_tests::write_the_memw_a_kernel_vectors`): record-mode walks
// of asm programs, Rust guests and an ethrex block, each whole MEMW_A chunk
// as the card gets it (its parts: refs, the window's accesses from a base,
// its mixed list, its whole rows from a base), the walked windows' logs as
// the card's cache holds them, the program's DECODE rows, and the words the
// prover's host fill writes for the chunk's MEMW_A table with the WALK's rows.
// Nothing here computes an expected word.
//
// WHAT IT CHECKS, per chunk:
//   1. the replay writes the host fill's words, every cell, and counts nothing
//      into the check;
//   2. three edits of the chunk's records: two are refused (counted into the
//      check): a regular row of two bytes or more given a mixed entry whose
//      second byte's timestamp differs; a regular row's access stamped at the
//      row's own timestamp (no step for the limbs); one is not: a narrow
//      load's log with its written value's top bit flipped (past the width)
//      makes the same words, the fill;
//   3. each of the kernel's four mutants bites on some chunk: a store's old
//      read as its value moves a word of the clean replay; the sign fill
//      dropped moves a word under the third edit; the mixed list read as
//      uniform and the limbs' rule dropped let their edit through (nothing
//      counted). (A real log's written value is already filled, so the fill
//      moves no word of a clean replay.)
// Every cell starts as a sentinel, so a cell no thread writes is a mismatch.
//
// WHAT IT DOES NOT: nvcc, the grid, timing, the upload's plan (which tables
// derive, the cache's coverage). That is the card test's.
//
// Build and run (from the repository root):
//   MEMW_A_VECTORS=/tmp/memwa.bin cargo test -p lambda-vm-prover --lib -- --ignored --exact \
//     tables::trace_builder::windowed::history_rows_tests::write_the_memw_a_kernel_vectors
//   c++ -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Icrypto/math-cuda/tests/host_kat -Icrypto/math-cuda/kernels \
//     -o target/host_kat/memw_a_host_kat crypto/math-cuda/tests/host_kat/memw_a_host_kat.cpp
//   target/host_kat/memw_a_host_kat /tmp/memwa.bin

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

struct Part {
    std::vector<AlignedRefC> refs;
    std::vector<MemoryAccessC> accesses;
    uint64_t access_base = 0;
    std::vector<uint32_t> mixed;
    std::vector<AlignedRowC> whole;
    uint64_t whole_base = 0;
};

struct Chunk {
    uint64_t limbs = 0;
    std::vector<Part> parts;
    std::vector<uint64_t> want;  // column-major, n × 29
    uint64_t rows() const {
        uint64_t n = 0;
        for (const auto &p : parts) n += p.refs.size();
        return n;
    }
};

constexpr uint64_t SENTINEL = 0xdeadbeefcafef00dull;

// One chunk replayed, one thread a row. Returns (cells differing, check).
std::pair<uint64_t, unsigned long long> replay(const std::vector<DecodeEntryC> &dec,
                                               const std::vector<DecodeRunC> &runs,
                                               const std::vector<CachedLogsC> &entries,
                                               const Chunk &c, uint32_t fault) {
    const uint64_t n = c.rows();
    std::vector<uint64_t> out(n * memw_a_cols::NUM_COLUMNS, SENTINEL);
    std::vector<AlignedPartC> descs;
    uint64_t row0 = 0;
    for (const auto &p : c.parts) {
        descs.push_back(AlignedPartC{p.refs.data(), p.refs.size(), row0, row0, n, out.data(),
                                     p.accesses.data(), p.access_base, p.accesses.size(),
                                     p.mixed.data(), p.mixed.size() / 8, p.whole.data(),
                                     p.whole_base, p.whole.size(), c.limbs});
        row0 += p.refs.size();
    }
    unsigned long long check = 0;
    blockIdx.x = 0;
    blockDim.x = 0;
    for (uint64_t g = 0; g < n; ++g) {
        threadIdx.x = (unsigned)g;
        derive_memw_a_from_cache(entries.data(), (uint32_t)entries.size(), descs.data(),
                                 (uint32_t)descs.size(), n, dec.data(), runs.data(),
                                 (uint32_t)runs.size(), fault, &check);
    }
    uint64_t differ = 0;
    for (size_t k = 0; k < out.size(); ++k) differ += out[k] != c.want[k];
    return {differ, check};
}

// The first regular ref of `c` that `pick` admits, as (part, index).
template <class F>
bool find_regular(const Chunk &c, F pick, size_t &part, size_t &index) {
    for (size_t p = 0; p < c.parts.size(); ++p) {
        for (size_t j = 0; j < c.parts[p].refs.size(); ++j) {
            const AlignedRefC r = c.parts[p].refs[j];
            if (r.cycle & ALIGNED_ECALL) continue;
            if (pick(c.parts[p], r, j)) {
                part = p;
                index = j;
                return true;
            }
        }
    }
    return false;
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
    if (r.bytes.size() < 8 || std::memcmp(r.bytes.data(), "CTMEMWA1", 8) != 0) {
        std::fprintf(stderr, "not MEMW_A vectors\n");
        return 2;
    }
    r.at = 8;
    const uint64_t cases = r.u64();
    uint64_t chunks = 0, rows = 0, failures = 0, edits_refused = 0, edits_tried = 0;
    const char *names[4] = {"mixed list read as uniform", "sign fill dropped",
                            "store's old read as its value", "limbs' step rule dropped"};
    uint64_t bit[4] = {0, 0, 0, 0};
    for (uint64_t k = 0; k < cases; ++k) {
        const auto dec = r.records<DecodeEntryC>(r.u64());
        const auto runs = r.records<DecodeRunC>(r.u64());
        const uint64_t nwindows = r.u64();
        std::vector<std::vector<LogC>> logs(nwindows);
        std::vector<CachedLogsC> entries;
        for (uint64_t w = 0; w < nwindows; ++w) {
            const uint64_t first = r.u64();
            const uint64_t n = r.u64();
            logs[w] = r.records<LogC>(n);
            entries.push_back(CachedLogsC{first, n, logs[w].data()});
        }
        const uint64_t nchunks = r.u64();
        for (uint64_t j = 0; j < nchunks; ++j) {
            Chunk c;
            c.limbs = r.u64();
            const uint64_t nparts = r.u64();
            for (uint64_t p = 0; p < nparts; ++p) {
                Part part;
                part.refs = r.records<AlignedRefC>(r.u64());
                const uint64_t naccesses = r.u64();
                part.access_base = r.u64();
                part.accesses = r.records<MemoryAccessC>(naccesses);
                part.mixed = r.records<uint32_t>(r.u64());
                const uint64_t nwhole = r.u64();
                part.whole_base = r.u64();
                part.whole = r.records<AlignedRowC>(nwhole);
                c.parts.push_back(std::move(part));
            }
            const uint64_t cells = r.u64();
            c.want = r.records<uint64_t>(cells);
            if (cells != c.rows() * memw_a_cols::NUM_COLUMNS) {
                std::fprintf(stderr, "case %llu chunk %llu: malformed\n", (unsigned long long)k,
                             (unsigned long long)j);
                return 2;
            }
            ++chunks;
            rows += c.rows();
            const auto clean = replay(dec, runs, entries, c, 0);
            if (clean.first != 0 || clean.second != 0) {
                ++failures;
                std::printf("FAIL case %llu chunk %llu: %llu cells differ, check %llu\n",
                            (unsigned long long)k, (unsigned long long)j,
                            (unsigned long long)clean.first, clean.second);
            }
            // Edit 3: a load narrower than eight bytes whose log's written
            // value has its top bit flipped (past the load's width): the fill
            // makes the same row, the mutant does not.
            {
                const uint64_t n = c.rows();
                uint64_t g = 0;
                bool done = false;
                for (size_t q = 0; q < c.parts.size() && !done; ++q) {
                    for (size_t t = 0; t < c.parts[q].refs.size() && !done; ++t, ++g) {
                        const AlignedRefC ref = c.parts[q].refs[t];
                        if ((ref.cycle & ALIGNED_ECALL) || c.want[memw_a_cols::MU_READ * n + g] != 1 ||
                            c.want[memw_a_cols::WRITE8 * n + g] != 0) {
                            continue;
                        }
                        for (size_t w = 0; w < entries.size(); ++w) {
                            const CachedLogsC en = entries[w];
                            if (ref.cycle < en.first || ref.cycle - en.first >= en.n) continue;
                            std::vector<LogC> edited(en.logs, en.logs + en.n);
                            edited[ref.cycle - en.first].dst_val ^= 1ull << 63;
                            std::vector<CachedLogsC> moved = entries;
                            moved[w].logs = edited.data();
                            ++edits_tried;
                            const auto same = replay(dec, runs, moved, c, 0);
                            const auto sign = replay(dec, runs, moved, c, MEMW_A_FAULT_SIGN_FILL);
                            // Counted as refused when the clean replay keeps the row.
                            edits_refused += same.first == 0 && same.second == 0;
                            bit[1] += same.first == 0 && same.second == 0 && sign.first != 0;
                            done = true;
                            break;
                        }
                    }
                }
            }
            const auto store = replay(dec, runs, entries, c, MEMW_A_FAULT_STORE_OLD);
            bit[2] += store.first != 0 || store.second != 0;
            size_t p = 0, i = 0;
            // Edit 1: a mixed entry whose second byte differs, on a row of two
            // bytes or more (WRITE2, WRITE4 or WRITE8 set in the walk's words)
            // whose access shares one timestamp.
            uint64_t g = 0;
            bool found = false;
            for (size_t q = 0; q < c.parts.size() && !found; ++q) {
                for (size_t t = 0; t < c.parts[q].refs.size() && !found; ++t, ++g) {
                    const AlignedRefC ref = c.parts[q].refs[t];
                    if (ref.cycle & ALIGNED_ECALL) continue;
                    const auto &part = c.parts[q];
                    if (part.accesses[ref.at - part.access_base].mixed != NO_MIXED) continue;
                    const uint64_t n = c.rows();
                    if (c.want[memw_a_cols::WRITE2 * n + g] | c.want[memw_a_cols::WRITE4 * n + g] |
                        c.want[memw_a_cols::WRITE8 * n + g]) {
                        found = true;
                        p = q;
                        i = t;
                    }
                }
            }
            if (found) {
                Chunk e = c;
                Part &part = e.parts[p];
                MemoryAccessC &acc = part.accesses[part.refs[i].at - part.access_base];
                for (uint32_t b = 0; b < 8; ++b) part.mixed.push_back(acc.ts);
                part.mixed[part.mixed.size() - 7] = acc.ts - 1;
                acc.mixed = (uint32_t)(part.mixed.size() / 8 - 1);
                ++edits_tried;
                const auto refused = replay(dec, runs, entries, e, 0);
                edits_refused += refused.second != 0;
                const auto through = replay(dec, runs, entries, e, MEMW_A_FAULT_MIXED);
                bit[0] += refused.second != 0 && through.second == 0;
            }
            // Edit 2: an access stamped at its row's own timestamp.
            if (c.limbs &&
                find_regular(
                    c,
                    [&](const Part &part, AlignedRefC ref, size_t) {
                        return part.accesses[ref.at - part.access_base].mixed == NO_MIXED;
                    },
                    p, i)) {
                Chunk e = c;
                Part &part = e.parts[p];
                MemoryAccessC &acc = part.accesses[part.refs[i].at - part.access_base];
                acc.ts = (uint32_t)(4ull * part.refs[i].cycle + 4);
                ++edits_tried;
                const auto refused = replay(dec, runs, entries, e, 0);
                edits_refused += refused.second != 0;
                const auto through = replay(dec, runs, entries, e, MEMW_A_FAULT_LIMBS);
                bit[3] += refused.second != 0 && through.second == 0;
            }
        }
    }
    if (r.at != r.bytes.size()) {
        std::fprintf(stderr, "%zu bytes left over\n", r.bytes.size() - r.at);
        return 2;
    }
    bool bites = true;
    for (int f = 0; f < 4; ++f) {
        std::printf("MUTANT %s: bit on %llu of %llu chunks\n", names[f],
                    (unsigned long long)bit[f], (unsigned long long)chunks);
        bites &= bit[f] > 0;
    }
    std::printf(
        "MEMW_A KAT: %llu chunks, %llu rows, %llu failed · edits as expected %llu of %llu · mutants "
        "%s\n",
        (unsigned long long)chunks, (unsigned long long)rows, (unsigned long long)failures,
        (unsigned long long)edits_refused, (unsigned long long)edits_tried,
        bites ? "all bite" : "NOT all bite");
    return failures == 0 && bites && edits_refused == edits_tried && chunks >= 2 ? 0 : 1;
}
