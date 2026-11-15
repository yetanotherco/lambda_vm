// Known-answer replay of `kernels/fill.cu`'s `fill_mem_load` and `fill_mem_store`
// (the fused memory tables' card fills: an aligned CPU load or store checked in one
// row with its memory tokens), run on the host.
//
// WHY THIS EXISTS. The card test checks the kernels on a GPU; this checks their
// arithmetic and indexing without one. The vectors come from the prover (the
// ignored test `mem_card::tests::write_the_fused_fill_kernel_vectors`): tables of
// rows over every width, sign and step, at several heights (padded and whole),
// each its records as the card gets them and the words the prover's host fill
// writes. Nothing here computes an expected word.
//
// WHAT IT CHECKS, per table: every cell of the replay is the host fill's, the
// padding rows included. Every cell starts as a sentinel, so a cell no thread
// writes is a mismatch. And one mutant each (the sign bit read from byte 7 for
// every width; the step's limbs swapped) moves a word, so the check can fail.
//
// Build and run (from the repository root):
//   MEM_FILL_VECTORS=/tmp/memfill.bin cargo test -p lambda-vm-prover --lib -- --ignored --exact \
//     tables::mem_card::tests::write_the_fused_fill_kernel_vectors
//   c++ -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Icrypto/math-cuda/tests/host_kat -Icrypto/math-cuda/kernels \
//     -o target/host_kat/mem_fill_host_kat crypto/math-cuda/tests/host_kat/mem_fill_host_kat.cpp
//   target/host_kat/mem_fill_host_kat /tmp/memfill.bin

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

constexpr uint64_t SENTINEL = 0xdeadbeefcafef00dull;

template <class Row, class Kernel>
uint64_t replay(const std::vector<Row> &rows, uint64_t height, uint32_t cols, Kernel kernel,
                const std::vector<uint64_t> &want) {
    std::vector<uint64_t> out(height * cols, SENTINEL);
    blockIdx.x = 0;
    blockDim.x = 0;
    for (uint64_t r = 0; r < height; ++r) {
        threadIdx.x = (unsigned)r;
        kernel(rows.data(), rows.size(), height, out.data());
    }
    uint64_t differ = 0;
    for (size_t k = 0; k < out.size(); ++k) differ += out[k] != want[k];
    return differ;
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
    if (r.bytes.size() < 8 || std::memcmp(r.bytes.data(), "CTMEMFL1", 8) != 0) {
        std::fprintf(stderr, "not fused-fill vectors\n");
        return 2;
    }
    r.at = 8;
    const uint64_t sizes = r.u64();
    uint64_t tables = 0, rows = 0, failures = 0, sign_moved = 0, step_moved = 0;
    for (uint64_t k = 0; k < sizes; ++k) {
        const uint64_t n = r.u64();
        auto loads = r.records<MemLoadRowC>(n);
        const auto want_l = r.records<uint64_t>(r.u64());
        auto stores = r.records<MemStoreRowC>(n);
        const auto want_s = r.records<uint64_t>(r.u64());
        const uint64_t hl = want_l.size() / mem_load_cols::NUM_COLUMNS;
        const uint64_t hs = want_s.size() / mem_store_cols::NUM_COLUMNS;
        if (hl * mem_load_cols::NUM_COLUMNS != want_l.size() || hl < n ||
            hs * mem_store_cols::NUM_COLUMNS != want_s.size() || hs < n) {
            std::fprintf(stderr, "size %llu: malformed\n", (unsigned long long)n);
            return 2;
        }
        const uint64_t dl = replay(loads, hl, mem_load_cols::NUM_COLUMNS, fill_mem_load, want_l);
        const uint64_t ds = replay(stores, hs, mem_store_cols::NUM_COLUMNS, fill_mem_store, want_s);
        tables += 2;
        rows += 2 * n;
        if (dl || ds) {
            ++failures;
            std::printf("FAIL size %llu: MEM_LOAD %llu cells differ, MEM_STORE %llu\n",
                        (unsigned long long)n, (unsigned long long)dl, (unsigned long long)ds);
        }
        // Mutant 1: a narrow load whose top byte differs from byte 7, read as
        // width 8 (the sign bit from byte 7).
        for (auto &row : loads) {
            if (row.width < 8 && (row.res[row.width - 1] >> 7) != (row.res[7] >> 7)) {
                MemLoadRowC m = row;
                m.width = 8;
                std::vector<MemLoadRowC> one = loads;
                one[&row - loads.data()] = m;
                sign_moved += replay(one, hl, mem_load_cols::NUM_COLUMNS, fill_mem_load, want_l) > 0;
                break;
            }
        }
        // Mutant 2: a store whose step's limbs differ, swapped.
        for (auto &row : stores) {
            if ((row.step & 0xFFFF) != (row.step >> 16)) {
                std::vector<MemStoreRowC> one = stores;
                auto &m = one[&row - stores.data()];
                m.step = (m.step << 16) | (m.step >> 16);
                step_moved += replay(one, hs, mem_store_cols::NUM_COLUMNS, fill_mem_store, want_s) > 0;
                break;
            }
        }
    }
    if (r.at != r.bytes.size()) {
        std::fprintf(stderr, "%zu bytes left over\n", r.bytes.size() - r.at);
        return 2;
    }
    const bool bites = sign_moved > 0 && step_moved > 0;
    std::printf("MEM FILL KAT: %llu tables, %llu rows, %llu failed · mutants %s (sign %llu, step %llu)\n",
                (unsigned long long)tables, (unsigned long long)rows, (unsigned long long)failures,
                bites ? "all bite" : "NOT all bite", (unsigned long long)sign_moved,
                (unsigned long long)step_moved);
    return failures == 0 && bites && tables >= 4 ? 0 : 1;
}
