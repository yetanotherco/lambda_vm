// Known-answer replay of `kernels/fill.cu`'s seven word-family fills (`fill_mem_load8`,
// `fill_mem_store8`, `fill_mem_loadw`, `fill_mem_storew`, `fill_mem_load2w`,
// `fill_mem_store2w`, `fill_memw_w`: the word memory format's tables filled from the
// walk's rows), run on the host.
//
// WHY THIS EXISTS. The card test checks the kernels on a GPU; this checks their
// arithmetic and indexing without one. The vectors come from the prover (the ignored
// test `word_card::tests::write_the_word_fill_kernel_vectors`): for each family, tables
// of rows over every width, sign, offset, message kind and a spread of steps, at
// several heights (padded and whole), each its rows as the card gets them and the words
// the prover's host fill writes. Nothing here computes an expected word.
//
// WHAT IT CHECKS, per table: every cell of the replay is the host fill's, the padding
// rows included. Every cell starts as a sentinel, so a cell no thread writes is a
// mismatch. And one mutant a family moves a word, so each check can fail: MEM_LOAD8 its
// step's limbs swapped; MEM_STORE8 its old word's halves swapped; MEM_LOADW a narrow
// load read one byte further (its offset + 1); MEM_STOREW its width doubled;
// MEM_LOAD2W its sign flag flipped on a row whose sign bit is set; MEM_STORE2W its
// second old word's low byte changed; MEMW_W its message kind flipped.
//
// Build and run (from the repository root):
//   WORD_FILL_VECTORS=/tmp/wordfill.bin cargo test -p lambda-vm-prover --lib -- --ignored --exact \
//     tables::word_card::tests::write_the_word_fill_kernel_vectors
//   c++ -std=c++17 -O2 -Wall -Wno-unknown-pragmas -Icrypto/math-cuda/tests/host_kat -Icrypto/math-cuda/kernels \
//     -o target/host_kat/word_fill_host_kat crypto/math-cuda/tests/host_kat/word_fill_host_kat.cpp
//   target/host_kat/word_fill_host_kat /tmp/wordfill.bin

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

struct Totals {
    uint64_t tables = 0, rows = 0, failures = 0;
};

// One family's tables at every size: the replay against the host fill, then the
// family's mutant on the first row it applies to (`mutate` returns whether it did).
template <class Row, class Kernel, class Mutate>
uint64_t family(Reader &r, uint64_t sizes, uint32_t cols, Kernel kernel, const char *name, Mutate mutate,
                Totals &t) {
    uint64_t moved = 0;
    for (uint64_t k = 0; k < sizes; ++k) {
        const uint64_t n = r.u64();
        auto rows = r.records<Row>(n);
        const auto want = r.records<uint64_t>(r.u64());
        const uint64_t height = want.size() / cols;
        if (height * cols != want.size() || height < n) {
            std::fprintf(stderr, "%s size %llu: malformed\n", name, (unsigned long long)n);
            std::exit(2);
        }
        const uint64_t d = replay(rows, height, cols, kernel, want);
        t.tables += 1;
        t.rows += n;
        if (d) {
            ++t.failures;
            std::printf("FAIL %s size %llu: %llu cells differ\n", name, (unsigned long long)n,
                        (unsigned long long)d);
        }
        for (size_t i = 0; i < rows.size(); ++i) {
            std::vector<Row> one = rows;
            if (mutate(one[i])) {
                moved += replay(one, height, cols, kernel, want) > 0;
                break;
            }
        }
    }
    return moved;
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
    if (r.bytes.size() < 8 || std::memcmp(r.bytes.data(), "CTWRDFL1", 8) != 0) {
        std::fprintf(stderr, "not word-fill vectors\n");
        return 2;
    }
    r.at = 8;
    const uint64_t sizes = r.u64();
    Totals t;
    uint64_t moved[7] = {};
    moved[0] = family<Load8RowC>(
        r, sizes, mem_load8_cols::NUM_COLUMNS, fill_mem_load8, "MEM_LOAD8",
        [](Load8RowC &m) {
            if ((m.step & 0xFFFF) == (m.step >> 16)) return false;
            m.step = (m.step << 16) | (m.step >> 16);
            return true;
        },
        t);
    moved[1] = family<Store8RowC>(
        r, sizes, mem_store8_cols::NUM_COLUMNS, fill_mem_store8, "MEM_STORE8",
        [](Store8RowC &m) {
            if ((m.old & 0xFFFFFFFFull) == (m.old >> 32)) return false;
            m.old = (m.old << 32) | (m.old >> 32);
            return true;
        },
        t);
    moved[2] = family<LoadWRowC>(
        r, sizes, mem_loadw_cols::NUM_COLUMNS, fill_mem_loadw, "MEM_LOADW",
        [](LoadWRowC &m) {
            const uint32_t off = (uint32_t)(m.address & 7);
            if (off + m.width >= 8) return false;
            const uint8_t a = (m.word >> (8 * off)) & 0xFF, b = (m.word >> (8 * (off + m.width))) & 0xFF;
            if (a == b) return false;
            m.address += 1;
            return true;
        },
        t);
    moved[3] = family<StoreWRowC>(
        r, sizes, mem_storew_cols::NUM_COLUMNS, fill_mem_storew, "MEM_STOREW",
        [](StoreWRowC &m) {
            if (m.width != 1 && m.width != 2) return false;
            m.width *= 2;
            return true;
        },
        t);
    moved[4] = family<Load2WRowC>(
        r, sizes, mem_load2w_cols::NUM_COLUMNS, fill_mem_load2w, "MEM_LOAD2W",
        [](Load2WRowC &m) {
            if (m.width == 8) return false;
            const uint32_t top = (uint32_t)(m.address & 7) + m.width - 1;
            const uint64_t word = top < 8 ? m.words[0] : m.words[1];
            if (!((word >> (8 * (top & 7) + 7)) & 1)) return false;
            m.is_signed = m.is_signed ? 0 : 1;
            return true;
        },
        t);
    moved[5] = family<Store2WRowC>(
        r, sizes, mem_store2w_cols::NUM_COLUMNS, fill_mem_store2w, "MEM_STORE2W",
        [](Store2WRowC &m) {
            m.old[1] ^= 0x01;
            return true;
        },
        t);
    moved[6] = family<MemwWordRowC>(
        r, sizes, memw_w_cols::NUM_COLUMNS, fill_memw_w, "MEMW_W",
        [](MemwWordRowC &m) {
            m.kind = m.kind ? 0 : 1;
            return true;
        },
        t);
    if (r.at != r.bytes.size()) {
        std::fprintf(stderr, "%zu bytes left over\n", r.bytes.size() - r.at);
        return 2;
    }
    bool bites = true;
    for (uint64_t m : moved) bites = bites && m > 0;
    std::printf("WORD FILL KAT: %llu tables, %llu rows, %llu failed · mutants %s (%llu %llu %llu %llu %llu %llu %llu)\n",
                (unsigned long long)t.tables, (unsigned long long)t.rows, (unsigned long long)t.failures,
                bites ? "all bite" : "NOT all bite", (unsigned long long)moved[0], (unsigned long long)moved[1],
                (unsigned long long)moved[2], (unsigned long long)moved[3], (unsigned long long)moved[4],
                (unsigned long long)moved[5], (unsigned long long)moved[6]);
    return t.failures == 0 && bites && t.tables >= 7 * 4 ? 0 : 1;
}
