// Known-answer tests for the commit-time out-of-domain answers' column values
// (`kernels/sumcheck.cu`: `eq_dot_base_ext3_many`, then `sum_partials_ext3`
// over its partials, as `math_cuda::sumcheck::eq_dot_many_base` launches
// them), run on the host through `cuda_host_simt_shim.h` — every thread of a
// block a real thread, so the block reductions run as written. No GPU, no
// nvcc.
//
// WHAT IT COVERS, for every shape (the columns' variables, how many low ones
// the `lo` half takes, how many columns, the blocks a column gets — fewer
// than its hi rows, as many, more — and the block's threads, more than a hi
// row's lo slots and fewer):
//   each column's value against a written-out reference: the sum over its
//   rows of the value times `eq(p, row)` as a product over the row's bits,
//   the first variable the high bit (canonical values). The columns' words
//   include non-canonical representatives, as a card may hold them.
//
// WHAT IT DOES NOT COVER, and the card tests still own: whether nvcc accepts
// the file, the hardware's memory and timing, the driver's launch
// (`multilinear/tests/whir_commit_ood_device.rs` on a box).
//
// Build and run with `make test-eq-dot-host-kat`.

#include <cstdio>
#include <string>
#include <vector>

#include "cuda_host_simt_shim.h"

// The kernels' block-shared buffer: blocks run one at a time here, so one
// buffer serves every block. `extern "C"` because the kernels that name it
// are. The launch bounds mean nothing on the host.
#define __shared__
#define __launch_bounds__(...)
extern "C" {
uint64_t shared[3 * 1024];
}

#include "sumcheck.cu"

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const uint64_t P = 0xFFFFFFFF00000001ull;

uint64_t state = 0x13198A2E03707344ull;
uint64_t next_u64() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// A raw limb: canonical most of the time, and every eighth one a
// non-canonical representative in [p, 2^64).
uint64_t raw_limb() {
    uint64_t v = next_u64() % P;
    if ((next_u64() & 7) == 0 && v < 0xFFFFFFFFull) v += P;
    return v;
}

Fe3 random_ext() { return ext3::make(raw_limb(), raw_limb(), raw_limb()); }

bool same_value(const Fe3 &a, const Fe3 &b) {
    Fe3 x = ext3::canonical(a), y = ext3::canonical(b);
    return x.a == y.a && x.b == y.b && x.c == y.c;
}

// `eq(r, x)` for every `x` of `r.size()` bits, the first variable the high
// bit (`multilinear::eq::eq_evals`).
std::vector<uint64_t> eq_table(const std::vector<Fe3> &r) {
    size_t n = r.size();
    std::vector<Fe3> t(1, ext3::make(1, 0, 0));
    for (size_t j = 0; j < n; ++j) {
        std::vector<Fe3> next(t.size() * 2);
        Fe3 one_minus = ext3::sub(ext3::make(1, 0, 0), r[j]);
        for (size_t i = 0; i < t.size(); ++i) {
            next[2 * i] = ext3::mul(t[i], one_minus);
            next[2 * i + 1] = ext3::mul(t[i], r[j]);
        }
        t.swap(next);
    }
    std::vector<uint64_t> raw;
    for (const Fe3 &v : t) {
        raw.push_back(v.a);
        raw.push_back(v.b);
        raw.push_back(v.c);
    }
    return raw;
}

struct Shape {
    unsigned vars;
    unsigned lo_bits;
    unsigned columns;
    unsigned blocks;
    unsigned threads;
};

void run(const Shape &s) {
    uint64_t rows = 1ull << s.vars;
    std::vector<uint64_t> columns(rows * s.columns);
    for (auto &v : columns) v = raw_limb();
    std::vector<Fe3> point(s.vars);
    for (auto &c : point) c = random_ext();
    std::vector<Fe3> high(point.begin(), point.end() - s.lo_bits);
    std::vector<Fe3> low(point.end() - s.lo_bits, point.end());
    std::vector<uint64_t> hi = eq_table(high), lo = eq_table(low);

    std::vector<uint64_t> partials(3 * (uint64_t)s.columns * s.blocks, 0);
    for (unsigned y = 0; y < s.columns; ++y) {
        simt_launch(s.blocks, s.threads, [&] {
            blockIdx.y = y;
            eq_dot_base_ext3_many(columns.data(), rows, s.lo_bits, hi.data(), lo.data(),
                                  partials.data());
        });
    }
    std::vector<uint64_t> sums(3 * s.columns, 0);
    simt_launch(s.columns, 256, [&] {
        sum_partials_ext3(partials.data(), s.blocks, sums.data());
    });

    for (unsigned k = 0; k < s.columns; ++k) {
        Fe3 want = ext3::zero();
        for (uint64_t row = 0; row < rows; ++row) {
            Fe3 eq = ext3::make(1, 0, 0);
            for (unsigned j = 0; j < s.vars; ++j) {
                bool bit = (row >> (s.vars - 1 - j)) & 1;
                eq = ext3::mul(eq, bit ? point[j] : ext3::sub(ext3::make(1, 0, 0), point[j]));
            }
            want = ext3::add(want, ext3::mul_base(eq, columns[k * rows + row]));
        }
        Fe3 got = ext3::make(sums[k * 3], sums[k * 3 + 1], sums[k * 3 + 2]);
        check(same_value(got, want),
              "vars " + std::to_string(s.vars) + " lo " + std::to_string(s.lo_bits) +
                  " columns " + std::to_string(s.columns) + " blocks " +
                  std::to_string(s.blocks) + " threads " + std::to_string(s.threads) +
                  ": column " + std::to_string(k));
    }
}

}  // namespace

int main() {
    std::vector<Shape> shapes = {
        // One hi row, lo slots fewer than the threads.
        {1, 1, 1, 1, 32},
        {3, 3, 2, 1, 32},
        {5, 5, 3, 1, 64},
        // One hi row of 2^10 lo slots, four a thread.
        {10, 10, 3, 1, 256},
        // Several hi rows: blocks fewer than them (each strides), as many, and
        // more (the extra blocks write zero partials).
        {12, 10, 4, 2, 256},
        {12, 10, 2, 4, 256},
        {12, 10, 2, 8, 256},
        {13, 10, 2, 3, 128},
        // A short low half, as a narrow point takes it.
        {8, 4, 5, 7, 32},
        {11, 3, 2, 64, 64},
    };
    for (const Shape &s : shapes) run(s);
    if (failures) {
        printf("eq_dot host KAT: %d FAILED\n", failures);
        return 1;
    }
    printf("eq_dot host KAT: %zu shapes, every column's value the written-out sum\n",
           shapes.size());
    return 0;
}
