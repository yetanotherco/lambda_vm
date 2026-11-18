// Known-answer tests for the bench's double-step GKR round kernels (`kernels/gkr4_bench.cu`: the
// value form `gkr4_round_gruen` and `gkr4_round_gruen_lb2`, the one-thread coefficient form
// `gkr2_round_v2{,_h0}`, the thread-pair form `gkr2_round_v3{,_h0}` and `gkr2_round_v3b{,_h0}`),
// run on the host through `cuda_host_simt_shim.h`: every thread of a block a real thread, so the
// pair kernel's exchange and the warp-shuffle block reduction run as written. No GPU, no nvcc.
//
// WHAT IT COVERS, for every kernel, with and without `H(0)`, over layers of 2^10 and 2^12 cells
// (quarters of 2^8 and 2^10), every round down to a cube of 8, at the driver's launch shape and at a
// grid of two blocks (so a lane walks several cube indices):
//   1. A WRITTEN-OUT REFERENCE: the rows `H(1)..H(4)` (and `H(0)`) summed here, in this file's own
//      canonical arithmetic, from the four quarters composed as the two folds compose them, each
//      quarter folded here by the same challenges: EQUAL to the kernel's block partials summed
//      (canonical values), every round.
//   2. The folded quarters the kernel writes back: EQUAL to the reference's, every round.
//   3. Faults that must part: the last quarter's lower cell read one off in a round without a fold
//      (every kernel; `H(1)` reads no lower cell and must keep its value, `H(0)` must move), and
//      the pair kernels trading with lane ^ 2 instead of the partner.
//
// WHAT IT DOES NOT COVER, and the card gate (`gkr4_bench.rs`) owns: nvcc acceptance, registers,
// spills, the hardware's warps and memory, timing.
//
// Build and run with `make test-gkr2-bench-host-kat`.

#include <array>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_host_simt_shim.h"

#define __shared__
#define __launch_bounds__(...)
extern "C" {
uint64_t shared[3 * 1024 + 8];
}

#include "gkr4_bench.cu"

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const std::string &what) {
    ++checks;
    if (!ok) {
        printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

// ── this file's own arithmetic: canonical Goldilocks and its cubic extension w³ = 2 ──────────
typedef unsigned __int128 u128;
const uint64_t P = 0xFFFFFFFF00000001ull;
uint64_t fc(uint64_t a) { return a % P; }
uint64_t fadd(uint64_t a, uint64_t b) { return (uint64_t)(((u128)a + b) % P); }
uint64_t fsub(uint64_t a, uint64_t b) { return a >= b ? a - b : (uint64_t)((u128)a + P - b); }
uint64_t fmul(uint64_t a, uint64_t b) { return (uint64_t)((u128)a * b % P); }

struct E {
    uint64_t c[3];
    bool operator==(const E &o) const { return c[0] == o.c[0] && c[1] == o.c[1] && c[2] == o.c[2]; }
    bool operator!=(const E &o) const { return !(*this == o); }
};
E ezero() { return E{{0, 0, 0}}; }
E eint(uint64_t v) { return E{{fc(v), 0, 0}}; }
E eadd(const E &x, const E &y) {
    return E{{fadd(x.c[0], y.c[0]), fadd(x.c[1], y.c[1]), fadd(x.c[2], y.c[2])}};
}
E esub(const E &x, const E &y) {
    return E{{fsub(x.c[0], y.c[0]), fsub(x.c[1], y.c[1]), fsub(x.c[2], y.c[2])}};
}
E emul(const E &x, const E &y) {
    uint64_t c0 = fadd(fmul(x.c[0], y.c[0]),
                       fmul(2, fadd(fmul(x.c[1], y.c[2]), fmul(x.c[2], y.c[1]))));
    uint64_t c1 = fadd(fadd(fmul(x.c[0], y.c[1]), fmul(x.c[1], y.c[0])),
                       fmul(2, fmul(x.c[2], y.c[2])));
    uint64_t c2 = fadd(fadd(fmul(x.c[0], y.c[2]), fmul(x.c[1], y.c[1])), fmul(x.c[2], y.c[0]));
    return E{{c0, c1, c2}};
}
E eraw(const uint64_t *at) { return E{{fc(at[0]), fc(at[1]), fc(at[2])}}; }

uint64_t state = 0x3C6EF372FE94F82Bull;
uint64_t next_u64() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// A raw limb: every fourth one non-canonical, in [p, 2^64), as the card's arithmetic leaves them.
uint64_t raw_limb() {
    uint64_t v = next_u64() % P;
    if ((next_u64() & 3) == 0 && v < 0xFFFFFFFFull) v += P;
    return v;
}

std::vector<uint64_t> random_words(size_t n) {
    std::vector<uint64_t> v(n);
    for (auto &x : v) x = raw_limb();
    return v;
}

// The parent from the grandchildren as the two folds make it: `(0, 2)` into the middle layer's
// lower half, `(1, 3)` into its upper half; `v[0..4]` numerators, `v[4..8]` denominators.
E composed(const E *v, const E &lambda) {
    auto fold = [](const E &p1, const E &q1, const E &p2, const E &q2, E &p, E &q) {
        p = eadd(emul(p1, q2), emul(p2, q1));
        q = emul(q1, q2);
    };
    E lo_p, lo_q, hi_p, hi_q, p, q;
    fold(v[0], v[4], v[2], v[6], lo_p, lo_q);
    fold(v[1], v[5], v[3], v[7], hi_p, hi_q);
    fold(lo_p, lo_q, hi_p, hi_q, p, q);
    return eadd(p, emul(lambda, q));
}

enum Kind { V0, V1, V2, V3, V3B };
const char *kind_name(Kind k) {
    switch (k) {
        case V0: return "gkr4_round_gruen";
        case V1: return "gkr4_round_gruen_lb2";
        case V2: return "gkr2_round_v2";
        case V3: return "gkr2_round_v3";
        case V3B: return "gkr2_round_v3b";
    }
    return "?";
}
bool is_pair(Kind k) { return k == V3 || k == V3B; }

// The rows of one launch, at the driver's shape for `lanes` lanes, at most `max_grid` blocks.
std::vector<E> launch(Kind kind, bool h0, uint32_t fault, uint64_t *p, uint64_t *q, uint64_t stride,
                      uint64_t quarter, bool fold, const uint64_t s[3], const uint64_t lam[3],
                      const uint64_t *e_hi, const uint64_t *e_lo, uint32_t low, unsigned max_grid) {
    uint64_t lanes = is_pair(kind) ? 2 * quarter : quarter;
    uint64_t bl = 1;
    while (bl < lanes) bl <<= 1;
    unsigned block = (unsigned)(bl < 32 ? 32 : bl > 256 ? 256 : bl);
    uint64_t g = (lanes + block - 1) / block;
    unsigned grid = (unsigned)(g < 1 ? 1 : g > max_grid ? max_grid : g);
    unsigned rows = h0 ? 5 : 4;
    std::vector<uint64_t> partials(5 * grid * 3, 0xDEADull);
    uint32_t f = fold ? 1 : 0;
    simt_launch(grid, block, [&] {
        switch (kind) {
            case V0:
                gkr4_round_gruen(p, q, stride, quarter, f, s[0], s[1], s[2], lam[0], lam[1], lam[2],
                                 e_hi, e_lo, low, h0 ? 1 : 0, fault == 1 ? 1 : 0, partials.data());
                break;
            case V1:
                gkr4_round_gruen_lb2(p, q, stride, quarter, f, s[0], s[1], s[2], lam[0], lam[1],
                                     lam[2], e_hi, e_lo, low, h0 ? 1 : 0, fault == 1 ? 1 : 0,
                                     partials.data());
                break;
            case V2:
                (h0 ? gkr2_round_v2_h0 : gkr2_round_v2)(p, q, stride, quarter, f, s[0], s[1], s[2],
                                                       lam[0], lam[1], lam[2], e_hi, e_lo, low,
                                                       fault, partials.data());
                break;
            case V3:
                (h0 ? gkr2_round_v3_h0 : gkr2_round_v3)(p, q, stride, quarter, f, s[0], s[1], s[2],
                                                       lam[0], lam[1], lam[2], e_hi, e_lo, low,
                                                       fault, partials.data());
                break;
            case V3B:
                (h0 ? gkr2_round_v3b_h0 : gkr2_round_v3b)(p, q, stride, quarter, f, s[0], s[1],
                                                         s[2], lam[0], lam[1], lam[2], e_hi, e_lo,
                                                         low, fault, partials.data());
                break;
        }
    });
    std::vector<E> out(rows, ezero());
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned b = 0; b < grid; ++b)
            out[r] = eadd(out[r], eraw(&partials[(r * grid + b) * 3]));
    return out;
}

struct Layer {
    unsigned m;  // a quarter holds 2^m cells
    std::vector<uint64_t> p, q;
    uint64_t lam[3];
    std::vector<std::vector<uint64_t>> e_hi;  // a level a round
    std::vector<uint64_t> e_lo;
    std::vector<std::array<uint64_t, 3>> challenges;
};

const uint32_t LOW = 2;

Layer make_layer(unsigned m) {
    Layer L;
    L.m = m;
    uint64_t cells = 4ull << m;
    L.p = random_words(cells * 3);
    L.q = random_words(cells * 3);
    for (int i = 0; i < 3; ++i) L.lam[i] = raw_limb();
    unsigned rounds = m - LOW;
    L.e_lo = random_words((1ull << LOW) * 3);
    for (unsigned j = 0; j < rounds; ++j) L.e_hi.push_back(random_words((1ull << (rounds - 1 - j)) * 3));
    for (unsigned j = 0; j < rounds; ++j) L.challenges.push_back({raw_limb(), raw_limb(), raw_limb()});
    return L;
}

// Every round of one kernel against the reference: rows and folded quarters.
void run_all_rounds(Kind kind, bool h0, const Layer &L, unsigned max_grid) {
    std::vector<uint64_t> p = L.p, q = L.q;
    uint64_t stride = 1ull << L.m;
    // The reference's quarters: f[k], k = 0..4 numerators, 4..8 denominators.
    std::vector<std::vector<E>> f(8);
    for (int k = 0; k < 8; ++k) {
        const std::vector<uint64_t> &src = k < 4 ? L.p : L.q;
        for (uint64_t i = 0; i < stride; ++i) f[k].push_back(eraw(&src[((k & 3) * stride + i) * 3]));
    }
    E lam = eraw(L.lam);
    unsigned rounds = L.m - LOW;
    uint64_t mask = (1ull << LOW) - 1;
    char tag[160];
    for (unsigned j = 0; j < rounds; ++j) {
        bool fold = j > 0;
        uint64_t s[3] = {0, 0, 0};
        if (fold) {
            for (int i = 0; i < 3; ++i) s[i] = L.challenges[j - 1][i];
            E se = eraw(s);
            for (auto &fk : f) {
                size_t half = fk.size() / 2;
                std::vector<E> folded(half);
                for (size_t x = 0; x < half; ++x)
                    folded[x] = eadd(fk[x], emul(se, esub(fk[x + half], fk[x])));
                fk = folded;
            }
        }
        uint64_t quarter = 1ull << (L.m - j - 1);
        unsigned rows = h0 ? 5 : 4;
        std::vector<E> want(rows, ezero());
        const uint64_t ts[5] = {1, 2, 3, 4, 0};
        for (uint64_t x = 0; x < quarter; ++x) {
            E w = emul(eraw(&L.e_hi[j][(x >> LOW) * 3]), eraw(&L.e_lo[(x & mask) * 3]));
            for (unsigned row = 0; row < rows; ++row) {
                E t = eint(ts[row]);
                E v[8];
                for (int k = 0; k < 8; ++k) v[k] = eadd(f[k][x], emul(t, esub(f[k][x + quarter], f[k][x])));
                want[row] = eadd(want[row], emul(w, composed(v, lam)));
            }
        }
        std::vector<E> got = launch(kind, h0, 0, p.data(), q.data(), stride, quarter, fold, s, L.lam,
                                    L.e_hi[j].data(), L.e_lo.data(), LOW, max_grid);
        snprintf(tag, sizeof tag, "%s%s 2^%u grid<=%u round %u", kind_name(kind), h0 ? " h0" : "",
                 L.m + 2, max_grid, j);
        bool rows_ok = got.size() == want.size();
        for (unsigned r = 0; rows_ok && r < rows; ++r) rows_ok = got[r] == want[r];
        check(rows_ok, std::string(tag) + ": the rows = the reference's");
        // The cube the next round reads: 2·quarter cells of every quarter.
        bool cells_ok = true;
        for (int k = 0; k < 8 && cells_ok; ++k) {
            const std::vector<uint64_t> &dev = k < 4 ? p : q;
            for (uint64_t i = 0; i < 2 * quarter && cells_ok; ++i)
                cells_ok = eraw(&dev[((k & 3) * stride + i) * 3]) == f[k][i];
        }
        check(cells_ok, std::string(tag) + ": the folded quarters = the reference's");
    }
}

// Round 0 (no fold) of one kernel with a fault: parts from the clean round.
void run_fault(Kind kind, uint32_t fault, const Layer &L) {
    uint64_t stride = 1ull << L.m, quarter = stride / 2;
    uint64_t s[3] = {0, 0, 0};
    std::vector<uint64_t> p = L.p, q = L.q;
    std::vector<E> clean = launch(kind, true, 0, p.data(), q.data(), stride, quarter, false, s,
                                  L.lam, L.e_hi[0].data(), L.e_lo.data(), LOW, 2048);
    std::vector<E> bad = launch(kind, true, fault, p.data(), q.data(), stride, quarter, false, s,
                                L.lam, L.e_hi[0].data(), L.e_lo.data(), LOW, 2048);
    std::string tag = std::string(kind_name(kind)) + " 2^" + std::to_string(L.m + 2) + " fault " +
                      std::to_string(fault);
    check(bad != clean, tag + ": parts from the clean round");
    if (fault == 1) {
        check(bad[0] == clean[0], tag + ": H(1) reads no lower cell and keeps its value");
        check(bad[4] != clean[4], tag + ": H(0) reads the lower cell and moves");
    }
}

}  // namespace

int main() {
    const Kind kinds[] = {V0, V1, V2, V3, V3B};
    for (unsigned m : {8u, 10u}) {
        Layer L = make_layer(m);
        for (Kind k : kinds) {
            for (bool h0 : {false, true}) {
                run_all_rounds(k, h0, L, 2048);
                if (m == 8) run_all_rounds(k, h0, L, 2);
            }
        }
        if (m == 8) {
            for (Kind k : kinds) run_fault(k, 1, L);
            run_fault(V3, 2, L);
            run_fault(V3B, 2, L);
        }
    }
    printf("gkr2 bench host KAT: %d checks, %d failures\n", checks, failures);
    if (failures == 0) printf("GKR2 HOST KAT PASS\n");
    return failures == 0 ? 0 : 1;
}
