// Known-answer tests for the GKR rounds that prove a layer against its
// grandchildren (`kernels/sumcheck.cu`: `gkr2_round{,_real}{,_slot}`,
// `gkr2_round_h0{,_real}`, `gkr2_finish{,_real}`), run on the host through
// `cuda_host_simt_shim.h` — every thread of a block a real thread, so the
// lane pairs' shuffles and the block reductions run as written. No GPU, no
// nvcc.
//
// WHAT IT COVERS, for every shape (the quarters' variables `m`, where the
// padding of a kept input layer starts, the cube left to the host, which
// rounds sum `H(0)`, the launch's grid):
//   1. A WRITTEN-OUT REFERENCE: each round's `H(1)..H(4)` (and `H(0)` where
//      asked) computed here from the whole layer by the two folds composed —
//      the middle layer's halves `fold(Q_{0b₂}, Q_{1b₂})`, then their fold,
//      then `p + λq` — each factor extended linearly to the node, weighted by
//      its own `eq(u_{>j}, x)` as a product over the bits of `x`; EQUAL to the
//      card's (canonical values). The folded quarters after every round and
//      the finish's eight bound factors, in the proof's order, likewise.
//   2. THE KEPT LAYER: quarters 2 and 3 stored only to where the padding
//      starts, poison past that, rounds launched as the driver launches them
//      (`math_cuda::gkr::Gruen2Layer`: the kept kernel while a stored run is
//      shorter than the cube, the whole-layer kernel after) and the kept
//      kernel every round — against the whole layer's kernels: every round's
//      block partials, the quarters' stored cells after every round, `H(0)`
//      and the finish, RAW-identical; the poison never read (a read would move
//      a sum) and never written.
//   3. THE SLOT: the slot kernels' sums in `d_slot` RAW-equal to the block
//      partials summed in `sum_partials_ext3`'s order, and the ticket reset.
//   4. NEGATIVE CONTROLS: one cell of the layer changed, or `λ` passed where
//      the kernel takes `λ/2`, and the round parts from the reference.
//
// WHAT IT DOES NOT COVER, and the card tests own: nvcc's acceptance of the
// file, registers and spills, the hardware's warps and memory, and the
// driver's launches (`gkr2_gruen.rs`, `argue_gkr2_truncated.rs` on a box).
//
// Build and run with `make test-gkr2-host-kat`.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_host_simt_shim.h"

// The kernels' block-shared buffer: blocks run one at a time here, so one
// buffer serves every block. `extern "C"` because the kernels that name it
// are. The launch bounds mean nothing on the host.
#define __shared__
#define __launch_bounds__(...)
extern "C" {
uint64_t shared[4096];
}

#include "sumcheck.cu"

namespace {

int failures = 0;
uint64_t checks = 0;

void check(bool ok, const std::string &what) {
    ++checks;
    if (!ok) {
        printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const uint64_t P = 0xFFFFFFFF00000001ull;
// `1/2` in the base field: `(p + 1)/2`.
const uint64_t HALF = 0x7FFFFFFF80000001ull;

uint64_t state = 0x13198A2E03707344ull;
uint64_t next_u64() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// A raw limb: canonical most of the time, and every eighth one a
// non-canonical representative in [p, 2^64) — what a card may hold.
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

Fe3 at(const std::vector<uint64_t> &v, uint64_t i) {
    return ext3::make(v[i * 3], v[i * 3 + 1], v[i * 3 + 2]);
}

void put(std::vector<uint64_t> &v, uint64_t i, const Fe3 &x) {
    v[i * 3] = x.a;
    v[i * 3 + 1] = x.b;
    v[i * 3 + 2] = x.c;
}

// `math_cuda::gkr::gruen2_shape`: a lane pair a cube index, 256-lane blocks
// for a wide cube and down to a warp for a narrow one, at most `max_grid`
// blocks (2048 on the card; smaller here, so the grid-stride loop runs).
void gruen2_shape(uint64_t quarter, unsigned max_grid, unsigned &grid, unsigned &block) {
    uint64_t lanes = 2 * quarter, b = 1;
    while (b < lanes) b <<= 1;
    block = (unsigned)std::min<uint64_t>(std::max<uint64_t>(b, 32), 256);
    grid = (unsigned)std::min<uint64_t>(std::max<uint64_t>((lanes + block - 1) / block, 1),
                                        max_grid);
}

// `crate::round_slot::shared_bytes` and the round's own need, in u64.
const uint64_t POISON = 0xDEAD5EED0BADF00Dull;

struct Shape {
    unsigned m;        // the quarters' variables: each quarter `2^m` cells
    uint64_t upper;    // the upper quarters' real cells together, (0, 2^{m+1}]
    unsigned low;      // at most this many variables left to the host
    int h0;            // which rounds sum H(0): 0 none, 1 every one, 2 the odd ones
    unsigned max_grid; // the launch's grid cap
    bool slot;         // also run the slot kernels
};

std::string label(const Shape &s) {
    char b[200];
    snprintf(b, sizeof b, "m %u upper %llu low %u h0 %d grid<=%u slot %d", s.m,
             (unsigned long long)s.upper, s.low, s.h0, s.max_grid, (int)s.slot);
    return b;
}

bool want_h0(const Shape &s, uint64_t j) { return s.h0 == 1 || (s.h0 == 2 && (j & 1)); }

struct Run {
    Shape s;
    uint64_t stride, rounds, low;
    std::vector<uint64_t> point, e; // u (m ext3), the eq levels as the kernel lays them out
    Fe3 lambda, half_lambda;
    std::vector<Fe3> challenges;    // one a round
    std::vector<uint64_t> wp, wq;   // the whole layer, the whole-layer kernels
    std::vector<uint64_t> kp, kq;   // kept, the driver's choice of kernel
    std::vector<uint64_t> ap, aq;   // kept, the kept kernel every round
    std::vector<uint64_t> sp, sq;   // whole, the slot kernels
    std::vector<Fe3> ref[2][4];     // the reference: [p, q][quarter], whole
};

// `eq(u_{j+1..m−1}, x)` written out: a product over the bits of `x`.
Fe3 eq_weight(const Run &r, uint64_t j, uint64_t x) {
    uint64_t bits = r.s.m - j - 1;
    Fe3 acc = ext3::one();
    for (uint64_t i = 0; i < bits; ++i) {
        Fe3 u = at(r.point, j + 1 + i);
        bool set = (x >> (bits - 1 - i)) & 1;
        acc = ext3::mul(acc, set ? u : ext3::sub(ext3::one(), u));
    }
    return acc;
}

// One fold of two fractions: `(p q' + p' q, q q')`.
void fold_fraction(const Fe3 &p, const Fe3 &q, const Fe3 &p2, const Fe3 &q2, Fe3 &np, Fe3 &nq) {
    np = ext3::add(ext3::mul(p, q2), ext3::mul(p2, q));
    nq = ext3::mul(q, q2);
}

// The parent at one node from the grandchildren's values there, `v[f][j]`
// (`f` 0 for `p`, 1 for `q`; quarter `j = 2b₁ + b₂`), by the two folds.
Fe3 parent(const Fe3 v[2][4], const Fe3 &lambda) {
    Fe3 mp[2], mq[2];
    for (int b2 = 0; b2 < 2; ++b2)
        fold_fraction(v[0][b2], v[1][b2], v[0][2 + b2], v[1][2 + b2], mp[b2], mq[b2]);
    Fe3 lp, lq;
    fold_fraction(mp[0], mq[0], mp[1], mq[1], lp, lq);
    return ext3::add(lp, ext3::mul(lambda, lq));
}

// The reference's sums `H(0..4)` at round `j`, after folding its whole layer
// by the last challenge.
void reference_round(Run &r, uint64_t j, Fe3 sums[5]) {
    uint64_t quarter = r.stride >> (j + 1);
    if (j > 0) {
        Fe3 s = r.challenges[j - 1];
        for (auto &side : r.ref)
            for (auto &g : side) {
                for (uint64_t x = 0; x < 2 * quarter; ++x)
                    g[x] = ext3::add(g[x], ext3::mul(s, ext3::sub(g[x + 2 * quarter], g[x])));
                g.resize(2 * quarter);
            }
    }
    for (int t = 0; t < 5; ++t) sums[t] = ext3::zero();
    for (uint64_t x = 0; x < quarter; ++x) {
        Fe3 w = eq_weight(r, j, x);
        for (int t = 0; t < 5; ++t) {
            Fe3 node = ext3::make((uint64_t)t, 0, 0), v[2][4];
            for (int f = 0; f < 2; ++f)
                for (int k = 0; k < 4; ++k) {
                    Fe3 lo = r.ref[f][k][x], hi = r.ref[f][k][x + quarter];
                    v[f][k] = ext3::add(lo, ext3::mul(node, ext3::sub(hi, lo)));
                }
            sums[t] = ext3::add(sums[t], ext3::mul(w, parent(v, r.lambda)));
        }
    }
}

// The real cells of quarters 2 and 3, from the upper quarters' run.
void quarter_runs(uint64_t upper, uint64_t stride, uint64_t &stored2, uint64_t &stored3) {
    stored2 = std::min(upper, stride);
    stored3 = upper > stride ? upper - stride : 0;
}

// `math_cuda::gkr::stored_after_round`.
uint64_t stored_after_round(uint64_t stored, uint64_t quarter) {
    return std::min<uint64_t>(stored, 2 * quarter);
}

enum Kernel { WHOLE, KEPT, SLOT_WHOLE };

struct Launched {
    std::vector<uint64_t> partials;
    std::vector<uint64_t> slot; // the slot's rows (SLOT_WHOLE only)
    unsigned grid;
};

Launched launch_round(Run &r, Kernel kernel, uint64_t *p, uint64_t *q, uint64_t j,
                      uint64_t stored2, uint64_t stored3, const Fe3 &h) {
    uint64_t quarter = r.stride >> (j + 1);
    unsigned grid, block;
    gruen2_shape(quarter, r.s.max_grid, grid, block);
    uint32_t fold = j > 0;
    Fe3 s = j > 0 ? r.challenges[j - 1] : ext3::zero();
    uint64_t at_level = (1ull << r.low) + (1ull << r.rounds) - (1ull << (r.rounds - j));
    const uint64_t *e_lo = r.e.data();
    const uint64_t *e_hi = r.e.data() + at_level * 3;
    Launched out;
    out.grid = grid;
    out.partials.assign(5 * grid * 3, POISON);
    uint64_t stride = r.stride;
    uint32_t low = (uint32_t)r.low;
    unsigned int ticket = 0;
    std::vector<uint64_t> slot(97, 0);
    simt_launch(grid, block, [&] {
        if (kernel == WHOLE)
            gkr2_round(p, q, stride, quarter, fold, s.a, s.b, s.c, h.a, h.b, h.c, e_hi, e_lo, low,
                       0, 0, out.partials.data());
        else if (kernel == KEPT)
            gkr2_round_real(p, q, stride, quarter, fold, s.a, s.b, s.c, h.a, h.b, h.c, e_hi, e_lo,
                            low, stored2, stored3, out.partials.data());
        else
            gkr2_round_slot(p, q, stride, quarter, fold, s.a, s.b, s.c, h.a, h.b, h.c, e_hi, e_lo,
                            low, 0, 0, out.partials.data(), &ticket, slot.data(), 7 + j);
    });
    if (kernel == SLOT_WHOLE) {
        check(ticket == 0, label(r.s) + " round " + std::to_string(j) + ": the slot's ticket reset");
        check(slot[96] == 7 + j, label(r.s) + " round " + std::to_string(j) + ": the slot's seq");
        out.slot.assign(slot.begin(), slot.begin() + 12);
    }
    return out;
}

// `H(0)`'s pass, into row 4 of `partials`.
void launch_h0(Run &r, bool real, const uint64_t *p, const uint64_t *q, uint64_t j,
               uint64_t stored2, uint64_t stored3, unsigned grid, std::vector<uint64_t> &partials) {
    uint64_t quarter = r.stride >> (j + 1);
    unsigned g, block;
    gruen2_shape(quarter, r.s.max_grid, g, block);
    uint64_t at_level = (1ull << r.low) + (1ull << r.rounds) - (1ull << (r.rounds - j));
    const uint64_t *e_lo = r.e.data();
    const uint64_t *e_hi = r.e.data() + at_level * 3;
    uint64_t stride = r.stride;
    uint32_t low = (uint32_t)r.low;
    Fe3 l = r.lambda;
    simt_launch(grid, block, [&] {
        if (real)
            gkr2_round_h0_real(p, q, stride, quarter, l.a, l.b, l.c, e_hi, e_lo, low, stored2,
                               stored3, partials.data());
        else
            gkr2_round_h0(p, q, stride, quarter, l.a, l.b, l.c, e_hi, e_lo, low, 0, 0,
                          partials.data());
    });
}

// A row's sum over the blocks, in `sum_partials_ext3`'s order: 256 lanes, lane
// `v` adding partials `v, v + 256, …`, then the halving tree.
Fe3 row_sum(const std::vector<uint64_t> &partials, int row, unsigned grid) {
    std::vector<Fe3> lanes(256, ext3::zero());
    for (unsigned v = 0; v < 256; ++v)
        for (uint64_t i = v; i < grid; i += 256)
            lanes[v] = ext3::add(lanes[v], at(partials, (uint64_t)row * grid + i));
    for (unsigned width = 128; width > 0; width >>= 1)
        for (unsigned v = 0; v < width; ++v) lanes[v] = ext3::add(lanes[v], lanes[v + width]);
    return lanes[0];
}

std::vector<uint64_t> launch_finish(Run &r, bool real, const uint64_t *p, const uint64_t *q,
                                    uint64_t stored2, uint64_t stored3) {
    uint64_t cells = 1ull << r.low;
    uint32_t fold = r.rounds > 0;
    Fe3 s = fold ? r.challenges[r.rounds - 1] : ext3::zero();
    std::vector<uint64_t> out(8 * cells * 3, POISON);
    uint64_t stride = r.stride;
    simt_launch(2, 32, [&] {
        if (real)
            gkr2_finish_real(p, q, stride, cells, fold, s.a, s.b, s.c, stored2, stored3,
                             out.data());
        else
            gkr2_finish(p, q, stride, cells, fold, s.a, s.b, s.c, out.data());
    });
    return out;
}

// Quarter `k`'s first `n` cells of side `f`, raw-equal between two buffers.
bool same_cells(const std::vector<uint64_t> &a, const std::vector<uint64_t> &b, uint64_t at,
                uint64_t n) {
    return memcmp(a.data() + at * 3, b.data() + at * 3, n * 3 * sizeof(uint64_t)) == 0;
}

// The kept buffers' cells past the real ones still hold the poison.
bool poison_intact(const std::vector<uint64_t> &v, uint64_t real) {
    for (uint64_t i = real * 3; i < v.size(); ++i)
        if (v[i] != POISON) return false;
    return true;
}

uint64_t layers_run = 0, rounds_run = 0, kept_rounds = 0, whole_rounds_on_kept = 0;

void run_shape(const Shape &shape) {
    Run r;
    r.s = shape;
    r.stride = 1ull << shape.m;
    r.low = std::min<uint64_t>(shape.low, shape.m);
    r.rounds = shape.m - r.low;
    const std::string what = label(shape);
    const uint64_t stride = r.stride, full = 4 * stride, real = 2 * stride + shape.upper;

    r.wp.resize(full * 3);
    r.wq.resize(full * 3);
    for (uint64_t i = 0; i < full; ++i) {
        bool padding = i >= real;
        put(r.wp, i, padding ? ext3::zero() : random_ext());
        put(r.wq, i, padding ? ext3::one() : random_ext());
    }
    r.sp = r.wp;
    r.sq = r.wq;
    r.kp.assign(full * 3, POISON);
    r.kq.assign(full * 3, POISON);
    std::copy(r.wp.begin(), r.wp.begin() + real * 3, r.kp.begin());
    std::copy(r.wq.begin(), r.wq.begin() + real * 3, r.kq.begin());
    r.ap = r.kp;
    r.aq = r.kq;
    for (int f = 0; f < 2; ++f)
        for (int k = 0; k < 4; ++k) {
            const std::vector<uint64_t> &side = f == 0 ? r.wp : r.wq;
            r.ref[f][k].resize(stride);
            for (uint64_t x = 0; x < stride; ++x) r.ref[f][k][x] = at(side, k * stride + x);
        }
    r.point.resize(shape.m * 3);
    for (unsigned i = 0; i < shape.m; ++i) put(r.point, i, random_ext());
    r.lambda = random_ext();
    r.half_lambda = ext3::mul(r.lambda, ext3::make(HALF, 0, 0));
    for (uint64_t j = 0; j < r.rounds; ++j) r.challenges.push_back(random_ext());

    uint64_t levels = (1ull << r.low) + (1ull << r.rounds) - 1;
    r.e.assign(levels * 3, 0);
    {
        uint32_t low = (uint32_t)r.low, rounds = (uint32_t)r.rounds;
        std::vector<uint64_t> zero(3, 0);
        const uint64_t *pt = r.point.empty() ? zero.data() : r.point.data();
        simt_launch(1, 32, [&] { gkr_eq_levels_ext3(pt, low, rounds, r.e.data()); });
    }

    // Negative controls, on copies, before any round folds the layer: round 0
    // over a layer with one cell changed, and with `λ` where `λ/2` goes.
    if (r.rounds > 0) {
        Fe3 want[5];
        Run copy = r;
        reference_round(copy, 0, want);
        std::vector<uint64_t> np = r.wp, nq = r.wq;
        uint64_t victim = 3 * stride + (next_u64() % stride);
        if (victim >= real) victim = 2 * stride; // a real cell
        put(nq, victim, ext3::add(at(nq, victim), ext3::one()));
        Launched one = launch_round(r, WHOLE, np.data(), nq.data(), 0, 0, 0, r.half_lambda);
        bool parted = false;
        for (int row = 0; row < 4; ++row)
            parted |= !same_value(row_sum(one.partials, row, one.grid), want[row + 1]);
        check(parted, what + ": a layer one cell off parts from the reference");
        np = r.wp;
        nq = r.wq;
        Launched lam = launch_round(r, WHOLE, np.data(), nq.data(), 0, 0, 0, r.lambda);
        parted = false;
        for (int row = 0; row < 4; ++row)
            parted |= !same_value(row_sum(lam.partials, row, lam.grid), want[row + 1]);
        check(parted, what + ": λ where λ/2 goes parts from the reference");
    }

    uint64_t k2, k3, a2, a3;
    quarter_runs(shape.upper, stride, k2, k3);
    a2 = k2;
    a3 = k3;
    for (uint64_t j = 0; j < r.rounds; ++j) {
        const std::string rw = what + " round " + std::to_string(j);
        uint64_t quarter = stride >> (j + 1);
        uint64_t cube = (j > 0 ? 4 : 2) * quarter;
        Launched whole = launch_round(r, WHOLE, r.wp.data(), r.wq.data(), j, 0, 0, r.half_lambda);
        bool keeps = k2 < cube || k3 < cube;
        kept_rounds += keeps;
        whole_rounds_on_kept += !keeps;
        Launched kept = launch_round(r, keeps ? KEPT : WHOLE, r.kp.data(), r.kq.data(), j, k2, k3,
                                     r.half_lambda);
        Launched always = launch_round(r, KEPT, r.ap.data(), r.aq.data(), j, a2, a3,
                                       r.half_lambda);
        std::vector<uint64_t> rows4(whole.partials.begin(), whole.partials.begin() + 4 * whole.grid * 3);
        check(std::equal(rows4.begin(), rows4.end(), kept.partials.begin()),
              rw + ": the kept layer's partials = the whole layer's, raw");
        check(std::equal(rows4.begin(), rows4.end(), always.partials.begin()),
              rw + ": the kept kernel's partials = the whole layer's, raw (every round)");
        if (shape.slot) {
            Launched slotted = launch_round(r, SLOT_WHOLE, r.sp.data(), r.sq.data(), j, 0, 0,
                                            r.half_lambda);
            check(std::equal(rows4.begin(), rows4.end(), slotted.partials.begin()),
                  rw + ": the slot kernel's partials = the whole layer's, raw");
            bool same = true;
            for (int row = 0; row < 4; ++row) {
                Fe3 want = row_sum(whole.partials, row, whole.grid);
                Fe3 got = at(slotted.slot, row);
                same &= got.a == want.a && got.b == want.b && got.c == want.c;
            }
            check(same, rw + ": the slot's sums = the partials summed as the second launch sums them, raw");
        }
        uint64_t n2 = stored_after_round(k2, quarter), n3 = stored_after_round(k3, quarter);
        uint64_t m2 = stored_after_round(a2, quarter), m3 = stored_after_round(a3, quarter);
        if (want_h0(shape, j)) {
            launch_h0(r, false, r.wp.data(), r.wq.data(), j, 0, 0, whole.grid, whole.partials);
            launch_h0(r, n2 < 2 * quarter || n3 < 2 * quarter, r.kp.data(), r.kq.data(), j, n2, n3,
                      kept.grid, kept.partials);
            launch_h0(r, true, r.ap.data(), r.aq.data(), j, m2, m3, always.grid, always.partials);
            std::vector<uint64_t> row4(whole.partials.begin() + 4 * whole.grid * 3,
                                       whole.partials.begin() + 5 * whole.grid * 3);
            check(std::equal(row4.begin(), row4.end(), kept.partials.begin() + 4 * kept.grid * 3) &&
                      std::equal(row4.begin(), row4.end(),
                                 always.partials.begin() + 4 * always.grid * 3),
                  rw + ": the kept layer's H(0) partials = the whole layer's, raw");
        }
        Fe3 want[5];
        reference_round(r, j, want);
        for (int row = 0; row < 4; ++row)
            check(same_value(row_sum(whole.partials, row, whole.grid), want[row + 1]),
                  rw + ": H(" + std::to_string(row + 1) + ") = the reference's");
        if (want_h0(shape, j))
            check(same_value(row_sum(whole.partials, 4, whole.grid), want[0]),
                  rw + ": H(0) = the reference's");
        // The quarters after the round: each holds `2·quarter` live cells.
        uint64_t live = 2 * quarter;
        for (int f = 0; f < 2; ++f) {
            const std::vector<uint64_t> &w = f == 0 ? r.wp : r.wq;
            const std::vector<uint64_t> &k = f == 0 ? r.kp : r.kq;
            const std::vector<uint64_t> &a = f == 0 ? r.ap : r.aq;
            for (int qd = 0; qd < 4; ++qd) {
                uint64_t nk = qd < 2 ? live : std::min(qd == 2 ? n2 : n3, live);
                uint64_t na = qd < 2 ? live : std::min(qd == 2 ? m2 : m3, live);
                check(same_cells(k, w, qd * stride, nk),
                      rw + ": side " + std::to_string(f) + " quarter " + std::to_string(qd) +
                          "'s stored cells = the whole layer's, raw");
                check(same_cells(a, w, qd * stride, na),
                      rw + ": side " + std::to_string(f) + " quarter " + std::to_string(qd) +
                          "'s stored cells = the whole layer's, raw (every round)");
                bool folded = true;
                for (uint64_t x = 0; x < live; ++x)
                    folded &= same_value(at(w, qd * stride + x), r.ref[f][qd][x]);
                check(folded, rw + ": side " + std::to_string(f) + " quarter " +
                                  std::to_string(qd) + " folded = the reference's");
            }
        }
        check(poison_intact(r.kp, real) && poison_intact(r.kq, real) &&
                  poison_intact(r.ap, real) && poison_intact(r.aq, real),
              rw + ": the poison past the stored runs never written");
        k2 = n2;
        k3 = n3;
        a2 = m2;
        a3 = m3;
        ++rounds_run;
    }

    // The finish: the eight factors bound to the last challenge, in the
    // proof's order, against the reference's.
    uint64_t cells = 1ull << r.low;
    uint64_t cube = r.rounds > 0 ? 2 * cells : cells;
    std::vector<uint64_t> whole = launch_finish(r, false, r.wp.data(), r.wq.data(), 0, 0);
    std::vector<uint64_t> kept =
        launch_finish(r, k2 < cube || k3 < cube, r.kp.data(), r.kq.data(), k2, k3);
    std::vector<uint64_t> always = launch_finish(r, true, r.ap.data(), r.aq.data(), a2, a3);
    check(whole == kept && whole == always, what + ": the kept layer's finish = the whole layer's, raw");
    if (r.rounds > 0) {
        Fe3 s = r.challenges[r.rounds - 1];
        for (auto &side : r.ref)
            for (auto &g : side) {
                for (uint64_t y = 0; y < cells; ++y)
                    g[y] = ext3::add(g[y], ext3::mul(s, ext3::sub(g[y + cells], g[y])));
                g.resize(cells);
            }
    }
    bool bound = true;
    for (int half = 0; half < 2; ++half)
        for (int f = 0; f < 4; ++f) {
            int side = f >= 2, qd = half + ((f & 1) ? 2 : 0);
            for (uint64_t y = 0; y < cells; ++y)
                bound &= same_value(at(whole, (uint64_t)(4 * half + f) * cells + y),
                                    r.ref[side][qd][y]);
        }
    check(bound, what + ": the finish's eight factors = the reference's, in the proof's order");
    ++layers_run;
}

} // namespace

int main() {
    std::vector<Shape> shapes;
    for (unsigned m : {1u, 2u, 4u, 6u}) {
        uint64_t stride = 1ull << m;
        std::vector<uint64_t> uppers = {2 * stride, 1, stride / 2 + 1, stride, stride + 1,
                                        2 * stride - 1};
        for (uint64_t upper : uppers) {
            if (upper == 0 || upper > 2 * stride) continue;
            for (unsigned low : {0u, 1u, 3u}) {
                if (low > m) continue;
                for (int h0 : {0, 2})
                    shapes.push_back({m, upper, low, h0, 2, h0 == 2});
            }
        }
    }
    // The card's shapes at 2^10 and 2^12 quarters: the driver's grid, every
    // round's H(0), the slot.
    shapes.push_back({10, 1ull << 11, 6, 1, 2048, true});
    shapes.push_back({10, (1ull << 10) + 77, 6, 2, 2048, true});
    shapes.push_back({12, 1ull << 13, 6, 0, 2048, true});
    shapes.push_back({12, 300, 6, 2, 2048, false});
    for (const Shape &s : shapes) run_shape(s);
    printf("gkr2 host KAT: %llu checks over %llu layers, %llu rounds (%llu on the kept kernel, %llu on the whole-layer "
           "kernel over a kept layer), %d failures\n",
           (unsigned long long)checks, (unsigned long long)layers_run, (unsigned long long)rounds_run,
           (unsigned long long)kept_rounds, (unsigned long long)whole_rounds_on_kept, failures);
    return failures == 0 ? 0 : 1;
}
