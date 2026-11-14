// Known-answer tests for the GKR layer rounds over an input layer kept on the
// card without its padding (`kernels/sumcheck.cu`: `gkr_round_gruen_real`,
// `gkr_gruen_finish_real`; `LAMBDA_VM_ARGUE_INPUT_REAL`), run on the host
// through `cuda_host_simt_shim.h` — every thread of a block a real thread, so
// the kernels' block reductions run as written. No GPU, no nvcc.
//
// WHAT IT COVERS, for every shape (the layer's `m`, where its padding starts,
// the cube left to the host, which rounds sum `H(0)`):
//   1. TODAY'S KERNELS UNCHANGED: `gkr_round_gruen` and `gkr_gruen_finish` as
//      shipped against a copy of the source they had before the kept layer
//      (below), over the whole layer: every round's block partials and the
//      folded halves, RAW-identical.
//   2. THE KEPT LAYER: the upper halves stored only to where the padding
//      starts, poison past that, rounds launched the way the driver launches
//      them (`math_cuda::gkr::GruenLayer`: the kept kernel while the stored run
//      is shorter than the cube, today's after) and, on the smaller shapes,
//      the kept kernel every round — against today's kernels over the whole
//      layer: every round's block partials, the halves' stored cells after
//      every round, the finish's output, RAW-identical; the poison never read
//      (a read would move a sum) and never written.
//   3. A WRITTEN-OUT REFERENCE: the sums `H(1)`, `H(2)`, `H(0)` and the
//      finish's halves computed here from the whole layer — its own fold, its
//      own `eq(u_{>j}, x)` as a product over the bits of `x` — EQUAL to the
//      kernels' (canonical values).
//   4. Today's path keeps the padding as the raw limbs `0/1`: after every fold
//      the whole layer's cells past the stored run are those limbs, which is
//      what makes the kept kernel's constants raw-identical.
//
// WHAT IT DOES NOT COVER, and the card tests still own: whether nvcc accepts
// the file, the hardware's warps and memory, launch shapes from the driver
// (`gkr_gruen_truncated.rs`, `argue_gkr_truncated.rs` on a box).
//
// Build and run with `make test-gkr-host-kat`.

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
uint64_t shared[3 * 1024];
}

#include "sumcheck.cu"

// ── today's two kernels, as they were before the kept layer ─────────────────
// Copied from `sumcheck.cu` at the commit this change starts from, renamed.
// Item 1 runs them beside the shipped ones.
void gkr_round_gruen_before(
    uint64_t *__restrict__ p, uint64_t *__restrict__ q, uint64_t half, uint64_t quarter,
    uint32_t fold, uint64_t s0, uint64_t s1, uint64_t s2, uint64_t l0, uint64_t l1, uint64_t l2,
    const uint64_t *__restrict__ e_hi, const uint64_t *__restrict__ e_lo, uint32_t low,
    uint32_t want_h0, uint64_t *__restrict__ d_partials) {
    Fe3 s = ext3::make(s0, s1, s2);
    Fe3 lambda = ext3::make(l0, l1, l2);
    uint64_t *factors[4] = {p, p + half * 3, q, q + half * 3};
    uint64_t mask = ((uint64_t)1 << low) - 1;
    Fe3 acc1 = ext3::zero(), acc2 = ext3::zero(), acc0 = ext3::zero();
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < quarter;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 lo[4], hi[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            uint64_t *f = factors[k];
            if (fold) {
                Fe3 a = load_ext(f + x * 3);
                Fe3 b = load_ext(f + (x + quarter) * 3);
                Fe3 c = load_ext(f + (x + 2 * quarter) * 3);
                Fe3 d = load_ext(f + (x + 3 * quarter) * 3);
                lo[k] = ext3::add(a, ext3::mul(s, ext3::sub(c, a)));
                hi[k] = ext3::add(b, ext3::mul(s, ext3::sub(d, b)));
                f[x * 3 + 0] = lo[k].a;
                f[x * 3 + 1] = lo[k].b;
                f[x * 3 + 2] = lo[k].c;
                f[(x + quarter) * 3 + 0] = hi[k].a;
                f[(x + quarter) * 3 + 1] = hi[k].b;
                f[(x + quarter) * 3 + 2] = hi[k].c;
            } else {
                lo[k] = load_ext(f + x * 3);
                hi[k] = load_ext(f + (x + quarter) * 3);
            }
        }
        Fe3 w = ext3::mul(load_ext(e_hi + (x >> low) * 3), load_ext(e_lo + (x & mask) * 3));
        acc1 = ext3::add(acc1, ext3::mul(w, gkr_h(hi[0], hi[1], hi[2], hi[3], lambda)));
        Fe3 two[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            two[k] = ext3::sub(ext3::add(hi[k], hi[k]), lo[k]);
        }
        acc2 = ext3::add(acc2, ext3::mul(w, gkr_h(two[0], two[1], two[2], two[3], lambda)));
        if (want_h0) {
            acc0 = ext3::add(acc0, ext3::mul(w, gkr_h(lo[0], lo[1], lo[2], lo[3], lambda)));
        }
    }

    extern __shared__ uint64_t shared[];
    uint32_t rows = want_h0 ? 3 : 2;
    for (uint32_t row = 0; row < rows; row++) {
        Fe3 mine = row == 0 ? acc1 : row == 1 ? acc2 : acc0;
        shared[threadIdx.x * 3 + 0] = mine.a;
        shared[threadIdx.x * 3 + 1] = mine.b;
        shared[threadIdx.x * 3 + 2] = mine.c;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
            if (threadIdx.x < width) {
                Fe3 sum = ext3::add(load_ext(shared + threadIdx.x * 3),
                                    load_ext(shared + (threadIdx.x + width) * 3));
                shared[threadIdx.x * 3 + 0] = sum.a;
                shared[threadIdx.x * 3 + 1] = sum.b;
                shared[threadIdx.x * 3 + 2] = sum.c;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            uint64_t at = ((uint64_t)row * gridDim.x + blockIdx.x) * 3;
            d_partials[at + 0] = shared[0];
            d_partials[at + 1] = shared[1];
            d_partials[at + 2] = shared[2];
        }
        __syncthreads();
    }
}

// The rounds' end: each half bound to the last challenge `s` (with `fold`) from its `2·cells` cube
// down to `cells`, written in place and to `out[(k·cells + y)·3]`, halves in the order `p_lo, p_hi,
// q_lo, q_hi` — what the host tail starts from. Thread `y` reads cells `y` and `y + cells` and
// writes cell `y` only.
void gkr_gruen_finish_before(uint64_t *__restrict__ p, uint64_t *__restrict__ q,
                                            uint64_t half, uint64_t cells, uint32_t fold,
                                            uint64_t s0, uint64_t s1, uint64_t s2,
                                            uint64_t *__restrict__ out) {
    Fe3 s = ext3::make(s0, s1, s2);
    uint64_t total = 4 * cells;
    for (uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; task < total;
         task += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t k = task / cells;
        uint64_t y = task - k * cells;
        uint64_t *f = (k < 2 ? p : q) + (k & 1) * half * 3;
        Fe3 v = load_ext(f + y * 3);
        if (fold) {
            Fe3 hi = load_ext(f + (y + cells) * 3);
            v = ext3::add(v, ext3::mul(s, ext3::sub(hi, v)));
            f[y * 3 + 0] = v.a;
            f[y * 3 + 1] = v.b;
            f[y * 3 + 2] = v.c;
        }
        uint64_t *at = out + task * 3;
        at[0] = v.a;
        at[1] = v.b;
        at[2] = v.c;
    }
}

namespace {

int failures = 0;

void check(bool ok, const std::string &what) {
    if (!ok) {
        printf("FAIL: %s\n", what.c_str());
        ++failures;
    }
}

const uint64_t P = 0xFFFFFFFF00000001ull;

uint64_t state = 0x243F6A8885A308D3ull;
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

// `math_cuda::gkr::gruen_shape`.
void gruen_shape(uint64_t quarter, unsigned &grid, unsigned &block) {
    uint64_t b = 1;
    while (b < quarter) b <<= 1;
    block = (unsigned)std::min<uint64_t>(std::max<uint64_t>(b, 32), 256);
    grid = (unsigned)std::min<uint64_t>(std::max<uint64_t>((quarter + block - 1) / block, 1), 2048);
}

// A poison cell: no value a kernel computes from the layer, and never `0/1`.
const uint64_t POISON = 0xDEAD5EED0BADF00Dull;

struct Shape {
    unsigned m;         // the halves' variables: each half `2^m` cells
    uint64_t stored;    // the upper halves' real cells, (0, 2^m]
    unsigned low;       // at most this many variables left to the host
    int h0;             // which rounds sum H(0): 0 none, 1 every one, 2 the odd ones
    bool every_round;   // also run the kept kernel on every round
    bool before;        // also run the copy of today's kernels
};

// One whole layer and its kept copy, and every round over both.
struct Run {
    Shape s;
    uint64_t half, rounds, low;
    std::vector<uint64_t> point, e;  // u (m ext3), the eq levels as the kernel lays them out
    Fe3 lambda;
    std::vector<Fe3> challenges;     // one a round
    // Buffers: p and q, each the whole layer's 2·half cells.
    std::vector<uint64_t> wp, wq;    // whole, today's kernels
    std::vector<uint64_t> bp, bq;    // whole, the copy of today's kernels
    std::vector<uint64_t> kp, kq;    // kept, the driver's choice of kernel
    std::vector<uint64_t> ap, aq;    // kept, the kept kernel every round
    std::vector<Fe3> ref[4];         // the reference: p_lo, p_hi, q_lo, q_hi, whole
};

std::string label(const Shape &s) {
    char b[160];
    snprintf(b, sizeof b, "m %u stored %llu low %u h0 %d", s.m, (unsigned long long)s.stored, s.low,
             s.h0);
    return b;
}

bool want_h0(const Shape &s, uint64_t j) { return s.h0 == 1 || (s.h0 == 2 && (j & 1)); }

// `eq(u_{j+1..m−1}, x)` written out: a product over the bits of `x`, variable
// `j + 1` in the high bit.
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

Fe3 h_of(const Fe3 v[4], const Fe3 &lambda) {
    return ext3::add(ext3::add(ext3::mul(v[0], v[3]), ext3::mul(v[1], v[2])),
                     ext3::mul(lambda, ext3::mul(v[2], v[3])));
}

enum Kernel { TODAY, BEFORE, KEPT };

// One round's launch over one pair of buffers; returns its block partials.
std::vector<uint64_t> launch_round(Run &r, Kernel kernel, uint64_t *p, uint64_t *q, uint64_t j,
                                   uint64_t stored) {
    uint64_t quarter = r.half >> (j + 1);
    unsigned grid, block;
    gruen_shape(quarter, grid, block);
    uint32_t fold = j > 0;
    Fe3 s = j > 0 ? r.challenges[j - 1] : ext3::zero();
    uint64_t at_level = (1ull << r.low) + (1ull << r.rounds) - (1ull << (r.rounds - j));
    const uint64_t *e_lo = r.e.data();
    const uint64_t *e_hi = r.e.data() + at_level * 3;
    uint32_t want = want_h0(r.s, j);
    std::vector<uint64_t> partials(3 * grid * 3, 0);
    uint64_t half = r.half;
    uint32_t low = (uint32_t)r.low;
    Fe3 l = r.lambda;
    simt_launch(grid, block, [&] {
        if (kernel == TODAY)
            gkr_round_gruen(p, q, half, quarter, fold, s.a, s.b, s.c, l.a, l.b, l.c, e_hi, e_lo, low,
                            want, partials.data());
        else if (kernel == BEFORE)
            gkr_round_gruen_before(p, q, half, quarter, fold, s.a, s.b, s.c, l.a, l.b, l.c, e_hi,
                                   e_lo, low, want, partials.data());
        else
            gkr_round_gruen_real(p, q, half, quarter, fold, s.a, s.b, s.c, l.a, l.b, l.c, e_hi,
                                 e_lo, low, want, stored, partials.data());
    });
    partials.resize((want ? 3 : 2) * grid * 3);
    return partials;
}

std::vector<uint64_t> launch_finish(Run &r, Kernel kernel, uint64_t *p, uint64_t *q,
                                    uint64_t stored) {
    uint64_t cells = 1ull << r.low;
    uint32_t fold = r.rounds > 0;
    Fe3 s = fold ? r.challenges[r.rounds - 1] : ext3::zero();
    std::vector<uint64_t> out(4 * cells * 3, POISON);
    uint64_t half = r.half;
    simt_launch(1, 32, [&] {
        if (kernel == TODAY)
            gkr_gruen_finish(p, q, half, cells, fold, s.a, s.b, s.c, out.data());
        else if (kernel == BEFORE)
            gkr_gruen_finish_before(p, q, half, cells, fold, s.a, s.b, s.c, out.data());
        else
            gkr_gruen_finish_real(p, q, half, cells, fold, s.a, s.b, s.c, stored, out.data());
    });
    return out;
}

// The driver's choice (`GruenLayer::round_enqueue`, `finish`): the kept kernel
// while the stored run is shorter than the cube the halves hold.
bool driver_keeps(uint64_t stored, uint64_t cube) { return stored < cube; }

// `math_cuda::gkr::stored_after_round`.
uint64_t stored_after_round(uint64_t stored, uint64_t quarter) {
    return std::min<uint64_t>(stored, 2 * quarter);
}

// The reference's sums at round `j`, after folding its whole layer by the
// last challenge.
void reference_round(Run &r, uint64_t j, Fe3 sums[3]) {
    uint64_t quarter = r.half >> (j + 1);
    if (j > 0) {
        Fe3 s = r.challenges[j - 1];
        for (auto &g : r.ref) {
            for (uint64_t x = 0; x < 2 * quarter; ++x)
                g[x] = ext3::add(g[x], ext3::mul(s, ext3::sub(g[x + 2 * quarter], g[x])));
            g.resize(2 * quarter);
        }
    }
    for (int t = 0; t < 3; ++t) sums[t] = ext3::zero();
    for (uint64_t x = 0; x < quarter; ++x) {
        Fe3 lo[4], hi[4], two[4];
        for (int k = 0; k < 4; ++k) {
            lo[k] = r.ref[k][x];
            hi[k] = r.ref[k][x + quarter];
            two[k] = ext3::sub(ext3::add(hi[k], hi[k]), lo[k]);
        }
        Fe3 w = eq_weight(r, j, x);
        sums[0] = ext3::add(sums[0], ext3::mul(w, h_of(hi, r.lambda)));
        sums[1] = ext3::add(sums[1], ext3::mul(w, h_of(two, r.lambda)));
        sums[2] = ext3::add(sums[2], ext3::mul(w, h_of(lo, r.lambda)));
    }
}

// The cells of factor `k` (0 p_lo, 1 p_hi, 2 q_lo, 3 q_hi) at `[0, n)`.
const uint64_t *factor(const std::vector<uint64_t> &p, const std::vector<uint64_t> &q, int k,
                       uint64_t half) {
    return (k < 2 ? p.data() : q.data()) + (k & 1) * half * 3;
}

bool same_raw(const uint64_t *a, const uint64_t *b, uint64_t cells) {
    return memcmp(a, b, cells * 3 * sizeof(uint64_t)) == 0;
}

// The kept buffers' cells past the real ones still hold the poison.
bool poison_intact(const std::vector<uint64_t> &v, uint64_t half, uint64_t stored) {
    for (uint64_t i = (half + stored) * 3; i < v.size(); ++i)
        if (v[i] != POISON) return false;
    return true;
}

uint64_t layers_run = 0, rounds_run = 0, kept_rounds = 0, today_rounds_on_kept = 0;

void run_shape(const Shape &shape) {
    Run r;
    r.s = shape;
    r.half = 1ull << shape.m;
    r.low = std::min<uint64_t>(shape.low, shape.m);
    r.rounds = shape.m - r.low;
    const std::string what = label(shape);
    const uint64_t half = r.half, full = 2 * half;

    // The layer: random fractions, the padding written out as today writes it.
    r.wp.resize(full * 3);
    r.wq.resize(full * 3);
    for (uint64_t i = 0; i < full; ++i) {
        bool padding = i >= half + shape.stored;
        put(r.wp, i, padding ? ext3::zero() : random_ext());
        put(r.wq, i, padding ? ext3::one() : random_ext());
    }
    r.bp = r.wp;
    r.bq = r.wq;
    // The kept copies: the real cells, poison where the padding was.
    r.kp.assign(full * 3, POISON);
    r.kq.assign(full * 3, POISON);
    std::copy(r.wp.begin(), r.wp.begin() + (half + shape.stored) * 3, r.kp.begin());
    std::copy(r.wq.begin(), r.wq.begin() + (half + shape.stored) * 3, r.kq.begin());
    r.ap = r.kp;
    r.aq = r.kq;
    for (int k = 0; k < 4; ++k) {
        const uint64_t *f = factor(r.wp, r.wq, k, half);
        r.ref[k].resize(half);
        for (uint64_t x = 0; x < half; ++x) r.ref[k][x] = ext3::make(f[x * 3], f[x * 3 + 1], f[x * 3 + 2]);
    }
    r.point.resize(shape.m * 3);
    for (unsigned i = 0; i < shape.m; ++i) put(r.point, i, random_ext());
    r.lambda = random_ext();
    for (uint64_t j = 0; j < r.rounds; ++j) r.challenges.push_back(random_ext());

    // The eq levels, by the kernel the driver launches.
    uint64_t cells = (1ull << r.low) + (1ull << r.rounds) - 1;
    r.e.assign(cells * 3, 0);
    {
        uint32_t low = (uint32_t)r.low, rounds = (uint32_t)r.rounds;
        const uint64_t *pt = r.point.empty() ? nullptr : r.point.data();
        std::vector<uint64_t> zero(3, 0);
        if (!pt) pt = zero.data();
        simt_launch(1, 32, [&] { gkr_eq_levels_ext3(pt, low, rounds, r.e.data()); });
    }

    uint64_t stored_k = shape.stored, stored_a = shape.stored;
    for (uint64_t j = 0; j < r.rounds; ++j) {
        const std::string rw = what + " round " + std::to_string(j);
        uint64_t quarter = half >> (j + 1);
        uint64_t cube = (j > 0 ? 4 : 2) * quarter;
        std::vector<uint64_t> today = launch_round(r, TODAY, r.wp.data(), r.wq.data(), j, 0);
        if (shape.before) {
            std::vector<uint64_t> before = launch_round(r, BEFORE, r.bp.data(), r.bq.data(), j, 0);
            check(before == today, rw + ": today's round = the copy of it from before, raw");
            check(r.bp == r.wp && r.bq == r.wq, rw + ": today's folded halves = the copy's, raw");
        }
        bool keeps = driver_keeps(stored_k, cube);
        kept_rounds += keeps;
        today_rounds_on_kept += !keeps;
        std::vector<uint64_t> kept =
            launch_round(r, keeps ? KEPT : TODAY, r.kp.data(), r.kq.data(), j, stored_k);
        check(kept == today, rw + ": the kept layer's partials = the whole layer's, raw");
        if (shape.every_round) {
            std::vector<uint64_t> always = launch_round(r, KEPT, r.ap.data(), r.aq.data(), j, stored_a);
            check(always == today, rw + ": the kept kernel's partials = today's, raw (every round)");
        }
        // The reference.
        Fe3 want[3];
        reference_round(r, j, want);
        unsigned grid, block;
        gruen_shape(quarter, grid, block);
        int rows = want_h0(shape, j) ? 3 : 2;
        for (int row = 0; row < rows; ++row) {
            Fe3 got = ext3::zero();
            for (unsigned b = 0; b < grid; ++b) got = ext3::add(got, at(today, row * grid + b));
            check(same_value(got, want[row]),
                  rw + ": H(" + std::to_string(row == 0 ? 1 : row == 1 ? 2 : 0) + ") = the reference's");
        }
        // The halves after the round: the live half is `2·quarter`.
        uint64_t next_k = stored_after_round(stored_k, quarter);
        uint64_t next_a = stored_after_round(stored_a, quarter);
        for (int k = 0; k < 4; ++k) {
            bool upper = k & 1;
            uint64_t live = 2 * quarter;
            uint64_t n = upper ? std::min(next_k, live) : live;
            check(same_raw(factor(r.kp, r.kq, k, half), factor(r.wp, r.wq, k, half), n),
                  rw + ": factor " + std::to_string(k) + "'s stored cells = the whole layer's, raw");
            if (shape.every_round) {
                uint64_t na = upper ? std::min(next_a, live) : live;
                check(same_raw(factor(r.ap, r.aq, k, half), factor(r.wp, r.wq, k, half), na),
                      rw + ": factor " + std::to_string(k) + "'s stored cells = the whole layer's, raw (every round)");
            }
            if (upper) {
                // Item 4: the whole layer's padding is still the raw `0/1`.
                const uint64_t *f = factor(r.wp, r.wq, k, half);
                bool limbs = true;
                for (uint64_t x = next_k; x < live; ++x)
                    limbs &= f[x * 3] == (k == 3 ? 1u : 0u) && f[x * 3 + 1] == 0 && f[x * 3 + 2] == 0;
                check(limbs, rw + ": factor " + std::to_string(k) + "'s padding is the raw 0/1 after the fold");
            }
        }
        check(poison_intact(r.kp, half, shape.stored) && poison_intact(r.kq, half, shape.stored),
              rw + ": nothing written past the real cells");
        if (shape.every_round)
            check(poison_intact(r.ap, half, shape.stored) && poison_intact(r.aq, half, shape.stored),
                  rw + ": nothing written past the real cells (every round)");
        stored_k = next_k;
        stored_a = next_a;
        ++rounds_run;
    }

    // The finish.
    {
        const std::string rw = what + " finish";
        uint64_t cells_l = 1ull << r.low;
        uint64_t cube = r.rounds > 0 ? 2 * cells_l : cells_l;
        std::vector<uint64_t> today = launch_finish(r, TODAY, r.wp.data(), r.wq.data(), 0);
        if (shape.before) {
            std::vector<uint64_t> before = launch_finish(r, BEFORE, r.bp.data(), r.bq.data(), 0);
            check(before == today && r.bp == r.wp && r.bq == r.wq,
                  rw + ": today's finish = the copy of it from before, raw");
        }
        bool keeps = driver_keeps(stored_k, cube);
        std::vector<uint64_t> kept =
            launch_finish(r, keeps ? KEPT : TODAY, r.kp.data(), r.kq.data(), stored_k);
        check(kept == today, rw + ": the kept layer's halves = the whole layer's, raw");
        if (shape.every_round) {
            std::vector<uint64_t> always = launch_finish(r, KEPT, r.ap.data(), r.aq.data(), stored_a);
            check(always == today, rw + ": the kept kernel's halves = today's, raw (every round)");
        }
        // The reference's halves.
        bool same = true;
        for (int k = 0; k < 4; ++k) {
            for (uint64_t y = 0; y < cells_l; ++y) {
                Fe3 v = r.ref[k][y];
                if (r.rounds > 0) {
                    Fe3 s = r.challenges[r.rounds - 1];
                    v = ext3::add(v, ext3::mul(s, ext3::sub(r.ref[k][y + cells_l], v)));
                }
                same &= same_value(at(today, k * cells_l + y), v);
            }
        }
        check(same, rw + ": the halves = the reference's");
        uint64_t n = std::min(stored_k, cells_l);
        for (int k = 0; k < 4; ++k) {
            uint64_t m = (k & 1) ? n : cells_l;
            check(same_raw(factor(r.kp, r.kq, k, half), factor(r.wp, r.wq, k, half), m),
                  rw + ": factor " + std::to_string(k) + "'s stored cells = the whole layer's, raw");
        }
        check(poison_intact(r.kp, half, shape.stored) && poison_intact(r.kq, half, shape.stored),
              rw + ": nothing written past the real cells");
    }
    ++layers_run;
}

}  // namespace

int main() {
    std::vector<Shape> shapes;
    // Every split of card rounds and host tail, and every H(0) mode, on small
    // layers, with the padding's start at both ends and between.
    for (unsigned m = 1; m <= 6; ++m) {
        uint64_t half = 1ull << m;
        for (uint64_t stored : {uint64_t(1), half / 2 + 1, half - 1, half}) {
            if (stored == 0 || stored > half) continue;
            for (unsigned low : {0u, 1u, 2u, m}) {
                for (int h0 : {0, 1, 2}) shapes.push_back({m, stored, low, h0, true, true});
            }
        }
    }
    // m 7..14: the padding one cell in, one cell before the end, none, a CPU
    // table's (20 of 32 slots) and a Keccak-round table's (1,031 of 2,048) —
    // each a run of whole rows. From m 12 on, where a layer is thousands of
    // host threads a round, the two production shapes alone.
    const unsigned lows[] = {0, 6, 2, 5, 9};
    unsigned n = 0;
    for (unsigned m = 7; m <= 14; ++m) {
        uint64_t half = 1ull << m, full = 2 * half;
        uint64_t cpu = (20 - 16) * (full / 32), keccak = (1031 - 1024) * (full / 2048);
        std::vector<uint64_t> starts = {cpu};
        if (full >= 2048) starts.push_back(keccak);
        if (m <= 11) {
            starts.push_back(1);
            starts.push_back(half - 1);
            starts.push_back(half);
        }
        for (uint64_t stored : starts) {
            shapes.push_back({m, stored, lows[n % 5], (int)(n % 3), m <= 10, m <= 11});
            ++n;
        }
    }
    for (const Shape &s : shapes) run_shape(s);
    printf("gkr host KAT: %llu layers (m 1..14), %llu card rounds (%llu by the kept kernel, %llu by today's "
           "once the padding was spent): today's kernels = their copy from before; kept = whole, raw; sums "
           "and halves = the written-out reference\n",
           (unsigned long long)layers_run, (unsigned long long)rounds_run,
           (unsigned long long)kept_rounds, (unsigned long long)today_rounds_on_kept);
    if (failures) {
        printf("gkr host KAT: %d FAILED\n", failures);
        return 1;
    }
    printf("gkr host KAT: all passed\n");
    return 0;
}
