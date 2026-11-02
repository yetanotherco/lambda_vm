// Known-answer tests for `kernels/ntt_cm.cu`, the column-major LDE engine's
// passes, run on the host.
//
// WHY THIS EXISTS. The column-major LDE engine's correctness is index math and
// roots: which element a register holds, which root a butterfly takes, what
// the twist multiplies by. All of that is arithmetic, and this checks it
// without a GPU: the kernel source compiles through `cuda_host_shim.h`, and a
// launch is replayed phase by phase — every thread of a phase before any
// thread of the next, which is the ordering the kernel's `__syncthreads` give
// it on the device.
//
// WHAT IT COVERS, each against an independent reference (schoolbook
// `__int128` field arithmetic and naive O(n^2) transforms — no code shared
// with the kernel):
//   1. the windowed root table against repeated multiplication, and the wide
//      passes' radix tables (k = 9..12) against ω_{2^k}^j;
//   2. every pass shape (k = 4..12, DIT and DIF, contiguous and strided, every
//      start level s) against the textbook radix-2 levels [s, s + k);
//   3. whole transforms composed of passes against the naive DFT;
//   4. the LDE the driver runs — DIF iNTT out of place, bit-reversed weights
//      on the last pass, DIT with the fused coset spread — against direct
//      interpolation and evaluation on the coset, blowups 2..16, several
//      columns with a column stride, non-canonical inputs;
//   5. the coefficient-input form (F_SPREAD | F_GATHER) against direct
//      evaluation;
//   6. the single-column spread transform of WHIR's encoding (spacing 1..16,
//      no weights) against the naive DFT;
//   7. the same transform in the wide plan (`lde_cm::plan_wide`, passes of up
//      to 10 levels in three register rounds) against the naive DFT on small
//      sizes, and against today's plan — the oracle — at every size up to 2^20
//      where the two plans differ, edge words planted;
//   8. the tile layout: every slot of every (k, T) in bounds and distinct, and
//      a wide pass's first-round and staged accesses on 16 distinct banks a
//      half-warp (a speed property, checked because the layout exists for it);
//   9. the Möbius levels fused into the first pass (F_MOBIUS): the encoding's
//      other Möbius levels on the host, then the fused transform, against the
//      whole Möbius transform on the host then today's spread transform, in
//      both plans, spacings 1..16, edge words planted.
//
// WHAT IT DOES NOT COVER: whether nvcc accepts the file, and everything about
// execution — grid sizing, shared-memory size, races the replay order cannot
// show, register pressure. That is the GPU parity suite's
// (`tests/ntt_cm_parity.rs`). Passing here is necessary, never sufficient.
//
// Build and run with `make test-ntt-cm-host-kat`.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include "cuda_host_shim.h"

#include "ntt_cm.cu"

namespace {

using u64 = uint64_t;
using u128 = unsigned __int128;

constexpr u64 P = 0xFFFFFFFF00000001ull;
constexpr u64 ROOT32 = 1753635133440165772ull;  // TWO_ADIC_PRIMITVE_ROOT_OF_UNITY

int g_failures = 0;
int g_checks = 0;

#define CHECK(cond, ...)                                          \
    do {                                                          \
        ++g_checks;                                               \
        if (!(cond)) {                                            \
            ++g_failures;                                         \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__);      \
            std::printf(__VA_ARGS__);                             \
            std::printf("\n");                                    \
        }                                                         \
    } while (0)

// Schoolbook field arithmetic — the definition, independent of goldilocks.cuh.
u64 canon(u64 a) { return a >= P ? a - P : a; }
u64 fadd(u64 a, u64 b) { return (u64)(((u128)canon(a) + canon(b)) % P); }
u64 fsub(u64 a, u64 b) { return (u64)(((u128)canon(a) + P - canon(b)) % P); }
u64 fmul(u64 a, u64 b) { return (u64)(((u128)canon(a) * canon(b)) % P); }
u64 fpow(u64 b, u64 e) {
    u64 r = 1;
    b = canon(b);
    while (e) {
        if (e & 1) r = fmul(r, b);
        b = fmul(b, b);
        e >>= 1;
    }
    return r;
}
u64 finv(u64 a) { return fpow(a, P - 2); }

// ω_{2^m} = ROOT32^(2^(32-m)), by squaring — the math crate's own derivation.
u64 omega(unsigned m, bool inverse) {
    u64 r = ROOT32;
    for (unsigned i = 0; i < 32 - m; ++i) r = fmul(r, r);
    return inverse ? finv(r) : r;
}

std::vector<u64> root_table(bool inverse) {
    const u64 root = inverse ? finv(ROOT32) : ROOT32;
    std::vector<u64> t(1024);
    for (int w = 0; w < 4; ++w) {
        const u64 base = fpow(root, (u64)1 << (8 * w));
        u64 acc = 1;
        for (int j = 0; j < 256; ++j) {
            t[w * 256 + j] = acc;
            acc = fmul(acc, base);
        }
    }
    return t;
}

u64 rev(u64 x, unsigned bits) {
    u64 r = 0;
    for (unsigned i = 0; i < bits; ++i) r |= ((x >> i) & 1) << (bits - 1 - i);
    return r;
}

// Deterministic, deliberately non-canonical inputs: a slice lands in [P, 2^64).
u64 g_rng = 0x9E3779B97F4A7C15ull;
u64 next_u64() {
    g_rng ^= g_rng << 13;
    g_rng ^= g_rng >> 7;
    g_rng ^= g_rng << 17;
    return g_rng;
}
u64 random_felt() {
    const u64 r = next_u64();
    return (r & 7) == 0 ? P + (r % (0xFFFFFFFFull)) : r;
}

// Textbook levels [s, s + k) of a length-2^L transform, in place.
void textbook_levels(std::vector<u64> &x, unsigned L, unsigned s, unsigned k, bool dit,
                     bool inverse) {
    const u64 n = (u64)1 << L;
    auto one_level = [&](unsigned u) {
        const u64 half = (u64)1 << u;
        const u64 w1 = omega(u + 1, inverse);
        std::vector<u64> pw(half);
        u64 acc = 1;
        for (u64 j = 0; j < half; ++j) {
            pw[j] = acc;
            acc = fmul(acc, w1);
        }
        for (u64 i = 0; i < n; ++i) {
            if (i & half) continue;
            const u64 tw = pw[i & (half - 1)];
            const u64 a = x[i], b = x[i + half];
            if (dit) {
                const u64 t = fmul(b, tw);
                x[i] = fadd(a, t);
                x[i + half] = fsub(a, t);
            } else {
                x[i] = fadd(a, b);
                x[i + half] = fmul(fsub(a, b), tw);
            }
        }
    };
    if (dit) {
        for (unsigned u = s; u < s + k; ++u) one_level(u);
    } else {
        for (unsigned u = s + k; u-- > s;) one_level(u);
    }
}

// Naive DFT: out[j] = Σ_i in[i] ω^(ij).
std::vector<u64> naive_dft(const std::vector<u64> &in, unsigned L, bool inverse) {
    const u64 n = (u64)1 << L;
    const u64 w = omega(L, inverse);
    std::vector<u64> out(n);
    for (u64 j = 0; j < n; ++j) {
        const u64 wj = fpow(w, j);
        u64 acc = 0, pw = 1;
        for (u64 i = 0; i < n; ++i) {
            acc = fadd(acc, fmul(in[i], pw));
            pw = fmul(pw, wj);
        }
        out[j] = acc;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Launch replay: one pass over `cols` columns, exactly the geometry the Rust
// driver uses (see `lde_cm.rs`): T = min(4096 >> k, 2^s) for strided passes,
// T = min(4096 >> k, 2^(L - k)) for contiguous ones.
// ---------------------------------------------------------------------------

unsigned pass_log_t(unsigned L, unsigned s, unsigned k) {
    const unsigned cap = 12 - k;  // log2(4096 >> k)
    const unsigned lim = s > 0 ? s : L - k;
    return cap < lim ? cap : lim;
}

// Words past the tile that no phase may write (checked after every block).
constexpr size_t TILE_GUARD = 64;
constexpr u64 GUARD_WORD = 0xA5A5A5A5DEADF00Dull;

template <int KLOG, bool DIT>
void replay_pass_k(const u64 *in, u64 in_stride, u64 *out, u64 out_stride, const u64 *roots,
                   const u64 *wtab, unsigned L, unsigned s, unsigned lb, unsigned flags,
                   unsigned cols) {
    using namespace ntt_cm;
    constexpr int ROUNDS = (KLOG + 3) / 4;
    const unsigned log_t = pass_log_t(L, s, KLOG);
    const unsigned nthreads = threads_of<KLOG>(log_t);
    const unsigned blocks = 1u << (L - KLOG - log_t);
    const size_t words = tile_words<KLOG>(log_t);
    std::vector<u64> tile(words + TILE_GUARD, GUARD_WORD);
    std::vector<u64> rad(Radix<KLOG>::SIZE);
    for (unsigned col = 0; col < cols; ++col) {
        for (unsigned bx = 0; bx < blocks; ++bx) {
            for (unsigned tid = 0; tid < nthreads; ++tid)
                phase_rad<KLOG>(rad.data(), roots, tid, nthreads);
            if (s > 0) {
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_strided_first<KLOG, DIT>(in, in_stride, out, out_stride, roots, rad.data(),
                                                   tile.data(), L, s, log_t, tid, bx, col);
                if constexpr (ROUNDS == 3) {
                    for (unsigned tid = 0; tid < nthreads; ++tid)
                        phase_tile_round<KLOG, DIT, 1>(tile.data(), rad.data(), log_t, tid);
                }
                if constexpr (ROUNDS >= 2) {
                    for (unsigned tid = 0; tid < nthreads; ++tid)
                        phase_strided_last<KLOG, DIT>(out, out_stride, roots, rad.data(),
                                                      tile.data(), L, s, log_t, tid, bx, col);
                }
            } else {
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_contig_stage<KLOG>(in, in_stride, wtab, tile.data(), L, lb, log_t, flags,
                                             tid, bx, col);
                if (flags & F_MOBIUS) {
                    for (unsigned l = 0; l < KLOG + log_t - lb; ++l)
                        for (unsigned tid = 0; tid < nthreads; ++tid)
                            phase_contig_mobius<KLOG>(tile.data(), log_t, lb, l, tid);
                }
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_tile_round<KLOG, DIT, 0>(tile.data(), rad.data(), log_t, tid);
                if constexpr (ROUNDS >= 2) {
                    for (unsigned tid = 0; tid < nthreads; ++tid)
                        phase_tile_round<KLOG, DIT, 1>(tile.data(), rad.data(), log_t, tid);
                }
                if constexpr (ROUNDS == 3) {
                    for (unsigned tid = 0; tid < nthreads; ++tid)
                        phase_tile_round<KLOG, DIT, 2>(tile.data(), rad.data(), log_t, tid);
                }
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_contig_store<KLOG>(out, out_stride, wtab, tile.data(), log_t, flags, tid,
                                             bx, col);
            }
            bool guard_ok = true;
            for (size_t i = words; i < words + TILE_GUARD; ++i) guard_ok &= tile[i] == GUARD_WORD;
            if (!guard_ok) {
                CHECK(false, "k=%d L=%u s=%u: a phase wrote past the tile's %zu words", KLOG, L, s,
                      words);
                std::fill(tile.begin() + words, tile.end(), GUARD_WORD);
            }
        }
    }
}

void replay_pass(unsigned k, bool dit, const u64 *in, u64 in_stride, u64 *out, u64 out_stride,
                 const u64 *roots, const u64 *wtab, unsigned L, unsigned s, unsigned lb,
                 unsigned flags, unsigned cols) {
#define DISPATCH(KK)                                                                               \
    case KK:                                                                                       \
        if (dit)                                                                                   \
            replay_pass_k<KK, true>(in, in_stride, out, out_stride, roots, wtab, L, s, lb, flags,  \
                                    cols);                                                         \
        else                                                                                       \
            replay_pass_k<KK, false>(in, in_stride, out, out_stride, roots, wtab, L, s, lb, flags, \
                                     cols);                                                        \
        return;
    switch (k) {
        DISPATCH(4)
        DISPATCH(5)
        DISPATCH(6)
        DISPATCH(7)
        DISPATCH(8)
        DISPATCH(9)
        DISPATCH(10)
        DISPATCH(11)
        DISPATCH(12)
        default:
            std::printf("bad k %u\n", k);
            std::exit(2);
    }
#undef DISPATCH
}

// The driver's pass plan (mirrors `lde_cm::plan`): ceil(L/8) passes, sizes as
// equal as possible.
std::vector<unsigned> plan(unsigned L) {
    const unsigned p = (L + 7) / 8;
    std::vector<unsigned> ks(p, L / p);
    for (unsigned i = 0; i < L % p; ++i) ks[i] += 1;
    return ks;
}

// The encoding's wide plan (mirrors `lde_cm::plan_wide`): ceil(L/10) passes,
// sizes as equal as possible — [10, 9, 9] at the production 2^28.
std::vector<unsigned> plan_wide(unsigned L) {
    const unsigned p = (L + 9) / 10;
    std::vector<unsigned> ks(p, L / p);
    for (unsigned i = 0; i < L % p; ++i) ks[i] += 1;
    return ks;
}

bool eq_canon(const std::vector<u64> &a, const std::vector<u64> &b, size_t *where) {
    if (a.size() != b.size()) {
        *where = (size_t)-1;
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (canon(a[i]) != canon(b[i])) {
            *where = i;
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// 1. Windowed roots.
// ---------------------------------------------------------------------------
void test_root_table() {
    for (bool inverse : {false, true}) {
        const std::vector<u64> t = root_table(inverse);
        const u64 root = inverse ? finv(ROOT32) : ROOT32;
        for (int i = 0; i < 2000; ++i) {
            const uint32_t e = (uint32_t)next_u64();
            CHECK(canon(ntt_cm::pow_root32(t.data(), e)) == fpow(root, e),
                  "pow_root32 inverse=%d e=%u", (int)inverse, e);
        }
        CHECK(canon(ntt_cm::pow_root32(t.data(), 0)) == 1, "pow_root32(0) != 1");
        // The top window is the radix table ω_256^j.
        const u64 w256 = omega(8, inverse);
        for (int j = 0; j < 128; ++j)
            CHECK(t[768 + j] == fpow(w256, j), "radix table entry %d", j);
        // A wide pass's radix table, built from the windows: ω_{2^k}^j.
        auto wide = [&](auto kk) {
            constexpr int K = decltype(kk)::value;
            using RX = ntt_cm::Radix<K>;
            static_assert(RX::LOG == K - 1, "a wide pass takes ω_{2^k}");
            std::vector<u64> rad(RX::SIZE, 0);
            for (uint32_t tid = 0; tid < 256; ++tid)
                ntt_cm::phase_rad<K>(rad.data(), t.data(), tid, 256);
            const u64 wk = omega(K, inverse);
            for (int j = 0; j < RX::SIZE; ++j)
                CHECK(canon(rad[j]) == fpow(wk, j), "k=%d inverse=%d radix entry %d", K,
                      (int)inverse, j);
        };
        wide(std::integral_constant<int, 9>{});
        wide(std::integral_constant<int, 10>{});
        wide(std::integral_constant<int, 11>{});
        wide(std::integral_constant<int, 12>{});
        static_assert(ntt_cm::Radix<8>::SIZE == 128 && ntt_cm::Radix<4>::SIZE == 128,
                      "every k <= 8 keeps today's table");
    }
    // ω_{2^32} really has order 2^32 (and not 2^31).
    CHECK(fpow(ROOT32, (u64)1 << 31) == P - 1, "ROOT32^(2^31) != -1");
}

// ---------------------------------------------------------------------------
// 2. Every pass shape against the textbook levels.
// ---------------------------------------------------------------------------
void test_single_passes() {
    const std::vector<u64> fwd = root_table(false), inv = root_table(true);
    for (unsigned L = 4; L <= 14; ++L) {
        for (unsigned k = 4; k <= 12 && k <= L; ++k) {
            for (unsigned s = 0; s + k <= L; ++s) {
                // Strided passes need a warp's worth of lo only for speed; the
                // math holds for any s >= log T, which the plan guarantees.
                for (bool dit : {true, false}) {
                    for (bool inverse : {false, true}) {
                        const unsigned cols = 2;
                        const u64 n = (u64)1 << L;
                        const u64 stride = n + 3;  // a column stride that is not the length
                        std::vector<u64> buf(stride * cols);
                        for (auto &v : buf) v = random_felt();
                        std::vector<std::vector<u64>> want(cols);
                        for (unsigned c = 0; c < cols; ++c) {
                            want[c].assign(buf.begin() + c * stride, buf.begin() + c * stride + n);
                            textbook_levels(want[c], L, s, k, dit, inverse);
                        }
                        // Out of place for the strided form (as the iNTT's
                        // first pass runs), in place for the contiguous one.
                        std::vector<u64> out = s > 0 ? std::vector<u64>(stride * cols, 0) : buf;
                        replay_pass(k, dit, buf.data(), stride, out.data(), stride,
                                    inverse ? inv.data() : fwd.data(), nullptr, L, s, 0, 0, cols);
                        for (unsigned c = 0; c < cols; ++c) {
                            std::vector<u64> got(out.begin() + c * stride,
                                                 out.begin() + c * stride + n);
                            size_t where = 0;
                            CHECK(eq_canon(got, want[c], &where),
                                  "pass L=%u s=%u k=%u dit=%d inv=%d col=%u first diff at %zu", L, s,
                                  k, (int)dit, (int)inverse, c, where);
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Whole transforms composed of passes against the naive DFT.
// ---------------------------------------------------------------------------
void test_whole_transforms() {
    const std::vector<u64> fwd = root_table(false), inv = root_table(true);
    for (unsigned L = 4; L <= 12; ++L) {
        const u64 n = (u64)1 << L;
        // Today's plan and the encoding's wide one, forward and inverse.
        for (int variant = 0; variant < 4; ++variant) {
            const bool wide = variant & 2, inverse = variant & 1;
            std::vector<u64> x(n);
            for (auto &v : x) v = random_felt();
            const std::vector<u64> want = naive_dft(x, L, inverse);
            const std::vector<unsigned> ks = wide ? plan_wide(L) : plan(L);

            // DIT: bit-reversed input, passes ascending from s = 0.
            {
                std::vector<u64> buf(n);
                for (u64 i = 0; i < n; ++i) buf[i] = x[rev(i, L)];
                unsigned s = 0;
                for (unsigned k : ks) {
                    replay_pass(k, true, buf.data(), n, buf.data(), n,
                                inverse ? inv.data() : fwd.data(), nullptr, L, s, 0, 0, 1);
                    s += k;
                }
                size_t where = 0;
                CHECK(eq_canon(buf, want, &where), "DIT L=%u wide=%d inv=%d first diff at %zu", L,
                      (int)wide, (int)inverse, where);
            }
            // DIF: natural input, passes descending to s = 0, bit-reversed out.
            {
                std::vector<u64> buf = x;
                unsigned s = L;
                for (unsigned k : ks) {
                    s -= k;
                    replay_pass(k, false, buf.data(), n, buf.data(), n,
                                inverse ? inv.data() : fwd.data(), nullptr, L, s, 0, 0, 1);
                }
                std::vector<u64> nat(n);
                for (u64 i = 0; i < n; ++i) nat[rev(i, L)] = buf[i];
                size_t where = 0;
                CHECK(eq_canon(nat, want, &where), "DIF L=%u wide=%d inv=%d first diff at %zu", L,
                      (int)wide, (int)inverse, where);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 4. The LDE as the driver runs it, and 5. the coefficient-input form.
// ---------------------------------------------------------------------------

// weights[j] = g^j / n (the prover's `coset_weights`).
std::vector<u64> coset_weights(u64 n, u64 g, bool with_inv_n) {
    std::vector<u64> w(n);
    u64 acc = with_inv_n ? finv(n % P) : 1;
    for (u64 j = 0; j < n; ++j) {
        w[j] = acc;
        acc = fmul(acc, g);
    }
    return w;
}

// Direct: interpolate (naive inverse DFT / n), evaluate at g·ω_N^i.
std::vector<u64> naive_lde(const std::vector<u64> &evals, unsigned lb, u64 g) {
    const u64 n = evals.size();
    unsigned log_n = 0;
    while (((u64)1 << log_n) < n) ++log_n;
    const unsigned log_N = log_n + lb;
    const u64 N = (u64)1 << log_N;
    std::vector<u64> c = naive_dft(evals, log_n, true);
    const u64 inv_n = finv(n % P);
    for (auto &v : c) v = fmul(v, inv_n);
    std::vector<u64> out(N);
    const u64 wN = omega(log_N, false);
    for (u64 i = 0; i < N; ++i) {
        const u64 x = fmul(g, fpow(wN, i));
        u64 acc = 0, pw = 1;
        for (u64 j = 0; j < n; ++j) {
            acc = fadd(acc, fmul(c[j], pw));
            pw = fmul(pw, x);
        }
        out[i] = acc;
    }
    return out;
}

// Run the driver's sequence on host: DIF iNTT from `src` (column stride
// src_stride) into W (stride n), last pass weighted by wbr; DIT from W with the
// coset spread into dst (stride N).
void run_lde(const std::vector<u64> &src, u64 src_stride, std::vector<u64> &dst, unsigned cols,
             unsigned log_n, unsigned lb, const std::vector<u64> &weights) {
    const std::vector<u64> fwd = root_table(false), inv = root_table(true);
    const u64 n = (u64)1 << log_n;
    const unsigned log_N = log_n + lb;
    const u64 N = (u64)1 << log_N;
    std::vector<u64> wbr(n);
    for (u64 p = 0; p < n; ++p) wbr[p] = weights[rev(p, log_n)];
    std::vector<u64> W(n * cols, 0xDEADBEEFull);

    const std::vector<unsigned> ki = plan(log_n);
    unsigned s = log_n;
    for (size_t i = 0; i < ki.size(); ++i) {
        s -= ki[i];
        const bool first = i == 0, last = i + 1 == ki.size();
        replay_pass(ki[i], false, first ? src.data() : W.data(), first ? src_stride : n, W.data(),
                    n, inv.data(), wbr.data(), log_n, s, 0, last ? ntt_cm::F_STORE_W : 0, cols);
    }
    const std::vector<unsigned> kf = plan(log_N);
    s = 0;
    for (size_t i = 0; i < kf.size(); ++i) {
        const bool first = i == 0;
        replay_pass(kf[i], true, first ? W.data() : dst.data(), first ? n : N, dst.data(), N,
                    fwd.data(), nullptr, log_N, s, lb, first ? ntt_cm::F_SPREAD : 0, cols);
        s += kf[i];
    }
}

void test_lde() {
    const u64 g = 7;
    struct Shape {
        unsigned log_n, lb, cols;
    };
    const Shape shapes[] = {
        {4, 1, 1},  {4, 2, 2}, {5, 1, 3}, {6, 3, 2}, {7, 4, 1}, {8, 1, 2},
        {9, 2, 2},  {10, 1, 1}, {10, 3, 1}, {11, 2, 1}, {9, 4, 1},
    };
    for (const Shape &sh : shapes) {
        const u64 n = (u64)1 << sh.log_n;
        const u64 N = n << sh.lb;
        const u64 src_stride = n + 5;
        std::vector<u64> src(src_stride * sh.cols);
        for (auto &v : src) v = random_felt();
        const std::vector<u64> weights = coset_weights(n, g, true);
        std::vector<u64> dst(N * sh.cols, 0xBADC0FFEEull);
        const std::vector<u64> src_before = src;
        run_lde(src, src_stride, dst, sh.cols, sh.log_n, sh.lb, weights);
        for (unsigned c = 0; c < sh.cols; ++c) {
            std::vector<u64> evals(src.begin() + c * src_stride, src.begin() + c * src_stride + n);
            const std::vector<u64> want = naive_lde(evals, sh.lb, g);
            std::vector<u64> got(dst.begin() + c * N, dst.begin() + (c + 1) * N);
            size_t where = 0;
            CHECK(eq_canon(got, want, &where), "LDE log_n=%u lb=%u col=%u first diff at %zu",
                  sh.log_n, sh.lb, c, where);
        }
        // The source survives, raw u64 for raw u64: the iNTT's first pass
        // reads it and writes the scratch, which is what lets the main commit
        // keep its trace-domain snapshot.
        CHECK(src == src_before, "LDE log_n=%u lb=%u clobbered its source", sh.log_n, sh.lb);
    }
}

void test_coeff_form() {
    const u64 offset = 7;
    for (unsigned log_n = 4; log_n <= 9; ++log_n) {
        for (unsigned lb = 1; lb <= 3; ++lb) {
            const u64 n = (u64)1 << log_n;
            const unsigned log_N = log_n + lb;
            const u64 N = (u64)1 << log_N;
            const unsigned cols = 2;
            std::vector<u64> coef(n * cols);
            for (auto &v : coef) v = random_felt();
            const std::vector<u64> weights = coset_weights(n, offset, false);
            std::vector<u64> wbr(n);
            for (u64 p = 0; p < n; ++p) wbr[p] = weights[rev(p, log_n)];
            std::vector<u64> dst(N * cols, 0);
            const std::vector<u64> fwd = root_table(false);
            const std::vector<unsigned> kf = plan(log_N);
            unsigned s = 0;
            for (size_t i = 0; i < kf.size(); ++i) {
                const bool first = i == 0;
                replay_pass(kf[i], true, first ? coef.data() : dst.data(), first ? n : N,
                            dst.data(), N, fwd.data(), wbr.data(), log_N, s, lb,
                            first ? (ntt_cm::F_SPREAD | ntt_cm::F_GATHER) : 0, cols);
                s += kf[i];
            }
            const u64 wN = omega(log_N, false);
            for (unsigned c = 0; c < cols; ++c) {
                std::vector<u64> want(N);
                for (u64 i = 0; i < N; ++i) {
                    const u64 x = fmul(offset, fpow(wN, i));
                    u64 acc = 0, pw = 1;
                    for (u64 j = 0; j < n; ++j) {
                        acc = fadd(acc, fmul(coef[c * n + j], pw));
                        pw = fmul(pw, x);
                    }
                    want[i] = acc;
                }
                std::vector<u64> got(dst.begin() + c * N, dst.begin() + (c + 1) * N);
                size_t where = 0;
                CHECK(eq_canon(got, want, &where), "coeff form log_n=%u lb=%u col=%u diff at %zu",
                      log_n, lb, c, where);
            }
        }
    }
}

// 6. The single-column spread transform the WHIR commit's encoding runs
// (`lde_cm::spread_ntt_column`): position B·q of the DIT input is src[q], the
// rest zero, no weights — the legacy `lift_spread` then the DIT body. A DIT with
// bit-reversed input position p computes Σ_p z[p]·ω^(i·rev_N(p)), and
// rev_N(B·q) = rev_n(q), so the answer is the DFT of y[j] = src[rev_n(j)] (j < n).
void test_spread_transform() {
    const std::vector<u64> fwd = root_table(false);
    for (unsigned log_n = 4; log_n <= 11; ++log_n) {
        for (unsigned lb = 0; lb <= 4 && lb <= log_n; ++lb) {
            const u64 N = (u64)1 << log_n;
            const unsigned log_c = log_n - lb;
            const u64 n = (u64)1 << log_c;
            std::vector<u64> src(n);
            for (auto &v : src) v = random_felt();
            std::vector<u64> dst(N, 0xFEEDFACEull);
            const std::vector<unsigned> ks = plan(log_n);
            unsigned s = 0;
            for (size_t i = 0; i < ks.size(); ++i) {
                const bool first = i == 0;
                replay_pass(ks[i], true, first ? src.data() : dst.data(), 0, dst.data(), 0, fwd.data(),
                            nullptr, log_n, s, lb, first ? ntt_cm::F_SPREAD : 0, 1);
                s += ks[i];
            }
            std::vector<u64> y(N, 0);
            for (u64 j = 0; j < n; ++j) y[j] = src[log_c ? rev(j, log_c) : 0];
            const std::vector<u64> want = naive_dft(y, log_n, false);
            size_t where = 0;
            CHECK(eq_canon(dst, want, &where), "spread transform log_n=%u lb=%u first diff at %zu", log_n,
                  lb, where);
        }
    }
}

// The spread transform on one column in a given plan, as `spread_ntt_column`
// launches it, the first pass's extra flags (F_MOBIUS) or-ed in.
std::vector<u64> spread_transform(const std::vector<u64> &src, unsigned log_n, unsigned lb,
                                  const std::vector<unsigned> &ks, unsigned first_flags = 0) {
    const std::vector<u64> fwd = root_table(false);
    std::vector<u64> dst((u64)1 << log_n, 0xFEEDFACEull);
    unsigned s = 0;
    for (size_t i = 0; i < ks.size(); ++i) {
        const bool first = i == 0;
        replay_pass(ks[i], true, first ? src.data() : dst.data(), 0, dst.data(), 0, fwd.data(),
                    nullptr, log_n, s, lb, first ? (ntt_cm::F_SPREAD | first_flags) : 0, 1);
        s += ks[i];
    }
    return dst;
}

// Möbius levels [from, to) on the host, the definition: every index with the
// level's bit set loses its partner below.
void host_mobius(std::vector<u64> &x, unsigned from, unsigned to) {
    for (unsigned l = from; l < to; ++l) {
        const u64 h = (u64)1 << l;
        for (u64 i = 0; i < x.size(); ++i)
            if (i & h) x[i] = fsub(x[i], x[i ^ h]);
    }
}

// Raw words with an edge value at every seventh slot (the card test's mix).
std::vector<u64> edge_words(u64 n) {
    static const u64 EDGES[8] = {0, 1, P - 1, P, P + 1, ~0ull, 1ull << 63, 0xFFFFFFFFull};
    std::vector<u64> v(n);
    for (u64 i = 0; i < n; ++i) v[i] = i % 7 == 0 ? EDGES[(i / 7) % 8] : next_u64();
    return v;
}

// 7. The wide plan: against the naive DFT where that is cheap, and against
// today's plan — the 4..8-level passes the encoding runs today, themselves
// pinned by 2. and 6. — at every size up to 2^20 where the plans differ.
void test_wide_spread_transform() {
    for (unsigned log_n = 4; log_n <= 12; ++log_n) {
        for (unsigned lb = 0; lb <= 4 && lb <= log_n; ++lb) {
            const unsigned log_c = log_n - lb;
            const u64 n = (u64)1 << log_c, N = (u64)1 << log_n;
            const std::vector<u64> src = edge_words(n);
            const std::vector<u64> got = spread_transform(src, log_n, lb, plan_wide(log_n));
            std::vector<u64> y(N, 0);
            for (u64 j = 0; j < n; ++j) y[j] = src[log_c ? rev(j, log_c) : 0];
            size_t where = 0;
            CHECK(eq_canon(got, naive_dft(y, log_n, false), &where),
                  "wide spread transform log_n=%u lb=%u first diff at %zu", log_n, lb, where);
        }
    }
    unsigned differing = 0;
    for (unsigned log_n = 4; log_n <= 20; ++log_n) {
        const std::vector<unsigned> today = plan(log_n), wide = plan_wide(log_n);
        if (today == wide) continue;
        ++differing;
        for (unsigned lb : {0u, 1u, 2u, 4u}) {
            const u64 n = (u64)1 << (log_n - lb);
            const std::vector<u64> src = edge_words(n);
            const std::vector<u64> want = spread_transform(src, log_n, lb, today);
            const std::vector<u64> got = spread_transform(src, log_n, lb, wide);
            size_t where = 0;
            CHECK(eq_canon(got, want, &where),
                  "wide vs today's plan log_n=%u lb=%u (%zu vs %zu passes) first diff at %zu",
                  log_n, lb, wide.size(), today.size(), where);
        }
    }
    // 9, 10 and 17..20 (the production 2^28's windows, (0, 10), (10, 9) and
    // (19, 9), are pinned per pass by 2. and on the card).
    CHECK(differing == 6, "the plans differ at %u sizes up to 2^20, expected 6", differing);
    CHECK(plan_wide(28) == std::vector<unsigned>({10, 9, 9}), "plan_wide(28) is not [10, 9, 9]");
    CHECK(plan(28) == std::vector<unsigned>({7, 7, 7, 7}), "plan(28) is not [7, 7, 7, 7]");
}

// 8. The tile layout.
template <int KLOG>
void tile_layout_k() {
    using namespace ntt_cm;
    for (unsigned log_t = 0; log_t + KLOG <= 12; ++log_t) {
        const uint32_t words = tile_words<KLOG>(log_t);
        std::vector<uint8_t> seen(words, 0);
        bool ok = true;
        for (uint32_t mid = 0; mid < (1u << KLOG); ++mid)
            for (uint32_t t = 0; t < (1u << log_t); ++t) {
                const uint32_t i = tile_at<KLOG>(mid, log_t, t);
                if (i >= words || seen[i]) ok = false;
                else seen[i] = 1;
            }
        CHECK(ok, "k=%d log_t=%u: a tile slot out of bounds or shared", KLOG, log_t);
    }
    if constexpr (KLOG > 8 && KLOG < 12) {
        // At the driver's lane count T = 4096 >> k, a half-warp is 16 / T
        // groups of T lanes. 64-bit words on 32 four-byte banks: a half-warp's
        // 16 words are conflict-free iff they are distinct mod 16. (At k = 12
        // the single lane makes the pitch even; no kernel instantiates it.)
        const uint32_t log_t = 12 - KLOG, T = 1u << log_t;
        auto distinct16 = [](const uint32_t (&a)[16]) {
            bool hit[16] = {};
            for (uint32_t x : a) {
                if (hit[x % 16]) return false;
                hit[x % 16] = true;
            }
            return true;
        };
        bool first = true, staged = true;
        for (uint32_t h = 0; h < (threads_of<KLOG>(log_t) >> 4); ++h)
            for (int v = 0; v < E; ++v) {
                uint32_t a[16];
                for (uint32_t l = 0; l < 16; ++l) {
                    const uint32_t tid = h * 16 + l, t = tid & (T - 1), g = tid >> log_t;
                    a[l] = tile_at<KLOG>(mid_base<0>(g) | (uint32_t)v, log_t, t);
                }
                first &= distinct16(a);
            }
        for (uint32_t e0 = 0; e0 < ((1u << KLOG) << log_t); e0 += 16) {
            uint32_t a[16];
            for (uint32_t l = 0; l < 16; ++l) {
                const uint32_t e = e0 + l;
                a[l] = tile_at<KLOG>(e & ((1u << KLOG) - 1u), log_t, e >> KLOG);
            }
            staged &= distinct16(a);
        }
        CHECK(first, "k=%d: the first round's half-warp shares a bank", KLOG);
        CHECK(staged, "k=%d: the staged copy's half-warp shares a bank", KLOG);
    }
}

// 9. F_MOBIUS: the first pass's block holds the low F = k + log T - lb bits of
// a compact index, so the host runs levels [F, log_c) and the pass levels
// [0, F); the oracle runs every level, then today's transform.
void test_fused_mobius() {
    for (unsigned log_n = 4; log_n <= 20; ++log_n) {
        for (unsigned lb : {0u, 1u, 2u, 4u}) {
            if (lb > log_n || (log_n > 13 && log_n < 17 && lb != 1)) continue;
            const unsigned log_c = log_n - lb;
            const std::vector<u64> src = edge_words((u64)1 << log_c);
            std::vector<u64> full = src;
            host_mobius(full, 0, log_c);
            const std::vector<u64> want = spread_transform(full, log_n, lb, plan(log_n));
            for (bool wide : {false, true}) {
                const std::vector<unsigned> ks = wide ? plan_wide(log_n) : plan(log_n);
                const unsigned f = ks[0] + pass_log_t(log_n, 0, ks[0]) - lb;
                std::vector<u64> part = src;
                host_mobius(part, f, log_c);
                const std::vector<u64> got = spread_transform(part, log_n, lb, ks, ntt_cm::F_MOBIUS);
                size_t where = 0;
                CHECK(eq_canon(got, want, &where),
                      "fused Möbius log_n=%u lb=%u wide=%d (%u levels fused) first diff at %zu", log_n,
                      lb, (int)wide, f, where);
                if (log_n >= 12) CHECK(f == 12 - lb, "log_n=%u lb=%u: %u levels fused, not 12 - lb", log_n, lb, f);
            }
        }
    }
    // The fused levels are load-bearing: without the flag the same input is not the codeword.
    const unsigned log_n = 13, lb = 1;
    const std::vector<u64> src = edge_words((u64)1 << (log_n - lb));
    std::vector<u64> full = src, part = src;
    host_mobius(full, 0, log_n - lb);
    host_mobius(part, 12 - lb, log_n - lb);
    size_t where = 0;
    CHECK(!eq_canon(spread_transform(part, log_n, lb, plan(log_n)),
                    spread_transform(full, log_n, lb, plan(log_n)), &where),
          "the unfused partial transform equals the codeword: the check cannot fail");
}

void test_tile_layout() {
    tile_layout_k<4>();
    tile_layout_k<7>();
    tile_layout_k<8>();
    tile_layout_k<9>();
    tile_layout_k<10>();
    tile_layout_k<11>();
    tile_layout_k<12>();
    // Today's layout for every k <= 8: K x (T + 1), unchanged.
    CHECK(ntt_cm::tile_words<7>(5) == 128u * 33u, "k=7 tile is not today's");
    CHECK(ntt_cm::tile_at<8>(17, 4, 3) == 17u * 17u + 3u, "k=8 slot is not today's");
}

}  // namespace

int main() {
    test_root_table();
    std::printf("roots: %d checks, %d failures\n", g_checks, g_failures);
    test_single_passes();
    std::printf("+ single passes: %d checks, %d failures\n", g_checks, g_failures);
    test_whole_transforms();
    std::printf("+ whole transforms: %d checks, %d failures\n", g_checks, g_failures);
    test_lde();
    std::printf("+ LDE: %d checks, %d failures\n", g_checks, g_failures);
    test_coeff_form();
    std::printf("+ coefficient form: %d checks, %d failures\n", g_checks, g_failures);
    test_spread_transform();
    std::printf("+ spread transform: %d checks, %d failures\n", g_checks, g_failures);
    test_wide_spread_transform();
    std::printf("+ wide spread transform: %d checks, %d failures\n", g_checks, g_failures);
    test_tile_layout();
    std::printf("+ tile layout: %d checks, %d failures\n", g_checks, g_failures);
    test_fused_mobius();
    std::printf("+ fused Möbius: %d checks, %d failures\n", g_checks, g_failures);
    if (g_failures) {
        std::printf("NTT_CM HOST KAT: FAILED (%d of %d)\n", g_failures, g_checks);
        return 1;
    }
    std::printf("NTT_CM HOST KAT: OK (%d checks)\n", g_checks);
    return 0;
}
