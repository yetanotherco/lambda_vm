// Known-answer tests for `kernels/ntt_cm.cu` (GAP K1), run on the host.
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
//   1. the windowed root table against repeated multiplication;
//   2. every pass shape (k = 4..8, DIT and DIF, contiguous and strided, every
//      start level s) against the textbook radix-2 levels [s, s + k);
//   3. whole transforms composed of passes against the naive DFT;
//   4. the LDE the driver runs — DIF iNTT out of place, bit-reversed weights
//      on the last pass, DIT with the fused coset spread — against direct
//      interpolation and evaluation on the coset, blowups 2..16, several
//      columns with a column stride, non-canonical inputs;
//   5. the coefficient-input form (F_SPREAD | F_GATHER) against direct
//      evaluation.
//
// WHAT IT DOES NOT COVER: whether nvcc accepts the file, and everything about
// execution — grid sizing, shared-memory size, races the replay order cannot
// show, register pressure. That is the GPU parity suite's
// (`tests/ntt_cm_parity.rs`). Passing here is necessary, never sufficient.
//
// Build and run with `make test-ntt-cm-host-kat`.

#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template <int KLOG, bool DIT>
void replay_pass_k(const u64 *in, u64 in_stride, u64 *out, u64 out_stride, const u64 *roots,
                   const u64 *wtab, unsigned L, unsigned s, unsigned lb, unsigned flags,
                   unsigned cols) {
    using namespace ntt_cm;
    constexpr int ROUNDS = (KLOG + 3) / 4;
    const unsigned log_t = pass_log_t(L, s, KLOG);
    const unsigned nthreads = threads_of<KLOG>(log_t);
    const unsigned blocks = 1u << (L - KLOG - log_t);
    std::vector<u64> tile(((size_t)1 << KLOG) * (((size_t)1 << log_t) + 1));
    std::vector<u64> rad(RAD);
    for (unsigned col = 0; col < cols; ++col) {
        for (unsigned bx = 0; bx < blocks; ++bx) {
            for (unsigned tid = 0; tid < nthreads; ++tid) phase_rad(rad.data(), roots, tid, nthreads);
            if (s > 0) {
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_strided_first<KLOG, DIT>(in, in_stride, out, out_stride, roots, rad.data(),
                                                   tile.data(), L, s, log_t, tid, bx, col);
                if constexpr (ROUNDS == 2) {
                    for (unsigned tid = 0; tid < nthreads; ++tid)
                        phase_strided_second<KLOG, DIT>(out, out_stride, roots, rad.data(),
                                                        tile.data(), L, s, log_t, tid, bx, col);
                }
            } else {
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_contig_stage<KLOG>(in, in_stride, wtab, tile.data(), L, lb, log_t, flags,
                                             tid, bx, col);
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_contig_round<KLOG, DIT, 0>(tile.data(), rad.data(), log_t, tid);
                if constexpr (ROUNDS == 2) {
                    for (unsigned tid = 0; tid < nthreads; ++tid)
                        phase_contig_round<KLOG, DIT, 1>(tile.data(), rad.data(), log_t, tid);
                }
                for (unsigned tid = 0; tid < nthreads; ++tid)
                    phase_contig_store<KLOG>(out, out_stride, wtab, tile.data(), log_t, flags, tid,
                                             bx, col);
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
        for (unsigned k = 4; k <= 8 && k <= L; ++k) {
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
    for (unsigned L = 4; L <= 11; ++L) {
        const u64 n = (u64)1 << L;
        for (bool inverse : {false, true}) {
            std::vector<u64> x(n);
            for (auto &v : x) v = random_felt();
            const std::vector<u64> want = naive_dft(x, L, inverse);
            const std::vector<unsigned> ks = plan(L);

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
                CHECK(eq_canon(buf, want, &where), "DIT L=%u inv=%d first diff at %zu", L,
                      (int)inverse, where);
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
                CHECK(eq_canon(nat, want, &where), "DIF L=%u inv=%d first diff at %zu", L,
                      (int)inverse, where);
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
    if (g_failures) {
        std::printf("NTT_CM HOST KAT: FAILED (%d of %d)\n", g_failures, g_checks);
        return 1;
    }
    std::printf("NTT_CM HOST KAT: OK (%d checks)\n", g_checks);
    return 0;
}
