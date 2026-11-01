// Column-major coset LDE engine: the pass kernels.
//
// WHY THIS EXISTS. The row-major LDE in `ntt.cu` runs one butterfly level per
// launch over the whole matrix, so an LDE of a 2^22 x 245 table makes ~33
// whole-matrix DRAM passes at the DRAM roof. This engine transforms a few
// columns at a time — a chunk sized to stay resident in L2 — and runs 4..8
// levels per launch out of registers and shared memory, so DRAM sees the trace
// about once and the LDE about once (the WHIR encoding's wide plan takes up to
// 10, see below). The driver is `src/lde_cm.rs`.
//
// THE PASS. A launch runs one PASS: levels [s, s + k) of a length-2^L radix-2
// transform, over every column of a chunk (gridDim.y = column). Write an index
// as i = hi·2^(s+k) + mid·2^s + lo. For each (hi, lo) the pass is a length-2^k
// transform over `mid` together with the TWIST
//
//     x[mid] *= ω_{2^(s+k)}^(lo · rev_k(mid))
//
// applied BEFORE the transform for DIT and AFTER it for DIF. That is the
// four-step factorisation of levels [s, s + k): what is left inside the pass
// uses only the powers of ω_{2^k}, which fit a shared table of 2^(k-1)
// entries (128 for every k <= 8). The levels themselves are the textbook
// radix-2 ones:
//
//   DIT (bit-reversed in, natural out; levels ascending): level u pairs i and
//       i + 2^u (bit u of i clear): t = x[i + 2^u]·ω_{2^(u+1)}^(i mod 2^u),
//       (x[i], x[i + 2^u]) <- (x[i] + t, x[i] - t).
//   DIF (natural in, bit-reversed out; levels descending): (a, b) <-
//       (a + b, (a - b)·ω_{2^(u+1)}^(i mod 2^u)).
//
// With the inverse root table either direction computes the inverse transform
// WITHOUT the 1/2^L factor, which is how the prover's weights expect it (they
// carry the 1/n).
//
// THE TILE. A block owns T consecutive `lo` (or, when s = 0, T consecutive
// `hi`) and all 2^k `mid`: a K x T tile. A thread holds E = 16 elements in
// registers: 4 of the mid bits vary over its registers, the other k - 4 bits
// are its group `g`, and threadIdx = g·T + t. Each ROUND runs up to four
// levels in registers; a pass of k <= 8 levels is one or two rounds with one
// shared-memory exchange between them, a WIDE pass of k = 9..12 levels three
// rounds with two. Global loads and stores walk `lo` across the lanes of a
// warp, so they are coalesced for any s >= 4 (a wide pass's T = 4096 >> k
// lanes make row segments of 8 words at k = 9 and 4 at k = 10: whole 32-byte
// sectors); at s = 0 the tile is staged through shared memory instead.
//
// ROOTS. Every root is a power of root32, the field's primitive 2^32-th root
// (`TWO_ADIC_PRIMITVE_ROOT_OF_UNITY`), or of its inverse:
// ω_{2^M}^x = root32^(x << (32 - M)), a 32-bit exponent that wraps exactly
// as the root's order does. `roots` is the 4 x 256 windowed table
// roots[w·256 + j] = root32^(j << 8w), so any power is at most three
// multiplications, and roots[768 + j] = ω_256^j is the radix table of a pass
// of k <= 8 levels. A wide pass builds its ω_{2^k}^j from the windows.
//
// FUSIONS (contiguous passes only, flags):
//   F_SPREAD  DIT s = 0: the input is COMPACT — n = 2^(L - lb) values `y`, and
//             position B·q of the zero-padded transform input is y[q], every
//             other position zero (the coset spread of the LDE). The zeros are
//             never read or stored.
//   F_GATHER  with F_SPREAD: the compact input is in NATURAL order, so y[q] is
//             read at rev_{L-lb}(q) and multiplied by wtab[q].
//   F_STORE_W s = 0: multiply the value at column position p by wtab[p] on the
//             way out (the iNTT's last pass applies the bit-reversed coset
//             weights here).
//
// HOST REPLAY. Every phase is a plain function of (tid, bx, col) so that
// `tests/host_kat/ntt_cm_host_kat.cpp` can compile this file through the CUDA
// host shim and replay a launch phase by phase, every thread of a phase before
// the next — exactly the ordering `__syncthreads` gives the device. The
// `__global__` wrappers are device-only.

#include "goldilocks.cuh"

namespace ntt_cm {

using goldilocks::add;
using goldilocks::mul;
using goldilocks::sub;

// Elements one thread holds in registers, and log2 of it.
constexpr int ELOG = 4;
constexpr int E = 1 << ELOG;

// The radix table of a KLOG-level pass: rad[j] = ω_{2^(LOG+1)}^j for
// j < 2^LOG, the powers every level of the pass takes (level u reads entry
// xi << (LOG - u)). Every k <= 8 shares today's table, ω_256^j for j < 128.
template <int KLOG>
struct Radix {
    static constexpr int LOG = KLOG <= 8 ? 7 : KLOG - 1;
    static constexpr int SIZE = 1 << LOG;
};

constexpr uint32_t F_SPREAD = 1u;
constexpr uint32_t F_GATHER = 2u;
constexpr uint32_t F_STORE_W = 4u;

// root32^e from the windowed table (at most three multiplications).
__device__ __forceinline__ uint64_t pow_root32(const uint64_t *roots, uint32_t e) {
    uint64_t r = roots[e & 255u];
    const uint32_t e1 = (e >> 8) & 255u;
    const uint32_t e2 = (e >> 16) & 255u;
    const uint32_t e3 = e >> 24;
    if (e1) r = mul(r, roots[256 + e1]);
    if (e2) r = mul(r, roots[512 + e2]);
    if (e3) r = mul(r, roots[768 + e3]);
    return r;
}

// Reverse the low `bits` bits of x (bits >= 1).
__device__ __forceinline__ uint64_t rev_bits(uint64_t x, uint32_t bits) {
    return __brevll(x) >> (64 - bits);
}

// Reverse a 4-bit register index — a compile-time constant in every use.
__device__ __forceinline__ constexpr int rev4(int v) {
    return ((v & 1) << 3) | ((v & 2) << 1) | ((v & 4) >> 1) | ((v & 8) >> 3);
}

// Round J of a KLOG-level pass: its levels [U0, U1] and the lowest of the four
// mid bits the registers vary over, A. Full rounds cover four levels; when k
// is not a multiple of four the short round keeps four varying bits and some
// of them are passive (the round's butterflies pair only along its levels).
//   DIT: rounds ascend; the short round is the last and ends at bit k - 1.
//   DIF: rounds descend; the short round is the last and starts at bit 0.
template <int KLOG, bool DIT, int J>
struct Round {
    static constexpr int kDitU0 = 4 * J;
    static constexpr int kDitU1 = (4 * J + 3 < KLOG - 1) ? 4 * J + 3 : KLOG - 1;
    static constexpr int kDitA = (4 * J < KLOG - 4) ? 4 * J : KLOG - 4;
    static constexpr int kDifU1 = KLOG - 1 - 4 * J;
    static constexpr int kDifU0 = (kDifU1 - 3 > 0) ? kDifU1 - 3 : 0;
    static constexpr int U0 = DIT ? kDitU0 : kDifU0;
    static constexpr int U1 = DIT ? kDitU1 : kDifU1;
    static constexpr int A = DIT ? kDitA : kDifU0;
};

// The mid index of register v of group g in a round whose varying bits are
// [A, A + 4): g fills the other k - 4 bits in order.
template <int A>
__device__ __forceinline__ uint32_t mid_base(uint32_t g) {
    return ((g >> A) << (A + ELOG)) | (g & ((1u << A) - 1u));
}

// One level u of the in-register transform: pairs registers v and v | 2^(u-A).
template <bool DIT, int U, int A, int RLOG>
__device__ __forceinline__ void level(uint64_t (&x)[E], uint32_t base, const uint64_t *rad) {
    constexpr int O = U - A;
#pragma unroll
    for (int v = 0; v < E; ++v) {
        if (v & (1 << O)) continue;
        const int w = v | (1 << O);
        uint64_t tw = 1;
        if constexpr (U > 0) {
            const uint32_t xi = (base | ((uint32_t)v << A)) & ((1u << U) - 1u);
            tw = rad[xi << (RLOG - U)];
        }
        if constexpr (DIT) {
            const uint64_t t = (U > 0) ? mul(x[w], tw) : x[w];
            x[w] = sub(x[v], t);
            x[v] = add(x[v], t);
        } else {
            const uint64_t a = x[v];
            const uint64_t b = x[w];
            x[v] = add(a, b);
            x[w] = (U > 0) ? mul(sub(a, b), tw) : sub(a, b);
        }
    }
}

// Levels U .. UEND of a round, ascending (DIT) or descending (DIF).
template <bool DIT, int A, int U, int UEND, int RLOG>
__device__ __forceinline__ void levels(uint64_t (&x)[E], uint32_t base, const uint64_t *rad) {
    level<DIT, U, A, RLOG>(x, base, rad);
    if constexpr (U != UEND) {
        constexpr int NEXT = DIT ? U + 1 : U - 1;
        levels<DIT, A, NEXT, UEND, RLOG>(x, base, rad);
    }
}

template <int KLOG, bool DIT, int J>
__device__ __forceinline__ void run_round(uint64_t (&x)[E], uint32_t g, const uint64_t *rad) {
    using R = Round<KLOG, DIT, J>;
    constexpr int RLOG = Radix<KLOG>::LOG;
    static_assert(KLOG >= ELOG && KLOG <= 12, "a pass runs 4..12 levels");
    static_assert(J >= 0 && J < (KLOG + 3) / 4, "round index out of range");
    static_assert(R::U0 <= R::U1 && R::A >= 0 && R::A <= R::U0 && R::U1 < R::A + ELOG,
                  "a round's levels lie inside its four varying bits");
    const uint32_t base = mid_base<R::A>(g);
    if constexpr (DIT) {
        levels<true, R::A, R::U0, R::U1, RLOG>(x, base, rad);
    } else {
        levels<false, R::A, R::U1, R::U0, RLOG>(x, base, rad);
    }
}

// Tile row pitch: T + 1 keeps a column of the tile off a single bank.
__device__ __forceinline__ uint32_t pitch_of(uint32_t log_t) { return (1u << log_t) + 1u; }

// The tile slot of row `mid`, lane t. A wide pass (k > 8) spans T <= 8 lanes,
// so a half-warp holds 16 / T groups, and in the first round's layout (A = 0)
// their rows lie 16 apart: at pitch T + 1 alone they would share banks. A skew
// of T words every 16 rows puts them on distinct banks; the staged copies (16
// consecutive rows, one lane) keep distinct banks through the odd pitch.
// k <= 8 keeps today's layout.
template <int KLOG>
__device__ __forceinline__ uint32_t tile_at(uint32_t mid, uint32_t log_t, uint32_t t) {
    uint32_t i = mid * pitch_of(log_t) + t;
    if constexpr (KLOG > 8) i += (mid >> ELOG) << log_t;
    return i;
}

// Words the tile spans (the dynamic shared memory, mirrored by the driver's
// `tile_bytes`).
template <int KLOG>
__device__ __forceinline__ uint32_t tile_words(uint32_t log_t) {
    uint32_t w = (1u << KLOG) * pitch_of(log_t);
    if constexpr (KLOG > 8) w += (1u << (KLOG - ELOG)) << log_t;
    return w;
}

template <int KLOG, int A>
__device__ __forceinline__ void tile_write(const uint64_t (&x)[E], uint64_t *tile, uint32_t log_t,
                                           uint32_t t, uint32_t g) {
    const uint32_t base = mid_base<A>(g);
#pragma unroll
    for (int v = 0; v < E; ++v) tile[tile_at<KLOG>(base | ((uint32_t)v << A), log_t, t)] = x[v];
}

template <int KLOG, int A>
__device__ __forceinline__ void tile_read(uint64_t (&x)[E], const uint64_t *tile, uint32_t log_t,
                                          uint32_t t, uint32_t g) {
    const uint32_t base = mid_base<A>(g);
#pragma unroll
    for (int v = 0; v < E; ++v) x[v] = tile[tile_at<KLOG>(base | ((uint32_t)v << A), log_t, t)];
}

// The twist of a strided pass, in the A = 0 register layout that both the
// first DIT round and the last DIF round use: mid = (g << 4) | v, so
// rev_k(mid) = rev4(v)·2^(k-4) + rev_{k-4}(g) and
//   ω_{2^(s+k)}^(lo·rev_k(mid)) = c · d^rev4(v),
//   c = ω_{2^(s+k)}^(lo·rev_{k-4}(g)),  d = ω_{2^(s+4)}^lo.
template <int KLOG>
__device__ __forceinline__ void twist(uint64_t (&x)[E], const uint64_t *roots, uint32_t s,
                                      uint32_t lo, uint32_t g) {
    constexpr int GBITS = KLOG - ELOG;
    uint32_t rg = 0;
    if constexpr (GBITS > 0) rg = (uint32_t)rev_bits(g, GBITS);
    const uint64_t c = pow_root32(roots, (lo * rg) << (32 - s - KLOG));
    const uint64_t d = pow_root32(roots, lo << (32 - s - ELOG));
    uint64_t f = c;
#pragma unroll
    for (int j = 0; j < E; ++j) {
        x[rev4(j)] = mul(x[rev4(j)], f);
        if (j + 1 < E) f = mul(f, d);
    }
}

// Threads in a block: T lanes x K/16 groups.
template <int KLOG>
__device__ __forceinline__ uint32_t threads_of(uint32_t log_t) {
    return (1u << log_t) << (KLOG - ELOG);
}

// Load the pass's radix table into shared memory: today's ω_256^j straight
// from the top window, a wide pass's ω_{2^k}^j = root32^(j << (32 - k)) from
// the windows (at most two multiplications an entry).
template <int KLOG>
__device__ __forceinline__ void phase_rad(uint64_t *rad, const uint64_t *roots, uint32_t tid,
                                          uint32_t nthreads) {
    using RX = Radix<KLOG>;
    for (uint32_t j = tid; j < (uint32_t)RX::SIZE; j += nthreads) {
        if constexpr (RX::LOG == 7) {
            rad[j] = roots[768 + j];
        } else {
            rad[j] = pow_root32(roots, j << (31 - RX::LOG));
        }
    }
}

// ---------------------------------------------------------------------------
// Strided pass (s >= log T >= 0 with s > 0): a block is one hi and T
// consecutive lo. Element (t, mid) of the tile is column index
// hi·2^(s+k) + mid·2^s + lo0 + t.
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint64_t strided_tile_base(uint32_t bx, uint32_t s, uint32_t klog,
                                                      uint32_t log_t, uint32_t t) {
    const uint32_t lo_blocks_log = s - log_t;
    const uint64_t hi = (uint64_t)(bx >> lo_blocks_log);
    const uint64_t lo0 = (uint64_t)(bx & ((1u << lo_blocks_log) - 1u)) << log_t;
    return (hi << (s + klog)) + lo0 + t;
}

// First (or only) phase of a strided pass: load in the first round's layout,
// DIT twist, first round; then either the exchange write (two or three
// rounds) or the DIF twist and the store (one round).
template <int KLOG, bool DIT>
__device__ __forceinline__ void phase_strided_first(
    const uint64_t *in, uint64_t in_stride, uint64_t *out, uint64_t out_stride,
    const uint64_t *roots, const uint64_t *rad, uint64_t *tile, uint32_t L, uint32_t s,
    uint32_t log_t, uint32_t tid, uint32_t bx, uint32_t col) {
    constexpr int ROUNDS = (KLOG + 3) / 4;
    using R0 = Round<KLOG, DIT, 0>;
    const uint32_t t = tid & ((1u << log_t) - 1u);
    const uint32_t g = tid >> log_t;
    const uint64_t tb = strided_tile_base(bx, s, KLOG, log_t, t);
    const uint32_t lo = (uint32_t)(tb & ((1ull << s) - 1ull));
    const uint64_t *src = in + (uint64_t)col * in_stride;

    uint64_t x[E];
    {
        const uint32_t base = mid_base<R0::A>(g);
#pragma unroll
        for (int v = 0; v < E; ++v) x[v] = src[tb + ((uint64_t)(base | ((uint32_t)v << R0::A)) << s)];
    }
    if constexpr (DIT) twist<KLOG>(x, roots, s, lo, g);
    run_round<KLOG, DIT, 0>(x, g, rad);
    if constexpr (ROUNDS >= 2) {
        tile_write<KLOG, R0::A>(x, tile, log_t, t, g);
    } else {
        if constexpr (!DIT) twist<KLOG>(x, roots, s, lo, g);
        uint64_t *dst = out + (uint64_t)col * out_stride;
        const uint32_t base = mid_base<R0::A>(g);
#pragma unroll
        for (int v = 0; v < E; ++v) dst[tb + ((uint64_t)(base | ((uint32_t)v << R0::A)) << s)] = x[v];
    }
}

// Last phase of a strided pass of two or three rounds: exchange read, last
// round, DIF twist, store. (A three-round pass runs its middle round tile to
// tile between the two, `phase_tile_round`.)
template <int KLOG, bool DIT>
__device__ __forceinline__ void phase_strided_last(
    uint64_t *out, uint64_t out_stride, const uint64_t *roots, const uint64_t *rad,
    const uint64_t *tile, uint32_t L, uint32_t s, uint32_t log_t, uint32_t tid, uint32_t bx,
    uint32_t col) {
    constexpr int LAST = (KLOG + 3) / 4 - 1;
    using RL = Round<KLOG, DIT, LAST>;
    const uint32_t t = tid & ((1u << log_t) - 1u);
    const uint32_t g = tid >> log_t;
    const uint64_t tb = strided_tile_base(bx, s, KLOG, log_t, t);
    const uint32_t lo = (uint32_t)(tb & ((1ull << s) - 1ull));

    uint64_t x[E];
    tile_read<KLOG, RL::A>(x, tile, log_t, t, g);
    run_round<KLOG, DIT, LAST>(x, g, rad);
    if constexpr (!DIT) twist<KLOG>(x, roots, s, lo, g);
    uint64_t *dst = out + (uint64_t)col * out_stride;
    const uint32_t base = mid_base<RL::A>(g);
#pragma unroll
    for (int v = 0; v < E; ++v) dst[tb + ((uint64_t)(base | ((uint32_t)v << RL::A)) << s)] = x[v];
}

// ---------------------------------------------------------------------------
// Contiguous pass (s = 0): a block is T consecutive sub-transforms of K
// consecutive elements, hi0 = bx·T. The tile is staged through shared memory
// so that global accesses walk consecutive addresses.
// ---------------------------------------------------------------------------

// Stage the block's T·K elements into the tile.
template <int KLOG>
__device__ __forceinline__ void phase_contig_stage(const uint64_t *in, uint64_t in_stride,
                                                   const uint64_t *wtab, uint64_t *tile,
                                                   uint32_t L, uint32_t lb, uint32_t log_t,
                                                   uint32_t flags, uint32_t tid, uint32_t bx,
                                                   uint32_t col) {
    constexpr uint32_t K = 1u << KLOG;
    const uint32_t nthreads = threads_of<KLOG>(log_t);
    const uint32_t count = K << log_t;
    const uint64_t *src = in + (uint64_t)col * in_stride;
    const uint64_t block0 = (uint64_t)bx << (KLOG + log_t);
    if (flags & F_SPREAD) {
        // Compact input: the block covers compact positions
        // [block0 >> lb, (block0 + count) >> lb); position B·q of the
        // transform input is y[q], the rest is zero.
        const uint32_t bmask = (1u << lb) - 1u;
        const uint32_t log_n = L - lb;
        for (uint32_t e = tid; e < count; e += nthreads) {
            const uint32_t t = e >> KLOG;
            const uint32_t mid = e & (K - 1u);
            uint64_t v = 0;
            if ((mid & bmask) == 0) {
                const uint64_t q = (block0 + e) >> lb;
                if (flags & F_GATHER) {
                    v = mul(src[rev_bits(q, log_n)], wtab[q]);
                } else {
                    v = src[q];
                }
            }
            tile[tile_at<KLOG>(mid, log_t, t)] = v;
        }
    } else {
        for (uint32_t e = tid; e < count; e += nthreads) {
            const uint32_t t = e >> KLOG;
            const uint32_t mid = e & (K - 1u);
            tile[tile_at<KLOG>(mid, log_t, t)] = src[block0 + e];
        }
    }
}

// Round J tile to tile — every round of a contiguous pass, and the middle
// round of a three-round strided one: each thread reads and writes back
// exactly the tile slots of its own group, so no barrier is needed between
// its read and its write.
template <int KLOG, bool DIT, int J>
__device__ __forceinline__ void phase_tile_round(uint64_t *tile, const uint64_t *rad,
                                                 uint32_t log_t, uint32_t tid) {
    using R = Round<KLOG, DIT, J>;
    const uint32_t t = tid & ((1u << log_t) - 1u);
    const uint32_t g = tid >> log_t;
    uint64_t x[E];
    tile_read<KLOG, R::A>(x, tile, log_t, t, g);
    run_round<KLOG, DIT, J>(x, g, rad);
    tile_write<KLOG, R::A>(x, tile, log_t, t, g);
}

// Store the block's T·K elements from the tile, optionally weighted.
template <int KLOG>
__device__ __forceinline__ void phase_contig_store(uint64_t *out, uint64_t out_stride,
                                                   const uint64_t *wtab, const uint64_t *tile,
                                                   uint32_t log_t, uint32_t flags, uint32_t tid,
                                                   uint32_t bx, uint32_t col) {
    constexpr uint32_t K = 1u << KLOG;
    const uint32_t nthreads = threads_of<KLOG>(log_t);
    const uint32_t count = K << log_t;
    uint64_t *dst = out + (uint64_t)col * out_stride;
    const uint64_t block0 = (uint64_t)bx << (KLOG + log_t);
    for (uint32_t e = tid; e < count; e += nthreads) {
        const uint32_t t = e >> KLOG;
        const uint32_t mid = e & (K - 1u);
        uint64_t v = tile[tile_at<KLOG>(mid, log_t, t)];
        if (flags & F_STORE_W) v = mul(v, wtab[block0 + e]);
        dst[block0 + e] = v;
    }
}

}  // namespace ntt_cm

#if defined(__CUDACC__)

// One pass. `s == 0` selects the contiguous (staged) form. The dynamic shared
// memory is the tile (`tile_words`); the radix table is static.
template <int KLOG, bool DIT>
__device__ __forceinline__ void ntt_cm_pass(const uint64_t *in, uint64_t in_stride, uint64_t *out,
                                            uint64_t out_stride, const uint64_t *roots,
                                            const uint64_t *wtab, uint32_t L, uint32_t s,
                                            uint32_t log_t, uint32_t lb, uint32_t flags) {
    using namespace ntt_cm;
    extern __shared__ uint64_t ntt_cm_tile[];
    __shared__ uint64_t rad[Radix<KLOG>::SIZE];
    constexpr int ROUNDS = (KLOG + 3) / 4;
    const uint32_t tid = threadIdx.x;
    const uint32_t bx = blockIdx.x;
    const uint32_t col = blockIdx.y;
    const uint32_t nthreads = threads_of<KLOG>(log_t);

    phase_rad<KLOG>(rad, roots, tid, nthreads);
    if (s > 0) {
        __syncthreads();
        phase_strided_first<KLOG, DIT>(in, in_stride, out, out_stride, roots, rad, ntt_cm_tile, L, s,
                                       log_t, tid, bx, col);
        if constexpr (ROUNDS == 3) {
            __syncthreads();
            phase_tile_round<KLOG, DIT, 1>(ntt_cm_tile, rad, log_t, tid);
        }
        if constexpr (ROUNDS >= 2) {
            __syncthreads();
            phase_strided_last<KLOG, DIT>(out, out_stride, roots, rad, ntt_cm_tile, L, s, log_t,
                                          tid, bx, col);
        }
    } else {
        phase_contig_stage<KLOG>(in, in_stride, wtab, ntt_cm_tile, L, lb, log_t, flags, tid, bx, col);
        __syncthreads();
        phase_tile_round<KLOG, DIT, 0>(ntt_cm_tile, rad, log_t, tid);
        if constexpr (ROUNDS >= 2) {
            __syncthreads();
            phase_tile_round<KLOG, DIT, 1>(ntt_cm_tile, rad, log_t, tid);
        }
        if constexpr (ROUNDS == 3) {
            __syncthreads();
            phase_tile_round<KLOG, DIT, 2>(ntt_cm_tile, rad, log_t, tid);
        }
        __syncthreads();
        phase_contig_store<KLOG>(out, out_stride, wtab, ntt_cm_tile, log_t, flags, tid, bx, col);
    }
}

// The driver never launches more than 256 threads (T lanes x K/16 groups is
// capped at 4096/16); two resident blocks per SM bound a thread at 128
// registers, which holds the 16 elements and the round's temporaries. The
// wide passes are DIT only: they serve the WHIR encoding's forward transform
// (`lde_cm::plan_wide`), and at k = 10 the tile and the radix table take
// 46 KiB of the 48 a block has without an opt-in.
#define NTT_CM_KERNEL(NAME, KLOG, DIT)                                                          \
    extern "C" __global__ void __launch_bounds__(256, 2)                                        \
        NAME(const uint64_t *in, uint64_t in_stride, uint64_t *out, uint64_t out_stride,        \
             const uint64_t *roots, const uint64_t *wtab, uint32_t L, uint32_t s,               \
             uint32_t log_t, uint32_t lb, uint32_t flags) {                                     \
        ntt_cm_pass<KLOG, DIT>(in, in_stride, out, out_stride, roots, wtab, L, s, log_t, lb,    \
                               flags);                                                          \
    }

NTT_CM_KERNEL(ntt_cm_dit_k4, 4, true)
NTT_CM_KERNEL(ntt_cm_dit_k5, 5, true)
NTT_CM_KERNEL(ntt_cm_dit_k6, 6, true)
NTT_CM_KERNEL(ntt_cm_dit_k7, 7, true)
NTT_CM_KERNEL(ntt_cm_dit_k8, 8, true)
NTT_CM_KERNEL(ntt_cm_dit_k9, 9, true)
NTT_CM_KERNEL(ntt_cm_dit_k10, 10, true)
NTT_CM_KERNEL(ntt_cm_dif_k4, 4, false)
NTT_CM_KERNEL(ntt_cm_dif_k5, 5, false)
NTT_CM_KERNEL(ntt_cm_dif_k6, 6, false)
NTT_CM_KERNEL(ntt_cm_dif_k7, 7, false)
NTT_CM_KERNEL(ntt_cm_dif_k8, 8, false)

#endif  // __CUDACC__
