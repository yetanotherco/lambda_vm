// The values a first-round query's kept subtree holds, evaluated from the
// stacked multilinear instead of read out of its re-encoded codeword
// (`src/whir_coset.rs`, which states the identity and pins every kernel here
// against the host).
//
// The multilinear `f` has `2^n` hypercube values; its codeword is
// `P(ω^j)`, `P(x) = f̂(x^(2^(n-1)), …, x^2, x)` with hypercube bit `s` sent to
// `x^(2^(n-1-s))`. On the coset `ω^c·⟨ζ⟩`, `|⟨ζ⟩| = 2^L`, every bit `s < n-L`
// sees a constant `γ_s(c) = ω^(c·2^(n-1-s))` and folds away:
//
//   f'(rest) = f(rest, 0) + γ·(f(rest, 1) − f(rest, 0))
//
// Bit `s`'s constant depends on `c` modulo `2^(b+1+s)` only, so the low bits
// are folded once per residue class and shared by every coset in it: the
// tile kernel folds the low `A` bits of each `2^A`-value tile for every class
// any coset needs, the rest kernel folds the remaining bits per coset, and
// what is left (`2^L` values a coset) goes through its Möbius transform, a
// twist by `ω^(c·i)` and one radix-2 NTT over `ζ`.
//
// Values are non-canonical u64 throughout (`goldilocks.cuh`); the host
// compares canonical values.

#include "goldilocks.cuh"

using goldilocks::add;
using goldilocks::mul;
using goldilocks::sub;

// Tiles one block folds before it writes: a run of this many consecutive tile
// indices per class, so the writes are whole sectors.
#define COSET_TILES_PER_BLOCK 8
// Most tile levels (`A`): a tile of 256 values, the block's width.
#define COSET_MAX_TILE_LEVELS 8

__device__ __forceinline__ uint64_t coset_fold(uint64_t f0, uint64_t f1, uint64_t gamma) {
    return add(f0, mul(gamma, sub(f1, f0)));
}

__device__ __forceinline__ uint64_t coset_pow(uint64_t base, uint64_t exp) {
    uint64_t acc = 1;
    while (exp > 0) {
        if (exp & 1ull) acc = mul(acc, base);
        base = mul(base, base);
        exp >>= 1;
    }
    return acc;
}

// Folds the low `levels` bits of every `2^levels`-value tile of `f` for every
// class the plan lists.
//
// Level `t` lists `count[t]` classes at `offset[t]` in `parent`/`gamma`:
// class `k`'s parent is its residue one level down (row `parent[k]` of the
// level before; level 0's parent is the tile itself, row 0), and `gamma[k]` is
// its constant for bit `t`. Level `t` holds `count[t]` rows of
// `2^(levels-1-t)` values, so a level's buffer is `buf_len` at most.
//
// Output: `slab[k·num_tiles + T]`, `k` a class of the last level, `T` the tile.
//
// Shared memory (dynamic): gamma[total] · buf0[buf_len] · buf1[buf_len] ·
// fin[count[levels-1]·TILES] · parent[total] (u32).
extern "C" __global__ void __launch_bounds__(256)
coset_fold_tiles(const uint64_t *__restrict__ f, uint64_t num_tiles, uint32_t levels,
                 const uint32_t *__restrict__ count, const uint32_t *__restrict__ offset,
                 const uint32_t *__restrict__ parent, const uint64_t *__restrict__ gamma,
                 uint32_t total, uint32_t buf_len, uint64_t *__restrict__ slab) {
    extern __shared__ uint64_t coset_smem[];
    __shared__ uint32_t s_count[COSET_MAX_TILE_LEVELS];
    __shared__ uint32_t s_offset[COSET_MAX_TILE_LEVELS];

    const uint32_t tid = threadIdx.x;
    const uint32_t bdim = blockDim.x;
    if (tid < levels) {
        s_count[tid] = count[tid];
        s_offset[tid] = offset[tid];
    }
    __syncthreads();
    const uint32_t last_count = s_count[levels - 1];

    uint64_t *s_gamma = coset_smem;
    uint64_t *buf0 = s_gamma + total;
    uint64_t *buf1 = buf0 + buf_len;
    uint64_t *fin = buf1 + buf_len;
    uint32_t *s_parent = (uint32_t *)(fin + (uint64_t)last_count * COSET_TILES_PER_BLOCK);
    for (uint32_t x = tid; x < total; x += bdim) {
        s_gamma[x] = gamma[x];
        s_parent[x] = parent[x];
    }

    const uint32_t tile = 1u << levels;
    const uint64_t first = (uint64_t)blockIdx.x * COSET_TILES_PER_BLOCK;
    for (uint32_t i = 0; i < COSET_TILES_PER_BLOCK; ++i) {
        const uint64_t at = first + i;
        if (at >= num_tiles) break;  // the same for every thread of the block
        const uint64_t *src = f + at * tile;
        // The previous tile's last level read buf0's partner; the barrier at
        // that level's end ordered it before this write.
        for (uint32_t x = tid; x < tile; x += bdim) buf0[x] = src[x];
        __syncthreads();
        uint64_t *in = buf0;
        uint64_t *out = buf1;
        uint32_t width = tile;
        for (uint32_t t = 0; t < levels; ++t) {
            const uint32_t cnt = s_count[t];
            const uint32_t off = s_offset[t];
            const uint32_t log_half = levels - 1 - t;
            const uint32_t half = 1u << log_half;
            const uint32_t outputs = cnt << log_half;
            const bool last = t + 1 == levels;
            for (uint32_t o = tid; o < outputs; o += bdim) {
                const uint32_t k = o >> log_half;
                const uint32_t j = o & (half - 1);
                const uint64_t *row = in + (uint64_t)s_parent[off + k] * width + 2 * j;
                const uint64_t r = coset_fold(row[0], row[1], s_gamma[off + k]);
                if (last) {
                    fin[(uint64_t)k * COSET_TILES_PER_BLOCK + i] = r;
                } else {
                    out[o] = r;
                }
            }
            __syncthreads();
            uint64_t *swap = in;
            in = out;
            out = swap;
            width = half;
        }
    }

    for (uint32_t x = tid; x < last_count * COSET_TILES_PER_BLOCK; x += bdim) {
        const uint32_t k = x / COSET_TILES_PER_BLOCK;
        const uint64_t at = first + x % COSET_TILES_PER_BLOCK;
        if (at < num_tiles) slab[(uint64_t)k * num_tiles + at] = fin[x];
    }
}

// Folds the bits the tile kernel left (`rest` ≤ 5 of them) per coset: one
// warp a value, lane `l` holding slab entry `h·2^rest + l` of the coset's
// class, the pairs a level folds `2^l` lanes apart.
//
// `g[q·2^log_free + h]`, `h < 2^log_free`. `rest_gamma[q·rest + l]` is coset
// `q`'s constant for its `l`-th remaining bit; `slab_of[q]` its class row.
extern "C" __global__ void __launch_bounds__(256)
coset_fold_rest(const uint64_t *__restrict__ slab, uint64_t num_tiles, uint32_t rest,
                uint32_t cosets, const uint32_t *__restrict__ slab_of,
                const uint64_t *__restrict__ rest_gamma, uint32_t log_free,
                uint64_t *__restrict__ g) {
    const uint64_t warp = ((uint64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const uint32_t lane = threadIdx.x & 31;
    if (warp >= ((uint64_t)cosets << log_free)) return;  // whole warps only
    const uint64_t q = warp >> log_free;
    const uint64_t h = warp & ((1ull << log_free) - 1);
    const uint64_t *src = slab + (uint64_t)slab_of[q] * num_tiles + (h << rest);
    uint64_t v = lane < (1u << rest) ? src[lane] : 0;
    for (uint32_t l = 0; l < rest; ++l) {
        const uint64_t partner =
            (uint64_t)__shfl_down_sync(0xffffffffu, (unsigned long long)v, 1u << l);
        if ((lane & ((2u << l) - 1)) == 0) v = coset_fold(v, partner, rest_gamma[q * rest + l]);
    }
    if (lane == 0) g[warp] = v;
}

// One level of the Möbius transform over every coset's `2^log_size` values,
// for a level the chunk kernel does not reach: `hi −= lo`, `2^level` apart.
extern "C" __global__ void coset_mobius_level(uint64_t *__restrict__ g, uint64_t pairs,
                                              uint32_t log_size, uint32_t level) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const uint64_t q = idx >> (log_size - 1);
    const uint64_t r = idx & ((1ull << (log_size - 1)) - 1);
    const uint64_t lo = ((r >> level) << (level + 1)) | (r & ((1ull << level) - 1));
    uint64_t *base = g + (q << log_size);
    base[lo + (1ull << level)] = sub(base[lo + (1ull << level)], base[lo]);
}

// One `2^chunk_levels` chunk of a coset's values, in shared memory: the
// Möbius levels below `chunk_levels`, the twist, and the NTT's first
// `chunk_levels` stages.
//
// After the Möbius transform position `h` holds the coefficient of
// `x^bitrev_L(h)` — the input order a decimation-in-time NTT takes — so the
// twist multiplies it by `w^bitrev_L(h)`, `w = twist[q] = ω^c`, and the stages
// run in place. `twiddles[j] = ζ^j`, `j < 2^(log_size-1)`. One thread a pair.
extern "C" __global__ void coset_chunk(uint64_t *__restrict__ g, uint32_t log_size,
                                       uint32_t chunk_levels,
                                       const uint64_t *__restrict__ twist,
                                       const uint64_t *__restrict__ twiddles) {
    extern __shared__ uint64_t coset_chunk_smem[];
    uint64_t *x = coset_chunk_smem;
    const uint32_t tid = threadIdx.x;
    const uint32_t half = 1u << (chunk_levels - 1);
    const uint64_t chunks = 1ull << (log_size - chunk_levels);
    const uint64_t q = blockIdx.x / chunks;
    const uint64_t at = (blockIdx.x % chunks) << chunk_levels;
    uint64_t *base = g + (q << log_size) + at;
    x[tid] = base[tid];
    x[tid + half] = base[tid + half];
    __syncthreads();

    for (uint32_t l = 0; l < chunk_levels; ++l) {
        const uint32_t lo = ((tid >> l) << (l + 1)) | (tid & ((1u << l) - 1));
        x[lo + (1u << l)] = sub(x[lo + (1u << l)], x[lo]);
        __syncthreads();
    }

    const uint64_t w = twist[q];
    for (uint32_t e = tid; e < 2 * half; e += half) {
        const uint64_t i = __brevll(at + e) >> (64 - log_size);
        x[e] = mul(x[e], coset_pow(w, i));
    }
    __syncthreads();

    for (uint32_t s = 1; s <= chunk_levels; ++s) {
        const uint32_t h = 1u << (s - 1);
        const uint32_t i0 = ((tid >> (s - 1)) << s) | (tid & (h - 1));
        const uint64_t tw = twiddles[(uint64_t)(tid & (h - 1)) << (log_size - s)];
        const uint64_t u = x[i0];
        const uint64_t v = mul(x[i0 + h], tw);
        x[i0] = add(u, v);
        x[i0 + h] = sub(u, v);
        __syncthreads();
    }

    base[tid] = x[tid];
    base[tid + half] = x[tid + half];
}

// One NTT stage `s` (pairs `2^(s-1)` apart) over every coset's values, for a
// stage past the chunk kernel's.
extern "C" __global__ void coset_ntt_stage(uint64_t *__restrict__ g, uint64_t pairs,
                                           uint32_t log_size, uint32_t s,
                                           const uint64_t *__restrict__ twiddles) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= pairs) return;
    const uint64_t q = idx >> (log_size - 1);
    const uint64_t r = idx & ((1ull << (log_size - 1)) - 1);
    const uint64_t h = 1ull << (s - 1);
    const uint64_t j = r & (h - 1);
    const uint64_t i0 = ((r >> (s - 1)) << s) | j;
    uint64_t *base = g + (q << log_size);
    const uint64_t u = base[i0];
    const uint64_t v = mul(base[i0 + h], twiddles[j << (log_size - s)]);
    base[i0] = add(u, v);
    base[i0 + h] = sub(u, v);
}

// The coset's values, in domain order (`g[q][u]` = the codeword at
// `c + u·2^(D-L)`), written in the order its kept subtree's leaves take them:
// leaf `r` (position `r` under the node) is `j = c + bitrev_kept(r)·2^(D-L)`,
// and its `2^fold` values are `u = bitrev_kept(r) + t·2^kept`, `t` ascending.
// `stride == 0`: leaf-major, leaf `r` of coset `q` at `(q·2^kept + r)·2^fold`.
// Otherwise strided as the coset leaf kernels read a codeword: value `t` of
// gathered leaf `q·2^kept + r` at `(q·2^kept + r) + t·stride`.
extern "C" __global__ void coset_leaf_order(const uint64_t *__restrict__ g, uint64_t total,
                                            uint32_t log_size, uint32_t kept, uint64_t stride,
                                            uint64_t *__restrict__ out) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    const uint64_t q = idx >> log_size;
    const uint64_t u = idx & ((1ull << log_size) - 1);
    const uint64_t low = u & ((1ull << kept) - 1);
    const uint64_t r = kept == 0 ? 0 : __brevll(low) >> (64 - kept);
    const uint64_t t = u >> kept;
    if (stride == 0) {
        out[(q << log_size) | (r << (log_size - kept)) | t] = g[idx];
    } else {
        out[((q << kept) | r) + t * stride] = g[idx];
    }
}
