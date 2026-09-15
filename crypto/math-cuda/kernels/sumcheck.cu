// One sumcheck round over the Boolean hypercube, for a batch given as
// straight-line code over its factors (`multilinear::program::Program`, lowered
// by `crypto/multilinear/src/gpu.rs`).
//
// Design:
//   * One thread per (cube index, interpolation node), grid-stride in both:
//     `blockIdx.x` walks the cube and `blockIdx.y` the nodes. A late round has
//     a cube smaller than the device is wide and nothing left to hide the slot
//     file's latency behind, and the nodes are the only parallelism left to
//     give it. The nodes stay the inner loop, so a factor's `lo` and `hi` are
//     read once and stay in cache across the nodes a block owns.
//   * Every value is ext3 — the sumcheck runs in one field, and the factors
//     were lifted when they were built.
//   * The lowering assigns each step a slot with liveness reuse, so the
//     per-thread scratch is the program's max live set, not its step count
//     (25k steps on the precompile tables). Slots are strided by thread:
//     slot `s` component `k` is `d_slots[(s*3 + k) * num_threads + tid]`.
//   * The round sums over the whole cube, so each block reduces its threads in
//     shared memory and writes one partial per interpolation node. Field
//     addition is associative and Goldilocks compares canonically, so the
//     order the partials are summed in is not observable.
//
// Op tags and the node packing MUST stay in sync with
// `crypto/multilinear/src/gpu.rs`.

#include "goldilocks.cuh"
#include "ext3.cuh"

using ext3::Fe3;

#define OP_FIXED 0u
#define OP_VAR 1u
#define OP_ADD 2u
#define OP_SUB 3u
#define OP_MUL 4u
#define OP_NEG 5u

// Interpolation nodes a round can ask for. The host declines above this.
#define MAX_NODES 16

// A lowered step: `word0 = op | (a << 32)`, `word1 = b | (res << 32)`.
struct Node {
    uint32_t op, a, b, res;
};

__device__ __forceinline__ Node load_node(const uint64_t *d_nodes, uint64_t i) {
    uint64_t w0 = d_nodes[2 * i];
    uint64_t w1 = d_nodes[2 * i + 1];
    Node n;
    n.op = (uint32_t)(w0 & 0xFFFFFFFFull);
    n.a = (uint32_t)(w0 >> 32);
    n.b = (uint32_t)(w1 & 0xFFFFFFFFull);
    n.res = (uint32_t)(w1 >> 32);
    return n;
}

__device__ __forceinline__ Fe3 load_ext(const uint64_t *p) {
    return ext3::make(p[0], p[1], p[2]);
}

__device__ __forceinline__ Fe3 load_slot(const uint64_t *slots, uint64_t stride, uint32_t slot) {
    const uint64_t *p = slots + (uint64_t)slot * 3 * stride;
    return ext3::make(p[0], p[stride], p[2 * stride]);
}

__device__ __forceinline__ void store_slot(uint64_t *slots, uint64_t stride, uint32_t slot,
                                           const Fe3 &v) {
    uint64_t *p = slots + (uint64_t)slot * 3 * stride;
    p[0] = v.a;
    p[stride] = v.b;
    p[2 * stride] = v.c;
}

// The program's value at cube index `j`, with every factor extended to the
// interpolation node `t`: `f(j) + t·(f(j + half) − f(j))`.
__device__ __forceinline__ Fe3 eval_program(const uint64_t *__restrict__ d_nodes,
                                            uint64_t num_nodes,
                                            const uint64_t *__restrict__ d_consts,
                                            const uint64_t *const *__restrict__ d_factors,
                                            uint64_t j, uint64_t half, const Fe3 &t,
                                            uint64_t *slots, uint64_t stride, uint32_t root_slot) {
    for (uint64_t i = 0; i < num_nodes; i++) {
        Node nd = load_node(d_nodes, i);
        switch (nd.op) {
        case OP_VAR: {
            const uint64_t *column = d_factors[nd.a];
            Fe3 lo = load_ext(column + j * 3);
            Fe3 hi = load_ext(column + (j + half) * 3);
            Fe3 v = ext3::add(lo, ext3::mul(t, ext3::sub(hi, lo)));
            store_slot(slots, stride, nd.res, v);
            break;
        }
        case OP_FIXED:
            store_slot(slots, stride, nd.res, load_ext(d_consts + (uint64_t)nd.a * 3));
            break;
        case OP_ADD:
            store_slot(slots, stride, nd.res,
                       ext3::add(load_slot(slots, stride, nd.a), load_slot(slots, stride, nd.b)));
            break;
        case OP_SUB:
            store_slot(slots, stride, nd.res,
                       ext3::sub(load_slot(slots, stride, nd.a), load_slot(slots, stride, nd.b)));
            break;
        case OP_MUL:
            store_slot(slots, stride, nd.res,
                       ext3::mul(load_slot(slots, stride, nd.a), load_slot(slots, stride, nd.b)));
            break;
        case OP_NEG:
            store_slot(slots, stride, nd.res, ext3::neg(load_slot(slots, stride, nd.a)));
            break;
        default:
            break;
        }
    }
    return load_slot(slots, stride, root_slot);
}

// Capped at the block size the launcher uses and two blocks per SM: without a
// bound the compiler spends 67 registers a thread and occupancy stops at half.
extern "C" __global__ __launch_bounds__(256, 2) void sumcheck_round_ext3(
    // one device pointer per factor; factor `k` at cube index `j`, component
    // `c`, is `d_factors[k][j*3 + c]`
    const uint64_t *const *__restrict__ d_factors,
    // cube indices this round: `lo` at `j`, `hi` at `j + half`
    uint64_t half,
    // the program
    const uint64_t *__restrict__ d_nodes, uint64_t num_nodes,
    const uint64_t *__restrict__ d_consts, uint32_t root_slot,
    // interpolation nodes, ext3
    const uint64_t *__restrict__ d_t, uint32_t num_t,
    // per-thread slot file
    uint64_t *__restrict__ d_slots,
    // out: one partial per (node, block), ext3
    uint64_t *__restrict__ d_partials) {
    // The slot file is per thread and a thread is a (cube index, node) pair,
    // so both grid dimensions go into the address.
    uint64_t tid = ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * gridDim.y * blockDim.x;
    uint64_t *slots = d_slots + tid;

    uint64_t index = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t index_stride = (uint64_t)gridDim.x * blockDim.x;

    // Each node belongs to exactly one `blockIdx.y`, so the partial it writes
    // below has one writer whatever the grid's second dimension is.
    Fe3 acc[MAX_NODES];
    for (uint32_t ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
        acc[ti] = ext3::make(0, 0, 0);
    }

    for (uint64_t j = index; j < half; j += index_stride) {
        for (uint32_t ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
            Fe3 t = load_ext(d_t + (uint64_t)ti * 3);
            Fe3 v = eval_program(d_nodes, num_nodes, d_consts, d_factors, j, half, t, slots,
                                 num_threads, root_slot);
            acc[ti] = ext3::add(acc[ti], v);
        }
    }

    // One node at a time through the same shared buffer: the round's degree is
    // a handful, and a buffer per node would bound the block size instead.
    extern __shared__ uint64_t shared[];
    for (uint32_t ti = blockIdx.y; ti < num_t; ti += gridDim.y) {
        shared[threadIdx.x * 3 + 0] = acc[ti].a;
        shared[threadIdx.x * 3 + 1] = acc[ti].b;
        shared[threadIdx.x * 3 + 2] = acc[ti].c;
        __syncthreads();
        for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
            if (threadIdx.x < width) {
                Fe3 x = load_ext(shared + threadIdx.x * 3);
                Fe3 y = load_ext(shared + (threadIdx.x + width) * 3);
                Fe3 sum = ext3::add(x, y);
                shared[threadIdx.x * 3 + 0] = sum.a;
                shared[threadIdx.x * 3 + 1] = sum.b;
                shared[threadIdx.x * 3 + 2] = sum.c;
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            uint64_t at = ((uint64_t)ti * gridDim.x + blockIdx.x) * 3;
            d_partials[at + 0] = shared[0];
            d_partials[at + 1] = shared[1];
            d_partials[at + 2] = shared[2];
        }
        __syncthreads();
    }
}

// Sums a round's per-block partials, one block per interpolation node.
//
// The round leaves `num_t × blocks` extension values and the transcript wants
// `num_t`. Reducing them here instead of on the host is what keeps a round's
// answer at a few dozen bytes: the partials are megabytes for a wide launch,
// and every round of every sumcheck waits for that copy.
extern "C" __global__ void sum_partials_ext3(const uint64_t *__restrict__ d_partials,
                                             uint64_t blocks, uint64_t *__restrict__ out) {
    const uint64_t *base = d_partials + (uint64_t)blockIdx.x * blocks * 3;
    Fe3 acc = ext3::make(0, 0, 0);
    for (uint64_t i = threadIdx.x; i < blocks; i += blockDim.x) {
        acc = ext3::add(acc, load_ext(base + i * 3));
    }
    extern __shared__ uint64_t shared[];
    shared[threadIdx.x * 3 + 0] = acc.a;
    shared[threadIdx.x * 3 + 1] = acc.b;
    shared[threadIdx.x * 3 + 2] = acc.c;
    __syncthreads();
    for (uint32_t width = blockDim.x / 2; width > 0; width >>= 1) {
        if (threadIdx.x < width) {
            Fe3 x = load_ext(shared + threadIdx.x * 3);
            Fe3 y = load_ext(shared + (threadIdx.x + width) * 3);
            Fe3 sum = ext3::add(x, y);
            shared[threadIdx.x * 3 + 0] = sum.a;
            shared[threadIdx.x * 3 + 1] = sum.b;
            shared[threadIdx.x * 3 + 2] = sum.c;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        uint64_t *at = out + (uint64_t)blockIdx.x * 3;
        at[0] = shared[0];
        at[1] = shared[1];
        at[2] = shared[2];
    }
}

// The program's value at every row, written out rather than summed: the LogUp
// input layer is one of these per interaction per side.
//
// Reuses the round's walk with `half = 0` and `t = 0`, which makes `OP_VAR`
// read `lo` and extend it by nothing — the plain value at the row.
extern "C" __global__ void program_map_ext3(const uint64_t *const *__restrict__ d_factors,
                                            uint64_t num_rows,
                                            const uint64_t *__restrict__ d_nodes,
                                            uint64_t num_nodes,
                                            const uint64_t *__restrict__ d_consts,
                                            uint32_t root_slot, uint64_t *__restrict__ d_slots,
                                            uint64_t *__restrict__ out) {
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;
    uint64_t *slots = d_slots + tid;
    Fe3 zero = ext3::make(0, 0, 0);

    for (uint64_t row = tid; row < num_rows; row += num_threads) {
        Fe3 v = eval_program(d_nodes, num_nodes, d_consts, d_factors, row, 0, zero, slots,
                             num_threads, root_slot);
        uint64_t *at = out + row * 3;
        at[0] = v.a;
        at[1] = v.b;
        at[2] = v.c;
    }
}

// Builds a table's factors out of its base columns: factor `slot` at row `j` is
// the column it reads, taken at its frame-step offset and lifted into the
// extension.
//
// `d_plan` is three u64 per committed factor — where its column starts inside
// `d_columns`, how far it is shifted (already reduced mod `rows`), and the slot
// it fills. The public factors are not here: they are in the extension already
// and are copied in as they are.
//
// The point of this kernel is that the columns are a third of what the factors
// are, so what crosses the bus is the trace and not its lift.
extern "C" __global__ void factors_from_columns_ext3(const uint64_t *__restrict__ d_columns,
                                                     const uint64_t *__restrict__ d_plan,
                                                     uint64_t num_plan, uint64_t rows,
                                                     uint64_t *__restrict__ out) {
    uint64_t total = num_plan * rows;
    for (uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; task < total;
         task += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t p = task / rows;
        uint64_t j = task - p * rows;
        uint64_t base = d_plan[p * 3 + 0];
        uint64_t shift = d_plan[p * 3 + 1];
        uint64_t slot = d_plan[p * 3 + 2];
        uint64_t k = j + shift;
        if (k >= rows) k -= rows;
        uint64_t *at = out + (slot * rows + j) * 3;
        at[0] = d_columns[base + k];
        at[1] = 0;
        at[2] = 0;
    }
}

// Fills a range with one ext3 value — the padding interactions of an input
// layer, whose numerators vanish and whose denominators are one.
extern "C" __global__ void fill_ext3(uint64_t *__restrict__ dst, uint64_t count,
                                     const uint64_t *__restrict__ value) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    uint64_t *at = dst + j * 3;
    at[0] = value[0];
    at[1] = value[1];
    at[2] = value[2];
}

// Lifts a base-field table into the extension: `out[j] = {in[j], 0, 0}`.
extern "C" __global__ void mle_lift_base_ext3(const uint64_t *__restrict__ in, uint64_t count,
                                              uint64_t *__restrict__ out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    uint64_t *at = out + j * 3;
    at[0] = in[j];
    at[1] = 0;
    at[2] = 0;
}

// `dst[j] += scale · src[j]`, the shape a weight takes when a round adds the
// next claim to it.
extern "C" __global__ void add_scaled_ext3(uint64_t *__restrict__ dst,
                                           const uint64_t *__restrict__ src, uint64_t count,
                                           const uint64_t *__restrict__ scale) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    Fe3 term = ext3::mul(load_ext(src + j * 3), load_ext(scale));
    Fe3 sum = ext3::add(load_ext(dst + j * 3), term);
    uint64_t *at = dst + j * 3;
    at[0] = sum.a;
    at[1] = sum.b;
    at[2] = sum.c;
}

// One level of the eq table's doubling: `dst[j + half] = dst[j]·r` and
// `dst[j] = dst[j]·(1 − r)`, the halves disjoint so one thread owns both. The
// host seeds `dst[0]` and walks the variables back to front, which is what
// puts variable 0 in the high bit.
extern "C" __global__ void eq_expand_level_ext3(uint64_t *__restrict__ dst, uint64_t half,
                                                const uint64_t *__restrict__ r) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    Fe3 value = load_ext(dst + j * 3);
    Fe3 challenge = load_ext(r);
    Fe3 hi = ext3::mul(value, challenge);
    Fe3 lo = ext3::sub(value, hi);
    uint64_t *at = dst + j * 3;
    at[0] = lo.a;
    at[1] = lo.b;
    at[2] = lo.c;
    uint64_t *up = dst + (j + half) * 3;
    up[0] = hi.a;
    up[1] = hi.b;
    up[2] = hi.c;
}

// A stacked polynomial's weight is one `eq` table per column, each in its own
// subcube. Building them one at a time is a launch per level per column —
// tens of thousands of launches for a hundred milliseconds of work — so these
// two do every column at once.
//
// The shares arrive sorted by variable count, descending, so the ones still
// doubling at level `l` are exactly the first `active` of them. Three u64 per
// share: where its subcube starts, how many variables it has, and where its
// point starts in `d_points`.

// Seeds every share's first cell with its scale. The table is a product over
// the variables, so one more factor at the start scales every cell.
extern "C" __global__ void eq_seed_shares_ext3(uint64_t *__restrict__ dst,
                                               const uint64_t *__restrict__ d_shares,
                                               const uint64_t *__restrict__ d_scales,
                                               uint64_t count) {
    uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint64_t *at = dst + d_shares[i * 3] * 3;
    at[0] = d_scales[i * 3 + 0];
    at[1] = d_scales[i * 3 + 1];
    at[2] = d_scales[i * 3 + 2];
}

// One level of the doubling, for every share still doubling at it. The body is
// `eq_expand_level_ext3`'s, with the share's own coordinate and subcube.
extern "C" __global__ void eq_expand_level_shares_ext3(uint64_t *__restrict__ dst,
                                                       const uint64_t *__restrict__ d_shares,
                                                       const uint64_t *__restrict__ d_points,
                                                       uint64_t active, uint64_t level) {
    uint64_t filled = (uint64_t)1 << level;
    uint64_t total = active * filled;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t s = t >> level;
        uint64_t j = t & (filled - 1);
        uint64_t offset = d_shares[s * 3 + 0];
        uint64_t vars = d_shares[s * 3 + 1];
        uint64_t point_at = d_shares[s * 3 + 2];

        uint64_t *cube = dst + offset * 3;
        // Variables go back to front, which is what leaves variable 0 in the
        // high bit.
        Fe3 challenge = load_ext(d_points + (point_at + vars - 1 - level) * 3);
        Fe3 value = load_ext(cube + j * 3);
        Fe3 hi = ext3::mul(value, challenge);
        Fe3 lo = ext3::sub(value, hi);
        uint64_t *at = cube + j * 3;
        at[0] = lo.a;
        at[1] = lo.b;
        at[2] = lo.c;
        uint64_t *up = cube + (j + filled) * 3;
        up[0] = hi.a;
        up[1] = hi.b;
        up[2] = hi.c;
    }
}

// One level of the fraction tree: `p' = p_lo·q_hi + p_hi·q_lo`, `q' = q_lo·q_hi`
// over the halves of the layer below. Out of place — the halves are read by
// threads that write the level above.
extern "C" __global__ void fraction_fold_ext3(const uint64_t *__restrict__ p,
                                              const uint64_t *__restrict__ q, uint64_t half,
                                              uint64_t *__restrict__ p_out,
                                              uint64_t *__restrict__ q_out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    Fe3 p_lo = load_ext(p + j * 3);
    Fe3 p_hi = load_ext(p + (j + half) * 3);
    Fe3 q_lo = load_ext(q + j * 3);
    Fe3 q_hi = load_ext(q + (j + half) * 3);
    Fe3 numerator = ext3::add(ext3::mul(p_lo, q_hi), ext3::mul(p_hi, q_lo));
    Fe3 denominator = ext3::mul(q_lo, q_hi);
    uint64_t *at = p_out + j * 3;
    at[0] = numerator.a;
    at[1] = numerator.b;
    at[2] = numerator.c;
    uint64_t *down = q_out + j * 3;
    down[0] = denominator.a;
    down[1] = denominator.b;
    down[2] = denominator.c;
}

// The same, for a layer whose upper half runs out: the interactions are padded
// up to a power of two with the fraction 0/1, and a fraction nobody wrote is
// one nobody has to store. The lower half is always there — the padding is
// less than half the cube, because the count is rounded *up* to the power of
// two above it.
extern "C" __global__ void fraction_fold_padded_ext3(const uint64_t *__restrict__ p,
                                                     const uint64_t *__restrict__ q, uint64_t half,
                                                     uint64_t real,
                                                     uint64_t *__restrict__ p_out,
                                                     uint64_t *__restrict__ q_out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    Fe3 p_lo = load_ext(p + j * 3);
    Fe3 q_lo = load_ext(q + j * 3);
    uint64_t hi = j + half;
    Fe3 p_hi = {0, 0, 0};
    Fe3 q_hi = {1, 0, 0};
    if (hi < real) {
        p_hi = load_ext(p + hi * 3);
        q_hi = load_ext(q + hi * 3);
    }
    Fe3 numerator = ext3::add(ext3::mul(p_lo, q_hi), ext3::mul(p_hi, q_lo));
    Fe3 denominator = ext3::mul(q_lo, q_hi);
    uint64_t *at = p_out + j * 3;
    at[0] = numerator.a;
    at[1] = numerator.b;
    at[2] = numerator.c;
    uint64_t *down = q_out + j * 3;
    down[0] = denominator.a;
    down[1] = denominator.b;
    down[2] = denominator.c;
}

// The first fold of a base-field table, which lifts it:
//   out[j] = in[j] + r·(in[j + half] − in[j])
// with `in` base and `out` ext3. Later folds stay in the extension and go
// through `sumcheck_fold_ext3` with a single factor.
extern "C" __global__ void mle_fold_base_ext3(const uint64_t *__restrict__ in, uint64_t half,
                                              const uint64_t *__restrict__ r,
                                              uint64_t *__restrict__ out) {
    uint64_t j = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= half) return;
    uint64_t lo = in[j];
    uint64_t delta = goldilocks::sub(in[j + half], lo);
    // `delta·r` with `delta` in the base field, then `lo +` it: the mixed-field
    // shortcuts, bit-identical to the full ext3 ops on the embedding.
    Fe3 scaled = ext3::mul_base(ext3::make(r[0], r[1], r[2]), delta);
    uint64_t *at = out + j * 3;
    at[0] = goldilocks::add(lo, scaled.a);
    at[1] = scaled.b;
    at[2] = scaled.c;
}

// The same for many base tables at once, laid out end to end: one thread per
// (table, index) pair, and the output is one ext3 half per table in the same
// order. Everything above this level is ext3, so the folds that follow are the
// ordinary ones over a list of factors.
extern "C" __global__ void mle_fold_base_ext3_many(const uint64_t *__restrict__ in, uint64_t half,
                                                   uint64_t num_tables,
                                                   const uint64_t *__restrict__ r,
                                                   uint64_t *__restrict__ out) {
    uint64_t total = half * num_tables;
    for (uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; task < total;
         task += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t table = task / half;
        uint64_t j = task - table * half;
        const uint64_t *src = in + table * half * 2;
        uint64_t lo = src[j];
        uint64_t delta = goldilocks::sub(src[j + half], lo);
        Fe3 scaled = ext3::mul_base(ext3::make(r[0], r[1], r[2]), delta);
        uint64_t *at = out + task * 3;
        at[0] = goldilocks::add(lo, scaled.a);
        at[1] = scaled.b;
        at[2] = scaled.c;
    }
}

// Binds the round's variable: `f(j) <- f(j) + r·(f(j + half) − f(j))` for every
// factor, halving the cube. One thread per (factor, index) pair.
extern "C" __global__ void sumcheck_fold_ext3(uint64_t *const *__restrict__ d_factors,
                                              uint64_t half, uint64_t width,
                                              const uint64_t *__restrict__ d_r) {
    uint64_t total = width * half;
    Fe3 r = load_ext(d_r);
    for (uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; task < total;
         task += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t k = task / half;
        uint64_t j = task - k * half;
        uint64_t *column = d_factors[k];
        Fe3 lo = load_ext(column + j * 3);
        Fe3 hi = load_ext(column + (j + half) * 3);
        Fe3 v = ext3::add(lo, ext3::mul(r, ext3::sub(hi, lo)));
        column[j * 3 + 0] = v.a;
        column[j * 3 + 1] = v.b;
        column[j * 3 + 2] = v.c;
    }
}
