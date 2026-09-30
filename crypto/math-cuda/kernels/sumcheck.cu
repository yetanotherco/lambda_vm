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
//
// `INT` (D-ARGUE S1-1, `LAMBDA_VM_ARGUE_INT_NODES`): the node is a base-field
// integer `(k, 0, 0)` — every round's nodes are `1..=D` — so `t·(hi − lo)` is
// the componentwise `k·(hi − lo)`: three products where the full multiply
// spends nine. The same value; the host only takes this variant when every
// node's upper limbs are zero.
template <bool INT>
__device__ __forceinline__ Fe3 eval_program_t(const uint64_t *__restrict__ d_nodes,
                                              uint64_t num_nodes,
                                              const uint64_t *__restrict__ d_consts,
                                              const uint64_t *const *__restrict__ d_factors,
                                              uint64_t j, uint64_t half, const Fe3 &t,
                                              uint64_t *slots, uint64_t stride,
                                              uint32_t root_slot) {
    for (uint64_t i = 0; i < num_nodes; i++) {
        Node nd = load_node(d_nodes, i);
        switch (nd.op) {
        case OP_VAR: {
            const uint64_t *column = d_factors[nd.a];
            Fe3 lo = load_ext(column + j * 3);
            Fe3 hi = load_ext(column + (j + half) * 3);
            Fe3 v = INT ? ext3::add(lo, ext3::mul_base(ext3::sub(hi, lo), t.a))
                        : ext3::add(lo, ext3::mul(t, ext3::sub(hi, lo)));
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

__device__ __forceinline__ Fe3 eval_program(const uint64_t *__restrict__ d_nodes,
                                            uint64_t num_nodes,
                                            const uint64_t *__restrict__ d_consts,
                                            const uint64_t *const *__restrict__ d_factors,
                                            uint64_t j, uint64_t half, const Fe3 &t,
                                            uint64_t *slots, uint64_t stride, uint32_t root_slot) {
    return eval_program_t<false>(d_nodes, num_nodes, d_consts, d_factors, j, half, t, slots,
                                 stride, root_slot);
}

// Capped at the block size the launcher uses and two blocks per SM: without a
// bound the compiler spends 67 registers a thread and occupancy stops at half.
template <bool INT>
__device__ __forceinline__ void sumcheck_round_body(
    const uint64_t *const *__restrict__ d_factors, uint64_t half,
    const uint64_t *__restrict__ d_nodes, uint64_t num_nodes,
    const uint64_t *__restrict__ d_consts, uint32_t root_slot,
    const uint64_t *__restrict__ d_t, uint32_t num_t, uint64_t *__restrict__ d_slots,
    uint64_t *__restrict__ d_partials);

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
    sumcheck_round_body<false>(d_factors, half, d_nodes, num_nodes, d_consts, root_slot, d_t,
                               num_t, d_slots, d_partials);
}

// The same round with integer nodes (D-ARGUE S1-1): see `eval_program_t`.
extern "C" __global__ __launch_bounds__(256, 2) void sumcheck_round_ext3_int(
    const uint64_t *const *__restrict__ d_factors, uint64_t half,
    const uint64_t *__restrict__ d_nodes, uint64_t num_nodes,
    const uint64_t *__restrict__ d_consts, uint32_t root_slot,
    const uint64_t *__restrict__ d_t, uint32_t num_t, uint64_t *__restrict__ d_slots,
    uint64_t *__restrict__ d_partials) {
    sumcheck_round_body<true>(d_factors, half, d_nodes, num_nodes, d_consts, root_slot, d_t,
                              num_t, d_slots, d_partials);
}

template <bool INT>
__device__ __forceinline__ void sumcheck_round_body(
    const uint64_t *const *__restrict__ d_factors, uint64_t half,
    const uint64_t *__restrict__ d_nodes, uint64_t num_nodes,
    const uint64_t *__restrict__ d_consts, uint32_t root_slot,
    const uint64_t *__restrict__ d_t, uint32_t num_t, uint64_t *__restrict__ d_slots,
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
            Fe3 v = eval_program_t<INT>(d_nodes, num_nodes, d_consts, d_factors, j, half, t,
                                        slots, num_threads, root_slot);
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

// The claim reduce's batched column for one offset, out of the epoch's resident
// columns: `out[row] = Σ_m weight[m]·column[member[m]][row]`. The columns are
// base-field and laid end to end, `rows` each, from the start of the table's run;
// `members` is each term's column in that run and `weights` its ext3 weight.
//
// One thread per row, looping over the members: the threads of a warp read one
// column's consecutive rows, and every thread reads the same weight. The sum is
// the host's (`claim_reduce::batched_column`) term for term — base × ext3 is the
// componentwise product either way — in a different order, which a field does
// not see.
extern "C" __global__ void batched_column_ext3(const uint64_t *__restrict__ columns,
                                               uint64_t rows,
                                               const uint64_t *__restrict__ members,
                                               const uint64_t *__restrict__ weights,
                                               uint64_t num_members,
                                               uint64_t *__restrict__ out) {
    for (uint64_t row = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; row < rows;
         row += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 acc = ext3::zero();
        for (uint64_t m = 0; m < num_members; ++m) {
            uint64_t value = columns[members[m] * rows + row];
            acc = ext3::add(acc, ext3::mul_base(load_ext(weights + m * 3), value));
        }
        uint64_t *at = out + row * 3;
        at[0] = acc.a;
        at[1] = acc.b;
        at[2] = acc.c;
    }
}

// A session's end, read back in one copy: factor `k`'s first `cells` ext3
// values — what the folds left of it, at its own base — gathered to
// `out[k·cells·3 ..]`. One thread per u64, so a warp reads one factor's
// consecutive words and writes consecutive words.
extern "C" __global__ void gather_factor_heads_ext3(uint64_t *const *__restrict__ d_factors,
                                                    uint64_t width, uint64_t cells,
                                                    uint64_t *__restrict__ out) {
    uint64_t span = cells * 3;
    uint64_t total = width * span;
    for (uint64_t word = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; word < total;
         word += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t k = word / span;
        out[word] = d_factors[k][word - k * span];
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

// ── D-ARGUE stage 1: the fused zerocheck (S1-2, S1-4, S1-5) ──────────────────
//
// A table's zerocheck batch is `eq(r,x)·C(x) + eq(ρ,x)·L(x)`, `C = Σ β_i·sel_i·root_i`
// the constraint part and `L` the bus's two rules as one affine column
// (`multilinear::fused`, the host reference these kernels are checked against).
// Rounds 0 and 1 come from one base-field pass over the 4-row groups
// `(a, b, x'')` — rows `a·2q + b·q + x''`, `q` a quarter of the cube — on the
// grid `{0..d}²` (`zc_grid01`, `zc_bus_u`); the factors are then folded by both
// challenges at once into a quarter-size ext3 buffer (`zc_fold2`), `L` is built
// there (`zc_bus_column`), and the later rounds walk `C` alone at the integer
// nodes `{0, 2..d}` under split `eq` weights (`zc_round_gruen`).
//
// The constraint program is the table's base DAG with one extra op, `ACC`:
// `acc += β[res]·(value[a]·factor[b])` (`b == NO_SELECTOR`: no factor), placed
// right after each root — so the roots are summed as they are made and never
// held to the end. Op tags and the packing MUST stay in sync with
// `crypto/multilinear/src/gpu_fused.rs`.

#define OP_ACC 6u
#define NO_SELECTOR 0xFFFFFFFFu

// A factor's value on the grid point `(a, b)` of group `x`, from its four rows
// — base limbs of the lifted ext3 factor, whose upper limbs are zero before
// the first fold (the base-field precondition). Bilinear, exact.
__device__ __forceinline__ uint64_t grid_value(const uint64_t *__restrict__ column, uint64_t x,
                                               uint64_t q, uint64_t a, uint64_t b) {
    uint64_t f00 = column[x * 3];
    uint64_t f01 = column[(x + q) * 3];
    uint64_t f10 = column[(x + 2 * q) * 3];
    uint64_t f11 = column[(x + 3 * q) * 3];
    uint64_t da = goldilocks::sub(f10, f00);
    uint64_t db = goldilocks::sub(f01, f00);
    uint64_t dab = goldilocks::sub(goldilocks::sub(f11, f10), db);
    uint64_t v = f00;
    if (a) v = goldilocks::add(v, goldilocks::mul(da, a));
    if (b) v = goldilocks::add(v, goldilocks::mul(db, b));
    if (a && b) v = goldilocks::add(v, goldilocks::mul(dab, goldilocks::mul(a, b)));
    return v;
}

// One thread per (group, grid point): `gridDim.y` is the point count and each
// block owns one point. `T(a, b) = Σ_x eq_r[x]·C(a, b, x)` summed per block
// into `d_partials[(point·gridDim.x + block)·3]`. A corner (`a, b < 2`) is on
// the list only when the corners are checked or kept. Checked
// (`keep_corners == 0`): its `C` must be zero on every row, the first row that
// is not is kept in `d_violation` (atomicMin), and it adds nothing to `T`.
// Kept (`keep_corners == 1`, random columns in a test): it adds to `T` like
// any point.
extern "C" __global__ __launch_bounds__(256) void zc_grid01(
    const uint64_t *const *__restrict__ d_factors, uint64_t q,
    const uint64_t *__restrict__ d_nodes, uint64_t num_nodes,
    const uint64_t *__restrict__ d_consts, const uint64_t *__restrict__ d_betas,
    const uint64_t *__restrict__ d_points, const uint64_t *__restrict__ d_eq_r,
    uint64_t *__restrict__ d_slots, uint64_t *__restrict__ d_partials,
    unsigned long long *__restrict__ d_violation, uint32_t keep_corners) {
    uint64_t tid = ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * gridDim.y * blockDim.x;
    uint64_t *slots = d_slots + tid;
    uint64_t point = d_points[blockIdx.y];
    uint64_t a = point & 0xFFFFFFFFull;
    uint64_t b = point >> 32;
    bool corner = a < 2 && b < 2;

    Fe3 t_acc = ext3::zero();
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < q;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 c = ext3::zero();
        for (uint64_t i = 0; i < num_nodes; i++) {
            Node nd = load_node(d_nodes, i);
            uint64_t *out = slots + (uint64_t)nd.res * num_threads;
            switch (nd.op) {
            case OP_VAR:
                *out = grid_value(d_factors[nd.a], x, q, a, b);
                break;
            case OP_FIXED:
                *out = d_consts[nd.a];
                break;
            case OP_ADD:
                *out = goldilocks::add(slots[(uint64_t)nd.a * num_threads],
                                       slots[(uint64_t)nd.b * num_threads]);
                break;
            case OP_SUB:
                *out = goldilocks::sub(slots[(uint64_t)nd.a * num_threads],
                                       slots[(uint64_t)nd.b * num_threads]);
                break;
            case OP_MUL:
                *out = goldilocks::mul(slots[(uint64_t)nd.a * num_threads],
                                       slots[(uint64_t)nd.b * num_threads]);
                break;
            case OP_NEG:
                *out = goldilocks::neg(slots[(uint64_t)nd.a * num_threads]);
                break;
            case OP_ACC: {
                uint64_t v = slots[(uint64_t)nd.a * num_threads];
                if (nd.b != NO_SELECTOR) {
                    v = goldilocks::mul(v, grid_value(d_factors[nd.b], x, q, a, b));
                }
                c = ext3::add(c, ext3::mul_base(load_ext(d_betas + (uint64_t)nd.res * 3), v));
                break;
            }
            default:
                break;
            }
        }
        if (corner && !keep_corners) {
            Fe3 k = ext3::canonical(c);
            if (k.a | k.b | k.c) {
                atomicMin(d_violation, (unsigned long long)(a * 2 * q + b * q + x));
            }
            continue;
        }
        t_acc = ext3::add(t_acc, ext3::mul(load_ext(d_eq_r + x * 3), c));
    }

    extern __shared__ uint64_t shared[];
    shared[threadIdx.x * 3 + 0] = t_acc.a;
    shared[threadIdx.x * 3 + 1] = t_acc.b;
    shared[threadIdx.x * 3 + 2] = t_acc.c;
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
        uint64_t at = ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * 3;
        d_partials[at + 0] = shared[0];
        d_partials[at + 1] = shared[1];
        d_partials[at + 2] = shared[2];
    }
}

// `U(b, c) = Σ_x eq_rho[x]·L(b, c, x)` for `(b, c) ∈ {0,1}²`, `L = a₀ + Σ a_k·f_k`
// from each term's four base rows. One thread per group; the block's four sums
// go to `d_partials[(bc·gridDim.x + block)·3]`, `bc = 2b + c`.
extern "C" __global__ __launch_bounds__(256) void zc_bus_u(
    const uint64_t *const *__restrict__ d_factors, uint64_t q,
    const uint32_t *__restrict__ d_term_slots, const uint64_t *__restrict__ d_term_coeffs,
    uint64_t num_terms, const uint64_t *__restrict__ d_constant,
    const uint64_t *__restrict__ d_eq_rho, uint64_t *__restrict__ d_partials) {
    Fe3 u[4] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
    Fe3 a0 = load_ext(d_constant);
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < q;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 l[4] = {a0, a0, a0, a0};
        for (uint64_t k = 0; k < num_terms; k++) {
            const uint64_t *column = d_factors[d_term_slots[k]];
            Fe3 coeff = load_ext(d_term_coeffs + k * 3);
            for (uint32_t bc = 0; bc < 4; bc++) {
                l[bc] = ext3::add(l[bc], ext3::mul_base(coeff, column[(x + bc * q) * 3]));
            }
        }
        Fe3 w = load_ext(d_eq_rho + x * 3);
        for (uint32_t bc = 0; bc < 4; bc++) {
            u[bc] = ext3::add(u[bc], ext3::mul(w, l[bc]));
        }
    }
    extern __shared__ uint64_t shared[];
    for (uint32_t bc = 0; bc < 4; bc++) {
        shared[threadIdx.x * 3 + 0] = u[bc].a;
        shared[threadIdx.x * 3 + 1] = u[bc].b;
        shared[threadIdx.x * 3 + 2] = u[bc].c;
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
            uint64_t at = ((uint64_t)bc * gridDim.x + blockIdx.x) * 3;
            d_partials[at + 0] = shared[0];
            d_partials[at + 1] = shared[1];
            d_partials[at + 2] = shared[2];
        }
        __syncthreads();
    }
}

// Both folds at once, base to ext3: `out_k[x] = Σ_{a,b} w[2a+b]·f_k(a, b, x)` over
// the four rows of each group, into factor `k`'s quarter-size slab of `out`.
// One thread per (factor, group).
extern "C" __global__ void zc_fold2(const uint64_t *const *__restrict__ d_factors, uint64_t q,
                                    uint64_t width, const uint64_t *__restrict__ d_w,
                                    uint64_t *__restrict__ out) {
    Fe3 w0 = load_ext(d_w), w1 = load_ext(d_w + 3), w2 = load_ext(d_w + 6),
        w3 = load_ext(d_w + 9);
    uint64_t total = width * q;
    for (uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; task < total;
         task += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t k = task / q;
        uint64_t x = task - k * q;
        const uint64_t *column = d_factors[k];
        Fe3 v = ext3::mul_base(w0, column[x * 3]);
        v = ext3::add(v, ext3::mul_base(w1, column[(x + q) * 3]));
        v = ext3::add(v, ext3::mul_base(w2, column[(x + 2 * q) * 3]));
        v = ext3::add(v, ext3::mul_base(w3, column[(x + 3 * q) * 3]));
        uint64_t *at = out + task * 3;
        at[0] = v.a;
        at[1] = v.b;
        at[2] = v.c;
    }
}

// `L[x] = a₀ + Σ a_k·F_k[x]` over the folded (ext3) factors. One thread per row.
extern "C" __global__ void zc_bus_column(const uint64_t *const *__restrict__ d_factors,
                                         uint64_t rows, const uint32_t *__restrict__ d_term_slots,
                                         const uint64_t *__restrict__ d_term_coeffs,
                                         uint64_t num_terms,
                                         const uint64_t *__restrict__ d_constant,
                                         uint64_t *__restrict__ out) {
    Fe3 a0 = load_ext(d_constant);
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < rows;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 acc = a0;
        for (uint64_t k = 0; k < num_terms; k++) {
            acc = ext3::add(acc, ext3::mul(load_ext(d_term_coeffs + k * 3),
                                           load_ext(d_factors[d_term_slots[k]] + x * 3)));
        }
        uint64_t *at = out + x * 3;
        at[0] = acc.a;
        at[1] = acc.b;
        at[2] = acc.c;
    }
}

// `eq(a_{>j+1}, ·)` from `eq(a_{>j}, ·)` in place: `t[x] += t[x + half]`.
extern "C" __global__ void zc_halve(uint64_t *__restrict__ table, uint64_t half) {
    for (uint64_t x = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; x < half;
         x += (uint64_t)gridDim.x * blockDim.x) {
        Fe3 v = ext3::add(load_ext(table + x * 3), load_ext(table + (x + half) * 3));
        table[x * 3 + 0] = v.a;
        table[x * 3 + 1] = v.b;
        table[x * 3 + 2] = v.c;
    }
}

// A later round (S1-4 with S1-1): `gridDim.y` is `num_a + 2` and each block owns
// one of them. Block row `y < num_a` sums `eq_r[j]·C(j, node[y])`, every factor
// read at the integer node `k = node[y]` as `lo + k·(hi − lo)` (`lo` itself at
// `k = 0`); row `num_a + b` sums `eq_rho[j]·L[j + b·half]`. The per-block sums go
// to `d_partials[(y·gridDim.x + block)·3]`.
extern "C" __global__ __launch_bounds__(256, 2) void zc_round_gruen(
    const uint64_t *const *__restrict__ d_factors, uint64_t half,
    const uint64_t *__restrict__ d_nodes, uint64_t num_nodes,
    const uint64_t *__restrict__ d_consts, const uint64_t *__restrict__ d_betas,
    const uint32_t *__restrict__ d_node_ints, uint32_t num_a,
    const uint64_t *__restrict__ d_eq_r, const uint64_t *__restrict__ d_eq_rho,
    const uint64_t *__restrict__ d_l, uint64_t *__restrict__ d_slots,
    uint64_t *__restrict__ d_partials) {
    uint64_t tid = ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * gridDim.y * blockDim.x;
    uint64_t *slots = d_slots + tid;
    uint64_t start = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t step = (uint64_t)gridDim.x * blockDim.x;

    Fe3 acc = ext3::zero();
    if (blockIdx.y < num_a) {
        uint64_t k = d_node_ints[blockIdx.y];
        for (uint64_t j = start; j < half; j += step) {
            Fe3 c = ext3::zero();
            for (uint64_t i = 0; i < num_nodes; i++) {
                Node nd = load_node(d_nodes, i);
                switch (nd.op) {
                case OP_VAR: {
                    const uint64_t *column = d_factors[nd.a];
                    Fe3 lo = load_ext(column + j * 3);
                    Fe3 v = lo;
                    if (k) {
                        Fe3 hi = load_ext(column + (j + half) * 3);
                        v = ext3::add(lo, ext3::mul_base(ext3::sub(hi, lo), k));
                    }
                    store_slot(slots, num_threads, nd.res, v);
                    break;
                }
                case OP_FIXED:
                    store_slot(slots, num_threads, nd.res, load_ext(d_consts + (uint64_t)nd.a * 3));
                    break;
                case OP_ADD:
                    store_slot(slots, num_threads, nd.res,
                               ext3::add(load_slot(slots, num_threads, nd.a),
                                         load_slot(slots, num_threads, nd.b)));
                    break;
                case OP_SUB:
                    store_slot(slots, num_threads, nd.res,
                               ext3::sub(load_slot(slots, num_threads, nd.a),
                                         load_slot(slots, num_threads, nd.b)));
                    break;
                case OP_MUL:
                    store_slot(slots, num_threads, nd.res,
                               ext3::mul(load_slot(slots, num_threads, nd.a),
                                         load_slot(slots, num_threads, nd.b)));
                    break;
                case OP_NEG:
                    store_slot(slots, num_threads, nd.res,
                               ext3::neg(load_slot(slots, num_threads, nd.a)));
                    break;
                case OP_ACC: {
                    Fe3 v = load_slot(slots, num_threads, nd.a);
                    if (nd.b != NO_SELECTOR) {
                        const uint64_t *column = d_factors[nd.b];
                        Fe3 lo = load_ext(column + j * 3);
                        Fe3 s = lo;
                        if (k) {
                            Fe3 hi = load_ext(column + (j + half) * 3);
                            s = ext3::add(lo, ext3::mul_base(ext3::sub(hi, lo), k));
                        }
                        v = ext3::mul(v, s);
                    }
                    c = ext3::add(c, ext3::mul(load_ext(d_betas + (uint64_t)nd.res * 3), v));
                    break;
                }
                default:
                    break;
                }
            }
            acc = ext3::add(acc, ext3::mul(load_ext(d_eq_r + j * 3), c));
        }
    } else {
        uint64_t b = blockIdx.y - num_a;
        for (uint64_t j = start; j < half; j += step) {
            acc = ext3::add(acc, ext3::mul(load_ext(d_eq_rho + j * 3),
                                           load_ext(d_l + (j + b * half) * 3)));
        }
    }

    extern __shared__ uint64_t shared[];
    shared[threadIdx.x * 3 + 0] = acc.a;
    shared[threadIdx.x * 3 + 1] = acc.b;
    shared[threadIdx.x * 3 + 2] = acc.c;
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
        uint64_t at = ((uint64_t)blockIdx.y * gridDim.x + blockIdx.x) * 3;
        d_partials[at + 0] = shared[0];
        d_partials[at + 1] = shared[1];
        d_partials[at + 2] = shared[2];
    }
}

// ── a GKR layer's rounds with Gruen's split (D-ARGUE S1-3, `crate::gkr::GruenLayer`) ─────────────
//
// The layer relation is `eq(u, x)·h(x)`, `h = p_lo·q_hi + p_hi·q_lo + λ·q_lo·q_hi`, over `m`
// variables. Round `j` needs only `H(t) = Σ_{x'} eq(u_{>j}, x')·h(s_{<j}, t, x')` at two nodes: the
// host puts `eq(u_{≤j})` back in and takes the third value from the claim. The weight is never a
// table the size of the layer: the last `L` variables' `eq` (`e_lo`, the host tail's) times the
// variables still to bind on the card (`e_hi`, one level per round), both built once a layer.

// Every `eq` level a layer's device rounds read, from its point `u` (`J + L` ext3 values): first
// `eq(u_{J..m−1}, ·)` over `2^L` cells, then for `j = 0..J−1` the level `eq(u_{j+1..J−1}, ·)` over
// `2^{J−1−j}` cells, level `j` at `2^L + 2^J − 2^{J−j}`. Variable `first` sits in the high bit, the
// indexing every table here folds on. One thread per cell, each a product of its bits' factors.
extern "C" __global__ void gkr_eq_levels_ext3(const uint64_t *__restrict__ point, uint32_t low,
                                              uint32_t rounds,
                                              uint64_t *__restrict__ out) {
    uint64_t lo_cells = (uint64_t)1 << low;
    uint64_t total = lo_cells + ((uint64_t)1 << rounds) - 1;
    for (uint64_t t = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total;
         t += (uint64_t)gridDim.x * blockDim.x) {
        uint32_t first, bits;
        uint64_t index;
        if (t < lo_cells) {
            first = rounds;
            bits = low;
            index = t;
        } else {
            uint64_t r = t - lo_cells;
            uint32_t j = 0;
            while (r >= ((uint64_t)1 << (rounds - 1 - j))) {
                r -= (uint64_t)1 << (rounds - 1 - j);
                j++;
            }
            first = j + 1;
            bits = rounds - 1 - j;
            index = r;
        }
        Fe3 acc = ext3::one();
        for (uint32_t i = 0; i < bits; i++) {
            Fe3 u = load_ext(point + (uint64_t)(first + i) * 3);
            bool set = (index >> (bits - 1 - i)) & 1;
            acc = ext3::mul(acc, set ? u : ext3::sub(ext3::one(), u));
        }
        uint64_t *at = out + t * 3;
        at[0] = acc.a;
        at[1] = acc.b;
        at[2] = acc.c;
    }
}

// `h` at one node, from the four halves' values there.
__device__ __forceinline__ Fe3 gkr_h(const Fe3 &a, const Fe3 &b, const Fe3 &c, const Fe3 &d,
                                     const Fe3 &lambda) {
    Fe3 cd = ext3::mul(c, d);
    return ext3::add(ext3::add(ext3::mul(a, d), ext3::mul(b, c)), ext3::mul(lambda, cd));
}

// One round over `quarter` cube indices. The four halves are `p`, `p + half`, `q`, `q + half`
// (`half` ext3 cells each). With `fold`, they still hold the previous round's cube, twice as long,
// and each is bound to `s` on the way in — `f(x) + s·(f(x + 2Q) − f(x))` at `x` and `x + Q` — and
// written back where the next round reads it: thread `x` owns cells `x, x+Q, x+2Q, x+3Q` and no
// one else touches them. Then `h` at `t = 1` (`hi`), `t = 2` (`2·hi − lo`) and, with `want_h0`,
// `t = 0` (`lo`), each weighted by `e_hi[x >> low]·e_lo[x mod 2^low]`. Block partials go to
// `d_partials[(row·gridDim.x + block)·3]`, rows `H(1), H(2), H(0)`.
extern "C" __global__ __launch_bounds__(256) void gkr_round_gruen(
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
extern "C" __global__ void gkr_gruen_finish(uint64_t *__restrict__ p, uint64_t *__restrict__ q,
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
