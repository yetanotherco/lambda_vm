// One sumcheck round over the Boolean hypercube, for a batch given as
// straight-line code over its factors (`multilinear::program::Program`, lowered
// by `crypto/multilinear/src/gpu.rs`).
//
// Design:
//   * One thread per cube index, grid-stride, so the launch is fixed at any
//     size. The interpolation nodes `t` are the inner loop: a factor's `lo` and
//     `hi` are read once per node but stay in cache across them, which is why
//     the loops are this way round and not the other.
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
                                            const uint64_t *__restrict__ d_factors,
                                            uint64_t factor_stride, uint64_t j, uint64_t half,
                                            const Fe3 &t, uint64_t *slots, uint64_t stride,
                                            uint32_t root_slot) {
    for (uint64_t i = 0; i < num_nodes; i++) {
        Node nd = load_node(d_nodes, i);
        switch (nd.op) {
        case OP_VAR: {
            const uint64_t *column = d_factors + (uint64_t)nd.a * factor_stride * 3;
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

extern "C" __global__ void sumcheck_round_ext3(
    // factors: factor `k` at cube index `j`, component `c`, is
    // `d_factors[(k*factor_stride + j)*3 + c]`
    const uint64_t *__restrict__ d_factors, uint64_t factor_stride,
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
    uint64_t tid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t num_threads = (uint64_t)gridDim.x * blockDim.x;
    uint64_t *slots = d_slots + tid;

    Fe3 acc[MAX_NODES];
    for (uint32_t ti = 0; ti < num_t; ti++) {
        acc[ti] = ext3::make(0, 0, 0);
    }

    for (uint64_t j = tid; j < half; j += num_threads) {
        for (uint32_t ti = 0; ti < num_t; ti++) {
            Fe3 t = load_ext(d_t + (uint64_t)ti * 3);
            Fe3 v = eval_program(d_nodes, num_nodes, d_consts, d_factors, factor_stride, j, half, t,
                                 slots, num_threads, root_slot);
            acc[ti] = ext3::add(acc[ti], v);
        }
    }

    // One node at a time through the same shared buffer: the round's degree is
    // a handful, and a buffer per node would bound the block size instead.
    extern __shared__ uint64_t shared[];
    for (uint32_t ti = 0; ti < num_t; ti++) {
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

// Binds the round's variable: `f(j) <- f(j) + r·(f(j + half) − f(j))` for every
// factor, halving the cube. One thread per (factor, index) pair.
extern "C" __global__ void sumcheck_fold_ext3(uint64_t *__restrict__ d_factors,
                                              uint64_t factor_stride, uint64_t half, uint64_t width,
                                              const uint64_t *__restrict__ d_r) {
    uint64_t total = width * half;
    Fe3 r = load_ext(d_r);
    for (uint64_t task = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x; task < total;
         task += (uint64_t)gridDim.x * blockDim.x) {
        uint64_t k = task / half;
        uint64_t j = task - k * half;
        uint64_t *column = d_factors + k * factor_stride * 3;
        Fe3 lo = load_ext(column + j * 3);
        Fe3 hi = load_ext(column + (j + half) * 3);
        Fe3 v = ext3::add(lo, ext3::mul(r, ext3::sub(hi, lo)));
        column[j * 3 + 0] = v.a;
        column[j * 3 + 1] = v.b;
        column[j * 3 + 2] = v.c;
    }
}
