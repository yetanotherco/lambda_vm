// Enough of the CUDA language to compile a `.cu` kernel file as ordinary host
// C++, so its arithmetic can be checked without a GPU.
//
// This exists because the GPU parity suite (`crypto/math-cuda/tests/blake3_*.rs`)
// runs only where a GPU does, and per-PR CI has none. Including a kernel through
// this shim turns its device functions into plain functions a host program can
// call, which is all a known-answer test needs.
//
// ⚠ What it CANNOT check, and what therefore still belongs to the GPU tests:
// anything about execution rather than arithmetic — thread/block indexing,
// `__syncthreads` ordering, memory alignment on device, register pressure, and
// whether nvcc accepts the file at all. A kernel that passes here can still be
// wrong on a GPU. Treat this as a lower bound on correctness, never a substitute.
#pragma once

#include <cstdint>

// The execution-space and inlining qualifiers carry no meaning on host.
#define __device__
#define __constant__
#define __forceinline__ inline
#define __global__

// Single-threaded host execution: one thread, block 0, and a barrier that has
// nothing to wait for. Kernel thread coordinates are ordinary mutable globals,
// so a caller can drive them (see `CUDA_HOST_FOR_EACH_THREAD`) and replay a
// whole launch's worth of thread slices one at a time.
#define __syncthreads() ((void)0)
struct CudaHostDim3 {
    unsigned x = 0, y = 0, z = 0;
};
static CudaHostDim3 blockIdx;
static CudaHostDim3 threadIdx;
static CudaHostDim3 cuda_host_block_dim;
#define blockDim cuda_host_block_dim

// `goldilocks.cuh`'s field multiply needs this intrinsic. `blake3.cu` only uses
// `goldilocks::canonical`, but the header compiles as a whole, so supply it.
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
}

// Bit-reverse a 64-bit word. Every leaf kernel derives its row index as
// `__brevll(tid) >> (64 - log_num_rows)`, so replaying one on host needs it.
// Written out rather than deferring to a compiler builtin so the shim stays
// toolchain-neutral.
static inline uint64_t __brevll(uint64_t x) {
    x = ((x & 0x5555555555555555ull) << 1) | ((x >> 1) & 0x5555555555555555ull);
    x = ((x & 0x3333333333333333ull) << 2) | ((x >> 2) & 0x3333333333333333ull);
    x = ((x & 0x0F0F0F0F0F0F0F0Full) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0Full);
    x = ((x & 0x00FF00FF00FF00FFull) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFull);
    x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFull);
    return (x << 32) | (x >> 32);
}

// Replay a `__global__` kernel once per thread index, sequentially, by driving
// the shim's thread coordinates. A kernel computing
// `tid = blockIdx.x * blockDim.x + threadIdx.x` sees `tid = i` on iteration `i`,
// so a whole launch can be reproduced on host:
//
//     CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) some_leaf_kernel(args...);
//
// ⚠ Only valid for kernels whose threads are independent — which the leaf
// kernels are (one thread, one leaf, disjoint output) and the Merkle *tail* is
// not. It says nothing about `__syncthreads` ordering, races or occupancy.
#define CUDA_HOST_FOR_EACH_THREAD(i, n)                                          \
    for (unsigned i = 0;                                                         \
         i < (unsigned)(n) &&                                                    \
         (blockIdx.x = 0, blockDim.x = 0, threadIdx.x = i, true);                \
         ++i)
