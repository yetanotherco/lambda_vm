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
// nothing to wait for. Kernels indexed off these run their thread-0 slice, which
// is why only device *functions* are worth calling through this shim.
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
