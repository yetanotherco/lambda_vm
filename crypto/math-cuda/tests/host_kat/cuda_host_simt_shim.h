// Enough of the CUDA language to run a kernel that COOPERATES ACROSS LANES as
// host C++: every thread of a block is a real `std::thread`, `__shfl_sync` is an
// exchange through a per-warp slot array between two barriers, `__syncthreads`
// is a block-wide barrier, and the atomics are host atomics.
//
// `cuda_host_shim.h` replays a launch one thread at a time, which is exact for
// kernels whose threads never talk to each other and meaningless for the ones
// that do (a shuffle from lane 5 needs lane 5 to be running). This shim exists
// for the second kind: the half-warp permutation and the work-queue grind.
//
// ⚠ What it CANNOT check, as for the single-thread shim: nvcc acceptance,
// register pressure, timing, and anything the hardware does that a converged
// warp of host threads does not (it assumes every lane of a warp reaches every
// `__shfl_sync`, which is what the kernels guarantee and what the hardware
// requires). Blocks run one after another, not concurrently.
#pragma once

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#define __device__
#define __constant__
#define __forceinline__ inline
#define __global__

struct CudaHostDim3 {
    unsigned x = 0, y = 0, z = 0;
};
static thread_local CudaHostDim3 threadIdx;
static thread_local CudaHostDim3 blockIdx;
static CudaHostDim3 cuda_host_block_dim;
static CudaHostDim3 cuda_host_grid_dim;
#define blockDim cuda_host_block_dim
#define gridDim cuda_host_grid_dim

// A reusable barrier for a fixed party count (C++17 has none).
class SimtBarrier {
  public:
    explicit SimtBarrier(unsigned parties) : parties_(parties) {}
    void arrive_and_wait() {
        std::unique_lock<std::mutex> lock(m_);
        const unsigned gen = generation_;
        if (++arrived_ == parties_) {
            arrived_ = 0;
            ++generation_;
            cv_.notify_all();
        } else {
            cv_.wait(lock, [&] { return generation_ != gen; });
        }
    }

  private:
    std::mutex m_;
    std::condition_variable cv_;
    unsigned parties_;
    unsigned arrived_ = 0;
    unsigned generation_ = 0;
};

struct SimtWarp {
    explicit SimtWarp(unsigned lanes) : bar(lanes) {}
    SimtBarrier bar;
    unsigned long long slots[32] = {};
};

// The running block's warps and block barrier (blocks run one at a time).
static std::vector<std::unique_ptr<SimtWarp>> *simt_warps = nullptr;
static SimtBarrier *simt_block_barrier = nullptr;

inline unsigned long long __shfl_sync(unsigned /*mask*/, unsigned long long v, int src) {
    SimtWarp &w = *(*simt_warps)[threadIdx.x / 32];
    w.slots[threadIdx.x & 31u] = v;
    w.bar.arrive_and_wait();
    const unsigned long long r = w.slots[(unsigned)src & 31u];
    w.bar.arrive_and_wait();
    return r;
}

#define __syncthreads() (simt_block_barrier->arrive_and_wait())

inline unsigned long long atomicAdd(unsigned long long *p, unsigned long long v) {
    return __atomic_fetch_add(p, v, __ATOMIC_SEQ_CST);
}

inline unsigned long long atomicMin(unsigned long long *p, unsigned long long v) {
    unsigned long long old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v < old &&
           !__atomic_compare_exchange_n(p, &old, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    return old;
}

inline unsigned long long atomicMax(unsigned long long *p, unsigned long long v) {
    unsigned long long old = __atomic_load_n(p, __ATOMIC_SEQ_CST);
    while (v > old &&
           !__atomic_compare_exchange_n(p, &old, v, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
    }
    return old;
}

// `goldilocks.cuh`'s multiply and the leaf kernels' row index, as in the
// single-thread shim.
static inline uint64_t __umul64hi(uint64_t a, uint64_t b) {
    return (uint64_t)(((unsigned __int128)a * (unsigned __int128)b) >> 64);
}

static inline uint64_t __brevll(uint64_t x) {
    x = ((x & 0x5555555555555555ull) << 1) | ((x >> 1) & 0x5555555555555555ull);
    x = ((x & 0x3333333333333333ull) << 2) | ((x >> 2) & 0x3333333333333333ull);
    x = ((x & 0x0F0F0F0F0F0F0F0Full) << 4) | ((x >> 4) & 0x0F0F0F0F0F0F0F0Full);
    x = ((x & 0x00FF00FF00FF00FFull) << 8) | ((x >> 8) & 0x00FF00FF00FF00FFull);
    x = ((x & 0x0000FFFF0000FFFFull) << 16) | ((x >> 16) & 0x0000FFFF0000FFFFull);
    return (x << 32) | (x >> 32);
}

// Launch `kernel` over `grid` blocks of `block` threads: each block's threads
// run concurrently as real threads; blocks run in order. `block` must be a
// multiple of 32, as the warp kernels require.
inline void simt_launch(unsigned grid, unsigned block, const std::function<void()> &kernel) {
    cuda_host_grid_dim.x = grid;
    cuda_host_block_dim.x = block;
    for (unsigned b = 0; b < grid; ++b) {
        std::vector<std::unique_ptr<SimtWarp>> warps;
        for (unsigned w = 0; w < block / 32; ++w) warps.emplace_back(new SimtWarp(32));
        SimtBarrier block_barrier(block);
        simt_warps = &warps;
        simt_block_barrier = &block_barrier;
        std::vector<std::thread> threads;
        threads.reserve(block);
        for (unsigned t = 0; t < block; ++t) {
            threads.emplace_back([&kernel, b, t] {
                blockIdx.x = b;
                threadIdx.x = t;
                kernel();
            });
        }
        for (auto &th : threads) th.join();
        simt_warps = nullptr;
        simt_block_barrier = nullptr;
    }
}
