// ZisK's production LDE + Merkle commit on one matrix (GAP-W3/K0): the same
// calls and kernels cargo-zisk 1.3.0-alpha runs, built from the published
// proofman-starks-src 1.3.0-alpha sources by zisk_profile.sh.
// Calls mirror extendAndMerkelize_inplace (starks_gpu.cu:304-307):
//   ntt.LDE(dst,0,src,0,nBits,nBitsExt,nCols,timer,stream,true,pNodes,numNodes)
//   buildMerkleTreeGPU(arity, pNodes, dst, nCols, NExt, ColMajor, stream)
// Usage: zk_k0 <nBits> <blowupBits> <nCols> <p1|b3|none> <iters>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include "goldilocks_base_field.hpp"
#include "ntt_goldilocks.cuh"
#include "poseidon_goldilocks.cuh"
#include "blake3_goldilocks.cuh"
#include "gpu_timer.cuh"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(2);} } while (0)

static __global__ void initK(gl64_t *d, uint64_t n) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) d[i] = gl64_t((i * 0x9E3779B97F4A7C15ULL) % 0xFFFFFFFF00000001ULL);
}

static uint64_t treeElems(uint64_t n, uint64_t arity) {
    // getNumNodes: CAPACITY(4) elements per node, levels padded to arity.
    uint64_t nodes = n, total = n;
    while (nodes > 1) {
        uint64_t extra = (arity - (nodes % arity)) % arity;
        total += extra;
        nodes = (nodes + arity - 1) / arity;
        total += nodes;
    }
    return total * 4;
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    if (argc < 6) { fprintf(stderr, "usage\n"); return 1; }
    uint64_t nBits = atoi(argv[1]), bb = atoi(argv[2]), nCols = atoi(argv[3]);
    std::string hash = argv[4];
    int iters = atoi(argv[5]);
    uint64_t nBitsExt = nBits + bb, N = 1ULL << nBits, NE = 1ULL << nBitsExt;
    uint32_t gpu = 0;
    CK(cudaSetDevice(0));
    cudaStream_t s; CK(cudaStreamCreate(&s));
    if (hash == "p1") PoseidonGoldilocksGPU<16>::initConstants(&gpu, 1);
    uint64_t arity = hash == "p1" ? 4 : 2;
    uint64_t te = treeElems(NE, arity);
    gl64_t *src, *dst; uint64_t *tree;
    CK(cudaMalloc(&src, N * nCols * 8));
    CK(cudaMalloc(&dst, NE * nCols * 8));
    CK(cudaMalloc(&tree, te * 8));
    initK<<<(N * nCols + 255) / 256, 256, 0, s>>>(src, N * nCols);
    CK(cudaStreamSynchronize(s));
    NTTGoldilocksGPU ntt;
    TimerGPU timer(s);
    for (int it = 0; it <= iters; it++) {
        double t0 = now_ms();
        ntt.LDE(dst, 0, src, 0, nBits, nBitsExt, nCols, timer, s, true, (gl64_t *)tree, te);
        CK(cudaStreamSynchronize(s));
        double t1 = now_ms();
        if (hash == "p1")
            PoseidonGoldilocksGPU<16>::merkletree(4, tree, (uint64_t *)dst, nCols, NE, Layout::ColMajor, s);
        else if (hash == "b3")
            Blake3GoldilocksGPU::merkletree(2, tree, (uint64_t *)dst, nCols, NE, Layout::ColMajor, s);
        CK(cudaStreamSynchronize(s));
        double t2 = now_ms();
        uint64_t root0 = 0;
        if (hash != "none") CK(cudaMemcpy(&root0, tree + te - 4, 8, cudaMemcpyDeviceToHost));
        printf("K0 zisk hash=%s log_n=%lu m=%lu blowup=%lu iter=%d lde_ms=%.3f tree_ms=%.3f root0=%016lx\n",
               hash.c_str(), nBits, nCols, 1UL << bb, it, t1 - t0, t2 - t1, root0);
        fflush(stdout);
    }
    CK(cudaGetLastError());
    return 0;
}
