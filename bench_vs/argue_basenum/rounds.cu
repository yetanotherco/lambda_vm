// Depth-24 two-layer input rounds, with the production slot read-back.
// Build on a CUDA box (sm_120):
// nvcc -O3 -std=c++17 -arch sm_120 -I crypto/math-cuda/kernels \
//     bench_vs/argue_basenum/rounds.cu -o /tmp/argue-basenum-rounds
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <immintrin.h>

#include "sumcheck.cu"

constexpr uint64_t stride = uint64_t{1} << 22;
constexpr uint64_t stored2 = stride;
constexpr uint64_t stored3 = stride / 2;
constexpr uint64_t real = 3 * stride + stored3;
constexpr unsigned grid = 2048, block = 256, low = 6, rounds = 22 - low;
constexpr unsigned shared_bytes = (12 * block + 12 * (block / 32)) * sizeof(uint64_t);

__device__ uint64_t random_limb(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

__global__ void inputs(uint64_t *p, uint64_t *base_p, uint64_t *q) {
    for (uint64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < real;
         x += gridDim.x * blockDim.x) {
        base_p[x] = random_limb(4 * x);
        gkr2_store(p, x, ext3::make(base_p[x], 0, 0));
        gkr2_store(q, x, ext3::make(random_limb(4 * x + 1), random_limb(4 * x + 2),
                                  random_limb(4 * x + 3)));
    }
}

// At seven live slots the bound quarters are all full after round 1.
__global__ void check_folds(const uint64_t *p, const uint64_t *folded_p, unsigned *counts) {
    unsigned raw = 0, canonical = 0;
    for (uint64_t x = blockIdx.x * blockDim.x + threadIdx.x; x < 6 * stride;
         x += gridDim.x * blockDim.x) {
        const uint64_t old = (x / (3 * stride / 2)) * (3 * stride) + x % (3 * stride / 2);
        raw += p[old] != folded_p[x];
        canonical += goldilocks::canonical(p[old]) != goldilocks::canonical(folded_p[x]);
    }
    if (raw) atomicAdd(counts, raw);
    if (canonical) atomicAdd(counts + 1, canonical);
}

void check(cudaError_t rc) {
    if (rc == cudaSuccess) return;
    std::fprintf(stderr, "%s\n", cudaGetErrorString(rc));
    std::exit(1);
}

template <typename T = uint64_t>
T *allocate(size_t count) {
    T *dev = nullptr;
    check(cudaMalloc(&dev, count * sizeof(T)));
    return dev;
}

using Clock = std::chrono::steady_clock;
using Sums = std::array<uint64_t, 12>;
using Pair = std::array<double, 2>;

int main() {
    check(cudaSetDeviceFlags(cudaDeviceMapHost));
    cudaStream_t stream;
    check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    uint64_t *p = allocate(real * 3), *p_input = allocate(real * 3), *base_p = allocate(real);
    uint64_t *q = allocate(real * 3), *q_input = allocate(real * 3);
    uint64_t *folded_p = allocate(6 * stride);
    uint64_t *partials = allocate(4 * grid * 3), *eq = allocate(3 * ((1 << low) + (1 << rounds) - 1));
    unsigned *ticket = allocate<unsigned>(1);
    uint64_t *slot = nullptr, *slot_dev = nullptr;
    check(cudaHostAlloc(&slot, 97 * sizeof(uint64_t), cudaHostAllocMapped));
    check(cudaHostGetDevicePointer(&slot_dev, slot, 0));
    slot[96] = 0;
    check(cudaMemsetAsync(ticket, 0, sizeof(*ticket), stream));

    uint64_t point[3 * 22];
    uint64_t state = 0xd24a713;
    for (auto &limb : point) {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        limb = state;
    }
    uint64_t *point_dev = allocate(3 * 22);
    check(cudaMemcpyAsync(point_dev, point, sizeof(point), cudaMemcpyHostToDevice, stream));
    gkr_eq_levels_ext3<<<256, block, 0, stream>>>(point_dev, low, rounds, eq);
    inputs<<<grid, block, 0, stream>>>(p_input, base_p, q_input);
    check(cudaStreamSynchronize(stream));
    const Fe3 s{point[0], point[1], point[2]}, h{point[3], point[4], point[5]};

    uint64_t seq = 0;
    auto measure = [&](bool base, std::array<Sums, 2> &sums) {
        check(cudaMemcpyAsync(q, q_input, real * 24, cudaMemcpyDeviceToDevice, stream));
        if (!base) check(cudaMemcpyAsync(p, p_input, real * 24, cudaMemcpyDeviceToDevice, stream));
        check(cudaStreamSynchronize(stream));
        Pair duration{};
        for (unsigned j = 0; j < 2; ++j) {
            ++seq;
            const uint64_t quarter = stride >> (j + 1);
            auto e_hi = eq + 3 * ((1 << low) + (1 << rounds) - (1 << (rounds - j)));
            auto start = Clock::now();
            if (!base) {
                gkr2_round_real_slot<<<grid, block, shared_bytes, stream>>>(
                    p, q, stride, quarter, j, s.a, s.b, s.c, h.a, h.b, h.c, e_hi, eq, low,
                    stored2, stored3, partials, ticket, slot_dev, seq);
            } else if (j == 0) {
                gkr2_round_base_slot<<<grid, block, shared_bytes, stream>>>(
                    base_p, q, stride, quarter, j, s.a, s.b, s.c, h.a, h.b, h.c, e_hi,
                    eq, low, stored2, stored3, partials, nullptr, ticket, slot_dev, seq);
            } else {
                gkr2_round_bind_base_slot<<<grid, block, shared_bytes, stream>>>(
                    base_p, q, stride, quarter, j, s.a, s.b, s.c, h.a, h.b, h.c, e_hi,
                    eq, low, stored2, stored3, partials, folded_p, ticket, slot_dev, seq);
            }
            check(cudaGetLastError());
            auto words = static_cast<volatile uint64_t *>(slot);
            while (words[96] != seq) {
                _mm_pause();
                if (Clock::now() - start > std::chrono::seconds(1)) {
                    check(cudaStreamSynchronize(stream));
                    if (words[96] != seq) std::abort();
                }
            }
            std::atomic_thread_fence(std::memory_order_acquire);
            for (unsigned i = 0; i < sums[j].size(); ++i) sums[j][i] = words[i];
            duration[j] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        return duration;
    };

    unsigned *counts = allocate<unsigned>(2);
    Pair today_total{}, base_total{};
    constexpr unsigned warmups = 3, reps = 10;
    for (unsigned i = 0; i < warmups + reps; ++i) {
        std::array<Sums, 2> today_sums{}, base_sums{};
        Pair today, base;
        if (i % 2 == 0) {
            today = measure(false, today_sums);
            base = measure(true, base_sums);
        } else {
            base = measure(true, base_sums);
            today = measure(false, today_sums);
        }
        for (unsigned j = 0; j < 2; ++j) {
            for (unsigned k = 0; k < today_sums[j].size(); ++k) {
                auto a = today_sums[j][k], b = base_sums[j][k];
                if (goldilocks::PRIME <= a) a -= goldilocks::PRIME;
                if (goldilocks::PRIME <= b) b -= goldilocks::PRIME;
                if (a != b) {
                    std::fprintf(stderr, "round %u limb %u differs canonically\n", j, k);
                    return 1;
                }
            }
        }
        if (i == 0) {
            check(cudaMemsetAsync(counts, 0, 2 * sizeof(unsigned), stream));
            check_folds<<<grid, block, 0, stream>>>(p, folded_p, counts);
            unsigned folded[2];
            check(cudaMemcpyAsync(folded, counts, sizeof(folded), cudaMemcpyDeviceToHost, stream));
            check(cudaStreamSynchronize(stream));
            std::printf("folded p mismatches (raw/canonical): %u/%u\n", folded[0], folded[1]);
            if (folded[1] != 0) return 1;
        }
        if (i < warmups) continue;
        for (unsigned j = 0; j < 2; ++j) {
            today_total[j] += today[j];
            base_total[j] += base[j];
        }
        std::printf("rep %2u: today %.6f %.6f ms, base %.6f %.6f ms, raw sums %s\n",
                    i - warmups + 1, today[0], today[1], base[0], base[1],
                    today_sums == base_sums ? "equal" : "differ");
    }
    for (unsigned j = 0; j < 2; ++j) {
        std::printf("round %u: %.6f %.6f ms, %.6fx\n", j, today_total[j] / reps,
                    base_total[j] / reps, base_total[j] / today_total[j]);
    }
    const double a = today_total[0] + today_total[1], b = base_total[0] + base_total[1];
    std::printf("pair: %.6f %.6f ms, %.6fx, %s\n", a / reps, b / reps, b / a,
                b > 0.90 * a ? "STOP" : (b <= 0.85 * a ? "GO" : "BETWEEN"));
}
