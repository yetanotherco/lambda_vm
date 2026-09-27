// Host checks for the RPX kernels whose lanes COOPERATE — the half-warp
// permutation and Merkle kernels, and the work-queue grind — run through
// `cuda_host_simt_shim.h`, where every lane is a real thread.
//
// Every check here is against the SHIPPED kernel or the shipped device
// function, byte for byte (raw words for the permutation): these kernels exist
// to produce exactly what `permute`, `rpx_merkle_level` / `rpx_merkle_tail` and
// `rpx_grind_search` produce, faster. The shipped side is itself pinned to the
// Rust oracle by `rpx_host_kat.cpp`.
//
// Build and run with `make test-rpx-host-kat` (the second binary). What the
// GPU tests still own: nvcc acceptance, the real warp scheduler, timing.

#include <cstdio>
#include <cstring>
#include <vector>

#include "cuda_host_simt_shim.h"

#define RPX_HOST_SIMT
#include "rpx.cu"

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

const uint64_t P = 0xFFFFFFFF00000001ull;

uint64_t splitmix(uint64_t &seed) {
    seed += 0x9E3779B97F4A7C15ull;
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// Every fifth word a raw representation in [p, 2^64), as an LDE buffer holds.
uint64_t sample(uint64_t &seed, uint64_t i) {
    const uint64_t x = splitmix(seed);
    return (i % 5 == 0) ? (x % 0xFFFFFFFFull) + P : x % P;
}

// ---------------------------------------------------------------------------
// K3 — the half-warp permutation against `permute`, raw.
// ---------------------------------------------------------------------------
void warp_permutation_matches_the_shipped_permutation() {
    // An odd count, so the last warp has a dead half.
    const uint64_t n = 21;
    std::vector<uint64_t> in(n * 12), out(n * 12, ~0ull);
    uint64_t seed = 0x5117;
    for (uint64_t t = 0; t < n * 12; ++t) in[t] = sample(seed, t);
    for (int i = 0; i < 12; ++i) in[i] = 0;          // the all-zero state
    for (int i = 0; i < 12; ++i) in[12 + i] = P - 1;  // the all-(p-1) state
    for (int i = 0; i < 12; ++i) in[24 + i] = ~0ull;  // raw maxima
    // 64 threads = 2 warps = 4 states per block.
    simt_launch((unsigned)((n + 3) / 4), 64,
                [&] { rpx_permute_warp_probe(in.data(), n, out.data()); });
    int matched = 0;
    for (uint64_t t = 0; t < n; ++t) {
        uint64_t s[12];
        memcpy(s, &in[t * 12], sizeof(s));
        rpx::permute(s);
        if (memcmp(s, &out[t * 12], sizeof(s)) == 0) {
            ++matched;
        } else {
            printf("FAIL warp permutation state %llu\n", (unsigned long long)t);
            ++failures;
        }
    }
    printf("K3 half-warp permutation vs rpx::permute, raw: %d/%llu states (odd count, edges, raw words)\n",
           matched, (unsigned long long)n);
}

// Random node bytes: both kernels decode big-endian words without reducing
// them, and the permutation reads raw words as field elements, so any bytes are
// a fair input for a kernel-versus-kernel comparison.
std::vector<uint8_t> tree_with_random_leaves(uint64_t num_leaves, uint64_t seed) {
    std::vector<uint8_t> nodes((2 * num_leaves - 1) * 32, 0);
    for (uint64_t i = (num_leaves - 1) * 32; i < nodes.size(); ++i) {
        nodes[i] = (uint8_t)splitmix(seed);
    }
    return nodes;
}

// The shipped walk: one `rpx_merkle_level` launch per level, threads replayed.
void shipped_tree(std::vector<uint8_t> &nodes, uint64_t num_leaves) {
    uint64_t level_begin = num_leaves - 1;
    while (level_begin != 0) {
        const uint64_t new_begin = level_begin / 2;
        const uint64_t n_pairs = level_begin - new_begin;
        const unsigned grid = (unsigned)((n_pairs + 63) / 64);
        simt_launch(grid, 64, [&] { rpx_merkle_level(nodes.data(), new_begin, n_pairs); });
        level_begin = new_begin;
    }
}

void warp_merkle_kernels_match_the_shipped_tree() {
    int trees = 0;
    for (uint64_t log_n = 1; log_n <= 7; ++log_n) {
        const uint64_t num_leaves = 1ull << log_n;
        std::vector<uint8_t> want = tree_with_random_leaves(num_leaves, 0x7EE + log_n);
        std::vector<uint8_t> by_level = want, by_tail = want, mixed = want;
        shipped_tree(want, num_leaves);

        // Every level through `rpx_merkle_level_warp` (2 warps per block).
        uint64_t level_begin = num_leaves - 1;
        while (level_begin != 0) {
            const uint64_t new_begin = level_begin / 2;
            const uint64_t n_pairs = level_begin - new_begin;
            simt_launch((unsigned)((n_pairs + 3) / 4), 64,
                        [&] { rpx_merkle_level_warp(by_level.data(), new_begin, n_pairs); });
            level_begin = new_begin;
        }
        check(by_level == want, "rpx_merkle_level_warp must reproduce the shipped tree");

        // The whole tree in one `rpx_merkle_tail_warp` block of 2 warps: 4
        // pairs per pass, so the wide levels take several passes.
        simt_launch(1, 64, [&] { rpx_merkle_tail_warp(by_tail.data(), num_leaves - 1); });
        check(by_tail == want, "rpx_merkle_tail_warp must reproduce the shipped tree");

        // The walker's split: shipped levels while wide, warp levels, then the
        // warp tail — here cut at 8 and 2 pairs.
        level_begin = num_leaves - 1;
        while (level_begin != 0) {
            const uint64_t new_begin = level_begin / 2;
            const uint64_t n_pairs = level_begin - new_begin;
            if (n_pairs <= 2) {
                simt_launch(1, 32, [&] { rpx_merkle_tail_warp(mixed.data(), level_begin); });
                break;
            }
            if (n_pairs <= 8) {
                simt_launch((unsigned)((n_pairs + 3) / 4), 64,
                            [&] { rpx_merkle_level_warp(mixed.data(), new_begin, n_pairs); });
            } else {
                simt_launch((unsigned)((n_pairs + 63) / 64), 64,
                            [&] { rpx_merkle_level(mixed.data(), new_begin, n_pairs); });
            }
            level_begin = new_begin;
        }
        check(mixed == want, "shipped + warp levels + warp tail must reproduce the shipped tree");
        ++trees;
    }
    // A control that can fail: one flipped leaf byte moves the root.
    std::vector<uint8_t> a = tree_with_random_leaves(8, 0xF00), b = a;
    b[(8 - 1) * 32 + 5] ^= 1;
    simt_launch(1, 64, [&] { rpx_merkle_tail_warp(a.data(), 7); });
    simt_launch(1, 64, [&] { rpx_merkle_tail_warp(b.data(), 7); });
    check(memcmp(a.data(), b.data(), 32) != 0, "a flipped leaf byte must move the warp tail's root");
    printf("K3 warp level + warp tail + mixed walk vs the shipped level walk: %d trees (2..128 leaves) + a failing control\n",
           trees);
}

// ---------------------------------------------------------------------------
// K4 — the work-queue grind against the shipped grind.
// ---------------------------------------------------------------------------

// The shipped kernel as ONE thread: stride 1, so it scans `[base, base+count)`
// in order and returns the smallest valid nonce there, or the sentinel.
uint64_t shipped_grind(const uint64_t inner[4], uint64_t limit, uint64_t base, uint64_t count) {
    unsigned long long result = ~0ull;
    simt_launch(1, 1, [&] {
        rpx_grind_search(inner, limit, base, count, (volatile unsigned long long *)&result);
    });
    return result;
}

struct QueueRun {
    uint64_t nonce, executed, max_iters, ran_to_end;
};

QueueRun queue_grind(const uint64_t inner[4], uint64_t limit, uint64_t base, uint64_t count,
                     unsigned grid, unsigned block) {
    unsigned long long state[2] = {~0ull, 0};
    unsigned long long counts[3] = {0, 0, 0};
    simt_launch(grid, block, [&] {
        rpx_grind_search_queue_counted(inner, limit, base, count, state, counts);
    });
    unsigned long long plain[2] = {~0ull, 0};
    simt_launch(grid, block, [&] { rpx_grind_search_queue(inner, limit, base, count, plain); });
    check(plain[0] == state[0], "the plain and counted queue kernels must agree");
    return QueueRun{state[0], counts[0], counts[1], counts[2]};
}

void queue_grind_returns_the_shipped_nonce() {
    int runs = 0, found = 0;
    double past_answer = 0;
    uint64_t seed = 0x9A1;
    for (int k = 0; k < 24; ++k) {
        uint64_t inner[4];
        for (int i = 0; i < 4; ++i) inner[i] = splitmix(seed) % P;
        for (unsigned factor : {5u, 7u, 9u}) {
            const uint64_t limit = 1ull << (64 - factor);
            for (uint64_t base : {0ull, 37ull}) {
                // A range that ends before most answers, and one that holds them.
                for (uint64_t count : {100ull, 4096ull}) {
                    const uint64_t want = shipped_grind(inner, limit, base, count);
                    const QueueRun got = queue_grind(inner, limit, base, count, 2, 64);
                    char what[160];
                    snprintf(what, sizeof what,
                             "queue grind must return the shipped nonce (k %d, factor %u, base %llu, count %llu)",
                             k, factor, (unsigned long long)base, (unsigned long long)count);
                    check(got.nonce == want, what);
                    if (want != ~0ull) {
                        ++found;
                        // What is GUARANTEED: every chunk up to the one holding
                        // the answer is hashed in full, and nothing is hashed
                        // twice. How far past the answer the other warps got
                        // depends on scheduling (here, the host's), so it is
                        // reported, not asserted.
                        const uint64_t through = ((want - base) / 32 + 1) * 32;
                        check(got.executed >= through && got.executed <= count,
                              "queue grind must hash every chunk through the answer's, once");
                        past_answer += (double)(got.executed - through);
                    } else {
                        check(got.executed == count, "a miss must hash the whole range exactly once");
                    }
                    ++runs;
                }
            }
        }
    }
    // A control that can fail: a different inner hash finds a different nonce.
    uint64_t a[4] = {1, 2, 3, 4}, b[4] = {1, 2, 3, 5};
    const uint64_t limit = 1ull << (64 - 7);
    check(queue_grind(a, limit, 0, 4096, 2, 64).nonce != queue_grind(b, limit, 0, 4096, 2, 64).nonce,
          "different inner hashes must (here) give different nonces");
    printf("K4 queue grind vs the shipped grind: %d runs (%d with a nonce), misses hash the range once, a failing control\n",
           runs, found);
    printf("   (informational: %.1f nonces hashed past the answer's chunk on average, 4 host warps)\n",
           found ? past_answer / found : 0.0);
}

}  // namespace

int main() {
    printf("RPX cooperative-lane kernels (half-warp Merkle, queue grind), host SIMT replay of crypto/math-cuda/kernels/rpx.cu\n\n");
    warp_permutation_matches_the_shipped_permutation();
    warp_merkle_kernels_match_the_shipped_tree();
    queue_grind_returns_the_shipped_nonce();
    if (failures != 0) {
        printf("\n*** %d FAILURE(S) ***\n", failures);
        return 1;
    }
    printf("\nALL SIMT HOST KAT CHECKS PASS\n");
    return 0;
}
