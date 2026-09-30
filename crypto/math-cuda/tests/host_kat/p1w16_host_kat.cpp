// Known-answer tests for `kernels/p1w16.cu` (the D-HASH stage-1 Poseidon1
// width-16 measurement kernels), run on the HOST through the shim: no GPU, no
// nvcc. Every check runs at every variant (the two multiplies and the
// Fourier-domain partial rounds).
//
// The vectors (`p1w16_kat_vectors.h`) come from the Python reference
// (`scripts/poseidon1/p1_params.py cuda`); the permutation vectors are the ones
// the Python reference, Plonky3's generator's permutation, Plonky3's Rust
// `Poseidon1` and the Rust host reference agree on.
//
// ⚠ Arithmetic and index math only — execution (threads, occupancy, the grid
// reduction) stays with `tests/p1w16_device.rs` on a GPU.
//
//   make test-p1w16-host-kat

#include <cstdio>
#include <cstring>

#include "cuda_host_shim.h"
#include "p1w16.cu"
#include "p1w16_kat_vectors.h"

static int g_fail = 0;

static void check(bool ok, const char *what, int v) {
    if (!ok) {
        std::printf("FAIL [v%d] %s\n", v, what);
        g_fail++;
    }
}

template <int V>
static void run() {
    // Permutation: the four vectors through the probe kernel.
    uint64_t out[4 * 16];
    uint64_t in[4 * 16];
    std::memcpy(in, P1_PERM_IN, sizeof(in));
    CUDA_HOST_FOR_EACH_THREAD(t, 4) permute_probe<V>(in, 4, out);
    check(std::memcmp(out, P1_PERM_OUT, sizeof(out)) == 0, "permutation vectors", V);

    // Mutation control: one flipped input bit must move the output.
    in[15] ^= 1;
    uint64_t out2[4 * 16];
    CUDA_HOST_FOR_EACH_THREAD(t, 4) permute_probe<V>(in, 4, out2);
    check(std::memcmp(out2, P1_PERM_OUT, 16 * 8) != 0, "a flipped input moves the output", V);

    // Leaves: a one-leaf coset of n felts is the leaf of those n felts.
    for (int k = 0; k < (int)(sizeof(P1_LEAF_N) / sizeof(P1_LEAF_N[0])); ++k) {
        const uint64_t n = P1_LEAF_N[k];
        uint64_t felts[64 + 1];
        for (uint64_t i = 0; i < n; ++i) felts[i] = i * 0x0123456789abcdefull + n;
        uint64_t d[4] = {~0ull, ~0ull, ~0ull, ~0ull};
        CUDA_HOST_FOR_EACH_THREAD(t, 1) leaves_base_coset<V>(felts, 1, n, d);
        char what[64];
        std::snprintf(what, sizeof(what), "leaf of %llu felts", (unsigned long long)n);
        check(std::memcmp(d, P1_LEAF_DIGEST[k], 32) == 0, what, V);
    }

    // Coset geometry: 4 leaves of block 16 over a strided codeword equal the
    // one-leaf form over each gathered coset.
    {
        const uint64_t num_leaves = 4, block = 16;
        uint64_t cw[64];
        for (int i = 0; i < 64; ++i) cw[i] = 0x9e3779b97f4a7c15ull * (i + 1);
        uint64_t got[16];
        CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) leaves_base_coset<V>(cw, num_leaves, block, got);
        bool ok = true;
        for (uint64_t j = 0; j < num_leaves; ++j) {
            uint64_t coset[16], want[4];
            for (uint64_t t = 0; t < block; ++t) coset[t] = cw[j + t * num_leaves];
            CUDA_HOST_FOR_EACH_THREAD(t, 1) leaves_base_coset<V>(coset, 1, block, want);
            ok &= std::memcmp(got + 4 * j, want, 32) == 0;
        }
        check(ok, "base coset stride", V);

        // Ext3: element `t` of leaf `j` is the triple at `(j + t·num_leaves)·3`;
        // its leaf is the base leaf over the 3·block components in order.
        uint64_t cw3[4 * 4 * 3];
        for (int i = 0; i < 48; ++i) cw3[i] = 0x1234567ull * (i + 7);
        uint64_t got3[16];
        CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) leaves_ext3_coset<V>(cw3, num_leaves, 4, got3);
        ok = true;
        for (uint64_t j = 0; j < num_leaves; ++j) {
            uint64_t flat[12], want[4];
            for (uint64_t t = 0; t < 4; ++t)
                for (int c = 0; c < 3; ++c) flat[3 * t + c] = cw3[(j + t * num_leaves) * 3 + c];
            CUDA_HOST_FOR_EACH_THREAD(t, 1) leaves_base_coset<V>(flat, 1, 12, want);
            ok &= std::memcmp(got3 + 4 * j, want, 32) == 0;
        }
        check(ok, "ext3 coset stride", V);
    }

    // The 4-ary node over the four digests P1_PERM_OUT[0][4c..4c+4].
    {
        uint64_t parent[4];
        CUDA_HOST_FOR_EACH_THREAD(t, 1) merkle_level4<V>(P1_PERM_OUT[0], parent, 1);
        check(std::memcmp(parent, P1_NODE4, 32) == 0, "4-ary node", V);
    }

    // Grind: with limit = head(k) + 1 for the k-th vector, the search over
    // [0, 8) returns the smallest nonce whose head is below it.
    for (int k = 0; k < 8; ++k) {
        const uint64_t limit = P1_GRIND_HEAD[k] + 1;
        uint64_t want = ~0ull;
        for (int n = 0; n < 8; ++n)
            if (P1_GRIND_HEAD[n] < limit) {
                want = n;
                break;
            }
        unsigned long long result = ~0ull;
        CUDA_HOST_SINGLE_THREAD();
        grind_search<V>(P1_GRIND_INNER, limit, 0, 8, &result);
        char what[64];
        std::snprintf(what, sizeof(what), "grind, limit head[%d]+1", k);
        check(result == want, what, V);
    }
}

int main() {
    run<0>();
    run<1>();
    run<2>();
    if (g_fail == 0) {
        std::printf("p1w16 host KAT: all checks pass (variants 0, 1 and 2)\n");
        return 0;
    }
    std::printf("p1w16 host KAT: %d FAILED\n", g_fail);
    return 1;
}
