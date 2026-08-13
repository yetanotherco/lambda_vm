// Known-answer tests for `kernels/blake3.cu`, run on the host.
//
// WHY THIS EXISTS. The GPU parity suite (`tests/blake3_compress_parity.rs` and
// friends) is the authority on these kernels, but it runs only where a GPU does,
// and per-PR CI has none — GPU CI is merge_group-only. Without this the kernels
// have no per-PR gate at all: an edit to `blake3.cu` that broke the hash would
// reach the merge queue before anything noticed. This compiles the real kernel
// source through `cuda_host_shim.h` and runs external known-answer vectors
// through it, in seconds, with no GPU and no cargo.
//
// WHAT IT COVERS: the compression function at both round counts, the field
// element serialization, the 64-byte block framing, and the Merkle parent.
//
// WHAT IT DOES NOT COVER, and what the GPU tests are still required for:
// whether nvcc accepts the file, and every property of execution rather than
// arithmetic — grid indexing, `__syncthreads` ordering up the Merkle levels,
// device memory alignment, and register pressure. Passing here is necessary,
// never sufficient.
//
// Build and run with `make test-blake3-host-kat`.

#include <cstdio>
#include <cstring>
#include <string>

#include "cuda_host_shim.h"

// The kernel under test. Included, not linked: the shim turns its device
// functions into host functions, and there is no other way to call them.
#include "blake3.cu"

#include "blake3_kat_vectors.h"

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

// The official vectors' input: the first `len` bytes of the repeating 251-byte
// sequence 0, 1, ..., 250.
void official_input(uint32_t len, uint8_t *out) {
    for (uint32_t i = 0; i < len; ++i) out[i] = (uint8_t)(i % 251);
}

// The 32-byte digest of a message of at most 64 bytes: ONE compression with
// `h = IV`, `t = 0`, the block zero-padded and read as little-endian words,
// `block_len` the true length, and the one-block flag set. The digest is the low
// eight output words, little-endian.
std::string hash_one_block(const uint8_t *msg, uint32_t len, int rounds) {
    uint8_t block[64] = {0};
    memcpy(block, msg, len);
    uint32_t m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = (uint32_t)block[4 * i] | ((uint32_t)block[4 * i + 1] << 8) |
               ((uint32_t)block[4 * i + 2] << 16) | ((uint32_t)block[4 * i + 3] << 24);
    }
    uint32_t out[16];
    if (rounds == 6) {
        blake3_compress<6>(BLAKE3_IV, m, 0, len, BLAKE3_FLAGS_ONE_BLOCK, out);
    } else {
        blake3_compress<7>(BLAKE3_IV, m, 0, len, BLAKE3_FLAGS_ONE_BLOCK, out);
    }
    char hex[65];
    for (int i = 0; i < 8; ++i) {
        for (int b = 0; b < 4; ++b) {
            snprintf(hex + (i * 4 + b) * 2, 3, "%02x", (unsigned)((out[i] >> (8 * b)) & 0xff));
        }
    }
    return std::string(hex, 64);
}

// ★ The external anchor. At 7 rounds the kernel must BE standard BLAKE3.
//
// Run over every length a single block can hold rather than one: the length keys
// both `block_len` and the zero-padding, so a port that ignored either would
// still pass at a single length.
void official_vectors_at_seven_rounds() {
    check(NUM_OFFICIAL_VECTORS == 11, "official vector table lost entries");
    for (int i = 0; i < NUM_OFFICIAL_VECTORS; ++i) {
        const OfficialVector &v = OFFICIAL_VECTORS[i];
        uint8_t msg[64];
        official_input(v.input_len, msg);
        std::string got = hash_one_block(msg, v.input_len, 7);
        if (got != v.hash_hex) {
            printf("FAIL official vector len=%u\n  got  %s\n  want %s\n", v.input_len, got.c_str(),
                   v.hash_hex);
            ++failures;
        }
    }
    printf("official BLAKE3 vectors at 7 rounds: %d checked\n", NUM_OFFICIAL_VECTORS);
}

// NEGATIVE CONTROL for the anchor above: at 6 rounds nothing must match.
//
// Without this the anchor would pass just as well if the round count were being
// ignored — the one bug that makes the whole external-anchor argument vacuous,
// since the 6-round arm's only defence is "the same code path with the loop
// bound changed". The zero-length case is skipped: an empty message is the one
// input where the rounds have nothing to diffuse and a collision would not be
// evidence of anything.
void six_rounds_is_not_standard_blake3() {
    int discriminated = 0;
    for (int i = 0; i < NUM_OFFICIAL_VECTORS; ++i) {
        const OfficialVector &v = OFFICIAL_VECTORS[i];
        if (v.input_len == 0) continue;
        uint8_t msg[64];
        official_input(v.input_len, msg);
        check(hash_one_block(msg, v.input_len, 6) != v.hash_hex,
              "6 rounds reproduced an official 7-round vector");
        ++discriminated;
    }
    printf("6-round negative control: %d lengths discriminated\n", discriminated);
}

// ★ The 6-round known-answer test, and the reason it is worth more than a
// self-comparison: `out6` came from #903's Python oracle, not from any code in
// this tree. All 16 output words are checked, not just the chaining value.
void canonical_vectors_at_both_round_counts() {
    check(NUM_CANONICAL_VECTORS == 10, "canonical vector table lost entries");
    for (int i = 0; i < NUM_CANONICAL_VECTORS; ++i) {
        const CanonicalVector &v = CANONICAL_VECTORS[i];
        uint32_t out6[16], out7[16];
        blake3_compress<6>(v.h, v.m, v.t, v.block_len, v.flags, out6);
        blake3_compress<7>(v.h, v.m, v.t, v.block_len, v.flags, out7);
        for (int w = 0; w < 16; ++w) {
            if (out6[w] != v.out6[w]) {
                printf("FAIL canonical %d word %d at 6 rounds: got %08x want %08x\n", i, w, out6[w],
                       v.out6[w]);
                ++failures;
            }
            if (out7[w] != v.out7[w]) {
                printf("FAIL canonical %d word %d at 7 rounds: got %08x want %08x\n", i, w, out7[w],
                       v.out7[w]);
                ++failures;
            }
        }
    }
    printf("canonical vectors at 6 AND 7 rounds: %d checked, all 16 words each\n",
           NUM_CANONICAL_VECTORS);
}

// The serialization: one field element becomes the two message words its
// canonical big-endian bytes are read as, little-endian. The non-canonical raws
// are the cases where the reduction is the only thing that matters.
void serialization_is_the_canonical_big_endian_bytes() {
    const uint64_t P = 0xFFFFFFFF00000001ull;
    const uint64_t raws[] = {0, 1, P - 1, P, P + 1, P + 12345, ~0ull, 0x0123456789ABCDEFull};
    for (uint64_t raw : raws) {
        uint32_t w0, w1;
        blake3_words_of_felt(raw, w0, w1);
        uint64_t canon = raw >= P ? raw - P : raw;
        uint8_t be[8];
        for (int i = 0; i < 8; ++i) be[i] = (uint8_t)(canon >> (56 - 8 * i));
        uint32_t e0 = (uint32_t)be[0] | ((uint32_t)be[1] << 8) | ((uint32_t)be[2] << 16) |
                      ((uint32_t)be[3] << 24);
        uint32_t e1 = (uint32_t)be[4] | ((uint32_t)be[5] << 8) | ((uint32_t)be[6] << 16) |
                      ((uint32_t)be[7] << 24);
        check(w0 == e0 && w1 == e1, "blake3_words_of_felt");
    }
    printf("serialization: %zu elements checked, non-canonical raws included\n",
           sizeof(raws) / sizeof(raws[0]));
}

// The block framing: nine elements are eighteen words, so one block completes and
// a two-word tail stays pending with fourteen words of zero padding behind it.
void block_framing_completes_and_pads() {
    Blake3Block b;
    b.init();
    int completed = 0;
    uint32_t blocks[2][16] = {{0}};
    for (int i = 0; i < 9; ++i) {
        uint32_t w0, w1;
        blake3_words_of_felt((uint64_t)(i + 1) * 0x1111111111111111ull, w0, w1);
        if (b.push_word(w0)) {
            memcpy(blocks[completed++], b.m, 64);
            b.reset();
        }
        if (b.push_word(w1)) {
            memcpy(blocks[completed++], b.m, 64);
            b.reset();
        }
    }
    check(completed == 1, "exactly one block should have completed");
    check(b.pending_bytes() == 8, "the pending tail should be 8 bytes");
    memcpy(blocks[1], b.m, 64);
    bool padded = true;
    for (int k = 2; k < 16; ++k) padded = padded && blocks[1][k] == 0;
    check(padded, "the tail block must be zero-padded");
    printf("block framing: 1 completed block + an 8-byte zero-padded tail\n");
}

// The Merkle parent: one compression over the 64 bytes of two child digests, so
// it must equal the one-block hash of their concatenation — which at 7 rounds is
// a plain `blake3::hash` call, and is what makes the parent framing externally
// anchored rather than merely self-consistent.
void parent_is_the_one_block_hash_of_its_children() {
    uint8_t children[64];
    official_input(64, children);
    uint8_t nodes[3 * 32];
    memcpy(nodes + 32, children, 32);      // node 1 = left child
    memcpy(nodes + 64, children + 32, 32); // node 2 = right child
    blake3_hash_merkle_parent(nodes, 0, 1, 0);
    char hex[65];
    for (int i = 0; i < 32; ++i) snprintf(hex + i * 2, 3, "%02x", (unsigned)nodes[i]);
    check(std::string(hex, 64) == hash_one_block(children, 64, BLAKE3_ROUNDS),
          "parent must equal the one-block hash of left || right");
    printf("Merkle parent at BLAKE3_ROUNDS=%d: %s\n", BLAKE3_ROUNDS, hex);
}

} // namespace

int main() {
    printf("BLAKE3 device-kernel known-answer tests, host-compiled from "
           "crypto/math-cuda/kernels/blake3.cu\n\n");
    official_vectors_at_seven_rounds();
    six_rounds_is_not_standard_blake3();
    canonical_vectors_at_both_round_counts();
    serialization_is_the_canonical_big_endian_bytes();
    block_framing_completes_and_pads();
    parent_is_the_one_block_hash_of_its_children();
    if (failures != 0) {
        printf("\n*** %d FAILURE(S) ***\n", failures);
        return 1;
    }
    printf("\nALL HOST KAT CHECKS PASS\n");
    printf("NOTE: arithmetic only. nvcc acceptance and GPU execution are covered "
           "by tests/blake3_*.rs, which need a GPU.\n");
    return 0;
}
