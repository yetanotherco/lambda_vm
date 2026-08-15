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
// element serialization, the 64-byte block framing, the Merkle parent, the
// `Blake3Chain` construction over multi-block messages, and every leaf kernel's
// byte stream (replayed thread by thread through the shim).
//
// WHAT IT DOES NOT COVER, and what the GPU tests are still required for:
// whether nvcc accepts the file, and every property of execution rather than
// arithmetic — grid indexing, `__syncthreads` ordering up the Merkle levels,
// device memory alignment, and register pressure. Passing here is necessary,
// never sufficient.
//
// HOW THE ANCHORING LAYERS. Nothing here is checked against itself:
//   1. The compression function is anchored by the OFFICIAL BLAKE3 vectors at 7
//      rounds (Table 1) and by the oracle-derived canonical vectors at 6
//      (Table 2).
//   2. `HostChain` below — a byte-level transcription of the construction — is
//      anchored by the OFFICIAL multi-block vectors at 7 rounds (Table 3) and
//      the committed 6-round chain KAT (Table 4). It is built ON the device
//      compression, so layer 1 carries into it.
//   3. The device `Blake3Chain` is checked against `HostChain`, at word
//      granularity, which is all the kernels ever need.
//   4. Each leaf kernel is replayed on host and checked against `HostChain` over
//      the byte stream `leaves_bit_reversed_grouped` specifies — so the read
//      pattern and the hash are anchored separately rather than together.
//
// Build and run with `make test-blake3-host-kat`.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

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

// ===========================================================================
// The chain construction.
// ===========================================================================

// The device compression with the round count as a run-time argument, so the
// reference below can be evaluated at either arm from one code path.
void compress_dyn(const uint32_t *h, const uint32_t *m, uint64_t t, uint32_t block_len,
                  uint32_t flags, int rounds, uint32_t *out) {
    if (rounds == 6) {
        blake3_compress<6>(h, m, t, block_len, flags, out);
    } else {
        blake3_compress<7>(h, m, t, block_len, flags, out);
    }
}

// `Blake3Chain` at BYTE granularity — PA-PLAN §1.7.1 written out directly.
//
// Why this exists when `blake3.cu` already has a `Blake3Chain`: the device one
// is word-granular, because every message the kernels hash is a whole number of
// 8-byte field elements. The official vectors are not — 65, 127, 1023 — and
// those lengths are where a final-block `block_len` bug lives. So the vectors
// anchor THIS, and the device chain is then checked against it at the word
// lengths it can actually reach.
//
// It is a transcription of the same spec as the host Rust `Blake3Chain`, not of
// the device struct, and it holds a full block rather than compressing it for
// the same reason: whether a block is the last is unknown until the message ends.
struct HostChain {
    uint32_t cv[8];
    uint8_t block[64];
    uint32_t block_len;
    bool started;
    int rounds;

    void init(int r) {
        memcpy(cv, BLAKE3_IV, sizeof(cv));
        memset(block, 0, sizeof(block));
        block_len = 0;
        started = false;
        rounds = r;
    }

    void block_words(uint32_t *m) const {
        for (int i = 0; i < 16; ++i) {
            m[i] = (uint32_t)block[4 * i] | ((uint32_t)block[4 * i + 1] << 8) |
                   ((uint32_t)block[4 * i + 2] << 16) | ((uint32_t)block[4 * i + 3] << 24);
        }
    }

    uint32_t flags(bool is_final) const {
        return (started ? 0u : BLAKE3_FLAG_CHUNK_START) |
               (is_final ? (BLAKE3_FLAG_CHUNK_END | BLAKE3_FLAG_ROOT) : 0u);
    }

    void compress_pending() {
        uint32_t m[16], out[16];
        block_words(m);
        compress_dyn(cv, m, 0, 64, flags(false), rounds, out);
        memcpy(cv, out, sizeof(cv));
        memset(block, 0, sizeof(block));
        block_len = 0;
        started = true;
    }

    void update(const uint8_t *in, size_t n) {
        while (n != 0) {
            // Only now is the pending block known not to be the last.
            if (block_len == 64) compress_pending();
            size_t take = 64 - block_len;
            if (take > n) take = n;
            memcpy(block + block_len, in, take);
            block_len += (uint32_t)take;
            in += take;
            n -= take;
        }
    }

    void finalize(uint8_t *out32) const {
        uint32_t m[16], out[16];
        block_words(m);
        compress_dyn(cv, m, 0, block_len, flags(true), rounds, out);
        for (int i = 0; i < 8; ++i) {
            out32[4 * i] = (uint8_t)(out[i] & 0xff);
            out32[4 * i + 1] = (uint8_t)((out[i] >> 8) & 0xff);
            out32[4 * i + 2] = (uint8_t)((out[i] >> 16) & 0xff);
            out32[4 * i + 3] = (uint8_t)((out[i] >> 24) & 0xff);
        }
    }
};

void host_chain(const uint8_t *msg, size_t len, int rounds, uint8_t *out32) {
    HostChain c;
    c.init(rounds);
    c.update(msg, len);
    c.finalize(out32);
}

std::string to_hex(const uint8_t *b, size_t n) {
    std::string s(n * 2, '\0');
    for (size_t i = 0; i < n; ++i) snprintf(&s[i * 2], 3, "%02x", (unsigned)b[i]);
    return s;
}

// The KAT message for Table 4: byte `i` is `37i + 11 (mod 256)`.
void kat_message(size_t len, std::vector<uint8_t> &out) {
    out.resize(len);
    for (size_t i = 0; i < len; ++i) out[i] = (uint8_t)((i * 37 + 11) & 0xff);
}

// ★ The chain's external anchor: over multi-block messages of at most one
// chunk, at 7 rounds, the construction IS standard BLAKE3, so the official
// vectors are direct known-answer tests for the framing — the flag schedule
// across blocks, the chaining value, and the final block's `block_len`.
//
// The `agrees == false` rows are the P3 negative control and are not decoration:
// without them, every matching row would pass identically if the full chunk tree
// had been implemented instead of the single unbounded chunk.
void chain_against_official_multiblock_vectors() {
    check(NUM_CHAIN_VECTORS == 8, "chain vector table lost entries");
    int matched = 0, diverged = 0;
    for (int i = 0; i < NUM_CHAIN_VECTORS; ++i) {
        const ChainVector &v = CHAIN_VECTORS[i];
        std::vector<uint8_t> msg(v.input_len);
        for (uint32_t k = 0; k < v.input_len; ++k) msg[k] = (uint8_t)(k % 251);
        uint8_t digest[32];
        host_chain(msg.data(), msg.size(), 7, digest);
        std::string got = to_hex(digest, 32);
        if (v.agrees) {
            if (got != v.hash_hex) {
                printf("FAIL chain vector len=%u\n  got  %s\n  want %s\n", v.input_len, got.c_str(),
                       v.hash_hex);
                ++failures;
            } else {
                ++matched;
            }
        } else {
            check(got != v.hash_hex,
                  "past one chunk the chain must LEAVE standard BLAKE3 (P3 control)");
            ++diverged;
        }
    }
    printf("chain vs official multi-block vectors at 7 rounds: %d must-match, %d P3 controls\n",
           matched, diverged);
}

// ★ The 6-round chain anchor: the committed table, whose digests came from the
// Python oracle rather than from any code in this tree.
void chain_against_the_committed_six_round_table() {
    check(NUM_CHAIN_KAT_6ROUND == 12, "6-round chain KAT table lost entries");
    for (int i = 0; i < NUM_CHAIN_KAT_6ROUND; ++i) {
        const ChainKat6Round &v = CHAIN_KAT_6ROUND[i];
        std::vector<uint8_t> msg;
        kat_message(v.input_len, msg);
        uint8_t digest[32];
        host_chain(msg.data(), msg.size(), 6, digest);
        std::string got = to_hex(digest, 32);
        if (got != v.hash_hex) {
            printf("FAIL 6-round chain KAT len=%u\n  got  %s\n  want %s\n", v.input_len, got.c_str(),
                   v.hash_hex);
            ++failures;
        }
    }
    printf("chain vs committed 6-round KAT: %d lengths checked\n", NUM_CHAIN_KAT_6ROUND);
}

// The DEVICE chain against the anchored reference, at every word-multiple length
// through several block boundaries. This is what carries the anchors above onto
// the struct the kernels actually use.
//
// The step of 4 is the device chain's granularity, and the range crosses the
// first, second and eighth block boundaries — the places a mis-set CHUNK_START,
// an eagerly compressed final block, or a wrong `block_len` would show.
void device_chain_matches_the_reference() {
    int checked = 0;
    for (size_t len = 0; len <= 600; len += 4) {
        std::vector<uint8_t> msg;
        kat_message(len, msg);

        Blake3Chain dev;
        dev.init();
        for (size_t i = 0; i < len; i += 4) {
            uint32_t w = (uint32_t)msg[i] | ((uint32_t)msg[i + 1] << 8) |
                         ((uint32_t)msg[i + 2] << 16) | ((uint32_t)msg[i + 3] << 24);
            dev.push_word(w);
        }
        uint8_t got[32];
        dev.finalize(got);

        uint8_t want[32];
        host_chain(msg.data(), msg.size(), BLAKE3_ROUNDS, want);
        if (memcmp(got, want, 32) != 0) {
            printf("FAIL device chain at len=%zu\n  got  %s\n  want %s\n", len,
                   to_hex(got, 32).c_str(), to_hex(want, 32).c_str());
            ++failures;
            break;
        }
        ++checked;
    }
    printf("device chain vs reference at BLAKE3_ROUNDS=%d: %d lengths (0..600 step 4)\n",
           BLAKE3_ROUNDS, checked);
}

// ★ P2 on the device struct: a 64-byte message through the chain must be the
// parent compression. This is the invariant that lets the leaf and parent layers
// be one hash, and it is why `blake3_hash_merkle_parent` needs no chaining.
void device_chain_at_64_bytes_is_the_parent() {
    uint8_t children[64];
    official_input(64, children);
    uint8_t nodes[3 * 32];
    memcpy(nodes + 32, children, 32);
    memcpy(nodes + 64, children + 32, 32);
    blake3_hash_merkle_parent(nodes, 0, 1, 0);

    Blake3Chain dev;
    dev.init();
    for (int i = 0; i < 16; ++i) {
        uint32_t w = (uint32_t)children[4 * i] | ((uint32_t)children[4 * i + 1] << 8) |
                     ((uint32_t)children[4 * i + 2] << 16) | ((uint32_t)children[4 * i + 3] << 24);
        dev.push_word(w);
    }
    uint8_t got[32];
    dev.finalize(got);
    check(memcmp(got, nodes, 32) == 0, "a 64-byte device chain must be the parent compression");
    printf("P2: a 64-byte chain is the Merkle parent compression\n");
}

// ===========================================================================
// The leaf kernels, replayed thread by thread.
// ===========================================================================

const uint64_t GOLDILOCKS_P = 0xFFFFFFFF00000001ull;

uint64_t canon(uint64_t raw) { return raw >= GOLDILOCKS_P ? raw - GOLDILOCKS_P : raw; }

// `reverse_index(i, n)` — the CPU commit's row permutation, and what the kernels
// compute as `__brevll(tid) >> (64 - log_num_rows)`.
uint64_t reverse_index(uint64_t i, uint32_t log_n) { return __brevll(i) >> (64 - log_n); }

// Append a field element's canonical BIG-endian bytes — the serialization
// `leaves_bit_reversed_grouped` writes and every leaf kernel must reproduce.
void push_be(std::vector<uint8_t> &buf, uint64_t raw) {
    uint64_t c = canon(raw);
    for (int i = 0; i < 8; ++i) buf.push_back((uint8_t)(c >> (56 - 8 * i)));
}

// A deterministic value stream, including deliberately non-canonical raws so the
// reduction is exercised rather than assumed.
uint64_t sample(uint64_t seed, uint64_t i) {
    uint64_t x = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull;
    x ^= x >> 31;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 29;
    // Every fifth value is left above the modulus.
    return (i % 5 == 0) ? x : x % GOLDILOCKS_P;
}

void check_leaves(const std::vector<uint8_t> &got, const std::vector<std::vector<uint8_t>> &want,
                  const char *what) {
    if (got.size() != want.size() * 32) {
        printf("FAIL %s: leaf count %zu vs %zu\n", what, got.size() / 32, want.size());
        ++failures;
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        uint8_t expect[32];
        host_chain(want[i].data(), want[i].size(), BLAKE3_ROUNDS, expect);
        if (memcmp(got.data() + i * 32, expect, 32) != 0) {
            printf("FAIL %s: leaf %zu\n  got  %s\n  want %s\n", what, i,
                   to_hex(got.data() + i * 32, 32).c_str(), to_hex(expect, 32).c_str());
            ++failures;
            return;
        }
    }
}

// The two column-major base kernels: one leaf per bit-reversed row, and one per
// bit-reversed row pair. The expected byte stream is built from the CPU leaf
// spec — rows in bit-reversed order, each written column by column in canonical
// big-endian — so what is compared is the kernel's READ PATTERN against that
// spec, with the hash anchored separately above.
void base_leaf_kernels_read_the_specified_bytes() {
    for (uint32_t log_n : {2u, 4u, 6u}) {
        for (uint64_t num_cols : {1ull, 5ull, 8ull, 17ull}) {
            uint64_t n = 1ull << log_n;
            std::vector<uint64_t> cols(num_cols * n);
            for (uint64_t c = 0; c < num_cols; ++c) {
                for (uint64_t r = 0; r < n; ++r) cols[c * n + r] = sample(log_n * 31 + num_cols, c * n + r);
            }

            // rows_per_leaf = 1
            {
                std::vector<uint8_t> out(n * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, n) {
                    blake3_leaves_base_batched(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint8_t>> want(n);
                for (uint64_t leaf = 0; leaf < n; ++leaf) {
                    uint64_t br = reverse_index(leaf, log_n);
                    for (uint64_t c = 0; c < num_cols; ++c) push_be(want[leaf], cols[c * n + br]);
                }
                check_leaves(out, want, "blake3_leaves_base_batched");
            }

            // rows_per_leaf = 2
            {
                uint64_t num_leaves = n / 2;
                std::vector<uint8_t> out(num_leaves * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                    blake3_leaves_base_row_pair_batched(cols.data(), n, num_cols, n, log_n,
                                                        out.data());
                }
                std::vector<std::vector<uint8_t>> want(num_leaves);
                for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                    for (int k = 0; k < 2; ++k) {
                        uint64_t br = reverse_index(2 * leaf + k, log_n);
                        for (uint64_t c = 0; c < num_cols; ++c) push_be(want[leaf], cols[c * n + br]);
                    }
                }
                check_leaves(out, want, "blake3_leaves_base_row_pair_batched");
            }
        }
    }
    printf("base leaf kernels: read pattern matches the CPU leaf spec\n");
}

// The ext3 kernels, over the de-interleaved three-slab layout. An ext3 element
// is three consecutive components, each 8 big-endian bytes — six words, so
// elements straddle block boundaries routinely, which is the case the
// word-granular block builder exists for.
void ext3_leaf_kernels_read_the_specified_bytes() {
    for (uint32_t log_n : {2u, 4u, 6u}) {
        for (uint64_t num_cols : {1ull, 3ull, 11ull}) {
            uint64_t n = 1ull << log_n;
            std::vector<uint64_t> cols(num_cols * 3 * n);
            for (uint64_t s = 0; s < num_cols * 3; ++s) {
                for (uint64_t r = 0; r < n; ++r) cols[s * n + r] = sample(log_n * 17 + num_cols, s * n + r);
            }

            // One leaf per bit-reversed row.
            {
                std::vector<uint8_t> out(n * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, n) {
                    blake3_leaves_ext3_batched(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint8_t>> want(n);
                for (uint64_t leaf = 0; leaf < n; ++leaf) {
                    uint64_t br = reverse_index(leaf, log_n);
                    for (uint64_t c = 0; c < num_cols; ++c) {
                        for (uint64_t k = 0; k < 3; ++k) push_be(want[leaf], cols[(c * 3 + k) * n + br]);
                    }
                }
                check_leaves(out, want, "blake3_leaves_ext3_batched");
            }

            // Row pairs — the comp-poly kernel, which the aux trace also uses.
            {
                uint64_t num_leaves = n / 2;
                std::vector<uint8_t> out(num_leaves * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                    blake3_comp_poly_leaves_ext3(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint8_t>> want(num_leaves);
                for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                    for (int j = 0; j < 2; ++j) {
                        uint64_t br = reverse_index(2 * leaf + j, log_n);
                        for (uint64_t c = 0; c < num_cols; ++c) {
                            for (uint64_t k = 0; k < 3; ++k)
                                push_be(want[leaf], cols[(c * 3 + k) * n + br]);
                        }
                    }
                }
                check_leaves(out, want, "blake3_comp_poly_leaves_ext3");
            }
        }
    }
    printf("ext3 + comp-poly leaf kernels: read pattern matches the CPU leaf spec\n");
}

// FRI leaves: two consecutive ext3 values from an interleaved vector, 48 bytes,
// no bit reversal. Under one block, so this is the chain's single-compression
// case at a length that is neither 64 nor a block multiple.
void fri_leaf_kernel_reads_the_specified_bytes() {
    for (uint64_t num_leaves : {1ull, 2ull, 8ull, 33ull}) {
        std::vector<uint64_t> evals(num_leaves * 2 * 3);
        for (size_t i = 0; i < evals.size(); ++i) evals[i] = sample(0xF41, i);
        std::vector<uint8_t> out(num_leaves * 32, 0);
        CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
            blake3_fri_leaves_ext3(evals.data(), num_leaves, out.data());
        }
        std::vector<std::vector<uint8_t>> want(num_leaves);
        for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
            for (int i = 0; i < 6; ++i) push_be(want[leaf], evals[leaf * 6 + i]);
        }
        check_leaves(out, want, "blake3_fri_leaves_ext3");
    }
    printf("FRI leaf kernel: read pattern matches the CPU leaf spec\n");
}

// The row-major row-pair kernels, plain and column-ranged. `m` is the row
// stride; the ranged variant hashes only `[col_start, col_end)` while the stride
// stays full, which is how preprocessed tables commit two column ranges to
// separate trees over one LDE.
void row_major_leaf_kernels_read_the_specified_bytes() {
    for (uint32_t log_n : {2u, 4u, 6u}) {
        for (uint64_t m : {1ull, 5ull, 13ull}) {
            uint64_t n = 1ull << log_n;
            uint64_t num_leaves = n / 2;
            std::vector<uint64_t> data(n * m);
            for (size_t i = 0; i < data.size(); ++i) data[i] = sample(log_n * 7 + m, i);

            {
                std::vector<uint8_t> out(num_leaves * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                    blake3_leaves_base_row_major_row_pair(data.data(), m, n, log_n, out.data());
                }
                std::vector<std::vector<uint8_t>> want(num_leaves);
                for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                    for (int k = 0; k < 2; ++k) {
                        uint64_t br = reverse_index(2 * leaf + k, log_n);
                        for (uint64_t c = 0; c < m; ++c) push_be(want[leaf], data[br * m + c]);
                    }
                }
                check_leaves(out, want, "blake3_leaves_base_row_major_row_pair");
            }

            // Every non-empty column range, so the boundary handling is checked
            // rather than sampled.
            for (uint64_t cs = 0; cs < m; ++cs) {
                for (uint64_t ce = cs + 1; ce <= m; ++ce) {
                    std::vector<uint8_t> out(num_leaves * 32, 0);
                    CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                        blake3_leaves_base_row_major_row_pair_range(data.data(), m, cs, ce, n,
                                                                    log_n, out.data());
                    }
                    std::vector<std::vector<uint8_t>> want(num_leaves);
                    for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                        for (int k = 0; k < 2; ++k) {
                            uint64_t br = reverse_index(2 * leaf + k, log_n);
                            for (uint64_t c = cs; c < ce; ++c) push_be(want[leaf], data[br * m + c]);
                        }
                    }
                    check_leaves(out, want, "blake3_leaves_base_row_major_row_pair_range");
                }
            }
        }
    }
    printf("row-major leaf kernels: read pattern matches the CPU leaf spec, all column ranges\n");
}

// The full-range ranged kernel must be the unranged one — the same bytes by two
// code paths. A cheap check that the range arithmetic has no off-by-one at the
// boundary it is most likely to have one at.
void the_full_range_variant_equals_the_plain_one() {
    const uint32_t log_n = 5;
    const uint64_t n = 1ull << log_n, m = 7, num_leaves = n / 2;
    std::vector<uint64_t> data(n * m);
    for (size_t i = 0; i < data.size(); ++i) data[i] = sample(0xBEEF, i);

    std::vector<uint8_t> plain(num_leaves * 32, 0), ranged(num_leaves * 32, 0);
    CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
        blake3_leaves_base_row_major_row_pair(data.data(), m, n, log_n, plain.data());
    }
    CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
        blake3_leaves_base_row_major_row_pair_range(data.data(), m, 0, m, n, log_n, ranged.data());
    }
    check(plain == ranged, "the full-range kernel must equal the unranged one");
    printf("row-major range [0, m) equals the plain kernel\n");
}

// NEGATIVE CONTROL for the leaf checks: the leaves must depend on the data and
// on the row index. Every check above compares kernel output to an expectation
// built from the same buffer, and all of them would pass if the kernel emitted a
// constant and the expectation happened to be that constant.
void leaves_depend_on_data_and_row() {
    const uint32_t log_n = 4;
    const uint64_t n = 1ull << log_n, num_cols = 3;
    std::vector<uint64_t> cols(num_cols * n);
    for (size_t i = 0; i < cols.size(); ++i) cols[i] = sample(0xD00D, i);

    std::vector<uint8_t> a(n * 32, 0), b(n * 32, 0);
    CUDA_HOST_FOR_EACH_THREAD(t, n) {
        blake3_leaves_base_batched(cols.data(), n, num_cols, n, log_n, a.data());
    }
    cols[n + 3] ^= 1ull;  // one element of one column
    CUDA_HOST_FOR_EACH_THREAD(t, n) {
        blake3_leaves_base_batched(cols.data(), n, num_cols, n, log_n, b.data());
    }
    check(a != b, "a one-element change must move some leaf");

    bool all_same = true;
    for (uint64_t i = 1; i < n; ++i) {
        if (memcmp(a.data(), a.data() + i * 32, 32) != 0) {
            all_same = false;
            break;
        }
    }
    check(!all_same, "all leaves identical — the kernel is not reading its row index");
    printf("negative control: leaves depend on the data and on the row index\n");
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
    printf("\n-- chain construction --\n");
    chain_against_official_multiblock_vectors();
    chain_against_the_committed_six_round_table();
    device_chain_matches_the_reference();
    device_chain_at_64_bytes_is_the_parent();
    printf("\n-- leaf kernels --\n");
    base_leaf_kernels_read_the_specified_bytes();
    ext3_leaf_kernels_read_the_specified_bytes();
    fri_leaf_kernel_reads_the_specified_bytes();
    row_major_leaf_kernels_read_the_specified_bytes();
    the_full_range_variant_equals_the_plain_one();
    leaves_depend_on_data_and_row();
    if (failures != 0) {
        printf("\n*** %d FAILURE(S) ***\n", failures);
        return 1;
    }
    printf("\nALL HOST KAT CHECKS PASS\n");
    printf("NOTE: arithmetic only. nvcc acceptance and GPU execution are covered "
           "by tests/blake3_*.rs, which need a GPU.\n");
    return 0;
}
