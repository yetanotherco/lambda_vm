// Known-answer tests for `kernels/rpx.cu`, run on the host.
//
// WHY THIS EXISTS. The GPU parity suite runs only where a GPU does, and per-PR
// CI has none — GPU CI is merge_group-only. This compiles the real kernel
// source through `cuda_host_shim.h` and pins its arithmetic in seconds, with no
// GPU and no cargo, exactly as `blake3_host_kat.cpp` does for BLAKE3.
//
// WHAT IT COVERS: the field primitives the kernel is built from, the MDS, both
// S-boxes, the cubic extension, the seven-round schedule, the rate-8 overwrite
// duplex leaf, the Merkle parent, and the raw-vs-canonical representation.
//
// WHAT IT DOES NOT COVER, and what the GPU tests are still required for:
// whether nvcc accepts the file, and every property of execution rather than
// arithmetic — grid indexing, register pressure, local-memory spills from the
// sponge's dynamic indexing. Passing here is necessary, never sufficient.
//
// HOW THE ANCHORING LAYERS. Nothing here is checked only against itself:
//   1. The field primitives (`goldilocks::mul/add`, `ext3::dot3`) against
//      schoolbook `__int128` arithmetic — the definition, no shared code.
//   2. The MDS against its per-term definition; the S-boxes against generic
//      exponentiation (including `x^{1/7}` as `x^INV_ALPHA`); the cubic
//      extension against naive polynomial multiplication reduced by
//      `φ³ = φ + 1` — the same independent algorithms `rpx.rs`'s own tests use.
//   3. ★ EXTERNAL: RPX's FB round IS RPO's round with RPO's constants. Seven
//      `fb_round(s, r)` compose to RPO256, and that composition is replayed over
//      miden-crypto's nineteen `hash_elements` vectors, which nothing in this
//      tree produced. That pins ARK1/ARK2, the MDS row and orientation, both
//      S-box chains and the sponge lane convention from outside.
//   4. ★ THE ORACLE: the Rust host `Rpx256` (`prover/src/lfm/rpx.rs`), through
//      the tables `prover/tests/rpx_host_kat_vectors.rs` prints — the bare
//      permutation, the leaf sponge at seven lengths, the parent. miden
//      publishes no RPX vector, so the E round and the schedule rest on this
//      layer alone, as the Rust module's own provenance note says they must.
//   5. Negative controls: RPX ≠ RPO on the same state; every input lane
//      reaches the output; raw (`≥ p`) and canonical inputs agree; outputs are
//      canonical.
//   6. The cost model, COUNTED rather than asserted from a comment.
//   7. Every leaf kernel, both Merkle compressors and the permutation probe
//      replayed thread by thread through the shim against the CPU leaf spec
//      and the host parent — the read patterns and the node encoding, with the
//      hash over them anchored by the layers above.
//
// Build and run with `make test-rpx-host-kat`.

#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include "cuda_host_shim.h"

// The kernel under test. Included, not linked: the shim turns its device
// functions into host functions, and there is no other way to call them.
// RPX_HOST_OP_COUNT turns on its field-op counters (layer 6).
#define RPX_HOST_OP_COUNT
#include "rpx.cu"

#include "rpx_kat_vectors.h"

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

typedef unsigned __int128 u128;
const uint64_t P = 0xFFFFFFFF00000001ull;
// `7^{-1} mod (p − 1)` — rpo.rs:96. Re-derived below rather than trusted.
const uint64_t INV_ALPHA = 10540996611094048183ull;

uint64_t canon(uint64_t x) { return x >= P ? x - P : x; }

// ===========================================================================
// Reference arithmetic: schoolbook over `__int128`. It shares no code with the
// kernel — it is the definition the kernel's shortcuts are checked against.
// ===========================================================================

uint64_t ref_mul(uint64_t a, uint64_t b) {
    return (uint64_t)(((u128)canon(a) * (u128)canon(b)) % P);
}

uint64_t ref_add(uint64_t a, uint64_t b) {
    return (uint64_t)(((u128)canon(a) + (u128)canon(b)) % P);
}

uint64_t ref_pow(uint64_t x, uint64_t e) {
    uint64_t r = 1, b = canon(x);
    while (e != 0) {
        if (e & 1) r = ref_mul(r, b);
        b = ref_mul(b, b);
        e >>= 1;
    }
    return r;
}

struct RefExt {
    uint64_t c[3];
};

// Naive polynomial multiplication reduced by `φ³ = φ + 1`, `φ⁴ = φ² + φ` — the
// obvious slow way, as `rpx.rs:341-352` writes it, so it shares no structure
// with the kernel's regrouped closed form.
RefExt ref_ext_mul(const RefExt &a, const RefExt &b) {
    uint64_t c[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) c[i + j] = ref_add(c[i + j], ref_mul(a.c[i], b.c[j]));
    }
    RefExt r;
    r.c[0] = ref_add(c[0], c[3]);
    r.c[1] = ref_add(ref_add(c[1], c[3]), c[4]);
    r.c[2] = ref_add(c[2], c[4]);
    return r;
}

RefExt ref_ext_pow(RefExt a, unsigned e) {
    RefExt r = {{1, 0, 0}};
    while (e != 0) {
        if (e & 1) r = ref_ext_mul(r, a);
        a = ref_ext_mul(a, a);
        e >>= 1;
    }
    return r;
}

// The MDS as defined: `out_i = Σ_j ROW[(j − i) mod 12] · s_j`, one reduced
// field multiplication per term (rpo.rs:522-524).
void ref_mds(const uint64_t in[12], uint64_t out[12]) {
    static const uint64_t ROW[12] = {7, 23, 8, 26, 13, 10, 9, 7, 6, 22, 21, 8};
    for (int i = 0; i < 12; ++i) {
        uint64_t acc = 0;
        for (int j = 0; j < 12; ++j) acc = ref_add(acc, ref_mul(ROW[(j + 12 - i) % 12], in[j]));
        out[i] = acc;
    }
}

// A deterministic value stream. Every fifth value is a RAW representation in
// `[p, 2^64)` — the field's non-canonical storage, which the kernel must read
// as `value − p` — so the reduction paths are exercised rather than assumed.
uint64_t splitmix(uint64_t &seed) {
    seed += 0x9E3779B97F4A7C15ull;
    uint64_t z = seed;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

uint64_t sample(uint64_t &seed, uint64_t i) {
    uint64_t x = splitmix(seed);
    // Raw values above p exist only for canonical values below 2^32 − 1.
    return (i % 5 == 0) ? (x % 0xFFFFFFFFull) + P : x % P;
}

// Values at every edge of the representation: zero, one, the modulus and its
// neighbours (raw zero, raw one), EPSILON and 2^32, the top of the u64 range.
const uint64_t EDGES[] = {0ull,          1ull,          2ull,          P - 1,       P,
                          P + 1,         0xFFFFFFFFull, 0x100000000ull, 1ull << 63, ~0ull,
                          ~0ull - 1,     0x0123456789ABCDEFull};
const int NUM_EDGES = (int)(sizeof(EDGES) / sizeof(EDGES[0]));

// ===========================================================================
// Layer 1 — the field primitives the kernel is built from.
// ===========================================================================

void field_primitives_match_schoolbook_arithmetic() {
    int checked = 0;
    for (int i = 0; i < NUM_EDGES; ++i) {
        for (int j = 0; j < NUM_EDGES; ++j) {
            const uint64_t a = EDGES[i], b = EDGES[j];
            check(canon(goldilocks::mul(a, b)) == ref_mul(a, b), "goldilocks::mul at an edge");
            check(canon(goldilocks::add(a, b)) == ref_add(a, b), "goldilocks::add at an edge");
            // Three equal products: the 128-bit sum overflows for the large edges.
            const uint64_t want = ref_add(ref_add(ref_mul(a, b), ref_mul(a, b)), ref_mul(a, b));
            check(canon(ext3::dot3(a, b, a, b, a, b)) == want, "ext3::dot3 at an edge (3 equal terms)");
            ++checked;
        }
    }
    // The two-overflow case explicitly: six maximal operands.
    {
        const uint64_t m = ~0ull;
        const uint64_t want = ref_add(ref_add(ref_mul(m, m), ref_mul(m, m)), ref_mul(m, m));
        check(canon(ext3::dot3(m, m, m, m, m, m)) == want, "ext3::dot3 with two 2^128 overflows");
        const uint64_t want1 = ref_add(ref_mul(m, m), ref_mul(m, m));
        check(canon(ext3::dot3(m, m, m, m, 0, 0)) == want1, "ext3::dot3 with one 2^128 overflow");
    }
    uint64_t seed = 0xF1E1D;
    for (int k = 0; k < 500; ++k) {
        uint64_t v[6];
        for (int t = 0; t < 6; ++t) v[t] = sample(seed, (uint64_t)k * 6 + t);
        const uint64_t want =
            ref_add(ref_add(ref_mul(v[0], v[1]), ref_mul(v[2], v[3])), ref_mul(v[4], v[5]));
        check(canon(ext3::dot3(v[0], v[1], v[2], v[3], v[4], v[5])) == want, "ext3::dot3 on random");
        check(canon(goldilocks::mul(v[0], v[1])) == ref_mul(v[0], v[1]), "goldilocks::mul on random");
        ++checked;
    }
    printf("field primitives vs schoolbook __int128: %d edge pairs + random, mul/add/dot3\n", checked);
}

// ===========================================================================
// Layer 2 — the building blocks against independent algorithms.
// ===========================================================================

void mds_matches_its_per_term_definition() {
    std::vector<std::vector<uint64_t>> states;
    states.push_back(std::vector<uint64_t>(12, 0));
    states.push_back(std::vector<uint64_t>(12, P - 1));
    states.push_back(std::vector<uint64_t>(12, ~0ull));  // the raw maximum: the u128 bound's worst case
    for (int k = 0; k < 12; ++k) {  // one-hot lanes pin the orientation
        std::vector<uint64_t> s(12, 0);
        s[k] = 1;
        states.push_back(s);
    }
    uint64_t seed = 0x3D5;
    for (int k = 0; k < 64; ++k) {
        std::vector<uint64_t> s(12);
        for (int i = 0; i < 12; ++i) s[i] = sample(seed, (uint64_t)k * 12 + i);
        states.push_back(s);
    }
    for (size_t n = 0; n < states.size(); ++n) {
        uint64_t got[12], want[12];
        memcpy(got, states[n].data(), sizeof(got));
        rpx::mds(got);
        ref_mds(states[n].data(), want);
        for (int i = 0; i < 12; ++i) {
            if (canon(got[i]) != want[i]) {
                printf("FAIL mds state %zu lane %d: got %llu want %llu\n", n, i,
                       (unsigned long long)canon(got[i]), (unsigned long long)want[i]);
                ++failures;
                break;
            }
        }
    }
    printf("MDS (u128-property port) vs per-term definition: %zu states incl. raw-max and one-hot\n",
           states.size());
}

void sboxes_are_the_seventh_power_and_its_inverse() {
    // `7 · INV_ALPHA ≡ 1 (mod p − 1)`, re-derived as rpo.rs:797-806 does.
    const u128 p_minus_one = (u128)P - 1;
    check(((u128)7 * (u128)INV_ALPHA) % p_minus_one == 1, "INV_ALPHA must invert 7 in the exponent group");

    std::vector<uint64_t> xs(EDGES, EDGES + NUM_EDGES);
    uint64_t seed = 0x5B0;
    for (int k = 0; k < 48; ++k) xs.push_back(sample(seed, (uint64_t)k));
    for (size_t n = 0; n < xs.size(); ++n) {
        const uint64_t x = xs[n];
        check(canon(rpx::sbox(x)) == ref_pow(x, 7), "sbox(x) must be x^7");
        check(canon(rpx::inv_sbox(x)) == ref_pow(x, INV_ALPHA), "inv_sbox(x) must be x^INV_ALPHA");
        check(canon(rpx::sbox(rpx::inv_sbox(x))) == canon(x), "sbox(inv_sbox(x)) must be x");
        check(canon(rpx::inv_sbox(rpx::sbox(x))) == canon(x), "inv_sbox(sbox(x)) must be x");
    }
    check(rpx::inv_sbox(0) == 0, "inv_sbox(0) must be 0 (the padding row's fixed point)");
    check(canon(rpx::inv_sbox(P)) == 0, "inv_sbox(raw zero) must be 0");
    check(canon(rpx::inv_sbox(1)) == 1, "inv_sbox(1) must be 1");
    printf("S-boxes vs generic exponentiation: %zu values, x^7, x^{1/7}, both compositions\n",
           xs.size());
}

void cubic_extension_matches_naive_polynomial_arithmetic() {
    // The reduction rule itself, pinned on the basis: φ·φ² = φ³ = 1 + φ, and
    // φ²·φ² = φ⁴ = φ + φ².
    {
        rpx::CubicExt phi = {0, 1, 0}, phi2 = {0, 0, 1}, one = {1, 0, 0};
        rpx::CubicExt r = rpx::ext_mul(phi, phi2);
        check(canon(r.c0) == 1 && canon(r.c1) == 1 && canon(r.c2) == 0, "φ·φ² must be 1 + φ");
        r = rpx::ext_mul(phi2, phi2);
        check(canon(r.c0) == 0 && canon(r.c1) == 1 && canon(r.c2) == 1, "φ²·φ² must be φ + φ²");
        r = rpx::ext_mul(phi2, one);
        check(canon(r.c0) == 0 && canon(r.c1) == 0 && canon(r.c2) == 1, "1 must be the identity");
    }
    std::vector<RefExt> as, bs;
    as.push_back(RefExt{{P - 1, P - 1, P - 1}});
    bs.push_back(RefExt{{P - 1, P - 1, P - 1}});
    as.push_back(RefExt{{~0ull, ~0ull, ~0ull}});  // raw maxima
    bs.push_back(RefExt{{~0ull, ~0ull, ~0ull}});
    as.push_back(RefExt{{0, 0, 0}});
    bs.push_back(RefExt{{P - 1, 0, 1}});
    uint64_t seed = 0xE3;
    for (int k = 0; k < 64; ++k) {
        RefExt a, b;
        for (int t = 0; t < 3; ++t) {
            a.c[t] = sample(seed, (uint64_t)k * 6 + t);
            b.c[t] = sample(seed, (uint64_t)k * 6 + 3 + t);
        }
        as.push_back(a);
        bs.push_back(b);
    }
    for (size_t n = 0; n < as.size(); ++n) {
        const rpx::CubicExt a = {as[n].c[0], as[n].c[1], as[n].c[2]};
        const rpx::CubicExt b = {bs[n].c[0], bs[n].c[1], bs[n].c[2]};
        const rpx::CubicExt m = rpx::ext_mul(a, b);
        const RefExt mw = ref_ext_mul(as[n], bs[n]);
        check(canon(m.c0) == mw.c[0] && canon(m.c1) == mw.c[1] && canon(m.c2) == mw.c[2],
              "ext_mul must equal the naive polynomial product");
        const rpx::CubicExt s = rpx::ext_square(a);
        const RefExt sw = ref_ext_mul(as[n], as[n]);
        check(canon(s.c0) == sw.c[0] && canon(s.c1) == sw.c[1] && canon(s.c2) == sw.c[2],
              "ext_square must equal the naive square");
        const rpx::CubicExt p7 = rpx::ext_power7(a);
        const RefExt pw = ref_ext_pow(as[n], 7);
        check(canon(p7.c0) == pw.c[0] && canon(p7.c1) == pw.c[1] && canon(p7.c2) == pw.c[2],
              "ext_power7 must equal generic exponentiation to 7");
    }
    printf("cubic extension (φ³ = φ + 1) vs naive polynomial arithmetic: %zu pairs, mul/square/power7\n",
           as.size());
}

// ===========================================================================
// Layer 3 — ★ the EXTERNAL anchor: seven FB rounds are RPO256.
// ===========================================================================

// RPO256's permutation composed from the kernel's FB round — rpo.rs:567-583.
void rpo_permute(uint64_t s[12]) {
    rpx::fb_round(s, 0);
    rpx::fb_round(s, 1);
    rpx::fb_round(s, 2);
    rpx::fb_round(s, 3);
    rpx::fb_round(s, 4);
    rpx::fb_round(s, 5);
    rpx::fb_round(s, 6);
    for (int i = 0; i < 12; ++i) s[i] = goldilocks::canonical(s[i]);
}

// miden's `hash_elements` in this lane convention — a transcription of the
// test-only `rpo.rs:747-767`: capacity lane 8 takes `len % 8`, the rate is
// OVERWRITTEN, the tail zero-padded, the digest is lanes 0-3.
void miden_hash_elements(const uint64_t *elements, size_t n, uint64_t out[4]) {
    uint64_t state[12] = {0};
    state[8] = (uint64_t)(n % 8);
    size_t i = 0;
    for (size_t k = 0; k < n; ++k) {
        state[i++] = elements[k];
        if (i == 8) {
            rpo_permute(state);
            i = 0;
        }
    }
    if (i > 0) {
        for (; i < 8; ++i) state[i] = 0;
        rpo_permute(state);
    }
    for (int d = 0; d < 4; ++d) out[d] = state[d];
}

void seven_fb_rounds_reproduce_the_miden_rpo_vectors() {
    check(NUM_MIDEN_HASH_ELEMENTS == 19, "miden vector table lost entries");
    int matched = 0;
    for (int n = 0; n < NUM_MIDEN_HASH_ELEMENTS; ++n) {
        uint64_t elements[19];
        for (int k = 0; k <= n; ++k) elements[k] = (uint64_t)k;
        uint64_t got[4];
        miden_hash_elements(elements, (size_t)n + 1, got);
        bool ok = true;
        for (int d = 0; d < 4; ++d) ok = ok && got[d] == MIDEN_HASH_ELEMENTS[n][d];
        if (!ok) {
            printf("FAIL miden hash_elements(0..=%d)\n  got  %llu %llu %llu %llu\n  want %llu %llu %llu %llu\n",
                   n, (unsigned long long)got[0], (unsigned long long)got[1],
                   (unsigned long long)got[2], (unsigned long long)got[3],
                   (unsigned long long)MIDEN_HASH_ELEMENTS[n][0],
                   (unsigned long long)MIDEN_HASH_ELEMENTS[n][1],
                   (unsigned long long)MIDEN_HASH_ELEMENTS[n][2],
                   (unsigned long long)MIDEN_HASH_ELEMENTS[n][3]);
            ++failures;
        } else {
            ++matched;
        }
    }
    // The compress layout, pinned the way rpo.rs:789-795 pins it: one
    // permutation of `[0..8 ‖ 0⁴]` is the eight-element vector, so
    // `[left ‖ right ‖ zero capacity]` with left = 0..4, right = 4..8 IS
    // `Rpo256::merge` — the layout `rpx::compress` builds.
    {
        uint64_t s[12] = {0, 1, 2, 3, 4, 5, 6, 7, 0, 0, 0, 0};
        rpo_permute(s);
        bool ok = true;
        for (int d = 0; d < 4; ++d) ok = ok && s[d] == MIDEN_HASH_ELEMENTS[7][d];
        check(ok, "permute([0..8 ‖ 0⁴]) must be miden's eight-element vector (compress layout)");
    }
    printf("★ EXTERNAL: seven fb_round(s, r) = RPO256 vs miden-crypto hash_elements: %d/19 matched\n",
           matched);
}

// ===========================================================================
// Layer 4 — ★ the Rust oracle.
//
// ⚠ Every comparison here is RAW: `s[i] == v.output[i]`, never
// `canon(s[i]) == …`. The tables are canonical by construction (the generator
// canonicalises), and `permute` ends in a canonicalisation loop that makes
// digests byte-comparable to the host's; a check that canonicalised the kernel
// side would pass with that loop deleted, and so would a raw check on outputs
// that merely happen to be canonical — all but a 2^-32 slice per lane. The
// "canonicalisation witness" row and `the_canonicalisation_loop_is_pinned…`
// below are what make the loop observable.
// ===========================================================================

void rpx_permutation_matches_the_rust_oracle() {
    check(NUM_RPX_PERMUTATION_VECTORS >= 8,
          "Rust-oracle permutation table must hold >= 8 vectors (run the generator, see rpx_kat_vectors.h)");
    bool saw_zero = false, saw_p_minus_one = false;
    int matched = 0;
    for (int n = 0; n < NUM_RPX_PERMUTATION_VECTORS; ++n) {
        const RpxPermutationVector &v = RPX_PERMUTATION_VECTORS[n];
        bool all_zero = true, all_pm1 = true;
        uint64_t s[12];
        for (int i = 0; i < 12; ++i) {
            s[i] = v.input[i];
            all_zero = all_zero && v.input[i] == 0;
            all_pm1 = all_pm1 && v.input[i] == P - 1;
        }
        saw_zero = saw_zero || all_zero;
        saw_p_minus_one = saw_p_minus_one || all_pm1;
        rpx::permute(s);
        bool ok = true;
        for (int i = 0; i < 12; ++i) ok = ok && s[i] == v.output[i];
        if (!ok) {
            printf("FAIL rpx permutation vector %d (%s)\n", n, v.name);
            for (int i = 0; i < 12; ++i) {
                if (s[i] != v.output[i]) {
                    printf("  lane %2d got %llu (raw) want %llu\n", i, (unsigned long long)s[i],
                           (unsigned long long)v.output[i]);
                }
            }
            ++failures;
        } else {
            ++matched;
        }
    }
    check(saw_zero, "the permutation table must include the all-zero state");
    check(saw_p_minus_one, "the permutation table must include the all-(p-1) state");
    printf("★ ORACLE: rpx::permute vs Rust Rpx256::permute: %d/%d vectors matched\n", matched,
           NUM_RPX_PERMUTATION_VECTORS);
}

// The array-form transcription of `algebraic_commit::sponge_leaf` (:169-184),
// over the kernel's permutation — so the STREAMING struct's block bookkeeping
// is checked against the direct transcription at every length, independently
// of which lengths the oracle table carries.
void ref_sponge_leaf(const uint64_t *felts, size_t n, uint64_t digest[4]) {
    uint64_t state[12] = {0};
    state[8] = (uint64_t)(n % 8);
    state[9] = 0x4C4D464Cull;  // u32::from_le_bytes(b"LFML")
    if (n == 0) {
        for (int d = 0; d < 4; ++d) digest[d] = state[d];
        return;
    }
    for (size_t start = 0; start < n; start += 8) {
        for (size_t lane = 0; lane < 8; ++lane) {
            state[lane] = (start + lane < n) ? felts[start + lane] : 0;
        }
        rpx::permute(state);
    }
    for (int d = 0; d < 4; ++d) digest[d] = state[d];
}

void leaf_sponge_matches_the_rust_oracle() {
    // The streaming struct against the array transcription, lengths 0..40.
    {
        uint64_t seed = 0x1EAF;
        std::vector<uint64_t> felts(40);
        for (size_t i = 0; i < felts.size(); ++i) felts[i] = sample(seed, i);
        for (size_t n = 0; n <= felts.size(); ++n) {
            uint64_t got[4], want[4];
            rpx::sponge_leaf(felts.data(), n, got);
            ref_sponge_leaf(felts.data(), n, want);
            check(memcmp(got, want, sizeof(got)) == 0, "rpx::Sponge must equal the sponge_leaf transcription");
        }
        uint64_t empty[4] = {1, 1, 1, 1};
        rpx::sponge_leaf(felts.data(), 0, empty);
        check(empty[0] == 0 && empty[1] == 0 && empty[2] == 0 && empty[3] == 0,
              "the empty leaf's digest is the zero rate lanes, with NO permutation");
        uint64_t one[4];
        rpx::sponge_leaf(felts.data(), 1, one);
        check(one[0] != 0 || one[1] != 0 || one[2] != 0 || one[3] != 0, "a one-felt leaf must permute");
        printf("leaf: rpx::Sponge vs sponge_leaf transcription at 41 lengths (0..40)\n");
    }
    // The oracle table: exactly the gate's seven lengths.
    std::set<uint32_t> lengths;
    for (int n = 0; n < NUM_RPX_LEAF_VECTORS; ++n) lengths.insert(RPX_LEAF_VECTORS[n].len);
    const uint32_t required[7] = {0, 1, 7, 8, 9, 16, 17};
    bool all_present = NUM_RPX_LEAF_VECTORS > 0;
    for (int k = 0; k < 7; ++k) all_present = all_present && lengths.count(required[k]) == 1;
    check(all_present,
          "Rust-oracle leaf table must hold lengths 0, 1, 7, 8, 9, 16, 17 (run the generator, see rpx_kat_vectors.h)");
    int matched = 0;
    for (int n = 0; n < NUM_RPX_LEAF_VECTORS; ++n) {
        const RpxLeafVector &v = RPX_LEAF_VECTORS[n];
        check(v.len <= (uint32_t)RPX_LEAF_KAT_MAX_FELTS, "leaf vector wider than the table row");
        uint64_t got[4];
        rpx::sponge_leaf(v.felts, v.len, got);
        bool ok = true;
        for (int d = 0; d < 4; ++d) ok = ok && got[d] == v.digest[d];
        if (!ok) {
            printf("FAIL rpx leaf vector len=%u\n  got  %llu %llu %llu %llu\n  want %llu %llu %llu %llu\n",
                   v.len, (unsigned long long)got[0], (unsigned long long)got[1],
                   (unsigned long long)got[2], (unsigned long long)got[3],
                   (unsigned long long)v.digest[0], (unsigned long long)v.digest[1],
                   (unsigned long long)v.digest[2], (unsigned long long)v.digest[3]);
            ++failures;
        } else {
            ++matched;
        }
    }
    printf("★ ORACLE: rpx::sponge_leaf vs Rust sponge_leaf(Rpx): %d/%d lengths matched\n", matched,
           NUM_RPX_LEAF_VECTORS);
}

void parent_matches_the_rust_oracle() {
    check(NUM_RPX_PARENT_VECTORS >= 1,
          "Rust-oracle parent table must hold >= 1 vector (run the generator, see rpx_kat_vectors.h)");
    int matched = 0;
    for (int n = 0; n < NUM_RPX_PARENT_VECTORS; ++n) {
        const RpxParentVector &v = RPX_PARENT_VECTORS[n];
        uint64_t got[4];
        rpx::compress(v.left, v.right, got);
        bool ok = true;
        for (int d = 0; d < 4; ++d) ok = ok && got[d] == v.digest[d];
        if (!ok) {
            printf("FAIL rpx parent vector %d (%s)\n  got  %llu %llu %llu %llu\n  want %llu %llu %llu %llu\n",
                   n, v.name, (unsigned long long)got[0], (unsigned long long)got[1],
                   (unsigned long long)got[2], (unsigned long long)got[3],
                   (unsigned long long)v.digest[0], (unsigned long long)v.digest[1],
                   (unsigned long long)v.digest[2], (unsigned long long)v.digest[3]);
            ++failures;
        } else {
            ++matched;
        }
        // Structure: a parent is ONE permutation of `[l ‖ r ‖ 0⁴]`, and the
        // order of the children matters.
        uint64_t s[12] = {v.left[0], v.left[1], v.left[2], v.left[3], v.right[0], v.right[1],
                          v.right[2], v.right[3], 0, 0, 0, 0};
        rpx::permute(s);
        check(memcmp(s, got, sizeof(got)) == 0, "compress must be permute([l ‖ r ‖ 0⁴]) truncated");
        uint64_t swapped[4];
        rpx::compress(v.right, v.left, swapped);
        bool same_children = memcmp(v.left, v.right, sizeof(swapped)) == 0;
        check(same_children || memcmp(swapped, got, sizeof(got)) != 0, "compress(r, l) must differ from compress(l, r)");
    }
    printf("★ ORACLE: rpx::compress vs Rust HasherKind::Rpx.compress: %d/%d parents matched\n", matched,
           NUM_RPX_PARENT_VECTORS);
}

// ★ The pin on the canonicalisation loop. The witness row's M-round MDS output
// lane 0 is `p − ARK1[6][0] + 1`, so the device's final `add` returns the raw
// twin `p + 1` for a field value of 1 — deterministically, since neither that
// sum nor the MDS reduction can wrap there. Replaying the rounds without the
// loop must therefore show a lane ≥ p (or the witness has gone stale and no
// longer witnesses anything), and `permute` must then return the oracle's
// canonical digits RAW — which a kernel without the loop cannot.
void the_canonicalisation_loop_is_pinned_by_the_witness() {
    const RpxPermutationVector *w = nullptr;
    for (int n = 0; n < NUM_RPX_PERMUTATION_VECTORS; ++n) {
        if (strcmp(RPX_PERMUTATION_VECTORS[n].name, "canonicalisation witness") == 0) {
            w = &RPX_PERMUTATION_VECTORS[n];
        }
    }
    check(w != nullptr,
          "the permutation table must carry the 'canonicalisation witness' row (run the generator, see rpx_kat_vectors.h)");
    if (w == nullptr) return;

    uint64_t s[12];
    memcpy(s, w->input, sizeof(s));
    rpx::fb_round(s, 0);
    rpx::ext_round(s, 1);
    rpx::fb_round(s, 2);
    rpx::ext_round(s, 3);
    rpx::fb_round(s, 4);
    rpx::ext_round(s, 5);
    rpx::final_round(s, 6);
    int twins = 0;
    for (int i = 0; i < 12; ++i) twins += (s[i] >= P) ? 1 : 0;
    check(twins > 0, "the witness must leave a raw lane >= p before the canonicalisation loop");
    check(s[0] == P + 1, "the witness's lane 0 must be the raw twin p + 1 before the loop");
    for (int i = 0; i < 12; ++i) {
        check(canon(s[i]) == w->output[i], "the witness's field values must be the oracle's");
    }

    uint64_t full[12];
    memcpy(full, w->input, sizeof(full));
    rpx::permute(full);
    const bool loop_present = memcmp(full, w->output, sizeof(full)) == 0;
    check(loop_present,
          "permute must return the witness's digits RAW — the canonicalisation loop is missing");
    if (loop_present) {
        printf("★ canonicalisation pin: witness leaves %d raw lane(s) >= p before the loop; permute() returns them canonical\n",
               twins);
    }
}

// ===========================================================================
// Layer 5 — negative controls and the representation.
// ===========================================================================

void rpx_is_not_rpo() {
    // rpx.rs:577-580: the two share constants, an MDS and three of seven
    // rounds, so a schedule bug could collapse one into the other.
    uint64_t a[12], b[12];
    for (int i = 0; i < 12; ++i) a[i] = b[i] = (uint64_t)i;
    rpx::permute(a);
    rpo_permute(b);
    check(memcmp(a, b, sizeof(a)) != 0, "RPX must not be RPO on the same state");
    uint64_t z[12] = {0};
    rpx::permute(z);
    bool nonzero = false;
    for (int i = 0; i < 12; ++i) nonzero = nonzero || z[i] != 0;
    check(nonzero, "with its constants present, permute(0) must not be 0");
    printf("negative control: RPX(0..12) != RPO(0..12); permute(0) != 0\n");
}

void raw_and_canonical_inputs_agree_and_outputs_are_canonical() {
    uint64_t seed = 0xCA0;
    for (int k = 0; k < 32; ++k) {
        uint64_t raw[12], can[12];
        for (int i = 0; i < 12; ++i) {
            // Canonical values below 2^32 − 1 have a raw twin `c + p`; alternate
            // lanes between the twin and a plain canonical value.
            const uint64_t c = splitmix(seed) % 0xFFFFFFFFull;
            const bool twin = ((k + i) % 3) != 0;
            can[i] = twin ? c : splitmix(seed) % P;
            raw[i] = twin ? c + P : can[i];
        }
        uint64_t r1[12], c1[12];
        memcpy(r1, raw, sizeof(r1));
        memcpy(c1, can, sizeof(c1));
        rpx::permute(r1);
        rpx::permute(c1);
        check(memcmp(r1, c1, sizeof(r1)) == 0, "permute(raw) must equal permute(canonical)");
        for (int i = 0; i < 12; ++i) check(c1[i] < P, "permute output must be canonical");

        uint64_t d_raw[4], d_can[4];
        rpx::sponge_leaf(raw, 12, d_raw);
        rpx::sponge_leaf(can, 12, d_can);
        check(memcmp(d_raw, d_can, sizeof(d_raw)) == 0, "sponge_leaf(raw) must equal sponge_leaf(canonical)");

        uint64_t p_raw[4], p_can[4];
        rpx::compress(raw, raw + 4, p_raw);
        rpx::compress(can, can + 4, p_can);
        check(memcmp(p_raw, p_can, sizeof(p_raw)) == 0, "compress(raw) must equal compress(canonical)");
    }
    printf("representation: raw [p, 2^64) inputs agree with canonical; outputs canonical (32 states)\n");
}

void every_input_lane_reaches_the_output() {
    uint64_t seed = 0x1A4E;
    uint64_t base[12];
    for (int i = 0; i < 12; ++i) base[i] = splitmix(seed) % P;
    uint64_t out0[12];
    memcpy(out0, base, sizeof(out0));
    rpx::permute(out0);
    for (int k = 0; k < 12; ++k) {
        uint64_t s[12];
        memcpy(s, base, sizeof(s));
        s[k] = (s[k] + 1) % P;
        rpx::permute(s);
        check(memcmp(s, out0, sizeof(s)) != 0, "changing one input lane must move the output");
    }
    printf("negative control: each of the 12 input lanes moves the output\n");
}

// ===========================================================================
// Layer 6 — the cost model, counted.
// ===========================================================================

struct Counted {
    unsigned long long mul, dot3, add;
};

template <typename F>
Counted count_ops(F f) {
    rpx::g_ops = rpx::OpCount{0, 0, 0};
    f();
    return Counted{rpx::g_ops.mul, rpx::g_ops.dot3, rpx::g_ops.add};
}

void the_cost_model_is_what_the_header_claims() {
    uint64_t s[12];
    for (int i = 0; i < 12; ++i) s[i] = (uint64_t)i + 1;
    const Counted fb = count_ops([&] { rpx::fb_round(s, 0); });
    const Counted ext = count_ops([&] { rpx::ext_round(s, 1); });
    const Counted fin = count_ops([&] { rpx::final_round(s, 6); });
    const Counted all = count_ops([&] { rpx::permute(s); });
    const Counted rpo = count_ops([&] { rpo_permute(s); });
    const Counted inv = count_ops([&] { (void)rpx::inv_sbox(s[0]); });
    const Counted fwd = count_ops([&] { (void)rpx::sbox(s[0]); });
    const Counted emul = count_ops([&] {
        rpx::CubicExt a = {s[0], s[1], s[2]};
        (void)rpx::ext_mul(a, a);
    });

    printf("op counts (Goldilocks mul | 3-term dot3 | add); MDS = 288 narrow 32x32 MACs each, uncounted:\n");
    printf("  x^7 (sbox)          %4llu | %3llu | %3llu\n", fwd.mul, fwd.dot3, fwd.add);
    printf("  x^{1/7} (inv_sbox)  %4llu | %3llu | %3llu   (63 squarings + 9 products)\n", inv.mul,
           inv.dot3, inv.add);
    printf("  ext_mul             %4llu | %3llu | %3llu   (9 wide products in 3 reductions)\n", emul.mul,
           emul.dot3, emul.add);
    printf("  FB round            %4llu | %3llu | %3llu   + 2 MDS\n", fb.mul, fb.dot3, fb.add);
    printf("  E round             %4llu | %3llu | %3llu   (4 triples x power7)\n", ext.mul, ext.dot3,
           ext.add);
    printf("  M round             %4llu | %3llu | %3llu   + 1 MDS\n", fin.mul, fin.dot3, fin.add);
    printf("  RPX permutation     %4llu | %3llu | %3llu   + 7 MDS (2016 MACs)\n", all.mul, all.dot3,
           all.add);
    printf("  RPO permutation     %4llu | %3llu | %3llu   + 14 MDS (4032 MACs), for comparison\n",
           rpo.mul, rpo.dot3, rpo.add);
    printf("  inverse S-box share of RPX field multiplications: %llu / %llu\n", 3ull * 12ull * inv.mul,
           all.mul);

    check(fwd.mul == 4 && inv.mul == 72, "S-box costs must be 4 and 72 multiplications");
    check(emul.mul == 0 && emul.dot3 == 3 && emul.add == 2, "ext_mul must be 3 dot3 + 2 adds");
    check(fb.mul == 912 && fb.dot3 == 0 && fb.add == 48, "FB round must be 912 mul / 48 add");
    check(ext.mul == 0 && ext.dot3 == 48 && ext.add == 44, "E round must be 48 dot3 / 44 add");
    check(fin.mul == 0 && fin.dot3 == 0 && fin.add == 24, "M round must be 24 add");
    check(all.mul == 2736 && all.dot3 == 144 && all.add == 300, "RPX permutation must be 2736 mul / 144 dot3 / 300 add");
    check(rpo.mul == 6384 && rpo.dot3 == 0 && rpo.add == 336, "RPO permutation must be 6384 mul / 336 add");
}

// ===========================================================================
// Layer 7 — the leaf kernels, the Merkle compressors and the probe, replayed
// thread by thread through the shim.
//
// What a leaf hashes is the CPU `leaves_bit_reversed_grouped` sequence —
// bit-reversed rows, each column by column, an ext3 element as its three
// components — and the hash over it is the `sponge_leaf` transcription pinned
// in layer 4. So each kernel is checked for its READ PATTERN and its node
// ENCODING (`digest_to_commitment`: four canonical felts, big-endian), with the
// permutation anchored separately above. Raw `[p, 2^64)` values are fed in,
// since that is what an LDE buffer holds.
// ===========================================================================

uint64_t reverse_index(uint64_t i, uint32_t log_n) { return __brevll(i) >> (64 - log_n); }

// The host leaf over `felts`: `sponge_leaf`, then `digest_to_commitment`.
void expected_leaf(const std::vector<uint64_t> &felts, uint8_t out[32]) {
    uint64_t d[4];
    ref_sponge_leaf(felts.data(), felts.size(), d);
    for (int i = 0; i < 4; ++i) {
        const uint64_t c = canon(d[i]);
        for (int b = 0; b < 8; ++b) out[i * 8 + b] = (uint8_t)(c >> (56 - 8 * b));
    }
}

std::string hex32(const uint8_t *b) {
    std::string s(64, '\0');
    for (int i = 0; i < 32; ++i) snprintf(&s[i * 2], 3, "%02x", (unsigned)b[i]);
    return s;
}

void check_leaves(const std::vector<uint8_t> &got, const std::vector<std::vector<uint64_t>> &want,
                  const char *what) {
    if (got.size() != want.size() * 32) {
        printf("FAIL %s: leaf count %zu vs %zu\n", what, got.size() / 32, want.size());
        ++failures;
        return;
    }
    for (size_t i = 0; i < want.size(); ++i) {
        uint8_t expect[32];
        expected_leaf(want[i], expect);
        if (memcmp(got.data() + i * 32, expect, 32) != 0) {
            printf("FAIL %s: leaf %zu\n  got  %s\n  want %s\n", what, i, hex32(got.data() + i * 32).c_str(),
                   hex32(expect).c_str());
            ++failures;
            return;
        }
    }
}

// The two column-major base kernels: one leaf per bit-reversed row, and one per
// bit-reversed row pair.
void base_leaf_kernels_read_the_specified_felts() {
    for (uint32_t log_n : {2u, 4u, 6u}) {
        for (uint64_t num_cols : {1ull, 5ull, 8ull, 17ull}) {
            const uint64_t n = 1ull << log_n;
            std::vector<uint64_t> cols(num_cols * n);
            uint64_t seed = log_n * 31 + num_cols;
            for (size_t i = 0; i < cols.size(); ++i) cols[i] = sample(seed, i);
            {
                std::vector<uint8_t> out(n * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, n) {
                    rpx_leaves_base_batched(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint64_t>> want(n);
                for (uint64_t leaf = 0; leaf < n; ++leaf) {
                    const uint64_t br = reverse_index(leaf, log_n);
                    for (uint64_t c = 0; c < num_cols; ++c) want[leaf].push_back(cols[c * n + br]);
                }
                check_leaves(out, want, "rpx_leaves_base_batched");
            }
            {
                const uint64_t num_leaves = n / 2;
                std::vector<uint8_t> out(num_leaves * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                    rpx_leaves_base_row_pair_batched(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint64_t>> want(num_leaves);
                for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                    for (int k = 0; k < 2; ++k) {
                        const uint64_t br = reverse_index(2 * leaf + k, log_n);
                        for (uint64_t c = 0; c < num_cols; ++c) want[leaf].push_back(cols[c * n + br]);
                    }
                }
                check_leaves(out, want, "rpx_leaves_base_row_pair_batched");
            }
        }
    }
    printf("base leaf kernels: read pattern + node encoding match the CPU leaf spec\n");
}

// The ext3 kernels over the de-interleaved three-slab layout.
void ext3_leaf_kernels_read_the_specified_felts() {
    for (uint32_t log_n : {2u, 4u, 6u}) {
        for (uint64_t num_cols : {1ull, 3ull, 11ull}) {
            const uint64_t n = 1ull << log_n;
            std::vector<uint64_t> cols(num_cols * 3 * n);
            uint64_t seed = log_n * 17 + num_cols;
            for (size_t i = 0; i < cols.size(); ++i) cols[i] = sample(seed, i);
            {
                std::vector<uint8_t> out(n * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, n) {
                    rpx_leaves_ext3_batched(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint64_t>> want(n);
                for (uint64_t leaf = 0; leaf < n; ++leaf) {
                    const uint64_t br = reverse_index(leaf, log_n);
                    for (uint64_t c = 0; c < num_cols; ++c) {
                        for (uint64_t k = 0; k < 3; ++k) want[leaf].push_back(cols[(c * 3 + k) * n + br]);
                    }
                }
                check_leaves(out, want, "rpx_leaves_ext3_batched");
            }
            {
                const uint64_t num_leaves = n / 2;
                std::vector<uint8_t> out(num_leaves * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                    rpx_comp_poly_leaves_ext3(cols.data(), n, num_cols, n, log_n, out.data());
                }
                std::vector<std::vector<uint64_t>> want(num_leaves);
                for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                    for (int j = 0; j < 2; ++j) {
                        const uint64_t br = reverse_index(2 * leaf + j, log_n);
                        for (uint64_t c = 0; c < num_cols; ++c) {
                            for (uint64_t k = 0; k < 3; ++k) want[leaf].push_back(cols[(c * 3 + k) * n + br]);
                        }
                    }
                }
                check_leaves(out, want, "rpx_comp_poly_leaves_ext3");
            }
        }
    }
    printf("ext3 + comp-poly leaf kernels: read pattern + node encoding match the CPU leaf spec\n");
}

// FRI leaves: two consecutive ext3 values from an interleaved vector, six felts,
// no bit reversal — the Pair backend's `hash_data`.
void fri_leaf_kernel_reads_the_specified_felts() {
    for (uint64_t num_leaves : {1ull, 2ull, 8ull, 33ull}) {
        std::vector<uint64_t> evals(num_leaves * 6);
        uint64_t seed = 0xF41;
        for (size_t i = 0; i < evals.size(); ++i) evals[i] = sample(seed, i);
        std::vector<uint8_t> out(num_leaves * 32, 0);
        CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) { rpx_fri_leaves_ext3(evals.data(), num_leaves, out.data()); }
        std::vector<std::vector<uint64_t>> want(num_leaves);
        for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
            for (int i = 0; i < 6; ++i) want[leaf].push_back(evals[leaf * 6 + i]);
        }
        check_leaves(out, want, "rpx_fri_leaves_ext3");
    }
    printf("FRI leaf kernel: read pattern + node encoding match the Pair backend's leaf\n");
}

// The row-major row-pair kernels, plain and column-ranged, every non-empty
// range.
void row_major_leaf_kernels_read_the_specified_felts() {
    for (uint32_t log_n : {2u, 4u, 6u}) {
        for (uint64_t m : {1ull, 5ull, 13ull}) {
            const uint64_t n = 1ull << log_n;
            const uint64_t num_leaves = n / 2;
            std::vector<uint64_t> data(n * m);
            uint64_t seed = log_n * 7 + m;
            for (size_t i = 0; i < data.size(); ++i) data[i] = sample(seed, i);
            {
                std::vector<uint8_t> out(num_leaves * 32, 0);
                CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                    rpx_leaves_base_row_major_row_pair(data.data(), m, n, log_n, out.data());
                }
                std::vector<std::vector<uint64_t>> want(num_leaves);
                for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                    for (int k = 0; k < 2; ++k) {
                        const uint64_t br = reverse_index(2 * leaf + k, log_n);
                        for (uint64_t c = 0; c < m; ++c) want[leaf].push_back(data[br * m + c]);
                    }
                }
                check_leaves(out, want, "rpx_leaves_base_row_major_row_pair");
            }
            for (uint64_t cs = 0; cs < m; ++cs) {
                for (uint64_t ce = cs + 1; ce <= m; ++ce) {
                    std::vector<uint8_t> out(num_leaves * 32, 0);
                    CUDA_HOST_FOR_EACH_THREAD(t, num_leaves) {
                        rpx_leaves_base_row_major_row_pair_range(data.data(), m, cs, ce, n, log_n, out.data());
                    }
                    std::vector<std::vector<uint64_t>> want(num_leaves);
                    for (uint64_t leaf = 0; leaf < num_leaves; ++leaf) {
                        for (int k = 0; k < 2; ++k) {
                            const uint64_t br = reverse_index(2 * leaf + k, log_n);
                            for (uint64_t c = cs; c < ce; ++c) want[leaf].push_back(data[br * m + c]);
                        }
                    }
                    check_leaves(out, want, "rpx_leaves_base_row_major_row_pair_range");
                }
            }
        }
    }
    printf("row-major leaf kernels: read pattern + node encoding match the CPU leaf spec, all column ranges\n");
}

// The host parent over two nodes: decode big-endian, compress, encode.
void expected_parent(const uint8_t *left, const uint8_t *right, uint8_t out[32]) {
    uint64_t l[4], r[4], d[4];
    for (int i = 0; i < 4; ++i) {
        l[i] = r[i] = 0;
        for (int b = 0; b < 8; ++b) {
            l[i] = (l[i] << 8) | left[i * 8 + b];
            r[i] = (r[i] << 8) | right[i * 8 + b];
        }
    }
    rpx::compress(l, r, d);
    for (int i = 0; i < 4; ++i) {
        for (int b = 0; b < 8; ++b) out[i * 8 + b] = (uint8_t)(canon(d[i]) >> (56 - 8 * b));
    }
}

// The Merkle level kernel replayed thread by thread up a 16-leaf tree, and the
// tail kernel replayed as a one-thread block (the shim's barrier is a no-op,
// so a single thread walking every pair in order is the tail's sequential
// meaning), both against the host parent over the same node buffer.
void merkle_compressors_match_the_host_parent() {
    const uint64_t num_leaves = 16;
    const uint64_t total = 2 * num_leaves - 1;
    // Nodes must be VALID digests (canonical big-endian felts) for the decode to
    // be meaningful, so the leaves are hashes of random felts, not random bytes.
    std::vector<uint8_t> leaves(num_leaves * 32);
    uint64_t seed = 0x3E11;
    for (uint64_t i = 0; i < num_leaves; ++i) {
        std::vector<uint64_t> f = {sample(seed, i), sample(seed, i + 1000)};
        expected_leaf(f, leaves.data() + i * 32);
    }

    std::vector<uint8_t> want(total * 32, 0);
    memcpy(want.data() + (num_leaves - 1) * 32, leaves.data(), leaves.size());
    for (uint64_t parent = num_leaves - 1; parent-- > 0;) {
        expected_parent(want.data() + (2 * parent + 1) * 32, want.data() + (2 * parent + 2) * 32,
                        want.data() + parent * 32);
    }

    // Level by level.
    std::vector<uint8_t> by_level(total * 32, 0);
    memcpy(by_level.data() + (num_leaves - 1) * 32, leaves.data(), leaves.size());
    uint64_t level_begin = num_leaves - 1;
    while (level_begin != 0) {
        const uint64_t new_begin = level_begin / 2;
        const uint64_t n_pairs = level_begin - new_begin;
        CUDA_HOST_FOR_EACH_THREAD(t, n_pairs) { rpx_merkle_level(by_level.data(), new_begin, n_pairs); }
        level_begin = new_begin;
    }
    check(by_level == want, "rpx_merkle_level must reproduce the host tree");

    // The tail, in one go.
    std::vector<uint8_t> by_tail(total * 32, 0);
    memcpy(by_tail.data() + (num_leaves - 1) * 32, leaves.data(), leaves.size());
    blockIdx.x = 0;
    threadIdx.x = 0;
    blockDim.x = 1;
    rpx_merkle_tail(by_tail.data(), num_leaves - 1);
    check(by_tail == want, "rpx_merkle_tail must reproduce the host tree");
    printf("Merkle compressors: level and tail kernels reproduce the host parent over a 16-leaf tree\n");
}

// The permutation probe replayed over the oracle table: pins its indexing.
void permute_probe_matches_the_oracle_table() {
    std::vector<uint64_t> in(NUM_RPX_PERMUTATION_VECTORS * 12), out(NUM_RPX_PERMUTATION_VECTORS * 12, 0);
    for (int n = 0; n < NUM_RPX_PERMUTATION_VECTORS; ++n) {
        for (int i = 0; i < 12; ++i) in[n * 12 + i] = RPX_PERMUTATION_VECTORS[n].input[i];
    }
    CUDA_HOST_FOR_EACH_THREAD(t, NUM_RPX_PERMUTATION_VECTORS) {
        rpx_permute_probe(in.data(), (uint64_t)NUM_RPX_PERMUTATION_VECTORS, out.data());
    }
    bool ok = true;
    for (int n = 0; n < NUM_RPX_PERMUTATION_VECTORS; ++n) {
        for (int i = 0; i < 12; ++i) ok = ok && out[n * 12 + i] == RPX_PERMUTATION_VECTORS[n].output[i];
    }
    check(ok, "rpx_permute_probe must reproduce the oracle table, raw");
    printf("permute probe: %d oracle states reproduced through the kernel entry point\n",
           NUM_RPX_PERMUTATION_VECTORS);
}

}  // namespace

int main() {
    printf("RPX device-kernel known-answer tests, host-compiled from crypto/math-cuda/kernels/rpx.cu\n\n");
    printf("-- layer 1/2: primitives and building blocks vs independent algorithms --\n");
    field_primitives_match_schoolbook_arithmetic();
    mds_matches_its_per_term_definition();
    sboxes_are_the_seventh_power_and_its_inverse();
    cubic_extension_matches_naive_polynomial_arithmetic();
    printf("\n-- layer 3: the external anchor --\n");
    seven_fb_rounds_reproduce_the_miden_rpo_vectors();
    printf("\n-- layer 4: the Rust oracle --\n");
    rpx_permutation_matches_the_rust_oracle();
    leaf_sponge_matches_the_rust_oracle();
    parent_matches_the_rust_oracle();
    the_canonicalisation_loop_is_pinned_by_the_witness();
    printf("\n-- layer 5: negative controls and representation --\n");
    rpx_is_not_rpo();
    raw_and_canonical_inputs_agree_and_outputs_are_canonical();
    every_input_lane_reaches_the_output();
    printf("\n-- layer 6: cost model --\n");
    the_cost_model_is_what_the_header_claims();
    printf("\n-- layer 7: leaf kernels, Merkle compressors and the probe, replayed thread by thread --\n");
    base_leaf_kernels_read_the_specified_felts();
    ext3_leaf_kernels_read_the_specified_felts();
    fri_leaf_kernel_reads_the_specified_felts();
    row_major_leaf_kernels_read_the_specified_felts();
    merkle_compressors_match_the_host_parent();
    permute_probe_matches_the_oracle_table();
    if (failures != 0) {
        printf("\n*** %d FAILURE(S) ***\n", failures);
        return 1;
    }
    printf("\nALL HOST KAT CHECKS PASS\n");
    printf("NOTE: arithmetic only. nvcc acceptance and GPU execution are phase 2's GPU tests.\n");
    return 0;
}
