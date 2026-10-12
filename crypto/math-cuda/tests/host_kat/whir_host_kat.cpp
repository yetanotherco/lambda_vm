// Known-answer tests for the WHIR round kernels in `kernels/whir_fold.cu`, run
// on the host through `cuda_host_shim.h` — no GPU, no nvcc, a second.
//
// WHAT IT COVERS:
//   1. The FUSED fold (`whir_fold_coset`, every level of a round in one pass)
//      against the level-by-level kernels it replaces (`whir_fold_base_ext3`,
//      `whir_fold_ext3`), replayed thread by thread: RAW-identical, limb for
//      limb, over base and extension inputs, k = 1..6, with non-canonical raw
//      inputs mixed in — the proof serializes raw limbs, so equal values are not
//      enough for this one.
//   2. The LEAN first rounds (`whir_lean_round_at`, `whir_lean_materialize_at`,
//      `whir_lean_colmap`) against the materialised opening they replace: a
//      weight written out from its shares as `scale·eq(z, row)` coordinate by
//      coordinate (no half tables), the message lifted, the round evaluated at
//      the nodes 1 and 2 over the pairs `(j, j + half)` and bound in place as
//      `lo + α(hi − lo)` — the order `Mle::fix_first_variable` and the device
//      session bind in. Six rounds, then the bound tables. EQUAL values
//      (canonical), which is what the lean path promises; the transcript
//      absorbs canonical bytes.
//
// WHAT IT DOES NOT COVER, and the GPU tests are still required for: whether
// nvcc accepts the file, the shared-memory squares and block reductions,
// grid-stride bounds under a real launch, register pressure. The field
// primitives themselves are pinned by `rpx_host_kat.cpp`'s schoolbook layer.
//
// Build and run with `make test-whir-host-kat`.

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

#include "cuda_host_shim.h"

// The kernels' shared memory, for a host that has no such thing. The KAT calls
// the device functions directly, never the block-reducing kernels, so these
// are only here for the file to link.
// `extern "C"` because the kernels that name it are.
#define __shared__
extern "C" {
uint64_t shared[3 * 1024];
}

#include "whir_fold.cu"

namespace {

int failures = 0;

void check(bool ok, const char *what) {
    if (!ok) {
        printf("FAIL: %s\n", what);
        ++failures;
    }
}

const uint64_t P = 0xFFFFFFFF00000001ull;

uint64_t state = 0x9E3779B97F4A7C15ull;
uint64_t next_u64() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

// A raw limb: canonical most of the time, and every eighth one a non-canonical
// representative in [p, 2^64) — the representation a device may hold.
uint64_t raw_limb() {
    uint64_t v = next_u64() % P;
    if ((next_u64() & 7) == 0 && v < 0xFFFFFFFFull) {
        v += P;
    }
    return v;
}

Fe3 random_ext() { return ext3::make(raw_limb(), raw_limb(), raw_limb()); }

bool same_value(const Fe3 &a, const Fe3 &b) {
    Fe3 x = ext3::canonical(a), y = ext3::canonical(b);
    return x.a == y.a && x.b == y.b && x.c == y.c;
}

// ─────────────────────────────── the fused fold ──────────────────────────────

// Level by level, as the device folds today: one launch per level, replayed.
std::vector<uint64_t> fold_level_by_level(const std::vector<uint64_t> &input, bool base,
                                          uint32_t k, uint64_t two_inv,
                                          const std::vector<uint64_t> &g_invs,
                                          const std::vector<uint64_t> &alphas) {
    std::vector<uint64_t> current = input;
    bool current_base = base;
    uint64_t elements = base ? input.size() : input.size() / 3;
    for (uint32_t level = 0; level < k; ++level) {
        uint64_t half = elements / 2;
        std::vector<uint64_t> out(half * 3);
        if (current_base) {
            CUDA_HOST_FOR_EACH_THREAD(j, half)
            whir_fold_base_ext3(current.data(), half, two_inv, g_invs[level],
                                alphas.data() + level * 3, out.data());
        } else {
            CUDA_HOST_FOR_EACH_THREAD(j, half)
            whir_fold_ext3(current.data(), half, two_inv, g_invs[level],
                           alphas.data() + level * 3, out.data());
        }
        current = out;
        current_base = false;
        elements = half;
    }
    return current;
}

std::vector<uint64_t> fold_fused(const std::vector<uint64_t> &input, bool base, uint32_t k,
                                 uint64_t two_inv, const std::vector<uint64_t> &g_invs,
                                 const std::vector<uint64_t> &alphas) {
    uint64_t elements = base ? input.size() : input.size() / 3;
    uint64_t n_out = elements >> k;
    std::vector<uint64_t> squares(WHIR_MAX_FOLD * WHIR_POW_BITS);
    for (uint32_t level = 0; level < k; ++level) {
        whir_fold_squares(squares.data(), level, g_invs[level]);
    }
    std::vector<uint64_t> out(n_out * 3);
    for (uint64_t j = 0; j < n_out; ++j) {
        Fe3 v = whir_fold_coset(input.data(), base, j, n_out, k, two_inv, squares.data(),
                                alphas.data());
        out[j * 3] = v.a;
        out[j * 3 + 1] = v.b;
        out[j * 3 + 2] = v.c;
    }
    return out;
}

void fused_fold_is_raw_identical() {
    int cases = 0;
    for (int base = 1; base >= 0; --base) {
        for (uint32_t k = 1; k <= WHIR_MAX_FOLD; ++k) {
            for (uint32_t log_n : {k, k + 1u, k + 4u, 13u}) {
                if (log_n < k) continue;
                uint64_t elements = 1ull << log_n;
                std::vector<uint64_t> input(base ? elements : elements * 3);
                for (auto &limb : input) limb = raw_limb();
                std::vector<uint64_t> g_invs(k), alphas(k * 3);
                for (auto &g : g_invs) g = raw_limb();
                for (auto &a : alphas) a = raw_limb();
                uint64_t two_inv = 0x7FFFFFFF80000001ull;  // 2^{-1} mod p
                std::vector<uint64_t> want =
                    fold_level_by_level(input, base, k, two_inv, g_invs, alphas);
                std::vector<uint64_t> got = fold_fused(input, base, k, two_inv, g_invs, alphas);
                char what[128];
                snprintf(what, sizeof what, "fused fold, %s, k=%u, n=2^%u: raw limbs",
                         base ? "base" : "ext3", k, log_n);
                check(want == got, what);
                ++cases;
            }
        }
    }
    // A control: the comparison sees a single limb.
    std::vector<uint64_t> input(1u << 8);
    for (auto &limb : input) limb = raw_limb();
    std::vector<uint64_t> g_invs = {raw_limb(), raw_limb()}, alphas(6);
    for (auto &a : alphas) a = raw_limb();
    std::vector<uint64_t> a = fold_fused(input, true, 2, 5, g_invs, alphas);
    input[17] = goldilocks::add(input[17], 1);
    std::vector<uint64_t> b = fold_fused(input, true, 2, 5, g_invs, alphas);
    check(a != b, "control: one input limb moves the fused output");
    printf("fused fold: %d cases raw-identical to the level-by-level kernels\n", cases);
}

// ───────────────────────────── the lean first rounds ─────────────────────────

struct Share {
    uint64_t offset;
    uint32_t num_vars;
    std::vector<Fe3> point;
    Fe3 scale;
};

// `eq(z, i)` with `i`'s most significant bit on `z[0]` — the layout every
// `eq` table in the prover has (`eq::eq_evals_into`).
Fe3 eq_at(const std::vector<Fe3> &z, uint64_t i) {
    Fe3 acc = ext3::one();
    uint32_t m = z.size();
    for (uint32_t c = 0; c < m; ++c) {
        bool bit = (i >> (m - 1 - c)) & 1;
        Fe3 factor = bit ? z[c] : ext3::sub(ext3::one(), z[c]);
        acc = ext3::mul(acc, factor);
    }
    return acc;
}

std::vector<uint64_t> eq_table(const std::vector<Fe3> &z) {
    std::vector<uint64_t> out;
    for (uint64_t i = 0; i < (1ull << z.size()); ++i) {
        Fe3 v = eq_at(z, i);
        out.push_back(v.a);
        out.push_back(v.b);
        out.push_back(v.c);
    }
    return out;
}

void lean_rounds_equal_the_materialised_opening() {
    const uint32_t n = 11;
    const uint64_t len = 1ull << n;
    // A stack of mixed heights, aligned to their sizes, with gaps: a table of
    // three columns sharing a point, a tall column crossing the first rounds'
    // blocks, short ones inside one block, and padding at the end.
    std::vector<Share> shares;
    std::vector<Fe3> table_point;
    for (int i = 0; i < 7; ++i) table_point.push_back(random_ext());
    for (int c = 0; c < 3; ++c) {
        shares.push_back({(uint64_t)c << 7, 7, table_point, random_ext()});
    }
    std::vector<Fe3> tall;
    for (int i = 0; i < 10; ++i) tall.push_back(random_ext());
    shares.push_back({1ull << 10, 10, tall, random_ext()});
    for (int c = 0; c < 6; ++c) {
        std::vector<Fe3> p;
        uint32_t vars = 2 + c % 3;
        for (uint32_t i = 0; i < vars; ++i) p.push_back(random_ext());
        uint64_t at = (1ull << 9) + ((uint64_t)c << 5);
        shares.push_back({at, vars, p, random_ext()});
    }
    // In stack order: the column map is a binary search over the offsets, and
    // the lean opening takes its shares sorted (`LeanRound0::new` refuses
    // otherwise).
    std::sort(shares.begin(), shares.end(),
              [](const Share &a, const Share &b) { return a.offset < b.offset; });
    // The message: base values where a column is, zero elsewhere.
    std::vector<uint64_t> data(len, 0);
    for (const Share &s : shares) {
        for (uint64_t r = 0; r < (1ull << s.num_vars); ++r) data[s.offset + r] = raw_limb();
    }

    // ── the materialised opening ──
    std::vector<Fe3> w(len, ext3::zero()), f(len);
    for (uint64_t x = 0; x < len; ++x) f[x] = ext3::make(data[x], 0, 0);
    for (const Share &s : shares) {
        for (uint64_t r = 0; r < (1ull << s.num_vars); ++r) {
            w[s.offset + r] = ext3::mul(s.scale, eq_at(s.point, r));
        }
    }

    // ── the lean opening's inputs ──
    std::vector<uint64_t> eqbuf, rows, starts, ends;
    for (const Share &s : shares) {
        uint32_t lo_bits = s.num_vars / 2;
        std::vector<Fe3> hi(s.point.begin(), s.point.end() - lo_bits);
        std::vector<Fe3> lo(s.point.end() - lo_bits, s.point.end());
        uint64_t hi_at = eqbuf.size() / 3;
        for (uint64_t v : eq_table(hi)) eqbuf.push_back(v);
        uint64_t lo_at = eqbuf.size() / 3;
        for (uint64_t v : eq_table(lo)) eqbuf.push_back(v);
        uint64_t row[WHIR_LEAN_SHARE_WORDS] = {s.offset, s.offset, lo_bits, hi_at, lo_at,
                                               s.scale.a, s.scale.b, s.scale.c, 0};
        rows.insert(rows.end(), row, row + WHIR_LEAN_SHARE_WORDS);
        starts.push_back(s.offset);
        ends.push_back(s.offset + (1ull << s.num_vars));
    }
    std::vector<uint16_t> colmap(len);
    CUDA_HOST_SINGLE_THREAD();
    whir_lean_colmap(colmap.data(), len, starts.data(), ends.data(), shares.size());
    uint64_t covered = 0;
    for (uint64_t x = 0; x < len; ++x) covered += colmap[x] != WHIR_LEAN_NONE;
    uint64_t cells = 0;
    for (const Share &s : shares) cells += 1ull << s.num_vars;
    check(covered == cells, "the column map covers exactly the columns' cells");

    // The share-form value at every position is the materialised one.
    bool values_ok = true;
    for (uint64_t x = 0; x < len; ++x) {
        Fe3 wv;
        uint64_t fv;
        whir_lean_value(colmap.data(), rows.data(), eqbuf.data(), data.data(), x, wv, fv);
        values_ok &= same_value(wv, w[x]) && goldilocks::canonical(fv) ==
                                                 goldilocks::canonical(f[x].a);
    }
    check(values_ok, "the share form reads back the materialised weight and message");

    const uint64_t t[6] = {1, 0, 0, 2, 0, 0};
    std::vector<uint64_t> bound;  // three limbs per challenge
    uint64_t width = len;
    for (uint32_t s = 1; s <= 6; ++s) {
        // The materialised round: pairs `(j, j + half)`, nodes 1 and 2.
        uint64_t half = width / 2;
        Fe3 want[2] = {ext3::zero(), ext3::zero()};
        for (uint64_t j = 0; j < half; ++j) {
            for (int ti = 0; ti < 2; ++ti) {
                Fe3 node = ext3::make(t[ti * 3], 0, 0);
                Fe3 wt = ext3::add(w[j], ext3::mul(node, ext3::sub(w[j + half], w[j])));
                Fe3 ft = ext3::add(f[j], ext3::mul(node, ext3::sub(f[j + half], f[j])));
                want[ti] = ext3::add(want[ti], ext3::mul(wt, ft));
            }
        }
        // The lean round over the shares.
        std::vector<Fe3> prefix;
        for (size_t i = 0; i < bound.size(); i += 3) {
            prefix.push_back(ext3::make(bound[i], bound[i + 1], bound[i + 2]));
        }
        std::vector<uint64_t> eqc = eq_table(prefix);
        Fe3 got[WHIR_LEAN_MAX_T] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
        for (uint64_t rest = 0; rest < (1ull << (n - s)); ++rest) {
            whir_lean_round_at(colmap.data(), rows.data(), eqbuf.data(), data.data(), n, s,
                               eqc.data(), t, 2, rest, got);
        }
        char what[96];
        snprintf(what, sizeof what, "round %u: the evaluation at node 1", s);
        check(same_value(got[0], want[0]), what);
        snprintf(what, sizeof what, "round %u: the evaluation at node 2", s);
        check(same_value(got[1], want[1]), what);

        // Bind: in place for the materialised tables, a challenge for the lean.
        Fe3 alpha = random_ext();
        for (uint64_t j = 0; j < half; ++j) {
            w[j] = ext3::add(w[j], ext3::mul(alpha, ext3::sub(w[j + half], w[j])));
            f[j] = ext3::add(f[j], ext3::mul(alpha, ext3::sub(f[j + half], f[j])));
        }
        width = half;
        bound.push_back(alpha.a);
        bound.push_back(alpha.b);
        bound.push_back(alpha.c);
    }

    // The tables the lean opening writes once its rounds are spent.
    std::vector<Fe3> prefix;
    for (size_t i = 0; i < bound.size(); i += 3) {
        prefix.push_back(ext3::make(bound[i], bound[i + 1], bound[i + 2]));
    }
    std::vector<uint64_t> eqfull = eq_table(prefix);
    bool tables_ok = true;
    for (uint64_t y = 0; y < width; ++y) {
        Fe3 wy, fy;
        whir_lean_materialize_at(colmap.data(), rows.data(), eqbuf.data(), data.data(), n, 6,
                                 eqfull.data(), y, wy, fy);
        tables_ok &= same_value(wy, w[y]) && same_value(fy, f[y]);
    }
    check(tables_ok, "the materialised tables after six rounds");

    // A control: a column whose scale moves moves the first round.
    rows[5] = goldilocks::add(rows[5], 1);
    std::vector<uint64_t> eqc = eq_table({});
    Fe3 moved[WHIR_LEAN_MAX_T] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
    Fe3 base_line[WHIR_LEAN_MAX_T] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
    for (uint64_t rest = 0; rest < (1ull << (n - 1)); ++rest) {
        whir_lean_round_at(colmap.data(), rows.data(), eqbuf.data(), data.data(), n, 1,
                           eqc.data(), t, 2, rest, moved);
    }
    rows[5] = goldilocks::sub(rows[5], 1);
    for (uint64_t rest = 0; rest < (1ull << (n - 1)); ++rest) {
        whir_lean_round_at(colmap.data(), rows.data(), eqbuf.data(), data.data(), n, 1,
                           eqc.data(), t, 2, rest, base_line);
    }
    check(!same_value(moved[1], base_line[1]), "control: a share's scale moves the round");
    printf("lean first rounds: 6 rounds and the bound tables equal the materialised opening\n");
}

}  // namespace

int main() {
    fused_fold_is_raw_identical();
    lean_rounds_equal_the_materialised_opening();
    if (failures) {
        printf("whir host KAT: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("whir host KAT: all passed\n");
    return 0;
}
