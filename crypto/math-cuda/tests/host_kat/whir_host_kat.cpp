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
//   3. The TILE form of the lean rounds (`whir_tile_plan_at`,
//      `whir_tile_contract_at`, `whir_tile_round_at`, `whir_tile_fold_at`) —
//      one pass over the message, then every round on tables `2^u` times
//      smaller — against BOTH the lean rounds of 2 and the materialised
//      opening, round by round, on random stacks: one to eight classes meeting
//      in a column of tiles, gaps, a column taller than the bound blocks, equal
//      points apart, columns read from a store that is not laid out as the
//      stack, one to six rounds. EQUAL values. The planner of these stacks is
//      this file's own: the host planner in Rust is not what it is checked
//      against.
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

// ───────────────────────────────── the tile form ─────────────────────────────

// A stacked polynomial as the tile form reads it: shares of at least `2^u`
// positions each, their points' tables split at `u` in every row, one `lo`
// table per class (points whose last `u` coordinates are equal, limb for
// limb), and the columns' data in a store that is NOT laid out as the stack.
struct TileStack {
    uint32_t n, u, lean, slots, classes;
    std::vector<Share> shares;
    std::vector<uint64_t> store;                      // the columns, back to front
    std::vector<uint64_t> eqbuf, rows, starts, ends;  // the share form's inputs, `lo_bits = u`
    std::vector<uint64_t> tile_starts, tile_ends;     // the same bounds in tiles
};

bool same_limbs(const std::vector<Fe3> &a, const std::vector<Fe3> &b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].a != b[i].a || a[i].b != b[i].b || a[i].c != b[i].c) return false;
    }
    return true;
}

// `suffixes` is how many distinct last-`u`-coordinate tails the points draw
// from: the classes. `tall` puts one column of `2^(n−1)` first.
TileStack tile_stack(uint32_t n, uint32_t u, uint32_t lean, uint32_t suffixes, bool tall) {
    TileStack t;
    t.n = n;
    t.u = u;
    t.lean = lean;
    std::vector<std::vector<Fe3>> tails(suffixes);
    for (auto &tail : tails) {
        for (uint32_t i = 0; i < u; ++i) tail.push_back(random_ext());
    }
    // A few heads per height, so columns of one height share whole points too.
    std::vector<std::vector<std::vector<Fe3>>> heads(n + 1);
    auto point_for = [&](uint32_t m) {
        auto &pool = heads[m];
        size_t pick = next_u64() % 3;
        while (pool.size() <= pick) {
            std::vector<Fe3> head;
            for (uint32_t i = 0; i + u < m; ++i) head.push_back(random_ext());
            pool.push_back(head);
        }
        std::vector<Fe3> point = pool[pick];
        const auto &tail = tails[next_u64() % suffixes];
        point.insert(point.end(), tail.begin(), tail.end());
        return point;
    };
    uint64_t len = 1ull << n, cursor = 0, stop = len;
    if (tall) {
        uint64_t at = (next_u64() & 1) ? len / 2 : 0;
        t.shares.push_back({at, n - 1, point_for(n - 1), random_ext()});
        if (at == 0) {
            cursor = len / 2;
        } else {
            stop = len / 2;
        }
    }
    while (true) {
        // Mostly the sizes around a bound block, where columns of several
        // points meet in one column of tiles.
        uint32_t top = n - lean + 1 < n - 1 ? n - lean + 1 : n - 1;
        if (top < u) top = u;
        uint32_t m = u + next_u64() % (top - u + 1);
        if (next_u64() % 4 == 0) m = u + next_u64() % 2;
        if (m > n - 1) m = n - 1;
        uint64_t size = 1ull << m;
        uint64_t at = (cursor + size - 1) / size * size;
        if (next_u64() % 5 == 0) at += size;  // a gap
        if (at + size > stop) break;
        t.shares.push_back({at, m, point_for(m), random_ext()});
        cursor = at + size;
    }
    std::sort(t.shares.begin(), t.shares.end(),
              [](const Share &a, const Share &b) { return a.offset < b.offset; });

    // The store: the last column first.
    std::vector<uint64_t> data_at(t.shares.size());
    for (size_t c = t.shares.size(); c-- > 0;) {
        data_at[c] = t.store.size();
        for (uint64_t r = 0; r < (1ull << t.shares[c].num_vars); ++r) t.store.push_back(raw_limb());
    }
    std::vector<std::pair<std::vector<Fe3>, uint64_t>> lo_tables;  // (tail, where)
    for (size_t c = 0; c < t.shares.size(); ++c) {
        const Share &s = t.shares[c];
        std::vector<Fe3> hi(s.point.begin(), s.point.end() - u);
        std::vector<Fe3> lo(s.point.end() - u, s.point.end());
        uint64_t hi_at = t.eqbuf.size() / 3;
        for (uint64_t v : eq_table(hi)) t.eqbuf.push_back(v);
        uint64_t lo_at = 0;
        bool known = false;
        for (const auto &seen : lo_tables) {
            if (same_limbs(seen.first, lo)) {
                lo_at = seen.second;
                known = true;
            }
        }
        if (!known) {
            lo_at = t.eqbuf.size() / 3;
            for (uint64_t v : eq_table(lo)) t.eqbuf.push_back(v);
            lo_tables.push_back({lo, lo_at});
        }
        uint64_t row[WHIR_LEAN_SHARE_WORDS] = {s.offset, data_at[c], u, hi_at, lo_at,
                                               s.scale.a, s.scale.b, s.scale.c, 0};
        t.rows.insert(t.rows.end(), row, row + WHIR_LEAN_SHARE_WORDS);
        t.starts.push_back(s.offset);
        t.ends.push_back(s.offset + (1ull << s.num_vars));
        t.tile_starts.push_back(s.offset >> u);
        t.tile_ends.push_back((s.offset + (1ull << s.num_vars)) >> u);
    }
    t.classes = lo_tables.size();
    t.slots = t.classes < (1u << lean) ? t.classes : (1u << lean);
    return t;
}

// The tile form's tables for a stack: the tile map, each column's slots, `Ω`
// and `Φ` (`slots` tables of `2^(n−u)` each).
struct TileTables {
    std::vector<uint16_t> tilemap;
    std::vector<uint64_t> slot_lo, omega, phi;
};

TileTables tile_tables(const TileStack &t) {
    const uint32_t nu = t.n - t.u;
    const uint64_t tiles = 1ull << nu, cols = 1ull << (nu - t.lean);
    TileTables out;
    out.tilemap.resize(tiles);
    out.slot_lo.resize(cols * t.slots);
    out.omega.resize(t.slots * tiles * 3);
    out.phi.resize(t.slots * tiles * 3);
    CUDA_HOST_SINGLE_THREAD();
    whir_lean_colmap(out.tilemap.data(), tiles, t.tile_starts.data(), t.tile_ends.data(),
                     t.shares.size());
    for (uint64_t col = 0; col < cols; ++col) {
        whir_tile_plan_at(out.tilemap.data(), t.rows.data(), t.eqbuf.data(), nu, t.lean, t.u,
                          t.slots, col, out.slot_lo.data(), out.omega.data());
    }
    for (uint64_t tile = 0; tile < tiles; ++tile) {
        Fe3 acc[WHIR_TILE_MAX_SLOTS];
        whir_tile_contract_at(out.tilemap.data(), t.rows.data(), t.eqbuf.data(), t.store.data(),
                              out.slot_lo.data(), nu, t.lean, t.u, t.slots, tile, 0, 1, acc);
        for (uint32_t s = 0; s < t.slots; ++s) {
            whir_store3(out.phi.data() + (s * tiles + tile) * 3, acc[s]);
        }
    }
    return out;
}

// One stack through the three: the materialised opening, the lean rounds and
// the tile form. Returns the most slots a column of tiles used.
uint32_t tile_form_on(const TileStack &t, const char *name) {
    const uint32_t n = t.n, u = t.u, lean = t.lean, slots = t.slots, nu = n - u;
    const uint64_t len = 1ull << n, tiles = 1ull << nu, cols = 1ull << (nu - lean);
    char what[160];

    // ── the materialised opening ──
    std::vector<Fe3> w(len, ext3::zero()), f(len, ext3::zero());
    for (size_t c = 0; c < t.shares.size(); ++c) {
        const Share &s = t.shares[c];
        for (uint64_t r = 0; r < (1ull << s.num_vars); ++r) {
            w[s.offset + r] = ext3::mul(s.scale, eq_at(s.point, r));
            f[s.offset + r] = ext3::make(t.store[t.rows[c * WHIR_LEAN_SHARE_WORDS + 1] + r], 0, 0);
        }
    }
    Fe3 claim = ext3::zero();
    for (uint64_t x = 0; x < len; ++x) claim = ext3::add(claim, ext3::mul(w[x], f[x]));

    // ── the lean rounds' position map against the tile form's tile map ──
    std::vector<uint16_t> colmap(len);
    CUDA_HOST_SINGLE_THREAD();
    whir_lean_colmap(colmap.data(), len, t.starts.data(), t.ends.data(), t.shares.size());
    TileTables tt = tile_tables(t);
    bool map_ok = true;
    for (uint64_t x = 0; x < len; ++x) map_ok &= tt.tilemap[x >> u] == colmap[x];
    snprintf(what, sizeof what, "%s: a tile lies in one column or one gap", name);
    check(map_ok, what);

    // ── the plan and the one pass ──
    uint32_t most = 0;
    bool prefix_ok = true;
    for (uint64_t col = 0; col < cols; ++col) {
        uint32_t used = 0;
        while (used < slots && tt.slot_lo[col * slots + used] != WHIR_TILE_NO_SLOT) ++used;
        for (uint32_t s = used; s < slots; ++s) {
            prefix_ok &= tt.slot_lo[col * slots + s] == WHIR_TILE_NO_SLOT;
        }
        most = used > most ? used : most;
    }
    snprintf(what, sizeof what, "%s: a column's slots are a prefix", name);
    check(prefix_ok, what);
    bool lanes_ok = true;
    const uint64_t lanes = (1ull << u) < 4 ? (1ull << u) : 4;
    for (uint64_t tile = 0; tile < tiles; ++tile) {
        Fe3 part[WHIR_TILE_MAX_SLOTS], sum[WHIR_TILE_MAX_SLOTS];
        for (uint32_t s = 0; s < slots; ++s) sum[s] = ext3::zero();
        for (uint64_t lane = 0; lane < lanes; ++lane) {
            whir_tile_contract_at(tt.tilemap.data(), t.rows.data(), t.eqbuf.data(), t.store.data(),
                                  tt.slot_lo.data(), nu, lean, u, slots, tile, lane, lanes, part);
            for (uint32_t s = 0; s < slots; ++s) sum[s] = ext3::add(sum[s], part[s]);
        }
        for (uint32_t s = 0; s < slots; ++s) {
            lanes_ok &= same_value(sum[s], whir_load3(tt.phi.data() + (s * tiles + tile) * 3));
        }
    }
    snprintf(what, sizeof what, "%s: the pass in lanes sums to the pass in one", name);
    check(lanes_ok, what);
    Fe3 tile_claim = ext3::zero();
    for (uint64_t i = 0; i < slots * tiles; ++i) {
        tile_claim = ext3::add(tile_claim, ext3::mul(whir_load3(tt.omega.data() + i * 3),
                                                     whir_load3(tt.phi.data() + i * 3)));
    }
    snprintf(what, sizeof what, "%s: the tiles' sum is the opening's claim", name);
    check(same_value(tile_claim, claim), what);

    // ── the rounds ──
    const uint64_t nodes[6] = {1, 0, 0, 2, 0, 0};
    std::vector<uint64_t> bound;
    uint64_t width = len, tile_half = tiles / 2;
    for (uint32_t s = 1; s <= lean; ++s) {
        uint64_t half = width / 2;
        Fe3 want[2] = {ext3::zero(), ext3::zero()};
        for (uint64_t j = 0; j < half; ++j) {
            for (int ti = 0; ti < 2; ++ti) {
                Fe3 node = ext3::make(nodes[ti * 3], 0, 0);
                Fe3 wt = ext3::add(w[j], ext3::mul(node, ext3::sub(w[j + half], w[j])));
                Fe3 ft = ext3::add(f[j], ext3::mul(node, ext3::sub(f[j + half], f[j])));
                want[ti] = ext3::add(want[ti], ext3::mul(wt, ft));
            }
        }
        std::vector<Fe3> prefix;
        for (size_t i = 0; i < bound.size(); i += 3) prefix.push_back(whir_load3(&bound[i]));
        std::vector<uint64_t> eqc = eq_table(prefix);
        Fe3 share_form[WHIR_LEAN_MAX_T] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
        for (uint64_t rest = 0; rest < (1ull << (n - s)); ++rest) {
            whir_lean_round_at(colmap.data(), t.rows.data(), t.eqbuf.data(), t.store.data(), n, s,
                               eqc.data(), nodes, 2, rest, share_form);
        }
        Fe3 got[WHIR_LEAN_MAX_T] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
        for (uint64_t i = 0; i < tile_half; ++i) {
            whir_tile_round_at(tt.omega.data(), tt.phi.data(), tiles, slots, tile_half, nodes, 2, i,
                               got);
        }
        for (int ti = 0; ti < 2; ++ti) {
            snprintf(what, sizeof what, "%s: round %u node %d: tiles = the materialised opening",
                     name, s, ti + 1);
            check(same_value(got[ti], want[ti]), what);
            snprintf(what, sizeof what, "%s: round %u node %d: tiles = the lean round", name, s,
                     ti + 1);
            check(same_value(got[ti], share_form[ti]), what);
        }
        Fe3 alpha = random_ext();
        for (uint64_t j = 0; j < half; ++j) {
            w[j] = ext3::add(w[j], ext3::mul(alpha, ext3::sub(w[j + half], w[j])));
            f[j] = ext3::add(f[j], ext3::mul(alpha, ext3::sub(f[j + half], f[j])));
        }
        for (uint32_t k = 0; k < slots; ++k) {
            for (uint64_t i = 0; i < tile_half; ++i) {
                whir_tile_fold_at(tt.omega.data() + k * tiles * 3, tile_half, i, alpha);
                whir_tile_fold_at(tt.phi.data() + k * tiles * 3, tile_half, i, alpha);
            }
        }
        width = half;
        tile_half /= 2;
        bound.push_back(alpha.a);
        bound.push_back(alpha.b);
        bound.push_back(alpha.c);
    }

    // ── the bound weight is `Σ_κ Ω_κ^bound(R)·eq(λ_κ, j)`, and `Φ^bound` the
    //    bound message's tiles at their slots' points ──
    bool weight_ok = true, message_ok = true;
    for (uint64_t y = 0; y < width; ++y) {
        uint64_t tile = y >> u, j = y & ((1ull << u) - 1);
        Fe3 sum = ext3::zero();
        for (uint32_t k = 0; k < slots; ++k) {
            uint64_t lo_at = tt.slot_lo[(tile & (cols - 1)) * slots + k];
            if (lo_at == WHIR_TILE_NO_SLOT) continue;
            sum = ext3::add(sum, ext3::mul(whir_load3(tt.omega.data() + (k * tiles + tile) * 3),
                                           whir_load3(t.eqbuf.data() + (lo_at + j) * 3)));
        }
        weight_ok &= same_value(sum, w[y]);
    }
    for (uint64_t tile = 0; tile < (width >> u); ++tile) {
        for (uint32_t k = 0; k < slots; ++k) {
            uint64_t lo_at = tt.slot_lo[(tile & (cols - 1)) * slots + k];
            if (lo_at == WHIR_TILE_NO_SLOT) continue;
            Fe3 sum = ext3::zero();
            for (uint64_t j = 0; j < (1ull << u); ++j) {
                sum = ext3::add(sum, ext3::mul(whir_load3(t.eqbuf.data() + (lo_at + j) * 3),
                                               f[(tile << u) + j]));
            }
            message_ok &= same_value(sum, whir_load3(tt.phi.data() + (k * tiles + tile) * 3));
        }
    }
    snprintf(what, sizeof what, "%s: the bound weight from the bound slot tables", name);
    check(weight_ok, what);
    snprintf(what, sizeof what, "%s: the bound slot tables of the message", name);
    check(message_ok, what);
    return most;
}

// The first round's node-2 value from the tile form alone, for the controls.
Fe3 tile_first_round(const TileStack &t) {
    const uint64_t tiles = 1ull << (t.n - t.u);
    TileTables tt = tile_tables(t);
    const uint64_t nodes[6] = {1, 0, 0, 2, 0, 0};
    Fe3 got[WHIR_LEAN_MAX_T] = {ext3::zero(), ext3::zero(), ext3::zero(), ext3::zero()};
    for (uint64_t i = 0; i < tiles / 2; ++i) {
        whir_tile_round_at(tt.omega.data(), tt.phi.data(), tiles, t.slots, tiles / 2, nodes, 2, i,
                           got);
    }
    return got[1];
}

void tile_form_equals_the_lean_rounds_and_the_materialised_opening() {
    int stacks = 0, by_slots[WHIR_TILE_MAX_SLOTS + 1] = {0}, gaps = 0, talls = 0;
    for (uint32_t trial = 0; trial < 240; ++trial) {
        uint32_t n = 9 + trial % 4;            // 9..12
        uint32_t u = 2 + (trial / 4) % 4;      // 2..5
        uint32_t lean = 1 + (trial / 16) % 6;  // 1..6
        if (lean > n - u) lean = n - u;
        uint32_t suffixes = 1 + trial % 9;     // classes offered: 1..9
        bool tall = trial % 5 == 0;
        TileStack t = tile_stack(n, u, lean, suffixes, tall);
        if (t.shares.empty() || t.slots > WHIR_TILE_MAX_SLOTS) continue;
        uint64_t cells = 0;
        for (const Share &s : t.shares) cells += 1ull << s.num_vars;
        gaps += cells < (1ull << n);
        talls += tall;
        char name[96];
        snprintf(name, sizeof name, "tile form #%u (n=%u u=%u rounds=%u classes=%u)", trial, n, u,
                 lean, t.classes);
        uint32_t most = tile_form_on(t, name);
        ++by_slots[most];
        ++stacks;
    }
    check(by_slots[1] > 0 && by_slots[2] > 0 && by_slots[4] > 0 && by_slots[8] > 0,
          "the stacks cover one, two, four and eight classes meeting in a column of tiles");
    check(gaps > 0 && talls > 0, "the stacks cover gaps and a column taller than the bound blocks");

    // Controls: the comparison sees one cell of the message, one entry of a
    // class's table, and a share's scale.
    TileStack t = tile_stack(10, 3, 3, 2, false);
    int before = failures;
    tile_form_on(t, "control: untouched");
    check(failures == before, "control: the untouched stack passes");
    Fe3 base_line = tile_first_round(t);
    TileStack cell = t;
    cell.store[cell.rows[1]] = goldilocks::add(cell.store[cell.rows[1]], 1);
    check(!same_value(tile_first_round(cell), base_line),
          "control: one cell of the message moves the round");
    TileStack table = t;
    table.eqbuf[(table.rows[4] + 1) * 3] = goldilocks::add(table.eqbuf[(table.rows[4] + 1) * 3], 1);
    check(!same_value(tile_first_round(table), base_line),
          "control: one entry of a class's table moves the round");
    TileStack scale = t;
    scale.rows[5] = goldilocks::add(scale.rows[5], 1);
    check(!same_value(tile_first_round(scale), base_line), "control: a share's scale moves the round");
    printf("tile form: %d stacks (most classes meeting in a column of tiles 1/2/3/4/5/6/7/8: "
           "%d/%d/%d/%d/%d/%d/%d/%d; %d with gaps, %d with a tall column) equal the lean rounds "
           "and the materialised opening\n",
           stacks, by_slots[1], by_slots[2], by_slots[3], by_slots[4], by_slots[5], by_slots[6],
           by_slots[7], by_slots[8], gaps, talls);
}

}  // namespace

int main() {
    fused_fold_is_raw_identical();
    lean_rounds_equal_the_materialised_opening();
    tile_form_equals_the_lean_rounds_and_the_materialised_opening();
    if (failures) {
        printf("whir host KAT: %d FAILURE(S)\n", failures);
        return 1;
    }
    printf("whir host KAT: all passed\n");
    return 0;
}
