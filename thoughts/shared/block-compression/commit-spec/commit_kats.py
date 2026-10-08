"""
KATs for the LFM-native commitment layer — C1..C10.

DRAFT — PENDING MAURO RATIFICATION.

Run:  python3 commit_kats.py            (check against commit_kats.json)
      python3 commit_kats.py --write    (regenerate the vectors)

Discipline, inherited from `leaf_kats.py` / `transcript_kats.py`: every negative
control is paired with an HONEST leg, because a construction that rejected
everything would pass a negative-only suite.
"""

from __future__ import annotations

import json
import os
import sys

import commit_ref as cr

_HERE = os.path.dirname(os.path.abspath(__file__))
VECTORS = os.path.join(_HERE, "commit_kats.json")

ROUNDS = (6, 7)
P = cr.P

results: list[tuple[str, str, str]] = []      # (id, what, verdict)
vectors: dict = {}


def check(cid: str, what: str, ok: bool) -> None:
    results.append((cid, what, "PASS" if ok else "FAIL"))
    if not ok:
        print(f"  !! {cid} FAILED: {what}")


def hexw(w: list[int]) -> str:
    return "".join(f"{x:08x}" for x in w)


# --- fixtures ---------------------------------------------------------------

def base_row(m: int, seed: int) -> list[int]:
    return [(seed * 1000 + i * 7 + 1) % P for i in range(m)]


def ext_row(m: int, seed: int) -> list[list[int]]:
    return [[(seed * 1000 + i * 7 + c + 1) % P for c in range(3)]
            for i in range(m)]


# ===========================================================================
# C1 — the wide leaf over a BASE matrix, both round counts
# ===========================================================================
def c1() -> None:
    m = 5                                    # 2*5*1 = 10 felts -> 3 chunks (pad 2)
    ev, sym = base_row(m, 1), base_row(m, 2)
    for r in ROUNDS:
        d = cr.wide_leaf(ev, sym, cr.KIND_BASE, m, rounds=r)
        vectors[f"C1.base.m{m}.r{r}"] = hexw(d)
        check("C1", f"base m={m} r={r} digest is 4 u32 lanes",
              len(d) == 4 and all(0 <= x <= cr.MASK32 for x in d))
    # determinism
    a = cr.wide_leaf(ev, sym, cr.KIND_BASE, m)
    b = cr.wide_leaf(ev, sym, cr.KIND_BASE, m)
    check("C1", "deterministic", a == b)
    check("C1", "cost formula matches the chain length (10 felts / rate 4)",
          cr.wide_leaf_compressions(m, cr.KIND_BASE) == 3)


# ===========================================================================
# C2 — the wide leaf over an EXT3 matrix
# ===========================================================================
def c2() -> None:
    m = 3                                    # 2*3*3 = 18 felts -> 5 chunks (pad 2)
    ev, sym = ext_row(m, 3), ext_row(m, 4)
    for r in ROUNDS:
        d = cr.wide_leaf(ev, sym, cr.KIND_EXT3, m, rounds=r)
        vectors[f"C2.ext3.m{m}.r{r}"] = hexw(d)
        check("C2", f"ext3 m={m} r={r} digest well formed",
              len(d) == 4 and all(0 <= x <= cr.MASK32 for x in d))
    check("C2", "cost formula matches the chain length (18 felts / rate 4)",
          cr.wide_leaf_compressions(m, cr.KIND_EXT3) == 5)
    # base and ext3 over the SAME felt count must differ (the kind is bound)
    m_b = 9                                  # 2*9*1 = 18 felts, same as above
    d_base = cr.wide_leaf(base_row(m_b, 3), base_row(m_b, 4), cr.KIND_BASE, m_b)
    d_ext = cr.wide_leaf(ev, sym, cr.KIND_EXT3, m)
    check("C2", "same felt count, different kind -> different leaf",
          d_base != d_ext)


# ===========================================================================
# C3 — ★ WIDTH BINDING: the recorded live break, and the honest leg
# ===========================================================================
def c3() -> None:
    # The break (verifier.rs:633-639): a prover moves one column from the main
    # tree into the aux tree, choosing it after the LogUp challenges. Under the
    # keccak leaf both leaves still hash the bytes they were given and nothing
    # in the leaf noticed. Here the width is IN the preimage.
    m = 6
    ev, sym = base_row(m, 5), base_row(m, 6)

    honest = cr.wide_leaf(ev, sym, cr.KIND_BASE, m)
    check("C3", "HONEST leg: the true width still verifies",
          honest == cr.wide_leaf(ev, sym, cr.KIND_BASE, m))

    # A verifier that built the header from the AIR (m) while the prover shipped
    # a different width gets a different leaf -> authentication fails.
    shrunk = cr.wide_leaf(ev[:m - 1], sym[:m - 1], cr.KIND_BASE, m - 1)
    check("C3", "a narrower opening yields a different leaf", shrunk != honest)

    # ★ The decisive one, and it is the main<->aux confusion in miniature:
    # 6 BASE columns and 2 EXT3 columns both serialize to the SAME 12 felts.
    # Under the keccak leaf those two openings are byte-identical preimages, so
    # one leaf hash authenticates both — exactly the shape that let a prover
    # move columns between the main (base) and aux (ext) trees. The header
    # separates them because it binds the KIND as well as the width.
    a, b, c, d, e, f = ev
    g, h, i, j, k, l = sym
    same_felts_base = cr.wide_leaf(ev, sym, cr.KIND_BASE, 6)
    same_felts_ext = cr.wide_leaf([[a, b, c], [d, e, f]],
                                  [[g, h, i], [j, k, l]], cr.KIND_EXT3, 2)
    check("C3", "★ identical felt stream, base vs ext3 -> different leaf "
                "(the main<->aux confusion, closed by the header)",
          same_felts_base != same_felts_ext)

    # The reference REFUSES to derive the width from the data.
    try:
        cr.wide_leaf(ev, sym, cr.KIND_BASE, m + 1)
        check("C3", "a width disagreeing with the data is refused", False)
    except AssertionError:
        check("C3", "a width disagreeing with the data is refused", True)


# ===========================================================================
# C4 — padding is unambiguous BECAUSE the header binds the count
# ===========================================================================
def c4() -> None:
    # m=1 base: 2 felts, padded with 2 zeros. m=2 base: 4 felts, no padding.
    # If the padded stream of m=1 equalled the stream of m=2 with two zero
    # columns, only the header would separate them. Construct exactly that.
    ev1, sym1 = [7], [9]                                  # -> [7, 9, 0, 0]
    ev2, sym2 = [7, 9], [0, 0]                            # -> [7, 9, 0, 0]
    d1 = cr.wide_leaf(ev1, sym1, cr.KIND_BASE, 1)
    d2 = cr.wide_leaf(ev2, sym2, cr.KIND_BASE, 2)
    check("C4", "★ colliding padded felt streams separated by the header",
          d1 != d2)
    vectors["C4.pad.m1"] = hexw(d1)
    vectors["C4.pad.m2"] = hexw(d2)

    # honest leg: padding is stable, not random
    check("C4", "HONEST leg: padded leaf is deterministic",
          d1 == cr.wide_leaf(ev1, sym1, cr.KIND_BASE, 1))

    # a zero-width matrix is still a well-defined (header-only) leaf
    d0 = cr.wide_leaf([], [], cr.KIND_BASE, 0)
    check("C4", "zero-width leaf is the bare header",
          d0 == cr.leaf_header(0, cr.KIND_BASE))


# ===========================================================================
# C5 — the byte -> cell absorb encoding
# ===========================================================================
def c5() -> None:
    # O1: every lane is exactly four bytes, hence < 2^32, with no gate.
    blob = bytes(range(37))
    cells = cr.bytes_to_cells(blob)
    check("C5", "every lane is a u32 (O1 automatic)",
          all(0 <= x <= cr.MASK32 for c in cells for x in c))
    check("C5", "cell count is header + ceil(len/16)",
          len(cells) == 1 + 3 and cr.absorb_bytes_compressions(37) == 4)

    # ★ injectivity under zero-padding — the reason for the length prefix.
    check("C5", "★ b'\\x01' and b'\\x01\\x00' encode differently",
          cr.bytes_to_cells(b"\x01") != cr.bytes_to_cells(b"\x01\x00"))
    check("C5", "empty string is header-only", len(cr.bytes_to_cells(b"")) == 1)

    # a pinned end-to-end vector through the B1 chain
    import transcript_ref as tr
    for r in ROUNDS:
        t = tr.Transcript(rounds=r)
        cr.absorb_bytes(t, b"LAMBDAVM_LFM_STATEMENT_V1")
        vectors[f"C5.absorb.r{r}"] = hexw(t.state)
    check("C5", "absorb advances the chain", True)

    # HONEST leg: the encoding round-trips the bytes it claims to carry
    recovered = b""
    for c in cells[1:]:
        for lane in c:
            recovered += int(lane).to_bytes(4, "little")
    check("C5", "HONEST leg: body bytes recover the input under its length",
          recovered[:len(blob)] == blob)


# ===========================================================================
# C6 — node embedding and STRICT decode (S2 malleability)
# ===========================================================================
def c6() -> None:
    word = [0x01234567, 0x89abcdef, 0x00000000, 0xffffffff]
    packed = cr.pack_digest(word)
    check("C6", "pack is 32 bytes", len(packed) == 32)
    check("C6", "HONEST leg: pack -> strict_unpack round-trips",
          cr.strict_unpack_digest(packed) == word)
    check("C6", "the sixteen padding bytes are zero",
          all(packed[8 * i + 4:8 * i + 8] == b"\x00" * 4 for i in range(4)))
    vectors["C6.pack"] = packed.hex()

    # ★ every non-canonical flavour must REJECT, not reduce.
    def rejects(b: bytes, label: str) -> None:
        try:
            cr.strict_unpack_digest(b)
            check("C6", f"★ rejects {label}", False)
        except ValueError:
            check("C6", f"★ rejects {label}", True)

    ba = bytearray(packed); ba[4] = 0x01
    rejects(bytes(ba), "a set high byte in lane 0 (the cheap forgery)")
    ba = bytearray(packed); ba[8 * 3 + 7] = 0x80
    rejects(bytes(ba), "a set top byte in lane 3")
    # the lane p + 1, which unpack_digest would reduce to 1
    ba = bytearray(packed); ba[0:8] = (P + 1).to_bytes(8, "little")
    rejects(bytes(ba), "a lane congruent to 1 mod p")
    rejects(packed[:31], "a short commitment")

    # and the honest control that the fix is not "reject everything"
    check("C6", "HONEST leg: an all-zero digest still decodes",
          cr.strict_unpack_digest(b"\x00" * 32) == [0, 0, 0, 0])


# ===========================================================================
# C7 — tree arity and padding (S6)
# ===========================================================================
def c7() -> None:
    leaves = [[i, i + 1, i + 2, i + 3] for i in range(8)]
    for r in ROUNDS:
        root = cr.merkle_root(leaves, rounds=r)
        vectors[f"C7.root.n8.r{r}"] = hexw(root)
    check("C7", "HONEST leg: a power-of-two tree builds",
          len(cr.merkle_root(leaves)) == 4)
    check("C7", "a single leaf is its own root",
          cr.merkle_root([leaves[0]]) == leaves[0])
    try:
        cr.merkle_root(leaves[:7])
        check("C7", "★ a non-power-of-two leaf count is refused", False)
    except AssertionError:
        check("C7", "★ a non-power-of-two leaf count is refused", True)


# ===========================================================================
# C8 — the 96-bit question, costed
# ===========================================================================
def c8() -> None:
    import transcript_ref as tr
    t1 = tr.Transcript()
    t1.absorb([1, 2, 3, 4])
    before = t1.compressions
    e1 = cr.squeeze_ext_1(t1)
    cost1 = t1.compressions - before
    check("C8", "squeeze_ext_1 costs one compression", cost1 == 1)
    check("C8", "★ its coordinates are u32-bounded (96 bits total)",
          all(0 <= x <= cr.MASK32 for x in e1))

    t2 = tr.Transcript()
    t2.absorb([1, 2, 3, 4])
    before = t2.compressions
    e2 = cr.squeeze_ext_2(t2)
    cost2 = t2.compressions - before
    check("C8", "squeeze_ext_2 costs two compressions (+1 flat)", cost2 == 2)
    check("C8", "its coordinates span the full field",
          all(0 <= x < P for x in e2) and any(x > cr.MASK32 for x in e2))
    vectors["C8.ext1"] = [f"{x:08x}" for x in e1]
    vectors["C8.ext2"] = [f"{x:016x}" for x in e2]


# ===========================================================================
# C9 — the crate anchor survives: LFML rows are still plain blake3 at 7 rounds
# ===========================================================================
def c9() -> None:
    import leaf_ref as lr
    felts = [1, 2**32, P - 1, 0]
    word = lr.leaf_compress(felts, 7)
    byte = lr.leaf_compress_bytelevel(felts, 7)
    check("C9", "★ the wide leaf's LFML rows keep the byte-level anchor @7r",
          word == byte)
    # the fold is the ratified LFMC socket, unchanged
    import socket_ref as sk
    a, b = [1, 2, 3, 4], [5, 6, 7, 8]
    check("C9", "the fold is the honest LFMC socket",
          sk.socket_digest_wordlevel(a, b, sk.Framing(rounds=7))
          == sk.socket_digest(a, b, sk.Framing(rounds=7)))


# ===========================================================================
# C10 — the header is load-bearing (domain separation of the construction)
# ===========================================================================
def c10() -> None:
    import socket_ref as sk
    import leaf_ref as lr
    # A one-chunk wide leaf must NOT equal the bare LFML digest of those felts,
    # nor an LFMC of them: the header fold is what separates them.
    felts = [11, 22, 33, 44]
    wide = cr.wide_leaf([11, 22], [33, 44], cr.KIND_BASE, 2)
    bare = lr.leaf_compress(felts, 7)
    check("C10", "★ a wide leaf is not the bare LFML digest", wide != bare)
    check("C10", "★ a wide leaf is not an unheaded LFMC fold",
          wide != sk.socket_digest_wordlevel([0, 0, 0, 0], bare,
                                             sk.Framing(rounds=7)))
    # honest leg: at RATE = 4 these four felts are exactly one row, no padding.
    check("C10", "HONEST leg: it is exactly lfml_chain_row(header, felts)",
          wide == cr.lfml_chain_row(cr.leaf_header(2, cr.KIND_BASE), felts))


# ===========================================================================
# C11 — grinding under B1 (D3 ratified: grinding STAYS)
# ===========================================================================
def c11() -> None:
    import socket_ref as sk
    import leaf_ref as lr
    state = [0x11111111, 0x22222222, 0x33333333, 0x44444444]
    FACTOR = 12                                   # ~4096 trials, fast in python

    nonce = cr.find_nonce(state, FACTOR)
    check("C11", "HONEST leg: a mined nonce satisfies the difficulty",
          nonce is not None and cr.pow_is_valid(state, nonce, FACTOR))
    check("C11", "the difficulty actually bites (mining was not trivial)",
          nonce is not None and nonce > 0)
    for r in ROUNDS:
        vectors[f"C11.pow.r{r}"] = hexw(cr.pow_digest(state, 338, FACTOR, r))

    # ★ the difficulty is IN the preimage: a nonce mined at one factor is
    # worthless at another. Without `factor` in the operand a prover mines once
    # at factor 1 and presents the result at factor 20.
    check("C11", "★ the factor is bound into the digest",
          cr.pow_digest(state, 338, 12) != cr.pow_digest(state, 338, 13))

    # ★ the seed is bound: a nonce is not portable across transcript states.
    other = [0x11111111, 0x22222222, 0x33333333, 0x44444445]
    check("C11", "★ the transcript state is bound into the digest",
          cr.pow_digest(state, 338, 12) != cr.pow_digest(other, 338, 12))

    # ★ GRIND_MARK is load-bearing (defence in depth, per the docstring).
    import transcript_ref as tr
    unmarked = tr.compress_t(state, [338, 0, 0, 12], 7)
    check("C11", "★ GRIND_MARK changes the digest",
          cr.pow_digest(state, 338, 12) != unmarked)

    # ★★ THE IDENTITY THE FIXED-SEQUENCE ARGUMENT MUST CARRY.
    # A PoW step and an ABSORB of the operand cell are the SAME FUNCTION — both
    # are compress_T(state, cell). No KAT can separate them and none pretends to:
    # the separation is the program's compile-time operation sequence, exactly as
    # TRANSCRIPT.md §1.1 says for absorb-vs-squeeze. This leg asserts the identity
    # so the reliance is VISIBLE in the board rather than buried in prose — if a
    # future change gives the PoW its own tag (D6a), this leg flips and says so.
    op = cr.grind_operand(338, 12)
    t_absorb = tr.Transcript()
    t_absorb.state = list(state)
    t_absorb.absorb(op)
    check("C11", "★★ a PoW step IS an absorb of its operand cell — separation "
                 "rests on the fixed program sequence, NOT on the hash (D6a)",
          cr.pow_digest(state, 338, 12) == t_absorb.state)

    check("C11", "★ a PoW step is not an LFMC Merkle parent of the same cells",
          cr.pow_digest(state, 338, 12)
          != sk.socket_digest_wordlevel(state, op, sk.Framing(rounds=7)))
    check("C11", "★ a PoW step is not an LFML leaf of the same felts",
          cr.pow_digest(state, 338, 12) != lr.leaf_compress(op, 7))

    # the difficulty rule: lane 0 for factor <= 32, lane 1 above.
    w = cr.pow_digest(state, 338, 20)
    check("C11", "factor <= 32 reads lane 0 only",
          cr.pow_is_valid(state, 338, 20) == (w[0] % (1 << 20) == 0))
    check("C11", "factor > 32 requires lane 0 fully zero (so this sample fails)",
          w[0] != 0 and not cr.pow_is_valid(state, 338, 33))

    # range discipline, mirroring grinding.rs:22's 1..=64
    for bad, label in ((0, "factor 0"), (65, "factor 65")):
        try:
            cr.grind_operand(1, bad)
            check("C11", f"★ rejects {label}", False)
        except ValueError:
            check("C11", f"★ rejects {label}", True)
    try:
        cr.grind_operand(2**64, 20)
        check("C11", "★ rejects an out-of-range nonce", False)
    except ValueError:
        check("C11", "★ rejects an out-of-range nonce", True)

    check("C11", "verification is ONE compression, independent of difficulty",
          cr.pow_verify_compressions() == 1)


# ===========================================================================
# C12 — ★ THE LEAF RATE (the parameter that decides Gate D1)
# ===========================================================================
def c12() -> None:
    check("C12", "the spec parameter is 4 felts/row with a 4-lane accumulator",
          cr.LFML_FELTS_PER_ROW == 4 and cr.LFML_ACC_LANES == 4)
    # ★★ the binding constraint: a hash row reads whole CELLS of 4 felts
    # (instr.rs:99-110 + word.rs:15), so the rate MUST be a multiple of 4.
    check("C12", "★★ the rate is a multiple of 4 (whole machine cells)",
          cr.LFML_FELTS_PER_ROW % 4 == 0)
    check("C12", "★★ it fits the EXISTING 2-cells-in bus contract "
                 "(acc cell + one felt cell = 12 lanes + tag = 13 of 16 words)",
          cr.LFML_ACC_LANES + 2 * cr.LFML_FELTS_PER_ROW + 1 <= 16)

    # ★ one compression per row, not two: the fold is gone.
    m = 10                                   # 20 felts -> 5 rows exactly
    ev, sym = base_row(m, 11), base_row(m, 12)
    check("C12", "★ rate is 4 felts/compression (was 2)",
          cr.wide_leaf_compressions(m, cr.KIND_BASE) == 5
          and cr.wide_leaf_v0_compressions(m, cr.KIND_BASE) == 10)
    check("C12", "★ that is a 2.0x improvement on the dominant cost",
          cr.wide_leaf_v0_compressions(m, cr.KIND_BASE)
          / cr.wide_leaf_compressions(m, cr.KIND_BASE) == 2.0)

    # the widened row is ONE blake3 block: 16 acc + 40 felt halves + 4 tag = 60
    d = cr.lfml_chain_row([1, 2, 3, 4], [7, 8, 9, 10])
    check("C12", "★ a full row is 52 bytes — still one BLAKE3 block (<= 64)",
          len(d) == 4)
    for r in ROUNDS:
        vectors[f"C12.row.r{r}"] = hexw(cr.lfml_chain_row([1, 2, 3, 4],
                                                          [7, 8, 9, 10], r))

    # ★ the crate anchor survives the widening — this is the reason to prefer
    # the in-message accumulator over carrying it in the chaining value h.
    import blake3_oracle as ora
    msg = (b"".join(int(x).to_bytes(4, "little") for x in [1, 2, 3, 4])
           + b"".join(int(x).to_bytes(4, "little") for x in
                      [7, 0, 8, 0, 9, 0, 10, 0])
           + b"LFML")
    full = ora.hash_bytes(msg, 32, rounds=7)
    check("C12", "★ row == plain blake3::hash(52 bytes) @7r (anchor intact)",
          len(msg) == 52
          and cr.lfml_chain_row([1, 2, 3, 4], [7, 8, 9, 10], 7)
          == [int.from_bytes(full[4 * i:4 * i + 4], "little") for i in range(4)])

    # the header still binds shape at the new rate (C3/C4 properties survive)
    a, b, c, dd, e, f = base_row(6, 5)
    g, h, i, j, k, l = base_row(6, 6)
    check("C12", "★ base-vs-ext3 separation survives the rate change",
          cr.wide_leaf([a, b, c, dd, e, f], [g, h, i, j, k, l], cr.KIND_BASE, 6)
          != cr.wide_leaf([[a, b, c], [dd, e, f]], [[g, h, i], [j, k, l]],
                          cr.KIND_EXT3, 2))
    check("C12", "★ padding still separated by the header at rate 4",
          cr.wide_leaf([7], [9], cr.KIND_BASE, 1)
          != cr.wide_leaf([7, 9], [0, 0], cr.KIND_BASE, 2))

    # non-canonical felts are still REJECTED, not reduced
    try:
        cr.lfml_chain_row([1, 2, 3, 4], [cr.P, 1, 2, 3])
        check("C12", "★ a non-canonical felt still rejects", False)
    except ValueError:
        check("C12", "★ a non-canonical felt still rejects", True)

    # per-query tower cost at the real widths, old rate vs new.
    # LFM_HASH is 3444 MAIN columns, not cols::NUM_COLUMNS' 3457: the 13
    # preprocessed columns are committed in the precomputed tree, not the main
    # tree this census is scoped to (COMMIT.md §1.5 note (4)).
    widths = [3444, 1480, 792, 196, 25, 23, 21, 16, 14, 10, 7, 7, 6, 2]
    old = sum(cr.wide_leaf_v0_compressions(w, cr.KIND_BASE) for w in widths)
    new = sum(cr.wide_leaf_compressions(w, cr.KIND_BASE) for w in widths)
    check("C12", f"★ per-query main-tree leaf cost {old} -> {new} "
                 f"({old / new:.2f}x)", old > new)
    vectors["C12.per_query_old"] = old
    vectors["C12.per_query_new"] = new


def main() -> int:
    write = "--write" in sys.argv
    for fn in (c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12):
        fn()

    if write:
        with open(VECTORS, "w") as f:
            json.dump(vectors, f, indent=2, sort_keys=True)
        print(f"wrote {VECTORS} ({len(vectors)} vectors)")
    elif os.path.exists(VECTORS):
        with open(VECTORS) as f:
            pinned = json.load(f)
        for k, v in vectors.items():
            check("PIN", f"{k} matches the pinned vector", pinned.get(k) == v)
    else:
        print("no vector file yet — run with --write")

    width = max(len(w) for _, w, _ in results)
    by_id: dict[str, list[int]] = {}
    for cid, _, verdict in results:
        by_id.setdefault(cid, [0, 0])
        by_id[cid][0 if verdict == "PASS" else 1] += 1
    print()
    for cid, what, verdict in results:
        print(f"  {cid:<4} {what:<{width}}  {verdict}")
    print()
    npass = sum(1 for _, _, v in results if v == "PASS")
    print(f"BOARD: {' | '.join(f'{k} {v[0]}/{v[0]+v[1]}' for k, v in by_id.items())}")
    print(f"TOTAL: {npass}/{len(results)} PASS")
    return 0 if npass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
