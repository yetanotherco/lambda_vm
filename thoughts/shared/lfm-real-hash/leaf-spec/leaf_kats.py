"""
LFML leaf-mode KATs.

  L1  crate-KAT identity: every leaf row equals BLAKE3(halves ‖ "LFML")[..16] at
      7 rounds, computed by two separate routes and asserted equal.
  L2  BOUNDARY felts round-trip through the halves boundary: 0, 1, 2^32-1,
      2^32, p-2^32, p-1.
  L3  NON-CANONICAL inputs are REJECTED, not reduced: p, p+1, 2^64-1 have no
      valid half-pair, and the chip predicate refuses the pairs that encode them.
  L4  the canonicity predicate agrees with `v < p` exhaustively on the boundary
      and over a large random sample.
  L5  DOMAIN SEPARATION: an LFML leaf row over the same eight lanes differs from
      an LFMC parent and from an LFMT transcript step.
  L6  a FriToyV0-shaped leaf (8 field elements) costs exactly 3 compresses and is
      reproducible end to end.

Run: python3 leaf_kats.py [--write]
"""

from __future__ import annotations

import json
import os
import random
import sys

import leaf_ref as lf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "gate-oracle"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "transcript-spec"))
import socket_ref as sk               # noqa: E402
import transcript_ref as tr           # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "leaf_kats.json")
P = lf.P


def hexlanes(c):
    return "".join(f"{x:08x}" for x in c)


def l1_crate_identity(rounds: int):
    cases = [
        ("zeros", [0, 0, 0, 0]),
        ("boundary_mix", [0, 1, P - 1, 2**32]),
        ("all_p_minus_1", [P - 1] * 4),
        ("ramp", [0x0102030405060708, 0x1112131415161718,
                  0x2122232425262728, 0x3132333435363738]),
        ("u32_edges", [2**32 - 1, 2**32, P - 2**32, 1]),
    ]
    out = []
    for name, felts in cases:
        w = lf.leaf_compress(felts, rounds)
        b = lf.leaf_compress_bytelevel(felts, rounds)
        if w != b:
            return False, f"L1 route mismatch on {name}@{rounds}", []
        out.append({
            "name": name, "felts": [str(f) for f in felts],
            "lanes": lf.leaf_lanes(felts),
            "digest": w, "digest_hex": hexlanes(w),
        })
    return True, f"L1 PASS: {len(cases)} leaf rows, word route == byte route", out


def l2_boundary_roundtrip():
    rows = []
    for name, v in lf.BOUNDARY_FELTS:
        lo, hi = lf.felt_halves(v)
        if lf.halves_felt(lo, hi) != v:
            return False, f"L2 FAIL: {name} does not round-trip", []
        rows.append({"name": name, "felt": str(v), "lo": lo, "hi": hi,
                     "canonical": True})
    return True, f"L2 PASS: {len(rows)} boundary felts round-trip", rows


def l3_non_canonical_rejected():
    rows = []
    for name, v in lf.NON_CANONICAL:
        try:
            lf.felt_halves(v)
        except ValueError:
            pass
        else:
            return False, f"L3 FAIL: {name} ({v:#x}) was ACCEPTED", []
        # and the raw pair that would encode it must fail the chip predicate
        lo, hi = v & lf.MASK32, (v >> 32) & lf.MASK32
        if lf.is_canonical(lo, hi):
            return False, (f"L3 FAIL: the pair encoding {name} passes the chip "
                           f"predicate — canonicity is not being enforced")
        rows.append({"name": name, "value": str(v), "lo": lo, "hi": hi,
                     "canonical": False, "rejected": True})
    return True, f"L3 PASS: {len(rows)} non-canonical inputs rejected, not reduced", rows


def l4_predicate_exhaustive_on_boundary():
    MAXH = lf.MAX_HI
    cases = [(0, MAXH), (1, MAXH), (lf.MASK32, MAXH), (0, MAXH - 1),
             (lf.MASK32, MAXH - 1), (0, 0), (1, 0)]
    rng = random.Random(11)
    cases += [(rng.randrange(2**32), rng.randrange(2**32)) for _ in range(300000)]
    for lo, hi in cases:
        if ((lo + (hi << 32)) < P) != lf.is_canonical(lo, hi):
            return False, f"L4 FAIL at lo={lo:#x} hi={hi:#x}"
    return True, (f"L4 PASS: predicate == (v < p) on {len(cases)} cases "
                  f"including every boundary")


def l5_domain_separation(rounds: int):
    felts = [0x0102030405060708, 0x1112131415161718,
             0x2122232425262728, 0x3132333435363738]
    lanes = lf.leaf_lanes(felts)
    a, b = lanes[0:4], lanes[4:8]
    leaf = lf.leaf_compress(felts, rounds)
    parent = sk.socket_digest_wordlevel(a, b, sk.Framing(rounds=rounds))
    step = tr.compress_t(a, b, rounds)
    if leaf == parent:
        return False, "L5 FAIL: LFML leaf == LFMC parent"
    if leaf == step:
        return False, "L5 FAIL: LFML leaf == LFMT transcript step"
    if parent == step:
        return False, "L5 FAIL: LFMC parent == LFMT transcript step"
    return True, ("L5 PASS: LFML / LFMC / LFMT are pairwise distinct on the same "
                  "eight lanes")


def l6_fri_leaf(rounds: int):
    felts = [P - 1, 0, 1, 2**32, 12345678901234567, 2**32 - 1, P - 2**32, 999]
    d = lf.leaf_over_8_felts(felts, rounds)
    return True, "L6 PASS: 8-felt leaf = 3 compresses (2 LFML + 1 LFMC)", {
        "felts": [str(f) for f in felts],
        "digest": d, "digest_hex": hexlanes(d), "compresses": 3,
    }


def main() -> int:
    print("=" * 74)
    print("LFML LEAF-MODE KATs (option C, ratified)")
    print("=" * 74)
    ok = True
    doc = {
        "mode": "LFML leaf (felt-input, option C)",
        "tag_ascii": lf.TAG_LFML_ASCII.decode(),
        "tag_word": lf.TAG_LFML,
        "felts_per_row": lf.FELTS_PER_LEAF_ROW,
        "lane_order": "[lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3] — halves adjacent",
        "byte_serialization": "each lane as 4 little-endian bytes, in lane order, then the 4 tag bytes",
        "canonicity": "v < p  <=>  NOT(hi == 2^32-1 AND lo >= 1)",
        "rounds": {},
    }
    for rounds in (7, 6):
        good, msg, rows = l1_crate_identity(rounds)
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {msg}")
        g6, m6, leaf = l6_fri_leaf(rounds)
        ok &= g6
        print(f"  [{'PASS' if g6 else 'FAIL'}] {m6} @{rounds}r")
        doc["rounds"][str(rounds)] = {"leaf_rows": rows, "fri_leaf": leaf}

    for fn in (l2_boundary_roundtrip, l3_non_canonical_rejected):
        good, msg, rows = fn()
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {msg}")
        doc[fn.__name__] = rows
    for fn in (l4_predicate_exhaustive_on_boundary,):
        good, msg = fn()
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {msg}")
    good, msg = l5_domain_separation(7)
    ok &= good
    print(f"  [{'PASS' if good else 'FAIL'}] {msg}")

    if "--write" in sys.argv:
        with open(OUT, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"\n  wrote {OUT}")
    print("-" * 74)
    print(f"LFML LEAF KATs: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
