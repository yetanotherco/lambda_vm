"""Phase-D0 oracle anchors for the lincomb2 precompile.

Three anchors, all differential against implementations that do NOT share a
code path with `lincomb2_ref.py`:

  L-A  >=500 random lincomb2 differentials, 3-way:
         `lincomb2_ref.lincomb2`  (affine, MSB-first, ec_ref.py lineage)
       vs `jacobian_ref.lincomb2` (Jacobian, LSB-first, inversion-free)
       vs `ecdsa` PyPI            (third lineage, independent package)
       plus, per case, the NUMS-blinded joint-chain row list must land on the
       same Q and every emitted row must re-satisfy its own group law.

  L-B  small-joint-scalar unrollings, u1, u2 in [1, 16] (the phase-E L7 anchor).
       Ground truth is FULLY ENUMERATED: u·P computed by u-1 repeated group
       additions, no scalar-multiplication algorithm involved at all. Row
       schedules are dumped to `lincomb2_small_vectors.json` for the gate.

  L-C  row-shape census over the L-A corpus (row counts per ecrecover), so the
       layout lock's mean/max are re-derived here rather than asserted.

Run:  <venv>/bin/python lincomb2_anchors.py      (needs the `ecdsa` package)
"""

import json
import random
import sys

import ec_ref
import jacobian_ref
import lincomb2_ref
from ec_ref import GX, GY, N, P, pt_add, pt_double, recover_even_y

from ecdsa import SECP256k1
from ecdsa.ellipticcurve import INFINITY, Point

rng = random.Random(20260724)

T0, T0_COUNTER = lincomb2_ref.t0_ref()

# Third lineage: the `ecdsa` package's own group law.
LIB_CURVE = SECP256k1.curve
LIB_G = SECP256k1.generator


def lib_lincomb2(u1, pt1, u2, pt2):
    """Q = u1*pt1 + u2*pt2 using the `ecdsa` package's Point arithmetic."""
    a = Point(LIB_CURVE, pt1[0], pt1[1]) * u1
    b = Point(LIB_CURVE, pt2[0], pt2[1]) * u2
    q = a + b
    if q == INFINITY:
        return None
    return (q.x(), q.y())


def random_point():
    """A uniformly random on-curve point, both parities exercised."""
    while True:
        x = rng.randrange(P)
        y = recover_even_y(x)
        if y is not None:
            break
    if rng.random() < 0.5:
        y = (P - y) % P
    return (x, y)


def check_rows(rows, P1, P2, T0_pt, u1, u2, length):
    """Re-verify every emitted row from first principles: the slope, the group
    law, on-curveness of the result, and the schedule's bit bookkeeping.
    Returns a list of failure strings (empty = clean)."""
    bad = []
    on_curve = lambda pt: (pt[1] * pt[1] - pt[0] ** 3 - 7) % P == 0

    seen_digits = {}
    acc = None
    for i, r in enumerate(rows):
        a, g, res, lam, op = r["a"], r["addend"], r["r"], r["lam"], r["op"]
        # slope + group law, recomputed independently of the emitter
        if op == 0:
            want_lam = (3 * a[0] * a[0] * pow((2 * a[1]) % P, P - 2, P)) % P
            want = pt_double(a)
        else:
            if a[0] == g[0]:
                bad.append(f"row {i} ({r['sel']}): degenerate add, xa == xg")
                continue
            want_lam = ((g[1] - a[1]) * pow((g[0] - a[0]) % P, P - 2, P)) % P
            want = pt_add(a, g)
        if lam != want_lam:
            bad.append(f"row {i} ({r['sel']}): lambda mismatch")
        if res != want:
            bad.append(f"row {i} ({r['sel']}): result mismatch")
        if not on_curve(res):
            bad.append(f"row {i} ({r['sel']}): result off curve")

        # telescoping: a == previous r, EXCEPT the precompute row (standalone
        # chord P1 + P2, off the accumulator line) and the first double (seeded
        # at T0). This is the layout-lock's flagged special case.
        if r["sel"] == lincomb2_ref.SEL_PRECOMPUTE:
            if a != P1 or g != P2:
                bad.append(f"row {i}: precompute operands are not (P1, P2)")
        else:
            expect_a = T0_pt if acc is None else acc
            if a != expect_a:
                bad.append(f"row {i} ({r['sel']}): accumulator not telescoped")
            acc = res
        if r["sel"] == lincomb2_ref.SEL_PRECOMPUTE:
            continue

        if r["sel"] == lincomb2_ref.SEL_DOUBLE:
            if g != (0, 0):
                bad.append(f"row {i}: double row carries a nonzero addend")
            # the digit bits and the round-successor flag ride the double row
            want = ((u1 >> r["round"]) & 1, (u2 >> r["round"]) & 1)
            if (r["d1"], r["d2"]) != want:
                bad.append(f"row {i}: double row digits {(r['d1'], r['d2'])} != {want}")
            if r["nb"] != (want[0] | want[1]):
                bad.append(f"row {i}: nb != d1|d2")
        elif r["nb"] != 0:
            bad.append(f"row {i} ({r['sel']}): nb set on a non-double row")
        if r["op"] == 1 and r["sel"] != lincomb2_ref.SEL_CORRECTION:
            seen_digits[r["round"]] = (r["d1"], r["d2"])

    # every nonzero joint digit below `length` consumed exactly once, and no
    # add row invented for a zero digit
    for rr in range(length):
        d = ((u1 >> rr) & 1, (u2 >> rr) & 1)
        if d == (0, 0):
            if rr in seen_digits:
                bad.append(f"round {rr}: add row for a zero joint digit")
        else:
            if seen_digits.get(rr) != d:
                bad.append(f"round {rr}: joint digit {d} not consumed")
    n_dbl = sum(1 for r in rows if r["sel"] == lincomb2_ref.SEL_DOUBLE)
    if n_dbl != length:
        bad.append(f"double count {n_dbl} != len {length}")
    return bad


# ── L-A: >=500 random lincomb differentials ─────────────────────────────────

def anchor_a(count=600):
    fails = 0
    skipped = 0
    row_counts = []
    dbl_counts = []
    add_counts = []
    cases = []

    edges = [
        (1, 1), (1, 2), (2, 1), (3, 5),
        (N - 1, N - 1), (N - 1, 1), (1, N - 1),
        (2**255, 2**255 - 1), (2**255 - 1, 2**255),
        (2**128, 2**128 + 1), ((N - 1) // 2, (N + 1) // 2),
    ]
    for i in range(count + len(edges)):
        if i < len(edges):
            u1, u2 = edges[i]
        else:
            u1 = rng.randrange(1, N)
            u2 = rng.randrange(1, N)
        P1 = (GX, GY) if i % 3 == 0 else random_point()
        P2 = random_point()
        if P1[0] == P2[0]:
            skipped += 1
            continue

        q_ref = lincomb2_ref.lincomb2(u1, P1, u2, P2)
        q_jac = jacobian_ref.lincomb2(u1, P1, u2, P2)
        q_lib = lib_lincomb2(u1, P1, u2, P2)
        if q_ref is None or q_jac is None or q_lib is None:
            # Q = infinity: all three must agree that it is degenerate.
            if not (q_ref is None and q_jac is None and q_lib is None):
                fails += 1
                print(f"FAIL L-A[{i}] infinity disagreement: "
                      f"ref={q_ref} jac={q_jac} lib={q_lib}")
            skipped += 1
            continue

        if q_ref != q_jac:
            fails += 1
            print(f"FAIL L-A[{i}] ref != jacobian: u1={u1:x} u2={u2:x} "
                  f"P1={P1[0]:x} P2={P2[0]:x}")
            continue
        if q_ref != q_lib:
            fails += 1
            print(f"FAIL L-A[{i}] ref != ecdsa-lib: u1={u1:x} u2={u2:x} "
                  f"P1={P1[0]:x} P2={P2[0]:x}")
            continue

        # the blinded joint chain must land on the same Q, row by row
        try:
            q_chain, length, rows = lincomb2_ref.lincomb2_rows(u1, P1, u2, P2, T0)
        except ValueError as e:
            fails += 1
            print(f"FAIL L-A[{i}] blinded chain rejected ({e}) on a case the "
                  f"references computed: u1={u1:x} u2={u2:x}")
            continue
        if q_chain != q_ref:
            fails += 1
            print(f"FAIL L-A[{i}] blinded chain Q != reference Q: u1={u1:x} u2={u2:x}")
            continue
        bad = check_rows(rows, P1, P2, T0, u1, u2, length)
        if bad:
            fails += 1
            print(f"FAIL L-A[{i}] row check: {bad[:4]}")
            continue

        row_counts.append(len(rows))
        dbl_counts.append(sum(1 for r in rows if r["sel"] == lincomb2_ref.SEL_DOUBLE))
        add_counts.append(sum(1 for r in rows
                              if r["sel"] in (lincomb2_ref.SEL_ADD_P1,
                                              lincomb2_ref.SEL_ADD_P2,
                                              lincomb2_ref.SEL_ADD_P12)))
        cases.append((u1, u2, P1, P2, q_ref))

    n = len(row_counts)
    print(f"ANCHOR L-A (3-way lincomb2 differential + blinded-chain row check): "
          f"{'PASS' if fails == 0 else 'FAIL'} "
          f"({n} cases compared 3 ways, {skipped} degenerate/skipped, {fails} failures)")
    if n:
        print(f"  row census: doubles mean {sum(dbl_counts)/n:.1f} max {max(dbl_counts)}   "
              f"adds mean {sum(add_counts)/n:.1f} max {max(add_counts)}")
        print(f"  total rows: mean {sum(row_counts)/n:.1f}   max {max(row_counts)}   "
              f"min {min(row_counts)}")
    return fails, cases


# ── L-B: small-joint-scalar unrollings, u1, u2 in [1, 16] ───────────────────

def enumerated_mul(u, pt):
    """u*pt by u-1 REPEATED GROUP ADDITIONS. No double-and-add, no windowing,
    no recursion: the fully-unrolled definition of scalar multiplication.
    Returns affine or None (infinity)."""
    assert u >= 1
    acc = pt
    for _ in range(u - 1):
        if acc is None:
            return None
        if acc[0] == pt[0]:
            if acc[1] == pt[1]:
                acc = pt_double(acc)
            else:
                acc = None  # acc = -pt -> infinity
        else:
            acc = pt_add(acc, pt)
    return acc


def enumerated_lincomb2(u1, pt1, u2, pt2):
    a = enumerated_mul(u1, pt1)
    b = enumerated_mul(u2, pt2)
    if a is None:
        return b
    if b is None:
        return a
    if a[0] == b[0]:
        return pt_double(a) if a[1] == b[1] else None
    return pt_add(a, b)


def anchor_b(dump_path="lincomb2_small_vectors.json"):
    """u1, u2 in [1, 16] x several point pairs, exhaustively."""
    fails = 0
    checked = 0
    vectors = []

    G = (GX, GY)
    # Point pairs: (label, P1, P2). The (G, 3G) pair carries an extra algebraic
    # cross-check -- u1*G + u2*(3G) must equal (u1 + 3*u2)*G.
    three_g = ec_ref.scalar_mul(3, G)
    pairs = [("G,3G", G, three_g)]
    for j in range(3):
        pairs.append((f"G,R{j}", G, random_point()))
    for j in range(2):
        pairs.append((f"R{j},R{j}'", random_point(), random_point()))

    for label, P1, P2 in pairs:
        assert P1[0] != P2[0]
        for u1 in range(1, 17):
            for u2 in range(1, 17):
                truth = enumerated_lincomb2(u1, P1, u2, P2)
                q_ref = lincomb2_ref.lincomb2(u1, P1, u2, P2)
                q_jac = jacobian_ref.lincomb2(u1, P1, u2, P2)
                q_lib = lib_lincomb2(u1, P1, u2, P2)
                checked += 1
                if not (truth == q_ref == q_jac == q_lib):
                    fails += 1
                    print(f"FAIL L-B[{label} u1={u1} u2={u2}]: "
                          f"enumerated={truth} ref={q_ref} jac={q_jac} lib={q_lib}")
                    continue
                if label == "G,3G":
                    want = ec_ref.scalar_mul((u1 + 3 * u2) % N, G)
                    if want != truth:
                        fails += 1
                        print(f"FAIL L-B[{label} u1={u1} u2={u2}]: "
                              f"(u1+3u2)*G cross-check disagrees")
                        continue

                try:
                    q_chain, length, rows = lincomb2_ref.lincomb2_rows(u1, P1, u2, P2, T0)
                except ValueError as e:
                    fails += 1
                    print(f"FAIL L-B[{label} u1={u1} u2={u2}]: chain rejected ({e})")
                    continue
                if q_chain != truth:
                    fails += 1
                    print(f"FAIL L-B[{label} u1={u1} u2={u2}]: chain Q != enumerated Q")
                    continue
                bad = check_rows(rows, P1, P2, T0, u1, u2, length)
                if bad:
                    fails += 1
                    print(f"FAIL L-B[{label} u1={u1} u2={u2}] row check: {bad[:4]}")
                    continue

                rec = {
                    # P1/P2 live once in the `pairs` header, keyed by `pair`.
                    "pair": label,
                    "u1": u1, "u2": u2,
                    "len": length,
                    "rows": len(rows),
                    "Q": [f"{truth[0]:064x}", f"{truth[1]:064x}"],
                    # compact schedule shape: the row-role sequence, which is
                    # what the L6/L7 counting arguments are about.
                    "shape": ",".join(r["sel"] for r in rows),
                }
                # Full per-row values for a pinned subset only (keeps the file
                # in the same size class as `vectors.json`); everything else is
                # regenerable by re-running this script.
                if label == "G,3G" and u1 <= 8 and u2 <= 8:
                    rec["schedule"] = [
                        {"sel": r["sel"], "round": r["round"], "op": r["op"],
                         "d1": r["d1"], "d2": r["d2"],
                         "a": [f"{r['a'][0]:064x}", f"{r['a'][1]:064x}"],
                         "addend": [f"{r['addend'][0]:064x}", f"{r['addend'][1]:064x}"],
                         "lam": f"{r['lam']:064x}",
                         "r": [f"{r['r'][0]:064x}", f"{r['r'][1]:064x}"]}
                        for r in rows
                    ]
                vectors.append(rec)

    detailed = sum(1 for v in vectors if "schedule" in v)
    with open(dump_path, "w") as fh:
        json.dump({
            "description": "lincomb2 small-joint-scalar unrollings (phase-E L7 anchor). "
                           "Ground truth by repeated group addition; row schedules mirror "
                           "ecsm::lincomb2_witness. `shape` is the row-role sequence for "
                           "every case; `schedule` (full per-row values) is present on the "
                           "pinned G,3G u1,u2<=8 subset.",
            "T0": {"x": f"{T0[0]:064x}", "y": f"{T0[1]:064x}", "counter": T0_COUNTER},
            "pairs": {
                label: {"P1": [f"{a[0]:064x}", f"{a[1]:064x}"],
                        "P2": [f"{b[0]:064x}", f"{b[1]:064x}"]}
                for label, a, b in pairs
            },
            "count": len(vectors),
            "detailed": detailed,
            "vectors": vectors,
        }, fh, separators=(",", ":"))

    print(f"ANCHOR L-B (small joint scalars u1,u2 in [1,16], {len(pairs)} point pairs): "
          f"{'PASS' if fails == 0 else 'FAIL'} "
          f"({checked} pairs enumerated, {fails} failures)")
    print(f"  wrote {len(vectors)} vectors ({detailed} with full row values) to {dump_path}")
    return fails


if __name__ == "__main__":
    print(f"T0 = ({T0[0]:#x}, {T0[1]:#x})  counter={T0_COUNTER}")
    fa, _ = anchor_a()
    fb = anchor_b()
    total = fa + fb
    print(f"PHASE D0 ORACLE: {'ALL GREEN' if total == 0 else f'{total} FAILURES'}")
    sys.exit(1 if total else 0)
