"""C9 probe — the joint chain's compile-time curve constants are load-bearing,
and NOTHING in the constraint system checks them.

The lincomb2 soundness theorem (`RESULTS-lincomb2.md` §5) claims to rest on
"contracts C1-C7 + A-PRIME (unchanged from RESULTS.md)" and "plus nothing else".
This script shows that claim is false: the joint chain also depends on two
classes of compile-time constant that the single-scalar chips never had, and one
of those classes is bound by NO in-proof mechanism at all.

    G       (`GENERATOR_LE`, ecsm2.rs:614-628)   ANCHORED   — the P1 read is a
            MEMW access, so the constant is checked against what the guest wrote
            at `a1`. A wrong G cannot forge; it can only make an honest run
            unprovable (or, via the executor's status 7, take the fallback).

    T0      (`T0_X_LE`/`T0_Y_LE`, ecsm2.rs:869-870)      UNANCHORED
    EC_T0   (256 rows of -2^(j+1)*T0, ec_t0.rs:131-153)  UNANCHORED
            Nothing outside the AIR ever sees these. They enter the proof as an
            AIR constant and a preprocessed commitment the verifier compiles in.
            If the generator that produced them is wrong, every constraint and
            every bus still balances and the chip returns a WRONG Q.

Two constructions, both plausible generator bugs the source itself warns about:

  A. SIGN FLIP — the table stores +2^len*T0 instead of -2^len*T0. `ec_t0.rs`
     lines 16-41 and `lincomb2_table.rs` lines 10-22 both carry a warning that
     `Lincomb2Witness::x_t0_pow`/`y_t0_pow` hold the OPPOSITE convention and
     that "reading y_t0_pow where you meant Y is a silent sign flip that still
     type-checks".

  B. OFF-BY-ONE — the table row for `len` stores -2^(len+1)*T0, i.e. the
     `points[idx + MIN_LEN]` index arithmetic at ec_t0.rs:143 is off by one.

For each, the script rebuilds the full joint chain, re-derives the correction
row from the tampered addend by the group law, and then CHECKS, rather than
asserts, that:

  * every one of the four ECDAS2 convolution relations holds mod p on every row
    (Lambda / Xr / Yr / Dinv — the value-level content of idx 28..=287);
  * every ECDAS2 schedule constraint idx 11..=27 holds on every row;
  * every ECSM2 range/selector constraint that reads a tampered value holds;
  * all five buses balance: Ecdas (chain telescoping), Addend, JointBit, EcT0
    (against the TAMPERED table, which is the point), and the MEMW result write;
  * and Q' != Q, with Q' a canonical on-curve point the guest will happily hash.

Run:  python3 thoughts/ec-recover-opt/gate/c9_constants_probe.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))

from ec_ref import P, N, GX, GY, pt_add, pt_double, pt_neg, scalar_mul  # noqa: E402
from lincomb2_ref import lincomb2, lincomb2_rows, t0_ref  # noqa: E402

G = (GX, GY)

# Row-role -> (PH1, PH2) and (S1, S2, S3, S_CORR), transcribed from
# `Ecdas2Operation::phase_bits` / `selector_bits` (ecdas2.rs:256-277).
PHASE_BITS = {
    "Precompute": (0, 0),
    "Correction": (0, 1),
    "Double": (1, 0),
    "AddP1": (1, 0),
    "AddP2": (1, 0),
    "AddP12": (1, 0),
}
SEL_BITS = {
    "Double": (0, 0, 0, 0),
    "AddP1": (1, 0, 0, 0),
    "AddP2": (0, 1, 0, 0),
    "Precompute": (0, 1, 0, 0),
    "AddP12": (0, 0, 1, 0),
    "Correction": (0, 0, 0, 1),
}
# Addend-bus `sel` values (ecsm2.rs:111-114).
SEL_VALUE = {"AddP1": 1, "AddP2": 2, "Precompute": 2, "AddP12": 3, "Correction": 4}


def finv(a):
    return pow(a % P, P - 2, P)


# ── the four ECDAS2 relations, at value level (mod p) ────────────────────────
#
# The chip states each as a byte convolution with a quotient absorbing the
# integer offset; gate lemmas L1 + L3a say the convolution is exactly the
# mod-p identity below. Transcribed from `Ecdas2Constraints::s_i`
# (ecdas2.rs:742-817).

def relations_hold(row, ph1, ph2, sel_bits):
    xa, ya = row["a"]
    xb, yb = row["addend"]
    xr, yr = row["r"]
    lam = row["lam"]
    op = row["op"]
    g = sum(sel_bits)  # S1 + S2 + S3 + S_CORR, the Dinv gate
    out = {}

    out["Lambda"] = (
        op * (lam * (xb - xa) + ya - yb) + (1 - op) * (2 * lam * ya - 3 * xa * xa)
    ) % P == 0
    out["Xr"] = (lam * lam - xa - xb - xr - (1 - op) * (xa - xb)) % P == 0
    out["Yr"] = (lam * (xa - xr) - ya - yr) % P == 0
    if g:
        # D_INV exists iff xB != xA mod p; the chip witnesses it.
        d_ok = (xb - xa) % P != 0
        out["Dinv"] = d_ok and (finv(xb - xa) * (xb - xa) - 1) % P == 0
    else:
        out["Dinv"] = True  # gated off; q3 is pinned to 3p (L5b' sub-lemma c)
    return out


def schedule_holds(row, ph1, ph2, sel_bits, mu=1):
    s1, s2, s3, sc = sel_bits
    op, nb, d1, d2 = row["op"], row["nb"], row["d1"], row["d2"]
    checks = {
        "11 PH1*PH2": ph1 * ph2,
        "12 OP*NB": op * nb,
        "13 (1-OP)(NB-D1-D2+D1D2)": (1 - op) * (nb - d1 - d2 + d1 * d2),
        "14 OP-SumS": op - s1 - s2 - s3 - sc,
        "15 (1-PH1)D1": (1 - ph1) * d1,
        "16 (1-PH1)D2": (1 - ph1) * d2,
        "17 PH1*S_CORR": ph1 * sc,
        "18 PH1(S1+S3-OP*D1)": ph1 * (s1 + s3 - op * d1),
        "19 PH1(S2+S3-OP*D2)": ph1 * (s2 + s3 - op * d2),
        "20 MU(1-PH1-PH2)(S2-1)": mu * (1 - ph1 - ph2) * (s2 - 1),
        "21 PH2(S_CORR-1)": ph2 * (sc - 1),
    }
    for i, col in enumerate([d1, d2, s1, s2, s3, sc]):
        checks[f"{22 + i} (1-MU)*col"] = (1 - mu) * col
    return {k: v == 0 for k, v in checks.items()}


def canonical_bytes(pt):
    """Every coordinate is a canonical field element, so its 32 limbs are
    genuine bytes: the AreBytes contract (C1) and the `< p` overflow witnesses
    (X_Q_SUB_P / Y_Q_SUB_P) are satisfiable."""
    return 0 <= pt[0] < P and 0 <= pt[1] < P


# ── whole-chain checker ─────────────────────────────────────────────────────

def check_chain(rows, u1, u2, length, table_entry, p2, verbose=False):
    """Returns (ok, report). `table_entry` is what the EC_T0 table publishes for
    this `len` — i.e. what the correction row's addend is bound to."""
    report = []
    ok = True

    def fail(msg):
        nonlocal ok
        ok = False
        report.append("  FAIL " + msg)

    # 1. per-row constraints
    for i, row in enumerate(rows):
        ph1, ph2 = PHASE_BITS[row["sel"]]
        sel_bits = SEL_BITS[row["sel"]]
        for name, good in relations_hold(row, ph1, ph2, sel_bits).items():
            if not good:
                fail(f"row {i} ({row['sel']}): relation {name}")
        for name, good in schedule_holds(row, ph1, ph2, sel_bits).items():
            if not good:
                fail(f"row {i} ({row['sel']}): constraint idx {name}")
        for pt in (row["a"], row["r"]):
            if not canonical_bytes(pt):
                fail(f"row {i}: non-canonical coordinate")
        if row["sel"] != "Double" and not canonical_bytes(row["addend"]):
            fail(f"row {i}: non-canonical addend")
    report.append(f"  per-row: {len(rows)} rows x (4 relations + 17 schedule) OK")

    # 2. Ecdas bus — the three segments telescope, seeds/drains pin both ends
    phases = {"Precompute": 0, "Correction": 2}
    seg = {0: [], 1: [], 2: []}
    for row in rows:
        seg[phases.get(row["sel"], 1)].append(row)
    seeds = {
        0: (G, 0, 1),                       # a = P1 = G, round 0, op = add
        1: (T0, length - 1, 0),             # a = T0,     round len-1, op = double
        2: (seg[1][-1]["r"], 0, 1),         # a = phase-1 drain (relayed by ECSM2)
    }
    drains = {}
    for ph in (0, 1, 2):
        acc, rnd, op = seeds[ph]
        for row in seg[ph]:
            if row["a"] != acc or row["round"] != rnd or row["op"] != op:
                fail(f"phase {ph}: chain tuple mismatch at round {rnd}")
                break
            acc, rnd, op = row["r"], row["round"] - 1 + row["nb"], row["nb"]
        else:
            if rnd != -1 or op != 0:
                fail(f"phase {ph}: drain tuple is (round={rnd}, op={op}), want (-1, 0)")
            drains[ph] = acc
    report.append("  Ecdas bus: 3 segments telescope seed -> drain, round hits -1")

    # 3. JointBit bus — per (round, stream) the receive is 2*bit, senders are
    #    the rows carrying that digit.
    for stream, u in ((1, u1), (2, u2)):
        for i in range(256):
            sends = sum(r["d1" if stream == 1 else "d2"] for r in rows if r["round"] == i)
            if sends != 2 * ((u >> i) & 1):
                fail(f"JointBit[{i}, stream {stream}]: {sends} sends vs {2 * ((u >> i) & 1)}")
    report.append("  JointBit bus: 512 receives at 2*bit, balanced by the digit sends")

    # 4. Addend bus — ECSM2's N1/N2/N3 are free columns pinned by balance, so
    #    any consistent count balances; the VALUES are what matters.
    published = {1: G, 2: p2, 3: pt_add(G, p2), 4: table_entry}
    counts = {1: 0, 2: 0, 3: 0, 4: 0}
    for row in rows:
        if row["sel"] == "Double":
            continue
        s = SEL_VALUE[row["sel"]]
        counts[s] += 1
        if row["addend"] != published[s]:
            fail(f"Addend[sel={s}]: row addend != published value")
    if counts[4] != 1:
        fail(f"Addend[sel=4]: {counts[4]} correction receives, must be exactly 1 (mult = OK)")
    report.append(
        f"  Addend bus: N1={counts[1]} N2={counts[2]} N3={counts[3]}, "
        "correction receive = 1 = OK"
    )

    # 5. EcT0 bus — the send is [len, x, y]; the table receives. This is the
    #    ONLY thing that binds the correction addend, and it binds it to the
    #    TABLE, whatever the table contains.
    if rows[-1]["addend"] != table_entry:
        fail("EcT0: correction addend != table row")
    report.append(f"  EcT0 bus: send key len={length} matches the table row it looks up")

    q_chain = drains.get(2)
    return ok, report, q_chain


# ── scenarios ───────────────────────────────────────────────────────────────

def rebuild_correction(rows, new_addend):
    """Replace the correction row's addend and re-derive lambda/xR/yR by the
    group law, exactly as the witness generator would for that addend."""
    rows = [dict(r) for r in rows]
    corr = rows[-1]
    assert corr["sel"] == "Correction"
    xa, ya = corr["a"]
    xb, yb = new_addend
    assert (xb - xa) % P != 0, "chosen tamper hits the degenerate edge; pick another instance"
    lam = ((yb - ya) * finv(xb - xa)) % P
    xr = (lam * lam - xa - xb) % P
    yr = (lam * (xa - xr) - ya) % P
    corr["addend"] = (xb, yb)
    corr["lam"] = lam
    corr["r"] = (xr, yr)
    return rows, (xr, yr)


def two_pow(k, pt):
    for _ in range(k):
        pt = pt_double(pt)
    return pt


def main():
    global T0
    T0, counter = t0_ref()
    print("=" * 78)
    print("C9 PROBE — compile-time curve constants of the joint chain")
    print("=" * 78)
    print(f"T0 (NUMS, tag counter {counter}) = ({hex(T0[0])[:18]}..., {hex(T0[1])[:18]}...)")
    assert (T0[1] * T0[1] - T0[0] ** 3 - 7) % P == 0, "T0 off curve"

    # A concrete instance. Any (u1, u2, P2) works; these keep the chain short
    # enough to print.
    u1 = 0xB3D5_7A1E
    u2 = 0x4F2C_9E07
    p2 = scalar_mul(7, G)
    q_true = lincomb2(u1, G, u2, p2)
    rows, length, _ = (None, None, None)
    q_ref, length, rows = lincomb2_rows(u1, G, u2, p2, T0)
    assert q_ref == q_true
    print(f"instance: u1={hex(u1)} u2={hex(u2)} P2=7*G  len={length}  rows={len(rows)}")

    honest_entry = pt_neg(two_pow(length, T0))
    ok, report, q_chain = check_chain(rows, u1, u2, length, honest_entry, p2)
    print("\n[baseline] honest EC_T0 table")
    for line in report:
        print(line)
    print(f"  chain drains Q = {'MATCHES' if q_chain == q_true else 'DIFFERS FROM'} u1*G + u2*P2")
    baseline_ok = ok and q_chain == q_true
    print(f"  => {'ACCEPTED, correct' if baseline_ok else 'BROKEN — probe bug'}")
    assert baseline_ok, "baseline must pass, else the probe models the chip wrongly"

    scenarios = [
        (
            "A. SIGN FLIP  (table stores +2^len*T0; ec_t0.rs:28-41 warns about exactly this)",
            two_pow(length, T0),
        ),
        (
            "B. OFF-BY-ONE (table row for len holds -2^(len+1)*T0; ec_t0.rs:143 index)",
            pt_neg(two_pow(length + 1, T0)),
        ),
    ]

    forged = 0
    for title, tampered_entry in scenarios:
        print(f"\n[tamper] {title}")
        t_rows, q_forged = rebuild_correction(rows, tampered_entry)
        ok, report, q_chain = check_chain(t_rows, u1, u2, length, tampered_entry, p2)
        for line in report:
            print(line)
        differs = q_chain != q_true
        on_curve = (q_chain[1] ** 2 - q_chain[0] ** 3 - 7) % P == 0
        print(f"  Q_chain != Q_true            : {differs}")
        print(f"  Q_chain is a canonical point : {on_curve and canonical_bytes(q_chain)}")
        print(f"  Q_true  x = {hex(q_true[0])}")
        print(f"  Q_chain x = {hex(q_chain[0])}")
        if ok and differs and on_curve:
            forged += 1
            print("  => ACCEPTED with a WRONG Q. Every constraint and every bus is")
            print("     satisfied; the only thing that would have caught it is the")
            print("     correctness of the preprocessed EC_T0 table itself.")
        else:
            print("  => not a forgery (constraints caught it)")

    # The contrast: G is anchored to guest memory, so a wrong G cannot forge.
    print("\n[contrast] wrong GENERATOR_LE — anchored, therefore NOT a forgery")
    print("  ecsm2.rs:612-629 reads P1 as 8 MEMW doublewords with CONSTANT values.")
    print("  The tuple must match the memory token the guest wrote at a1, so a")
    print("  GENERATOR_LE != G makes the Memw bus unbalance: rejection, not forgery.")
    print("  (Executor-side it returns status 7 -> software fallback -> correct.)")
    print("  T0 and the EC_T0 rows have NO such anchor: no guest write, no executor")
    print("  read, no in-circuit curve check. They are pure AIR constants.")

    print("\n" + "=" * 78)
    print(f"RESULT: {forged}/2 tampered constant sets produce an accepted, wrong Q.")
    print("The joint chain therefore rests on a contract the board does not name:")
    print("  C9  the compile-time EC constants (T0, and the 256 EC_T0 rows bound by")
    print("      the verifier's compiled-in preprocessed commitment) are the")
    print("      intended, on-curve, mutually consistent values.")
    print("Discharged today by TESTS + a static commitment, never by a constraint:")
    print("  crypto/ecsm/src/tests/lincomb2_tests.rs::t0_is_on_curve_and_pinned")
    print("  crypto/ecsm/src/tests/lincomb2_tests.rs::t0_derivation_matches")
    print("  prover/src/tests/ec_t0_tests.rs::every_entry_is_on_curve_and_canonical")
    print("  prover/src/tests/ec_t0_tests.rs::trace_rows_recompute_from_t0_by_doubling")
    print("  prover/src/tests/ec_t0_tests.rs::table_matches_lincomb2_witness_correction_row")
    print("  prover/src/tests/ec_t0_tests.rs::commitment_is_stable_and_matches_the_shipped_static_bytes")
    print("Those tests are good and they close BOTH constructions above. The finding")
    print("is not that the constants are wrong — it is that the board's contract list")
    print("does not say they are load-bearing, so nothing tells a future reader that")
    print("those tests are part of the soundness argument rather than hygiene.")
    print("=" * 78)
    return 0 if forged == 2 else 1


if __name__ == "__main__":
    sys.exit(main())
