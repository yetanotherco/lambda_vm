"""Positive control + L7 for the lincomb2 chips.

Two jobs in one pass over real witnesses:

  POSITIVE ANCHOR — every row of a real `ecsm::lincomb2_witness` (dumped by the
  oracle harness) must satisfy every transcribed ECDAS2 constraint mod p_g,
  every range contract, and every schedule/bookkeeping relation. This is the
  transcription-faithfulness anchor: a sign or index error in the model fails
  HERE, on honest data, before any UNSAT verdict is trusted.

  L7 — the chain's drained `Q` must equal FULLY ENUMERATED ground truth for
  small joint scalars (`u1, u2 ∈ [1,16]`, ground truth by repeated group
  addition), and must equal the independent references on random inputs.

Constraint enumeration mirrors `Ecdas2Constraints::eval` (idx 0..=216):
  0..=10   IS_BIT on MU, OP, NB, D1, D2, S1, S2, S3, S_CORR, PH1, PH2
  11..=21  the schedule constraints
  22..=216 3 × (64 ConvCarry + 1 ColIsZero)

`PH*` and `S*` are not columns of the witness struct — the chip derives them
from `JointSel` (`Ecdas2Operation::phase_bits` / `selector_bits`). That mapping
is reproduced here and is a MODELLED step, flagged as such in RESULTS.md.

Run:  <venv>/bin/python positive_real_witness2.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))

import lincomb2_ref
from ec_ref import GX, GY, N, P, pt_add, pt_double, recover_even_y, scalar_mul
from gate_common import OFF, PG, s_ecdas_lambda, s_ecdas_xr, s_ecdas_yr
from gate2_common import lincomb2_witness

checks = 0
failures = []


def ck(cond, what):
    global checks
    checks += 1
    if not cond:
        failures.append(what)


P_LIMBS = list(P.to_bytes(32, "little"))
R_LIMBS = list((3 * P).to_bytes(33, "little"))


def _rq_int(q_limbs, i, mu=1):
    """µ·Σ R_j·P_{i−j} − Σ q_j·P_{i−j}, the shared offset term."""
    rp = qp = 0
    for j in range(i + 1):
        pij = P_LIMBS[i - j] if i - j < 32 else 0
        rp += (R_LIMBS[j] if j < 33 else 0) * pij
        qp += (q_limbs[j] if j < len(q_limbs) else 0) * pij
    return mu * rp - qp


def le_bytes(int_hex, n=32):
    return list(int(int_hex, 16).to_bytes(n, "little"))


def hex_bytes(s):
    return list(bytes.fromhex(s))


# `Ecdas2Operation::phase_bits` / `selector_bits`, reproduced.
PHASE = {"Precompute": (0, 0), "Correction": (0, 1),
         "Double": (1, 0), "AddP1": (1, 0), "AddP2": (1, 0), "AddP12": (1, 0)}
SELECT = {"Double": (0, 0, 0, 0), "AddP1": (1, 0, 0, 0),
          "AddP2": (0, 1, 0, 0), "Precompute": (0, 1, 0, 0),
          "AddP12": (0, 0, 1, 0), "Correction": (0, 0, 0, 1)}


def check_row(row, tag):
    """All 217 ECDAS2 constraints on one live (MU = 1) row, plus the ranges."""
    sel = row["sel"]
    ph1, ph2 = PHASE[sel]
    s1, s2, s3, sc = SELECT[sel]
    op, nb, d1, d2 = row["op"], row["nb"], row["d1"], row["d2"]
    mu = 1

    # idx 0..=10 — IS_BIT
    for nm, val in (("MU", mu), ("OP", op), ("NB", nb), ("D1", d1), ("D2", d2),
                    ("S1", s1), ("S2", s2), ("S3", s3), ("S_CORR", sc),
                    ("PH1", ph1), ("PH2", ph2)):
        ck(val in (0, 1), f"{tag}: IS_BIT({nm}) = {val}")

    # idx 11..=21 — the schedule constraints, evaluated as integers (all are
    # products of small values, so mod p_g is the same as over Z here).
    ck(ph1 * ph2 == 0, f"{tag}: idx11 PH1·PH2")
    ck(op * nb == 0, f"{tag}: idx12 OP·NB")
    ck((1 - op) * (nb - d1 - d2 + d1 * d2) == 0, f"{tag}: idx13 NB = D1∨D2")
    ck(op - s1 - s2 - s3 - sc == 0, f"{tag}: idx14 OP = ΣS")
    ck((1 - ph1) * d1 == 0, f"{tag}: idx15 (1−PH1)·D1")
    ck((1 - ph1) * d2 == 0, f"{tag}: idx16 (1−PH1)·D2")
    ck(ph1 * sc == 0, f"{tag}: idx17 PH1·S_CORR")
    ck(ph1 * (s1 + s3 - op * d1) == 0, f"{tag}: idx18 addend vs u1 digit")
    ck(ph1 * (s2 + s3 - op * d2) == 0, f"{tag}: idx19 addend vs u2 digit")
    ck(mu * (1 - ph1 - ph2) * (s2 - 1) == 0, f"{tag}: idx20 precompute adds P2")
    ck(ph2 * (sc - 1) == 0, f"{tag}: idx21 correction adds −2^len·T₀")

    # Range contracts: bytes and the byte-checked ROUND.
    v = {
        "lam": le_bytes(row["lambda"]), "xa": le_bytes(row["x_a"]),
        "ya": le_bytes(row["y_a"]), "xg": le_bytes(row["x_b"]),
        "yg": le_bytes(row["y_b"]), "xr": le_bytes(row["x_r"]),
        "yr": le_bytes(row["y_r"]),
        "q0": hex_bytes(row["q0"]), "q1": hex_bytes(row["q1"]),
        "q2": hex_bytes(row["q2"]),
    }
    for nm in ("lam", "xa", "ya", "xg", "yg", "xr", "yr", "q0", "q1", "q2"):
        ck(all(0 <= b < 256 for b in v[nm]), f"{tag}: AreBytes({nm})")
    ck(0 <= row["round"] < 256, f"{tag}: AreBytes(ROUND)")

    # idx 22..=27 — (1−MU)·x for every column that is a bus multiplicity.
    for nm, val in (("D1", d1), ("D2", d2), ("S1", s1), ("S2", s2),
                    ("S3", s3), ("S_CORR", sc)):
        ck((1 - mu) * val == 0, f"{tag}: idx22-27 (1−MU)·{nm}")

    # idx 28..=287 — the four convolution relations.
    for relation, builder, cs, off in (
        ("lambda", s_ecdas_lambda, row["c0"], OFF["ecdas_lambda"]),
        ("xr", s_ecdas_xr, row["c1"], OFF["ecdas_xr"]),
        ("yr", s_ecdas_yr, row["c2"], OFF["ecdas_yr"]),
    ):
        for i in range(64):
            s_i = builder(v, i, op) if relation != "yr" else builder(v, i)
            c_i = cs[i]
            c_prev = cs[i - 1] if i else 0
            ck((256 * c_i - c_prev - s_i) % PG == 0,
               f"{tag}: ConvCarry({relation}, {i})")
        ck(cs[63] == 0, f"{tag}: ColIsZero(c_63, {relation})")
        for i in range(63):
            ck(0 <= cs[i] + off < (1 << 16),
               f"{tag}: IsHalfword(c_{i} + {off}, {relation})")

    # The Dinv block (idx 223..=287). `dinv_witness` now lives in
    # `crypto/ecsm/src/witness.rs`, so the harness dumps the PROVER'S OWN
    # columns: this is a genuine transcription check, not merely a completeness
    # check. The group-law derivation is kept alongside as a differential.
    #   gated:  g·(Σ d_j·(xB − xA)_{i−j} − [i=0]) + rq(q3),  g = ΣS
    g = s1 + s2 + s3 + sc
    ck(g == op, f"{tag}: D_INV gate ΣS equals OP")
    xb_v = int(row["x_b"], 16)
    xa_v = int(row["x_a"], 16)
    dl_p = hex_bytes(row["d_inv"])          # prover's columns
    q3b_p = hex_bytes(row["q3"])
    c3_p = row["c3"]
    ck(all(0 <= b < 256 for b in dl_p + q3b_p), f"{tag}: AreBytes(D_INV, Q3)")

    if g == 1:
        delta = (xb_v - xa_v) % P
        ck(delta != 0, f"{tag}: D_INV — the add is a genuine chord (xB ≢ xA)")
        # independent derivation, then a differential against the prover
        d_inv = pow(delta, P - 2, P)
        q3 = 3 * P + (d_inv * (xb_v - xa_v) - 1) // P
        ck((d_inv * (xb_v - xa_v) - 1) % P == 0, f"{tag}: D_INV numerator ÷ p")
        ck(0 <= q3 < (1 << 264), f"{tag}: D_INV quotient fits 33 bytes")
        ck(dl_p == list(d_inv.to_bytes(32, "little")),
           f"{tag}: D_INV column == independently derived inverse")
        ck(q3b_p == list(q3.to_bytes(33, "little")),
           f"{tag}: Q3 column == independently derived quotient")
    else:
        # gated off: only rq survives. It does NOT leave q3 free — telescoping
        # gives p·(µ·R − q3) = 0, so q3 is PINNED to 3p on a live doubling.
        ck(q3b_p == list((3 * P).to_bytes(33, "little")),
           f"{tag}: D_INV gated-off pins Q3 = 3p")
        ck(all(c == 0 for c in c3_p), f"{tag}: D_INV gated-off has zero carries")

    # the relation itself, on the prover's columns
    at = lambda a, m: a[m] if m < len(a) else 0
    for i in range(64):
        si = g * (sum(at(dl_p, j) * (at(v["xg"], i - j) - at(v["xa"], i - j))
                      for j in range(i + 1)) - (1 if i == 0 else 0))
        si += _rq_int(q3b_p, i)
        c_i = c3_p[i]
        c_prev = c3_p[i - 1] if i else 0
        ck((256 * c_i - c_prev - si) % PG == 0, f"{tag}: ConvCarry(dinv, {i})")
    ck(c3_p[63] == 0, f"{tag}: ColIsZero(c_63, dinv)")
    for i in range(63):
        ck(0 <= c3_p[i] + OFF["ecdas_xr"] < (1 << 16),
           f"{tag}: IsHalfword(c_{i} + {OFF['ecdas_xr']}, dinv)")

    return 288


def main():
    T0, _ = lincomb2_ref.t0_ref()
    G = (GX, GY)
    print("POSITIVE ANCHOR + L7 — real lincomb2 witnesses vs the transcribed model")
    print()

    # ── the corpus ───────────────────────────────────────────────────────────
    cases = []
    # L7: small joint scalars, ground truth by repeated group addition
    P2_small = scalar_mul(7, G)
    for u1 in range(1, 17):
        for u2 in range(1, 17):
            cases.append(("L7-small", u1, u2, G, P2_small))
    # random + edge, for the positive anchor's volume
    import random
    rng = random.Random(90210)
    for _ in range(6):
        while True:
            x = rng.randrange(P)
            y = recover_even_y(x)
            if y is not None:
                break
        cases.append(("random", rng.randrange(1, N), rng.randrange(1, N), G,
                      (x, y if rng.random() < 0.5 else (P - y) % P)))
    for u1, u2 in [(1, 1), (N - 1, N - 1), (2**255, 2**255 - 1)]:
        cases.append(("edge", u1, u2, G, P2_small))

    def enumerated_mul(u, pt):
        acc = pt
        for _ in range(u - 1):
            acc = pt_double(acc) if acc == pt else pt_add(acc, pt)
        return acc

    rows_seen = 0
    l7_checked = 0
    l7_fail = 0
    for kind, u1, u2, p1, p2 in cases:
        w = lincomb2_witness(u1, u2, p1, p2)
        q = (int(w["x_q"], 16), int(w["y_q"], 16))

        for idx, row in enumerate(w["rows"]):
            check_row(row, f"{kind} u1={u1:x} u2={u2:x} row{idx}")
            rows_seen += 1

        # ── L7 ───────────────────────────────────────────────────────────────
        if kind == "L7-small":
            a, b = enumerated_mul(u1, p1), enumerated_mul(u2, p2)
            truth = pt_double(a) if a == b else pt_add(a, b)
            l7_checked += 1
            if truth != q:
                l7_fail += 1
                failures.append(f"L7: enumerated ground truth != chip Q "
                                f"(u1={u1}, u2={u2})")
        else:
            ref = lincomb2_ref.lincomb2(u1, p1, u2, p2)
            l7_checked += 1
            if ref != q:
                l7_fail += 1
                failures.append(f"L7: reference != chip Q (u1={u1:x}, u2={u2:x})")

        # the drained Q must be canonical (the chip's X_Q_SUB_P / Y_Q_SUB_P)
        ck(q[0] < P and q[1] < P, f"{kind}: Q canonical")
        # and on the curve
        ck((q[1] * q[1] - q[0] ** 3 - 7) % P == 0, f"{kind}: Q on curve")

    print(f"   cases            : {len(cases)} "
          f"({sum(1 for c in cases if c[0] == 'L7-small')} L7-small, "
          f"{sum(1 for c in cases if c[0] == 'random')} random, "
          f"{sum(1 for c in cases if c[0] == 'edge')} edge)")
    print(f"   ECDAS2 rows      : {rows_seen:,}")
    print(f"   constraint/range checks : {checks:,}")
    print(f"   L7 Q comparisons : {l7_checked} ({l7_fail} mismatches)")
    print()
    if failures:
        print(f"   FAILURES: {len(failures)}")
        for f in failures[:20]:
            print(f"      {f}")
    else:
        print("   ALL CHECKS PASS — the transcription is faithful on honest data,")
        print("   and every drained Q matches enumerated / independent ground truth.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
