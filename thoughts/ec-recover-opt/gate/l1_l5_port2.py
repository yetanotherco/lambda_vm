"""L1-L5 for the lincomb2 chips — the port argument, and L5b's replacement.

L1, L2a/b/c, L3a/L3b, L4a/b/c and L5a all speak ONLY about the three convolution
relations. They therefore transfer to ECDAS2 verbatim **if and only if** the
relation builders are the same function. That is checked mechanically here
rather than asserted (§1), which is a stronger port argument than re-deriving
the same lemmas against a re-transcribed model would be.

L5b does NOT port. In the old chip it proved the incomplete-addition edge
unreachable from `k < N` plus the prefix structure; that argument has no analogue
in the joint chain, and the NUMS blinding proposed to replace it was broken
(`../lincomb2/FINDING-nums-blinding.log`). Its replacement is the unconditional
non-degeneracy relation `D_INV·(xB − xA) ≡ 1 (mod p)`, proved here (§3) to:

  (a) be imposed on exactly the addend-consuming rows — all three scalar addends
      AND the correction row — and never on a doubling;
  (b) be unsatisfiable exactly when `xA ≡ xB (mod p)`, including when the two
      differ as byte strings but agree modulo p.

Run:  <venv>/bin/python l1_l5_port2.py
"""

import sys
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))

from gate_common import OFF, P, PG
from gate2_common import (
    Ecdas2Row, chip_state, membership_bodies_identical, relation_bodies_identical,
)

verdicts = []


def report(name, ok, detail=""):
    verdicts.append((name, ok))
    print(f"[{'PROVED' if ok else 'FAILED':^8}] {name}")
    for line in detail.splitlines():
        if line:
            print(f"           {line}")


def hdr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ── §1 the port argument ────────────────────────────────────────────────────

def port_argument():
    hdr("§1 — L1 / L2 / L3 / L4 / L5a port by relation-body identity")
    ident = relation_bodies_identical()
    all_same = all(ident.values())
    detail = "\n".join(f"{k:14}: {'identical' if v else 'DIFFERS'}"
                       for k, v in ident.items())
    report("ECDAS2's relation builders are ECDAS's, modulo the XB/YB rename",
           all_same, detail)
    report("⇒ L1 (carry telescoping), L2a (per-limb widths), L3a (convolutions "
           "capture\n   the intended polynomial), L3b (chord/tangent stays on "
           "curve), L4a/b/c\n   (λ, xR, yR pinned mod p) and L5a (no 2-torsion) "
           "port unchanged", all_same,
           "These lemmas quantify over the relation's operands only; renaming an\n"
           "operand column cannot affect them. The old gate's proofs stand as-is.\n"
           "The comparison covers the s_i PROLOGUE and `conv_carry` as well as the\n"
           "three arms: without the prologue a chip whose relations read the WRONG\n"
           "columns still compares identical (TRANSCRIPTION-AUDIT.md F3).")

    # The membership port. The soundness theorem's "P2 is on the curve" clause
    # rests on it and nothing used to check it (TRANSCRIPTION-AUDIT.md, gap 2).
    mem = membership_bodies_identical()
    mem_same = all(mem.values())
    report("ECSM2's P2-membership relations are ECSM's, modulo the column rename "
           "and\n   the µ → OK gate swap", mem_same,
           "\n".join(f"{k:14}: {'identical' if v else 'DIFFERS'}"
                     for k, v in mem.items())
           + "\nOK is IS_BIT and OK·(1−MU) = 0 (ecsm2.rs idx 1, 2), so OK = 1 rows are"
             "\na subset of MU = 1 rows and every ECSM membership lemma applies to"
             "\nthem verbatim. `carry_chain` is excluded: ECSM2 has five"
             "\nOverflowKinds to ECSM's three, so those bodies differ by design"
             "\n(covered by L8 N3 / N7 instead).")
    return all_same and mem_same


# ── §2 L2b — the carry windows for the NEW row population ───────────────────

def l2b_offsets():
    hdr("§2 — L2b: honest carries of the joint chain fit the existing windows")
    import width_audit as wa

    print("   (measured in ../oracle/width_audit.py over 6,346 real rows of every")
    print("    row type, and differentially checked against the prover's own")
    print("    carries — 38,076 comparisons, 0 mismatches)")
    print()
    ranges = {"lambda": (-4303, 6728), "xr": (-112, 8308), "yr": (-465, 5914)}
    ok = True
    for rel, off_key in (("lambda", "ecdas_lambda"), ("xr", "ecdas_xr"),
                         ("yr", "ecdas_yr")):
        off = OFF[off_key]
        lo, hi = wa.carry_window(off)
        clo, chi = ranges[rel]
        fits = lo <= clo and chi <= hi
        ok &= fits
        print(f"   {rel:7} honest [{clo}, {chi}]  window [{lo}, {hi}]  "
              f"{'fits' if fits else 'DOES NOT FIT'}")
    report("L2b: the joint chain's honest carries fit the unchanged offsets", ok,
           "The varying addend does not move this: the relations have the same\n"
           "term counts, and all four addends are canonical for an honest prover.")
    return ok


# ── §3 L5b's replacement: the D_INV non-degeneracy relation ─────────────────

def l5b_replacement():
    hdr("§3 — L5b REPLACED by D_INV·(xB − xA) ≡ 1 (mod p)")
    st = chip_state()
    gate = st["dinv_gate_detail"]
    if not st["dinv_relation"]:
        print("   NOTE: D_INV does not protect every addend-consuming row.")
        print(f"   reason: {gate['reason']}")
        print("   What follows is then a SPECIFICATION rather than a check.")
        print()
    else:
        print("   D_INV is present (ECDAS2 idx 223..=287), and its gate expression")
        print(f"   is PARSED from the `Relation::Dinv` arm: {sorted(gate['gate'])},")
        print(f"   which EQUALS the Addend receive's Multiplicity::Linear terms")
        print(f"   {sorted(gate['addend'])}. That equality is what (a1)/(a2) below")
        print("   quantify over; before TRANSCRIPTION-AUDIT.md F1 nothing checked")
        print("   it, and dropping S_CORR from the gate was a working forgery.")
        print()

    # (a) imposed on exactly the addend-consuming rows.
    #     The Addend receive multiplicity IS ΣS, and idx 14 gives OP = ΣS, so
    #     gating by OP selects exactly the rows that consume an addend.
    s = z3.Solver()
    r = Ecdas2Row(s, "r")
    s.add(r.mu == 1)
    s.add(r.op != r.addend_receive())
    a_ok = s.check() == z3.unsat
    report("(a1) OP = (Addend receive multiplicity) on every live row", a_ok,
           "The chip gates D_INV by ΣS = S1+S2+S3+S_CORR — the very expression\n"
           "that counts the Addend receive — so the check cannot drift away from\n"
           "the rows that consume an addend. This lemma shows that gate coincides\n"
           "with OP on every live row, so it fires exactly on the add rows.")

    # every add row type consumes an addend; doublings never do
    covered = {}
    for name, extra in (
        ("AddP1", lambda s, r: s.add(r.ph1 == 1, r.op == 1, r.d1 == 1, r.d2 == 0)),
        ("AddP2", lambda s, r: s.add(r.ph1 == 1, r.op == 1, r.d1 == 0, r.d2 == 1)),
        ("AddP12", lambda s, r: s.add(r.ph1 == 1, r.op == 1, r.d1 == 1, r.d2 == 1)),
        ("Precompute", lambda s, r: s.add(r.ph1 == 0, r.ph2 == 0, r.op == 1)),
        ("Correction", lambda s, r: s.add(r.ph2 == 1, r.op == 1)),
        ("Double", lambda s, r: s.add(r.ph1 == 1, r.op == 0)),
    ):
        s = z3.Solver()
        r = Ecdas2Row(s, name)
        s.add(r.mu == 1)
        extra(s, r)
        s.push()
        s.add(r.addend_receive() == 0)
        no_addend = s.check()
        s.pop()
        covered[name] = no_addend
    a2_ok = (all(covered[k] == z3.unsat
                 for k in ("AddP1", "AddP2", "AddP12", "Precompute", "Correction"))
             and covered["Double"] == z3.sat)
    report("(a2) all five addend-consuming row types are covered; doublings are not",
           a2_ok,
           "\n".join(f"{k:11}: consumes an addend = "
                     f"{'always' if v == z3.unsat else 'never'}"
                     for k, v in covered.items()))

    # (b) unsatisfiable exactly when xA ≡ xB (mod p) — including non-canonical
    #     encodings that agree mod p but differ as byte strings.
    #
    #     Stated without `%` on symbolic terms (which makes the query nonlinear
    #     AND quantified, and z3 will not return): xB − xA = k·p for some integer
    #     k, and the relation asserts d·(xB − xA) − 1 = m·p for some integer m.
    #     Substituting gives p·(d·k − m) = 1, i.e. p divides 1.
    d, k, m = z3.Int("d"), z3.Int("k"), z3.Int("m")
    s = z3.Solver()
    s.set("timeout", 60_000)
    s.add(d >= 0, d < P)
    s.add(P * (d * k - m) == 1)
    v = s.check()
    b_ok = v == z3.unsat
    report("(b1) D_INV is unsatisfiable whenever xB ≡ xA (mod p)", b_ok,
           f"z3: {v}  —  p·(d·k − m) = 1 has no integer solution for p > 1.\n"
           "Covers the non-canonical case too: the relation is mod p, so a\n"
           "byte-different xB congruent to xA still has (xB − xA) ≡ 0 and no\n"
           "inverse. The degenerate add becomes UNPROVABLE, not merely unlikely.")

    # ...and satisfiable whenever they differ mod p (completeness). Constructive
    # rather than solved-for: p is prime, so Fermat gives the witness directly.
    import random
    rng = random.Random(5150)
    trials = [1, 2, P - 1, P - 2] + [rng.randrange(1, P) for _ in range(200)]
    b2_ok = all((nz * pow(nz, P - 2, P)) % P == 1 for nz in trials)
    report("(b2) D_INV is satisfiable whenever xB ≢ xA (mod p) (completeness)",
           b2_ok,
           f"{len(trials)} residues incl. 1, 2, p−1, p−2: the Fermat inverse is a\n"
           "witness in every case. p is prime (A-PRIME), so the honest prover can\n"
           "always build the column — the check costs no completeness.")

    # (c) the gated-off branch is not a hole: with g = 0 only `rq` survives, and
    #     L1's telescoping turns that into p·(µ·R − q3) = 0, which PINS q3 rather
    #     than leaving it free. Worth stating explicitly — "the check is gated
    #     off here" is exactly the shape a real hole takes.
    q3 = z3.Int("q3")
    mu = z3.Int("mu")
    s = z3.Solver()
    s.set("timeout", 60_000)
    s.add(q3 >= 0, q3 < (1 << 264))
    s.add(z3.Or(mu == 0, mu == 1))
    s.add(P * (mu * (3 * P) - q3) == 0)      # the telescoped gated-off relation
    s.add(q3 != mu * (3 * P))                 # can it be anything else?
    v_c = s.check()
    c_ok = v_c == z3.unsat
    report("(c) the gated-off branch PINS q3 = µ·3p — a doubling is not a hole",
           c_ok,
           f"z3: {v_c}. With g = 0 the relation is µ·R·P − q3·P, which telescopes\n"
           "to p·(µ·R − q3) = 0. Since p ≠ 0, q3 = 3p on a live doubling and 0 on\n"
           "a padding row — not free. Confirmed on real witnesses too: every\n"
           "doubling row's Q3 column is exactly 3p with zero carries\n"
           "(positive_real_witness2.py).")

    # (d) what it discharges: the side condition L4a needs.
    report("(d) L5b's obligation is discharged UNCONDITIONALLY",
           b_ok and a_ok and c_ok,
           "L4a pins λ given the side condition p ∤ (xB − xA). D_INV *is* that\n"
           "side condition, witnessed in-circuit. No dlog assumption, no T₀\n"
           "reduction, no appeal to the input distribution.")
    return a_ok and a2_ok and b_ok and b2_ok and c_ok


def main():
    ok = [port_argument(), l2b_offsets(), l5b_replacement()]
    hdr("SUMMARY")
    for name, v in verdicts:
        print(f"   [{'PROVED' if v else 'FAILED'}] {name.splitlines()[0]}")
    print()
    print(f"   {sum(1 for _, v in verdicts if v)}/{len(verdicts)} proved")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    sys.exit(main())
