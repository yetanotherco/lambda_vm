"""Adversarial transcription audit of the lincomb2 gate — model ⊆ chip.

Companion to `TRANSCRIPTION-AUDIT.md`. Everything here answers ONE question:

    does the z3 model assert something `prover/src/tables/ecdas2.rs` /
    `ecsm2.rs` do not actually enforce?

That direction is the dangerous one: a model STRONGER than the chip turns a
genuinely forgeable chip into a green UNSAT. The positive anchor
(`positive_real_witness2.py`) structurally cannot see it — honest witnesses
satisfy a correct model and an over-strong model equally well.

The audit found no LIVE over-strong assertion (§A is the proof: the boolean core
is exactly equivalent, in both directions). It found four places where the gate
ASSERTED a property of the chip that nothing read. Those are now fixed, and this
file is the **regression suite for the fixes**: every section re-runs the tamper
that used to pass and requires the detector to fire.

  A  brute-force equivalence of `Ecdas2Row` with an INDEPENDENT second
     transcription of `Ecdas2Constraints::eval` idx 0..=27 (all 2^11 cases,
     both directions).
  B  F4 — N4b re-pointed at the rows where idx 14 is actually load-bearing.
  C  F3 — the PORT lemma now covers the `s_i` prologue, `conv_carry`, and the
     ECSM/ECSM2 membership pair.
  D  F2 — `chip_state()` parses the emitted expression, and enforces
     "gated columns == raw bus multiplicities" in both directions.
  E  F1 — the D_INV gate expression is parsed and cross-checked against the
     Addend receive, with the forgery a narrowed gate would hide.
  F  the `JointSel → PH*/S*` derivation is compared arm for arm.
  G  fail-closed behaviour: an unrecognised chip shape reports ABSENT.

No `.rs` file is modified — every tamper is applied to an in-memory copy.

Run:  <venv>/bin/python audit_transcription.py  (z3; §E's construction needs ../oracle)
"""

import itertools
import sys
from pathlib import Path

import z3

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "oracle"))

import gate2_common as gc
from gate2_common import (
    ECDAS2, ECSM2, Ecdas2Row, chip_state, dinv_gate_state, joint_sel_maps,
    membership_bodies_identical, padding_gate_state, relation_bodies_identical,
)

ECDAS2_SRC = ECDAS2.read_text()
ECSM2_SRC = ECSM2.read_text()
verdicts = []


def hdr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


def check(name, ok, note=""):
    verdicts.append((name, ok))
    print(f"   [{'PASS' if ok else 'FAIL':^6}] {name}")
    for line in note.splitlines():
        if line:
            print(f"            {line}")


def _with_ecdas2(src, fn):
    """Run `fn()` with `gate2_common.ECDAS2` pointing at a tampered copy."""
    tmp = Path("/tmp/_audit_ecdas2.rs")
    tmp.write_text(src)
    orig, gc.ECDAS2 = gc.ECDAS2, tmp
    try:
        return fn()
    finally:
        gc.ECDAS2 = orig


def _with_ecsm2(src, fn):
    tmp = Path("/tmp/_audit_ecsm2.rs")
    tmp.write_text(src)
    orig, gc.ECSM2 = gc.ECSM2, tmp
    try:
        return fn()
    finally:
        gc.ECSM2 = orig


def _replace_once(src, old, new, what):
    assert old in src, f"tamper target not found verbatim: {what}"
    out = src.replace(old, new, 1)
    assert out != src
    return out


# ── A. the schedule block, both directions ──────────────────────────────────

NAMES = ("mu", "op", "nb", "d1", "d2", "s1", "s2", "s3", "sc", "ph1", "ph2")


def chip_admits(vals, with_2227=True):
    """SECOND, independent transcription of ecdas2.rs::eval idx 0..=27."""
    mu, op, nb, d1, d2, s1, s2, s3, sc, ph1, ph2 = vals
    c = [x * (1 - x) for x in vals]                          # 0..=10
    c += [
        ph1 * ph2,                                           # 11
        op * nb,                                             # 12
        (1 - op) * (nb - d1 - d2 + d1 * d2),                 # 13
        op - s1 - s2 - s3 - sc,                              # 14
        (1 - ph1) * d1,                                      # 15
        (1 - ph1) * d2,                                      # 16
        ph1 * sc,                                            # 17
        ph1 * (s1 + s3 - op * d1),                           # 18
        ph1 * (s2 + s3 - op * d2),                           # 19
        mu * (1 - ph1 - ph2) * (s2 - 1),                     # 20
        ph2 * (sc - 1),                                      # 21
    ]
    if with_2227:
        c += [(1 - mu) * x for x in (d1, d2, s1, s2, s3, sc)]  # 22..=27
    return all(v == 0 for v in c)


def model_admits(vals, padding_gate, ablate=()):
    s = z3.Solver()
    r = Ecdas2Row(s, "x", ablate=ablate, padding_gate=padding_gate)
    for n, v in zip(NAMES, vals):
        s.add(getattr(r, n) == v)
    return s.check() == z3.sat


def section_a():
    hdr("A — Ecdas2Row vs an independent transcription of idx 0..=27 (2^11 cases)")
    space = list(itertools.product((0, 1), repeat=11))
    m = {v for v in space if model_admits(v, True)}
    ch = {v for v in space if chip_admits(v, True)}
    stronger, weaker = ch - m, m - ch
    print(f"   model {len(m)} / chip {len(ch)} admitted assignments")
    check("model ⊆ chip (no over-strong boolean constraint)", not stronger,
          "" if not stronger else f"chip admits, model rejects: {sorted(stronger)}")
    check("chip ⊆ model (no omitted boolean constraint)", not weaker,
          "" if not weaker else f"model admits, chip rejects: {sorted(weaker)}")

    m0 = {v for v in space if model_admits(v, False)}
    check("`padding_gate` defaults to the chip (True), not to a weaker model",
          Ecdas2Row.__init__.__defaults__[-1] is True,
          f"padding_gate=False still models the pre-fix chip ({len(m0)} "
          f"assignments vs {len(ch)}) — ablations must now opt in explicitly.")


# ── B. F4: where idx 14 is actually load-bearing ────────────────────────────

def section_b():
    hdr("B — F4: where `OP = ΣS` (idx 14) is actually load-bearing")

    def q(ablate, mu, extra, padding_gate=True):
        s = z3.Solver()
        r = Ecdas2Row(s, "r", ablate=ablate, padding_gate=padding_gate)
        s.add(r.mu == mu)
        extra(s, r)
        s.add(r.op == 0, r.addend_receive() > 0)
        return s.check()

    pad = lambda s, r: s.add(r.ph1 == 0, r.ph2 == 0)
    old_model = q((14,), 0, pad, padding_gate=False)
    real_chip = q((14,), 0, pad, padding_gate=True)
    check("the OLD N4b target (MU=0 padding row) is NOT a forgery on the chip",
          old_model == z3.sat and real_chip == z3.unsat,
          f"padding_gate=False (as N4b used to run): {old_model}\n"
          f"padding_gate=True  (the chip as written): {real_chip}\n"
          "idx 24..=27 zero every selector on a MU=0 row regardless of idx 14.")

    live = {
        "precompute (MU=1, PH1=PH2=0)": lambda s, r: s.add(r.ph1 == 0, r.ph2 == 0),
        "correction  (MU=1, PH2=1)": lambda s, r: s.add(r.ph2 == 1),
    }
    ok, lines = True, []
    for name, extra in live.items():
        b, a = q((), 1, extra), q((14,), 1, extra)
        lines.append(f"{name:30} untampered {b}   ablate-14 {a}")
        ok &= (b == z3.unsat and a == z3.sat)
    check("the NEW N4b target (live PH1=0 phases) IS a forgery", ok,
          "\n".join(lines) + "\nidx 14 stays — the conclusion survives; only the "
          "justification was wrong.")


# ── C. F3: what the PORT lemma covers ───────────────────────────────────────

def section_c():
    hdr("C — F3: the PORT lemma now covers the prologue, conv_carry, and ECSM2")

    base = relation_bodies_identical()
    check("untampered ECDAS/ECDAS2 port is clean", all(base.values()),
          "  ".join(base))

    mem = membership_bodies_identical()
    check("untampered ECSM/ECSM2 membership port is clean", all(mem.values()),
          "  ".join(mem)
          + "\nThis pair was never compared before; the soundness theorem's "
            '"P2 is on the curve" clause depends on it.')

    # TAMPER 1: rebind an operand column in the s_i prologue.
    t = _replace_once(
        ECDAS2_SRC,
        "let xa = |j: usize| Self::byte_at(b, cols::XA, 32, j);",
        "let xa = |j: usize| Self::byte_at(b, cols::XR, 32, j);",
        "s_i prologue xa binding")
    res = _with_ecdas2(t, relation_bodies_identical)
    check("a rebound operand column (xa → cols::XR) is DETECTED",
          not all(res.values()),
          f"flagged: {sorted(k for k, v in res.items() if not v)}\n"
          "Every value lemma (L3a, L4a/b/c) is false on that chip. This tamper "
          "used to report all-identical.")

    # TAMPER 2: break the carry recurrence itself.
    t = _replace_once(
        ECDAS2_SRC,
        "two_pow_8 * c_i - c_prev - Self::s_i(b, relation, i)",
        "two_pow_8 * c_i - Self::s_i(b, relation, i)",
        "conv_carry recurrence")
    res = _with_ecdas2(t, relation_bodies_identical)
    check("a broken carry recurrence (dropped c_prev) is DETECTED",
          not res["conv_carry"],
          "L1's telescoping IS this recurrence; conv_carry was never compared "
          "before.")

    # TAMPER 3: drop the curve constant from ECSM2's membership relation.
    t = _replace_once(
        ECSM2_SRC, "s = s - ok * curve_b;", "s = s - ok * ok;",
        "ECSM2 Yg curve constant")
    res = _with_ecsm2(t, membership_bodies_identical)
    check("a tampered ECSM2 membership relation is DETECTED",
          not all(res.values()),
          f"flagged: {sorted(k for k, v in res.items() if not v)}\n"
          "Without `b` the y² relation no longer pins P2 to the curve.")


# ── D. F2: the padding gate, parsed and matched to the multiplicities ───────

def section_d():
    hdr("D — F2: the padding gate is parsed, and matched against the multiplicities")

    pad = padding_gate_state()
    check("untampered: gate present, and gated set == raw multiplicity set",
          pad["present"] and pad["exact"],
          f"gated              {sorted(pad['gated'])}\n"
          f"raw multiplicities {sorted(pad['raw_multiplicity'])}\n"
          f"unparsed           {pad['unresolved_multiplicities']}")

    # TAMPER 1: delete the emitting loop, keep every comment.
    start = ECDAS2_SRC.index("        for (i, col) in [\n            cols::D1,")
    end = ECDAS2_SRC.index("b.emit_base(22 + i, (one - mu) * x);", start)
    loop = ECDAS2_SRC[start:end + len("b.emit_base(22 + i, (one - mu) * x);")] \
        + "\n        }\n"
    t = _replace_once(ECDAS2_SRC, loop, "        // defence removed\n",
                      "idx 22..=27 emitting loop")
    assert "emit_base(22 + i" not in t
    assert "(1 − MU)·{D1" in t, "the header comment must survive this tamper"
    st = padding_gate_state(t)
    check("deleting the defence while KEEPING its comment is DETECTED",
          not st["present"],
          f"reason: {st['reason']}\n"
          "The old detector matched the comment, so this reported the defence "
          "present and scored N1 as a passing ablation.")

    # TAMPER 2: the original JointBit bug's shape — digits escape the gate.
    t = _replace_once(
        ECDAS2_SRC,
        "        for (i, col) in [\n            cols::D1,\n            cols::D2,\n",
        "        for (i, col) in [\n",
        "gate column list")
    st = padding_gate_state(t)
    check("a digit send escaping the gate is DETECTED (the original bug's shape)",
          st["present"] and not st["exact"]
          and st["ungated_multiplicities"] == {"D1", "D2"},
          f"gated {sorted(st['gated'])}\n"
          f"UNGATED MULTIPLICITY: {sorted(st['ungated_multiplicities'])}")

    # TAMPER 3: a NEW ungated multiplicity appears on a bus.
    t = _replace_once(ECDAS2_SRC,
                      "            Multiplicity::Column(col),",
                      "            Multiplicity::Column(cols::NB),",
                      "JointBit send multiplicity")
    st = padding_gate_state(t)
    check("a NEW ungated multiplicity column (NB) is DETECTED",
          "NB" in st["ungated_multiplicities"],
          f"UNGATED MULTIPLICITY: {sorted(st['ungated_multiplicities'])}\n"
          "The invariant is an EQUALITY, so it fires on a new raw multiplicity "
          "as well as on a deleted gate. That is the direction the original "
          "JointBit bug arrived from.")

    # TAMPER 4: the Dinv block is never emitted.
    t = _replace_once(ECDAS2_SRC, "            (Relation::Dinv, cols::C3),\n", "",
                      "Dinv emit-loop entry")
    st = dinv_gate_state(t)
    check("dropping Dinv from the emit loop is DETECTED", not st["present"],
          f"reason: {st['reason']}\n"
          'The old detector was `"D_INV" in src and "Dinv" in src`, which '
          "survives this untouched.")


# ── E. F1: the D_INV gate expression, and the forgery it hides ──────────────

def section_e():
    hdr("E — F1: the D_INV gate is parsed and matched to the Addend receive")

    st = dinv_gate_state()
    check("untampered: gate expression == Addend receive multiplicity",
          st["present"] and st["matches_addend"],
          f"gate    {sorted(st['gate'])}\naddend  {sorted(st['addend'])}\n"
          f"applied as `g * s`: {st['applied']}   emitted: {st['emitted']}")

    narrowed = _replace_once(
        ECDAS2_SRC,
        """                let g = b.main(0, cols::S1)
                    + b.main(0, cols::S2)
                    + b.main(0, cols::S3)
                    + b.main(0, cols::S_CORR);""",
        """                let g = b.main(0, cols::S1)
                    + b.main(0, cols::S2)
                    + b.main(0, cols::S3);""",
        "Dinv gate expression")
    st_n = _with_ecdas2(narrowed, lambda: chip_state()["dinv_gate_detail"])
    port_n = _with_ecdas2(narrowed, relation_bodies_identical)
    check("dropping S_CORR from the gate is DETECTED", not st_n["present"],
          f"reason: {st_n['reason']}\n"
          f"gate {sorted(st_n['gate'])} vs addend {sorted(st_n['addend'])}\n"
          f"(the PORT check still reports all-identical: {all(port_n.values())} "
          "— it excludes the Dinv arm by design, which is exactly why the gate "
          "needed an invariant of its own.)")

    # the forgery that tamper hides, constructed
    try:
        from ec_ref import GX, GY, P, pt_add, pt_double
        import lincomb2_ref
    except Exception as e:
        print(f"   (construction skipped: {e})")
        return

    T0, _ = lincomb2_ref.t0_ref()
    G = (GX, GY)
    neg = lambda pt: (pt[0], (P - pt[1]) % P)

    # acc = 2^len·T0 + u1·G + u2·P2 ; W = −2^len·T0.  Degenerate ⟺ acc == W
    #   ⟺  u1·G + u2·P2 = −2^(len+1)·T0.  Take u1 = u2 = 1 (len = 1).
    u1 = u2 = 1
    length = max(u1.bit_length(), u2.bit_length())
    t = T0
    for _ in range(length + 1):
        t = pt_double(t)
    P2 = pt_add(neg(t), neg(G))

    acc = T0
    for rr in range(length - 1, -1, -1):
        acc = pt_double(acc)
        e1, e2 = (u1 >> rr) & 1, (u2 >> rr) & 1
        if e1 or e2:
            acc = pt_add(acc, pt_add(G, P2) if (e1 and e2) else (G if e1 else P2))
    tpow = T0
    for _ in range(length):
        tpow = pt_double(tpow)
    W = neg(tpow)

    lam_rel = lambda lam: (lam * (W[0] - acc[0]) + acc[1] - W[1]) % P
    degenerate = acc == W and lam_rel(12345) == 0 and lam_rel(67890) == 0

    def step(lam):
        xr = (lam * lam - 2 * acc[0]) % P
        return xr, (lam * (acc[0] - xr) - acc[1]) % P

    honest = pt_double(acc)
    forged = {step(l) for l in (2, 3, 5, 7, 11)}

    d = z3.Int("d")
    s = z3.Solver()
    s.add(d >= 0, d < P, (d * ((W[0] - acc[0]) % P) - 1) % P == 0)
    blocked = s.check() == z3.unsat

    print()
    print("   THE FORGERY (no discrete log — one point subtraction):")
    print(f"      u1 = u2 = 1, len = {length}, P2 = -2^(len+1)*T0 - G")
    print(f"      P2 = ({P2[0]:#x},")
    print(f"            {P2[1]:#x})")
    print(f"      correction row: xA == xB {acc[0] == W[0]}, "
          f"yA == yB {acc[1] == W[1]}")
    print(f"      lambda relation identically 0 (free lambda): {degenerate}")
    print(f"      honest  Q = ({honest[0]:#x}, ...)")
    print(f"      forged  Q' family: {len(forged)} distinct, none honest: "
          f"{honest not in forged}")
    check("with the real gate, the forged correction row is UNSATISFIABLE",
          degenerate and blocked,
          "d·(xB − xA) = 1 with xB = xA has no solution — the chip as written "
          "blocks it.")


# ── F. the JointSel → PH*/S* derivation ─────────────────────────────────────

def section_f():
    hdr("F — the `JointSel → phase_bits/selector_bits` mapping, arm for arm")
    import positive_real_witness2 as anchor

    rows = anchor.check_sel_maps()
    check("the anchor's hand copy matches the chip's `match` arms",
          all(ok for _, ok, _ in rows),
          "\n".join(f"{n:18}: {'MATCHES' if ok else 'DIFFERS'} {note}"
                    for n, ok, note in rows))

    chip = joint_sel_maps()
    tampered = _replace_once(ECDAS2_SRC, "JointSel::Correction => (0, 1),",
                             "JointSel::Correction => (1, 1),", "phase_bits arm")
    got = _with_ecdas2(tampered, lambda: joint_sel_maps()["phase_bits"])
    check("a changed phase_bits arm is DETECTED",
          got != chip["phase_bits"] and got["Correction"] == (1, 1),
          f"chip now maps Correction → {got['Correction']} while the anchor's "
          "dict still says (0, 1), so `check_sel_maps` fails.\n"
          "RESULTS §7 called this the anchor's last modelled step; it is now "
          "machine-checked.")


# ── G. fail-closed ──────────────────────────────────────────────────────────

def section_g():
    hdr("G — the detectors fail CLOSED on an unrecognised chip shape")

    t = _replace_once(ECDAS2_SRC, "b.emit_base(22 + i, (one - mu) * x);",
                      "b.emit_base(22 + i, gate_helper(b, col));",
                      "idx 22..=27 emission")
    st = padding_gate_state(t)
    check("an unparsed gate shape reports ABSENT, not present", not st["present"],
          f"reason: {st['reason']}\n"
          "A detector that guesses 'probably fine' is the failure mode this "
          "file exists to prevent.")

    t = _replace_once(ECDAS2_SRC, "                let g = b.main(0, cols::S1)",
                      "                let g = gate_expr(b)\n"
                      "                    * b.main(0, cols::S1)", "Dinv gate")
    st = dinv_gate_state(t)
    check("an opaque D_INV gate helper is not silently accepted",
          not st["present"] or not st["matches_addend"],
          f"present={st['present']}  reason: {st['reason']}\n"
          f"parsed gate columns {sorted(st['gate'])}")


def main():
    section_a()
    section_b()
    section_c()
    section_d()
    section_e()
    section_f()
    section_g()

    hdr("SUMMARY")
    for name, ok in verdicts:
        print(f"   [{'PASS' if ok else 'FAIL':^6}] {name}")
    bad = [n for n, ok in verdicts if not ok]
    print()
    print(f"   {len(verdicts) - len(bad)}/{len(verdicts)} checks pass")
    if bad:
        print("   FAILED:")
        for n in bad:
            print(f"      - {n}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
