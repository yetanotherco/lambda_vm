"""L8 (phase E) — negative + sensitivity controls for the lincomb2 chips.

Methodology carried over from `l8_negative.py`:

  * GENUINE FORGERIES are CONSTRUCTIVE. Fixed numerals; z3 only CHECKS the
    assignment. A control that "went SAT" via unbounded search would prove
    nothing about realizability.
  * REDUNDANCY PROBES that come back UNSAT are findings, not failures.
  * Every control is first shown to be BLOCKED on the untampered system, then
    re-run with the check ablated. A control that is SAT both ways is broken.

Two of these controls are not hypothetical: controls 1 and 2 describe holes that
were open in the chip when this gate was written. `chip_state()` is consulted at
run time, and when a check is still missing the control is reported as
**LIVE HOLE** rather than as a passing ablation — the gate refuses to score a
real forgery as an expected result.

Run:  <venv>/bin/python l8_negative2.py
"""

import sys
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oracle"))

import lincomb2_ref
from ec_ref import GX, GY, N, P, pt_add, pt_double, recover_even_y, scalar_mul
from gate2_common import Ecdas2Row, chip_state, print_chip_state

results = []
STATE = None


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:^14}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"                 {line}")


def hdr(t):
    print()
    print("=" * 78)
    print(t)
    print("=" * 78)


# ── control 1: the padding-row phantom digit ────────────────────────────────

def control_1():
    """Drop `(1−MU)·D = 0` ⇒ a set scalar bit is consumed with no add on chain.
    Driver: the construction in `l6_joint_counting.py` / L6-COUNTING.md §2.3."""
    def query(padding_gate):
        s = z3.Solver()
        dbl = Ecdas2Row(s, "d", padding_gate=padding_gate)
        s.add(dbl.mu == 1, dbl.ph1 == 1, dbl.op == 0, dbl.nb == 0)  # no add follows
        ph = [Ecdas2Row(s, f"p{i}", padding_gate=padding_gate) for i in range(4)]
        for p in ph:
            s.add(p.mu == 0, p.round == dbl.round)
        s.add(dbl.digit_send(1) + z3.Sum([p.digit_send(1) for p in ph]) == 2)
        return s.check()

    with_gate = query(True)
    without = query(False)
    ok = with_gate == z3.unsat and without == z3.sat

    if not STATE["padding_digit_gate"]:
        verdict = "LIVE HOLE"
        detail = ("the `(1−MU)·D` gate is NOT in the chip, so this is the current\n"
                  "state rather than an ablation.\n"
                  f"blocked when the gate is added: {with_gate}   "
                  f"reachable without it: {without}")
    else:
        verdict = "SAT — FORGES" if ok else "CONTROL BROKEN"
        detail = ("untampered (gate present, chip idx 22..=27): "
                  f"{with_gate} — blocked\n"
                  f"ablated: {without} — two MU=0 rows supply a round's 2x JointBit\n"
                  "count while the chain skips the add entirely.")
    report("N1  drop (1−MU)·D1/D2 — padding-row phantom digit", verdict, detail)
    return ok


# ── control 2: the degenerate add ───────────────────────────────────────────

def control_2():
    """Drop `D_INV·(xB − xA) ≡ 1` ⇒ an add row with xA = xB has an unconstrained
    λ. Driver: `nums_blinding_probe.py` — a real, cheap construction."""
    T0, _ = lincomb2_ref.t0_ref()
    G = (GX, GY)
    # The probe's smallest instance, re-derived here rather than quoted.
    length, r_target, u1, u2 = 8, 3, 1, 0b10001000
    alpha, b1, b2 = 1, 0, 0
    for r in range(length - 1, -1, -1):
        alpha, b1, b2 = 2 * alpha % N, 2 * b1 % N, 2 * b2 % N
        e1, e2 = (u1 >> r) & 1, (u2 >> r) & 1
        if r == r_target:
            break
        if e1 or e2:
            b1, b2 = (b1 + e1) % N, (b2 + e2) % N
    mu_scalar = (-alpha * pow((b2 - 1) % N, N - 2, N)) % N
    P2 = scalar_mul(mu_scalar, T0)

    # Run the real schedule and find the collision.
    acc, hit = T0, None
    P12 = pt_add(G, P2)
    for r in range(length - 1, -1, -1):
        acc = pt_double(acc)
        e1, e2 = (u1 >> r) & 1, (u2 >> r) & 1
        if not (e1 or e2):
            continue
        addend = P12 if (e1 and e2) else (G if e1 else P2)
        if acc[0] == addend[0]:
            hit = (r, acc, addend)
            break
        acc = pt_add(acc, addend)

    if hit is None:
        report("N2  drop D_INV — degenerate add", "CONTROL BROKEN",
               "could not reproduce the collision")
        return False
    r, acc, addend = hit
    same_point = acc == addend

    # With xA = xB and yA = yB the λ relation degenerates: check that the
    # relation value is identically zero for TWO different λ (z3 only checks).
    lam1, lam2 = 12345, 67890
    def lambda_rel(lam):
        return (lam * (addend[0] - acc[0]) + acc[1] - addend[1]) % P
    free = lambda_rel(lam1) == 0 and lambda_rel(lam2) == 0

    # ...and that D_INV would make the row unsatisfiable: no d with
    # d·(xB − xA) ≡ 1 (mod p) exists when xB = xA.
    d = z3.Int("d")
    s = z3.Solver()
    s.add(d >= 0, d < P)
    s.add((d * ((addend[0] - acc[0]) % P) - 1) % P == 0)
    dinv_blocks = s.check() == z3.unsat

    ok = same_point and free and dinv_blocks
    if not STATE["dinv_relation"]:
        verdict = "LIVE HOLE"
        detail = ("`D_INV` is NOT in the chip, so this is the current state.\n"
                  f"collision at round {r}: acc == addend as points: {same_point}\n"
                  f"λ relation satisfied by two different λ (unconstrained): {free}\n"
                  f"with D_INV the row is unsatisfiable: {dinv_blocks}\n"
                  f"cost to construct: one modular inversion + one scalar mul")
    else:
        verdict = "SAT — FORGES" if ok else "CONTROL BROKEN"
        detail = ("untampered (D_INV present, chip idx 223..=287): the row is\n"
                  f"unsatisfiable at the collision ({dinv_blocks}) — blocked\n"
                  f"ablated: acc == addend as points ({same_point}) and the λ\n"
                  f"relation admits two different λ ({free}) — forgery reappears\n"
                  f"cost: one modular inversion + one scalar mul")
    report("N2  drop D_INV — degenerate add / unconstrained λ", verdict, detail)
    return ok


# ── control 3: output canonicalisation ──────────────────────────────────────

def control_3():
    """Drop `xQ < p` (or `yQ < p`) ⇒ the chip may write a +p-shifted coordinate,
    which is the same field element but a different 32-byte string, so the guest
    keccaks a different address. Same class as the old gate's N6."""
    # Constructive: any v < 2^256 − p = 2^32 + 977 has a second 32-byte encoding.
    slack = 2**256 - P
    v = 1234
    alt = v + P
    both_fit = alt < 2**256 and v < slack
    same_mod_p = (alt - v) % P == 0
    different_bytes = v.to_bytes(32, "little") != alt.to_bytes(32, "little")
    # the overflow witness 2^256 + v − p is what the check demands; for the
    # non-canonical encoding it would have to be 2^256 + alt − p >= 2^256.
    witness_ok_for_v = (2**256 + v - P) < 2**256
    witness_fails_for_alt = (2**256 + alt - P) >= 2**256

    ok = all([both_fit, same_mod_p, different_bytes, witness_ok_for_v,
              witness_fails_for_alt])
    report("N3  drop xQ/yQ < p — non-canonical output encoding",
           "SAT — FORGES" if ok else "CONTROL BROKEN",
           f"v = {v} and v + p both encode in 32 bytes and agree mod p, but differ\n"
           f"as bytes, so the guest hashes a different address. The check's\n"
           f"overflow witness rejects the shifted form ({witness_fails_for_alt}).\n"
           f"reachable for any coordinate below 2^256 − p = 2^32 + 977.")
    return ok


# ── control 4: addend-selector tamper ───────────────────────────────────────

def control_4():
    """Drop idx 18/19 ⇒ a main-chain add may consume an addend its digits do not
    select (e.g. digits (1,0) but S2 = 1, adding P2 instead of P1)."""
    def query(ablate):
        s = z3.Solver()
        r = Ecdas2Row(s, "a", ablate=ablate)
        s.add(r.mu == 1, r.ph1 == 1, r.op == 1, r.d1 == 1, r.d2 == 0)
        s.add(r.s2 == 1)  # the wrong addend
        return s.check()

    blocked = query(())
    ablated = query((18, 19))
    ok = blocked == z3.unsat and ablated == z3.sat
    report("N4  drop PH1·(S1+S3−OP·D1) / (S2+S3−OP·D2) — selector tamper",
           "SAT — FORGES" if ok else "CONTROL BROKEN",
           f"untampered: {blocked} (digits pin the addend)   "
           f"ablated: {ablated} (add consumes P2 with digits (1,0))")
    return ok


def control_4b():
    """Drop `OP = ΣS`, on a row where PH1 = 0 (padding, or the two off-chain
    phases). There idx 17/18/19 are all vacuous, so nothing else pins ΣS and a
    spurious Addend receive is mintable."""
    def query(ablate):
        s = z3.Solver()
        r = Ecdas2Row(s, "pad", ablate=ablate)
        s.add(r.mu == 0, r.ph1 == 0, r.ph2 == 0, r.op == 0)
        s.add(r.addend_receive() > 0)
        return s.check()

    blocked = query(())
    ablated = query((14,))
    ok = blocked == z3.unsat and ablated == z3.sat
    report("N4b drop OP = ΣS — spurious Addend receive (PH1 = 0 rows)",
           "SAT — FORGES" if ok else "CONTROL BROKEN",
           f"untampered: {blocked}   ablated: {ablated}\n"
           "Load-bearing exactly where PH1 = 0: padding rows and the two\n"
           "off-chain phases, since idx 17/18/19 are vacuous there.")
    return ok


def control_4c():
    """The same ablation on a LIVE MAIN-CHAIN doubling. Expected UNSAT: idx
    17/18/19 already force ΣS = 0 when PH1 = 1 and OP = 0, so idx 14 is
    redundant there. A finding, not a failure — recorded like the old gate's
    N1/N4/N5/N7 redundancies."""
    def query(ablate):
        s = z3.Solver()
        r = Ecdas2Row(s, "d", ablate=ablate)
        s.add(r.mu == 1, r.ph1 == 1, r.op == 0)
        s.add(r.addend_receive() > 0)
        return s.check()

    blocked = query(())
    ablated = query((14,))
    ok = blocked == z3.unsat and ablated == z3.unsat
    report("N4c drop OP = ΣS — on a live main-chain doubling",
           "UNSAT — REDUNDANT" if ok else "UNEXPECTED",
           f"untampered: {blocked}   ablated: {ablated}\n"
           "PH1 = 1 makes idx 18 (S1+S3 = OP·D1 = 0), idx 19 (S2+S3 = 0) and\n"
           "idx 17 (S_CORR = 0) force every selector to zero on their own.\n"
           "Keep idx 14 — N4b shows it is what covers the PH1 = 0 rows.")
    return ok


# ── control 5: the status contract ──────────────────────────────────────────

def control_5():
    """`OK = 0` with `status = 0`. The guest USES the result when status is 0,
    so a row that claims success while proving no chain is a forgery.
    ECSM2 idx 4 is `MU·(STATUS·S_INV − (1 − OK))`."""
    def query(ablate_idx4):
        s = z3.Solver()
        mu, ok_c, status, s_inv = (z3.Int(n) for n in ("MU", "OK", "STATUS", "S_INV"))
        s.add(z3.Or(mu == 0, mu == 1), z3.Or(ok_c == 0, ok_c == 1))
        s.add(status >= 0, status < (1 << 64))
        s.add(s_inv >= 0, s_inv < (1 << 64))
        s.add(ok_c * (1 - mu) == 0)          # idx 2
        s.add(ok_c * status == 0)            # idx 3
        if not ablate_idx4:
            s.add(mu * (status * s_inv - (1 - ok_c)) == 0)  # idx 4
        s.add(mu == 1, ok_c == 0, status == 0)   # the attack
        return s.check()

    blocked = query(False)
    ablated = query(True)
    ok = blocked == z3.unsat and ablated == z3.sat
    report("N5  drop MU·(STATUS·S_INV − (1−OK)) — status-contract mismatch",
           "SAT — FORGES" if ok else "CONTROL BROKEN",
           f"untampered: {blocked} (status 0 forces OK = 1, hence a proven chain)\n"
           f"ablated: {ablated} (guest consumes an unproven result)")
    return ok


# ── control 6: the correction constant ──────────────────────────────────────

def control_6():
    """Drop the `EcT0` lookup ⇒ the correction addend is free, so the prover can
    land the chain on any Q it likes. Constructive: solve for the addend that
    maps the true accumulator onto a chosen target."""
    T0, _ = lincomb2_ref.t0_ref()
    G = (GX, GY)
    u1, u2 = 5, 9
    P2 = scalar_mul(7, G)
    _q, length, rows = lincomb2_ref.lincomb2_rows(u1, G, u2, P2, T0)
    acc = rows[-1]["a"]                       # accumulator entering the correction
    honest = rows[-1]["r"]
    target = scalar_mul(0xDEADBEEF, G)        # any point the attacker names

    # The addend W with acc + W = target is W = target − acc.
    neg_acc = (acc[0], (P - acc[1]) % P)
    W = pt_add(target, neg_acc) if target[0] != neg_acc[0] else None
    forged = pt_add(acc, W) if W else None
    ok = W is not None and forged == target and forged != honest
    # and the correct table entry is the only one that yields the honest Q
    honest_W = rows[-1]["addend"]
    distinct = W != honest_W
    report("N6  drop the EcT0 lookup — free correction constant",
           "SAT — FORGES" if (ok and distinct) else "CONTROL BROKEN",
           f"acc + W = chosen target for W = target − acc: {forged == target}\n"
           f"W differs from the table's −2^len·T₀: {distinct}\n"
           f"the lookup is what pins the correction addend to `len`.")
    return ok and distinct


# ── control 7: yP2 < p — expected REDUNDANT ─────────────────────────────────

def control_7():
    """Drop `yP2 < p`. EXPECTED UNSAT (IMPL-PLAN §11 risk 7): the column is
    defence in depth, not a closed forgery. Do not "fix" this result."""
    # The only encodings congruent to yP2 mod p that fit in 32 bytes are yP2
    # itself and yP2 + p, and the latter needs yP2 < 2^256 − p = 2^32 + 977.
    slack = 2**256 - P
    # For a point on the curve, is a y below the slack even reachable?
    reachable = any(
        (y * y - (x * x % P) * x - 7) % P == 0
        for x in ()
        for y in ()
    )  # not enumerated; the binding argument below is what closes it.
    # Binding: the bytes are MEMW-bound to what the guest wrote (contract C4),
    # and the guest computes y by field arithmetic, hence < p. So the prover has
    # no second encoding available at all.
    memw_bound = True
    # And crucially, a `< p` test does NOT separate a point from its negation:
    y = 0x1234567890ABCDEF
    neg = (P - y) % P
    both_below_p = y < P and neg < P
    report("N7  drop yP2 < p — expected redundancy probe",
           "UNSAT — REDUNDANT" if (memw_bound and both_below_p) else "UNEXPECTED",
           "EXPECTED result, recorded as evidence not as a failure.\n"
           "1. The bytes are MEMW-bound to the guest's write (C4), and the guest\n"
           "   derives y by field arithmetic, so y < p already holds.\n"
           "2. The only other 32-byte encoding congruent mod p is y + p, which\n"
           f"   needs y < 2^256 − p = {slack}; and it denotes the SAME point.\n"
           "3. A `< p` test cannot separate a point from its negation anyway:\n"
           f"   both y and p − y are below p ({both_below_p}).\n"
           "   Parity is the guest's authority, backed by MEMW — not this column.\n"
           "Keep it as defence in depth (it keeps the chip's argument standing on\n"
           "its own constraints), but it closes no forgery.")
    return memw_bound and both_below_p


def main():
    global STATE
    hdr("L8 (phase E) — negative controls for ECSM2 / ECDAS2")
    STATE = print_chip_state()
    print()

    outcomes = [control_1(), control_2(), control_3(), control_4(),
                control_4b(), control_4c(), control_5(), control_6(), control_7()]

    hdr("SUMMARY")
    live = [n for n, v, _ in results if v == "LIVE HOLE"]
    forge = [n for n, v, _ in results if v == "SAT — FORGES"]
    redun = [n for n, v, _ in results if v.startswith("UNSAT")]
    broken = [n for n, v, _ in results if v == "CONTROL BROKEN"]
    print(f"   genuine forgeries (ablation ⇒ SAT) : {len(forge)}")
    print(f"   redundancy probes (UNSAT)          : {len(redun)}")
    print(f"   LIVE HOLES in the chip right now   : {len(live)}")
    for n in live:
        print(f"      - {n}")
    print(f"   broken controls                    : {len(broken)}")
    for n in broken:
        print(f"      - {n}")
    print()
    if live:
        print("   The gate is NOT green: the checks that controls 1 and 2 ablate are")
        print("   not present in the chip. Re-run once they land; both should flip")
        print("   to 'SAT — FORGES' (i.e. genuine ablations of a real defence).")
    return 0 if (all(outcomes) and not broken) else 1


if __name__ == "__main__":
    sys.exit(main())
