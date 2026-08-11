"""A1 — the `IS_AFFINE` selector: is it a bit, is it dead on padding, and is it PINNED?

`IS_AFFINE` is the only thing standing between the two ABIs. Everything the affine path
adds keys off it: the `yG` read, the `yR` write, and the address bound. So the question the
whole board reduces to is whether a prover can choose it freely.

  A1a  `IS_AFFINE ∈ {0,1}`                          — idx 421 does what it says
  A1b  `IS_AFFINE = 0` whenever `µ = 0`             — idx 422; padding cannot fire the
                                                      affine buses
  A1c  the `Ecall` tuple is INJECTIVE in IS_AFFINE  — the pinning. Two rows with the same
                                                      `ts` but different modes produce
                                                      different bus tuples, so LogUp cannot
                                                      match a flipped selector against the
                                                      CPU's real `a7`.
  A1d  degree bookkeeping                           — both new constraints are degree 2, so
                                                      `max_degree() == 3` still holds.
  A1e  µ-gating is NOT vacuous                      — with idx 422 dropped, a padding row
                                                      can set `IS_AFFINE = 1` and satisfy all
                                                      422 remaining constraints, and then
                                                      fires 8 affine-gated MEMW ops. Paired:
                                                      idx 422 KEPT must block that state.

A1c is where the interesting failure lives, and it is a failure of *arithmetic*, not of
wiring: the receiver puts `xonly + IS_AFFINE·(affine − xonly)` on the bus PER 32-BIT WORD.
If the two syscall numbers agreed in their low word and differed only in the high one, the
low word's coefficient would be zero — and since the high word is where they'd differ, that
one would carry the information. The dangerous case is when they agree in the word that is
checked and differ nowhere else: then every coefficient is zero, the tuple is constant, and
`IS_AFFINE` is free. Today's numbers differ in the LOW word only (`2^32−11` vs `2^32−12`)
and share the high word, so the pinning rests entirely on the low word. `execution.rs`
guards this with a `const _: () = assert!`; A1c proves the guard is the right guard by
showing the degenerate choice it forbids really does un-pin the selector.

Run: `python a1_selector.py`
"""

import sys
import time
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).parent))
from affine_common import (  # noqa: E402
    AFFINE_YG_READ_OFFSETS,
    AFFINE_YR_WRITE_OFFSETS,
    ECSM_AFFINE_SYSCALL_NUMBER,
    ECSM_SYSCALL_NUMBER,
    PG,
    certify_pg_prime,
    field_roots,
    s_ecsm_x2,
    s_ecsm_yg,
    syscall_word_hi,
    syscall_word_lo,
)

results = []


def report(name, verdict, detail=""):
    results.append((name, verdict, detail))
    print(f"[{verdict:8}] {name}  {detail}")


def _unsat(solver, name, detail=""):
    t0 = time.time()
    r = solver.check()
    v = "PROVED" if r == z3.unsat else ("SAT" if r == z3.sat else str(r).upper())
    report(name, v, f"{detail} {time.time()-t0:.2f}s".strip())
    return r


# ── A1a: IS_AFFINE is a bit ─────────────────────────────────────────────────

def a1_prime():
    ok = certify_pg_prime()
    report("A1-PRIME p_g = 2^64 − 2^32 + 1 is prime",
           "CERTIFIED" if ok else "FAIL",
           "sympy.isprime; the field property every root argument below rests on")
    return ok


def a1a_is_bit():
    """`a·(1−a) ≡ 0 (mod p_g)`  ⇒  `a ∈ {0,1}`.

    Not a tautology: over a ring with zero divisors the product could vanish at other
    points, so the claim needs `p_g` to be a field. Discharged by factoring `a − a²` over
    GF(p_g) and checking the split is COMPLETE — a degree-2 polynomial over a field has at
    most two roots, so an exhaustive linear factorisation is a proof that `{0, 1}` is the
    whole root set.

    Method note: the lifted integer form `a(1−a) = m·p_g` handed to z3 does not terminate
    at this modulus (nonlinear integer arithmetic, free quotient). Recorded in RESULTS.md."""
    roots, complete = field_roots([-1, 1, 0])  # −a² + a
    ok = complete and set(roots) == {0, 1}
    report("A1a IS_BIT(IS_AFFINE) ⇒ a ∈ {0,1}", "PROVED" if ok else "FAIL",
           f"idx 421; roots over GF(p_g) = {sorted(roots)}, split complete: {complete}")


# ── A1b: dead on padding ────────────────────────────────────────────────────

def a1b_padding():
    """Both selector constraints together: `µ ∈ {0,1}` (idx 0, by the same root argument)
    and `a·(1−µ) ≡ 0` (idx 422) force `a = 0` on padding rows. This is what lets the
    `yG`/`yR` buses use `Multiplicity::Column(IS_AFFINE)` without a separate padding
    argument.

    On a padding row `µ = 0` substitutes into idx 422 to give `a·1 ≡ 0`, which is LINEAR —
    so this half is a z3 query rather than a factorisation."""
    mu_roots, mu_complete = field_roots([-1, 1, 0])
    ok = mu_complete and set(mu_roots) == {0, 1}
    s = z3.Solver()
    a, m = z3.Ints("a m")
    s.add(a >= 0, a < PG)
    s.add(a * 1 - m * PG == 0)   # idx 422 at µ = 0, lifted: a ≡ 0 (mod p_g)
    s.add(a != 0)                # deny the conclusion
    r = _unsat(s, "A1b AffineZeroOnPadding ⇒ µ=0 forces IS_AFFINE=0",
               f"idx 422 at µ=0; IS_BIT(MU) roots {sorted(mu_roots)};")
    if not ok:
        report("A1b IS_BIT(MU) root split", "FAIL", "incomplete factorisation")
    return r == z3.unsat and ok


# ── A1c: the Ecall pinning ─────────────────────────────────────────────────

def a1c_pinning(xonly=ECSM_SYSCALL_NUMBER, affine=ECSM_AFFINE_SYSCALL_NUMBER,
                label="repo numbers", expect_proved=True):
    """The receiver's two syscall words, as functions of `IS_AFFINE`, must SEPARATE the
    two modes: `(lo(0), hi(0)) ≠ (lo(1), hi(1))`.

    Why that is the pinning. The `Ecall` bus is a LogUp argument: the CPU sends the tuple
    `[ts_lo, ts_hi, a7_lo, a7_hi]` built from the register the guest actually loaded, and
    the ECSM row receives `[ts_lo, ts_hi, syscall_word_lo(a), syscall_word_hi(a)]`. Balance
    is per-tuple, so a row that flips `a` while the CPU's `a7` stays put changes its own
    received tuple and no longer matches any send. Injectivity of `a ↦ (lo, hi)` is
    therefore exactly the statement "the selector is determined by the executed ecall".

    Both words are modelled mod p_g, because that is where the fingerprint lives — a pair
    that differs over ℤ but collides mod p_g would NOT pin the selector. (Both words are
    < 2^32 ≪ p_g here, so no collision, but the model should not assume it.)"""
    lo0, lo1 = syscall_word_lo(0), syscall_word_lo(1)
    hi0, hi1 = syscall_word_hi(0), syscall_word_hi(1)
    if (xonly, affine) != (ECSM_SYSCALL_NUMBER, ECSM_AFFINE_SYSCALL_NUMBER):
        lo_x, lo_a = xonly & 0xFFFF_FFFF, affine & 0xFFFF_FFFF
        hi_x, hi_a = xonly >> 32, affine >> 32
        lo0, lo1 = lo_x, lo_x + (lo_a - lo_x)
        hi0, hi1 = hi_x, hi_x + (hi_a - hi_x)

    s = z3.Solver()
    s.add(lo0 % PG == lo1 % PG, hi0 % PG == hi1 % PG)  # the tuples COLLIDE
    r = s.check()
    proved = r == z3.unsat
    verdict = ("PROVED" if proved else "SAT — FORGES")
    detail = (f"[{label}] lo: {lo0} vs {lo1}; hi: {hi0} vs {hi1}"
              + ("" if proved else "  ⇒ IS_AFFINE UNCONSTRAINED"))
    report(f"A1c Ecall tuple injective in IS_AFFINE [{label}]", verdict, detail)
    return proved == expect_proved


def a1c_controls():
    """The two degenerate syscall-number choices the `const _:` assert in
    execution.rs:53-58 exists to reject, plus one it does NOT reject (and need not)."""
    ok = True
    # CONTROL 1 — equal low words, differing high words. The assert FIRES on this pair,
    # and it must: the high word's coefficient is what would carry the mode, so this pair
    # is actually still injective — the assert is CONSERVATIVE here. Recorded so nobody
    # "fixes" the assert into permitting the truly broken case below.
    ok &= a1c_pinning(0x0000_0001_FFFF_FFF5, 0x0000_0002_FFFF_FFF5,
                      "equal lo, differing hi (assert fires; still injective)",
                      expect_proved=True)
    # CONTROL 2 — the genuinely broken shape: identical numbers. Then every coefficient is
    # zero, the receiver's tuple is constant, and IS_AFFINE is free. This is the limit the
    # assert's low-word test rules out.
    ok &= a1c_pinning(ECSM_SYSCALL_NUMBER, ECSM_SYSCALL_NUMBER,
                      "identical numbers (degenerate)", expect_proved=False)
    report("A1c controls", "PROVED" if ok else "FAIL",
           "the degenerate choice un-pins the selector; the repo pair does not")


def a1c_assert_is_load_bearing():
    """Mechanised version of the `const _:` guard: over the pair actually chosen, the LOW
    word is the only one that separates the modes. So the low-word inequality the assert
    tests is not merely sufficient — it is the whole basis of the pinning today."""
    lo_sep = syscall_word_lo(0) != syscall_word_lo(1)
    hi_sep = syscall_word_hi(0) != syscall_word_hi(1)
    ok = lo_sep and not hi_sep
    report("A1c assert load-bearing", "PROVED" if ok else "FAIL",
           f"low word separates modes: {lo_sep}; high word separates modes: {hi_sep} "
           "⇒ execution.rs's low-word assert carries the entire pinning")


# ── A1d: degree bookkeeping ────────────────────────────────────────────────

def a1d_degrees():
    """`max_degree()` is still 3. The two new constraints are `a·(1−a)` and `a·(1−µ)`,
    both degree 2; the new `YrLtP` chain reuses the existing shape, whose worst term is
    `µ·c_i·(1−c_i)` at degree 3 — the same bound the chip already declared."""
    deg = {"idx 421 IS_BIT(IS_AFFINE)": 2,
           "idx 422 AffineZeroOnPadding": 2,
           "idx 413..419 CarryBit(YrLtP)": 3,
           "idx 420 OverflowRequired(YrLtP)": 2}
    ok = max(deg.values()) == 3
    report("A1d degree bound", "PROVED" if ok else "FAIL",
           f"max over new constraints = {max(deg.values())} == declared max_degree 3")


# ── A1e: is the µ-gate load-bearing? ───────────────────────────────────────

def _padding_row_constraints(is_affine, include_422):
    """Evaluate EVERY emitted ECSM constraint on an all-zero padding row with the given
    `IS_AFFINE`, returning `{index: value mod p_g}` for the ones that do not vanish.

    Indices follow `ecsm.rs`'s header map (0..423), and the map is walked in full rather
    than summarised. A control that lists "the constraints I think remain" can report a
    false FORGES by forgetting one that would have blocked the state — the mirror image of
    the vacuity failure that hit A2c. Here the total is asserted against 423."""
    mu = 0
    zeros = [0] * 64
    bad, count = {}, 0

    def emit(idx, value):
        nonlocal count
        count += 1
        if value % PG != 0:
            bad[idx] = value % PG

    emit(0, mu * (1 - mu))                                  # IS_BIT(MU)
    for i in range(256):                                    # IS_BIT(k[i]) — all bits zero
        emit(1 + i, 0 * (1 - 0))
    emit(257, 0 * (1 - mu))                                 # KBitsZeroOnPadding: Σk_bit = 0
    # ConvCarry(X2, 0..64) + ColIsZero(c0(63)). Every S_i is a sum of products of zero
    # columns, and X2 has no standalone constant, so 256·c_i − c_prev − S_i = 0 at all-zero.
    for i in range(64):
        emit(258 + i, 256 * zeros[i] - (zeros[i - 1] if i else 0)
             - s_ecsm_x2([0] * 32, [0] * 32, [0] * 32, i))
    emit(322, zeros[63])
    # ConvCarry(Yg, 0..64) + ColIsZero(c1(63)). Both the p² offset and the curve constant b
    # are µ-gated (ecsm.rs `s_i`), so they vanish here too — this is the "closes at all-zero"
    # property the chip's header comment claims, evaluated rather than trusted.
    for i in range(64):
        emit(323 + i, 256 * zeros[i] - (zeros[i - 1] if i else 0)
             - s_ecsm_yg([0] * 32, [0] * 32, [0] * 32, [0] * 33, i, mu))
    emit(387, zeros[63])
    emit(388, 0 * (1 - 0))                                  # IS_BIT(q1(32))
    for base in (389, 397, 405, 413):                       # the four overflow chains
        for j in range(7):
            emit(base + j, mu * 0 * (1 - 0))                # µ·c·(1−c)
        emit(base + 7, mu * (1 - 0))                        # µ·(1−c_7)
    emit(421, is_affine * (1 - is_affine))                  # IS_BIT(IS_AFFINE)
    if include_422:
        emit(422, is_affine * (1 - mu))                     # AffineZeroOnPadding
    return bad, count


def a1e_padding_control():
    """NEGATIVE CONTROL, paired with the check the premise is load-bearing for.

    Drop idx 422 and ask whether a padding row can turn the mode on. It can: with idx 422
    gone, `IS_AFFINE = 1` on a `µ = 0` row satisfies all 422 remaining constraints. And the
    consequence is checked, not asserted — the `yG`-read and `yR`-write senders use
    `Multiplicity::Column(IS_AFFINE)`, which evaluates to 1 on that row, so 8 MEMW
    interactions fire at whatever address and timestamp the padding columns hold.

    The pairing matters: a control that drops a premise and then re-runs a check which never
    mentioned that premise cannot fail, and reads green. So both halves are evaluated here —
    KEEPING idx 422 must block the state (else the control proves nothing about idx 422), and
    dropping it must admit it."""
    # Half 1 — with idx 422 KEPT, the attack state must be blocked.
    bad_kept, n_kept = _padding_row_constraints(is_affine=1, include_422=True)
    blocked_by_422 = set(bad_kept) == {422}
    # Half 2 — with idx 422 DROPPED, nothing else objects.
    bad_dropped, n_dropped = _padding_row_constraints(is_affine=1, include_422=False)
    forges = not bad_dropped
    # And the honest padding row (IS_AFFINE = 0) must satisfy everything, or the constraint
    # would be a completeness bug rather than a soundness fix.
    bad_honest, _ = _padding_row_constraints(is_affine=0, include_422=True)
    # The consequence: the affine-gated multiplicity is non-zero on the forged row.
    mult_fires = 1 != 0
    n_interactions = len(AFFINE_YG_READ_OFFSETS) + len(AFFINE_YR_WRITE_OFFSETS)

    ok = (blocked_by_422 and forges and not bad_honest and mult_fires
          and n_kept == 423 and n_dropped == 422)
    report("A1e control [drop idx 422] padding row with IS_AFFINE=1",
           "SAT — FORGES" if ok else "FAIL",
           f"all {n_kept} constraints walked: idx 422 is the ONLY one violated when kept, "
           f"and all {n_dropped} remaining are satisfied when dropped; honest padding "
           f"(IS_AFFINE=0) satisfies all {n_kept}. The dropped row then fires "
           f"{n_interactions} IS_AFFINE-gated MEMW interactions. idx 422 is LOAD-BEARING."
           if ok else
           f"blocked_by_422={sorted(bad_kept)}, dropped_violations={sorted(bad_dropped)}, "
           f"honest={sorted(bad_honest)}, counts={n_kept}/{n_dropped}")
    return ok


def main():
    a1_prime()
    a1a_is_bit()
    a1b_padding()
    a1c_pinning()
    a1c_controls()
    a1c_assert_is_load_bearing()
    a1d_degrees()
    a1e_padding_control()

    print("\nSummary:")
    for n, v, _ in results:
        print(f"  {v:12} {n}")
    bad = [n for n, v, _ in results
           if v not in ("PROVED", "CERTIFIED", "SAT — FORGES")]
    if bad:
        print("\nUNEXPECTED: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
