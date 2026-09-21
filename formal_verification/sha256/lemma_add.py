"""
Lemma 3: the two addition constraints force out_a and out_e to the round's
outputs, and nothing else.

sha256_round.rs:227-235 emits
    temp1 + temp2 - half(OUT_A) - 2^32 * CARRY_A = 0
    D     + temp1 - half(OUT_E) - 2^32 * CARRY_E = 0
with temp1 = H + S1 + ch + K + W and temp2 = S0 + maj. Lemmas 1 and 2 have
already discharged S0/S1 and ch/maj, so they enter here as free 32-bit values.

What is proven: given the range checks, half(OUT_A) is UNIQUELY the low 32 bits
of temp1+temp2, and half(OUT_E) the low 32 bits of d+temp1. A prover cannot
choose a different split.

What is NOT proven here: that temp1 and temp2 are the right expressions. Both
sides of this query are built from the same temp1, so dropping a term from it
changes them together and the query stays unsat — a control for that would be
vacuous in this file, and it is deliberately absent rather than listed green.
That claim belongs to lemma_compose.py.

WIDTH AUDIT for this lemma
  OUT_A/E halves  16 bits  the four `IsHalfword` sends (sha256_round.rs:163-166).
                           THIS is what makes the split unique: with out < 2^32
                           the carry is determined. Dropping it is falsifiable
                           here (`bug="drop_halfword_bounds"` -> sat).
  CARRY_A/E       8 bits   the paired `AreBytes` send (:168). NOT needed for
                           uniqueness — it is needed so the field arithmetic
                           cannot wrap the modulus, which QF-BV cannot see. The
                           true carry is at most 6; the byte bound is loose but
                           sufficient. This is the one bound whose necessity
                           this file cannot demonstrate, and it is the same
                           scope gap #950 documents for the keccak gate.
  h, d, S0, S1,   32 bits  from the ShaRound bus (previous row's range-checked
  ch, maj, k, w            halves), lemma 1, lemma 2, and the ShaK/ShaM lookups.

Non-overflow side condition: temp1+temp2 sums seven 32-bit values, so it is
below 7*2^32 < 2^35 and the 48-bit model arithmetic cannot wrap where the field
arithmetic would not.
"""
import sys

from z3 import BitVec, BitVecVal, LShR, Solver, ULE, sat, unsat

W = 48
M32 = (1 << 32) - 1


def build(bug=None):
    s = Solver()

    def v(name, bound):
        x = BitVec(name, W)
        s.add(ULE(x, bound))
        return x

    h, d = v("h", M32), v("d", M32)
    k, w = v("k", M32), v("w", M32)
    S0, S1 = v("S0", M32), v("S1", M32)
    ch, maj = v("ch", M32), v("maj", M32)

    hi_bound = M32 if bug == "drop_halfword_bounds" else 0xFFFF
    a_lo, a_hi = v("out_a_lo", hi_bound), v("out_a_hi", hi_bound)
    e_lo, e_hi = v("out_e_lo", hi_bound), v("out_e_hi", hi_bound)
    carry_bound = M32 if bug == "drop_carry_bounds" else 255
    ca, ce = v("carry_a", carry_bound), v("carry_e", carry_bound)

    temp1 = h + S1 + ch + k + w
    temp2 = S0 + maj

    out_a = a_lo + 65536 * a_hi
    out_e = e_lo + 65536 * e_hi
    two32 = BitVecVal(1 << 32, W)
    s.add(temp1 + temp2 == out_a + two32 * ca)
    s.add(d + temp1 == out_e + two32 * ce)

    ref_a = (temp1 + temp2) & BitVecVal(M32, W)
    ref_e = (d + temp1) & BitVecVal(M32, W)
    return s, out_a, out_e, ref_a, ref_e


def check(bug=None):
    s, out_a, out_e, ra, re = build(bug)
    from z3 import Or
    s.add(Or(out_a != ra, out_e != re))
    return s.check()


def positive_control():
    """Non-vacuity: models exist, and in one the outputs equal the reference."""
    s, out_a, out_e, ra, re = build()
    if s.check() != sat:
        return None
    m = s.model()
    return (m.eval(out_a).as_long() == m.eval(ra).as_long()
            and m.eval(out_e).as_long() == m.eval(re).as_long())


if __name__ == "__main__":
    print("=== positive control (non-vacuity) ===")
    print("  models exist and agree with the reference:", positive_control())

    print("\n=== negative controls (each MUST be sat) ===")
    for bug in ["drop_halfword_bounds"]:
        r = check(bug)
        print(f"  {bug:<22} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")

    print("\n=== the carry bound is NOT needed for uniqueness (documented gap) ===")
    r = check("drop_carry_bounds")
    print(f"  drop_carry_bounds      {str(r):<6} "
          f"{'unsat = the split stays unique without it, as the audit says' if r == unsat else 'sat'}")

    print("\n=== the lemma: the additions force out_a and out_e ===")
    r = check()
    print(f"  out_a, out_e uniquely the low 32 bits: {r}")

    bad = 0
    if positive_control() is not True:
        print("  FAIL positive control"); bad += 1
    if check("drop_halfword_bounds") != sat:
        print("  FAIL drop_halfword_bounds did not flip"); bad += 1
    if check() != unsat:
        print("  FAIL the lemma is not unsat"); bad += 1
    sys.exit(1 if bad else 0)
