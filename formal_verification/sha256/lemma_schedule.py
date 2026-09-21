"""
Lemma 5: SHA256MSGSCHED computes the message-schedule recurrence.

prover/src/tables/sha256_schedule.rs emits, for each row i in 16..64:
    BACK16 + sigma0(B15) + BACK7 + sigma1(B2) - half(OUT) - 2^32*CARRY = 0   (:166-172)
    (1 - MU) * AMOUNT = 0                                                     (:174)
against the spec's
    w[i] = w[i-16] + sigma0(w[i-15]) + w[i-7] + sigma1(w[i-2])   (FIPS 180-4 6.2.2)

sigma0/sigma1 over the bit columns are already discharged by lemma_sigma.py
(kinds 0 and 1), so they enter here as free 32-bit values.

COLUMN-ROLE MAP (sha256_schedule.rs:28-43)
  TS     0..2   timestamp
  INDEX  2      i, keys the four ShaM sends at i-2, i-7, i-15, i-16
  AMOUNT 3      how many later rows read w[i]; the receive's multiplicity
  OUT    4..6   w[i] as WordHL: lo + 2^16*hi
  BACK7  6      w[i-7], a single field element
  BACK16 7      w[i-16]
  B15    8..40  w[i-15], 32 bit columns (it is rotated, so it is held as bits)
  B2     40..72 w[i-2], 32 bit columns
  CARRY  72     carry of the four-term sum
  MU     73     row-is-real flag

WIDTH AUDIT
  B15/B2 bits  1 bit   `check_bits(b, &mut id, B15, 32)` / `..., B2, 32)`
                       (:158-160). Modeled as 8-bit with an explicit ULE(.,1),
                       never a 1-bit sort — see lemma_sigma.py.
  OUT halves   16 bits the two `IsHalfword` sends (:98-100). This is what makes
                       the (out, carry) split unique.
  CARRY        8 bits  the paired `AreBytes` send (:101-106), which also
                       range-checks INDEX-16. The spec's argument: adding four
                       range-checked words needs only the carry bounded, and
                       the carry of four 32-bit values does not reach a byte.
  BACK7/16     32 bits from the ShaM bus, discharged by whoever produced them.

Non-overflow side condition: four 32-bit values sum below 2^34, and the model
computes in 48 bits.
"""
import sys

from z3 import BitVec, BitVecVal, Or, Solver, ULE, ZeroExt, sat, unsat

from lemma_sigma import circuit_sigma

W = 48
M32 = (1 << 32) - 1


def build(bug=None):
    s = Solver()

    def v(name, bound=M32):
        x = BitVec(name, W)
        s.add(ULE(x, bound))
        return x

    back16, back7 = v("back16"), v("back7")

    # B15 and B2 as bit columns, so that which sigma is applied to which word is
    # visible to the query. With sigma0/sigma1 as free values the `sigma_swap`
    # control is vacuous — addition commutes, so swapping two frees is a
    # relabeling. Deriving them from the bits is what makes it falsifiable.
    B15 = [BitVec(f"b15_bit_{i}", 8) for i in range(32)]
    B2 = [BitVec(f"b2_bit_{i}", 8) for i in range(32)]
    if bug != "drop_bit_bounds":
        for x in B15 + B2:
            s.add(ULE(x, 1))
    k0, k1 = (1, 0) if bug == "sigma_swap" else (0, 1)
    s0 = ZeroExt(W - 32, circuit_sigma(B15, k0))
    s1 = ZeroExt(W - 32, circuit_sigma(B2, k1))
    hb = M32 if bug == "drop_halfword_bounds" else 0xFFFF
    out_lo, out_hi = v("out_lo", hb), v("out_hi", hb)
    carry = v("carry", M32 if bug == "drop_carry_bounds" else 255)
    MASK = BitVecVal(M32, W)

    lhs = back16 + s0 + back7 + s1
    if bug == "drop_back7":
        lhs = back16 + s0 + s1
    out = out_lo + 65536 * out_hi
    s.add(lhs == out + BitVecVal(1 << 32, W) * carry)

    # the spec side always uses the correct pairing, whatever the circuit did
    spec_s0 = ZeroExt(W - 32, circuit_sigma(B15, 0))
    spec_s1 = ZeroExt(W - 32, circuit_sigma(B2, 1))
    spec = (back16 + spec_s0 + back7 + spec_s1) & MASK
    return s, out, spec


def check(bug=None):
    s, out, spec = build(bug)
    s.add(out != spec)
    return s.check()


def check_mu_gate(bug=None):
    """(1 - MU)*AMOUNT = 0 with MU a bit: a row that carries nothing may not
    claim reads of it. Prove AMOUNT is forced to 0 when MU is 0."""
    s = Solver()
    mu, amount = BitVec("mu", 8), BitVec("amount", 8)
    if bug != "drop_mu_bit_bound":
        s.add(ULE(mu, 1))
    if bug != "drop_mu_gate":
        s.add((BitVecVal(1, 8) - mu) * amount == 0)
    if bug == "drop_mu_bit_bound":
        # The real attack the bit bound stops is mu OUTSIDE {0,1}: with mu = 3
        # the gate reads (1-3)*amount = -2*amount, which is 0 mod 256 at
        # amount = 128, so a row can claim reads while its multiplicity is not
        # a bit. Pinning mu = 0 as the honest case does would make this control
        # vacuous, which is why it asks for the dishonest one.
        s.add(mu != 0, mu != 1, amount != 0)
    else:
        s.add(mu == 0, amount != 0)       # a padding row claiming reads
    return s.check()


def positive_control():
    s, out, spec = build()
    s.add(out != 0)
    if s.check() != sat:
        return None
    m = s.model()
    return m.eval(out).as_long() == m.eval(spec).as_long()


if __name__ == "__main__":
    print("=== positive control (non-vacuity) ===")
    print("  models exist with nonzero out and agree:", positive_control())

    print("\n=== negative controls (each MUST be sat) ===")
    # `drop_bit_bounds` is deliberately absent: both sides of this query build
    # sigma from the same bit columns, so dropping the bound moves them together
    # and the control would sit here green while testing nothing. Its necessity
    # is proven in lemma_sigma.py, where the spec side is the word-level
    # rotate-xor and therefore independent of the bits.
    for bug in ["drop_halfword_bounds", "drop_back7", "sigma_swap"]:
        r = check(bug)
        print(f"  {bug:<22} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")
    for bug in ["drop_mu_gate", "drop_mu_bit_bound"]:
        r = check_mu_gate(bug)
        print(f"  {bug:<22} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")

    print("\n=== the lemma ===")
    print("  w[i] uniquely the low 32 bits of the recurrence:", check())
    print("  a zero-MU row cannot claim reads:                ",
          "unsat (forced)" if check_mu_gate() == unsat else "sat !!!")

    bad = 0
    if positive_control() is not True:
        print("  FAIL positive control"); bad += 1
    for bug in ["drop_halfword_bounds", "drop_back7", "sigma_swap"]:
        if check(bug) != sat:
            print(f"  FAIL control {bug} did not flip"); bad += 1
    for bug in ["drop_mu_gate", "drop_mu_bit_bound"]:
        if check_mu_gate(bug) != sat:
            print(f"  FAIL control {bug} did not flip"); bad += 1
    if check() != unsat or check_mu_gate() != unsat:
        print("  FAIL a lemma is not unsat"); bad += 1
    sys.exit(1 if bad else 0)
