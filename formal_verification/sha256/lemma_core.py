"""
Lemma 6-8: the SHA256 core chip (prover/src/tables/sha256.rs).

Lemma 7 is NOT a QF-BV query, and that is the point. The core's pointer
arithmetic goes through `emit_add_pair` (prover/src/constraints/templates.rs:334-374),
which recovers each carry as

    carry = (lhs_limb + rhs_limb - sum_limb) * INV_SHIFT_32

where `INV_SHIFT_32` is the inverse of 2^32 **in the Goldilocks field**. A
bitvector model cannot represent that: mod 2^n the factor 2^32 is a zero
divisor and has no inverse, so any QF-BV encoding of this construction is
either unsound or vacuous. Modeling it needs integers and congruences mod p —
which is the companion check `../keccak/README.md` lists as its first
follow-up, here done for this chip.

What lemma 7 establishes is the thing QF-BV cannot: that the range bounds are
SUFFICIENT mod p. An attacker who could make a limb hold more than the model
assumes, or make the field arithmetic wrap, escapes a bitvector model that
silently assumed the bound.
"""
import sys

from z3 import And, Int, Implies, Not, Or, Solver, sat, unsat

P = (1 << 64) - (1 << 32) + 1          # Goldilocks
TWO32 = 1 << 32
M32 = TWO32 - 1


# ---------------------------------------------------------------- lemma 6 ----
def feed_forward(bug=None):
    """`MU * (h_i + last_i - out_i - 2^32 * carry_i) = 0` with carry_i a bit
    (sha256.rs `check_bits(b, &mut id, CARRY, 8)`), h_i and out_i the big-endian
    recompositions of four memory bytes.

    Claim: out_i is uniquely (h_i + last_i) mod 2^32.
    """
    s = Solver()
    h, last, out, carry, kf = (Int(n) for n in ("h", "last", "out", "carry", "kf"))
    s.add(h >= 0, h <= M32)            # four bytes, big-endian recomposition
    s.add(last >= 0, last <= M32)      # from the ShaRound bus
    if bug == "drop_out_range":
        s.add(out >= 0, out < P)
    else:
        s.add(out >= 0, out <= M32)    # AreBytes on the output byte pairs
    if bug == "drop_carry_bit":
        s.add(carry >= 0, carry < P)
    else:
        s.add(Or(carry == 0, carry == 1))
    # The circuit emits this over the FIELD, so it is a congruence mod p, not an
    # integer identity. Writing `== 0` instead let the integer ranges force the
    # carry to a bit on their own and made `drop_carry_bit` sit green while
    # testing nothing — the constraint has to be modeled where it lives.
    s.add(h + last - out - TWO32 * carry == kf * P)
    # counterexample sought: out is not the true low 32 bits
    s.add(out != (h + last) % TWO32)
    return s.check()


# ---------------------------------------------------------------- lemma 7 ----
def add_pair(bug=None):
    """`emit_add_pair` over the integers, with the carries recovered by the
    field inverse of 2^32 and pinned to bits.

    The field equation `(a + b - c) * inv(2^32) = carry` is, cleared of the
    inverse, `a + b - c = carry * 2^32 (mod p)`. That is what is encoded: an
    integer congruence, not a bitvector identity.

    `carry * (1 - carry) = 0` over a PRIME field forces carry in {0, 1}; that
    step is where primality of p is used, and it is the assumption stated here
    rather than re-derived.
    """
    s = Solver()
    a_lo, b_lo, c_lo = Int("a_lo"), Int("b_lo"), Int("c_lo")
    a_hi, b_hi, c_hi = Int("a_hi"), Int("b_hi"), Int("c_hi")
    k0, k1 = Int("k0"), Int("k1")      # the multiples of p the congruences allow
    c0, c1 = Int("c0"), Int("c1")

    # Each limb is a DWordHL half: two 16-bit columns, so 0..2^32.
    bound = P - 1 if bug == "drop_limb_range" else M32
    for v in (a_lo, b_lo, c_lo, a_hi, b_hi, c_hi):
        s.add(v >= 0, v <= bound)
    for cv in (c0, c1):
        if bug == "drop_carry_bit":
            s.add(cv >= 0, cv < P)
        else:
            s.add(Or(cv == 0, cv == 1))

    # the two field congruences
    s.add(a_lo + b_lo - c_lo - TWO32 * c0 == k0 * P)
    s.add(a_hi + b_hi + c0 - c_hi - TWO32 * c1 == k1 * P)

    # counterexample sought: the low half is not the true sum mod 2^32
    s.add(c_lo != (a_lo + b_lo) % TWO32)
    return s.check()


def add_pair_high(bug=None):
    """Same, for the high half: c_hi must be (a_hi + b_hi + c0) mod 2^32."""
    s = Solver()
    a_lo, b_lo, c_lo = Int("a_lo"), Int("b_lo"), Int("c_lo")
    a_hi, b_hi, c_hi = Int("a_hi"), Int("b_hi"), Int("c_hi")
    k0, k1, c0, c1 = Int("k0"), Int("k1"), Int("c0"), Int("c1")
    bound = P - 1 if bug == "drop_limb_range" else M32
    for v in (a_lo, b_lo, c_lo, a_hi, b_hi, c_hi):
        s.add(v >= 0, v <= bound)
    for cv in (c0, c1):
        s.add(Or(cv == 0, cv == 1))
    s.add(a_lo + b_lo - c_lo - TWO32 * c0 == k0 * P)
    s.add(a_hi + b_hi + c0 - c_hi - TWO32 * c1 == k1 * P)
    s.add(c_hi != (a_hi + b_hi + c0) % TWO32)
    return s.check()


# ---------------------------------------------------------------- lemma 8 ----
def be_cast():
    """The memory boundary is big-endian while the chips are little-endian
    inside. `be(c)` in sha256.rs sums byte_j * 2^(8*(3-j)); check that it really
    is the big-endian reading, and that swapping it for little-endian is a
    different function (so the check is not vacuous)."""
    s = Solver()
    bs = [Int(f"byte_{j}") for j in range(4)]
    for v in bs:
        s.add(v >= 0, v <= 255)
    be = sum(bs[j] * (1 << (8 * (3 - j))) for j in range(4))
    spec = bs[0] * (1 << 24) + bs[1] * (1 << 16) + bs[2] * (1 << 8) + bs[3]
    s.add(be != spec)
    ok = s.check()

    s2 = Solver()
    bs2 = [Int(f"byte_{j}") for j in range(4)]
    for v in bs2:
        s2.add(v >= 0, v <= 255)
    le = sum(bs2[j] * (1 << (8 * j)) for j in range(4))
    spec2 = bs2[0] * (1 << 24) + bs2[1] * (1 << 16) + bs2[2] * (1 << 8) + bs2[3]
    s2.add(le != spec2)
    return ok, s2.check()


def positive_control():
    """Non-vacuity: the add_pair system has models."""
    s = Solver()
    a_lo, b_lo, c_lo, c0, k0 = (Int(n) for n in ("a_lo", "b_lo", "c_lo", "c0", "k0"))
    for v in (a_lo, b_lo, c_lo):
        s.add(v >= 0, v <= M32)
    s.add(Or(c0 == 0, c0 == 1))
    s.add(a_lo + b_lo - c_lo - TWO32 * c0 == k0 * P)
    s.add(a_lo > 0, b_lo > 0, c0 == 1)      # an actual carry, not the trivial model
    return s.check()


if __name__ == "__main__":
    print("=== positive control (non-vacuity) ===")
    print("  add_pair has models with a real carry:", positive_control())

    print("\n=== negative controls (each MUST be sat) ===")
    for name, r in [("feed_forward drop_out_range", feed_forward("drop_out_range")),
                    ("feed_forward drop_carry_bit", feed_forward("drop_carry_bit")),
                    ("add_pair    drop_limb_range", add_pair("drop_limb_range")),
                    ("add_pair    drop_carry_bit", add_pair("drop_carry_bit"))]:
        print(f"  {name:<30} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")

    print("\n=== lemma 8: the big-endian cast ===")
    be_ok, le_ok = be_cast()
    print(f"  be(c) is the big-endian reading: {be_ok}")
    print(f"  little-endian would differ:      {le_ok} (sat = the check is not vacuous)")

    print("\n=== lemma 6: the feed-forward ===")
    print("  out_i uniquely (h_i + last_i) mod 2^32:", feed_forward())

    print("\n=== lemma 7: emit_add_pair over the integers, mod p ===")
    print("  the bounds SUFFICE mod p, low half: ", add_pair())
    print("  the bounds SUFFICE mod p, high half:", add_pair_high())

    bad = 0
    if positive_control() != sat:
        print("  FAIL positive control"); bad += 1
    if feed_forward("drop_out_range") != sat or feed_forward("drop_carry_bit") != sat:
        print("  FAIL a feed_forward control did not flip"); bad += 1
    if add_pair("drop_limb_range") != sat or add_pair("drop_carry_bit") != sat:
        print("  FAIL an add_pair control did not flip"); bad += 1
    be_ok, le_ok = be_cast()
    if be_ok != unsat or le_ok != sat:
        print("  FAIL the big-endian cast check"); bad += 1
    if feed_forward() != unsat or add_pair() != unsat or add_pair_high() != unsat:
        print("  FAIL a lemma is not unsat"); bad += 1
    sys.exit(1 if bad else 0)
