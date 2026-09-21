"""
Lemma 4: the round chip's assembled output is the FIPS 180-4 round.

Lemmas 1-3 discharge the pieces: Sigma0/Sigma1 are the spec's rotate-xor over
the bit columns (lemma_sigma), ch/maj are the spec's Ch/Maj out of the BYTE_ALU
results (lemma_chmaj), and the two addition constraints force out_a/out_e to be
the low 32 bits of their sums (lemma_add). What is left is the grouping: that
    temp1 = h + Sigma1 + Ch + K + W
    temp2 = Sigma0 + Maj
    out_a = temp1 + temp2,  out_e = d + temp1
is the spec's
    T1 = h + Sigma1(e) + Ch(e,f,g) + K + W
    T2 = Sigma0(a) + Maj(a,b,c)
    a' = T1 + T2,  e' = d + T1

The pieces enter as free 32-bit values, already proven equal to their spec
counterparts, so this query is pure arithmetic and every dropped or misplaced
term is falsifiable here.
"""
import sys

from z3 import BitVec, BitVecVal, Or, Solver, ULE, sat, unsat

W = 48
M32 = (1 << 32) - 1


def build(bug=None):
    s = Solver()

    def v(name):
        x = BitVec(name, W)
        s.add(ULE(x, M32))
        return x

    h, d, k, w = v("h"), v("d"), v("k"), v("w")
    S0, S1, ch, maj = v("S0"), v("S1"), v("ch"), v("maj")
    MASK = BitVecVal(M32, W)

    temp1 = h + S1 + ch + k + w
    temp2 = S0 + maj
    if bug == "drop_h_from_temp1":
        temp1 = S1 + ch + k + w
    elif bug == "drop_k":
        temp1 = h + S1 + ch + w
    elif bug == "maj_in_temp1":
        temp1 = h + S1 + maj + k + w
        temp2 = S0 + ch
    elif bug == "out_e_uses_temp2":
        temp2 = S0 + maj
    circuit_a = (temp1 + temp2) & MASK
    circuit_e = ((d + (temp2 if bug == "out_e_uses_temp2" else temp1))) & MASK

    spec_t1 = h + S1 + ch + k + w
    spec_t2 = S0 + maj
    spec_a = (spec_t1 + spec_t2) & MASK
    spec_e = (d + spec_t1) & MASK
    return s, circuit_a, circuit_e, spec_a, spec_e


def check(bug=None):
    s, ca, ce, sa, se = build(bug)
    s.add(Or(ca != sa, ce != se))
    return s.check()


def positive_control():
    s, ca, ce, sa, se = build()
    s.add(ca != 0, ce != 0)
    if s.check() != sat:
        return None
    m = s.model()
    return (m.eval(ca).as_long() == m.eval(sa).as_long()
            and m.eval(ce).as_long() == m.eval(se).as_long())


if __name__ == "__main__":
    print("=== positive control (non-vacuity) ===")
    print("  models exist with nonzero output and agree:", positive_control())

    print("\n=== negative controls (each MUST be sat) ===")
    for bug in ["drop_h_from_temp1", "drop_k", "maj_in_temp1", "out_e_uses_temp2"]:
        r = check(bug)
        print(f"  {bug:<20} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")

    print("\n=== the lemma: the grouping is the spec's ===")
    print("  circuit round == FIPS 180-4 round:", check())

    bad = 0
    if positive_control() is not True:
        print("  FAIL positive control"); bad += 1
    for bug in ["drop_h_from_temp1", "drop_k", "maj_in_temp1", "out_e_uses_temp2"]:
        if check(bug) != sat:
            print(f"  FAIL control {bug} did not flip"); bad += 1
    if check() != unsat:
        print("  FAIL the lemma is not unsat"); bad += 1
    sys.exit(1 if bad else 0)
