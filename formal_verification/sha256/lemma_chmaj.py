"""
Lemma 2: the five BYTE_ALU results, summed the way the AIR sums them, are Ch
and Maj.

sha256_round.rs:222-223 computes
    ch  = word(E_AND_F) + word(NOT_E_AND_G)
    maj = word(A_AND_B) + word(C_AND_A_XOR_B)
with ADDITION where FIPS 180-4 has XOR. That is only sound because the operands
are bit-disjoint, and disjointness is a property of SHA-256, not of the circuit
— so it is proven here rather than assumed.

Contracts assumed (each a separately verified chip):
  ByteAlu(op, x, y, z)  x,y,z are bytes and z = x op y, op in {AND, XOR}
The `255 - e_byte` operand is the complement form sha256_common.rs `byte_bits`
emits so that NOT needs no extra opsel; that it really is bitwise NOT on a byte
is part of what this lemma checks.
"""
import sys

from z3 import BitVec, BitVecVal, Concat, Solver, sat, unsat

WB = 8


def check_ch(bug=None):
    """ch = (e&f) + (!e&g) equals Ch(e,f,g) = (e&f) ^ (!e&g), byte by byte."""
    s = Solver()
    e, f, g = BitVec("e", WB), BitVec("f", WB), BitVec("g", WB)
    e_and_f = e & f
    not_e = e if bug == "ch_no_not" else (BitVecVal(255, WB) - e)
    not_e_and_g = not_e & g
    # the AIR's sum, widened so a carry between the two terms would show
    circuit = Concat(BitVecVal(0, WB), e_and_f) + Concat(BitVecVal(0, WB), not_e_and_g)
    spec = Concat(BitVecVal(0, WB), (e & f) ^ ((~e) & g))
    s.add(circuit != spec)
    return s.check()


def check_maj(bug=None):
    """maj = (a&b) + (c&(a^b)) equals Maj(a,b,c) = (a&b)^(a&c)^(b&c)."""
    s = Solver()
    a, b, c = BitVec("a", WB), BitVec("b", WB), BitVec("c", WB)
    a_and_b = a & b
    a_xor_b = a ^ b
    src = b if bug == "maj_uses_b_not_c" else c
    c_and_axb = src & a_xor_b
    circuit = Concat(BitVecVal(0, WB), a_and_b) + Concat(BitVecVal(0, WB), c_and_axb)
    spec = Concat(BitVecVal(0, WB), (a & b) ^ (a & c) ^ (b & c))
    s.add(circuit != spec)
    return s.check()


def check_complement():
    """`255 - x` is bitwise NOT on a byte — the identity byte_bits relies on."""
    s = Solver()
    x = BitVec("x", WB)
    s.add((BitVecVal(255, WB) - x) != (~x))
    return s.check()


def positive_control():
    """Non-vacuity: the systems have models and agree in them."""
    s = Solver()
    e, f, g = BitVec("e", WB), BitVec("f", WB), BitVec("g", WB)
    s.add(e != 0, f != 0, g != 0)
    if s.check() != sat:
        return None
    m = s.model()
    ev, fv, gv = (m[v].as_long() for v in (e, f, g))
    return ((ev & fv) + ((255 - ev) & gv)) == ((ev & fv) ^ ((~ev & 255) & gv))


if __name__ == "__main__":
    print("=== positive control (non-vacuity) ===")
    print("  circuit and spec agree on a concrete model:", positive_control())

    print("\n=== negative controls (each MUST be sat) ===")
    for name, r in [("ch_no_not", check_ch("ch_no_not")),
                    ("maj_uses_b_not_c", check_maj("maj_uses_b_not_c"))]:
        print(f"  {name:<18} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")

    print("\n=== the lemma ===")
    for name, r in [("255-x is bitwise NOT", check_complement()),
                    ("ch:  (e&f)+(!e&g) == Ch", check_ch()),
                    ("maj: (a&b)+(c&(a^b)) == Maj", check_maj())]:
        print(f"  {name:<30} {r}")

    bad = 0
    if positive_control() is not True:
        print("  FAIL positive control"); bad += 1
    if check_ch("ch_no_not") != sat or check_maj("maj_uses_b_not_c") != sat:
        print("  FAIL a control did not flip"); bad += 1
    if check_complement() != unsat or check_ch() != unsat or check_maj() != unsat:
        print("  FAIL a lemma is not unsat"); bad += 1
    sys.exit(1 if bad else 0)
