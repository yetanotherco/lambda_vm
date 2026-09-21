"""
Lemma 1 of the gate: the Sigma/sigma expressions over the bit columns are the
rotate-xor the spec defines.

This is split out from the round wiring on purpose. Keccak's round is purely
bitwise, so its whole transition fits one QF-BV query; SHA-256's round adds five
32-bit values, and the carry chains make a single monolithic query intractable.
The split is the same assume-guarantee move the method already makes for helper
chips: prove the rotations here, then let the round query treat S0/S1 as free
32-bit values it has already discharged.

What is proven: for every assignment of the 32 bit columns satisfying
`check_bits` (sha256_common.rs `check_bits`, x*(x-1)=0), the value
`sigma(b, c, kind)` emits equals rotr/shr-xor of the word those bits recompose.
"""
import sys

from z3 import BitVec, BitVecVal, LShR, Solver, ULE, ZeroExt, sat, unsat

WB = 8    # bit columns: 8-bit vectors with an explicit ULE(.,1), never a 1-bit
          # sort, so that dropping the bound is a falsifiable perturbation
WW = 32

PARAMS = {0: (7, 18, 3, 1), 1: (17, 19, 10, 1), 2: (2, 13, 22, 0), 3: (6, 11, 25, 0)}


def circuit_sigma(bits, kind, rot_delta=0):
    """Exactly sha256_common.rs `sigma`: a rotation is an index permutation of
    the bit columns, and each output bit is x+y+z-2(xy+xz+yz)+4xyz, collapsing
    to x+y-2xy where a shift contributes a zero bit."""
    ra, rb, rc, is_shift = PARAMS[kind]
    ra += rot_delta
    acc = BitVecVal(0, WW)
    for i in range(32):
        x, y = bits[(i + ra) % 32], bits[(i + rb) % 32]
        if is_shift and i + rc >= 32:
            bit = x + y - 2 * (x * y)
        else:
            z = bits[i + rc] if is_shift else bits[(i + rc) % 32]
            bit = x + y + z - 2 * (x * y + x * z + y * z) + 4 * (x * y * z)
        acc = acc + (ZeroExt(WW - WB, bit) << i)
    return acc


def spec_sigma(word, kind):
    """The FIPS 180-4 definition, on the recomposed word."""
    ra, rb, rc, is_shift = PARAMS[kind]

    # LShR, not `>>`: z3py's `>>` on a BitVec is an ARITHMETIC shift, which
    # sign-extends once the top bit is set. Using it here silently made the
    # oracle wrong and the first run of this lemma came back sat — the circuit
    # was right and the reference was not. That is what the concrete mirror in
    # model_dataflow.py exists to catch.
    def rotr(v, n):
        return LShR(v, n) | (v << (32 - n))

    third = LShR(word, rc) if is_shift else rotr(word, rc)
    return rotr(word, ra) ^ rotr(word, rb) ^ third


def check(kind, bug=None):
    s = Solver()
    bits = [BitVec(f"bit_{i}", WB) for i in range(32)]
    if bug != "drop_bit_bounds":
        for v in bits:
            s.add(ULE(v, 1))
    # the word the circuit recomposes: `bits(c, 32)` = sum of bit_i * 2^i
    word = BitVecVal(0, WW)
    for i in range(32):
        word = word + (ZeroExt(WW - WB, bits[i]) << i)
    delta = 1 if bug == "off_by_one_rot" else 0
    lhs = circuit_sigma(bits, kind, rot_delta=delta)
    if bug == "wrong_kind":
        lhs = circuit_sigma(bits, (kind + 1) % 4)
    s.add(lhs != spec_sigma(word, kind))
    return s.check()


def positive_control(kind):
    """Non-vacuity: the system has models, and in them the circuit value equals
    the spec value."""
    s = Solver()
    bits = [BitVec(f"bit_{i}", WB) for i in range(32)]
    for v in bits:
        s.add(ULE(v, 1))
    word = BitVecVal(0, WW)
    for i in range(32):
        word = word + (ZeroExt(WW - WB, bits[i]) << i)
    s.add(word != 0)
    if s.check() != sat:
        return None
    m = s.model()
    return m.eval(circuit_sigma(bits, kind)).as_long() == m.eval(spec_sigma(word, kind)).as_long()


NAMES = {0: "sigma0 (rotr 7,18 / shr 3)", 1: "sigma1 (rotr 17,19 / shr 10)",
         2: "Sigma0 (rotr 2,13,22)", 3: "Sigma1 (rotr 6,11,25)"}

if __name__ == "__main__":
    import time
    print("=== positive control (non-vacuity) ===")
    for k in range(4):
        print(f"  {NAMES[k]:<30} circuit == spec in a model: {positive_control(k)}")

    print("\n=== negative controls (each MUST be sat) ===")
    for bug in ["off_by_one_rot", "wrong_kind", "drop_bit_bounds"]:
        r = check(2, bug)
        print(f"  {bug:<18} {str(r):<6} {'CAUGHT' if r == sat else '!!! MISSED'}")

    print("\n=== the lemma: circuit sigma == spec rotate-xor, all four kinds ===")
    for k in range(4):
        t0 = time.time()
        r = check(k)
        print(f"  {NAMES[k]:<30} {str(r):<6} ({time.time() - t0:.1f}s)")

    bad = 0
    for k in range(4):
        if positive_control(k) is not True:
            print(f"  FAIL positive control {k}"); bad += 1
    for bug in ["off_by_one_rot", "wrong_kind", "drop_bit_bounds"]:
        if check(2, bug) != sat:
            print(f"  FAIL control {bug} did not flip"); bad += 1
    for k in range(4):
        if check(k) != unsat:
            print(f"  FAIL lemma kind {k}"); bad += 1
    sys.exit(1 if bad else 0)
