"""
Formal (z3 / QF_BV) assume-guarantee check that one `keccak_rnd` round, as
wired in prover/src/tables/keccak_rnd.rs, computes a correct Keccak-f[1600]
round GIVEN the helper-chip contracts (ByteAlu XOR/AND, Hwsl, KeccakRc; the
AreBytes/IS_BIT range checks are captured by byte-width + explicit bit/byte
domain constraints).

Method: every trace column is a FREE bitvector. Each bus interaction and eval
constraint becomes an equation relating those free vars (under the referenced
chip's contract). The output lanes are whatever the constraints force. We assert
`output != reference_round(input)` and ask z3 for a counterexample:
  UNSAT -> for all constraint-satisfying assignments, output == reference.
  SAT   -> the constraints permit a wrong output (under-constrained / mis-wired).

The reference round (`zref_round`) is written directly from FIPS-202 with 64-bit
BV ops (RotateLeft / xor / and / not) — structurally independent of the circuit's
byte-level HWSL wiring.

Contracts assumed (assume-guarantee):
  ByteAlu(op,a,b,c): a,b,c are bytes and c = a `op` b (op in {XOR,AND}). Operands
      given as linear combos are required to be bytes (the lookup only has byte
      rows) -> modeled as `sum <= 255` on the field value, low 8 bits used.
  Hwsl(in16, s, left16, right16): left16 = (in16 << s) mod 2^16,
      right16 = in16 >> (16 - s)  (right16 = 0 when s = 0).
  KeccakRc(round, rc[8]): rc = little-endian bytes of KECCAK_RC[round].
"""
import sys
from z3 import (
    BitVec, BitVecVal, Concat, Extract, LShR, RotateLeft, ULE, ZeroExt, Or, And,
    Solver, sat, unsat,
)
from keccak_ref import RHO, RC

# --------------------------------------------------------------------------
# byte<->column helpers mirroring keccak_rnd.rs::cols
# --------------------------------------------------------------------------
def cxz_right_bit_for_byte(b):        # rs:126-132
    return (b // 2 + 3) % 4 if b % 2 == 0 else None

def pi_src(X, Y, z):                   # rs:161-174
    sx = (X + 3 * Y) % 5
    sy = X
    rbc = RHO[sx][sy] // 16
    l, r = [(z, (z + 6) % 8),
            ((z + 6) % 8, (z + 4) % 8),
            ((z + 4) % 8, (z + 2) % 8),
            ((z + 2) % 8, z)][rbc]
    return sx, sy, l, r


# --------------------------------------------------------------------------
# Independent z3-native reference round (FIPS-202, 64-bit lanes)
# --------------------------------------------------------------------------
def zref_round(lanes, rc_val):
    # lanes[x][y] : 64-bit BV.  The reference is ALWAYS correct — the `bug=`
    # injections in build_circuit perturb the circuit model only, never this.
    C = [lanes[x][0] ^ lanes[x][1] ^ lanes[x][2] ^ lanes[x][3] ^ lanes[x][4]
         for x in range(5)]
    D = [C[(x + 4) % 5] ^ RotateLeft(C[(x + 1) % 5], 1) for x in range(5)]
    a = [[lanes[x][y] ^ D[x] for y in range(5)] for x in range(5)]
    B = [[None] * 5 for _ in range(5)]
    for X in range(5):
        for Y in range(5):
            sx = (X + 3 * Y) % 5
            sy = X
            B[X][Y] = RotateLeft(a[sx][sy], RHO[sx][sy])
    out = [[None] * 5 for _ in range(5)]
    for x in range(5):
        for y in range(5):
            out[x][y] = B[x][y] ^ ((~B[(x + 1) % 5][y]) & B[(x + 2) % 5][y])
    out[0][0] = out[0][0] ^ BitVecVal(rc_val, 64)
    return out


# --------------------------------------------------------------------------
# Build the circuit-model constraint system as free vars + equations.
# Returns (constraints, out_byte(x,y,b), start_byte(x,y,b)).
# --------------------------------------------------------------------------
def build_circuit(round_idx, tag, bug=None):
    C = []                             # list of z3 Bool constraints
    def V(name, w=8):                  # fresh free var
        return BitVec(f"{tag}_{name}", w)

    # free columns -----------------------------------------------------------
    start = {(x, y, b): V(f"start_{x}_{y}_{b}") for x in range(5) for y in range(5) for b in range(8)}
    cxz = {(x, s, b): V(f"cxz_{x}_{s}_{b}") for x in range(5) for s in range(4) for b in range(8)}
    cxzL = {(x, b): V(f"cxzL_{x}_{b}") for x in range(5) for b in range(8)}
    cxzR = {(x, hw): V(f"cxzR_{x}_{hw}") for x in range(5) for hw in range(4)}
    dxz = {(x, b): V(f"dxz_{x}_{b}") for x in range(5) for b in range(8)}
    theta = {(x, y, b): V(f"theta_{x}_{y}_{b}") for x in range(5) for y in range(5) for b in range(8)}
    rotL = {(x, y, b): V(f"rotL_{x}_{y}_{b}") for x in range(5) for y in range(5) for b in range(8)}
    rotR = {(x, y, b): V(f"rotR_{x}_{y}_{b}") for x in range(5) for y in range(5) for b in range(8)}
    chA = {(x, y, b): V(f"chiand_{x}_{y}_{b}") for x in range(5) for y in range(5) for b in range(8)}
    chi = {(x, y, b): V(f"chi_{x}_{y}_{b}") for x in range(5) for y in range(5) for b in range(8)}
    rc = {b: V(f"rc_{b}") for b in range(8)}
    iota = {b: V(f"iota_{b}") for b in range(8)}

    def hw16(lo, hi):                  # 16-bit from (low byte, high byte)
        return Concat(hi, lo)

    def byte_op_operand(field_expr16):
        # ByteAlu operand contract: field value must be a byte.
        # MUST be ULE, not `<=`: z3py's comparison operators on bitvectors are
        # SIGNED (SLE), so `expr <= 255` would also admit every value with the
        # sign bit set (e.g. 0xFFFF passes at width 16). That models the
        # contract too weakly: the solver may exhibit an operand the real lookup
        # can never supply, reporting a spurious counterexample against an
        # honest chip. Harmless at the widths used here (the operands
        # are sums of two zero-extended bytes, ≤ 510), which is exactly why it
        # must be written correctly for the next chip that copies this contract.
        C.append(ULE(field_expr16, BitVecVal(255, 16)))
        return Extract(7, 0, field_expr16)

    # === theta: Cxz XOR chain === rs:539-588
    for x in range(5):
        for b in range(8):
            C.append(cxz[(x, 0, b)] == start[(x, 0, b)] ^ start[(x, 1, b)])
        for s in range(1, 4):
            yy = s + 1
            for b in range(8):
                C.append(cxz[(x, s, b)] == cxz[(x, s - 1, b)] ^ start[(x, yy, b)])

    # === theta: HWSL rotate-C-by-1 === rs:593-631  (+ eval IS_BIT rs:914-924)
    for x in range(5):
        for hw in range(4):
            inp = hw16(cxz[(x, 3, 2 * hw)], cxz[(x, 3, 2 * hw + 1)])
            left16 = inp << 1
            C.append(hw16(cxzL[(x, 2 * hw)], cxzL[(x, 2 * hw + 1)]) == left16)
            if bug != "drop_hwsl_carry":
                C.append(cxzR[(x, hw)] == ZeroExt(7, Extract(15, 15, inp)))   # carry bit
            # REMOVAL DEMO drop_hwsl_carry: no Hwsl lookup pins the carry —
            # only the IS_BIT eval constraint below survives (carry forgeable).
            C.append(Or(cxzR[(x, hw)] == 0, cxzR[(x, hw)] == 1))          # IS_BIT (redundant)

    def rotated_c(xp, b):              # rs:322-329 / 663-672
        hw = cxz_right_bit_for_byte(b)
        expr = ZeroExt(8, cxzL[(xp, b)])
        if hw is not None:
            expr = expr + ZeroExt(8, cxzR[(xp, hw)])
        return byte_op_operand(expr)

    # === theta: Dxz XOR === rs:661-690
    for x in range(5):
        for b in range(8):
            cm1 = cxz[((x + 4) % 5, 3, b)]
            if bug == "theta_no_rot":
                rc1 = cxz[((x + 1) % 5, 3, b)]           # drop rotate
            else:
                rc1 = rotated_c((x + 1) % 5, b)
            C.append(dxz[(x, b)] == cm1 ^ rc1)

    # === theta final XOR === rs:694-717
    for x in range(5):
        for y in range(5):
            for b in range(8):
                C.append(theta[(x, y, b)] == start[(x, y, b)] ^ dxz[(x, b)])

    # === rho: HWSL === rs:723-766
    rho_tbl = [[RHO[x][y] for y in range(5)] for x in range(5)]
    if bug == "rho_swap":
        rho_tbl[1][0], rho_tbl[2][0] = rho_tbl[2][0], rho_tbl[1][0]
    if bug == "rho_off_by_one":
        rho_tbl[3][2] += 1                 # one lane's shift amount off by 1
    for x in range(5):
        for y in range(5):
            rnc = rho_tbl[x][y] % 16
            for hw in range(4):
                inp = hw16(theta[(x, y, 2 * hw)], theta[(x, y, 2 * hw + 1)])
                left16 = inp << rnc
                C.append(hw16(rotL[(x, y, 2 * hw)], rotL[(x, y, 2 * hw + 1)]) == left16)
                if rnc == 0:
                    C.append(rotR[(x, y, 2 * hw)] == 0)
                    C.append(rotR[(x, y, 2 * hw + 1)] == 0)
                else:
                    right16 = LShR(inp, 16 - rnc)
                    C.append(hw16(rotR[(x, y, 2 * hw)], rotR[(x, y, 2 * hw + 1)]) == right16)

    def pi(X, Y, z):                   # rs:793-795 virtual pi
        sx, sy, l, r = pi_src(X, Y, z)
        return byte_op_operand(ZeroExt(8, rotL[(sx, sy, l)]) + ZeroExt(8, rotR[(sx, sy, r)]))

    # === chi: AND then XOR === rs:796-870
    for x in range(5):
        for y in range(5):
            for b in range(8):
                if bug == "drop_chi_xor_byte" and (x, y, b) == (2, 3, 5):
                    continue  # REMOVAL DEMO: this output byte's defining equations gone
                p0 = pi(x, y, b)
                p1 = pi((x + 1) % 5, y, b)
                p2 = pi((x + 2) % 5, y, b)
                if bug == "chi_no_not":
                    C.append(chA[(x, y, b)] == (p1 & p2))
                elif bug == "chi_swap":
                    C.append(chA[(x, y, b)] == ((BitVecVal(255, 8) - p2) & p1))
                else:
                    C.append(chA[(x, y, b)] == ((BitVecVal(255, 8) - p1) & p2))
                C.append(chi[(x, y, b)] == p0 ^ chA[(x, y, b)])

    # === iota === rs:872-894  (rc pinned by KeccakRc contract rs:518-535)
    rc_round = (round_idx + 1) % 24 if bug == "iota_wrong_rc" else round_idx
    rc_bytes = [(RC[rc_round] >> (8 * b)) & 0xFF for b in range(8)]
    for b in range(8):
        C.append(rc[b] == BitVecVal(rc_bytes[b], 8))
        if bug == "iota_no_rc":
            C.append(iota[b] == chi[(0, 0, b)])
        else:
            C.append(iota[b] == chi[(0, 0, b)] ^ rc[b])

    def out_byte(x, y, b):             # rs:496-509 handoff
        return iota[b] if (x == 0 and y == 0) else chi[(x, y, b)]

    return C, out_byte, start


# --------------------------------------------------------------------------
def check_round(round_idx, bug=None):
    tag = f"r{round_idx}" + (f"_{bug}" if bug else "")
    C, out_byte, start = build_circuit(round_idx, tag, bug=bug)

    # symbolic input lanes from the SAME free start bytes
    lanes = [[Concat(*[start[(x, y, b)] for b in reversed(range(8))]) for y in range(5)]
             for x in range(5)]
    ref = zref_round(lanes, RC[round_idx])
    ref_byte = lambda x, y, b: Extract(8 * b + 7, 8 * b, ref[x][y])

    s = Solver()
    s.add(And(*C))
    # counterexample: circuit output differs from reference somewhere
    s.add(Or(*[out_byte(x, y, b) != ref_byte(x, y, b)
               for x in range(5) for y in range(5) for b in range(8)]))
    return s.check()


def positive_control(round_idx, seed):
    # Non-vacuity: fix start to concrete bytes, solve the constraint system
    # (no diff assertion), confirm SAT and that the pinned output == reference.
    import random
    rng = random.Random(seed)
    tag = f"pos{round_idx}"
    C, out_byte, start = build_circuit(round_idx, tag)
    s = Solver()
    s.add(And(*C))
    concrete = {}
    for x in range(5):
        for y in range(5):
            for b in range(8):
                v = rng.randrange(0, 256)
                concrete[(x, y, b)] = v
                s.add(start[(x, y, b)] == v)
    if s.check() != sat:
        return False, "constraint system UNSAT for a concrete input (VACUOUS!)"
    m = s.model()
    # reference from concrete input, indexed x + 5y
    from keccak_ref import keccak_round
    in_lanes = [0] * 25
    for x in range(5):
        for y in range(5):
            in_lanes[x + 5 * y] = sum(concrete[(x, y, b)] << (8 * b) for b in range(8))
    exp = keccak_round(in_lanes, RC[round_idx])
    for x in range(5):
        for y in range(5):
            got = sum(int(str(m.evaluate(out_byte(x, y, b)))) << (8 * b) for b in range(8))
            if got != exp[x + 5 * y]:
                return False, f"pinned output != reference at lane ({x},{y})"
    return True, "output uniquely pinned to reference"


if __name__ == "__main__":
    bugs = ["theta_no_rot", "rho_swap", "chi_no_not", "chi_swap", "iota_no_rc"]

    print("=== POSITIVE CONTROL (non-vacuity): constraints SAT & pin output ===")
    ok, msg = positive_control(5, seed=1)
    print(f"  round 5: {ok}  ({msg})")
    assert ok

    print("\n=== NEGATIVE CONTROLS (round 1): each buggy model must be SAT ===")
    for bug in bugs:
        r = check_round(1, bug=bug)
        print(f"  bug={bug:14s} -> {r}   (want sat)")
        assert r == sat, f"VACUOUS ENCODING: buggy model {bug} returned {r}"

    print("\n=== MAIN CHECK: clean model, all 24 rounds must be UNSAT ===")
    allunsat = True
    for r in range(24):
        res = check_round(r)
        allunsat &= (res == unsat)
        print(f"  round {r:2d} -> {res}")
        if res != unsat:
            print("  !!! COUNTEREXAMPLE FOUND — investigate")
    print()
    if allunsat:
        print("VERDICT: all 24 rounds UNSAT + all negative controls SAT + positive control OK")
        print("=> keccak_rnd round is provably correct GIVEN the chip contracts.")
    else:
        print("VERDICT: at least one round SAT — see above.")
        sys.exit(1)
