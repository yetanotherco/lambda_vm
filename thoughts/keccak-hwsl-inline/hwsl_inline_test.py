"""
REVERSE REWRITE experiment: replace the keccak_rnd Hwsl bus lookups with a
mu-gated LINEAR IDENTITY over the same committed cells, pinned to a unique
solution by the range bounds that ALREADY exist in the circuit.

The identity (field elements): for input halfword `in`, shift `rnc`,
    in * 2^rnc  ==  right * 2^16 + left
where left = shifted-halfword (Hwsl SLL) and right = carry-halfword (Hwsl SLLC).
Given 0<=left<2^16 and 0<=right<2^16 (from AreBytes byte cells) the pair
(right,left) = (quotient,remainder) of Euclidean division of in*2^rnc by 2^16 —
UNIQUE. Soundness relies on 2^16 being INVERTIBLE in the Goldilocks prime field.

Two models, deliberately:
  (Part 1) BV full-round variant, all bounds present (byte cells = 8-bit BVs =
     AreBytes; carry bit = IS_BIT). Here all honest values < 2^31 << p, so a wide
     BV with NO wraparound equals field integer arithmetic -> faithful. Used for
     the equivalence main-check (all 24 rounds) + the existing round-logic
     negative controls. The Hwsl `<<`/`>>` semantics are REMOVED; left/right are
     free byte cells related to `in` ONLY by the linear equation.
  (Part 2) FIELD (Int mod p) isolated decomposition. A BV model CANNOT show the
     drop-range / drop-IS_BIT ambiguity, because 2^16 is a ZERO DIVISOR mod 2^n
     (the carry can only absorb multiples of 2^16), so BV would wrongly keep the
     decomposition pinned. The prime field is required: 2^16 invertible => a wrong
     `left` admits a (large) field `right`. So bound-necessity is proven mod p.
"""
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from z3 import (
    BitVec, BitVecVal, Concat, Extract, ZeroExt, Or, And, Not, Solver, sat, unsat,
    Int, IntVal,
)
from keccak_ref import RHO, RC, keccak_round
from z3_verify import zref_round, pi_src, cxz_right_bit_for_byte

P = 2**64 - 2**32 + 1  # Goldilocks prime

# ===========================================================================
# PART 1 — BV full-round variant (Hwsl lookups -> linear identity)
# ===========================================================================
W = 40  # wide BV width: honest values < 2^33, no wraparound


def build_variant(round_idx, tag, bug=None):
    C = []
    def V(name, w=8):
        return BitVec(f"{tag}_{name}", w)

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

    def hw16(lo, hi):
        return Concat(hi, lo)

    def wb(b8):                       # widen an 8-bit byte cell to W bits
        return ZeroExt(W - 8, b8)

    def operand(sum16):
        C.append(sum16 <= BitVecVal(255, 16))
        return Extract(7, 0, sum16)

    # cxz XOR chain (unchanged)
    for x in range(5):
        for b in range(8):
            C.append(cxz[(x, 0, b)] == start[(x, 0, b)] ^ start[(x, 1, b)])
        for s in range(1, 4):
            yy = s + 1
            for b in range(8):
                C.append(cxz[(x, s, b)] == cxz[(x, s - 1, b)] ^ start[(x, yy, b)])

    # === THETA rotate-C-by-1: LINEAR IDENTITY replaces Hwsl (rnc=1) ===
    # in*2 == carry*2^16 + left ; carry in {0,1} via IS_BIT (now LOAD-BEARING);
    # left = cxzL halfword bounded to [0,2^16) by the 8-bit byte cells (AreBytes).
    for x in range(5):
        for hw in range(4):
            inpW = wb(cxz[(x, 3, 2 * hw)]) + BitVecVal(256, W) * wb(cxz[(x, 3, 2 * hw + 1)])
            leftW = wb(cxzL[(x, 2 * hw)]) + BitVecVal(256, W) * wb(cxzL[(x, 2 * hw + 1)])
            rightW = wb(cxzR[(x, hw)])
            C.append(inpW * BitVecVal(2, W) == rightW * BitVecVal(2 ** 16, W) + leftW)
            if bug != "drop_isbit":
                C.append(Or(cxzR[(x, hw)] == 0, cxzR[(x, hw)] == 1))   # IS_BIT (load-bearing)

    def rotated_c(xp, b):
        hw = cxz_right_bit_for_byte(b)
        expr = ZeroExt(8, cxzL[(xp, b)])
        if hw is not None:
            expr = expr + ZeroExt(8, cxzR[(xp, hw)])
        return operand(expr)

    for x in range(5):
        for b in range(8):
            cm1 = cxz[((x + 4) % 5, 3, b)]
            if bug == "theta_no_rot":
                rc1 = cxz[((x + 1) % 5, 3, b)]
            else:
                rc1 = rotated_c((x + 1) % 5, b)
            C.append(dxz[(x, b)] == cm1 ^ rc1)

    for x in range(5):
        for y in range(5):
            for b in range(8):
                C.append(theta[(x, y, b)] == start[(x, y, b)] ^ dxz[(x, b)])

    # === RHO: LINEAR IDENTITY replaces Hwsl ===
    # in*2^rnc == right*2^16 + left ; left,right = rotL/rotR halfwords bounded to
    # [0,2^16) by 8-bit byte cells (AreBytes rs:771-789).
    rho_tbl = [[RHO[x][y] for y in range(5)] for x in range(5)]
    if bug == "rho_swap":
        rho_tbl[1][0], rho_tbl[2][0] = rho_tbl[2][0], rho_tbl[1][0]
    for x in range(5):
        for y in range(5):
            rnc = rho_tbl[x][y] % 16
            for hw in range(4):
                inpW = wb(theta[(x, y, 2 * hw)]) + BitVecVal(256, W) * wb(theta[(x, y, 2 * hw + 1)])
                leftW = wb(rotL[(x, y, 2 * hw)]) + BitVecVal(256, W) * wb(rotL[(x, y, 2 * hw + 1)])
                rightW = wb(rotR[(x, y, 2 * hw)]) + BitVecVal(256, W) * wb(rotR[(x, y, 2 * hw + 1)])
                C.append(inpW * BitVecVal(2 ** rnc, W) == rightW * BitVecVal(2 ** 16, W) + leftW)

    def pi(X, Y, z):
        sx, sy, l, r = pi_src(X, Y, z)
        return operand(ZeroExt(8, rotL[(sx, sy, l)]) + ZeroExt(8, rotR[(sx, sy, r)]))

    for x in range(5):
        for y in range(5):
            for b in range(8):
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

    rc_bytes = [(RC[round_idx] >> (8 * b)) & 0xFF for b in range(8)]
    for b in range(8):
        C.append(rc[b] == BitVecVal(rc_bytes[b], 8))
        if bug == "iota_no_rc":
            C.append(iota[b] == chi[(0, 0, b)])
        else:
            C.append(iota[b] == chi[(0, 0, b)] ^ rc[b])

    def out_byte(x, y, b):
        return iota[b] if (x == 0 and y == 0) else chi[(x, y, b)]

    return C, out_byte, start


def variant_main_check(round_idx, bug=None):
    tag = f"v{round_idx}" + (f"_{bug}" if bug else "")
    C, out_byte, start = build_variant(round_idx, tag, bug)
    lanes = [[Concat(*[start[(x, y, b)] for b in reversed(range(8))]) for y in range(5)]
             for x in range(5)]
    ref = zref_round(lanes, RC[round_idx])
    s = Solver()
    s.add(And(*C))
    s.add(Or(*[out_byte(x, y, b) != Extract(8 * b + 7, 8 * b, ref[x][y])
               for x in range(5) for y in range(5) for b in range(8)]))
    return str(s.check())


# ===========================================================================
# PART 2 — FIELD (mod p) isolated decomposition: bound-necessity controls
# ===========================================================================
def _byte(s, name):
    v = Int(name)
    s.add(v >= 0, v < 256)
    return v


def field_rho(rnc, in_val, drop_right_bound):
    """Uniqueness of (left,right) for in*2^rnc == right*2^16 + left, mod p.
    Returns 'unsat' if a WRONG decomposition (!= Hwsl output) is impossible."""
    s = Solver()
    llo, lhi = _byte(s, "llo"), _byte(s, "lhi")     # left bytes (AreBytes always present)
    left = llo + 256 * lhi
    if drop_right_bound:
        right = Int("right")                         # UNBOUNDED field element [0,p)
        s.add(right >= 0, right < P)
    else:
        rlo, rhi = _byte(s, "rlo"), _byte(s, "rhi")  # right bytes (AreBytes)
        right = rlo + 256 * rhi
    # field identity
    s.add((in_val * (2 ** rnc) - right * (2 ** 16) - left) % P == 0)
    # honest Hwsl output
    prod = in_val * (2 ** rnc)
    left_ref = prod % (2 ** 16)
    # a decomposition that DIFFERS from the honest shifted-left is admissible?
    s.add(left != left_ref)
    return str(s.check())   # unsat => left pinned to Hwsl value; sat => ambiguous


def field_theta(in_val, drop_isbit):
    """rnc=1: in*2 == carry*2^16 + left, mod p. carry bounded by IS_BIT only."""
    s = Solver()
    llo, lhi = _byte(s, "llo"), _byte(s, "lhi")
    left = llo + 256 * lhi
    if drop_isbit:
        carry = Int("carry")                         # UNBOUNDED field element
        s.add(carry >= 0, carry < P)
    else:
        carry = Int("carry")
        s.add(Or(carry == 0, carry == 1))            # IS_BIT (load-bearing here)
    s.add((in_val * 2 - carry * (2 ** 16) - left) % P == 0)
    left_ref = (in_val * 2) % (2 ** 16)
    s.add(left != left_ref)
    return str(s.check())


# ===========================================================================
def main():
    print("=== PART 1: BV full-round variant (Hwsl -> linear identity) ===", flush=True)
    controls = ["theta_no_rot", "rho_swap", "chi_no_not", "chi_swap", "iota_no_rc"]
    results = {}
    with ProcessPoolExecutor(max_workers=10) as ex:
        futs = {}
        for r in range(24):
            futs[ex.submit(variant_main_check, r, None)] = ("main", r)
        for bug in controls:
            futs[ex.submit(variant_main_check, 1, bug)] = ("ctrl", bug)
        for f in as_completed(futs):
            results[futs[f]] = f.result()
            print(f"  {futs[f]} -> {f.result()}", flush=True)

    main_unsat = all(results[("main", r)] == "unsat" for r in range(24))
    ctrl_ok = all(results[("ctrl", b)] == "sat" for b in controls)
    print(f"\n  MAIN all-24 UNSAT: {main_unsat}", flush=True)
    print(f"  round-logic controls all SAT: {ctrl_ok}", flush=True)

    print("\n=== PART 2: FIELD (mod p) bound-necessity controls ===", flush=True)
    IN = 0x9C3A  # representative input halfword
    # rho with a shift whose right halfword is a genuine 2-byte value (rnc=12)
    a = field_rho(12, IN, drop_right_bound=False)
    b = field_rho(12, IN, drop_right_bound=True)
    print(f"  rho rnc=12  bounds present -> {a}  (want unsat: unique/pinned)", flush=True)
    print(f"  rho rnc=12  DROP right bound -> {b}  (want sat: ambiguous)", flush=True)
    c = field_theta(IN, drop_isbit=False)
    d = field_theta(IN, drop_isbit=True)
    print(f"  theta rnc=1 IS_BIT present -> {c}  (want unsat: carry pinned)", flush=True)
    print(f"  theta rnc=1 DROP IS_BIT   -> {d}  (want sat: ambiguous)", flush=True)

    print("\n================ SUMMARY ================", flush=True)
    verdict = (main_unsat and ctrl_ok and a == "unsat" and b == "sat"
               and c == "unsat" and d == "sat")
    print(f"  equivalent under rewrite (all bounds present): {'YES' if main_unsat else 'NO'}")
    print(f"  round-logic negative controls SAT: {ctrl_ok}")
    print(f"  DROP right-range -> SAT (ambiguous): {b == 'sat'}")
    print(f"  DROP IS_BIT -> SAT (now load-bearing): {d == 'sat'}")
    print(f"  bounds-present -> pinned (rho unsat={a=='unsat'}, theta unsat={c=='unsat'})")
    print(f"\n  OVERALL: {'REWRITE VALID + BOUNDS NECESSARY (as predicted)' if verdict else 'CHECK FAILURES ABOVE'}")
    sys.exit(0 if verdict else 1)


if __name__ == "__main__":
    main()
