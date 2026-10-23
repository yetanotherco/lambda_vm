"""
Formal (z3 / QF_BV) assume-guarantee gate for the BLAKE3 compression chip design.

Method (mirrors ../keccak-verify/z3_verify.py):
  * Every committed column of the designed chip is a FREE bitvector.
  * Every bus lookup (under its precomputed-table contract) and every eval
    constraint becomes an equation relating those free vars.
  * The chip OUTPUT is whatever the constraints force. We assert
    `output != reference(input)` and ask z3 for a counterexample:
        UNSAT -> for all constraint-satisfying assignments, output == reference
                 (the chip is correctly & tightly constrained).
        SAT   -> the constraints permit a wrong output (under-constrained / mis-wired).

The reference (`bref_*`) is written directly from the BLAKE3 spec with 32-bit
BV ops (RotateRight / + / ^) — structurally INDEPENDENT of the chip's byte-level
XOR / halfword-shift wiring, exactly like keccak's zref_round vs the byte circuit.

Chip contracts assumed (assume-guarantee, from prover/src/tables/bitwise.rs):
  ByteAlu[XOR](a,b)->c : a,b,c are bytes and c = a ^ b.  (8-bit width = byte
      range-check; output pinned by the precomputed table.)
  AreBytes[a,b]        : a,b are bytes (8-bit width).
  (HWSL is NOT used: rotations are inlined as the mu-gated linear shift identity
   in*2^r == SLLC*2^16 + SLL, whose soundness is proven by ../keccak-verify/
   hwsl_inline_test.py given the AreBytes 16-bit bounds + 2^16 invertible mod p.)

Add carries and shift decompositions are eval constraints (mu-gated, degree <=3);
here mu=1 (a real row), so mu drops out and we model the ungated equation.

DESIGN DECISIONS UNDER TEST (see DESIGN.md):
  * State stored as bytes; XOR byte-wise via ByteAlu[XOR].
  * rotr16 / rotr8  : FREE byte relabels (no columns, no lookups).
  * rotr12 / rotr7  : inner rotl r=4 / r=9 -> two halfword shift-identities +
                      cross-halfword recombine + halfword swap.
  * 2-operand add   : one carry bit, a+b == s + 2^32*carry, s range-checked.
  * 3-operand add   : O1 option (c) -- TWO summed carry bits c1,c2 in {0,1},
                      a+b+m == s + 2^32*(c1+c2). (No committed intermediate word;
                      degree stays <=3 after mu-gating, unlike k(k-1)(k-2).)
"""
import sys
import json
import os
from z3 import (
    BitVec, BitVecVal, Concat, ZeroExt, RotateRight, Or, And, Solver, sat, unsat,
    Int, IntVal,
)

# ---------------------------------------------------------------------------
# BLAKE3 constants (spec; cross-checked against Plonky3 in the oracle)
# ---------------------------------------------------------------------------
IV = [0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A,
      0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19]
MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
MASK32 = 0xFFFFFFFF
WIDE = 48            # wide BV width for add / shift identities (honest < 2^35 << 2^48)
P = 2**64 - 2**32 + 1  # Goldilocks prime (used in the width-audit field checks)

# G-function schedule: (a,b,c,d, mx_index, my_index) for the 8 calls of a round.
G_CALLS = [
    (0, 4,  8, 12, 0, 1),
    (1, 5,  9, 13, 2, 3),
    (2, 6, 10, 14, 4, 5),
    (3, 7, 11, 15, 6, 7),
    (0, 5, 10, 15, 8, 9),
    (1, 6, 11, 12, 10, 11),
    (2, 7,  8, 13, 12, 13),
    (3, 4,  9, 14, 14, 15),
]

# ===========================================================================
# Independent z3-native reference (BLAKE3 spec, 32-bit BV words)
# ===========================================================================
def bref_g(v, a, b, c, d, mx, my):
    v[a] = v[a] + v[b] + mx
    v[d] = RotateRight(v[d] ^ v[a], 16)
    v[c] = v[c] + v[d]
    v[b] = RotateRight(v[b] ^ v[c], 12)
    v[a] = v[a] + v[b] + my
    v[d] = RotateRight(v[d] ^ v[a], 8)
    v[c] = v[c] + v[d]
    v[b] = RotateRight(v[b] ^ v[c], 7)


def bref_round(v, m):
    for (a, b, c, d, ix, iy) in G_CALLS:
        bref_g(v, a, b, c, d, m[ix], m[iy])


def bref_permute(m):
    return [m[MSG_PERMUTATION[i]] for i in range(16)]


def bref_round_only(state16, msg16):
    """One round, free 16-word state + free 16-word message -> new state."""
    v = list(state16)
    bref_round(v, msg16)
    return v


def bref_compress(h, m, tlo, thi, bl, fl, rounds):
    """Full compression. h:8 BV32, m:16 BV32, counter split tlo/thi, bl, fl."""
    v = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
         BitVecVal(IV[0], 32), BitVecVal(IV[1], 32),
         BitVecVal(IV[2], 32), BitVecVal(IV[3], 32),
         tlo, thi, bl, fl]
    schedule = list(m)
    for r in range(rounds):
        bref_round(v, schedule)
        if r < rounds - 1:
            schedule = bref_permute(schedule)
    out = [None] * 16
    for i in range(8):
        out[i] = v[i] ^ v[i + 8]
        out[i + 8] = v[i + 8] ^ h[i]
    return out


# ===========================================================================
# Chip circuit model.  A "word" is a list of 4 free 8-bit BVs [b0,b1,b2,b3]
# (little-endian). Byte width == the ByteAlu/AreBytes range-check contract.
# ===========================================================================
class Circuit:
    def __init__(self, tag, bug=None):
        self.C = []
        self.tag = tag
        self.bug = bug
        self.n = 0

    def _fresh(self, w=8):
        v = BitVec(f"{self.tag}_v{self.n}", w)
        self.n += 1
        return v

    def fresh_word(self):
        return [self._fresh(8) for _ in range(4)]

    def const_word(self, val):
        return [BitVecVal((val >> (8 * i)) & 0xFF, 8) for i in range(4)]

    # -- value helpers -----------------------------------------------------
    def wval(self, word):
        """word as a WIDE-bit BV integer (little-endian byte combination)."""
        acc = BitVecVal(0, WIDE)
        for i in range(4):
            acc = acc + ZeroExt(WIDE - 8, word[i]) * BitVecVal(1 << (8 * i), WIDE)
        return acc

    def hwval(self, blo, bhi):
        """halfword (2 bytes) as a WIDE-bit BV."""
        return ZeroExt(WIDE - 8, blo) + ZeroExt(WIDE - 8, bhi) * BitVecVal(256, WIDE)

    def word32(self, word):
        return Concat(word[3], word[2], word[1], word[0])

    def fresh_bit(self, boolean=True):
        v = self._fresh(8)
        if boolean:
            self.C.append(Or(v == 0, v == 1))   # mu-gated IS_BIT (mu=1 here)
        return v

    # -- operations under contract ----------------------------------------
    def xor(self, A, B):
        """ByteAlu[XOR]: out byte-wise = A ^ B (auto byte range-check)."""
        out = self.fresh_word()
        for i in range(4):
            self.C.append(out[i] == A[i] ^ B[i])
        return out

    def rotr16(self, A):
        # rotate-right 16 == swap halfwords == byte relabel [b2,b3,b0,b1]. FREE.
        return [A[2], A[3], A[0], A[1]]

    def rotr8(self, A):
        # rotate-right 8 == byte relabel [b1,b2,b3,b0]. FREE.
        return [A[1], A[2], A[3], A[0]]

    def add2(self, A, B, drop_bool=False):
        """2-operand add mod 2^32: a+b == s + 2^32*carry, carry in {0,1}."""
        s = self.fresh_word()
        carry = self.fresh_bit(boolean=not drop_bool)
        self.C.append(
            self.wval(A) + self.wval(B)
            == self.wval(s) + ZeroExt(WIDE - 8, carry) * BitVecVal(1 << 32, WIDE)
        )
        return s

    def add3(self, A, B, M, drop_bool=False):
        """3-operand add mod 2^32 (O1 option c): TWO summed carry bits.
        a+b+m == s + 2^32*(c1+c2), c1,c2 in {0,1}."""
        s = self.fresh_word()
        c1 = self.fresh_bit(boolean=not drop_bool)
        c2 = self.fresh_bit(boolean=not drop_bool)
        csum = ZeroExt(WIDE - 8, c1) + ZeroExt(WIDE - 8, c2)
        self.C.append(
            self.wval(A) + self.wval(B) + self.wval(M)
            == self.wval(s) + csum * BitVecVal(1 << 32, WIDE)
        )
        return s

    def rotr(self, A, n, wrong_amount=False):
        """rotr12 / rotr7 via inner rotl r + halfword swap.

        r=4 for n=12 (rotl20=rotl16.rotl4); r=9 for n=7 (rotl25=rotl16.rotl9).
        Shift identity (inline, mu-gated): hw*2^r == SLLC*2^16 + SLL, with SLL
        the tight 16-bit remainder and SLLC the (loose 16-bit) quotient. Then
        Y_lo = SLL_hi + SLLC_lo, Y_hi = SLL_lo + SLLC_hi  (non-overlapping adds).
        """
        r = {12: 4, 7: 9}[n]
        if wrong_amount:
            r += 1                       # negative control: wrong rotation amount
        xlo = self.hwval(A[0], A[1])
        xhi = self.hwval(A[2], A[3])
        # SLL / SLLC as free halfwords (each = 2 free bytes -> AreBytes 16-bit).
        sll_lo = self.fresh_word()[:2]
        sllc_lo = self.fresh_word()[:2]
        sll_hi = self.fresh_word()[:2]
        sllc_hi = self.fresh_word()[:2]
        SLL_lo, SLLC_lo = self.hwval(*sll_lo), self.hwval(*sllc_lo)
        SLL_hi, SLLC_hi = self.hwval(*sll_hi), self.hwval(*sllc_hi)
        two_r = BitVecVal(1 << r, WIDE)
        two_16 = BitVecVal(1 << 16, WIDE)
        # shift identities
        self.C.append(xlo * two_r == SLLC_lo * two_16 + SLL_lo)
        self.C.append(xhi * two_r == SLLC_hi * two_16 + SLL_hi)
        # recombine (rotl_r) + halfword swap (rotl16)
        Y = self.fresh_word()
        self.C.append(self.hwval(Y[0], Y[1]) == SLL_hi + SLLC_lo)   # Y low  halfword
        self.C.append(self.hwval(Y[2], Y[3]) == SLL_lo + SLLC_hi)   # Y high halfword
        return Y


# ---------------------------------------------------------------------------
# Build one round of the chip (free input state + free message).
# ---------------------------------------------------------------------------
def build_g(cir, v, a, b, c, d, mx, my, bug, gflag):
    b_first = c if (bug == "swap_g_operand" and gflag) else b   # WRONG: v[c] for v[b]
    v[a] = cir.add3(v[a], v[b_first], mx)
    v[d] = cir.rotr16(cir.xor(v[d], v[a]))
    v[c] = cir.add2(v[c], v[d])
    v[b] = cir.rotr(cir.xor(v[b], v[c]), 12,
                    wrong_amount=(bug == "rot_wrong_amount" and gflag))
    v[a] = cir.add3(v[a], v[b], my,
                    drop_bool=(bug == "drop_carry_bool" and gflag))
    v[d] = cir.rotr8(cir.xor(v[d], v[a]))
    v[c] = cir.add2(v[c], v[d])
    v[b] = cir.rotr(cir.xor(v[b], v[c]), 7)


def build_round(cir, v, m, bug=None, bug_first_g_only=True):
    for gi, (a, b, c, d, ix, iy) in enumerate(G_CALLS):
        gflag = (gi == 0) if bug_first_g_only else True
        build_g(cir, v, a, b, c, d, m[ix], m[iy], bug, gflag)


def build_compress(cir, h, m, tlo, thi, bl, fl, rounds, bug=None):
    iv = list(IV)
    if bug == "wrong_iv":
        iv[0] ^= 1                                  # negative control
    v = [h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7],
         cir.const_word(iv[0]), cir.const_word(iv[1]),
         cir.const_word(iv[2]), cir.const_word(iv[3]),
         tlo, thi, bl, fl]
    perm = list(MSG_PERMUTATION)
    if bug == "wrong_msg_index":
        perm[0], perm[1] = perm[1], perm[0]         # negative control
    schedule = list(m)
    for r in range(rounds):
        # only inject round-logic bugs in round 0's first G
        rbug = bug if (r == 0 and bug in
                       ("rot_wrong_amount", "swap_g_operand", "drop_carry_bool")) else None
        build_round(cir, v, schedule, bug=rbug)
        if r < rounds - 1:
            schedule = [schedule[perm[i]] for i in range(16)]
    out = [None] * 16
    for i in range(8):
        out[i] = cir.xor(v[i], v[i + 8])
        out[i + 8] = cir.xor(v[i + 8], h[i])
        if bug == "drop_ff_xor" and i == 0:
            out[0] = cir.fresh_word()               # dropped: output left free
    return out


# ===========================================================================
# Checks
# ===========================================================================
def check_g(bug=None, timeout_ms=0):
    """Single G-function vs reference G. Free 4 state words + 2 message words.
    UNSAT = the G quarter-round is correctly & tightly constrained. A round is a
    fixed composition of 8 G-calls on specified indices, so a correct G under
    arbitrary inputs => correct round (the chaining argument)."""
    tag = "g" + (f"_{bug}" if bug else "")
    cir = Circuit(tag, bug)
    va, vb, vc, vd = (cir.fresh_word(), cir.fresh_word(),
                      cir.fresh_word(), cir.fresh_word())
    mx, my = cir.fresh_word(), cir.fresh_word()
    v = [None] * 16
    v[0], v[1], v[2], v[3] = va, vb, vc, vd
    build_g(cir, v, 0, 1, 2, 3, mx, my, bug, gflag=True)
    rv = [cir.word32(va), cir.word32(vb), cir.word32(vc), cir.word32(vd)]
    bref_g(rv, 0, 1, 2, 3, cir.word32(mx), cir.word32(my))
    s = Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.add(And(*cir.C))
    s.add(Or(cir.word32(v[0]) != rv[0], cir.word32(v[1]) != rv[1],
             cir.word32(v[2]) != rv[2], cir.word32(v[3]) != rv[3]))
    return s.check()


def check_round(bug=None, timeout_ms=0):
    """Round circuit vs reference round. Free state + free message. UNSAT = correct."""
    tag = "rnd" + (f"_{bug}" if bug else "")
    cir = Circuit(tag, bug)
    state = [cir.fresh_word() for _ in range(16)]
    msg = [cir.fresh_word() for _ in range(16)]
    v = list(state)
    build_round(cir, v, msg, bug=bug)
    ref = bref_round_only([cir.word32(w) for w in state],
                          [cir.word32(w) for w in msg])
    s = Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.add(And(*cir.C))
    s.add(Or(*[cir.word32(v[i]) != ref[i] for i in range(16)]))
    return s.check()


def check_compress(rounds, bug=None, timeout_ms=0):
    """Full compression vs reference. UNSAT = correct."""
    tag = f"cmp{rounds}" + (f"_{bug}" if bug else "")
    cir = Circuit(tag, bug)
    h = [cir.fresh_word() for _ in range(8)]
    m = [cir.fresh_word() for _ in range(16)]
    tlo, thi, bl, fl = (cir.fresh_word(), cir.fresh_word(),
                        cir.fresh_word(), cir.fresh_word())
    out = build_compress(cir, h, m, tlo, thi, bl, fl, rounds, bug=bug)
    ref = bref_compress([cir.word32(w) for w in h], [cir.word32(w) for w in m],
                        cir.word32(tlo), cir.word32(thi), cir.word32(bl),
                        cir.word32(fl), rounds)
    s = Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.add(And(*cir.C))
    s.add(Or(*[cir.word32(out[i]) != ref[i] for i in range(16)]))
    return s.check()


def positive_control_compress(rounds, h_i, m_i, tlo_i, thi_i, bl_i, fl_i, out_i):
    """Non-vacuity + external anchor: pin inputs to a concrete oracle vector,
    assert the chip output == the RECORDED oracle output, expect SAT."""
    tag = f"pos{rounds}"
    cir = Circuit(tag)
    h = [cir.fresh_word() for _ in range(8)]
    m = [cir.fresh_word() for _ in range(16)]
    tlo, thi, bl, fl = (cir.fresh_word(), cir.fresh_word(),
                        cir.fresh_word(), cir.fresh_word())
    out = build_compress(cir, h, m, tlo, thi, bl, fl, rounds)
    s = Solver()
    s.add(And(*cir.C))
    # pin inputs
    for wi, val in zip(h, h_i):
        s.add(cir.word32(wi) == BitVecVal(val, 32))
    for wi, val in zip(m, m_i):
        s.add(cir.word32(wi) == BitVecVal(val, 32))
    s.add(cir.word32(tlo) == BitVecVal(tlo_i, 32))
    s.add(cir.word32(thi) == BitVecVal(thi_i, 32))
    s.add(cir.word32(bl) == BitVecVal(bl_i, 32))
    s.add(cir.word32(fl) == BitVecVal(fl_i, 32))
    # pin output to the recorded oracle vector
    for wi, val in zip(out, out_i):
        s.add(cir.word32(wi) == BitVecVal(val, 32))
    return s.check()


# ===========================================================================
# WIDTH AUDIT: field-level (mod p) bound-necessity for the shift identity and
# the add carry.  A wide-BV model cannot show these (2^16 / 2^32 are zero
# divisors mod 2^n); the prime field is required, exactly as
# ../keccak-verify/hwsl_inline_test.py Part 2 demonstrates.
# ===========================================================================
def field_shift_bound(r, in_hw, drop_sll_bound):
    """hw*2^r == SLLC*2^16 + SLL (mod p). SLL bounded to [0,2^16) unless dropped.
    Returns 'unsat' if SLL is pinned to the honest value; 'sat' if ambiguous."""
    s = Solver()
    if drop_sll_bound:
        SLL = Int("SLL"); s.add(SLL >= 0, SLL < P)          # UNBOUNDED field elt
    else:
        lo, hi = Int("sll_lo"), Int("sll_hi")
        s.add(lo >= 0, lo < 256, hi >= 0, hi < 256)          # AreBytes: 2 bytes
        SLL = lo + 256 * hi
    SLLC = Int("SLLC")
    s.add(SLLC >= 0, SLLC < 2**16)                            # loose 16-bit is fine
    s.add((in_hw * (2 ** r) - SLLC * (2 ** 16) - SLL) % P == 0)
    sll_ref = (in_hw * (2 ** r)) % (2 ** 16)
    s.add(SLL != sll_ref)                                     # a WRONG SLL admissible?
    return str(s.check())


def field_add_carry(a, b, m3, drop_bool):
    """3-op: a+b+m == s + 2^32*(c1+c2) (mod p). s in [0,2^32). carries in {0,1}
    unless dropped. Returns 'unsat' if s pinned to honest, 'sat' if ambiguous."""
    s = Solver()
    S = Int("S"); s.add(S >= 0, S < 2**32)
    if drop_bool:
        c1 = Int("c1"); s.add(c1 >= 0, c1 < P)               # UNBOUNDED
        csum = c1
    else:
        c1, c2 = Int("c1"), Int("c2")
        s.add(Or(c1 == 0, c1 == 1), Or(c2 == 0, c2 == 1))
        csum = c1 + c2
    s.add((a + b + m3 - S - (2**32) * csum) % P == 0)
    s_ref = (a + b + m3) % (2**32)
    s.add(S != s_ref)
    return str(s.check())


# ===========================================================================
def load_canonical_6round():
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "..", "blake3-oracle", "canonical_6round_vectors.json")
    with open(path) as f:
        return json.load(f)


def gen_7round_vector():
    """Concrete 7-round compression vector from the validated oracle itself."""
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "..", "blake3-oracle"))
    import blake3_ref as ora
    import random
    rng = random.Random(12345)
    h = [rng.randrange(0, 1 << 32) for _ in range(8)]
    m = [rng.randrange(0, 1 << 32) for _ in range(16)]
    t = rng.randrange(0, 1 << 64)
    bl = rng.randrange(0, 65)
    fl = rng.randrange(0, 128)
    out = ora.compress(h, m, t, bl, fl, rounds=7)
    return h, m, t & MASK32, (t >> 32) & MASK32, bl, fl, out


def main():
    full = "--full" in sys.argv
    print("=" * 70)
    print("BLAKE3 compression-chip z3 gate")
    print("=" * 70)

    # --- MAIN CHECK 0: single G (fundamental unit; covers every G/round) --
    print("\n=== MAIN CHECK 0: one G-function, free inputs (covers every G) ===")
    g = check_g()
    print(f"  G (clean) -> {g}   (want unsat)")
    g_ok = (g == unsat)

    # --- MAIN CHECK 1: init-state layout + feed-forward (rounds=0) --------
    # Tiny & symbolic: v = initial state, then the feed-forward XORs. Isolates
    # the h/IV/counter-split placement and out[i]=v[i]^v[i+8], out[i+8]=v[i+8]^h[i].
    print("\n=== MAIN CHECK 1: init-state + feed-forward (rounds=0, symbolic) ===")
    r0 = check_compress(0)
    print(f"  compress rounds=0 -> {r0}   (want unsat)")
    wrapper_ok = (r0 == unsat)

    # --- Heavy symbolic multi-round UNSATs: BONUS, gated behind --full ----
    round_ok = None
    full6 = full7 = full2 = None
    if full:
        print("\n=== MAIN CHECK 2 (--full): one round, free state+message ===")
        rr = check_round(timeout_ms=1_800_000)
        print(f"  round (clean) -> {rr}   (want unsat)")
        round_ok = (rr == unsat)
        print("\n=== MAIN CHECK 3 (--full): compression rounds=2 (permutation+chaining) ===")
        full2 = check_compress(2, timeout_ms=1_800_000)
        print(f"  compress rounds=2 -> {full2}   (want unsat)")
        print("\n=== MAIN CHECK 4 (--full): FULL compression rounds=6 and rounds=7 ===")
        full6 = check_compress(6, timeout_ms=2_400_000)
        print(f"  compress rounds=6 -> {full6}   (want unsat)")
        full7 = check_compress(7, timeout_ms=2_400_000)
        print(f"  compress rounds=7 -> {full7}   (want unsat)")
    else:
        print("\n=== Heavy symbolic multi-round UNSATs skipped (pass --full) ===")
        print("  G-unsat + fixed G-composition (chaining) already prove every round;")
        print("  rounds=0 proves init+feed-forward; the message permutation is")
        print("  proven load-bearing by the wrong_msg_index control and exercised")
        print("  concretely by the full 6-/7-round positive controls below.")

    # --- NEGATIVE CONTROLS (must all be SAT) -----------------------------
    print("\n=== NEGATIVE CONTROLS — STRUCTURAL bugs (BV-observable, must be SAT) ===")
    # NB: 'dropped carry booleanity' is deliberately NOT here. Dropping a carry
    # column's booleanity is a FIELD-level soundness bug: an unconstrained
    # committed column is a full field element, but in a *bounded BV* model the
    # 8-bit carry + the s in [0,2^32) byte-range still pins s, so BV reports
    # UNSAT. It is demonstrated correctly in the WIDTH AUDIT below (drop -> SAT),
    # exactly as ../keccak-verify/hwsl_inline_test.py Part 2 requires the prime
    # field to show HWSL bound-necessity. This is a feature: the gate separates
    # BV-observable logic bugs from field-only soundness bugs.
    controls = {}
    controls["rot_wrong_amount"] = check_g(bug="rot_wrong_amount")   # wrong rotation amount
    controls["swap_g_operand"] = check_g(bug="swap_g_operand")       # swapped G operand
    controls["wrong_iv"] = check_compress(1, bug="wrong_iv")         # wrong IV constant
    controls["drop_ff_xor"] = check_compress(1, bug="drop_ff_xor")   # dropped feed-forward XOR
    controls["wrong_msg_index"] = check_compress(2, bug="wrong_msg_index")  # wrong msg-schedule index
    for name, res in controls.items():
        print(f"  bug={name:18s} -> {res}   (want sat)")
    controls_ok = all(res == sat for res in controls.values())

    # --- POSITIVE CONTROLS (external anchor: pin to oracle vectors) -------
    print("\n=== POSITIVE CONTROLS (pin input+output to oracle vectors -> SAT) ===")
    vecs = load_canonical_6round()
    pos_ok = True
    for vec in vecs[:3]:
        res = positive_control_compress(
            6, vec["h"], vec["m"], vec["t"] & MASK32, (vec["t"] >> 32) & MASK32,
            vec["block_len"], vec["flags"], vec["out"])
        ok = (res == sat)
        pos_ok &= ok
        print(f"  6round seed={vec['seed']} (canonical) -> {res}   (want sat)")
    h7, m7, tlo7, thi7, bl7, fl7, out7 = gen_7round_vector()
    res7 = positive_control_compress(7, h7, m7, tlo7, thi7, bl7, fl7, out7)
    pos_ok &= (res7 == sat)
    print(f"  7round (oracle-generated)   -> {res7}   (want sat)")

    # --- WIDTH AUDIT (field-level bound-necessity) -----------------------
    # These are the FIELD-level negative controls (BV provably cannot show them,
    # since 2^16 / 2^32 are zero divisors mod 2^n). 'DROP -> sat' == the bug is
    # exploitable in the prime field; 'present -> unsat' == the range check pins
    # the value. Includes the 'dropped carry booleanity' control (team-lead #4).
    print("\n=== WIDTH AUDIT + FIELD-LEVEL NEGATIVE CONTROLS (mod p bound necessity) ===")
    a_sh = field_shift_bound(9, 0x9C3A, drop_sll_bound=False)
    b_sh = field_shift_bound(9, 0x9C3A, drop_sll_bound=True)
    print(f"  shift r=9  AreBytes SLL bound present -> {a_sh}   (want unsat: pinned)")
    print(f"  shift r=9  DROP SLL bound (neg ctrl)  -> {b_sh}   (want sat: forgeable)")
    a_ad = field_add_carry(0xF0000000, 0xF0000000, 0xF0000000, drop_bool=False)
    b_ad = field_add_carry(0xF0000000, 0xF0000000, 0xF0000000, drop_bool=True)
    print(f"  3-add carry booleanity present        -> {a_ad}   (want unsat: pinned)")
    print(f"  3-add DROP booleanity (neg ctrl #4)   -> {b_ad}   (want sat: forgeable)")
    audit_ok = (a_sh == "unsat" and b_sh == "sat" and a_ad == "unsat" and b_ad == "sat")

    # --- VERDICT ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  G-function UNSAT (covers all G)   : {g_ok}")
    print(f"  init+feed-forward UNSAT (rounds=0): {wrapper_ok}")
    if full:
        print(f"  round UNSAT (direct)             : {round_ok}")
        print(f"  compress rounds=2 UNSAT          : {full2 == unsat}")
        print(f"  full 6-round UNSAT               : {full6 == unsat}")
        print(f"  full 7-round UNSAT               : {full7 == unsat}")
    print(f"  negative controls all SAT         : {controls_ok}")
    print(f"  positive controls all SAT         : {pos_ok}   (full 6-/7-round pipeline, concrete)")
    print(f"  width audit (bound necessity)     : {audit_ok}")
    # G correctness + fixed G-composition => round correctness (chaining);
    # rounds=0 => init+feed-forward; positive controls run the full pipeline
    # concretely; the direct multi-round UNSATs (--full) are bonus confirmation.
    base_ok = g_ok and wrapper_ok and controls_ok and pos_ok and audit_ok
    full_ok = (not full) or (round_ok and full2 == unsat
                             and full6 == unsat and full7 == unsat)
    ok = base_ok and full_ok
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL — investigate above'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
