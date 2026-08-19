"""
Formal (z3) assume-guarantee gate for the BLAKE3 chip's CHAINED-ABSORB mode.

Target: `prover/src/tables/blake3.rs` on `blake3-absorb-mode`, the mode
introduced by commit `9cf6c352`. Follows the method of
`formal_verification/keccak/` (the canonical template) and of the earlier
BLAKE3 compression gate `thoughts/blake3/blake3-chip/z3_blake_verify.py`.

===========================================================================
BINDING SCOPE RULE — READ BEFORE EXTENDING
===========================================================================
The 6 rounds are NEVER composed in one query. The monolithic full-compression
query does not close: on 2026-08-06 the predecessor gate's `--full` board ran
145 minutes and returned `unknown` (z3 timeout) for all four of its composed
queries — round, rounds=2, rounds=6, rounds=7 — which its verdict line scored
as failures (recorded in commit `89aeeb8c`). That was a RESOURCE limit, not a
finding: nothing was disproven.

This gate therefore verifies ONE round at a time, and factors each round into
two queries that each close in well under a second:

  * P1b  the byte-level G circuit against the spec G, free inputs. The
         circuit text of a G is identical for all 48 instances, so ONE query
         covers every G of every round — running 48 copies would re-decide the
         same formula.
  * P1a  round r's WIRING, with G abstracted as an uninterpreted function:
         does round r hand each G the state slots `G_INDICES` names and the
         message words `permute^r` names? Six queries, r = 0..5, each with
         that round's concrete schedule indices.

Composition of the two levels, and of round r into round r+1, is a STRUCTURAL
WIRING ARGUMENT, stated here and NOT claimed as SMT-proved:

  (i)  G-internal ∘ round-wiring: the wiring query treats G as a black box; the
       G query proves the black box is the spec's G for arbitrary inputs. Since
       the wiring query's UF is applied to exactly the operands the circuit
       feeds it, substituting the proved G into the proved wiring gives the
       round. This is sound because the G circuit is verified under FREE
       inputs, so it holds at every instantiation.
  (ii) round r → round r+1: `run_flow` (blake3.rs:342-373) is one loop whose
       body writes `v[ia] = a2; v[ib] = b2; v[ic] = c2; v[id] = vd2` back into
       the same array the next iteration reads. Round r's output COLUMNS are
       literally round r+1's input columns — there is no committed handoff to
       constrain and nothing for a prover to choose. ✓ VERIFIED by reading.

===========================================================================
OUT OF SCOPE (owned elsewhere — do not read this gate as covering them)
===========================================================================
  * Bus telescoping / multiset balance of the `Blake3Absorb` chain — that a
    group's rows form ONE chain, that a send has a receiver, that two groups
    cannot interleave. Owned by the in-tree falsification suite
    (`blake3.rs` tests `a_tampered_chained_cv_unbalances_the_chain`,
    `a_group_without_its_end_row_leaves_a_dangling_send`,
    `an_early_end_sends_a_zero_tuple_that_does_not_exist`) and by the
    `rev-absorb` review lane. Every claim below is ROW-LOCAL.
  * MEMW ordering / the memory argument (that a read returns what was written).
  * Fiat-Shamir, the transcript, and everything above the AIR.
  * The helper chips themselves: BITWISE's ByteAlu/AreBytes/Zero/IsB20/IsHalf
    rows are ASSUMED to obey the contracts in `CONTRACTS` below. Each is a
    separately verified, fully enumerated preprocessed table.

===========================================================================
TWO THEORIES, DELIBERATELY
===========================================================================
The compression core is boolean/byte algebra → QF-BV models it exactly (P1).

The absorb mode's own constraints are NOT byte algebra. They are field
arithmetic over selector columns, countdowns and packed limbs, and the
attacks they defend against are field-level: 256, 2^16 and 2^32 are INVERTIBLE
mod the Goldilocks prime while they are zero divisors mod 2^n. A bitvector
model of those constraints would report UNSAT for free and bless a chip that
is forgeable in the field. P2-P5 therefore run in integer arithmetic mod p,
with every congruence linearized by a bounded quotient. This mirrors the
`WIDTH AUDIT` section of the predecessor gate and discipline #3 of
`formal_verification/keccak/README.md`.

===========================================================================
MODELING ASSUMPTION, stated once
===========================================================================
`IS_BIT(x)` is emitted as `x·(1−x) = 0` over the field. Since p is prime and
committed values lie in [0,p), that is EQUIVALENT to `x ∈ {0,1}`, and the
model encodes it as the disjunction. z3 cannot derive this — it does not know
p is prime — so it is supplied. Same convention as the predecessor gate's
`fresh_bit`. (One-line proof: p | x(1−x) ⟹ p|x or p|(1−x) ⟹ x=0 or x=1.)

Run:  python3 z3_absorb_verify.py            (full board, ~seconds)
      python3 z3_absorb_verify.py --verbose  (also print witnesses)
"""
import sys
import time

from z3 import (
    And, BitVec, BitVecVal, BitVecSort, Concat, Function, Int, Or, RotateRight,
    Solver, ZeroExt, sat, unsat,
)

import blake3_ref as ref

# Goldilocks.
P = 2**64 - 2**32 + 1

# Widths the chip's helper lookups pin. See CONTRACTS.
BYTE = 256
HALF = 2**16
B20 = 2**20
W32 = 2**32

# `ABSORB_MAX_BLOCKS` / `ABSORB_CAP_SCALE` (blake3.rs:125-128).
ABSORB_MAX_BLOCKS = 1 << 10
ABSORB_CAP_SCALE = (1 << 20) // ABSORB_MAX_BLOCKS  # = 2^10

CONTRACTS = """
Typed helper-chip contracts ASSUMED by this gate (bitwise.rs:756-830):
  ByteAlu(op,a,b,c) : a,b ∈ [0,256) and c = a op b. Operands are byte
                      range-checked BY the lookup — the table has only byte rows.
  AreBytes(a,b)     : a,b ∈ [0,256).
  Zero(v) -> z      : v ∈ [0,2^20)  AND  z = 1 if v = 0 else 0.  ★ the DOMAIN
                      bound is as load-bearing as the output — see P4.6.
  IsB20(v)          : v ∈ [0,2^20).
  IsHalfword(v)     : v ∈ [0,2^16).
  Memw register read: the value's lo32 limb ∈ [0,2^32); the hi32 slot here is
                      the CONSTANT 0 in the tuple (blake3.rs:1385,1394).
"""

VERBOSE = "--verbose" in sys.argv


# ===========================================================================
# Field-model plumbing: congruences as bounded-quotient linear constraints.
# ===========================================================================
_kid = [0]


def cong0(cons, expr, emin, emax):
    """Assert expr ≡ 0 (mod p), linearly, given integer bounds on expr."""
    _kid[0] += 1
    k = Int(f"_q{_kid[0]}")
    cons.append(k >= emin // P - 1)
    cons.append(k <= emax // P + 1)
    cons.append(expr == k * P)


def fe(name, cons):
    """A committed column: an arbitrary field element in [0,p)."""
    v = Int(name)
    cons.append(v >= 0)
    cons.append(v < P)
    return v


def bit(name, cons):
    """A column carrying IS_BIT — see MODELING ASSUMPTION."""
    v = Int(name)
    cons.append(Or(v == 0, v == 1))
    return v


def distinct_names(*groups):
    """Guard for uniqueness queries: two z3 variables with the same NAME are
    the same variable, so a query that builds 'two independent models' out of
    identically-named vars is trivially UNSAT — vacuously green. This asserts
    the models really are independent. (A live bug caught by this check during
    development; kept as a permanent guard.)"""
    seen = set()
    for grp in groups:
        names = {str(x) for x in grp}
        if names & seen:
            raise AssertionError(
                f"aliased model variables — the query would be vacuous: "
                f"{sorted(names & seen)[:4]}"
            )
        seen |= names
    return True


def solve(cons, goal, timeout_ms=120_000):
    """UNSAT ⇒ the goal is impossible ⇒ the property holds."""
    s = Solver()
    s.set("timeout", timeout_ms)
    for c in cons:
        s.add(c)
    s.add(goal)
    t0 = time.perf_counter()
    r = s.check()
    dt = time.perf_counter() - t0
    model = s.model() if r == sat else None
    return r, dt, model


# ===========================================================================
# The absorb mode's row-local constraint set, transcribed from blake3.rs.
#
# Every entry names the construct it transcribes. `drop` removes one by name —
# that is how the negative controls are built (discipline #1: a constraint that
# can be removed from the MODEL is falsifiable; one carried by a variable's
# sort is not, and is called out in the README's width census).
# ===========================================================================
def mode_algebra(cons, drop=()):
    """blake3.rs:1721-1780 — IS_BIT band, μ = MU_S+MU_A, exclusivity, MU_C,
    the boundary lock, and FIRST·END = 0."""
    v = {}
    # idx 814 (IS_BIT MU), idx 815..819 (IS_BIT MU_S/MU_A/FIRST/END).
    for name in ("MU", "MU_S", "MU_A", "FIRST", "END"):
        v[name] = bit(name, cons) if f"is_bit_{name}" not in drop else fe(name, cons)
    # MU_C is NOT independently IS_BIT-constrained: it is a free column pinned
    # only by `MU_C = MU_A − END`. That it lands in {0,1} is a THEOREM (P3.2),
    # and it matters because MU_C is used as a bus MULTIPLICITY.
    v["MU_C"] = fe("MU_C", cons)

    # All operands here are in {0,1} (or [0,p) for a dropped IS_BIT), so each
    # polynomial's integer value is far below p and ≡0 (mod p) ⟺ = 0. Where an
    # IS_BIT is dropped the value can be large, so the congruence form is used.
    def rel(expr, lo, hi, name):
        if name in drop:
            return
        cong0(cons, expr, lo, hi)

    rel(v["MU"] - v["MU_S"] - v["MU_A"], -2 * P, P, "mu_partition")          # idx 819
    rel(v["MU_S"] * v["MU_A"], 0, P * P, "mode_exclusive")                    # idx 820
    rel(v["MU_C"] + v["END"] - v["MU_A"], -2 * P, 2 * P, "mu_c_def")          # idx 821
    rel((v["FIRST"] + v["END"]) * (1 - v["MU_A"]), -P * P, 2 * P * P,
        "boundary_lock")                                                      # idx 822
    rel(v["FIRST"] * v["END"], 0, P * P, "no_zero_block_group")               # idx 823
    return v


def countdown(cons, v, drop=()):
    """blake3.rs:1786-1792 (the countdown) + the Zero and IsB20 lookups
    (blake3.rs:1463-1481)."""
    v["REMAINING"] = fe("REMAINING", cons)
    v["REM_DECR"] = fe("REM_DECR", cons)

    # 10f. Zero[REMAINING] -> END, gated MU_A. Both halves of the contract.
    if "zero_lookup" not in drop:
        if "zero_domain" not in drop:
            cons.append(Or(v["MU_A"] == 0, v["REMAINING"] < B20))
        cons.append(Or(v["MU_A"] == 0,
                       And(v["REMAINING"] == 0, v["END"] == 1),
                       And(v["REMAINING"] != 0, v["END"] == 0)))

    # 10g. IsB20[REM_DECR · 2^10], gated FIRST. Linearized: the field product
    # reduced mod p must land in [0, 2^20).
    if "isb20_cap" not in drop:
        prod = fe("_capval", cons)
        cong0(cons, ABSORB_CAP_SCALE * v["REM_DECR"] - prod,
              -P, ABSORB_CAP_SCALE * P)
        cons.append(Or(v["FIRST"] == 0, prod < B20))

    # idx 824. MU_C·(REM_DECR + 1 − REMAINING) = 0.
    if "countdown" not in drop:
        cons.append(Or(v["MU_C"] == 0,
                       (v["REM_DECR"] + 1 - v["REMAINING"]) % P == 0))
    return v


def framing(cons, v, drop=()):
    """blake3.rs:1854-1869 — the interior schedule (t = 0, block_len = 64,
    flags only on FIRST) plus the byte range checks that make those word
    equations mean what they say.

    ✓ VERIFIED that the 16 bytes of input words 24..27 ARE ByteAlu operands on
    every row where the mixing core is live: `run_flow` puts them at v[12..16]
    (blake3.rs:334) and round 0's G-calls 0..3 have d = 12,13,14,15, whose first
    use is `f.xor(g, 0, vd, a1)` (blake3.rs:350). The mixing core is gated
    `MU − END`, which is 1 on exactly the rows these constraints gate.
    """
    words = {}
    for wname, widx in (("t_lo", 24), ("t_hi", 25), ("block_len", 26), ("flags", 27)):
        bytes_ = []
        for b in range(4):
            c = fe(f"{wname}_b{b}", cons)
            # ByteAlu operand range check, live iff MU − END = 1.
            if "byte_range_framing" not in drop:
                cons.append(Or(v["MU"] - v["END"] != 1, c < BYTE))
            bytes_.append(c)
        words[wname] = bytes_
    v["framing_bytes"] = words

    def word(bs):
        return bs[0] + 256 * bs[1] + 65536 * bs[2] + 16777216 * bs[3]

    v["framing_word"] = {k: word(bs) for k, bs in words.items()}

    # idx 844..847: MU_C·(t_lo) = 0, MU_C·(t_hi) = 0, MU_C·(block_len − 64) = 0.
    for wname, want in (("t_lo", 0), ("t_hi", 0), ("block_len", 64)):
        if f"schedule_{wname}" in drop:
            continue
        cons.append(Or(v["MU_C"] == 0, (word(words[wname]) - want) % P == 0))
    # idx 847: (MU_C − FIRST)·flags = 0.
    if "schedule_flags" not in drop:
        cons.append(Or(v["MU_C"] - v["FIRST"] == 0,
                       word(words["flags"]) % P == 0))
    return v


def end_row_cv(cons, drop=()):
    """The END row's `h` bytes: pinned as WORDS by the chain receive
    (blake3.rs:1453-1457, `word_of_bytes(IN + 4i)`), pinned as BYTES only by the
    END-gated AreBytes at blake3.rs:1245-1251. Their bytes are then written to
    memory verbatim by the `cv_out` store (blake3.rs:1414-1424).

    This is the sharpest instance of the "assumes there are bytes" class: the
    mixing core that range-checks `h` everywhere else is gated OFF on this row.
    """
    a, b = [], []
    for i in range(32):
        ca = fe(f"cvA_b{i}", cons)
        cb = fe(f"cvB_b{i}", cons)
        if "arebytes_end_cv" not in drop:
            cons.append(ca < BYTE)
            cons.append(cb < BYTE)
        a.append(ca)
        b.append(cb)
    # The chain receive pins the 8 WORD expressions — identical for both
    # assignments, which is what "the bus delivered one value" means.
    for w in range(8):
        lo = 4 * w
        wa = a[lo] + 256 * a[lo + 1] + 65536 * a[lo + 2] + 16777216 * a[lo + 3]
        wb = b[lo] + 256 * b[lo + 1] + 65536 * b[lo + 2] + 16777216 * b[lo + 3]
        cong0(cons, wa - wb, -(2**25) * P, (2**25) * P)
    return a, b


def pointer_add(cons, tag, drop=(), offset=64):
    """`emit_add_pair` (templates.rs:334-374) as the chip applies it to
    `M_BASE + 64 = M_BASE_INCR` (blake3.rs:1796-1803): two committed carry bits
    and the two 32-bit limb identities, with the limbs packed from IsHalfword
    halfwords (blake3.rs:1486-1499).

    ⚠ `tag` is not cosmetic. Two z3 variables with the same name ARE the same
    variable, so a uniqueness query that builds two models must name them
    apart — otherwise it is trivially UNSAT and the check is vacuous.
    """
    base, incr = [], []
    for i in range(4):
        hb = fe(f"{tag}_mbase_h{i}", cons)
        hi_ = fe(f"{tag}_mincr_h{i}", cons)
        if "ishalf_mbase" not in drop:
            cons.append(hb < HALF)
            cons.append(hi_ < HALF)
        base.append(hb)
        incr.append(hi_)

    def lo(h):
        return h[0] + 65536 * h[1]

    def hi(h):
        return h[2] + 65536 * h[3]

    c0 = bit(f"{tag}_addpair_c0", cons)
    c1 = bit(f"{tag}_addpair_c1", cons)
    # ⚠ The quotient bound must hold with IsHalfword DROPPED too, or the model
    # silently re-imposes the bound it is meant to remove and the negative
    # control cannot flip. A free halfword pair reaches ~2^16·p, so ±2^17·p.
    span = (2**17) * P
    cong0(cons, lo(base) + offset - lo(incr) - c0 * W32, -span, span)
    cong0(cons, hi(base) + c0 - hi(incr) - c1 * W32, -span, span)
    return base, incr, (c0, c1)


# ===========================================================================
# P1 — compression equivalence, QF-BV. Byte columns are 8-bit bitvectors,
# which IS the AreBytes/ByteAlu range-check contract.
# ===========================================================================
WIDE = 48  # honest intermediates stay < 2^35


class Circuit:
    """The chip's mixing core, transcribed from `run_flow` + the eval
    constraints (blake3.rs:326-378, 1647-1716). A word is 4 free 8-bit BVs."""

    def __init__(self, tag):
        self.C = []
        self.tag = tag
        self.n = 0

    def _fresh(self, w=8):
        v = BitVec(f"{self.tag}_v{self.n}", w)
        self.n += 1
        return v

    def fresh_word(self):
        return [self._fresh(8) for _ in range(4)]

    def const_word(self, val):
        return [BitVecVal((val >> (8 * i)) & 0xFF, 8) for i in range(4)]

    def wval(self, word):
        acc = BitVecVal(0, WIDE)
        for i in range(4):
            acc = acc + ZeroExt(WIDE - 8, word[i]) * BitVecVal(1 << (8 * i), WIDE)
        return acc

    def hwval(self, blo, bhi):
        return ZeroExt(WIDE - 8, blo) + ZeroExt(WIDE - 8, bhi) * BitVecVal(256, WIDE)

    def word32(self, word):
        return Concat(word[3], word[2], word[1], word[0])

    def fresh_bit(self):
        v = self._fresh(8)
        self.C.append(Or(v == 0, v == 1))  # μ-gated IS_BIT, μ = 1 on a real row
        return v

    def xor(self, A, B):
        """ByteAlu[XOR] — output pinned and all three operands byte-ranged."""
        out = self.fresh_word()
        for i in range(4):
            self.C.append(out[i] == A[i] ^ B[i])
        return out

    def rotr16(self, A):
        return [A[2], A[3], A[0], A[1]]  # free byte relabel

    def rotr8(self, A):
        return [A[1], A[2], A[3], A[0]]  # free byte relabel

    def add2(self, A, B):
        """Expression carry: a + b − s = 2^32·carry, carry IS_BIT."""
        s = self.fresh_word()
        carry = self.fresh_bit()
        self.C.append(
            self.wval(A) + self.wval(B)
            == self.wval(s) + ZeroExt(WIDE - 8, carry) * BitVecVal(1 << 32, WIDE)
        )
        return s

    def add3(self, A, B, M):
        """Two summed committed carry bits (degree stays ≤ 3 after gating)."""
        s = self.fresh_word()
        c1 = self.fresh_bit()
        c2 = self.fresh_bit()
        csum = ZeroExt(WIDE - 8, c1) + ZeroExt(WIDE - 8, c2)
        self.C.append(
            self.wval(A) + self.wval(B) + self.wval(M)
            == self.wval(s) + csum * BitVecVal(1 << 32, WIDE)
        )
        return s

    def rot_shift(self, A, n, wrong_amount=False):
        """rotr12/rotr7 as inline μ-gated shift identities + halfword swap."""
        r = {12: 4, 7: 9}[n] + (1 if wrong_amount else 0)
        xlo = self.hwval(A[0], A[1])
        xhi = self.hwval(A[2], A[3])
        sll_lo, sllc_lo = self.fresh_word()[:2], self.fresh_word()[:2]
        sll_hi, sllc_hi = self.fresh_word()[:2], self.fresh_word()[:2]
        SLL_lo, SLLC_lo = self.hwval(*sll_lo), self.hwval(*sllc_lo)
        SLL_hi, SLLC_hi = self.hwval(*sll_hi), self.hwval(*sllc_hi)
        two_r = BitVecVal(1 << r, WIDE)
        two16 = BitVecVal(1 << 16, WIDE)
        self.C.append(xlo * two_r == SLLC_lo * two16 + SLL_lo)
        self.C.append(xhi * two_r == SLLC_hi * two16 + SLL_hi)
        Y = self.fresh_word()
        self.C.append(self.hwval(Y[0], Y[1]) == SLL_hi + SLLC_lo)
        self.C.append(self.hwval(Y[2], Y[3]) == SLL_lo + SLLC_hi)
        return Y

    def build_g(self, v, a, b, c, d, mx, my, bug=None):
        """One G, exactly the order `run_flow` emits (blake3.rs:349-365)."""
        b_op = c if bug == "swap_g_operand" else b
        a1 = self.add3(v[a], v[b_op], mx)
        x1 = self.xor(v[d], a1)
        d1 = self.rotr16(x1)
        c1 = self.add2(v[c], d1)
        x2 = self.xor(v[b], c1)
        b1 = self.rot_shift(x2, 12, wrong_amount=(bug == "rot_wrong_amount"))
        a2 = self.add3(a1, b1, my)
        x3 = self.xor(d1, a2)
        d2 = self.rotr8(x3)
        c2 = self.add2(c1, d2)
        x4 = self.xor(b1, c2)
        b2 = self.rot_shift(x4, 7)
        v[a], v[b], v[c], v[d] = a2, b2, c2, d2


# ===========================================================================
# P1a — per-round WIRING, G abstracted. Six queries, one per round.
# ===========================================================================
def check_round_wiring(r, bug=None, timeout_ms=60_000):
    """Round r: does the chip hand each G the slots and message words the spec
    names for round r? G is an uninterpreted function, identical on both sides,
    so this decides the PLUMBING and nothing else."""
    bv32 = BitVecSort(32)
    sig = [bv32] * 6
    Gf = [Function(f"G{r}_{o}", *sig, bv32) for o in range(4)]

    state = [BitVec(f"s{r}_{i}", 32) for i in range(16)]
    msg = [BitVec(f"m{r}_{i}", 32) for i in range(16)]

    def apply_g(v, a, b, c, d, mx, my):
        args = (v[a], v[b], v[c], v[d], mx, my)
        outs = [f(*args) for f in Gf]
        v[a], v[b], v[c], v[d] = outs

    # --- chip side: `run_flow`'s own schedule recurrence, transcribed ---
    sched = list(range(16))
    for _ in range(r):
        prev = sched
        sched = [prev[p] for p in ref.MSG_PERMUTATION]
    if bug == "swap_sched_pair":
        sched[0], sched[1] = sched[1], sched[0]
    chip = list(state)
    for j, (a, b, c, d) in enumerate(ref.G_INDICES):
        ia, ib, ic, idd = (a, b, c, d)
        if bug == "swap_state_slot" and j == 0:
            ib, ic = ic, ib
        apply_g(chip, ia, ib, ic, idd, msg[sched[2 * j]], msg[sched[2 * j + 1]])

    # --- reference side: permute the message r times, then consume in order ---
    rmsg = list(msg)
    for _ in range(r):
        rmsg = ref.permute(rmsg)
    want = list(state)
    for j, (a, b, c, d) in enumerate(ref.G_INDICES):
        apply_g(want, a, b, c, d, rmsg[2 * j], rmsg[2 * j + 1])

    return solve([], Or(*[chip[i] != want[i] for i in range(16)]), timeout_ms)


# ===========================================================================
# P1b — the G circuit's internals against the spec G, free inputs.
# ===========================================================================
def check_g(bug=None, timeout_ms=600_000):
    cir = Circuit("g" + (f"_{bug}" if bug else ""))
    va, vb, vc, vd = (cir.fresh_word(), cir.fresh_word(),
                      cir.fresh_word(), cir.fresh_word())
    mx, my = cir.fresh_word(), cir.fresh_word()
    v = [None] * 16
    v[0], v[1], v[2], v[3] = va, vb, vc, vd
    cir.build_g(v, 0, 1, 2, 3, mx, my, bug=bug)
    rv = [cir.word32(va), cir.word32(vb), cir.word32(vc), cir.word32(vd)]
    ref.g(ref.BvOps, rv, 0, 1, 2, 3, cir.word32(mx), cir.word32(my))
    goal = Or(*[cir.word32(v[i]) != rv[i] for i in range(4)])
    return solve(cir.C, goal, timeout_ms)


# ===========================================================================
# P1c/P1d — the absorb framing's input feed, and the feed-forward that becomes
# the chain payload. Both are round-free, so they are exact and cheap.
# ===========================================================================
def check_input_feed(bug=None, timeout_ms=60_000):
    """The 16-word state entering round 0 on an absorbing row, under the values
    P2 proves the framing columns hold: v[8..12] = IV, v[12..16] = (0,0,64,fl).
    """
    cir = Circuit("feed" + (f"_{bug}" if bug else ""))
    h = [cir.fresh_word() for _ in range(8)]
    flags = cir.fresh_word()
    iv = list(ref.IV)
    if bug == "wrong_iv":
        iv[0] ^= 1
    bl = 64 ^ (1 if bug == "wrong_block_len" else 0)
    chip = (
        [cir.word32(w) for w in h]
        + [BitVecVal(x, 32) for x in iv[:4]]
        + [BitVecVal(0, 32), BitVecVal(0, 32), BitVecVal(bl, 32), cir.word32(flags)]
    )
    want = ref.initial_state(
        ref.BvOps, [cir.word32(w) for w in h],
        BitVecVal(0, 32), BitVecVal(0, 32), BitVecVal(64, 32), cir.word32(flags),
    )
    return solve(cir.C, Or(*[chip[i] != want[i] for i in range(16)]), timeout_ms)


def check_feed_forward(bug=None, timeout_ms=60_000):
    """out[i] = v[i]^v[i+8], out[i+8] = v[i+8]^h[i]; the chain carries out[0..8]
    (blake3.rs:1451, `chain_values(..., cols::OUT)` takes 8 words)."""
    cir = Circuit("ff" + (f"_{bug}" if bug else ""))
    h = [cir.fresh_word() for _ in range(8)]
    v = [cir.fresh_word() for _ in range(16)]
    out = []
    for i in range(8):
        out.append(cir.xor(v[i], v[i + 8]))
    for i in range(8):
        out.append(cir.xor(v[i + 8], h[i]))
    if bug == "drop_ff_xor":
        out[0] = cir.fresh_word()
    if bug == "chain_carries_wrong_half":
        payload = [cir.word32(out[8 + i]) for i in range(8)]
    else:
        payload = [cir.word32(out[i]) for i in range(8)]
    want = ref.feed_forward(ref.BvOps,
                            [cir.word32(w) for w in v],
                            [cir.word32(w) for w in h])[:8]
    return solve(cir.C, Or(*[payload[i] != want[i] for i in range(8)]), timeout_ms)


# ===========================================================================
# Board
# ===========================================================================
class Board:
    def __init__(self):
        self.rows = []
        self.ok = True

    def record(self, prop, name, want, res, dt, model=None, note=""):
        got = str(res)
        good = got == want
        self.ok &= good
        self.rows.append((prop, name, want, got, dt, good, note))
        mark = "OK " if good else "!! "
        print(f"  {mark}{name:<46s} {got:<8s} (want {want:<5s}) {dt:7.2f}s  {note}")
        if model is not None and (VERBOSE or (want == "unsat" and got == "sat")):
            print(f"      witness: {self.decode(model)}")
        return good

    @staticmethod
    def decode(model):
        keep = {}
        for d in model.decls():
            n = d.name()
            if n.startswith("_q") or n.startswith("_cap"):
                continue
            keep[n] = model[d]
        items = sorted(keep.items())[:14]
        return ", ".join(f"{k}={v}" for k, v in items)


def main():
    import z3

    print("=" * 78)
    print("BLAKE3 CHAINED-ABSORB MODE — z3 gate")
    print(f"z3 {z3.get_version_string()}   |   chip: prover/src/tables/blake3.rs")
    print("=" * 78)
    print(CONTRACTS)
    b = Board()

    # ---------------------------------------------------------------- P1 ---
    print("\n=== P1  compression equivalence under the absorb framing (QF-BV) ===")
    print("    One round at a time. See BINDING SCOPE RULE.")
    for r in range(6):
        res, dt, m = check_round_wiring(r)
        b.record("P1a", f"round {r} wiring (schedule = permute^{r})", "unsat", res, dt, m)
    res, dt, m = check_g()
    b.record("P1b", "G circuit vs spec G, free inputs", "unsat", res, dt, m,
             "covers all 48 G instances")
    res, dt, m = check_input_feed()
    b.record("P1c", "input feed: IV | t=0 | block_len=64 | flags", "unsat", res, dt, m)
    res, dt, m = check_feed_forward()
    b.record("P1d", "feed-forward + chain payload = CV (8 words)", "unsat", res, dt, m)

    print("\n  negative controls (must be SAT — a green board with a vacuous")
    print("  encoding is the fail-open mode discipline #1 exists to close):")
    for bug in ("swap_sched_pair", "swap_state_slot"):
        res, dt, m = check_round_wiring(3, bug=bug)
        b.record("P1a-neg", f"round 3, {bug}", "sat", res, dt)
    for bug in ("rot_wrong_amount", "swap_g_operand"):
        res, dt, m = check_g(bug=bug)
        b.record("P1b-neg", f"G, {bug}", "sat", res, dt)
    for bug in ("wrong_iv", "wrong_block_len"):
        res, dt, m = check_input_feed(bug=bug)
        b.record("P1c-neg", f"feed, {bug}", "sat", res, dt)
    for bug in ("drop_ff_xor", "chain_carries_wrong_half"):
        res, dt, m = check_feed_forward(bug=bug)
        b.record("P1d-neg", f"feed-forward, {bug}", "sat", res, dt)

    # ---------------------------------------------------------------- P3 ---
    # (P3 before P2: the mode algebra is what P2's gates are written over.)
    print("\n=== P3  mode gating: selectors, Σ modes = μ, no cross-mode bleed (field) ===")
    cons = []
    v = mode_algebra(cons)
    LEGAL = Or(
        And(v["MU"] == 0, v["MU_S"] == 0, v["MU_A"] == 0,   # padding
            v["FIRST"] == 0, v["END"] == 0, v["MU_C"] == 0),
        And(v["MU"] == 1, v["MU_S"] == 1, v["MU_A"] == 0,   # single compression
            v["FIRST"] == 0, v["END"] == 0, v["MU_C"] == 0),
        And(v["MU"] == 1, v["MU_S"] == 0, v["MU_A"] == 1,   # absorb, FIRST
            v["FIRST"] == 1, v["END"] == 0, v["MU_C"] == 1),
        And(v["MU"] == 1, v["MU_S"] == 0, v["MU_A"] == 1,   # absorb, interior
            v["FIRST"] == 0, v["END"] == 0, v["MU_C"] == 1),
        And(v["MU"] == 1, v["MU_S"] == 0, v["MU_A"] == 1,   # absorb, END
            v["FIRST"] == 0, v["END"] == 1, v["MU_C"] == 0),
    )
    res, dt, m = solve(cons, z3.Not(LEGAL))
    b.record("P3.1", "exactly 5 row shapes exist (no sixth)", "unsat", res, dt, m)

    for expr, nm in (
        (v["MU_C"], "MU_C"),
        (v["MU"] - v["END"], "MU − END (mixing-core gate)"),
        (v["MU_A"] - v["FIRST"], "MU_A − FIRST (chain receive)"),
        (v["MU_S"] + v["FIRST"], "MU_S + FIRST (x10 / cv read)"),
    ):
        res, dt, m = solve(cons, Or(expr < 0, expr > 1))
        b.record("P3.2", f"multiplicity {nm} ∈ {{0,1}}", "unsat", res, dt, m)

    res, dt, m = solve(cons, And(v["MU_S"] == 1,
                                 Or(v["MU_A"] != 0, v["FIRST"] != 0,
                                    v["END"] != 0, v["MU_C"] != 0)))
    b.record("P3.3", "single row: every absorb multiplicity is 0", "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["MU_A"] == 1, v["MU_S"] != 0))
    b.record("P3.4", "absorb row: MU_S-gated interactions are off", "unsat", res, dt, m)

    print("\n  negative controls:")
    for dropped, goal_name, goal in (
        ("boundary_lock", "padding row mints END (a free cv_out write)",
         lambda vv: And(vv["MU"] == 0, vv["END"] == 1)),
        ("boundary_lock", "single row mints FIRST (a second Ecall receive)",
         lambda vv: And(vv["MU_S"] == 1, vv["FIRST"] == 1)),
        ("no_zero_block_group", "zero-block group (FIRST = END = 1)",
         lambda vv: And(vv["FIRST"] == 1, vv["END"] == 1)),
        ("mu_c_def", "MU_C free ⇒ negative multiplicity",
         lambda vv: vv["MU_C"] > 1),
    ):
        c2 = []
        v2 = mode_algebra(c2, drop=(dropped,))
        res, dt, m = solve(c2, goal(v2))
        b.record("P3-neg", f"drop {dropped}: {goal_name}", "sat", res, dt, m)

    # The documented rationale for MU_S·MU_A = 0 is stronger than the algebra
    # needs; IS_BIT(MU) + μ = MU_S + MU_A already forbids MU_S = MU_A = 1.
    c3 = []
    v3 = mode_algebra(c3, drop=("mode_exclusive",))
    res, dt, m = solve(c3, And(v3["MU_S"] == 1, v3["MU_A"] == 1))
    b.record("P3.5", "MU_S·MU_A = 0 is IMPLIED by IS_BIT(MU) + partition",
             "unsat", res, dt, m, "informational — see README")

    # ---------------------------------------------------------------- P4 ---
    print("\n=== P4  countdown / END row-local logic + the boundary lock (field) ===")
    cons = []
    v = mode_algebra(cons)
    countdown(cons, v)

    res, dt, m = solve(cons, And(v["MU_A"] == 1, v["FIRST"] == 1,
                                 Or(v["REMAINING"] < 1,
                                    v["REMAINING"] > ABSORB_MAX_BLOCKS)))
    b.record("P4.1", "FIRST row: 1 ≤ REMAINING ≤ 1024 (the cap, in circuit)",
             "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["MU_A"] == 1, v["END"] == 1, v["REMAINING"] != 0))
    b.record("P4.2", "no early END (END ⇒ REMAINING = 0)", "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["MU_A"] == 1, v["REMAINING"] == 0, v["END"] != 1))
    b.record("P4.3", "no late END (REMAINING = 0 ⇒ END)", "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["END"] == 1,
                                 Or(v["MU_A"] != 1, v["MU_C"] != 0)))
    b.record("P4.4", "END row is inert: MU_A = 1, MU_C = 0 (no send, no core)",
             "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["MU_A"] == 1, v["END"] == 0,
                                 Or(v["MU_C"] != 1,
                                    (v["REM_DECR"] + 1 - v["REMAINING"]) % P != 0)))
    b.record("P4.5", "compressing row: MU_C = 1 and REM_DECR = REMAINING − 1",
             "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["FIRST"] == 1, v["REM_DECR"] >= ABSORB_MAX_BLOCKS))
    b.record("P4.6", "cap cannot be wrapped: REM_DECR < 2^10 on FIRST",
             "unsat", res, dt, m)

    print("\n  negative controls:")
    for dropped, goal_name, goal in (
        ("zero_lookup", "END is free ⇒ end a group early at REMAINING = 7",
         lambda vv: And(vv["MU_A"] == 1, vv["END"] == 1, vv["REMAINING"] == 7)),
        ("isb20_cap", "no cap ⇒ a group of 2^19 blocks",
         lambda vv: And(vv["MU_A"] == 1, vv["FIRST"] == 1,
                        vv["REMAINING"] == 2**19)),
        ("countdown", "no countdown ⇒ REM_DECR unrelated to REMAINING",
         lambda vv: And(vv["MU_A"] == 1, vv["END"] == 0,
                        vv["REMAINING"] == 5, vv["REM_DECR"] == 99)),
        ("zero_domain", "★ Zero's DOMAIN bound dropped ⇒ the cap wraps mod p",
         lambda vv: And(vv["MU_A"] == 1, vv["FIRST"] == 1,
                        vv["REM_DECR"] >= ABSORB_MAX_BLOCKS)),
    ):
        c2 = []
        v2 = mode_algebra(c2)
        countdown(c2, v2, drop=(dropped,))
        res, dt, m = solve(c2, goal(v2))
        b.record("P4-neg", f"drop {dropped}: {goal_name}", "sat", res, dt, m)

    # ---------------------------------------------------------------- P2 ---
    print("\n=== P2  flags schedule / interior framing (field) ===")
    cons = []
    v = mode_algebra(cons)
    countdown(cons, v)
    framing(cons, v)
    fw = v["framing_word"]
    fb = v["framing_bytes"]

    interior = And(v["MU_A"] == 1, v["END"] == 0, v["FIRST"] == 0)
    res, dt, m = solve(cons, And(interior,
                                 Or(*[fb["t_lo"][i] != 0 for i in range(4)],
                                    *[fb["t_hi"][i] != 0 for i in range(4)])))
    b.record("P2.1", "interior row: counter bytes are all zero", "unsat", res, dt, m)
    res, dt, m = solve(cons, And(interior,
                                 Or(fb["block_len"][0] != 64,
                                    *[fb["block_len"][i] != 0 for i in (1, 2, 3)])))
    b.record("P2.2", "interior row: block_len bytes are exactly (64,0,0,0)",
             "unsat", res, dt, m)
    res, dt, m = solve(cons, And(interior,
                                 Or(*[fb["flags"][i] != 0 for i in range(4)])))
    b.record("P2.3", "★ interior row cannot carry ANY flag byte "
                     "(the forged-shorter-message class)", "unsat", res, dt, m)
    res, dt, m = solve(cons, And(v["MU_A"] == 1, v["FIRST"] == 1, v["END"] == 0,
                                 fw["flags"] >= W32))
    b.record("P2.4", "FIRST row: flags word < 2^32 (fits the one-word column)",
             "unsat", res, dt, m)

    print("\n  negative controls:")
    c2 = []
    v2 = mode_algebra(c2)
    countdown(c2, v2)
    framing(c2, v2, drop=("schedule_flags",))
    res, dt, m = solve(c2, And(v2["MU_A"] == 1, v2["END"] == 0, v2["FIRST"] == 0,
                               v2["framing_bytes"]["flags"][0] == 0x0A))
    b.record("P2-neg", "drop the flags gate: interior block carries CHUNK_END|ROOT",
             "sat", res, dt, m, "= a digest for a PREFIX of the message")
    c2 = []
    v2 = mode_algebra(c2)
    countdown(c2, v2)
    framing(c2, v2, drop=("schedule_block_len",))
    res, dt, m = solve(c2, And(v2["MU_A"] == 1, v2["END"] == 0,
                               v2["framing_bytes"]["block_len"][0] == 7))
    b.record("P2-neg", "drop block_len = 64: absorb a block under a short framing",
             "sat", res, dt, m)

    # ---------------------------------------------------------------- P5 ---
    print("\n=== P5  byte/dword width assumptions — 'it assumes there are bytes' (field) ===")
    # 5a. The END row's cv bytes: pinned as words by the bus, as bytes only by
    #     the END-gated AreBytes, then written to memory verbatim.
    cons = []
    a, bb = end_row_cv(cons)
    distinct_names(a, bb)
    res, dt, m = solve(cons, Or(*[a[i] != bb[i] for i in range(32)]))
    b.record("P5.1", "END-row cv_out bytes are UNIQUE given the chain words",
             "unsat", res, dt, m)
    # Non-vacuity of the uniqueness model itself: the constraints must admit
    # SOME assignment, or the UNSAT above would be meaningless.
    res, dt, _ = solve(cons, a[0] == 200)
    b.record("P5.1", "…and that model is consistent (non-vacuity)", "sat", res, dt)
    c2 = []
    a2, b2 = end_row_cv(c2, drop=("arebytes_end_cv",))
    res, dt, m = solve(c2, Or(*[a2[i] != b2[i] for i in range(32)]))
    b.record("P5-neg", "★ drop the END-row AreBytes: a non-canonical cv_out "
                       "writes a 'byte' ≥ 256 to memory", "sat", res, dt, m)

    # 5b. The framing words mean what they say only because their bytes are
    #     ByteAlu operands. Same query as P2.2, with the sort-carried bound
    #     removed — the one gap a full board of green controls can hide.
    c2 = []
    v2 = mode_algebra(c2)
    countdown(c2, v2)
    framing(c2, v2, drop=("byte_range_framing",))
    res, dt, m = solve(c2, And(v2["MU_A"] == 1, v2["END"] == 0,
                               v2["framing_bytes"]["block_len"][0] != 64))
    b.record("P5-neg", "★ drop the byte range on block_len: 64 has other "
                       "decompositions mod p", "sat", res, dt, m)

    # 5c. Pointer limbs: IsHalfword is what makes `M_BASE + 64` a function.
    for dropped, want, label in (
        ((), "unsat", "M_BASE_INCR is unique given M_BASE (IsHalfword present)"),
        (("ishalf_mbase",), "sat",
         "drop IsHalfword: the next block's address is ambiguous"),
    ):
        both = []
        base, incr, _ = pointer_add(both, "A", drop=dropped)
        base2, incr2, _ = pointer_add(both, "B", drop=dropped)
        distinct_names(base + incr, base2 + incr2)
        both += [base[i] == base2[i] for i in range(4)]
        res, dt, m = solve(both, Or(*[incr[i] != incr2[i] for i in range(4)]))
        b.record("P5.2" if want == "unsat" else "P5-neg", label, want, res, dt, m)

    # ------------------------------------------------ positive control ------
    print("\n=== POSITIVE CONTROL (non-vacuity: an honest row must be admissible) ===")
    cons = []
    v = mode_algebra(cons)
    countdown(cons, v)
    framing(cons, v)
    honest = And(v["MU"] == 1, v["MU_S"] == 0, v["MU_A"] == 1, v["FIRST"] == 1,
                 v["END"] == 0, v["MU_C"] == 1, v["REMAINING"] == 3,
                 v["REM_DECR"] == 2,
                 v["framing_bytes"]["block_len"][0] == 64,
                 v["framing_bytes"]["flags"][0] == 0x01)
    res, dt, m = solve(cons, honest)
    b.record("POS", "honest FIRST row of a 3-block group is SAT", "sat", res, dt)
    honest_end = And(v["MU"] == 1, v["MU_A"] == 1, v["END"] == 1, v["FIRST"] == 0,
                     v["MU_C"] == 0, v["REMAINING"] == 0)
    res, dt, m = solve(cons, honest_end)
    b.record("POS", "honest END row is SAT", "sat", res, dt)

    # ------------------------------------------------------------ verdict ---
    print("\n" + "=" * 78)
    total = sum(r[4] for r in b.rows)
    bad = [r for r in b.rows if not r[5]]
    print(f"VERDICT: {'PASS' if b.ok else 'FAIL'}   "
          f"({len(b.rows) - len(bad)}/{len(b.rows)} queries as expected, "
          f"{total:.1f}s total)")
    if bad:
        print("\nUNEXPECTED:")
        for prop, name, want, got, dt, _, _ in bad:
            print(f"  [{prop}] {name}: got {got}, wanted {want}")
        print("\n  'unknown' = z3 timeout, NOT a finding. 'sat' where 'unsat'")
        print("  was wanted IS a finding — decode the witness above.")
    print("=" * 78)
    sys.exit(0 if b.ok else 1)


if __name__ == "__main__":
    main()
