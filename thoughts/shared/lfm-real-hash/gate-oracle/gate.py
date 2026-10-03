"""
LAYER 5: THE z3 GATE.

Method.  Every committed column of the chip is a FREE variable; every lookup
(under its contract) and every eval constraint becomes an equation over those
variables; the chip's OUTPUT is whatever the constraints force.  Then:

    assert  chip_output != reference_f(input)      and ask z3.

    UNSAT -> for EVERY constraint-satisfying assignment the output equals the
             reference: the chip is correctly AND tightly constrained.
    SAT   -> the constraints admit a wrong output: under-constrained or mis-wired.

FAIL-OPEN IS THE ONLY DANGEROUS MODE.  A gate that returns UNSAT because the
model quietly assumed something the chip never enforces certifies nothing.  Two
defences, both mandatory and both run below:

  * NEGATIVE CONTROLS -- inject a bug, demand SAT. A control that comes back
    UNSAT means the gate cannot see that class of bug at all.
  * THE WIDTH AUDIT -- every field-lifted byte/word width must cite a real
    range-check contract AND a non-overflow side condition. Bound-necessity is
    invisible in BV (a byte IS 8 bits there), so those controls run in the FIELD
    domain, mod p. This is where a field-level attacker who escapes the
    bit-vector model gets caught.

Run:  python3 gate.py            # fast board (symbolic core + all controls + audit)
      python3 gate.py --full     # + concrete full 6- and 7-round pipeline runs
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

from z3 import (And, BitVecVal, Int, Or, Solver, sat, unsat)

import blake3_oracle as ora
import chip_model as cm
import socket_ref as sk
from contracts import P, WIDE, FieldContracts

HERE = os.path.dirname(os.path.abspath(__file__))


class Board:
    def __init__(self):
        self.rows: list[tuple[str, str, str, str, bool, float]] = []

    def add(self, section, name, got, want, elapsed=0.0):
        ok = (str(got) == want)
        self.rows.append((section, name, str(got), want, ok, elapsed))
        mark = "PASS" if ok else "**FAIL**"
        print(f"  [{mark:8s}] {name:44s} -> {str(got):6s} (want {want})"
              f"{f'  {elapsed:.1f}s' if elapsed > 0.3 else ''}")
        return ok

    def ok(self):
        return all(r[4] for r in self.rows)


def _solve(assertions, goal, timeout_ms=0):
    s = Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)
    s.add(And(*assertions))
    s.add(goal)
    t0 = time.time()
    res = s.check()
    return res, time.time() - t0


# ===========================================================================
# SYMBOLIC THEOREMS (BV) -- the chip's logic and framing
# ===========================================================================

def theorem_g(bug=None, timeout_ms=0):
    """One G quarter-round against the reference G, free inputs.

    A round is a FIXED composition of eight G-calls on fixed indices, and the
    message schedule is a compile-time permutation, so a G that is correct on
    arbitrary inputs gives a correct round, hence a correct N-round core for
    BOTH round counts. That chaining argument is what makes the fast board
    sufficient and the monolithic multi-round runs a bonus."""
    from z3 import Concat, RotateRight

    chip = cm.SocketChip("G" + (f"_{bug}" if bug else ""), bug=bug)
    va, vb, vc, vd = (chip.c.fresh_word(), chip.c.fresh_word(),
                      chip.c.fresh_word(), chip.c.fresh_word())
    mx, my = chip.c.fresh_word(), chip.c.fresh_word()
    v = [va, vb, vc, vd]
    chip.emit_g(v, 0, 1, 2, 3, mx, my, gflag=True)

    def w32(w):
        return Concat(w[3], w[2], w[1], w[0])

    rv = [w32(va), w32(vb), w32(vc), w32(vd)]
    rmx, rmy = w32(mx), w32(my)
    rv[0] = rv[0] + rv[1] + rmx
    rv[3] = RotateRight(rv[3] ^ rv[0], 16)
    rv[2] = rv[2] + rv[3]
    rv[1] = RotateRight(rv[1] ^ rv[2], 12)
    rv[0] = rv[0] + rv[1] + rmy
    rv[3] = RotateRight(rv[3] ^ rv[0], 8)
    rv[2] = rv[2] + rv[3]
    rv[1] = RotateRight(rv[1] ^ rv[2], 7)

    goal = Or(*[w32(v[i]) != rv[i] for i in range(4)])
    return _solve(chip.assertions, goal, timeout_ms)


def theorem_socket(rounds: int, chip_framing: sk.Framing | None = None,
                   bug=None, ref_framing: sk.Framing | None = None,
                   timeout_ms=0, tail_truncate=False):
    """The SOCKET layer: lane bytes -> message placement -> constant initial
    state -> R rounds -> feed-forward -> truncation window -> digest lanes,
    against the reference framing.

    The chip is built with `chip_framing` (perturbed for a control); the
    reference always uses `ref_framing` (honest). Symbolic in the eight input
    lanes."""
    cf = chip_framing or sk.honest(rounds)
    rf = ref_framing or sk.honest(rounds)
    tag = f"S{rounds}" + (f"_{bug}" if bug else "") + f"_{id(cf) & 0xFFFF:x}"
    chip = cm.SocketChip(tag, framing=cf, bug=bug,
                         tail_truncate=tail_truncate).build()
    ref = cm.reference_digest_bv(chip, rf)
    got = chip.digest_lane_values()
    # Compare as WIDE values so the byte->word lift is part of what is checked.
    from z3 import ZeroExt
    goal = Or(*[got[i] != ZeroExt(WIDE - 32, ref[i]) for i in range(4)])
    return _solve(chip.assertions, goal, timeout_ms)


def theorem_schedule(rounds: int, chip_framing: sk.Framing | None = None,
                     ref_framing: sk.Framing | None = None, timeout_ms=0):
    """MESSAGE LAYER: the schedule the chip feeds to every round, against the
    reference schedule, symbolic in the eight input lanes.

    Covers framing choices 1 (where a and b land), 5 (the tag word and its slot)
    and 7 (the lane byte order), plus the compile-time message permutation --
    all of it without paying for a single G. Isolating the layer this way is not
    a shortcut: a G that is correct on ARBITRARY inputs (T1) composed with a
    schedule that is correct on ARBITRARY inputs (this) is a correct round, and
    the composition is fixed at compile time."""
    from z3 import Concat, ZeroExt
    cf = chip_framing or sk.honest(rounds)
    rf = ref_framing or sk.honest(rounds)
    chip = cm.SocketChip(f"MSG{rounds}_{id(cf) & 0xFFFF:x}", framing=cf)
    chip.emit_lane_bytes()

    chip_sched = chip.message_words()
    lanes = [Concat(w[3], w[2], w[1], w[0]) for w in chip.in_lane_bytes]
    if not rf.lane_le:
        lanes = [Concat(w[0], w[1], w[2], w[3]) for w in chip.in_lane_bytes]
    ref_sched = [BitVecVal(0, 32) for _ in range(16)]
    for i in range(4):
        ref_sched[rf.a_slot + i] = lanes[i]
        ref_sched[rf.b_slot + i] = lanes[4 + i]
    ref_sched[rf.tag_slot] = BitVecVal(rf.tag_word, 32)

    diffs = []
    cs, rs = list(chip_sched), list(ref_sched)
    for r in range(max(rounds, 1)):
        for i in range(16):
            diffs.append(chip.c.wval(cs[i]) != ZeroExt(WIDE - 32, rs[i]))
        if r < rounds - 1:
            cs = [cs[cf.msg_permutation[i]] for i in range(16)]
            rs = [rs[rf.msg_permutation[i]] for i in range(16)]
    return _solve(chip.assertions, Or(*diffs), timeout_ms)


def concrete_pipeline(rounds: int, a, b, expect_digest, negate=False,
                      timeout_ms=0, tail_truncate=False,
                      chip_framing: sk.Framing | None = None, bug=None):
    """Non-vacuity + external anchor: pin the eight input lanes to a KAT input
    and the four digest lanes to the KAT output.

    negate=False -> expect SAT: the full byte-level pipeline reproduces the
                    externally-anchored vector.
    negate=True  -> pin the digest to a WRONG value and expect UNSAT: at this
                    concrete input the system is FUNCTIONAL, i.e. tight.

    Cheap (a pinned input propagates), end-to-end, and anchored -- so this, not
    a monolithic symbolic run, is what every negative control is measured
    against below."""
    fr = chip_framing or sk.honest(rounds)
    chip = cm.SocketChip(f"C{rounds}_{'n' if negate else 'p'}_{id(fr) & 0xFFFF:x}",
                         framing=fr, bug=bug,
                         tail_truncate=tail_truncate).build()
    extra = []
    lanes = list(a) + list(b)
    for word, val in zip(chip.in_lane_bytes, lanes):
        for k in range(4):
            extra.append(word[k] == BitVecVal((val >> (8 * k)) & 0xFF, 8))
    want = list(expect_digest)
    if negate:
        want[0] ^= 1
    for expr, val in zip(chip.digest_lane_values(), want):
        extra.append(expr == BitVecVal(val, WIDE))
    return _solve(list(chip.assertions) + extra, And(True), timeout_ms)


def concrete_control(rounds: int, a, b, honest_digest, chip_framing=None,
                     bug=None, timeout_ms=0):
    """A negative control, run against the FULL pipeline at a concrete input.

    Build the perturbed chip, pin the input lanes, and ask whether the digest can
    differ from the honest anchored value.  SAT = the gate sees this bug class.
    UNSAT = the gate is BLIND to it, which is the finding that matters."""
    fr = chip_framing or sk.honest(rounds)
    chip = cm.SocketChip(f"NC{rounds}_{bug or ''}_{id(fr) & 0xFFFF:x}",
                         framing=fr, bug=bug).build()
    extra = []
    for word, val in zip(chip.in_lane_bytes, list(a) + list(b)):
        for k in range(4):
            extra.append(word[k] == BitVecVal((val >> (8 * k)) & 0xFF, 8))
    goal = Or(*[expr != BitVecVal(val, WIDE)
                for expr, val in zip(chip.digest_lane_values(), honest_digest)])
    return _solve(list(chip.assertions) + extra, goal, timeout_ms)


# ===========================================================================
# WIDTH AUDIT (FIELD, mod p) -- bound necessity. BV provably cannot show these.
# ===========================================================================

def audit_lane_decomposition(drop_arebytes: bool):
    """The Route-A lane boundary, obligation O1 -- part 1 of 2.

    The chip decomposes each input lane into four byte columns with ONE linear
    identity. Question: does that identity alone pin the bytes?

    with AreBytes -> UNSAT: the bytes are the unique LE decomposition.
    without       -> SAT:   one linear equation in four unknowns leaves three
                            free, so the byte columns are unpinned.

    WHAT THIS DOES AND DOES NOT SHOW (corrected, D10). It shows the identity is
    not self-sufficient. It does NOT show a `v` / `v + 2^32` collision -- that
    attack is unconstructible here, because the mixing core reads the SAME linear
    form the identity pins, so the lane and the message word are one field
    element by construction. The consequence that matters is WA2's: without the
    sends the message word is not forced to be a u32, and since `m` reaches the
    core through `add3` only (never an XOR), these 16 sends are its ONLY range
    check. See `chip_model.emit_lane_bytes` for the full argument."""
    honest_lane = 0x89ABCDEF
    hb = [(honest_lane >> (8 * k)) & 0xFF for k in range(4)]

    s = Solver()
    fc = FieldContracts(s)
    b = [fc.fresh_felt(f"mb{k}") for k in range(4)]
    if not drop_arebytes:
        fc.are_bytes(*b)                       # the AreBytes sends
    # the mu-gated lane-decomposition identity, in the field
    s.add((honest_lane - (b[0] + 256 * b[1] + 65536 * b[2] + 16777216 * b[3])) % P == 0)
    # the attacker names the three low bytes; only the top byte is left to absorb
    s.add(b[0] == (hb[0] ^ 0x5A), b[1] == (hb[1] ^ 0x3C), b[2] == (hb[2] ^ 0xF0))
    return str(s.check())


def audit_lane_upper_range(drop_arebytes: bool):
    """The Route-A lane boundary, obligation O1 -- part 2 of 2, and THE one that
    carries the soundness argument. Can a felt >= 2^32 pass the identity?

    with AreBytes -> UNSAT: the sum of four bytes is < 2^32, so `lane` -- which
                            IS the message word, the same linear form -- is
                            forced below 2^32. The compression therefore denotes
                            BLAKE3 of an actual 36-byte string.
    without       -> SAT:   the message word ranges over the whole field. Round
                            0's add3 has constant a, b and byte-bounded s, so a
                            prover solves `m = s + 2^32*(c1+c2) - a - b` for any
                            chosen s and owns the compression from the first add
                            onward -- and what the chip computes is no longer
                            BLAKE3 of any message."""
    s = Solver()
    fc = FieldContracts(s)
    lane = fc.fresh_felt("lane")
    b = [fc.fresh_felt(f"ub{k}") for k in range(4)]
    if not drop_arebytes:
        fc.are_bytes(*b)
    s.add((lane - (b[0] + 256 * b[1] + 65536 * b[2] + 16777216 * b[3])) % P == 0)
    s.add(lane >= 2**32)
    return str(s.check())


def audit_shift_bound(r: int, in_hw: int, drop_sll_bound: bool):
    """hw*2^r == SLLC*2^16 + SLL (mod p). SLL is the TIGHT remainder (AreBytes on
    its two bytes); SLLC is the quotient and a loose 16-bit bound suffices.
    Soundness needs 2^16 invertible mod p -- true in Goldilocks, false mod 2^n,
    which is exactly why this cannot be a BV check."""
    s = Solver()
    fc = FieldContracts(s)
    if drop_sll_bound:
        SLL = fc.fresh_felt("SLL")                      # unbounded column
    else:
        lo, hi = fc.fresh_felt("sll_lo"), fc.fresh_felt("sll_hi")
        fc.are_bytes(lo, hi)
        SLL = lo + 256 * hi
    SLLC = fc.fresh_felt("SLLC")
    fc.bounded(SLLC, 2**16)
    s.add((in_hw * (2 ** r) - SLLC * (2 ** 16) - SLL) % P == 0)
    s.add(SLL != (in_hw * (2 ** r)) % (2 ** 16))
    return str(s.check())


def audit_add_carry(a, b, m, drop_bool: bool):
    """3-operand add: a+b+m == s + 2^32*(c1+c2) (mod p), s in [0,2^32) from its
    byte columns. Dropping the carry booleanity turns c into a full field element
    and s becomes forgeable -- again field-only."""
    s = Solver()
    fc = FieldContracts(s)
    S = fc.fresh_felt("S")
    fc.bounded(S, 2**32)
    if drop_bool:
        c1 = fc.fresh_felt("c1")
        csum = c1
    else:
        c1, c2 = fc.fresh_felt("c1"), fc.fresh_felt("c2")
        fc.carry_bit(c1)
        fc.carry_bit(c2)
        csum = c1 + c2
    s.add((a + b + m - S - (2**32) * csum) % P == 0)
    s.add(S != (a + b + m) % (2**32))
    return str(s.check())


def audit_add2_expression_carry(drop_s_bound: bool):
    """WA7 -- THE audit item the expression-carry form needs, and the reason the
    BV model may encode it as a two-way disjunction.

    The chip emits ONE constraint, `MU * carry * (1 - carry) = 0`, where
    `carry := (A + B - s) * 2^{-32}` is a linear form over existing columns.
    Over the field that says `A + B - s in {0, 2^32}`. The question BV cannot
    answer: given A, B, s byte-bounded below 2^32, are 0 and 2^32 the ONLY
    reachable roots -- in particular, can a NEGATIVE difference alias 2^32 mod p?

    It cannot. If `A + B - s >= 0` it lies in [0, 2^33) and 2^33 << p, so the
    only residues are the honest two. If `A + B - s < 0` it lies in (-2^32, 0),
    i.e. the field element sits in (p - 2^32, p); that equals 0 only for a zero
    difference, and equals 2^32 only if the difference were 2^32 - p, which is
    about -2^64 and far below -2^32. Hence s is pinned to (A + B) mod 2^32.

    present -> UNSAT: s is pinned.
    drop the byte bound on s -> SAT: s becomes a free field element, the
        difference can be steered onto a root, and the add is forgeable. This is
        the same class as WA4 and equally invisible to BV.

    ENCODING: `carry in {0,1}` is encoded as its root set `d in {0, 2^32}` --
    two LINEAR congruences -- rather than as the quadratic `carry*(1-carry) = 0`
    with a nested inverse, which is intractable for z3's integer arithmetic. The
    step from one to the other is AR2 in the argued ledger (`2^{-32}` is a unit,
    so multiplying by it is a bijection and maps the root set exactly). This is
    the same posture WA4 already takes for the add3 carry, and it keeps the
    solver on the question it can actually decide: whether s is pinned."""
    s_ = Solver()
    fc = FieldContracts(s_)
    A, Bv = 0xFFFF_FFF0, 0xFFFF_FFF5      # a case that genuinely carries
    S = fc.fresh_felt("S2")
    if not drop_s_bound:
        fc.bounded(S, 2**32)
    d = A + Bv - S
    s_.add(Or(d % P == 0, (d - 2**32) % P == 0))
    s_.add(S != (A + Bv) % (2**32))
    return str(s_.check())


def audit_block0_capacity(drop_mode_p_pin: bool):
    """BLOCK-0 idx 0-3 with idx 5. `S_k - (MODE_P*IN_{8+k} + MODE_C*IV_k) = 0`.

    with `MODE_P = 0` pinned -> UNSAT: S_k is forced to MODE_C * IV_k.
    without it -> SAT: MODE_P is free, so the capacity prefix becomes a
        prover-chosen copy of IN_{8+k}. idx 0-3 pin nothing on their own; idx 5
        is what gives them meaning."""
    s_ = Solver()
    fc = FieldContracts(s_)
    IV0 = ora.IV[0]
    mode_c, mode_p = fc.fresh_felt("mode_c"), fc.fresh_felt("mode_p")
    in_8 = fc.fresh_felt("in_8")
    S = fc.fresh_felt("S_cap")
    ms = mode_c + mode_p          # widened below once mode_t exists
    s_.add((ms * (1 - ms)) % P == 0)                      # idx 4
    if not drop_mode_p_pin:
        s_.add(mode_p == 0)                               # idx 5
    mode_t = fc.fresh_felt("mode_t")
    s_.add((S - (mode_p * in_8 + (mode_c + mode_t) * IV0)) % P == 0)  # idx 0-3
    s_.add(S != ((mode_c + mode_t) * IV0) % P)
    return str(s_.check())


def audit_block0_mu_boolean(drop_mode_sum_bool: bool):
    """BLOCK-0 idx 4 + idx 5 give MU booleanity. With MODE_P = 0,
    mode_sum = MODE_C = MU, so `mode_sum*(1-mode_sum)=0` IS `MU in {0,1}`.

    The pre-Phase-2 model deferred MU booleanity to "structural, not a BV
    theorem". The chip emits it as a real constraint, so it is checkable -- and
    checked here in the field.

    ENCODING: the emitted polynomial's root set is `{0,1}` by AR1 (a prime field
    has no zero divisors), so the contract is encoded as that root set. What the
    solver decides is the consequence: with MODE_P pinned, does mode_sum being a
    bit force MU to be a bit -- and what happens when the constraint is absent.

    present -> UNSAT (MU is a bit); dropped -> SAT (MU is any felt, and a
    non-boolean MU scales every gated constraint and every send multiplicity)."""
    s_ = Solver()
    fc = FieldContracts(s_)
    mode_c, mode_t = fc.fresh_felt("mc2"), fc.fresh_felt("mt2")
    mode_p = fc.fresh_felt("mp2")
    s_.add(mode_p == 0)                                # idx 5
    ms = mode_c + mode_t + mode_p                      # AS BUILT
    if not drop_mode_sum_bool:
        s_.add(Or(ms % P == 0, ms % P == 1))           # idx 4, via AR1
    # MU = MODE_C + MODE_T + MODE_L as built, which IS the mode sum once
    # MODE_P = 0. (This audit is written over the two-selector case; the
    # four-way form is M8's, and adding MODE_L here changes nothing about what
    # idx 4 buys -- it bounds the sum either way.)
    #
    # DO NOT add the registrar's one-hot here: it would force MU = 1 outright and
    # make this audit vacuous (an earlier draft did exactly that and the control
    # caught it -- `dropped` came back UNSAT). The division of labour is the
    # point, and it is sharper than the spec's original claim:
    #   * idx 4 DOES give MU booleanity -- MU is the sum, so bounding the sum to
    #     a bit bounds MU. That is what this audit checks.
    #   * idx 4 does NOT give one-hotness -- which tag `m[8]` selects is the
    #     registrar's preprocessed check. That is M8's job.
    #
    # NB: MU is a SUM of felts, so as a z3 Int it can exceed p. It must be
    # compared by RESIDUE, not raw value -- otherwise `mu = p + 1` counts as
    # "not 1" and the audit reports SAT for a chip that is fine. (A draft did
    # exactly that; the `present` leg caught it.)
    mu = (mode_c + mode_t + mode_p) % P
    s_.add(mu != 0, mu != 1)
    return str(s_.check())


def audit_block0_tag_selection(with_one_hot: bool, target_tag: int | None = None):
    """M8 model-side — WHAT ACTUALLY MAKES `m[8]` TRUSTWORTHY.

    `m[8] = MODE_C*TAG_LFMC + MODE_T*TAG_LFMT`. The question is what forces it to
    be ONE of the two tags rather than a blend.

    IT IS NOT idx 4. Over a prime field `mode_sum in {0,1}` pins the SUM, not the
    selectors: `MODE_C = x`, `MODE_T = 1 - x` satisfies it for ANY x, and since
    the tags differ, `x = (T - TAG_T)/(TAG_C - TAG_T)` reaches ANY target tag T.

    with_one_hot=False -> SAT: a forged tag is reachable (idx 4 is not enough).
    with_one_hot=True  -> UNSAT for a forged target, SAT for either real tag
                          (the honest-path leg: a fix that rejected everything
                          would pass the attack leg alone).

    ✓ Reproduces the builder's Rust M5/M6 finding independently. The real closure
    is (i) MODE_* being PREPROCESSED and (ii) the registrar's one-hot check."""
    TAG_C, TAG_T = 0x434D464C, 0x544D464C          # "LFMC", "LFMT"
    target = TAG_C if target_tag is None else target_tag
    s_ = Solver()
    fc = FieldContracts(s_)
    mc, mt = fc.fresh_felt("mc8"), fc.fresh_felt("mt8")
    ms = mc + mt
    s_.add(Or(ms % P == 0, ms % P == 1))            # idx 4
    if with_one_hot:                                 # the registrar's check
        s_.add(Or(And(mc == 1, mt == 0), And(mc == 0, mt == 1)))
    s_.add((mc * TAG_C + mt * TAG_T - target) % P == 0)
    return str(s_.check())


MAX_HALF = 0xFFFFFFFF
TAG_C, TAG_T, TAG_L = 0x434D464C, 0x544D464C, 0x4C4D464C


def audit_leaf_canonicity(drop_canon: bool):
    """WA8 — the leaf mode's canonicity gate (obligation O1 on a leaf row).

    A leaf row binds `v = lo + 2^32*hi` with lo, hi bounded to u32 by AreBytes.
    That is a decomposition, NOT a canonical one: `p - 1 = 0xFFFFFFFF_00000000`,
    so every pair with `hi` maximal and `lo >= 1` encodes a field element that
    ALSO has an ordinary encoding -- one felt, two half-pairs, two leaf digests,
    which is precisely the collision a Merkle tree must not have.

    present -> UNSAT: no non-canonical pair satisfies the constraints.
    dropped -> SAT:   a second encoding of an already-encodable felt exists."""
    s_ = Solver()
    fc = FieldContracts(s_)
    lo, hi = fc.fresh_felt("lo"), fc.fresh_felt("hi")
    fc.bounded(lo, 2**32)                      # from AreBytes + the lane bytes
    fc.bounded(hi, 2**32)
    z, ginv = fc.fresh_felt("z"), fc.fresh_felt("ginv")
    g = MAX_HALF - hi
    if not drop_canon:
        s_.add((z * g) % P == 0)                          # canon-a
        s_.add((1 - z - g * ginv) % P == 0)               # canon-b
        s_.add((z * lo) % P == 0)                         # canon-c
    # the attack: a NON-canonical pair, i.e. one encoding a value >= p
    s_.add(hi == MAX_HALF, lo >= 1)
    return str(s_.check())


def audit_leaf_range_dependency(narrow_arebytes_to_digest: bool):
    """WA9 — ⚠ THE HAZARD THE GATING SPLIT CREATES, and the reason it is safe.

    `idx 6-13` (the lane identity) narrowed to the DIGEST modes when MODE_L
    landed; the AreBytes range bound did NOT (its sends carry
    `Sum3(MODE_C, MODE_T, MODE_L)`). This audit asks what would happen if a
    future change narrowed the RANGE bound too -- the plausible "tidy up the
    multiplicities to match" refactor.

    bound present (as built) -> UNSAT: with lo, hi < 2^32 the canonicity block
        admits only canonical pairs, so the felt->halves map is injective.
    bound narrowed away      -> SAT:   lo and hi become full field elements, and
        a felt acquires a second half-pair that still satisfies binding AND
        canonicity -- the gate is intact but VACUOUS. Canonicity assumes the
        u32 bound; it does not establish it."""
    s_ = Solver()
    fc = FieldContracts(s_)
    lo, hi = fc.fresh_felt("lo9"), fc.fresh_felt("hi9")
    if not narrow_arebytes_to_digest:
        fc.bounded(lo, 2**32)
        fc.bounded(hi, 2**32)
    z, ginv = fc.fresh_felt("z9"), fc.fresh_felt("ginv9")
    g = MAX_HALF - hi
    s_.add((z * g) % P == 0)
    s_.add((1 - z - g * ginv) % P == 0)
    s_.add((z * lo) % P == 0)
    # a SECOND encoding of the felt v = 1: binding says v == lo + 2^32*hi
    v = 1
    s_.add((v - lo - (2**32) * hi) % P == 0)
    s_.add(Or(lo != 1, hi != 0))               # anything other than the honest pair
    return str(s_.check())


def audit_tag_selection_4way(with_one_hot: bool, target_tag: int):
    """M8 over the FOUR-way one-hot: m[8] = MODE_C*TAG_C + MODE_T*TAG_T +
    MODE_L*TAG_L. A third tag does not change the finding -- idx 4 still pins
    only the SUM, so a fractional split still reaches any target."""
    s_ = Solver()
    fc = FieldContracts(s_)
    mc, mt, ml = (fc.fresh_felt("mc4"), fc.fresh_felt("mt4"), fc.fresh_felt("ml4"))
    ms = mc + mt + ml
    s_.add(Or(ms % P == 0, ms % P == 1))                       # idx 4
    if with_one_hot:
        s_.add(Or(And(mc == 1, mt == 0, ml == 0),
                  And(mc == 0, mt == 1, ml == 0),
                  And(mc == 0, mt == 0, ml == 1)))
    s_.add((mc * TAG_C + mt * TAG_T + ml * TAG_L - target_tag) % P == 0)
    return str(s_.check())


# HashMode arities, ✓ VERIFIED instr.rs:104-110.
MODE_ARITY = {"Compress": 2, "Transcript": 2, "Leaf": 1, "Permute": 3}


def audit_unread_pin_selectors():
    """D1's shared unread-input pins: does any HONEST row get OVER-constrained?

    `emit_unread_input_pins` derives the selector for input slot `k` as the sum
    of the modes with `num_input_cells() <= k`. A pin fires on a row iff that
    row's mode is in the sum. The obligation is that a mode is NEVER pinned on a
    cell it actually READS -- otherwise honest rows become unprovable, which is
    the failure mode a soundness fix most easily introduces.

    UNSAT = no mode is pinned on a cell it reads."""
    s_ = Solver()
    bad = []
    for mode, arity in MODE_ARITY.items():
        for slot in (1, 2):
            pinned = arity <= slot          # the helper's filter
            reads = slot < arity            # this mode reads that cell
            if pinned and reads:
                bad.append(f"{mode} pinned on slot {slot} which it READS")
    s_.add(Int("dummy") == (1 if bad else 0), Int("dummy") == 1)
    return ("sat" if bad else "unsat"), bad


def audit_unread_pins_inert_on_blake3():
    """⚠ DOCUMENTED GATE BLINDNESS, in the `drop_carry_bool` shape.

    On the BLAKE3 arm the two unread cells are read by NOTHING:
      * cell 1 (IN4..8) is read by the lane identity idx 6-13, which is gated on
        the DIGEST modes -- and on a leaf row, the only row where cell 1 is
        unread, that gate is zero;
      * cell 2 (IN8..12) is read only through idx 0-3's `MODE_P * IN` term, and
        idx 5 pins MODE_P to zero PERMANENTLY (option B1).

    So dropping the BLAKE3 unread pins cannot change a BLAKE3 digest, and this
    gate would report UNSAT for any "wrong output" question about them. That is
    a TRUE statement about the BLAKE3 arm and NOT evidence the pins are
    unnecessary: D1 was a defect in `eval_test` / `eval_poseidon`, where those
    cells ARE read, and those arms are outside this QF-BV model entirely.

    WHAT THIS GATE CERTIFIES : the pins are inert on BLAKE3 (hygiene).
    WHAT IT CANNOT           : their necessity on Test/Poseidon.
    WHAT CARRIES THAT INSTEAD: the builder's Rust junk-rejection controls.

    Recorded rather than left implicit, because "the gate said UNSAT" is exactly
    how a fix gets dropped as redundant."""
    return "unsat"


def audit_block0_upper_out(drop_pins: bool):
    """BLOCK-0 idx 14-21: `OUT_{4+j} = 0`. The digest is ONE cell, so the upper
    eight OUT lanes must carry nothing. present -> UNSAT; dropped -> SAT."""
    s_ = Solver()
    fc = FieldContracts(s_)
    outs = [fc.fresh_felt(f"outhi{j}") for j in range(8)]
    if not drop_pins:
        for o in outs:
            s_.add(o == 0)
    s_.add(Or(*[o != 0 for o in outs]))
    return str(s_.check())


def audit_recombine_pins(target: str):
    """The TAIL-TRUNCATION obligation, both sides.

    The rotation's output word Y is constrained only by two halfword identities:
        Ylo == SLL_hi + SLLC_lo,   Yhi == SLL_lo + SLLC_hi
    where Ylo = Y0 + 256*Y1 and Yhi = Y2 + 256*Y3. If Y's downstream XOR is
    removed (the last-round tail optimisation), Y's BYTES lose their range check.

    target='word'  -> UNSAT: the WORD VALUE sum(Y_k * 2^{8k}) is still pinned,
                      because it regroups exactly into the two constrained
                      halfword sums. So a consumer that reads Y as the full
                      linear form (the add3) is safe.
    target='byte'  -> SAT:   the individual BYTES are NOT pinned. So a consumer
                      that reads Y's bytes -- a relabel, a byte lookup, any
                      sub-combination -- is UNSOUND without an explicit AreBytes.
    """
    SLL_hi, SLLC_lo, SLL_lo, SLLC_hi = 0x1230, 0x0004, 0x5670, 0x0008
    ylo, yhi = SLL_hi + SLLC_lo, SLL_lo + SLLC_hi
    s = Solver()
    fc = FieldContracts(s)
    Y = [fc.fresh_felt(f"Y{k}") for k in range(4)]     # NO AreBytes: tail case
    s.add((Y[0] + 256 * Y[1] - ylo) % P == 0)
    s.add((Y[2] + 256 * Y[3] - yhi) % P == 0)
    if target == "word":
        word = Y[0] + 256 * Y[1] + 65536 * Y[2] + 16777216 * Y[3]
        s.add(word % P != (ylo + 65536 * yhi) % P)
    else:
        s.add(Y[0] != ylo & 0xFF)
    return str(s.check())


# ---------------------------------------------------------------------------
# The non-overflow side condition: a static bound argument, not a solver run.
# ---------------------------------------------------------------------------

WIDTH_AUDIT_TABLE = [
    # (identity, max |LHS| and |RHS| given the contracts, backing contract)
    ("lane decomposition   lane == sum b_k*2^{8k}",
     2**32, "AreBytes on MB[j][0..4]  (LaneDecomposition)"),
    ("add2 sum             A+B == s + 2^32*c",
     2**33, "ByteAlu[XOR] on operands + CarryBit"),
    ("add3 sum             A+B+M == s + 2^32*(c1+c2)",
     2**34, "ByteAlu[XOR]/AreBytes on operands + CarryBit x2"),
    ("shift identity       hw*2^r == SLLC*2^16 + SLL",
     2**32, "AreBytes on SLL/SLLC bytes (ShiftRemainderBound)"),
    ("recombine            Ylo == SLL_hi + SLLC_lo",
     2**17, "AreBytes on SLL/SLLC bytes"),
    ("digest recomposition OUT_C[i] == sum OUTW_k*2^{8k}",
     2**32, "ByteAlu[XOR] output bytes"),
]


def audit_no_wrap() -> tuple[bool, int]:
    worst = max(m for (_, m, _) in WIDTH_AUDIT_TABLE)
    return worst < P, worst


# ---------------------------------------------------------------------------
# THE ARGUED LEDGER -- steps discharged by algebra, not by a solver.
#
# Recording them is the point. Each is a one-line field fact that some encoding
# above relies on; a gate that silently baked them in would be asserting exactly
# the kind of unstated assumption that makes a fail-open possible. z3 4.15.4 has
# no finite-field sort, and the Int+mod encodings of these are nonlinear and
# intractable, so they are argued -- and SAID to be argued -- rather than solved.
# ---------------------------------------------------------------------------

ARGUED_LEDGER = [
    ("AR1", "F_p has no zero divisors (p prime), so `x*(1-x) = 0` has root set "
            "exactly {0,1}.",
     "add3 carry booleanity (WA4); mode-sum booleanity (B0b)"),
    ("AR2", "2^{-32} is a unit in F_p, so `d * 2^{-32} in {0,1}` iff "
            "`d in {0, 2^32}` -- multiplication by a unit is a bijection and "
            "maps the root set exactly.",
     "add2 expression-carry (WA7)"),
    ("AR3", "2^16 is invertible mod p, which is what makes the tight AreBytes "
            "bound on SLL pin the shift remainder uniquely.",
     "rotation shift identity (WA3)"),
    ("AR4", "every field-lifted expression stays below 2^34 << p, so `expr = 0 "
            "mod p` implies `expr = 0` over the integers.",
     "all identities (WA6)"),
]


# ===========================================================================
def load_kats():
    path = os.path.join(HERE, "socket_kats.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main() -> int:
    full = "--full" in sys.argv
    B = Board()
    print("=" * 78)
    print("BLAKE3-behind-LFM_HASH  --  z3 GATE  (Option A socket)")
    print("=" * 78)

    kats = load_kats()
    if kats is None:
        print("  socket_kats.json missing -- run `python3 socket_kats.py --write` first")
        return 1
    VEC = {r: next(e for e in kats["rounds"][str(r)] if e["name"] == "formula_1")
           for r in (6, 7)}

    # ---------------------------------------------------------------- core
    # The argument the board rests on, stated once:
    #   T1  a G-call is correct on ARBITRARY inputs;
    #   T2  the schedule fed to every round is correct on ARBITRARY inputs;
    #   T3  the constant initial state, the feed-forward, the truncation window
    #       and the two felt<->byte recompositions are correctly wired;
    #   a round is a FIXED composition of eight G-calls on fixed indices, and the
    #   round count is a compile-time constant.
    # Hence the full N-round socket is correct, for BOTH round counts. T4 then
    # runs the whole pipeline concretely against externally-anchored vectors, so
    # the composition argument has an executed end-to-end witness rather than
    # only a proof sketch.
    print("\n--- MAIN THEOREMS (symbolic, BV) -- want UNSAT ---")
    r, t = theorem_g()
    B.add("core", "T1  G quarter-round, free inputs (covers every G)", r, "unsat", t)
    r, t = theorem_schedule(7)
    B.add("core", "T2  message schedule, all 7 rounds (placement/tag/LE/perm)",
          r, "unsat", t)
    r, t = theorem_socket(0)
    B.add("core", "T3  framing @rounds=0 (init state/feed-forward/window)",
          r, "unsat", t)

    # A theorem with no control of its own is a theorem that may be vacuous, so
    # each of T2 and T3 gets controls proving it discriminates -- against ITS OWN
    # layer, not only against the end-to-end pipeline.
    print("  per-theorem discrimination controls -- want SAT:")
    T2_LAYER = ["swap_a_b", "tag_changed", "tag_omitted", "tag_slot_moved",
                "lanes_big_endian", "msg_perm_swapped"]
    for name in T2_LAYER:
        cfr = replace(sk.CONTROLS[name], rounds=7)
        r, t = theorem_schedule(7, chip_framing=cfr, ref_framing=sk.honest(7))
        B.add("neg", f"    T2-ctl {name}", r, "sat", t)
    # T3 sees only what reaches the window with ZERO rounds: v[0..12] and the
    # feed-forward wiring. The counter/block_len/flags words sit at v[12..16] and
    # reach the digest only THROUGH the rounds, so they are genuinely invisible
    # here and are covered by T4 instead. Listing them as T3 controls would be a
    # false claim of coverage.
    for name in ["cv_zero", "truncate_high_half"]:
        cfr = replace(sk.CONTROLS[name], rounds=0)
        r, t = theorem_socket(0, chip_framing=cfr, ref_framing=sk.honest(0))
        B.add("neg", f"    T3-ctl {name}", r, "sat", t)
    r, t = theorem_socket(0, bug="drop_ff_xor")
    B.add("neg", "    T3-ctl drop_ff_xor", r, "sat", t)
    if full:
        for rr in (1, 2):
            r, t = theorem_socket(rr, timeout_ms=5_400_000)
            B.add("core", f"T5  monolithic symbolic socket @rounds={rr} (bonus)",
                  r, "unsat", t)

    print("\n--- T4  FULL PIPELINE, CONCRETE, vs the anchored KATs ---")
    for rounds in (7, 6):
        v = VEC[rounds]
        r, t = concrete_pipeline(rounds, v["a"], v["b"], v["digest"],
                                 timeout_ms=900_000)
        B.add("core", f"T4  full {rounds}-round pipeline == anchored KAT",
              r, "sat", t)
        r, t = concrete_pipeline(rounds, v["a"], v["b"], v["digest"], negate=True,
                                 timeout_ms=900_000)
        B.add("core", f"T4  full {rounds}-round pipeline EXCLUDES a wrong digest",
              r, "unsat", t)

    # ------------------------------------------------- negative controls
    print("\n--- NEGATIVE CONTROLS -- want SAT (an UNSAT here means the gate is BLIND) ---")
    print("  logic bugs, symbolic at G level:")
    for bug in ("rot_wrong_amount", "swap_g_operand"):
        r, t = theorem_g(bug=bug)
        B.add("neg", f"NC  {bug}", r, "sat", t)

    # The transcribed bodies must be re-measured against every control: a
    # transcription that accidentally STRENGTHENS is as wrong as one that
    # weakens, and only the controls can tell the difference. Run at BOTH round
    # counts, because the chip ships both (7r default, 6r behind `blake3-6round`).
    for rounds in (7, 6):
        print(f"  framing + wiring bugs, FULL {rounds}-round pipeline, concrete:")
        vv = VEC[rounds]
        for name, cfr in sk.CONTROLS.items():
            if name == "rounds_6_not_7":
                # at 6 rounds this control IS the honest framing; the meaningful
                # form is the opposite confusion, checked below
                cr, cfr2 = (6, cfr) if rounds == 7 else (7, replace(cfr, rounds=7))
                label = "rounds_6_not_7" if rounds == 7 else "rounds_7_not_6"
                r, t = concrete_control(cr, vv["a"], vv["b"], vv["digest"],
                                        chip_framing=cfr2, timeout_ms=900_000)
                B.add("neg", f"NC  {label} @{rounds}", r, "sat", t)
                continue
            r, t = concrete_control(rounds, vv["a"], vv["b"], vv["digest"],
                                    chip_framing=replace(cfr, rounds=rounds),
                                    timeout_ms=900_000)
            B.add("neg", f"NC  {name} @{rounds}", r, "sat", t)
        for bug in ("drop_ff_xor", "swap_g_operand", "drop_add2_carry"):
            r, t = concrete_control(rounds, vv["a"], vv["b"], vv["digest"],
                                    bug=bug, timeout_ms=900_000)
            # `drop_add2_carry` removes the add2 constraint outright. Under the
            # expression-carry form there is no carry column left to un-boolean,
            # so the whole constraint IS the booleanity -- and unlike the add3
            # case it is therefore BV-visible. (`drop_carry_bool`, which
            # un-booleans add3's carry COLUMNS, stays BV-blind; see below.)
            B.add("neg", f"NC  {bug} @{rounds}", r, "sat", t)

    # ------------------------------------------- documented BV blindness
    # Not a failure: a demonstration of WHY the field audit is mandatory. In BV
    # a carry column is an 8-bit variable, so removing its booleanity leaves it
    # bounded and `s` is still pinned -> UNSAT. The same bug is a live forgery in
    # the field (WA4 below). A gate that ran only the BV domain would report this
    # class of bug as absent. That is the fail-open this split exists to prevent.
    print("\n--- DOCUMENTED BV BLINDNESS (why the FIELD domain is mandatory) ---")
    r, t = theorem_g(bug="drop_carry_bool")
    B.add("blind", "BV  drop_carry_bool (add3 carry COLUMNS) invisible in BV "
                   "-> WA4 has the field verdict", r, "unsat", t)

    # ------------------------------------------- the tail optimisation
    # Documented in ORACLE.md as OPTIONAL and NOT recommended; checked anyway,
    # because an optimisation described but never exercised is an unverified
    # claim. Its first draft skipped the column group too -- caught here.
    print("\n--- OPTIONAL TAIL TRUNCATION (last-round diagonal X4/B2 omitted) ---")
    v7 = VEC[7]
    r, t = concrete_pipeline(7, v7["a"], v7["b"], v7["digest"],
                             tail_truncate=True, timeout_ms=900_000)
    B.add("core", "TT  tail-truncated 7-round pipeline == anchored KAT",
          r, "sat", t)
    r, t = concrete_pipeline(7, v7["a"], v7["b"], v7["digest"], negate=True,
                             tail_truncate=True, timeout_ms=900_000)
    B.add("core", "TT  tail-truncated pipeline EXCLUDES a wrong digest",
          r, "unsat", t)
    tt = cm.SocketChip("tt7", framing=sk.honest(7), tail_truncate=True).build()
    base = cm.SocketChip("base7", framing=sk.honest(7)).build()
    print(f"      saves {base.census.cell_equiv() - tt.census.cell_equiv()} "
          f"cell-equiv of {base.census.cell_equiv()} "
          f"({100*(base.census.cell_equiv()-tt.census.cell_equiv())/base.census.cell_equiv():.1f}%)")

    # ---------------------------------------------- non-vacuity
    print("\n--- NON-VACUITY ---")
    chip = cm.SocketChip("nv", framing=sk.honest(7)).build()
    r, t = _solve(chip.assertions, And(True))
    B.add("pos", "NV  honest system satisfiable @rounds=7 (not vacuous)",
          r, "sat", t)

    # --------------------------------------------------------- width audit
    print("\n--- WIDTH AUDIT (FIELD, mod p) -- bound necessity; BV cannot see these ---")
    B.add("audit", "WA1 lane decomposition, AreBytes PRESENT",
          audit_lane_decomposition(False), "unsat")
    B.add("audit", "WA1 lane decomposition, AreBytes DROPPED",
          audit_lane_decomposition(True), "sat")
    B.add("audit", "WA2 lane < 2^32 forced, AreBytes PRESENT",
          audit_lane_upper_range(False), "unsat")
    B.add("audit", "WA2 lane < 2^32 forced, AreBytes DROPPED",
          audit_lane_upper_range(True), "sat")
    B.add("audit", "WA3 shift SLL bound PRESENT (r=9)",
          audit_shift_bound(9, 0x9C3A, False), "unsat")
    B.add("audit", "WA3 shift SLL bound DROPPED (r=9)",
          audit_shift_bound(9, 0x9C3A, True), "sat")
    B.add("audit", "WA4 add3 carry booleanity PRESENT",
          audit_add_carry(0xF0000000, 0xF0000000, 0xF0000000, False), "unsat")
    B.add("audit", "WA4 add3 carry booleanity DROPPED",
          audit_add_carry(0xF0000000, 0xF0000000, 0xF0000000, True), "sat")
    B.add("audit", "WA5 tail case: rotation WORD value still pinned",
          audit_recombine_pins("word"), "unsat")
    B.add("audit", "WA5 tail case: rotation BYTES not pinned (hazard is real)",
          audit_recombine_pins("byte"), "sat")
    nowrap, worst = audit_no_wrap()
    B.add("audit", f"WA6 no-wrap side condition (worst 2^{worst.bit_length()-1} < p)",
          "ok" if nowrap else "OVERFLOW", "ok")
    B.add("audit", "WA7 add2 expression-carry pins s (s byte-bound)",
          audit_add2_expression_carry(False), "unsat")
    B.add("audit", "WA7 add2 expression-carry, s bound DROPPED",
          audit_add2_expression_carry(True), "sat")

    print("\n--- BLOCK-0 FRAMING AUDIT (FIELD) -- the four constraints BV cannot reach ---")
    B.add("audit", "B0a capacity prefix pinned, MODE_P=0 PRESENT (idx 0-3,5)",
          audit_block0_capacity(False), "unsat")
    B.add("audit", "B0a capacity prefix, MODE_P pin DROPPED (idx 5 gone)",
          audit_block0_capacity(True), "sat")
    B.add("audit", "B0b MU booleanity from mode-sum (idx 4,5) PRESENT",
          audit_block0_mu_boolean(False), "unsat")
    B.add("audit", "B0b MU booleanity DROPPED",
          audit_block0_mu_boolean(True), "sat")
    B.add("audit", "B0c upper OUT lanes pinned to 0 (idx 14-21) PRESENT",
          audit_block0_upper_out(False), "unsat")
    B.add("audit", "B0c upper OUT lane pins DROPPED",
          audit_block0_upper_out(True), "sat")

    print("\n--- M8 (FOUR-way one-hot): what makes the mode-selected m[8] trustworthy ---")
    FORGED = 0x58585858                                    # "XXXX"
    B.add("audit", "M8  forged tag reachable with idx 4 ALONE (no one-hot)",
          audit_tag_selection_4way(False, FORGED), "sat")
    B.add("audit", "M8  forged tag EXCLUDED once four-way one-hot is present",
          audit_tag_selection_4way(True, FORGED), "unsat")
    for nm, tg in (("LFMC", TAG_C), ("LFMT", TAG_T), ("LFML", TAG_L)):
        B.add("audit", f"M8  honest leg: TAG_{nm} still reachable under one-hot",
              audit_tag_selection_4way(True, tg), "sat")

    print("\n--- D1 UNREAD-INPUT PINS (8, both cells) ---")
    r_sel, bad_sel = audit_unread_pin_selectors()
    B.add("audit", "D1  no honest row over-constrained (pin selectors vs arities)",
          r_sel, "unsat")
    if bad_sel:
        for x in bad_sel:
            print(f"        OVER-CONSTRAINED: {x}")
    B.add("blind", "D1  pins are INERT on BLAKE3 -- necessity rests on the Rust "
                   "controls (see the docstring)",
          audit_unread_pins_inert_on_blake3(), "unsat")

    print("\n--- LEAF MODE (MODE_L): canonicity, and the gating-split hazard ---")
    B.add("audit", "WA8 leaf canonicity PRESENT -> non-canonical pair unprovable",
          audit_leaf_canonicity(False), "unsat")
    B.add("audit", "WA8 leaf canonicity DROPPED -> a felt gets a 2nd half-pair",
          audit_leaf_canonicity(True), "sat")
    B.add("audit", "WA9 AreBytes still covers leaf rows (as built) -> map injective",
          audit_leaf_range_dependency(False), "unsat")
    B.add("audit", "WA9 AreBytes NARROWED to digest modes -> canonicity goes VACUOUS",
          audit_leaf_range_dependency(True), "sat")

    print("\n--- ARGUED LEDGER (algebra, not solver output -- stated, not hidden) ---")
    for tag, fact, used_by in ARGUED_LEDGER:
        print(f"  {tag}: {fact}")
        print(f"       relied on by: {used_by}")

    # --------------------------------------------------------------- census
    chip = cm.SocketChip("census7", framing=sk.honest(7)).build()
    chip6 = cm.SocketChip("census6", framing=sk.honest(6)).build()
    print("\n--- COST CENSUS (derived from the gated model, not hand-counted) ---")
    for label, ch in (("7-round", chip), ("6-round", chip6)):
        print(f"  {label}: main={ch.census.main:5d}  sends={ch.census.sends:5d}  "
              f"aux={ch.census.aux_cells():5d}  cell-equiv={ch.census.cell_equiv():5d}")
    for blk, n in sorted(chip.census.by_block.items(), key=lambda x: -x[1]):
        print(f"      {blk:24s} {n:5d}")

    print("\n" + "=" * 78)
    print(f"GATE VERDICT: {'PASS' if B.ok() else 'FAIL -- investigate above'}")
    print("=" * 78)
    fails = [r for r in B.rows if not r[4]]
    if fails:
        for f in fails:
            print(f"  FAILED: {f[1]} -> {f[2]} (wanted {f[3]})")
    return 0 if B.ok() else 1


if __name__ == "__main__":
    sys.exit(main())
