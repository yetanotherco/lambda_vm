"""
Transcription audit of `blake3-chip/z3_blake_verify.py` — an EXECUTABLE
regression suite for GATE-TRANSCRIPTION-AUDIT.md.

Two transcriptions are under test, and only one direction is dangerous.

  (a) blake3-oracle/blake3_ref.py  ->  the gate's `bref_*` BV reference.
      If these diverge, every UNSAT the gate reports proves the chip matches
      the WRONG function.

  (b) blake3-chip/DESIGN.md        ->  the gate's `build_g/build_round/
      build_compress` circuit model.
      A model WEAKER than the designed chip yields a spurious SAT — a false
      alarm, safe.  A model STRONGER yields UNSAT where the real object is
      forgeable — false assurance, and no positive control can see it,
      because an honest witness satisfies a correct model and an over-strong
      model equally well.

Every check below is paired with a TAMPER that must make it fail; a check
that does not bite is itself reported as a failure.  Nothing outside this
file is modified: tampers are applied to in-memory copies and reverted.

Run:  python3 audit_gate_transcription.py            (fast sections)
      python3 audit_gate_transcription.py --slow     (+ the BV UNSATs, ~5 min)
"""
import importlib.util
import itertools
import os
import random
import sys

from z3 import (
    And, BitVec, BitVecVal, Concat, Int, IntVal, Or, RotateRight, Solver,
    is_bv, sat, simplify, unsat,
)

HERE = os.path.dirname(os.path.abspath(__file__))
P = 2**64 - 2**32 + 1          # Goldilocks
MASK32 = 0xFFFFFFFF


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, relpath))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ORA = _load("blake3_ref", "blake3-oracle/blake3_ref.py")
GATE = _load("z3_blake_verify", "blake3-chip/z3_blake_verify.py")


# ---------------------------------------------------------------------------
# result bookkeeping
# ---------------------------------------------------------------------------
RESULTS = []


def record(section, name, ok, detail=""):
    RESULTS.append((section, name, bool(ok), detail))
    flag = "PASS" if ok else "**FAIL**"
    print(f"  [{flag}] {name}" + (f"   {detail}" if detail else ""))
    return ok


def note(text):
    """An observation that is reported but is not a pass/fail check."""
    print(f"  [note] {text}")


def ora_G_CALLS():
    """The G-call schedule RECOVERED from the oracle (blake3_ref.round_fn),
    with sentinel message words so the mx/my indices come back too."""
    calls = []
    orig = ORA.g

    def spy(state, a, b, c, d, mx, my):
        calls.append((a, b, c, d, mx - 1000, my - 1000))
    ORA.g = spy
    try:
        ORA.round_fn([0] * 16, [1000 + i for i in range(16)])
    finally:
        ORA.g = orig
    return calls


def bites(section, name, tamper_fn):
    """A check must FAIL under its tamper, or the check is decorative."""
    try:
        detected = tamper_fn()
    except Exception as exc:                      # a crash is also detection
        detected = True
        record(section, f"tamper bites: {name}", True, f"(raised {type(exc).__name__})")
        return True
    return record(section, f"tamper bites: {name}", detected,
                  "" if detected else "TAMPER NOT DETECTED — the check is vacuous")


# ===========================================================================
# SECTION A — transcription (a): blake3_ref.py  ->  bref_*
# ===========================================================================
def section_A(slow):
    print("\n" + "=" * 74)
    print("A  REFERENCE TRANSCRIPTION — blake3_ref.py -> bref_* (the BV oracle)")
    print("=" * 74)

    # -- A1 constant tables, element by element ----------------------------
    record("A", "IV identical (8 words, element-wise)",
           list(GATE.IV) == list(ORA.IV), f"{[hex(x) for x in GATE.IV[:2]]}...")
    record("A", "MSG_PERMUTATION identical (16 indices, element-wise)",
           list(GATE.MSG_PERMUTATION) == list(ORA.MSG_PERMUTATION),
           str(GATE.MSG_PERMUTATION))
    record("A", "MSG_PERMUTATION is a permutation of 0..15",
           sorted(GATE.MSG_PERMUTATION) == list(range(16)))
    record("A", "MASK32 identical", GATE.MASK32 == ORA.MASK32)

    def tamper_iv():
        old = GATE.IV[3]
        GATE.IV[3] ^= 1
        bad = list(GATE.IV) != list(ORA.IV)
        GATE.IV[3] = old
        return bad
    bites("A", "IV comparison", tamper_iv)

    # -- A2 the G-call schedule, RECOVERED from the oracle ------------------
    ora_calls = ora_G_CALLS()
    record("A", "G_CALLS == the oracle's round_fn call sequence (recovered)",
           [tuple(x) for x in GATE.G_CALLS] == ora_calls,
           f"{len(ora_calls)} calls")
    record("A", "every G quadruple has 4 DISTINCT state indices "
                "(so check_g's a,b,c,d=0,1,2,3 instance is general)",
           all(len({a, b, c, d}) == 4 for a, b, c, d, _, _ in GATE.G_CALLS))
    record("A", "the 8 G-calls touch each of the 16 state slots exactly twice",
           sorted(i for q in GATE.G_CALLS for i in q[:4]) ==
           sorted(list(range(16)) * 2))
    record("A", "the 8 G-calls consume message indices 0..15 exactly once",
           sorted(i for q in GATE.G_CALLS for i in q[4:]) == list(range(16)))

    def tamper_gcalls():
        old = GATE.G_CALLS[4]
        GATE.G_CALLS[4] = (0, 5, 10, 15, 9, 8)          # mx/my swapped
        bad = [tuple(x) for x in GATE.G_CALLS] != ora_calls
        GATE.G_CALLS[4] = old
        return bad
    bites("A", "G_CALLS comparison", tamper_gcalls)

    # -- A3 rotation amounts and their ORDER, recovered from both sides -----
    def oracle_rot_amounts():
        seen = []
        orig = ORA.rotr

        def spy(x, n):
            seen.append(n)
            return orig(x, n)
        ORA.rotr = spy
        try:
            ORA.g([0] * 4, 0, 1, 2, 3, 0, 0)
        finally:
            ORA.rotr = orig
        return seen

    def bref_rot_amounts():
        seen = []
        orig = GATE.RotateRight

        def spy(x, n):
            seen.append(n)
            return orig(x, n)
        GATE.RotateRight = spy
        try:
            GATE.bref_g([BitVec(f"a{i}", 32) for i in range(4)], 0, 1, 2, 3,
                        BitVec("mx", 32), BitVec("my", 32))
        finally:
            GATE.RotateRight = orig
        return seen

    ora_rots, bref_rots = oracle_rot_amounts(), bref_rot_amounts()
    record("A", "bref_g rotation amounts and order == oracle g",
           ora_rots == bref_rots == [16, 12, 8, 7], f"{bref_rots}")

    # -- A4 differential: bref_* vs the oracle on concrete values -----------
    rng = random.Random(0xB1A3E)

    def w32(v):
        return BitVecVal(v & MASK32, 32)

    def as_int(bv):
        return simplify(bv).as_long()

    def diff_g(n):
        for _ in range(n):
            st = [rng.randrange(1 << 32) for _ in range(4)]
            mx, my = rng.randrange(1 << 32), rng.randrange(1 << 32)
            ref = list(st)
            ORA.g(ref, 0, 1, 2, 3, mx, my)
            bv = [w32(x) for x in st]
            GATE.bref_g(bv, 0, 1, 2, 3, w32(mx), w32(my))
            if [as_int(x) for x in bv] != ref:
                return False, (st, mx, my)
        return True, None

    ok, cex = diff_g(300)
    record("A", "bref_g == oracle g  (300 random + carry/rotate edge inputs)", ok,
           "" if ok else f"counterexample {cex}")

    # edge cases: all-zero, all-ones, single bits (exercise every carry and
    # every rotate boundary)
    edges = [[0] * 4, [MASK32] * 4, [1, 0, 0, 0], [0, 0, 0, MASK32],
             [0x80000000] * 4, [0x0000FFFF, 0xFFFF0000, 0xF0F0F0F0, 0x0F0F0F0F]]
    ok_edge = True
    for st in edges:
        for msg in ([0, 0], [MASK32, MASK32], [0x80000000, 1]):
            ref = list(st)
            ORA.g(ref, 0, 1, 2, 3, msg[0], msg[1])
            bv = [w32(x) for x in st]
            GATE.bref_g(bv, 0, 1, 2, 3, w32(msg[0]), w32(msg[1]))
            ok_edge &= ([as_int(x) for x in bv] == ref)
    record("A", "bref_g == oracle g  (edge inputs: 0, 2^32-1, MSB, split words)",
           ok_edge)

    def diff_round(n):
        for _ in range(n):
            st = [rng.randrange(1 << 32) for _ in range(16)]
            m = [rng.randrange(1 << 32) for _ in range(16)]
            ref = list(st)
            ORA.round_fn(ref, m)
            got = GATE.bref_round_only([w32(x) for x in st], [w32(x) for x in m])
            if [as_int(x) for x in got] != ref:
                return False
        return True
    record("A", "bref_round_only == oracle round_fn (60 random states+messages)",
           diff_round(60))

    m0 = [rng.randrange(1 << 32) for _ in range(16)]
    record("A", "bref_permute == oracle permute (and is index-identical)",
           [as_int(x) for x in GATE.bref_permute([w32(x) for x in m0])]
           == ORA.permute(m0))

    def diff_compress(n, rounds_list):
        for _ in range(n):
            h = [rng.randrange(1 << 32) for _ in range(8)]
            m = [rng.randrange(1 << 32) for _ in range(16)]
            t = rng.randrange(1 << 64)
            bl = rng.randrange(65)
            fl = rng.randrange(128)
            for r in rounds_list:
                ref = ORA.compress(h, m, t, bl, fl, rounds=r)
                got = GATE.bref_compress([w32(x) for x in h], [w32(x) for x in m],
                                         w32(t & MASK32), w32((t >> 32) & MASK32),
                                         w32(bl), w32(fl), r)
                if [as_int(x) for x in got] != ref:
                    return False, (h, m, t, bl, fl, r)
        return True, None

    ok, cex = diff_compress(25, [0, 1, 2, 5, 6, 7, 8])
    record("A", "bref_compress == oracle compress (25 vectors x rounds "
                "{0,1,2,5,6,7,8}) — pins the rounds parameterisation", ok,
           "" if ok else f"counterexample {cex}")

    # counters straddling 2^32 — the split order t_lo=v[12], t_hi=v[13]
    ok_ctr = True
    for t in (0, 1, 2**32 - 1, 2**32, 2**32 + 1, 2**47 + 12345, 2**64 - 1):
        h = [rng.randrange(1 << 32) for _ in range(8)]
        m = [rng.randrange(1 << 32) for _ in range(16)]
        ref = ORA.compress(h, m, t, 64, 3, rounds=6)
        got = GATE.bref_compress([w32(x) for x in h], [w32(x) for x in m],
                                 w32(t & MASK32), w32((t >> 32) & MASK32),
                                 w32(64), w32(3), 6)
        ok_ctr &= ([as_int(x) for x in got] == ref)
    record("A", "counter split order matches across 2^32 "
                "(t_lo->v[12], t_hi->v[13]; 7 counters incl. 2^32-1, 2^32)", ok_ctr)

    def tamper_bref_perm():
        old = GATE.MSG_PERMUTATION[:]
        GATE.MSG_PERMUTATION[0], GATE.MSG_PERMUTATION[1] = old[1], old[0]
        bad = not diff_compress(3, [2, 6])[0]
        GATE.MSG_PERMUTATION[:] = old
        return bad
    bites("A", "bref_compress differential (permutation)", tamper_bref_perm)

    def tamper_bref_iv():
        old = GATE.IV[0]
        GATE.IV[0] ^= 1
        bad = not diff_compress(2, [1, 6])[0]
        GATE.IV[0] = old
        return bad
    bites("A", "bref_compress differential (IV)", tamper_bref_iv)

    # -- A5 how many times the permutation is applied ----------------------
    def ora_permute_count(rounds):
        n = [0]
        orig = ORA.permute

        def spy(m):
            n[0] += 1
            return orig(m)
        ORA.permute = spy
        try:
            ORA.compress([0] * 8, [0] * 16, 0, 64, 0, rounds=rounds)
        finally:
            ORA.permute = orig
        return n[0]

    def bref_permute_count(rounds):
        n = [0]
        orig = GATE.bref_permute

        def spy(m):
            n[0] += 1
            return orig(m)
        GATE.bref_permute = spy
        try:
            GATE.bref_compress([w32(0)] * 8, [w32(0)] * 16, w32(0), w32(0),
                               w32(0), w32(0), rounds)
        finally:
            GATE.bref_permute = orig
        return n[0]

    counts = [(r, ora_permute_count(r), bref_permute_count(r)) for r in range(9)]
    record("A", "permute applications per rounds r == max(r-1,0), oracle == bref "
                "(the classic off-by-one)",
           all(o == b == max(r - 1, 0) for r, o, b in counts),
           " ".join(f"r{r}:{b}" for r, _, b in counts))

    # -- A6 the initial-state layout, probed slot by slot -------------------
    # rounds=0 makes out[i]=v[i]^v[i+8] and out[i+8]=v[i+8]^h[i] read the
    # initial state directly, so each slot is individually observable.
    h = [0] * 8
    m = [0] * 16
    tlo, thi, bl, fl = 0xA1A2A3A4, 0xB1B2B3B4, 0xC1C2C3C4, 0xD1D2D3D4
    out0 = [as_int(x) for x in GATE.bref_compress(
        [w32(x) for x in h], [w32(x) for x in m], w32(tlo), w32(thi),
        w32(bl), w32(fl), 0)]
    layout_ok = (
        out0[0] == GATE.IV[0] and out0[1] == GATE.IV[1] and
        out0[2] == GATE.IV[2] and out0[3] == GATE.IV[3] and
        out0[4] == tlo and out0[5] == thi and out0[6] == bl and out0[7] == fl and
        out0[8] == GATE.IV[0] and out0[12] == tlo and out0[13] == thi)
    record("A", "initial v layout: v[8..12]=IV, v[12]=t_lo, v[13]=t_hi, "
                "v[14]=block_len, v[15]=flags (probed slot by slot)", layout_ok,
           f"out[4..8]={[hex(x) for x in out0[4:8]]}")

    hh = [0x11111111 * (i + 1) for i in range(8)]
    out1 = [as_int(x) for x in GATE.bref_compress(
        [w32(x) for x in hh], [w32(x) for x in m], w32(0), w32(0), w32(0),
        w32(0), 0)]
    ff_ok = (all(out1[i] == (hh[i] ^ [GATE.IV[0], GATE.IV[1], GATE.IV[2],
                                      GATE.IV[3], 0, 0, 0, 0][i])
                 for i in range(8)) and
             all(out1[i + 8] == ([GATE.IV[0], GATE.IV[1], GATE.IV[2], GATE.IV[3],
                                  0, 0, 0, 0][i] ^ hh[i]) for i in range(8)))
    record("A", "feed-forward: out[i]=v[i]^v[i+8] and out[i+8]=v[i+8]^h[i] "
                "with h the ORIGINAL input CV (not the mutated state)", ff_ok)


# ===========================================================================
# SECTION B — transcription (b): DESIGN.md -> build_g / build_round /
#             build_compress.  Structural, by instrumenting the model.
# ===========================================================================
class Traced(GATE.Circuit):
    """Circuit subclass that records the SSA dataflow the model builds.

    Words are lists of z3 byte expressions; rotr16/rotr8 return the SAME byte
    objects (they are relabels), so resolving an operand's bytes to their
    producing word automatically follows a free rotation back to its source
    XOR — which is exactly the provenance question DESIGN 4.3/4.4/5 raises.
    """

    def __init__(self, tag, bug=None):
        super().__init__(tag, bug)
        self.words = []            # wid -> {kind, cells, used}
        self.owner = {}            # byte-expr name -> wid
        self.ops = []              # {kind, ins:[wid], outs:[wid], perm:[..]}
        self._pending = []

    # -- registration ------------------------------------------------------
    def fresh_word(self):
        w = super().fresh_word()
        wid = len(self.words)
        self.words.append({"kind": "unassigned", "cells": w, "used": 4})
        for c in w:
            self.owner[str(c)] = wid
        self._pending.append(wid)
        return w

    def const_word(self, val):
        w = super().const_word(val)
        wid = len(self.words)
        self.words.append({"kind": "const", "cells": w, "used": 4})
        return w

    def _wids(self, word):
        return sorted({self.owner[str(c)] for c in word if str(c) in self.owner})

    def _op(self, kind, ins, out_kinds, perm=None):
        outs = self._pending[:]
        self._pending = []
        for wid, k in zip(outs, out_kinds):
            self.words[wid]["kind"] = k
        self.ops.append({"kind": kind,
                         "ins": [self._wids(w) for w in ins],
                         "in_words": ins, "outs": outs, "perm": perm})
        return outs

    # -- the operations under contract ------------------------------------
    def xor(self, A, B):
        self._pending = []
        out = super().xor(A, B)
        self._op("xor", [A, B], ["xor_out"])
        return out

    def add2(self, A, B, drop_bool=False):
        self._pending = []
        out = super().add2(A, B, drop_bool)
        self._op("add2", [A, B], ["add_out"])
        return out

    def add3(self, A, B, M, drop_bool=False):
        self._pending = []
        out = super().add3(A, B, M, drop_bool)
        self._op("add3", [A, B, M], ["add_out"])
        return out

    def rotr(self, A, n, wrong_amount=False):
        self._pending = []
        out = super().rotr(A, n, wrong_amount)
        # fresh_word order inside Circuit.rotr: sll_lo, sllc_lo, sll_hi,
        # sllc_hi, Y  (the first four are used two bytes wide)
        outs = self._op("rotr", [A], ["sll", "sllc", "sll", "sllc", "rot_out"],
                        perm=n)
        for wid in outs[:4]:
            self.words[wid]["used"] = 2
        return out

    def rotr16(self, A):
        out = super().rotr16(A)
        self.ops.append({"kind": "relabel16", "ins": [self._wids(A)],
                         "in_words": [A], "outs": [],
                         "perm": [A.index(c) if c in A else None for c in out]})
        return out

    def rotr8(self, A):
        out = super().rotr8(A)
        self.ops.append({"kind": "relabel8", "ins": [self._wids(A)],
                         "in_words": [A], "outs": [],
                         "perm": [A.index(c) if c in A else None for c in out]})
        return out

    # -- provenance analysis ----------------------------------------------
    def xor_consumed(self):
        """wids that appear as an operand of at least one ByteAlu[XOR]."""
        s = set()
        for op in self.ops:
            if op["kind"] == "xor":
                for group in op["ins"]:
                    s.update(group)
        return s

    def unchecked(self):
        """SSA words whose byte-range DESIGN 4.3/4.4/5 sources from a
        downstream XOR, but which no XOR in this scope consumes."""
        xc = self.xor_consumed()
        return [wid for wid, w in enumerate(self.words)
                if w["kind"] in ("add_out", "rot_out") and wid not in xc]


def _build_one_g(cls=Traced, build=None):
    cir = cls("aud")
    v = [None] * 16
    va, vb, vc, vd = (cir.fresh_word(), cir.fresh_word(),
                      cir.fresh_word(), cir.fresh_word())
    mx, my = cir.fresh_word(), cir.fresh_word()
    for w in (va, vb, vc, vd, mx, my):
        cir.words[cir.owner[str(w[0])]]["kind"] = "input"
    v[0], v[1], v[2], v[3] = va, vb, vc, vd
    (build or GATE.build_g)(cir, v, 0, 1, 2, 3, mx, my, None, False)
    return cir, v, (va, vb, vc, vd, mx, my)


def _build_compress(rounds):
    cir = Traced("audc")
    h = [cir.fresh_word() for _ in range(8)]
    m = [cir.fresh_word() for _ in range(16)]
    tlo, thi, bl, fl = (cir.fresh_word(), cir.fresh_word(),
                        cir.fresh_word(), cir.fresh_word())
    for w in h + m + [tlo, thi, bl, fl]:
        cir.words[cir.owner[str(w[0])]]["kind"] = "input"
    out = GATE.build_compress(cir, h, m, tlo, thi, bl, fl, rounds)
    return cir, out, dict(h=h, m=m, tlo=tlo, thi=thi, bl=bl, fl=fl)


def section_B(slow):
    print("\n" + "=" * 74)
    print("B  CIRCUIT TRANSCRIPTION — DESIGN.md -> build_g / build_round / "
          "build_compress")
    print("=" * 74)

    # -- B1 the free rotations are byte relabels, not BV rotates -----------
    cir = GATE.Circuit("rel")
    A = cir.fresh_word()
    n_before = cir.n
    r16, r8 = cir.rotr16(A), cir.rotr8(A)
    record("B", "rotr16 is the index relabel [b2,b3,b0,b1] on the SOURCE bytes "
                "(object identity, DESIGN 4.2/7.6)",
           [c is A[i] for c, i in zip(r16, (2, 3, 0, 1))] == [True] * 4)
    record("B", "rotr8 is the index relabel [b1,b2,b3,b0] on the SOURCE bytes",
           [c is A[i] for c, i in zip(r8, (1, 2, 3, 0))] == [True] * 4)
    record("B", "the free rotations commit NO new columns and emit NO "
                "constraints (DESIGN 3: 'produce no columns')",
           cir.n == n_before and cir.C == [])

    s = Solver()
    s.add(Or(cir.word32(r16) != RotateRight(cir.word32(A), 16),
             cir.word32(r8) != RotateRight(cir.word32(A), 8)))
    record("B", "and the relabels are VALUE-equal to RotateRight 16 / 8 "
                "(z3, all 2^32 inputs)", s.check() == unsat)

    def tamper_relabel():
        w = GATE.Circuit("t")
        B = w.fresh_word()
        wrong = [B[1], B[2], B[3], B[0]]          # rotr8 pattern used for 16
        s2 = Solver()
        s2.add(w.word32(wrong) != RotateRight(w.word32(B), 16))
        return s2.check() == sat
    bites("B", "relabel value check", tamper_relabel)

    # -- B2 range-check provenance: the load-bearing claim ------------------
    print("\n  -- B2  where does each SSA word's byte range actually come from? --")
    gcir, gv, _ = _build_one_g()
    kinds = {}
    for w in gcir.words:
        kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
    record("B", "one G commits 56 byte-cells + 6 carry bits (DESIGN 3 table)",
           sum(w["used"] for w in gcir.words
               if w["kind"] in ("add_out", "xor_out", "sll", "sllc", "rot_out")) == 56
           and sum(1 for c in gcir.C if "Or" in str(c)[:3] or str(c).startswith("Or")) == 6,
           f"cells={sum(w['used'] for w in gcir.words if w['kind'] not in ('input','unassigned','const'))}, "
           f"bool-constraints={sum(1 for c in gcir.C if str(c).startswith('Or'))}")
    record("B", "one G = 2 add3 + 2 add2 + 4 xor + 2 shift-rotations + "
                "1 rotr16 + 1 rotr8 (DESIGN 2/5)",
           [sum(1 for o in gcir.ops if o["kind"] == k)
            for k in ("add3", "add2", "xor", "rotr", "relabel16", "relabel8")]
           == [2, 2, 4, 2, 1, 1])

    xc_g = gcir.xor_consumed()
    add_outs = [wid for wid, w in enumerate(gcir.words) if w["kind"] == "add_out"]
    record("B", "all FOUR add outputs of a G (A1, C1, A2, C2) are consumed by a "
                "ByteAlu INSIDE the same G — so MAIN 0's byte declaration is "
                "derivable for them (the class where the range check is "
                "load-bearing; see section C)",
           len(add_outs) == 4 and all(w in xc_g for w in add_outs))

    unchecked_in_g = gcir.unchecked()
    detail = ", ".join(f"w{wid}({gcir.words[wid]['kind']})" for wid in unchecked_in_g)
    record("B", "INSIDE one G, exactly one SSA output has no ByteAlu consumer: "
                "the final rotr7 result B2 (its range check lives in the NEXT "
                "G / the feed-forward — outside MAIN 0's scope)",
           len(unchecked_in_g) == 1
           and gcir.words[unchecked_in_g[0]]["kind"] == "rot_out"
           and unchecked_in_g[0] == gcir.owner[str(gv[1][0])],
           f"unchecked in G-scope: [{detail}]")

    ccir, cout, cin = _build_compress(6)
    unchecked_full = ccir.unchecked()
    record("B", "in the FULL 6-round compression every add/shift output IS "
                "consumed by a ByteAlu[XOR] — DESIGN 5/7.4's premise verified "
                "mechanically (the gate never checks it)",
           unchecked_full == [],
           f"{sum(1 for w in ccir.words if w['kind'] in ('add_out','rot_out'))} "
           f"add/shift outputs, {len(unchecked_full)} unchecked")
    other_rounds = {r: len(_build_compress(r)[0].unchecked()) for r in (1, 2, 7)}
    record("B", "…and for ROUNDS = 1, 2 and 7 too, so the premise is a property "
                "of the layout, not of the round count",
           set(other_rounds.values()) == {0}, str(other_rounds))

    xc = ccir.xor_consumed()
    ins_xored = {k: all(ccir.owner[str(w[0])] in xc for w in
                        (cin[k] if isinstance(cin[k], list) and
                         isinstance(cin[k][0], list) else [cin[k]]))
                 for k in ("h", "m", "tlo", "thi", "bl", "fl")}
    record("B", "DESIGN 4.7 input claim: h, t_lo, t_hi, block_len, flags each "
                "feed an XOR; m does NOT (so m is the one input needing an "
                "explicit AreBytes)",
           ins_xored["h"] and ins_xored["tlo"] and ins_xored["thi"]
           and ins_xored["bl"] and ins_xored["fl"] and not ins_xored["m"],
           str(ins_xored))

    # which XOR is h[i]'s range check?  h[0..4] land in round-0 'a' slots,
    # which G only ever uses as an ADD operand — so their sole ByteAlu is the
    # UPPER feed-forward half, the half a CV-only caller (DESIGN 1.1) would
    # naturally drop.
    h_consumers = []
    for i in range(8):
        wid = ccir.owner[str(cin["h"][i][0])]
        cons = [oi for oi, o in enumerate(ccir.ops)
                if o["kind"] == "xor" and any(wid in gp for gp in o["ins"])]
        h_consumers.append(len(cons))
    ff_start = min(oi for oi, o in enumerate(ccir.ops)
                   if o["kind"] == "xor" and
                   any(ccir.owner[str(cin["h"][0][0])] in gp for gp in o["ins"]))
    record("B", "h[0..4] have exactly ONE ByteAlu consumer (the upper "
                "feed-forward out[i+8]=v[i+8]^h[i]) while h[4..8] have two "
                "(they are round-0 'b' slots, so X2 xors them too)",
           h_consumers == [1, 1, 1, 1, 2, 2, 2, 2],
           f"consumer counts {h_consumers}")
    record("B", "…and that single consumer really is a feed-forward XOR, not a "
                "round XOR (it is among the last 24 ops of the circuit)",
           ff_start >= len(ccir.ops) - 24,
           f"op #{ff_start} of {len(ccir.ops)}")

    def tamper_provenance():
        """A G-variant in which an add output has NO ByteAlu consumer —
        exactly the deviation DESIGN 7.4 warns about.  The detector must see
        it."""
        def build_g_leaky(cir, v, a, b, c, d, mx, my, bug, gflag):
            v[a] = cir.add3(v[a], v[b], mx)
            v[d] = cir.rotr16(cir.xor(v[d], v[a]))
            v[c] = cir.add2(v[c], v[d])
            v[b] = cir.rotr(cir.xor(v[b], v[c]), 12)
            v[a] = cir.add3(v[a], v[b], my)
            v[d] = cir.rotr8(cir.xor(v[d], v[a]))
            v[c] = cir.add2(v[c], v[d])          # C2: consumer removed below
            v[b] = cir.rotr(cir.xor(v[b], v[a]), 7)   # reads v[a], not v[c]
        cir2, _, _ = _build_one_g(build=build_g_leaky)
        return len(cir2.unchecked()) == 2
    bites("B", "range-check provenance detector", tamper_provenance)

    # -- B3 the message schedule as build_compress actually wires it -------
    print("\n  -- B3  message indexing under permute^r --")

    def capture_wiring(rounds=7):
        """What build_compress ACTUALLY feeds each G: the original message
        column index, recovered by tagging the committed message words."""
        seen = []
        orig_bg = GATE.build_g

        def spy_g(cir, v, a, b, c, d, mx, my, bug, gflag):
            seen.append((a, b, c, d, mx[0], my[0]))
            return orig_bg(cir, v, a, b, c, d, mx, my, bug, gflag)

        GATE.build_g = spy_g
        try:
            cir3 = Traced("sched")
            h3 = [cir3.fresh_word() for _ in range(8)]
            m3 = [cir3.fresh_word() for _ in range(16)]
            tg = {str(m3[i][0]): i for i in range(16)}
            t3 = [cir3.fresh_word() for _ in range(4)]
            GATE.build_compress(cir3, h3, m3, t3[0], t3[1], t3[2], t3[3], rounds)
        finally:
            GATE.build_g = orig_bg
        wired = [[(tg.get(str(mx)), tg.get(str(my)))
                  for (_, _, _, _, mx, my) in seen[r * 8:(r + 1) * 8]]
                 for r in range(rounds)]
        return wired, [q[:4] for q in seen]

    wired, quads = capture_wiring(7)
    # what the ORACLE says round r must consume: permute^r applied to the
    # identity schedule, then indexed by round_fn's own message positions
    expected = []
    sched = list(range(16))
    for r in range(7):
        expected.append([(sched[ix], sched[iy]) for (_, _, _, _, ix, iy)
                         in ora_G_CALLS()])
        sched = ORA.permute(sched)
    record("B", "build_compress feeds every G the ORIGINAL message column "
                "under permute^r, for all 7 rounds x 8 G-calls (56 index "
                "pairs), matching the oracle's permutation composition",
           wired == expected,
           f"round0 {wired[0][:2]}... round6 {wired[6][:2]}...")
    record("B", "the state quadruples build_round passes match the oracle's "
                "round_fn quadruples in order, all 7 rounds",
           quads == [tuple(c[:4]) for c in ora_G_CALLS()] * 7)

    def tamper_sched():
        old = GATE.MSG_PERMUTATION[:]
        GATE.MSG_PERMUTATION[3], GATE.MSG_PERMUTATION[4] = old[4], old[3]
        try:
            w2, _ = capture_wiring(7)
        finally:
            GATE.MSG_PERMUTATION[:] = old
        return w2 != expected
    bites("B", "message-schedule index check", tamper_sched)

    def tamper_quads():
        old = GATE.G_CALLS[5]
        GATE.G_CALLS[5] = (1, 6, 11, 12, 11, 10)      # mx/my swapped
        try:
            w2, q2 = capture_wiring(7)
        finally:
            GATE.G_CALLS[5] = old
        return w2 != expected
    bites("B", "G-call wiring check", tamper_quads)

    # -- B4 what the model does NOT carry ----------------------------------
    print("\n  -- B4  what the circuit model does not represent --")
    src = open(os.path.join(HERE, "blake3-chip/z3_blake_verify.py")).read()

    # Every variable the model creates is 8 bits wide, and every constraint it
    # emits is an equation or a carry booleanity.  There is no range-check
    # OBJECT, so "AreBytes present" and "AreBytes absent" are the same model.
    widths = set()
    for w in ccir.words:
        widths.update(c.size() for c in w["cells"])
    kinds = {}
    for c in ccir.C:
        kinds[c.decl().name()] = kinds.get(c.decl().name(), 0) + 1
    record("B", "every committed cell the model creates is BitVec(...,8): "
                "byte-ness is the DECLARATION, never a modelled lookup",
           widths == {8}, f"cell widths {sorted(widths)}")
    record("B", "and every emitted constraint is '=' (xor / sum / shift / "
                "recombine) or 'or' (carry booleanity) — no range constraint "
                "object exists to be present or absent",
           set(kinds) <= {"=", "or"}, str(kinds))
    cls_body = src[src.index("# Chip circuit model"):src.index("def build_g")]
    record("B", "inside the Circuit class, 'AreBytes' occurs only in comments "
                "(the class header ':118-119' and rotr ':212') — the gate "
                "DOCUMENTS the assumption ('Byte width == the ByteAlu/AreBytes "
                "range-check contract') but has no object for it",
           all(ln.strip().startswith("#") for ln in cls_body.split("\n")
               if "AreBytes" in ln))
    record("B", "the model has NO mu column: every eval identity is asserted "
                "ungated, which is exact for a live row and blind to padding "
                "rows (DESIGN 4.5 / 7.1 are therefore outside the gate)",
           not any("mu" in str(c).lower() for c in ccir.C)
           and "mu" not in "".join(str(w["cells"][0]) for w in ccir.words))
    record("B", "the model has no bus / multiplicity / timestamp layer at all "
                "(confirming README finding 1, not re-deriving it)",
           not any(k in src for k in ("Multiplicity", "TIMESTAMP", "bus_interaction",
                                      "receive(", "send(")))
    nunused = sum(1 for w in ccir.words if w["kind"] in ("sll", "sllc"))
    record("B", "each shift-rotation allocates 4 x fresh_word() but uses only "
                "2 bytes of each (fresh_word()[:2]) — 8 free unconstrained BVs "
                "per rotation, unread and harmless",
           nunused == 4 * 96, f"{nunused} halfword slots over 96 rotations")


# ===========================================================================
# SECTION C — the dangerous direction, in the field: where byte-ness
#             actually comes from, and the forgery the model cannot see.
# ===========================================================================
def _field_word(s, name, ranged):
    cells = [Int(f"{name}_{i}") for i in range(4)]
    for c in cells:
        s.add(c >= 0, c < (256 if ranged else P))
    return cells, sum(cells[i] * 2**(8 * i) for i in range(4))


def add_pinned(nops, out_ranged, ops_concrete=None, want_model=False):
    """DESIGN 4.3/4.4 add, modelled in the FIELD.  Is the committed sum word
    pinned to (sum of operands) mod 2^32?  unsat = pinned, sat = forgeable."""
    s = Solver()
    ops = []
    for k in range(nops):
        if ops_concrete:
            ops.append(IntVal(ops_concrete[k]))
        else:
            _, v = _field_word(s, f"in{k}", True)
            ops.append(v)
    scells, S = _field_word(s, "S", out_ranged)
    if nops == 2:
        c = Int("c")
        s.add(Or(c == 0, c == 1))
        csum = c
    else:
        c1, c2 = Int("c1"), Int("c2")
        s.add(Or(c1 == 0, c1 == 1), Or(c2 == 0, c2 == 1))
        csum = c1 + c2
    s.add((sum(ops) - S - 2**32 * csum) % P == 0)
    T, K = Int("T"), Int("K")
    s.add(K >= 0, K <= nops - 1, T >= 0, T < 2**32, sum(ops) == K * 2**32 + T)
    s.add((S - T) % P != 0)                    # a wrong FIELD VALUE, not just cells
    res = s.check()
    if res == sat and want_model:
        mo = s.model()
        g = lambda e: mo.eval(e, model_completion=True).as_long()
        return res, dict(operands=[hex(g(o)) for o in ops], honest=hex(g(T)),
                         forged=hex(g(S) % P), cells=[g(x) for x in scells],
                         carries=g(csum))
    return res, None


def rot_pinned(r, kept, want_model=False):
    """DESIGN 4.2 rotation in the FIELD, COMPOSED: both shift identities +
    both recombine identities + the byte range on Y that the downstream
    ByteAlu gives.  `kept` = which halfwords carry their AreBytes bound."""
    s = Solver()
    xlo, xhi = Int("xlo"), Int("xhi")
    s.add(xlo >= 0, xlo < 2**16, xhi >= 0, xhi < 2**16)
    hw = {}
    for n in ("SLL_lo", "SLLC_lo", "SLL_hi", "SLLC_hi"):
        if n in kept:
            lo, hi = Int(n + "_b0"), Int(n + "_b1")
            s.add(lo >= 0, lo < 256, hi >= 0, hi < 256)
            hw[n] = lo + 256 * hi
        else:
            v = Int(n)
            s.add(v >= 0, v < P)
            hw[n] = v
    s.add((xlo * 2**r - hw["SLLC_lo"] * 2**16 - hw["SLL_lo"]) % P == 0)
    s.add((xhi * 2**r - hw["SLLC_hi"] * 2**16 - hw["SLL_hi"]) % P == 0)
    Y = [Int(f"Y{i}") for i in range(4)]
    for y in Y:
        s.add(y >= 0, y < 256)
    Ylo, Yhi = Y[0] + 256 * Y[1], Y[2] + 256 * Y[3]
    s.add((Ylo - hw["SLL_hi"] - hw["SLLC_lo"]) % P == 0)
    s.add((Yhi - hw["SLL_lo"] - hw["SLLC_hi"]) % P == 0)
    X = xlo + 2**16 * xhi
    Q, R = Int("Q"), Int("R")
    s.add(Q >= 0, Q < 2**r, R >= 0, R < 2**32, X * 2**r == Q * 2**32 + R)
    wlo, whi = Int("wlo"), Int("whi")
    s.add(wlo >= 0, wlo < 2**16, whi >= 0, whi < 2**16, R + Q == wlo + 2**16 * whi)
    honest = whi + 2**16 * wlo
    s.push()
    s.add(Ylo + 2**16 * Yhi != honest)
    res = s.check()
    mdl = None
    if res == sat and want_model:
        mo = s.model()
        g = lambda e: mo.eval(e, model_completion=True).as_long()
        mdl = dict(X=hex(g(X)), honest_Y=hex(g(honest)),
                   forged_Y=hex(g(Ylo + 2**16 * Yhi)),
                   SLL_lo=hex(g(hw["SLL_lo"])), SLLC_lo=hex(g(hw["SLLC_lo"])),
                   SLL_hi=hex(g(hw["SLL_hi"])), SLLC_hi=hex(g(hw["SLLC_hi"])))
    s.pop()
    # non-vacuity: the honest witness must satisfy the model
    s.add(Ylo + 2**16 * Yhi == honest)
    live = s.check() == sat
    return res, mdl, live


def section_C(slow):
    print("\n" + "=" * 74)
    print("C  FIELD-LEVEL — what the byte range checks actually buy, and the "
          "forgery the\n   BV model cannot express")
    print("=" * 74)

    for n in (2, 3):
        res, _ = add_pinned(n, True)
        record("C", f"add{n}: WITH the output's byte range check the sum is "
                    f"pinned to (a+b{'+m' if n == 3 else ''}) mod 2^32, for "
                    f"ALL operands (symbolic, mod p)", res == unsat)
    for n in (2, 3):
        res, mdl = add_pinned(n, False, want_model=True)
        record("C", f"add{n}: WITHOUT it the committed sum is FORGEABLE — the "
                    f"prover commits the UNREDUCED sum with carry 0",
               res == sat, str(mdl))

    res, mdl = add_pinned(2, False, ops_concrete=[0x80000000, 0x80000000],
                          want_model=True)
    record("C", "concrete witness: a=b=0x80000000, honest s=0, forged s=2^32 "
                "with carry=0 (cells [2^32,0,0,0]) — every modelled constraint "
                "satisfied", res == sat and mdl["forged"] == hex(2**32), str(mdl))

    # the gate cannot tell the two apart: its `s` is 4 BitVec(8)s either way
    src = open(os.path.join(HERE, "blake3-chip/z3_blake_verify.py")).read()
    record("C", "…and the gate models BOTH chips identically: add2/add3 return "
                "`self.fresh_word()`, i.e. 4x BitVec(...,8), so the range check "
                "is DECLARED, never derived from a modelled lookup",
           "s = self.fresh_word()" in src and "AreBytes" not in
           src[src.index("def add2"):src.index("def rotr")])

    print("\n  -- C2  the rotation, composed (the gate tests it in isolation) --")
    lattice = {}
    for r in (4, 9):
        for k in range(4, -1, -1):
            for kept in itertools.combinations(
                    ("SLL_lo", "SLLC_lo", "SLL_hi", "SLLC_hi"), k):
                res, _, live = rot_pinned(r, set(kept))
                lattice[(r, kept)] = (res, live)
    all_live = all(live for _, live in lattice.values())
    record("C", "non-vacuity: the honest rotation witness satisfies the "
                "composed field model in all 32 bound configurations",
           all_live)
    record("C", "rotation with all four AreBytes bounds: Y is pinned to "
                "rotr12/rotr7 for ALL 2^32 inputs (symbolic, mod p — the gate "
                "only ever checked one concrete halfword in the field)",
           lattice[(4, ("SLL_lo", "SLLC_lo", "SLL_hi", "SLLC_hi"))][0] == unsat
           and lattice[(9, ("SLL_lo", "SLLC_lo", "SLL_hi", "SLLC_hi"))][0] == unsat)
    one_sll = all(lattice[(r, k)][0] == unsat for r in (4, 9)
                  for k in (("SLL_lo",), ("SLL_hi",)))
    no_sll = all(lattice[(r, k)][0] == sat for r in (4, 9)
                 for k in ((), ("SLLC_lo",), ("SLLC_hi",), ("SLLC_lo", "SLLC_hi")))
    record("C", "necessary AND sufficient bound set = at least one of "
                "{SLL_lo, SLL_hi}; every configuration with neither is "
                "forgeable, every configuration with either is pinned — the "
                "SLLC bounds are not load-bearing at all",
           one_sll and no_sll)
    res, mdl, _ = rot_pinned(9, set(("SLLC_lo", "SLLC_hi")), want_model=True)
    record("C", "the composed rotation forgery (both SLL bounds dropped) exists "
                "for exactly ONE input, X=0xFFFFFFFF -> forged Y=0 instead of "
                "0xFFFFFFFF — not 'any input', as the gate's isolated control "
                "suggests", res == sat and mdl["X"] == hex(0xFFFFFFFF), str(mdl))

    # exhaustively: is X = 0xFFFFFFFF the only one?
    def enumerate_bad_X(r, limit=4):
        found = []
        seen = set()
        for _ in range(limit):
            s = Solver()
            xlo, xhi = Int("xlo"), Int("xhi")
            s.add(xlo >= 0, xlo < 2**16, xhi >= 0, xhi < 2**16)
            hw = {}
            for n in ("SLL_lo", "SLL_hi"):
                v = Int(n)
                s.add(v >= 0, v < P)
                hw[n] = v
            for n in ("SLLC_lo", "SLLC_hi"):
                lo, hi = Int(n + "_b0"), Int(n + "_b1")
                s.add(lo >= 0, lo < 256, hi >= 0, hi < 256)
                hw[n] = lo + 256 * hi
            s.add((xlo * 2**r - hw["SLLC_lo"] * 2**16 - hw["SLL_lo"]) % P == 0)
            s.add((xhi * 2**r - hw["SLLC_hi"] * 2**16 - hw["SLL_hi"]) % P == 0)
            Y = [Int(f"Y{i}") for i in range(4)]
            for y in Y:
                s.add(y >= 0, y < 256)
            Ylo, Yhi = Y[0] + 256 * Y[1], Y[2] + 256 * Y[3]
            s.add((Ylo - hw["SLL_hi"] - hw["SLLC_lo"]) % P == 0)
            s.add((Yhi - hw["SLL_lo"] - hw["SLLC_hi"]) % P == 0)
            X = xlo + 2**16 * xhi
            Q, R = Int("Q"), Int("R")
            s.add(Q >= 0, Q < 2**r, R >= 0, R < 2**32, X * 2**r == Q * 2**32 + R)
            wlo, whi = Int("wlo"), Int("whi")
            s.add(wlo >= 0, wlo < 2**16, whi >= 0, whi < 2**16,
                  R + Q == wlo + 2**16 * whi)
            s.add(Ylo + 2**16 * Yhi != whi + 2**16 * wlo)
            for x in seen:
                s.add(X != x)
            if s.check() != sat:
                break
            xv = s.model().eval(X, model_completion=True).as_long()
            seen.add(xv)
            found.append(hex(xv))
        return found
    bad4, bad9 = enumerate_bad_X(4), enumerate_bad_X(9)
    record("C", "exhaustive: X=0xFFFFFFFF is the ONLY forgeable input for both "
                "r=4 and r=9 (all other X enumerated away -> unsat)",
           bad4 == bad9 == ["0xffffffff"], f"r=4 {bad4}  r=9 {bad9}")

    # …but the rotation OUTPUT does not need its own byte range check: both
    # recombine identities together pin its VALUE regardless of how its cells
    # decompose.  So the free-range-check argument is load-bearing for the add
    # outputs and for one SLL per rotation — and for nothing else.
    s = Solver()
    xlo, xhi = Int("xlo"), Int("xhi")
    s.add(xlo >= 0, xlo < 2**16, xhi >= 0, xhi < 2**16)
    hw = {}
    for n in ("SLL_lo", "SLLC_lo", "SLL_hi", "SLLC_hi"):
        lo, hi = Int(n + "_b0"), Int(n + "_b1")
        s.add(lo >= 0, lo < 256, hi >= 0, hi < 256)
        hw[n] = lo + 256 * hi
    r = 9
    s.add((xlo * 2**r - hw["SLLC_lo"] * 2**16 - hw["SLL_lo"]) % P == 0)
    s.add((xhi * 2**r - hw["SLLC_hi"] * 2**16 - hw["SLL_hi"]) % P == 0)
    Ycells = [Int(f"Yf{i}") for i in range(4)]
    for c in Ycells:
        s.add(c >= 0, c < P)                       # NO range check on Y
    Ylo, Yhi = Ycells[0] + 256 * Ycells[1], Ycells[2] + 256 * Ycells[3]
    s.add((Ylo - hw["SLL_hi"] - hw["SLLC_lo"]) % P == 0)
    s.add((Yhi - hw["SLL_lo"] - hw["SLLC_hi"]) % P == 0)
    X = xlo + 2**16 * xhi
    Q, R = Int("Q"), Int("R")
    s.add(Q >= 0, Q < 2**r, R >= 0, R < 2**32, X * 2**r == Q * 2**32 + R)
    wlo, whi = Int("wlo"), Int("whi")
    s.add(wlo >= 0, wlo < 2**16, whi >= 0, whi < 2**16, R + Q == wlo + 2**16 * whi)
    s.add((Ylo + 2**16 * Yhi - (whi + 2**16 * wlo)) % P != 0)
    record("C", "the rotation OUTPUT needs no range check of its own: the two "
                "recombine identities pin its value even with free field cells "
                "— so the 'free range check' is load-bearing only for the add "
                "outputs and one SLL halfword per rotation", s.check() == unsat)

    print("\n  -- C3  the width audit's two claims, re-derived symbolically --")
    # the gate proves each on ONE concrete input; prove them for all inputs
    s = Solver()
    inhw = Int("in_hw")
    s.add(inhw >= 0, inhw < 2**16)
    lo, hi = Int("lo"), Int("hi")
    s.add(lo >= 0, lo < 256, hi >= 0, hi < 256)
    SLL = lo + 256 * hi
    SLLC = Int("SLLC")
    s.add(SLLC >= 0, SLLC < 2**16)
    r = 9
    s.add((inhw * 2**r - SLLC * 2**16 - SLL) % P == 0)
    ref = Int("ref")
    s.add(ref >= 0, ref < 2**16, (inhw * 2**r - ref) % 2**16 == 0)
    s.add(SLL != ref)
    record("C", "field_shift_bound's UNSAT holds for ALL in_hw, not just "
                "0x9C3A (the gate tests one point)", s.check() == unsat)

    s = Solver()
    a, b, m3 = Int("a"), Int("b"), Int("m")
    for x in (a, b, m3):
        s.add(x >= 0, x < 2**32)
    S = Int("S")
    s.add(S >= 0, S < 2**32)
    c1, c2 = Int("c1"), Int("c2")
    s.add(Or(c1 == 0, c1 == 1), Or(c2 == 0, c2 == 1))
    s.add((a + b + m3 - S - 2**32 * (c1 + c2)) % P == 0)
    K, T = Int("K"), Int("T")
    s.add(K >= 0, K <= 2, T >= 0, T < 2**32, a + b + m3 == K * 2**32 + T)
    s.add(S != T)
    record("C", "field_add_carry's UNSAT holds for ALL (a,b,m), not just "
                "3x0xF0000000", s.check() == unsat)

    # dropping the booleanity really does free s, composed with s's byte range
    s = Solver()
    a, b, m3 = IntVal(0x12345678), IntVal(0x9ABCDEF0), IntVal(0x0F0F0F0F)
    scells, S = _field_word(s, "S", True)      # s STILL byte-range-checked
    k = Int("k")
    s.add(k >= 0, k < P)                       # booleanity dropped
    s.add((a + b + m3 - S - 2**32 * k) % P == 0)
    honest = (0x12345678 + 0x9ABCDEF0 + 0x0F0F0F0F) % 2**32
    s.add(S != honest)
    res = s.check()
    mdl = None
    if res == sat:
        mo = s.model()
        mdl = dict(honest=hex(honest),
                   forged=hex(mo.eval(S, model_completion=True).as_long()),
                   k=mo.eval(k, model_completion=True).as_long())
    record("C", "dropping the carry booleanity is forgeable even WITH the byte "
                "range check on s (so this control is faithful to the composed "
                "chip, unlike the shift one)", res == sat, str(mdl))

    print("\n  -- C4  the message columns (DESIGN 4.7): AreBytes vs the model --")
    # without AreBytes on m the cells bind only sum(m_i 2^8i): exhibit two
    # distinct cell vectors that satisfy every constraint identically.
    honest_cells = [0x9A, 0x00, 0x13, 0x7F]
    forged_cells = [0x9A + 256, 0x00 - 1, 0x13, 0x7F]
    same_value = (sum(honest_cells[i] * 2**(8 * i) for i in range(4)) % P ==
                  sum(forged_cells[i] * 2**(8 * i) for i in range(4)) % P)
    record("C", "without the explicit AreBytes, a message word has many cell "
                "representations with the same value (here [0x9A,0,0x13,0x7F] "
                "and [0x19A,-1,0x13,0x7F] = [.., p-1, ..]): the chip binds "
                "sum(m_i 2^8i), not the 64 bytes", same_value,
           f"forged cells over F_p: {[c % P for c in forged_cells]}")
    record("C", "the gate declares m as 16 x 4 BitVec(...,8), so it proves the "
                "SAME UNSAT for a chip with and without those 32 AreBytes sends",
           "m = [cir.fresh_word() for _ in range(16)]" in src)


# ===========================================================================
# SECTION D — gate hygiene: are the UNSATs non-vacuous, and is the model's
#             carry encoding the one DESIGN.md specifies?
# ===========================================================================
def section_D(slow):
    print("\n" + "=" * 74)
    print("D  GATE HYGIENE")
    print("=" * 74)

    cir, v, _ = _build_one_g()
    s = Solver()
    s.add(And(*cir.C))
    record("D", "the G circuit's constraint set is SATISFIABLE on its own — "
                "MAIN 0's UNSAT is not vacuous", s.check() == sat)
    counts = {}
    for opname, call in (("xor", lambda c: c.xor(c.fresh_word(), c.fresh_word())),
                         ("add2", lambda c: c.add2(c.fresh_word(), c.fresh_word())),
                         ("add3", lambda c: c.add3(c.fresh_word(), c.fresh_word(),
                                                   c.fresh_word())),
                         ("rotr12", lambda c: c.rotr(c.fresh_word(), 12)),
                         ("rotr16", lambda c: c.rotr16(c.fresh_word()))):
        c = GATE.Circuit("cnt")
        call(c)
        counts[opname] = len(c.C)
    record("D", "constraint counts per op match DESIGN 4.1-4.4: xor 4 (pure "
                "lookup, modelled as 4 byte equalities), add2 2 (sum + 1 "
                "booleanity), add3 3 (sum + 2 booleanities), rotr12 4 "
                "(2 shift + 2 recombine), rotr16 0 (free relabel)",
           counts == {"xor": 4, "add2": 2, "add3": 3, "rotr12": 4, "rotr16": 0},
           str(counts))

    ccir, cout, _ = _build_compress(6)
    s = Solver()
    s.add(And(*ccir.C))
    record("D", "the 6-round circuit's constraint set is SATISFIABLE on its own",
           s.check() == sat)

    # DESIGN 4.3 commits NO carry column for the 2-op add (carry is the linear
    # expression (a+b-s)*2^-32); the model commits a boolean column instead.
    # Prove the two are equivalent.
    s = Solver()
    A, B, S = Int("A"), Int("B"), Int("S")
    for x in (A, B, S):
        s.add(x >= 0, x < 2**32)
    c_derived = Int("cd")
    lhs = Or(And((A + B - S - 2**32 * 0) % P == 0),
             And((A + B - S - 2**32 * 1) % P == 0))     # committed-boolean form
    rhs = ((A + B - S) * pow(2**32, -1, P) % P == 0)
    # derived form: carry := (A+B-S)*2^-32 mod p, booleanity carry*(carry-1)=0
    cd = ((A + B - S) * pow(2**32, -1, P)) % P
    rhs = Or(cd == 0, cd == 1)
    s.add(lhs != rhs)
    record("D", "DESIGN 4.3's DERIVED carry (linear expr x INV_SHIFT_32, "
                "booleanity) and the model's COMMITTED boolean carry are "
                "equivalent over F_p — the difference is 1 column per add2, "
                "not a semantic one", s.check() == unsat)

    note("DESIGN 4.8's ledger row for the recombine identity says body degree "
         "2 -> 3 after x mu; the body mu*(Ylo - SLL_hi - SLLC_lo) is LINEAR in "
         "committed columns, so it is 1 -> 2. Over-stated in the safe "
         "direction; the 'no constraint exceeds 3' verdict is unaffected.")
    note("DESIGN 3's per-G table counts 1 committed carry bit for each add2, "
         "while DESIGN 4.3 makes that carry a DERIVED linear expression "
         "(a+b-s)*INV_SHIFT_32 with no column. The gate models the committed "
         "form. Equivalent as constraints (proved above); the two readings "
         "differ by 96 cells/compression in the DESIGN 6 cost table.")

    # The positive controls are the gate's only external anchor.  They pin the
    # circuit's output to a RECORDED vector, so a stale fixture would silently
    # anchor the gate to nothing.
    vecs = GATE.load_canonical_6round()
    ok_vec = all(ORA.compress_6round(v["h"], v["m"], v["t"], v["block_len"],
                                     v["flags"]) == v["out"] for v in vecs)
    record("D", f"all {len(vecs)} canonical 6-round fixture vectors reproduce "
                "from the oracle's compress_6round — the positive controls "
                "anchor to the live oracle, not a stale file", ok_vec)
    record("D", "…and they are genuinely 6-round: none of them equals the "
                "7-round compression of the same input",
           all(ORA.compress(v["h"], v["m"], v["t"], v["block_len"], v["flags"],
                            rounds=7) != v["out"] for v in vecs))
    h7, m7, tlo7, thi7, bl7, fl7, out7 = GATE.gen_7round_vector()
    record("D", "gen_7round_vector's output is the oracle's 7-round "
                "compression of its own inputs",
           ORA.compress(h7, m7, tlo7 | (thi7 << 32), bl7, fl7, rounds=7) == out7)

    # WIDE = 48 must be wide enough that the BV identities are integer
    # identities, and small-enough values that they coincide with mod-p.
    s = Solver()
    a, b, m3, S = Int("a"), Int("b"), Int("m"), Int("S")
    for x in (a, b, m3, S):
        s.add(x >= 0, x < 2**32)
    c1, c2 = Int("c1"), Int("c2")
    s.add(Or(c1 == 0, c1 == 1), Or(c2 == 0, c2 == 1))
    s.add((a + b + m3 == S + 2**32 * (c1 + c2)) !=
          (((a + b + m3 - S - 2**32 * (c1 + c2)) % P) == 0))
    record("D", "the 3-op sum identity over Z (what WIDE=48 BV computes) and "
                "over F_p (what the chip computes) are equivalent under the "
                "byte bounds — no wraparound is available on either side, "
                "confirming DESIGN 7.9", s.check() == unsat)
    # The gate ASSUMES the BITWISE contracts (assume-guarantee).  They are
    # cheap to verify against the real table, so verify them.
    bw = os.path.join(HERE, "..", "..", "prover", "src", "tables", "bitwise.rs")
    if os.path.exists(bw):
        rs = open(bw).read()
        record("D", "the assumed ByteAlu[XOR] contract is real: bitwise.rs "
                    "enumerates x,y in 0..256 and sets cols::XOR = x^y, and the "
                    "receiver pins (XOR, X, Y) -> XOR",
               "for x in 0u32..256 {" in rs and "for y in 0u32..256 {" in rs
               and "table.set_byte(row_idx, cols::XOR, (x ^ y) as u8);" in rs
               and "Multiplicity::Column(cols::MU_BYTE_ALU_XOR)" in rs)
        record("D", "the assumed AreBytes contract is real: an AreBytes "
                    "receiver over the same 0..256 x 0..256 domain",
               "BusId::AreBytes," in rs
               and "ARE_BYTES[X, Y] - range check two byte values" in rs)
    else:
        note(f"bitwise.rs not found at {bw}; the ByteAlu/AreBytes contracts "
             "were not cross-checked in this run.")

    note("block_len and flags are modelled as free 32-bit words; the design "
         "says 0..64 and 0..127. The model is WEAKER there, which is the safe "
         "direction, and DESIGN.md specifies no such constraint either.")
    note("DESIGN 4.1 allows a ByteAlu operand to be a linear combination of "
         "cells ('sum <= 255'); the model never uses one — rotr16/rotr8 are "
         "pure index relabels, so every operand is a single cell. Consistent, "
         "but the linear-combo contract is therefore unexercised.")

    if slow:
        print("\n  -- D2  the gate's own BV verdicts, re-run --")
        record("D", "check_g() == unsat (MAIN 0)", GATE.check_g() == unsat)
        record("D", "check_compress(0) == unsat (MAIN 1)",
               GATE.check_compress(0) == unsat)
        record("D", "check_g(bug='swap_g_operand') == sat",
               GATE.check_g(bug="swap_g_operand") == sat)


# ===========================================================================
def main():
    slow = "--slow" in sys.argv
    print("=" * 74)
    print("BLAKE3 GATE TRANSCRIPTION AUDIT — regression suite")
    print("=" * 74)
    section_A(slow)
    section_B(slow)
    section_C(slow)
    section_D(slow)

    print("\n" + "=" * 74)
    fails = [(s, n) for s, n, ok, _ in RESULTS if not ok]
    print(f"SUMMARY: {len(RESULTS) - len(fails)}/{len(RESULTS)} checks pass")
    for s, n in fails:
        print(f"  FAIL [{s}] {n}")
    print("=" * 74)
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
