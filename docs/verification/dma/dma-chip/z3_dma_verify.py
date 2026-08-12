"""
Formal (z3) assume-guarantee gate for the DMA memcpy chip (PR #874).

Method (mirrors blake3-chip/z3_blake_verify.py from PR #903, branch feat/blake3-accelerator):
  * every committed column of the table is a FREE variable;
  * every eval constraint and every bus lookup becomes an equation over those
    free variables;
  * the row's OUTPUT is whatever the constraints force. We assert
    `output != reference(input)` and ask z3 for a counterexample:
        UNSAT -> for every constraint-satisfying assignment the row does what
                 the oracle says (correctly and tightly constrained);
        SAT   -> the constraints permit a wrong row (under-constrained).

TWO LAYERS, and the split is deliberate.

  Layer 1 (field-exact, one or two rows). Every column is an element of
  Goldilocks `p = 2^64 - 2^32 + 1`, every constraint is an equation mod p, and
  each carry is extracted the way the templates extract it,
  `carry = (lhs + rhs - sum) * 2^-32`. A bit-vector model CANNOT do this job:
  the whole question here is whether a range check is missing, and in a bounded
  BV model an unconstrained column is silently bounded, so the bug disappears.
  Layer 1 proves the ROW ABSTRACTION -- `width = 8 - 7*tail`, `tail = count < 8`,
  `end = (count == 0)`, `src_incr = src + width` without wrapping,
  `count_decr = (count - width) mod 2^64` -- as INTEGER relations, out of the
  field-level constraints alone.

  Layer 2 (many rows). Takes the row abstraction as given, adds the `DmaNext`
  bus as a free BIJECTION between senders and receivers (not as an assumed
  chain), and proves the only balanced structure is a single chain whose data
  rows tile `[src, src + n)` exactly once with the oracle's widths. This is
  where "a source row skipped forward", "the copy ended early" and "a disjoint
  cycle of rows also balances the bus" get answered. Run at the integer level
  for depth, and re-run field-exact at small depth so the abstraction step is
  not taken on trust.

MODELLED CONTRACTS. Every lookup is modelled by the CONSTRAINTS OF THE TABLE
THAT RECEIVES IT, not by its advertised contract -- an advertised contract is
exactly the kind of premise that turns out to be declared and never derived
(finding F1 of GATE-TRANSCRIPTION-AUDIT.md of PR #903 (branch feat/blake3-accelerator), which in the EC
campaign hid a working forgery):

  IsHalfword[h]        h in [0, 2^16). Preprocessed, so the contract IS the
                       range.
  Zero[v] -> is_zero   `bitwise.rs`: the argument decomposes as
                       v = X + 256*Y + 65536*Z with X,Y bytes and Z in [0,16),
                       and the OUTPUT column is 1 iff X = Y = Z = 0. (X/Y/Z are
                       the table's own digit columns; the output is a separate
                       column, `cols::ZERO`. Do not reuse the name `z` for both,
                       as an earlier draft of this docstring did.) NOTE THE
                       DOMAIN -- the table only has rows for v < 2^20, so a send
                       outside it has no partner at all.
  Alu[a,b,LT] -> o     `lt.rs`'s own columns: a free `lhs_sub_rhs` of four
                       IsHalfword halves, two carries with booleanity,
                       `out = carry_1`, `lhs.hi = lhs[1] + 2^16*lhs[2]` with
                       both halves range-checked. NOT `o = (a < b)`. This is
                       what makes the `LHS_0` aliasing question below visible
                       at all: `lt.rs` range-checks `lhs[1]` and `lhs[2]` but
                       NOT the bare `LHS_0` word.
  Memw(addr, ...)      the base-address limbs are 32-bit          (MEMW-ADDR32)
  Memw register read   the three argument registers' limbs are 32-bit and equal
                       the register file's value                  (REG-32)

The last two are PREMISES THIS GATE DOES NOT PROVE. ../TRANSCRIPTION-AUDIT.md
records where each is discharged. Their control coverage is asymmetric, and
saying so is the point:

  REG-32 (A2) IS load-bearing -- `drop_reg32` flips `check_row_budget` to sat,
      because a bound lookup on a residue class caps only the residue.
  MEMW-ADDR32 (A1) is INERT for every claim this board makes: dropping it leaves
      MAIN 0/1/2/2b/2c/3, both chain layers and the sweep unchanged. It is
      modelled because the AIR really does carry those sends, but nothing here
      leans on it -- the limb-wise `DmaNext` link derives well-formedness from
      the sender's IsHalfword checks instead. It therefore has NO negative
      control, deliberately: a control that cannot fail is worse than none, and
      an earlier version of this docstring claimed "every negative control shows
      what breaks without them", which was false for exactly this premise.

WHAT THE GATE CANNOT SEE (same disclaimer shape as the BLAKE3 gate):
  * bus WIRING -- that the read tuple and the write tuple really reference the
    same `value` columns, that the timestamp offsets really are +1 and +2, that
    the multiplicities really are `mu - end` / `first` / `mu`. Those are textual
    facts about `dma.rs`, checked by ../audit_gate_transcription.py.
  * the MEMW consistency argument, hence the snapshot semantics of an
    overlapping copy. That is a timestamp-ordering property of the memory
    table; the oracle's `write_before_read` mutant covers the model side.
  * LogUp soundness. Bus balance is assumed to mean multiset equality.
  * the multi-call case. Layer 2 proves the tiling among groups containing
    exactly ONE head row; `ChainRow` carries no timestamp, and the `ts` in both
    `DmaNext` tuples is what separates two ecalls' rows.

    python3 z3_dma_verify.py             # the full board
    python3 z3_dma_verify.py --quick     # shorter completeness sweep and chains
"""

import os
import sys

from z3 import (
    And, Distinct, If, Implies, Int, IntVal, Not, Or, Solver, Sum, get_version,
    get_version_string, sat, unknown, unsat,
)

#: Solver version this board is known green on. NOT a hard pin -- the queries are
#: version-independent in meaning -- but older solvers are dramatically slower on
#: the field-exact chain (`CHAIN-F 2` measured 0.45 s on 5.0.0 vs 7.60 s on
#: 4.12.2, 17x), so they blow the per-query budgets and report `unknown`.
#: `unknown` is scored as FAILURE everywhere, never as success, so an old solver
#: gives a false alarm and never a false proof -- but the operator deserves to be
#: told which it is looking at.
VALIDATED_Z3 = (5, 0, 0)

# ---------------------------------------------------------------------------
# Constants -- transcribed from the Rust; ../audit_gate_transcription.py
# asserts each one against the source.
# ---------------------------------------------------------------------------

P = 2**64 - 2**32 + 1          # Goldilocks
INV_2_32 = pow(2**32, -1, P)   # `templates::INV_SHIFT_32`
B16, B32, B64 = 2**16, 2**32, 2**64

WIDE_WIDTH, TAIL_WIDTH = 8, 1
MAX_BYTES = 256                # `DMA_MEMCPY_MAX_BYTES`
ZERO_SUM = 4 * 65535           # the constant in the Zero sender's linear term
ZERO_DOMAIN = 2**20            # bitwise ZERO covers x + 256y + 65536z, z < 16

assert INV_2_32 == 18446744065119617026, "INV_SHIFT_32 transcription is wrong"
assert ZERO_SUM < ZERO_DOMAIN, "the Zero send can leave the receiving table's domain"


class Premises:
    """Which assume-guarantee premises are switched on.

    Every field names a lookup or range check that exists in `dma.rs`, or in a
    table `dma.rs` sends to. Turning one off is a negative control: the gate
    then reports the forgery that check is the sole obstacle to.
    """

    NAMES = (
        "halfword_src_incr", "halfword_dst_incr", "halfword_count_decr",
        "memw_addr32", "reg32", "lt_tail", "lt_bound", "zero_end",
        "no_overflow_src", "no_overflow_dst", "tail_lane_zero",
    )

    def __init__(self, **off):
        for name in self.NAMES:
            setattr(self, name, True)
        for name, value in off.items():
            assert name in self.NAMES, f"unknown premise {name}"
            setattr(self, name, value)


#: Default per-query solver budget. Never leave a query unbounded: an unlucky
#: solver version or platform then HANGS a CI job instead of failing it, and a
#: hang is far harder to diagnose than a timeout (measured: the whole board takes
#: ~96 s on z3 5.0.0 but ~1210 s on 4.12.2, where two queries blew their budgets).
DEFAULT_TIMEOUT_MS = 120_000

_EQ_COUNTER = [0]


def eq_mod(lhs, rhs, modulus):
    """`lhs == rhs (mod modulus)` as a linear constraint with a witness quotient.

    See the ENCODING NOTE in `FieldRow`: this is exactly `(lhs - rhs) % m == 0`,
    written so the query stays in linear integer arithmetic.
    """
    _EQ_COUNTER[0] += 1
    k = Int(f"q{_EQ_COUNTER[0]}")
    return lhs - rhs == k * modulus


def dmanext_link(sender, receiver):
    """The `DmaNext` bus binding, ELEMENT BY ELEMENT.

    THIS IS THE FACT AN EARLIER VERSION OF THIS GATE GOT WRONG, and the error
    produced a phantom soundness finding that was published as the campaign's
    headline result. It is worth stating precisely.

    `Packing::num_bus_elements()` (`crypto/stark/src/lookup.rs:227-241`) returns
    **2** for both `DWordWL` ("2x Direct") and `DWordHL` ("2x Word2L"), and
    `accumulate_fingerprint_with` (`:305-340`) gives each element its own alpha
    power. No `Packing` variant contains a `2^32` shift at all, so **a 64-bit
    value is never a single bus element anywhere in this codebase.**

    Both DmaNext tuples are therefore 8 elements (`1+1+2+2+2`) and align
    pairwise, so bus balance imposes TWO independent equations per 64-bit value:

        receiver.src0 == sender.si0 + 2^16*si1      (low word)
        receiver.src1 == sender.si2 + 2^16*si3      (high word)

    Modelling it as one equation on the fully packed value is strictly WEAKER
    than the AIR: it lets the receiver re-split the limbs freely, which
    manufactures an alias (`cnt1 = 2^32-1`, `cnt0 = V+1`) that the real bus
    rejects. Since the sender's halfwords are IsHalfword-bounded, the limb-wise
    form gives the receiver 32-bit limbs *for free* — no extra range check
    needed, which is why the "fix" that earlier version recommended (receive
    `count` as `DWordHL`) was a no-op, and why a no-op fix should have been
    read as evidence that the gap wasn't there.
    """
    return [
        eq_mod(sender.ts0, receiver.ts0, P),
        eq_mod(sender.ts1, receiver.ts1, P),
        eq_mod(sender.hl_lo(sender.si), receiver.src0, P),
        eq_mod(sender.hl_hi(sender.si), receiver.src1, P),
        eq_mod(sender.hl_lo(sender.di), receiver.dst0, P),
        eq_mod(sender.hl_hi(sender.di), receiver.dst1, P),
        eq_mod(sender.hl_lo(sender.cd), receiver.cnt0, P),
        eq_mod(sender.hl_hi(sender.cd), receiver.cnt1, P),
    ]


def solve(assertions, timeout_ms=DEFAULT_TIMEOUT_MS):
    s = Solver()
    if timeout_ms:
        s.set("timeout", timeout_ms)
    for a in assertions:
        s.add(a)
    return s.check()


# ===========================================================================
# Layer 1 -- field-exact model of a row
# ===========================================================================

class FieldRow:
    """One DMA row, every column a free Goldilocks element.

    Column names track `dma::cols` exactly. Lookups are placed under their own
    multiplicity: a lookup with multiplicity zero constrains NOTHING, and
    asserting it anyway would make the model stronger than the AIR -- the
    dangerous direction, since an over-strong model yields UNSAT where the real
    object is forgeable.
    """

    def __init__(self, tag: str, prem: Premises):
        self.tag, self.prem, self.n = tag, prem, 0
        self.C = []

        self.ts0, self.ts1 = self.col("ts0"), self.col("ts1")
        self.src0, self.src1 = self.col("src0"), self.col("src1")
        self.si = [self.col(f"si{i}") for i in range(4)]
        self.dst0, self.dst1 = self.col("dst0"), self.col("dst1")
        self.di = [self.col(f"di{i}") for i in range(4)]
        self.cnt0, self.cnt1 = self.col("cnt0"), self.col("cnt1")
        self.cd = [self.col(f"cd{i}") for i in range(4)]
        self.first = self.col("first")
        self.end = self.col("end")
        self.tail = self.col("tail")
        self.value = [self.col(f"value{i}") for i in range(8)]
        self.mu = self.col("mu")

        self._constraints()
        self._lookups()

    # -- plumbing ----------------------------------------------------------
    #
    # ENCODING NOTE. Two rewrites keep the model inside linear integer
    # arithmetic, where z3 is fast. Both are exact, not approximations:
    #
    #   `a == b (mod p)`  becomes  `a - b == k*p` for a fresh unbounded integer
    #       k. Identical semantics to `(a - b) % p == 0`, but linear in the
    #       columns (p is a constant), so no div/mod machinery is introduced.
    #
    #   `x*(1 - x) == 0 (mod p)`  becomes  `x == 0 or x == 1`. Exact because p
    #       is prime and every column is already confined to [0, p): the
    #       product vanishes mod p only if one factor does, giving x = 0 or
    #       x = 1 in that interval. Keeping the literal quadratic instead makes
    #       every query nonlinear over a 64-bit prime, which is what the first
    #       version of this gate died of.
    #
    # Products of a BOOLEAN column with anything else are likewise rewritten as
    # implications (`mu*(...)` -> `Implies(mu == 1, ...)`), which is exact once
    # the column is known boolean.

    def col(self, name: str):
        """A committed column: a free field element in [0, p)."""
        v = Int(f"{self.tag}_{name}")
        self.C.append(And(v >= 0, v < P))
        return v

    def aux(self, name: str):
        """A virtual value (a carry, or another table's column). Also in [0,p)."""
        self.n += 1
        v = Int(f"{self.tag}_{name}_{self.n}")
        self.C.append(And(v >= 0, v < P))
        return v

    def feq(self, lhs, rhs):
        """`lhs == rhs` in the field, as `lhs - rhs == k*p`."""
        self.n += 1
        k = Int(f"{self.tag}_k{self.n}")
        self.C.append(lhs - rhs == k * P)

    def is_bit(self, x):
        """`x*(1-x) = 0` for a column already in [0, p): x is 0 or 1."""
        self.C.append(Or(x == 0, x == 1))

    def scoped(self, condition, body):
        """Assert what `body` appends only where `condition` holds.

        Used for every bus lookup, so that a multiplicity-zero interaction
        contributes nothing.
        """
        mark = len(self.C)
        body()
        added, self.C[mark:] = self.C[mark:], []
        self.C.append(Implies(condition, And(*added)))

    # -- packings ----------------------------------------------------------
    def wl(self, lo, hi):
        return lo + B32 * hi

    def hl_lo(self, h):
        return h[0] + B16 * h[1]

    def hl_hi(self, h):
        return h[2] + B16 * h[3]

    def hl(self, h):
        return self.hl_lo(h) + B32 * self.hl_hi(h)

    @property
    def step_lo(self):
        """`step = 8 - 7*tail`, the `AddOperand::linear` in `DmaConstraints`."""
        return 8 - 7 * self.tail

    @property
    def src(self):
        return self.wl(self.src0, self.src1)

    @property
    def dst(self):
        return self.wl(self.dst0, self.dst1)

    @property
    def count(self):
        return self.wl(self.cnt0, self.cnt1)

    @property
    def src_incr(self):
        return self.hl(self.si)

    @property
    def dst_incr(self):
        return self.hl(self.di)

    @property
    def count_decr(self):
        return self.hl(self.cd)

    @property
    def width(self):
        return If(self.tail == 1, IntVal(TAIL_WIDTH), IntVal(WIDE_WIDTH))

    # -- eval constraints --------------------------------------------------
    def _add_pair(self, lhs_lo, lhs_hi, rhs_lo, rhs_hi, sum_lo, sum_hi,
                  name, no_overflow):
        """`templates::emit_add_pair[_no_overflow]`.

        `carry_0 = (lhs.lo + rhs.lo - sum.lo) * 2^-32` is always boolean.
        `carry_1 = (lhs.hi + rhs.hi + carry_0 - sum.hi) * 2^-32` is boolean in
        the plain form, and forced to ZERO on active non-terminal rows
        (`mu - end == 1`) in the no-overflow form -- leaving it a free field
        element on terminal and padding rows, whose successor is not consumed.
        """
        c0 = self.aux(f"{name}_c0")
        self.feq(lhs_lo + rhs_lo - sum_lo, c0 * B32)
        self.is_bit(c0)
        c1 = self.aux(f"{name}_c1")
        self.feq(lhs_hi + rhs_hi + c0 - sum_hi, c1 * B32)
        if no_overflow:
            # `(mu - end) * carry_1 == 0`, with mu and end boolean.
            self.C.append(Implies(self.mu - self.end == 1, c1 == 0))
        else:
            self.is_bit(c1)

    def _constraints(self):
        """`DmaConstraints::eval`, index by index."""
        for x in (self.first, self.end, self.tail, self.mu):   # idx 0-3
            self.is_bit(x)
        # idx 4: `(first + end)*(1 - mu) == 0` -- an inactive row cannot claim
        # first or end. With all three boolean this is exactly:
        self.C.append(Implies(self.mu == 0, And(self.first == 0, self.end == 0)))
        # idx 5-6, 7-8: src and dst advance by `step` without wrapping 2^64
        self._add_pair(self.src0, self.src1, self.step_lo, 0,
                       self.hl_lo(self.si), self.hl_hi(self.si), "src",
                       self.prem.no_overflow_src)
        self._add_pair(self.dst0, self.dst1, self.step_lo, 0,
                       self.hl_lo(self.di), self.hl_hi(self.di), "dst",
                       self.prem.no_overflow_dst)
        # idx 9-10: count_decr + step = count. The PLAIN pair, so it MAY wrap --
        # which is what lets the terminal row hold `0 - 1`.
        self._add_pair(self.hl_lo(self.cd), self.hl_hi(self.cd), self.step_lo, 0,
                       self.cnt0, self.cnt1, "cnt", False)
        # idx 11-17: `tail * value[i] == 0` -- unused lanes are zero on a
        # one-byte row (`tail` boolean, so the product form is this):
        if self.prem.tail_lane_zero:
            for lane in self.value[1:]:
                self.C.append(Implies(self.tail == 1, lane == 0))

    # -- lookups -----------------------------------------------------------
    def _lt(self, lhs_lo, lhs_hi, rhs_lo, name):
        """`Alu[lhs, rhs, LT] -> out`, as `lt.rs`'s own constraints.

        `lhs`/`rhs` cross the bus as DWordHHW -> [lo32, hi32]. The hi limb is
        `LHS_1 + 2^16*LHS_2` with both halves range-checked (IsHalfword on
        `[1]`, MSB16 -- whose argument is a halfword -- on `[2]`), so the hi
        limb is genuinely 32-bit. `LHS_0` is a bare `Word` column, pinned only
        through the carry relation; keeping that faithful is the whole point of
        modelling the table instead of its contract.
        """
        sub = [self.aux(f"{name}_sub{i}") for i in range(4)]
        for h in sub:
            self.C.append(h < B16)                      # IsHalfword[sub[i]]
        h1, h2 = self.aux(f"{name}_h1"), self.aux(f"{name}_h2")
        self.C.append(h1 < B16)                         # IsHalfword[lhs[1]]
        self.C.append(h2 < B16)                         # MSB16[lhs[2]]
        self.feq(lhs_hi, h1 + B16 * h2)

        sub_lo, sub_hi = sub[0] + B16 * sub[1], sub[2] + B16 * sub[3]
        c0 = self.aux(f"{name}_c0")
        self.feq(rhs_lo + sub_lo - lhs_lo, c0 * B32)
        self.is_bit(c0)
        c1 = self.aux(f"{name}_c1")
        self.feq(0 + sub_hi + c0 - lhs_hi, c1 * B32)
        self.is_bit(c1)
        return c1                                       # unsigned lt == carry_1

    def _zero(self, arg, out):
        """`Zero[arg] -> out`, as the receiving `bitwise.rs` row.

        `arg` must decompose as `x + 256y + 65536z` with x,y bytes and z in
        [0,16), i.e. `arg` must lie IN THE TABLE'S DOMAIN [0, 2^20). An `arg`
        outside it has no partner row, which is a completeness failure rather
        than a soundness hole -- but it means the halfword bounds on
        `count_decr` are doing two jobs at once, and the width audit separates
        them.
        """
        x, y, z = self.aux("zx"), self.aux("zy"), self.aux("zz")
        self.C.append(x < 256)
        self.C.append(y < 256)
        self.C.append(z < 16)
        self.feq(arg, x + 256 * y + 65536 * z)
        self.feq(out, If(And(x == 0, y == 0, z == 0), IntVal(1), IntVal(0)))

    def _lookups(self):
        p, active = self.prem, self.mu == 1
        # IsHalfword senders, multiplicity mu.
        for columns, present in ((self.cd, p.halfword_count_decr),
                                 (self.si, p.halfword_src_incr),
                                 (self.di, p.halfword_dst_incr)):
            if present:
                for h in columns:
                    self.C.append(Implies(active, h < B16))
        # MEMW data ops, multiplicity mu - end: bind the base-address limbs.
        if p.memw_addr32:
            for limb in (self.src0, self.src1, self.dst0, self.dst1):
                self.C.append(Implies(self.mu - self.end == 1, limb < B32))
        # MEMW register reads, multiplicity first: bind all three arguments.
        if p.reg32:
            for limb in (self.src0, self.src1, self.dst0, self.dst1,
                         self.cnt0, self.cnt1):
                self.C.append(Implies(self.first == 1, limb < B32))
        # Bus 16: end detection.
        if p.zero_end:
            self.scoped(active, lambda: self._zero(ZERO_SUM - Sum(self.cd), self.end))
        # Bus 20: tail = (count < 8).
        if p.lt_tail:
            self.scoped(active, lambda: self.feq(
                self.tail, self._lt(self.cnt0, self.cnt1, IntVal(WIDE_WIDTH), "lt_tail")))
        # Bus 21: the first row proves count < MAX + 1.
        if p.lt_bound:
            self.scoped(self.first == 1, lambda: self.feq(
                1, self._lt(self.cnt0, self.cnt1, IntVal(MAX_BYTES + 1), "lt_bound")))

    # -- the reference and the invariant -----------------------------------
    def reference(self):
        """What an active row must do, from `dma_ref.row_columns`.

        Stated over the INTEGERS, which is only meaningful where the limbs are
        genuine 32-bit words -- `well_formed()` is that hypothesis, and
        `check_invariant_propagates` is what shows it holds chain-wide.

        NOTE the `count_decr` clause is spelled as an explicit two-way
        disjunction rather than through `eq_mod`. This predicate is asserted
        NEGATED, and a modular equality carrying a free witness quotient becomes
        vacuously satisfiable under negation (pick a nonzero quotient) -- which
        is exactly how the first run of this gate reported a bogus SAT. Both
        representatives are in range here (`count_decr` is a bounded dword and
        `count - width` lies in `[-8, 2^64)`), so the disjunction is exact and
        witness-free.
        """
        return And(
            self.tail == If(self.count < WIDE_WIDTH, IntVal(1), IntVal(0)),
            self.end == If(self.count == 0, IntVal(1), IntVal(0)),
            Or(self.count_decr == self.count - self.width,
               self.count_decr == self.count - self.width + B64),
            Implies(self.end == 0,
                    And(self.src_incr == self.src + self.width,
                        self.src + self.width < B64,
                        self.dst_incr == self.dst + self.width,
                        self.dst + self.width < B64)),
        )

    def well_formed(self):
        """The limbs are genuine 32-bit words.

        Supplied on the head row by REG-32 and on every data row by
        MEMW-ADDR32 (addresses only); for `count` on a non-head row it is a
        CONCLUSION, not a hypothesis -- see `check_invariant_propagates`.
        """
        return And(*[x < B32 for x in
                     (self.src0, self.src1, self.dst0, self.dst1,
                      self.cnt0, self.cnt1)])


# ---------------------------------------------------------------------------
# Layer 1 checks
# ---------------------------------------------------------------------------

def check_row(prem=None, timeout_ms=180_000):
    """MAIN 0 -- an active, well-formed row does exactly what the oracle says."""
    prem = prem or Premises()
    r = FieldRow("row", prem)
    return solve(r.C + [r.mu == 1, r.well_formed(), Not(r.reference())], timeout_ms)


def check_end_detection(prem=None, timeout_ms=180_000):
    """MAIN 1 -- `end` fires if and only if `count == 0`.

    Split out from MAIN 0 because it is the single thing standing between the
    table and a silently truncated copy: an `end` row's memory sends have
    multiplicity `mu - end = 0`, so a row that wrongly claims `end` emits no
    reads and no writes at all, and every bus still balances.
    """
    prem = prem or Premises()
    r = FieldRow("endr", prem)
    wrong = Or(And(r.end == 1, r.count != 0), And(r.end == 0, r.count == 0))
    return solve(r.C + [r.mu == 1, r.well_formed(), wrong], timeout_ms)


def check_wrap_only_terminal(prem=None, timeout_ms=180_000):
    """MAIN 2 -- the `count` subtraction wraps only on the terminal row.

    The lemma the chain argument rests on. `count_decr` uses the PLAIN add pair,
    so `count - width` is allowed to wrap modulo 2^64; if it could wrap on a row
    that still sends to `DmaNext`, `count` would stop being strictly decreasing
    along the chain and a cycle of rows that balances the bus while copying
    nothing (or copying twice) becomes thinkable. UNSAT says a wrapping row
    always has `end = 1`, and an `end` row sends nothing.
    """
    prem = prem or Premises()
    r = FieldRow("wrap", prem)
    return solve(r.C + [r.mu == 1, r.well_formed(),
                        r.count < r.width, r.end == 0], timeout_ms)


def check_tail_lanes(prem=None, timeout_ms=180_000):
    """MAIN 2b -- a one-byte row carries seven zero lanes.

    LABELLED HONESTLY: this is a TRANSCRIPTION check, not a composed solver
    result. It is UNSAT from `Implies(tail == 1, lane == 0)` alone, with no other
    AIR fact participating, and its negative control is `sat` for the same
    trivial reason. Kept because the property matters and the pair documents it,
    but it earns no credit as evidence about the constraint system -- the textual
    equivalent in `../audit_gate_transcription.py` is the real guard.

    `value[1..8]` ride the MEMW tuple of a `w8 = 1 - tail` operation, so on a
    tail row the memory table must see the canonical one-byte encoding. Nothing
    else pins those lanes: they are not XOR-consumed and not range-checked, so
    without constraints 11-17 they are free field elements appearing in a bus
    tuple -- the aliasing shape `keccak.rs` documents for its address bytes.
    """
    prem = prem or Premises()
    r = FieldRow("lanes", prem)
    return solve(r.C + [r.mu == 1, r.tail == 1,
                        Or(*[lane != 0 for lane in r.value[1:]])], timeout_ms)


def check_row_budget(prem=None, timeout_ms=180_000):
    """MAIN 2c -- one ecall cannot ask for more than MAX_BYTES bytes.

    This is the bound that keeps a single guest instruction from adding an
    unbounded number of rows to a continuation epoch, and it is the only claim
    in the table that needs BOTH the first-row LT lookup (bus 21) and REG-32:
    the lookup caps the packed count, and REG-32 is what makes the packed count
    a genuine 64-bit integer rather than one representative of a residue class.
    Deliberately does NOT assume `well_formed()` -- that would hand REG-32 to
    the query for free and make its control vacuous.
    """
    prem = prem or Premises()
    r = FieldRow("budget", prem)
    return solve(r.C + [r.mu == 1, r.first == 1, r.count > MAX_BYTES], timeout_ms)


def check_invariant_propagates(prem=None, timeout_ms=600_000):
    """MAIN 3 -- well-formedness and the exact count cross a `DmaNext` hop.

    `DmaNext` binds each 64-bit value as TWO 32-bit elements, not one packed
    field element (see `dmanext_link`, which documents the error an earlier
    version of this gate made here). So the successor cannot re-split its limbs:
    with the sender's halfwords IsHalfword-bounded, the receiver's `count0` and
    `count1` are each pinned to a genuine 32-bit word.

    The claim is therefore unconditional -- no disjunctive escape branch:

        successor.count == predecessor.count - width   AND   successor well formed

    Proving it once on the head row (where the register read supplies
    well-formedness) proves it for the whole chain, which is what licenses
    Layer 2's integer abstraction.
    """
    prem = prem or Premises()
    a, b = FieldRow("inv_a", prem), FieldRow("inv_b", prem)
    link = [
        a.mu == 1, a.end == 0, a.well_formed(), a.count <= MAX_BYTES,
        b.mu == 1, b.first == 0,
    ] + dmanext_link(a, b)
    holds = And(b.count == a.count - a.width, b.well_formed())
    return solve(a.C + b.C + link + [Not(holds)], timeout_ms)


def completeness_sweep(prem=None, quick=False):
    """MAIN 4 -- every honest trace is accepted (no false rejection).

    For each length the ORACLE pins every column of every row (plus the padding
    row the generator emits) and asks whether the constraint system is
    satisfiable. A failure is a completeness bug: the AIR would reject a copy
    the executor performs. This is also the gate's non-vacuity check -- if the
    constraint set were contradictory, every UNSAT above would be worthless.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "dma-oracle"))
    import dma_ref as ref

    prem = prem or Premises()
    lengths = list(range(0, MAX_BYTES + 1)) if not quick else [0, 1, 7, 8, 9, 16, 27, 255, 256]
    checked = 0
    for n in lengths:
        memory = {0x2000 + i: (i * 7 + 3) & 0xFF for i in range(n)}
        rows = ref.row_decomposition(0x30, 0x1000, 0x2000, n, memory)
        for index, row in enumerate(rows):
            r = FieldRow(f"cs{n}_{index}", prem)
            verdict = solve(r.C + _pins(r, ref.row_columns(row)))
            # `unknown` is a TIMEOUT, not a rejection. Scoring `!= sat` would print
            # "the AIR REJECTS an honest row" for a slow solver -- a false
            # accusation against the chip, and exactly the confusion the board's
            # legend exists to prevent. Distinguish the two.
            if verdict == unknown:
                return None, (f"n={n} row {index}: solver TIMED OUT "
                              f"(not a rejection -- raise the budget or use a newer z3)")
            if verdict != sat:
                return False, f"n={n} row {index}: the AIR REJECTS an honest row"
            checked += 1
        r = FieldRow(f"pad{n}", prem)
        verdict = solve(r.C + _pins(r, ref.padding_columns()))
        if verdict == unknown:
            return None, f"n={n}: solver TIMED OUT on the padding row (not a rejection)"
        if verdict != sat:
            return False, f"n={n}: the AIR REJECTS the padding row"
        checked += 1
    return True, (f"{checked - len(lengths)} honest rows + {len(lengths)} padding rows "
                  f"over {len(lengths)} lengths, all accepted")


def _pins(r: FieldRow, cols: dict):
    pins = [
        r.ts0 == cols["timestamp"][0], r.ts1 == cols["timestamp"][1],
        r.src0 == cols["src"][0], r.src1 == cols["src"][1],
        r.dst0 == cols["dst"][0], r.dst1 == cols["dst"][1],
        r.cnt0 == cols["count"][0], r.cnt1 == cols["count"][1],
        r.first == cols["first"], r.end == cols["end"],
        r.tail == cols["tail"], r.mu == cols["mu"],
    ]
    pins += [r.si[i] == cols["src_incr"][i] for i in range(4)]
    pins += [r.di[i] == cols["dst_incr"][i] for i in range(4)]
    pins += [r.cd[i] == cols["count_decr"][i] for i in range(4)]
    pins += [r.value[i] == cols["value"][i] for i in range(8)]
    return pins


# ===========================================================================
# Layer 2 -- the chain, with DmaNext as a free bijection
# ===========================================================================

class ChainRow:
    """One row, abstracted to the integer relations Layer 1 proved.

    Using the abstraction instead of re-deriving the field model keeps the
    multi-row query tractable. The composition is valid because MAIN 0/1/2/3
    establish exactly these relations on every reachable row (with R1 as the
    stated residual).
    """

    def __init__(self, tag):
        self.src, self.dst = Int(f"{tag}_src"), Int(f"{tag}_dst")
        self.count, self.first = Int(f"{tag}_count"), Int(f"{tag}_first")
        self.C = [
            And(self.src >= 0, self.src < B64),
            And(self.dst >= 0, self.dst < B64),
            And(self.count >= 0, self.count < B64),
            Or(self.first == 0, self.first == 1),
        ]

    @property
    def width(self):
        return If(self.count < WIDE_WIDTH, IntVal(TAIL_WIDTH), IntVal(WIDE_WIDTH))

    @property
    def end(self):
        return If(self.count == 0, IntVal(1), IntVal(0))

    @property
    def src_incr(self):
        return self.src + self.width

    @property
    def dst_incr(self):
        return self.dst + self.width

    @property
    def count_decr(self):
        return self.count - self.width

    def no_overflow(self):
        return Implies(self.end == 0,
                       And(self.src_incr < B64, self.dst_incr < B64))


def check_chain(n_rows, prem=None, timeout_ms=300_000, premises_only=False,
                drop_link=None):
    """CHAIN -- the only bus-balanced structure is the oracle's decomposition.

    `n_rows` active rows. `DmaNext` is NOT assumed to be a chain: every row with
    `end = 0` sends one tuple, every row with `first = 0` receives one, and bus
    balance means a BIJECTION between those sets -- modelled as a free injective
    index map plus tuple equality. A cycle, a fork, a skipped source row or a
    duplicated one would be just as balanced a priori. The Ecall bus supplies
    "exactly one row may be `first`" (the CPU sends its tuple once).

    SCOPE, stated precisely: the `Ecall` bus supplies at most one `first` row
    **per timestamp**, not one per trace -- a real trace with k DMA calls has k
    head rows. `ChainRow` carries no timestamp field, so this check cannot
    express the property that separates two calls' rows (the `ts` in both
    DmaNext tuples, which `../audit_gate_transcription.py` pins textually).
    What is proved is therefore: *among a group of active rows containing
    exactly one head row*, the only bus-balanced structure is the reference
    tiling. Dropping the single-head premise makes this `sat`, so the
    multi-call case is genuinely out of model rather than covered.

    The tiling predicate asserts four facts: data-row widths summing to the head
    count, every data interval inside `[src, src + count)`, the intervals
    pairwise disjoint, and `dst - src` constant. Together they say the copy moves
    byte `j` of the source to byte `j` of the destination, for every
    `j < count`, exactly once. (The greedy width rule is not concluded here --
    it is baked into `ChainRow.width` as a definition, discharged by MAIN 0.)
    """
    prem = prem or Premises()
    rows = [ChainRow(f"c{n_rows}_{i}") for i in range(n_rows)]
    C = [c for r in rows for c in r.C]
    if prem.no_overflow_src:
        C += [r.no_overflow() for r in rows]

    # Ecall bus: exactly one head.
    C.append(Sum([r.first for r in rows]) == 1)
    # DmaNext balance: #senders == #receivers.
    senders = [If(r.end == 0, IntVal(1), IntVal(0)) for r in rows]
    C.append(Sum(senders) == Sum([1 - r.first for r in rows]))

    # sigma[j] = the sender matched to receiver j; a distinct negative sentinel
    # for the head so `Distinct` still expresses injectivity over receivers.
    sigma = [Int(f"s{n_rows}_{j}") for j in range(n_rows)]
    for j, row in enumerate(rows):
        C.append(Implies(row.first == 1, sigma[j] == -1 - j))
        C.append(Implies(row.first == 0, And(sigma[j] >= 0, sigma[j] < n_rows)))
        for k, sender in enumerate(rows):
            # `drop_link` omits one field of the DmaNext tuple -- the Layer 2
            # negative control, showing the bijection's contents are what force
            # the tiling and not the bijection's mere existence.
            tuple_eqs = [sender.end == 0]
            if drop_link != "src":
                tuple_eqs.append(sender.src_incr == row.src)
            if drop_link != "dst":
                tuple_eqs.append(sender.dst_incr == row.dst)
            if drop_link != "count":
                tuple_eqs.append(sender.count_decr == row.count)
            C.append(Implies(And(row.first == 0, sigma[j] == k), And(*tuple_eqs)))
    C.append(Distinct(*sigma))

    head_count = Sum([If(r.first == 1, r.count, IntVal(0)) for r in rows])
    head_src = Sum([If(r.first == 1, r.src, IntVal(0)) for r in rows])
    head_skew = Sum([If(r.first == 1, r.dst - r.src, IntVal(0)) for r in rows])
    if prem.lt_bound:
        C.append(head_count <= MAX_BYTES)

    if premises_only:
        return solve(C, timeout_ms)

    data = lambda r: r.end == 0                                       # noqa: E731
    tiling = And(
        Sum([If(data(r), r.width, IntVal(0)) for r in rows]) == head_count,
        And(*[Implies(data(r), And(r.src >= head_src,
                                  r.src + r.width <= head_src + head_count))
              for r in rows]),
        And(*[Implies(And(data(rows[i]), data(rows[j])),
                      Or(rows[i].src + rows[i].width <= rows[j].src,
                         rows[j].src + rows[j].width <= rows[i].src))
              for i in range(n_rows) for j in range(i + 1, n_rows)]),
        And(*[Implies(data(r), r.dst - r.src == head_skew) for r in rows]),
    )
    return solve(C + [Not(tiling)], timeout_ms)


def check_chain_field(n_rows, prem=None, timeout_ms=900_000, premises_only=False):
    """CHAIN-F -- the same question, field-exact, at small depth.

    The integer chain takes the row abstraction on trust. This one does not: it
    builds `n_rows` full `FieldRow`s, links them with the field-level `DmaNext`
    bijection, anchors the head with REG-32 plus the bound lookup, and asks for
    any group whose copied byte total differs from the head count. Small `n`
    only -- these queries are nonlinear over a 64-bit prime and get expensive
    fast -- but it means the abstraction step is confirmed rather than assumed.
    """
    prem = prem or Premises()
    rows = [FieldRow(f"f{n_rows}_{i}", prem) for i in range(n_rows)]
    C = [c for r in rows for c in r.C]
    C += [r.mu == 1 for r in rows]
    C.append(Sum([r.first for r in rows]) == 1)
    C.append(Sum([If(r.end == 0, IntVal(1), IntVal(0)) for r in rows])
             == Sum([1 - r.first for r in rows]))
    C.append(Sum([r.end for r in rows]) == 1)

    sigma = [Int(f"fs{n_rows}_{j}") for j in range(n_rows)]
    for j, row in enumerate(rows):
        C.append(Implies(row.first == 1, sigma[j] == -1 - j))
        C.append(Implies(row.first == 0, And(sigma[j] >= 0, sigma[j] < n_rows)))
        for k, sender in enumerate(rows):
            C.append(Implies(And(row.first == 0, sigma[j] == k),
                             And(*([sender.end == 0] + dmanext_link(sender, row)))))
    C.append(Distinct(*sigma))
    # Head anchor, and NOTHING MORE. Only the head row gets well-formedness (the
    # register read) and a count bound (the `Alu` lookup at multiplicity
    # `first`); every other row's limbs and count are *derived* through the
    # limb-wise link, per MAIN 3. An earlier version asserted
    # `r.count <= MAX_BYTES` for EVERY row in order to sidestep a phantom
    # residual -- which was the one genuinely over-strong assertion in this gate,
    # and over-strong assertions are the direction that yields a bogus UNSAT.
    for r in rows:
        C.append(Implies(r.first == 1, r.well_formed()))

    head_count = Sum([If(r.first == 1, r.count, IntVal(0)) for r in rows])
    covered = Sum([If(r.end == 0, r.width, IntVal(0)) for r in rows])
    if premises_only:
        # Positive control: is the premise set satisfiable at all? `Not(tiling)`
        # returning UNSAT is worthless if `C` is itself contradictory.
        return solve(C, timeout_ms)
    return solve(C + [covered != head_count], timeout_ms)


# ===========================================================================
# Width audit -- field-level bound necessity at the concrete boundary
# ===========================================================================

def audit_end_detection_bound(drop_bound: bool):
    """Is `sum(count_decr) == 4*65535  <=>  count_decr == 0xFFFF_FFFF_FFFF_FFFF`?

    The Zero send collapses four halfwords into ONE sum. With the IsHalfword
    bounds the only way to reach `4*65535` is all four at `0xFFFF`. Drop them
    and `(0xFFFF+d, 0xFFFF-d, 0xFFFF, 0xFFFF)` hits the same sum with a totally
    different `count_decr`, so `end` can be claimed at a nonzero count -- and an
    `end` row emits no memory operations. 'unsat' means the identity holds.
    """
    s = Solver()
    s.set("timeout", DEFAULT_TIMEOUT_MS)   # never leave a query unbounded (see above)
    cd = [Int(f"ae_cd{i}") for i in range(4)]
    for h in cd:
        s.add(h >= 0, h < P)
        if not drop_bound:
            s.add(h < B16)
    s.add(eq_mod(Sum(cd), ZERO_SUM, P))
    # The forged quantity is compared through its RESIDUE, not through a negated
    # modular equality: negating an equality that carries a free witness
    # quotient is vacuous (see `FieldRow.reference`).
    residue = Int("ae_res")
    s.add(residue >= 0, residue < P)
    s.add(eq_mod(cd[0] + B16 * cd[1] + B32 * cd[2] + B32 * B16 * cd[3], residue, P))
    s.add(residue != (B64 - 1) % P)
    return str(s.check())


def audit_no_overflow_bound(drop_bound: bool):
    """Does `carry_1 == 0` really mean `src + width < 2^64`?

    `carry_1 = (src1 + carry_0 - src_incr.hi) * 2^-32`, so `carry_1 == 0` says
    `src_incr.hi == src1 + carry_0` -- and at `src1 = 2^32 - 1` with a carry
    that is exactly `2^32`, which the IsHalfword pair forbids and an unbounded
    pair does not. The row then hands on a WRAPPED address that the executor's
    `checked_add` would have rejected. 'unsat' means the bound pins it.
    """
    s = Solver()
    s.set("timeout", DEFAULT_TIMEOUT_MS)   # never leave a query unbounded (see above)
    src0, src1, c0 = Int("an_src0"), Int("an_src1"), Int("an_c0")
    si = [Int(f"an_si{i}") for i in range(4)]
    step = WIDE_WIDTH
    s.add(src0 >= 0, src0 < B32, src1 >= 0, src1 < B32)
    for h in si:
        s.add(h >= 0, h < P)
        if not drop_bound:
            s.add(h < B16)
    s.add(Or(c0 == 0, c0 == 1))
    s.add(eq_mod(src0 + step - (si[0] + B16 * si[1]), c0 * B32, P))
    s.add(eq_mod(src1 + c0, si[2] + B16 * si[3], P))         # carry_1 == 0
    s.add(src0 + B32 * src1 + step >= B64)                   # the range DOES wrap
    return str(s.check())


def audit_tail_pin(drop_pin: bool):
    """Is the LT lookup the only thing stopping a SEVEN-byte truncation?

    `end` is claimed via the Zero check, which needs `count_decr` to be
    all-`0xFFFF`, i.e. `count == step - 1`. With `tail` free a row may take
    `tail = 0`, hence `step = 8`, hence `count == 7` satisfies it -- and an
    `end` row emits NO memory operations at all, because both its MEMW sends
    have multiplicity `mu - end`. Seven requested bytes are silently not copied
    while every bus balances.

    The count is 7 and not some smaller number for a reason worth recording:
    the two constraints compose, so `count = step - 1` is the ONLY reachable
    forgery here, and `step` in {1, 8} makes 7 the only value a free `tail` buys.
    'unsat' means the LT pin blocks it.
    """
    r = FieldRow("tp" + ("_drop" if drop_pin else ""),
                 Premises(lt_tail=not drop_pin))
    return str(solve(r.C + [r.mu == 1, r.well_formed(), r.count == 7, r.end == 1]))


# ===========================================================================

def check_solver_version():
    """Warn loudly if the solver is older than the one this board was green on."""
    current = get_version()[:3]
    if current < VALIDATED_Z3:
        print(f"  !! z3 {get_version_string()} is older than the validated "
              f"{'.'.join(map(str, VALIDATED_Z3))}.", flush=True)
        print("  !! The queries mean the same thing, but older solvers are much "
              "slower on the", flush=True)
        print("  !! field-exact chain and may report `unknown` (= TIMED OUT, "
              "scored as failure).", flush=True)
        print("  !! An `unknown` is a budget problem, NOT a soundness problem.",
              flush=True)
        return False
    return True


def main():
    quick = "--quick" in sys.argv
    print("=" * 76, flush=True)
    print("DMA memcpy chip -- z3 gate" + ("  (--quick)" if quick else ""), flush=True)
    print("=" * 76, flush=True)
    print(f"  solver: z3 {get_version_string()}", flush=True)
    check_solver_version()
    print("  legend: unsat = proved | sat = counterexample found | "
          "unknown = TIMED OUT (failure)", flush=True)

    print("\n=== LAYER 1: field-exact rows ===", flush=True)
    row = check_row()
    print(f"  MAIN 0  row == oracle row                 -> {row}   (want unsat)", flush=True)
    endd = check_end_detection()
    print(f"  MAIN 1  end <=> count == 0                -> {endd}   (want unsat)", flush=True)
    wrap = check_wrap_only_terminal()
    print(f"  MAIN 2  count wraps only on terminal row  -> {wrap}   (want unsat)", flush=True)
    lanes = check_tail_lanes()
    print(f"  MAIN 2b one-byte row has zero lanes 1..7  -> {lanes}   (want unsat)", flush=True)
    budget = check_row_budget()
    print(f"  MAIN 2c one ecall asks for <= {MAX_BYTES} bytes  -> {budget}   (want unsat)", flush=True)
    inv = check_invariant_propagates()
    print(f"  MAIN 3  successor exact + well formed     -> {inv}   (want unsat)", flush=True)
    layer1_ok = all(x == unsat for x in (row, endd, wrap, lanes, budget, inv))

    print("\n=== LAYER 2: chain structure, DmaNext as a free bijection ===", flush=True)
    chain = {}
    for n_rows in ((2, 3, 4) if quick else (2, 3, 4, 5)):
        chain[n_rows] = check_chain(n_rows)
        print(f"  CHAIN   {n_rows} rows, any balanced structure   -> {chain[n_rows]}   (want unsat)", flush=True)
    field_chain = {}
    for n_rows in ((2,) if quick else (2, 3)):
        field_chain[n_rows] = check_chain_field(n_rows)
        print(f"  CHAIN-F {n_rows} rows, field-exact              -> {field_chain[n_rows]}   (want unsat)", flush=True)
    layer2_ok = all(x == unsat for x in list(chain.values()) + list(field_chain.values()))

    # Layer 2 needs its own controls. `Not(tiling)` returning unsat proves
    # nothing if the premise set is itself unsatisfiable, and an earlier version
    # of this board had neither a positive nor a negative control here.
    print("\n  -- Layer 2 controls --", flush=True)
    l2_pos = {n: check_chain(n, premises_only=True) for n in (2, 3, 4)}
    l2_posf = check_chain_field(2, premises_only=True)
    for n, res in l2_pos.items():
        print(f"  positive: {n}-row premise set satisfiable  -> {res}   (want sat)", flush=True)
    print(f"  positive: 2-row field-exact premise set   -> {l2_posf}   (want sat)", flush=True)
    l2_neg = {f: check_chain(3, drop_link=f) for f in ("count", "src", "dst")}
    for field, res in l2_neg.items():
        print(f"  negative: drop `{field}` from the tuple{'':<7}-> {res}   (want sat)", flush=True)
    layer2_controls_ok = (all(r == sat for r in l2_pos.values()) and l2_posf == sat
                          and all(r == sat for r in l2_neg.values()))

    print("\n=== NEGATIVE CONTROLS -- drop one premise, expect a forgery ===", flush=True)
    # Each control drops ONE premise and re-runs the check that premise is
    # load-bearing for. Pairing matters: dropping `tail_lane_zero` and re-running
    # MAIN 0 would report unsat, because MAIN 0's reference says nothing about
    # the value lanes -- a control that cannot fail is not a control.
    controls = {
        "drop_halfword_count_decr": check_row(Premises(halfword_count_decr=False)),
        "drop_halfword_src_incr": check_row(Premises(halfword_src_incr=False)),
        "drop_zero_end": check_end_detection(Premises(zero_end=False)),
        "drop_lt_tail": check_row(Premises(lt_tail=False)),
        "drop_no_overflow_src": check_row(Premises(no_overflow_src=False)),
        "drop_tail_lane_zero": check_tail_lanes(Premises(tail_lane_zero=False)),
        "drop_lt_bound": check_row_budget(Premises(lt_bound=False)),
        "drop_reg32": check_row_budget(Premises(reg32=False)),
        # Previously undropped premises. `halfword_dst_incr` and `no_overflow_dst`
        # are the dst-side mirrors of checks only ever demonstrated on src, and
        # `DESIGN.md`'s "all twelve halfwords, each one" needs all three families.
        "drop_halfword_dst_incr": check_row(Premises(halfword_dst_incr=False)),
        "drop_no_overflow_dst": check_row(Premises(no_overflow_dst=False)),
    }
    for name, res in controls.items():
        print(f"  {name:28s} -> {res}   (want sat)", flush=True)
    controls_ok = all(res == sat for res in controls.values())

    print("\n=== WIDTH AUDIT -- bound necessity at the boundary (field level) ===", flush=True)
    audit = {
        "Zero sum identity, bounds present": (audit_end_detection_bound(False), "unsat"),
        "Zero sum identity, bounds DROPPED": (audit_end_detection_bound(True), "sat"),
        "no-overflow, halfword bounds present": (audit_no_overflow_bound(False), "unsat"),
        "no-overflow, halfword bounds DROPPED": (audit_no_overflow_bound(True), "sat"),
        "truncation at count=7, LT pin present": (audit_tail_pin(False), "unsat"),
        "truncation at count=7, LT pin DROPPED": (audit_tail_pin(True), "sat"),
    }
    for name, (got, want) in audit.items():
        print(f"  {name:40s} -> {got:6s} (want {want})", flush=True)
    audit_ok = all(got == want for got, want in audit.values())

    print("\n=== POSITIVE CONTROLS -- oracle-pinned completeness sweep ===", flush=True)
    sweep_ok, sweep_detail = completeness_sweep(quick=quick)
    label = {True: "PASS", False: "FAIL", None: "TIMEOUT"}[sweep_ok]
    print(f"  {label}  {sweep_detail}", flush=True)

    print("\n" + "=" * 76, flush=True)
    print("VERDICT", flush=True)
    print("=" * 76, flush=True)
    print(f"  layer 1 (row semantics)              : {layer1_ok}", flush=True)
    print(f"  layer 2 (chain structure)            : {layer2_ok}", flush=True)
    print(f"  layer 2 controls (pos + neg)         : {layer2_controls_ok}", flush=True)
    print(f"  negative controls all SAT            : {controls_ok}   "
          f"({sum(1 for r in controls.values() if r == sat)}/{len(controls)})")
    print(f"  width audit (bound necessity)        : {audit_ok}", flush=True)
    print(f"  completeness sweep SAT               : {sweep_ok}", flush=True)
    print("\n  Scope: Layer 2 proves the tiling among groups with exactly ONE head", flush=True)
    print("  row. Two DMA calls are separated by the `ts` in both DmaNext tuples,", flush=True)
    print("  which `ChainRow` does not model -- see `check_chain`'s docstring and", flush=True)
    print("  the textual guard in ../audit_gate_transcription.py.", flush=True)
    ok = (layer1_ok and layer2_ok and layer2_controls_ok and controls_ok
          and audit_ok and sweep_ok is True)
    if quick:
        print("\n  NOTE: --quick shortened the completeness sweep and the chain depths.", flush=True)
    print(f"\n  OVERALL: {'PASS' if ok else 'FAIL -- investigate above'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
