"""
LAYER 4: the COLUMN-ROLE MAP, executable -- and THE SEAM.

============================================================================
THIS FILE IS THE PHASE-2 SPECIFICATION.
============================================================================
Every committed column of the BLAKE3 arm of `LFM_HASH` appears here as a free
variable, and every constraint the chip must impose appears here as an equation
over those variables.  A chip that conforms to this file is one the gate proves
correct; a chip that does not conform is one the gate says nothing about.

============================================================================
THE SEAM -- how the real chip plugs in after Phase 2
============================================================================
The gate (`gate.py`) touches this module ONLY through `SocketChip`'s public
surface:

    chip = SocketChip(tag, framing)         # allocate columns
    chip.build()                            # emit every constraint
    chip.in_lane_bytes  ->  8 x [4 byte columns]     (the socket's two input cells)
    chip.digest_words   ->  4 x [4 byte columns]     (the socket's one output cell)
    chip.assertions     ->  the constraint system

To validate the REAL chip, replace the bodies of the `emit_*` methods with a
transcription of the corresponding arms of `HashConstraints::eval` (the BLAKE3
arm added in Phase 2), keeping the same method signatures.  Nothing else in the
gate changes.  Each `emit_*` carries a `CHIP CONSTRAINT` comment naming the
exact constraint the Rust body must contain; that comment is the conformance
checklist.

Deliberately, the model is written in terms of the same primitives the Rust body
will use -- byte columns, mu-gated linear identities, ByteAlu/AreBytes sends --
rather than in terms of 32-bit arithmetic.  A model written at word level would
be easy to make UNSAT and would prove nothing about the chip that exists.

============================================================================
MU-GATING
============================================================================
Every eval constraint in the real chip is multiplied by the MU column (1 on a
real compression row, 0 on padding) and every bus send carries
`Multiplicity::Column(MU)`; padding rows are all-zero.  The gate models a REAL
row, so mu = 1 and drops out.  MU's own obligations -- booleanity, and the
all-zero-padding property -- are NOT BV theorems; they are checked structurally
and recorded in ORACLE.md's degree ledger.
"""

from __future__ import annotations

from z3 import BitVecVal

import blake3_oracle as ora
import socket_ref as sk
from contracts import WIDE, BvContracts

# The eight G-calls of a round, as (a, b, c, d, mx_index, my_index).
G_CALLS = ora.G_CALLS


class ColumnCensus:
    """Cost accounting, kept in lockstep with the model so the numbers in
    ORACLE.md cannot drift from the constraints that were actually gated.

    Sends are NOT counted here: they are counted once, at the point of issue, by
    `BvContracts` (`byte_xor` and `are_bytes`), because a send is a lookup and a
    lookup only exists where a contract is invoked. A second counter incremented
    by hand at the call sites is how a census silently double-counts one family
    and drops another."""

    def __init__(self, contracts):
        self.main = 0          # committed main columns (cells)
        self.by_block: dict[str, int] = {}
        self._c = contracts
        self.io_sends = 0      # the host socket's LfmMem tuples

    def add(self, block: str, n: int):
        self.main += n
        self.by_block[block] = self.by_block.get(block, 0) + n

    @property
    def sends(self) -> int:
        return self._c.sends + self.io_sends

    def aux_cells(self) -> int:
        # LogUp aux width: 3 extension columns per pair of sends (the verified
        # Tier-2 cost model: a send costs ~1.5 base cells of aux).
        return 3 * ((self.sends + 1) // 2)

    def cell_equiv(self) -> int:
        return self.main + self.aux_cells()


class SocketChip:
    """The BLAKE3 arm of `LFM_HASH`: one row = one 2-to-1 compress."""

    def __init__(self, tag: str, framing: sk.Framing = sk.HONEST_7,
                 bug: str | None = None, tail_truncate: bool = False):
        self.fr = framing
        self.bug = bug
        self.tail_truncate = tail_truncate
        self.c = BvContracts(tag)
        self.census = ColumnCensus(self.c)
        self.in_lane_bytes: list[list] = []
        self.digest_words: list[list] = []
        self._built = False

    # -- convenience ------------------------------------------------------
    @property
    def assertions(self):
        return self.c.assertions

    def _bug(self, name: str, flag: bool = True) -> bool:
        return self.bug == name and flag

    # =====================================================================
    # BLOCK 0 -- socket I/O (shared with the LFM_HASH host)
    # =====================================================================
    # AS BUILT: the frozen 28-column shared prefix -- 12 `IN` + 4 `S` + 12 `OUT`
    # (of which 4 `IN` lanes and 8 `OUT` lanes are unused on a Compress row).
    # `MU = MODE_C` is a PREPROCESSED column, so it is outside the main-column
    # census entirely AND a prover cannot choose it. R1 is satisfied: the arm
    # re-exports `cols::{IN0, OUT0, S8}` rather than committing a second copy.
    #
    # These are felts, not bytes, so BLOCK 0 is not modelled in the BV domain.
    # The chip emits FOUR framing constraint families here that the pre-Phase-2
    # model did not cover; all four are over felts and mode selectors, so they
    # go to the FIELD/structural ledger and are checked in gate.py's
    # `audit_block0_*`, NOT in BV:
    #
    #   idx 0-3   S_k - (MODE_P*IN_{8+k} + MU*IV_k)   capacity prefix; MU is the
    #             FULL three-way sum -- a leaf row is a compress in framing too
    #   idx 4     mode_sum*(1 - mode_sum), mode_sum = MODE_C+MODE_T+MODE_P
    #   idx 5     MODE_P = 0                              no permute socket, PERMANENT
    #   idx 14-21 OUT_{4+j} = 0, j in 0..8                digest is ONE cell
    #   idx 22-25 digest recomposition
    #   idx 26-33 UNREAD INPUT PINS -- 8, both unread cells (was 4, one cell,
    #             before the D1 fix). Shared helper `emit_unread_input_pins`,
    #             derived ONCE from HashMode::num_input_cells:
    #               slot 1 (IN4..8):  modes with <=1 input cell  -> MODE_L
    #               slot 2 (IN8..12): modes with <=2 input cells -> MODE_L+MODE_C+MODE_T
    #   idx 34-49 the LEAF block (LEAF_IDX = UNREAD_IDX + NUM_UNREAD_INPUT_PINS)
    #   idx 50+   the mixing core (CORE_IDX)
    # NUM_CONSTRAINTS = 26 + 8 + 16 + 16*NUM_G = 946 @7r (was 942).
    #
    # AS BUILT POST-MODE_L (layout::hash): PREP_WIDTH = 13, MODE_C = 6,
    # MODE_P = 7, MODE_T = 8, MODE_L = 9, MULT0..2 = 10..12, NUM_SELECTORS = 4.
    # Every selector sits INSIDE the contiguous run read from MODE_C, because the
    # admission validator's one-hot check reads that span -- a selector parked
    # past the mults would be outside the check and silently unchecked.
    #
    # ⚠ TWO DIFFERENT MULTIPLICITIES, and the distinction is load-bearing:
    #   MU_COLUMNS          = MODE_C + MODE_T + MODE_L   (the is-real gate; also
    #                         the multiplicity on EVERY BITWISE send)
    #   DIGEST_MODE_COLUMNS = MODE_C + MODE_T            (gates idx 6-13 only)
    #
    # O1 IS TWO OBLIGATIONS AND ONLY ONE OF THEM NARROWED:
    #   * the LANE IDENTITY (IN_lane == m[lane]) narrowed to the digest modes.
    #     Correct: on a leaf row the eight lanes are four felts' HALVES, so
    #     IN_lane and m[lane] are deliberately different field elements, and
    #     gating this on the full mu would make every leaf row unprovable.
    #   * the AreBytes RANGE BOUND did NOT narrow -- ✓ VERIFIED the lane sends
    #     carry `Multiplicity::Sum3(MODE_C, MODE_T, MODE_L)`, so all 32 lane byte
    #     columns are bounded on leaf rows too.
    #
    # That second point is what makes the leaf block sound: canonicity ASSUMES
    # lo, hi < 2^32 and does not establish it. Had the range bound narrowed with
    # the identity, leaf halves would be unbounded field elements and the whole
    # canonicity gate would be vacuous. Audited as WA9.
    #
    # idx 0-3, 4, 5 and 14-21 are all UNGATED (no MU factor), which is correct:
    # they must hold on padding rows too, and padding is all-zero.
    #
    # The dependency worth executing, and the reason these are not merely
    # "structural": idx 0-3 only PIN anything because idx 5 kills the MODE_P
    # term. Without idx 5 the capacity prefix is a prover-chosen copy of
    # IN_{8+k}. That is checked, both ways, by `audit_block0_capacity`.

    # =====================================================================
    # BLOCK 1 -- the lane boundary (THE new soundness surface for Route A)
    # =====================================================================
    def emit_lane_bytes(self):
        """Columns: MB[j][k], j in 0..8 lanes, k in 0..4 bytes = 32 byte columns.

        CHIP CONSTRAINT (per lane j), mu-gated, degree 2:
            MU * ( LANE_j - (MB[j][0] + 2^8*MB[j][1] + 2^16*MB[j][2] + 2^24*MB[j][3]) ) = 0
        CHIP SENDS (per lane j): AreBytes(MB[j][0], MB[j][1]), AreBytes(MB[j][2], MB[j][3])

        WHY THE SENDS ARE REQUIRED (the verified argument -- see the CORRECTION
        below before citing any older wording).

        The 16 lane `AreBytes` are `m[0..8]`'s ONLY range check. The message
        reaches the mixing core through `add3` alone and is never an XOR operand
        -- ✓ VERIFIED: `message_word_ref` appears in `blake3_socket.rs` solely as
        an add3 `m` operand, and `blake3_chip.rs`'s header had already recorded
        the same property of `m`. Every OTHER committed word in this design gets
        its bytes range-checked for free by a downstream `ByteAlu[XOR]`; the
        message has no such consumer, so if these sends go, nothing bounds it.

        What breaks without them: `m` becomes a free field element instead of a
        u32. Round 0's `add3` has CONSTANT `a` and `b` (BLOCK 3 -- the entire
        initial state is compile-time constant) and a byte-bounded `s`, so a
        prover solves `m = s + 2^32*(c1+c2) - a - b` for ANY chosen `s` -- put
        the whole value in `MB[0]` and zero the other three bytes -- and owns
        the compression from the first add onward. The chip then computes
        something that is not BLAKE3 of any 36-byte string, which is exactly the
        freedom a forged Merkle path needs.

        CORRECTION (D10). Earlier revisions of this docstring justified the sends
        with a `v` / `v + 2^32` collision -- "two lanes that hash alike". That
        attack is UNCONSTRUCTIBLE against this chip and the claim was wrong: the
        mixing core reads the SAME linear form the decomposition identity pins
        (`message_word_ref` is `Sum MB[j][k]*2^{8k}`), so `IN_lane` and the
        message word are one field element by construction and there is no
        reduction step for two felts to alias through. The identity is what makes
        them the same element; the sends are what make that element a u32. Both
        are still required -- for the reason above, not that one.

        In the BV domain a byte IS 8 bits, so the necessity of the sends is NOT
        visible here; it is proved in gate.py's FIELD width audit (WA2 is the
        executable form: without the sends the lane is not forced below 2^32).
        """
        for _j in range(8):
            word = self.c.fresh_word()
            self.c.are_bytes(*word)          # 2 sends per lane
            self.in_lane_bytes.append(word)
        self.census.add("lane_bytes(MB)", 32)

    # =====================================================================
    # BLOCK 2 -- message words. Only m[0..8] are columns; m[8..16] are constants.
    # =====================================================================
    def message_words(self) -> list:
        """m[a_slot..+4] = a, m[b_slot..+4] = b, m[tag_slot] = tag, rest = 0.

        REQUIREMENT: m[8..16] carry NO columns and NO range checks. They are
        compile-time constants, which is what makes the 4-byte domain tag free.
        """
        fr = self.fr
        m = [self.c.const_word(0) for _ in range(16)]
        a = self.in_lane_bytes[0:4]
        b = self.in_lane_bytes[4:8]
        if not fr.lane_le:
            # Control: a big-endian lane serialisation. BV-observable, because
            # the message WORD changes even though the columns do not.
            a = [list(reversed(w)) for w in a]
            b = [list(reversed(w)) for w in b]
        for i in range(4):
            m[fr.a_slot + i] = a[i]
            m[fr.b_slot + i] = b[i]
        # m[8] AS BUILT: NOT a constant -- a linear form over the two
        # PREPROCESSED mode columns, `MODE_C*TAG_LFMC + MODE_T*TAG_LFMT`
        # (`WordRef::ModeSelected`, evaluated `sum col*tag`). On a real row
        # exactly one selector is 1, so the value equals that row's tag; the
        # model therefore carries the SELECTED tag, and the mechanism that makes
        # the selection trustworthy -- preprocessed-ness plus the registrar's
        # one-hot check, NOT idx 4 -- is audited in the FIELD domain
        # (gate.audit_block0_tag_selection / M8). Modelling it as a bare
        # constant here would describe something the chip does not do and would
        # still report PASS: the fail-open this gate exists to prevent.
        m[fr.tag_slot] = self.c.const_word(fr.tag_word)
        return m

    # =====================================================================
    # BLOCK 3 -- initial state. ALL SIXTEEN WORDS ARE COMPILE-TIME CONSTANTS.
    # =====================================================================
    def init_state(self) -> list:
        """v[0..8] = h = IV[0..8];  v[8..12] = IV[0..4];  v[12] = t_lo = 0;
        v[13] = t_hi = 0;  v[14] = block_len = 36;  v[15] = flags = 0x0B.

        Note the consequence of h = IV: the ENTIRE initial state is constant, so
        the socket costs zero input-state columns (a syscall-shaped chip pays 112
        bytes here). It also means constant-folding round 0 is possible -- see
        ORACLE.md; it is permitted but must be re-gated, because a folded round 0
        no longer matches this model.
        """
        fr = self.fr
        cv = list(fr.cv)
        return [self.c.const_word(cv[i]) for i in range(8)] + \
               [self.c.const_word(ora.IV[i]) for i in range(4)] + \
               [self.c.const_word(fr.counter & 0xFFFFFFFF),
                self.c.const_word((fr.counter >> 32) & 0xFFFFFFFF),
                self.c.const_word(fr.block_len),
                self.c.const_word(fr.flags)]

    # =====================================================================
    # BLOCK 4 -- per-G SSA logic
    # =====================================================================
    def emit_xor(self, A: list, B: list) -> list:
        """CHIP SENDS: 4 x ByteAlu[XOR]. No eval constraint.
        The lookup pins the output AND byte-range-checks both operands -- which
        is why nearly every word in this design needs no explicit AreBytes."""
        out = [self.c.byte_xor(A[i], B[i]) for i in range(4)]
        self.census.add("xor_out", 4)
        return out

    @staticmethod
    def rotr16(A: list) -> list:
        """FREE byte relabel [b0,b1,b2,b3] -> [b2,b3,b0,b1]. No columns."""
        return [A[2], A[3], A[0], A[1]]

    @staticmethod
    def rotr8(A: list) -> list:
        """FREE byte relabel -> [b1,b2,b3,b0]. No columns."""
        return [A[1], A[2], A[3], A[0]]

    def emit_add2(self, A: list, B: list, drop_carry_bool: bool = False) -> list:
        """s = (A + B) mod 2^32, in the implementation's EXPRESSION-CARRY form.

        CHIP COLUMNS: s[0..4] bytes. **NO carry column.**
        CHIP CONSTRAINT (mu-gated), the only one — `blake3_socket.rs:826-834`:
            carry := (wval(A) + wval(B) - wval(s)) * 2^{-32}      (a linear form)
            MU * carry * (1 - carry) = 0                          (degree 3)

        The carry is *derived*, not witnessed, so the sum identity and the
        booleanity collapse into one constraint: `carry in {0,1}` is exactly
        `wval(A) + wval(B) - wval(s) in {0, 2^32}`.

        MODELLING NOTE, and it is the whole reason WA7 exists. `2^{-32}` is a
        FIELD inverse; there is no faithful BV counterpart, so the BV domain
        models the post-audit statement -- the difference lies in {0, 2^32} --
        and the side condition that those are the ONLY reachable roots (in
        particular that a negative difference cannot alias 2^32 mod p) is
        discharged in the field by WA7. Encoding the disjunction here without
        that audit would be assuming the very thing that makes the form sound.

        This deviates from the pre-Phase-2 model, which witnessed the carry as a
        column and constrained it twice. The two are equivalent -- the model's
        pair asserts `exists carry in {0,1}` where the implementation eliminates
        an existential whose witness is determined -- but the gate must certify
        the chip that EXISTS, not a stronger cousin, so the model follows the
        chip. Saves 1 column per add2: 2 per G, 96 (6r) / 112 (7r) overall."""
        from z3 import Or as _Or
        s = self.c.fresh_word()
        lhs = self.c.wval(A) + self.c.wval(B)
        rhs = self.c.wval(s)
        if drop_carry_bool:
            pass                      # control: the difference is unconstrained
        else:
            self.c.assertions.append(
                _Or(lhs == rhs, lhs == rhs + BitVecVal(1 << 32, WIDE)))
        self.census.add("add2", 4)
        return s

    def emit_add3(self, A: list, B: list, M: list,
                  drop_carry_bool: bool = False) -> list:
        """s = (A + B + M) mod 2^32, carry in {0,1,2} as TWO summed carry bits.
        CHIP COLUMNS: s[0..4] bytes + 2 carry columns.
        CHIP CONSTRAINTS (mu-gated):
            MU * ( wval(A)+wval(B)+wval(M) - wval(s) - 2^32*(c1+c2) ) = 0  (deg 2)
            MU * c1 * (1 - c1) = 0 ;  MU * c2 * (1 - c2) = 0               (deg 3)

        NOT a single ternary carry k(k-1)(k-2)=0: that body is degree 3 already
        and mu-gating pushes it to 4, over the hard budget. This coupling between
        mu-gating and the 3-operand add is the tightest in the design."""
        s = self.c.fresh_word()
        c1 = self.c.carry_bit(enforce=not drop_carry_bool)
        c2 = self.c.carry_bit(enforce=not drop_carry_bool)
        csum = self.c.wide(c1) + self.c.wide(c2)
        self.c.assertions.append(
            self.c.wval(A) + self.c.wval(B) + self.c.wval(M)
            == self.c.wval(s) + csum * BitVecVal(1 << 32, WIDE))
        self.census.add("add3", 6)
        return s

    def emit_rotr(self, A: list, n: int, wrong_amount: bool = False) -> list:
        """rotr12 / rotr7, inlined as the mu-gated linear shift identity.

        rotr12 = rotl20 = rotl16 . rotl4  (inner r = 4)
        rotr7  = rotl25 = rotl16 . rotl9  (inner r = 9)

        CHIP COLUMNS: SLL_lo(2B), SLLC_lo(2B), SLL_hi(2B), SLLC_hi(2B), Y[0..4](4B).
        CHIP CONSTRAINTS (mu-gated, all linear bodies):
            MU * ( xlo*2^r - SLLC_lo*2^16 - SLL_lo ) = 0
            MU * ( xhi*2^r - SLLC_hi*2^16 - SLL_hi ) = 0
            MU * ( Ylo - SLL_hi - SLLC_lo ) = 0
            MU * ( Yhi - SLL_lo - SLLC_hi ) = 0
        CHIP SENDS: AreBytes over the 8 bytes of SLL_lo/SLLC_lo/SLL_hi/SLLC_hi
                    = 4 sends. THE SLL BOUND IS TIGHT AND LOAD-BEARING.

        Y is range-checked free by the XOR that consumes it. Soundness needs 2^16
        invertible mod p -- a BV model cannot see that, so it is audited in the
        FIELD domain."""
        r = {12: 4, 7: 9}[n]
        if wrong_amount:
            r += 1
        xlo = self.c.hwval(A[0], A[1])
        xhi = self.c.hwval(A[2], A[3])
        sll_lo = [self.c.fresh_byte(), self.c.fresh_byte()]
        sllc_lo = [self.c.fresh_byte(), self.c.fresh_byte()]
        sll_hi = [self.c.fresh_byte(), self.c.fresh_byte()]
        sllc_hi = [self.c.fresh_byte(), self.c.fresh_byte()]
        self.c.are_bytes(*sll_lo, *sllc_lo, *sll_hi, *sllc_hi)   # 4 sends
        SLL_lo = self.c.hwval(*sll_lo)
        SLLC_lo = self.c.hwval(*sllc_lo)
        SLL_hi = self.c.hwval(*sll_hi)
        SLLC_hi = self.c.hwval(*sllc_hi)
        two_r = BitVecVal(1 << r, WIDE)
        two16 = BitVecVal(1 << 16, WIDE)
        self.c.assertions.append(xlo * two_r == SLLC_lo * two16 + SLL_lo)
        self.c.assertions.append(xhi * two_r == SLLC_hi * two16 + SLL_hi)
        Y = self.c.fresh_word()
        self.c.assertions.append(self.c.hwval(Y[0], Y[1]) == SLL_hi + SLLC_lo)
        self.c.assertions.append(self.c.hwval(Y[2], Y[3]) == SLL_lo + SLLC_hi)
        self.census.add("rotr_shift", 12)
        return Y

    def emit_g(self, v: list, a: int, b: int, c: int, d: int,
               mx: list, my: list, gflag: bool, skip_tail: bool = False):
        """One G quarter-round, in SSA. 56 byte cells + 6 carry cells.

        `skip_tail` omits the final XOR + rotr7, which produce v[b] only. That is
        legal in the LAST round for the four G-calls whose b-position is outside
        the truncation window -- and it carries an obligation, spelled out in
        ORACLE.md, about how the surviving consumer reads B1."""
        b_first = c if self._bug("swap_g_operand", gflag) else b
        v[a] = self.emit_add3(v[a], v[b_first], mx)
        v[d] = self.rotr16(self.emit_xor(v[d], v[a]))
        v[c] = self.emit_add2(v[c], v[d],
                              drop_carry_bool=self._bug("drop_add2_carry", gflag))
        v[b] = self.emit_rotr(self.emit_xor(v[b], v[c]), 12,
                              wrong_amount=self._bug("rot_wrong_amount", gflag))
        v[a] = self.emit_add3(v[a], v[b], my,
                              drop_carry_bool=self._bug("drop_carry_bool", gflag))
        v[d] = self.rotr8(self.emit_xor(v[d], v[a]))
        v[c] = self.emit_add2(v[c], v[d],
                              drop_carry_bool=self._bug("drop_add2_carry", gflag))
        if not skip_tail:
            v[b] = self.emit_rotr(self.emit_xor(v[b], v[c]), 7)

    def emit_rounds(self, v: list, m: list):
        """R rounds of 8 G-calls; the schedule is permuted between rounds by the
        compile-time MSG_PERMUTATION, so a round references the ORIGINAL message
        columns under permute^r with zero runtime handoff."""
        fr = self.fr
        window = set(range(fr.out_window, fr.out_window + 4))
        needed = window | {i + 8 for i in window}
        schedule = list(m)
        for r in range(fr.rounds):
            last = (r == fr.rounds - 1)
            for gi, (a, b, c, d, ix, iy) in enumerate(G_CALLS):
                gflag = (gi == 0 and r == 0)
                # The tail is droppable only when NOTHING later reads v[b]. In
                # the last round that means the DIAGONAL group only (gi >= 4):
                # a column G's v[b] is consumed by the diagonal group that
                # follows it in the same round, so dropping its tail is a bug,
                # not an optimisation.
                skip = (self.tail_truncate and last and gi >= 4
                        and b not in needed)
                self.emit_g(v, a, b, c, d, schedule[ix], schedule[iy],
                            gflag, skip_tail=skip)
            if not last:
                schedule = [schedule[fr.msg_permutation[i]] for i in range(16)]

    # =====================================================================
    # BLOCK 5 -- feed-forward, truncation window, output recomposition
    # =====================================================================
    def emit_feedforward(self, v: list, h: list):
        """CHIP CONSTRAINT: out[i] = v[i] XOR v[i+8], for i in the truncation
        window ONLY.

        The socket produces FOUR of the sixteen output words. out[i+8] =
        v[i+8] XOR h[i] is never computed: h is the constant IV and those words
        are not part of the digest. That is where most of the saving over a
        syscall-shaped BLAKE3 chip comes from -- 12 words x 4 bytes of columns
        and the same number of XOR sends, never built."""
        fr = self.fr
        for i in range(fr.out_window, fr.out_window + 4):
            w = self.emit_xor(v[i], v[i + 8])
            if self._bug("drop_ff_xor", i == fr.out_window):
                w = self.c.fresh_word()      # control: output left free
            self.digest_words.append(w)

    def digest_lane_values(self):
        """CHIP CONSTRAINT (per output lane i), mu-gated, degree 2:
            MU * ( OUT_C[i] - (OUTW[i][0] + 2^8*OUTW[i][1]
                               + 2^16*OUTW[i][2] + 2^24*OUTW[i][3]) ) = 0
        No range check needed: OUTW's bytes are ByteAlu[XOR] outputs, hence
        already bytes. The sum is < 2^32 << p, so OUT_C is forced to the honest
        u32 -- and therefore the socket's OUTPUT always satisfies O1, which is
        why only leaf digests and prover-hinted siblings need the input check."""
        return [self.c.wval(w) for w in self.digest_words]

    # =====================================================================
    def build(self) -> "SocketChip":
        if self._built:
            return self
        self.emit_lane_bytes()
        m = self.message_words()
        v = self.init_state()
        h = list(v[0:8])
        self.emit_rounds(v, m)
        self.emit_feedforward(v, h)
        # AS BUILT: the frozen shared prefix is 28 value columns (12 IN + 4 S +
        # 12 OUT). MU is preprocessed, so it is NOT a main column.
        # PREP_WIDTH 12 -> 13 (MODE_L) is PREPROCESSED and so does NOT enter the
        # main-column census; the frozen VALUE prefix is unchanged at 28.
        self.census.add("frozen_socket_prefix(IN/S/OUT)", 28)
        self.census.io_sends = 6      # the LfmMem tuples of the host socket
        self._built = True
        return self


# ---------------------------------------------------------------------------
# The reference, expressed over the SAME symbolic lane values, so the gate
# compares like with like.
# ---------------------------------------------------------------------------

def reference_digest_bv(chip: SocketChip, fr: sk.Framing):
    """Word-level BLAKE3 over 32-bit BVs -- structurally independent of the
    chip's byte-level XOR / halfword-shift wiring, exactly as the keccak gate
    keeps zref_round independent of the byte circuit."""
    from z3 import Concat, RotateRight

    def w32(word):
        return Concat(word[3], word[2], word[1], word[0])

    def ref_g(v, a, b, c, d, mx, my):
        v[a] = v[a] + v[b] + mx
        v[d] = RotateRight(v[d] ^ v[a], 16)
        v[c] = v[c] + v[d]
        v[b] = RotateRight(v[b] ^ v[c], 12)
        v[a] = v[a] + v[b] + my
        v[d] = RotateRight(v[d] ^ v[a], 8)
        v[c] = v[c] + v[d]
        v[b] = RotateRight(v[b] ^ v[c], 7)

    lanes = [w32(w) for w in chip.in_lane_bytes]
    if not fr.lane_le:
        lanes = [w32(list(reversed(w))) for w in chip.in_lane_bytes]
    m = [BitVecVal(0, 32) for _ in range(16)]
    for i in range(4):
        m[fr.a_slot + i] = lanes[i]
        m[fr.b_slot + i] = lanes[4 + i]
    m[fr.tag_slot] = BitVecVal(fr.tag_word, 32)

    v = [BitVecVal(x, 32) for x in fr.cv] + \
        [BitVecVal(ora.IV[i], 32) for i in range(4)] + \
        [BitVecVal(fr.counter & 0xFFFFFFFF, 32),
         BitVecVal((fr.counter >> 32) & 0xFFFFFFFF, 32),
         BitVecVal(fr.block_len, 32), BitVecVal(fr.flags, 32)]

    schedule = list(m)
    for r in range(fr.rounds):
        for (a, b, c, d, ix, iy) in G_CALLS:
            ref_g(v, a, b, c, d, schedule[ix], schedule[iy])
        if r < fr.rounds - 1:
            schedule = [schedule[fr.msg_permutation[i]] for i in range(16)]
    return [v[i] ^ v[i + 8] for i in range(fr.out_window, fr.out_window + 4)]
