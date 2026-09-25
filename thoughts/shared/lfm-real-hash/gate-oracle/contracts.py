"""
LAYER 3: the CHIP-CONTRACT LIBRARY.

Assume-guarantee.  The gate proves the compression/framing layer *given* these
contracts; it does not re-prove the tables that supply them.  Those tables
(`prover/src/tables/bitwise.rs`) are existing, separately-audited chips, and
this is the same assumption the keccak gate makes.  What is NOT optional is
writing the contracts down: an unstated contract is how a fail-OPEN gate
happens -- the model quietly assumes a bound the chip never enforces, every
theorem comes back UNSAT, and the gate certifies nothing.

Each contract below records, in one place:
  * the GUARANTEE the gate is allowed to assume,
  * the OBLIGATION the chip must discharge to earn it (a real bus send),
  * the WIDTH it licenses -- which is the entry the width audit cites.

Two modelling domains, because they see different bugs:

  BV  (QF_BV, bytes as 8-bit bitvectors) -- sees logic/wiring bugs.  It CANNOT
      see bound-necessity bugs, because in a bounded BV model the bound is
      baked into the variable's width: dropping a range check is unrepresentable.
  FIELD (Int mod p, Goldilocks) -- sees exactly those.  A committed column with
      no range check is a full field element, and 2^16 / 2^32 are invertible
      mod p while being zero divisors mod 2^n.  Every field-lifted width in the
      design must be justified HERE, not in BV.

Getting that split wrong is the classic fail-open: model a dropped range check
in BV, observe UNSAT, and conclude the range check is unnecessary.
"""

from __future__ import annotations

from dataclasses import dataclass

from z3 import BitVec, BitVecVal, Int, Or, ZeroExt

# Goldilocks.
P = 2**64 - 2**32 + 1

# Wide BV width used for the add/shift identities.  Honest field expressions in
# this design stay < 2^35; 48 bits is comfortably above that and below 64, so a
# BV overflow inside the model would be a modelling bug, not a masked forgery.
WIDE = 48


@dataclass(frozen=True)
class Contract:
    name: str
    guarantee: str
    obligation: str
    width: str


CONTRACTS: dict[str, Contract] = {
    "AreBytes": Contract(
        name="AreBytes[x, y]",
        guarantee="x, y are integers in [0, 256).",
        obligation="one send to the precomputed BITWISE AreBytes receiver "
                   "(`bitwise.rs`, AreBytes), multiplicity Column(MU), per PAIR "
                   "of bytes.",
        width="licenses treating a committed column as an 8-bit value in a "
              "field-lifted linear form.",
    ),
    "ByteAlu_XOR": Contract(
        name="ByteAlu[XOR](x, y) -> z",
        guarantee="x, y, z in [0, 256) AND z = x XOR y, exactly.",
        obligation="one send to the precomputed BITWISE ByteAlu receiver with "
                   "op = XOR, multiplicity Column(MU), per output BYTE.",
        width="range-checks BOTH operands and the output for free -- this is why "
              "most words in the design need no explicit AreBytes: they are "
              "consumed by a later XOR. Operands may be linear combinations "
              "provided each stays <= 255 (which is what makes a free byte "
              "relabel legal in place).",
    ),
    "LaneDecomposition": Contract(
        name="lane = b0 + 2^8*b1 + 2^16*b2 + 2^24*b3, with AreBytes on b0..b4",
        guarantee="the felt `lane` is in [0, 2^32) and b0..b4 are its unique "
                  "little-endian byte decomposition.",
        obligation="ONE mu-gated eval constraint (the linear identity) AND TWO "
                   "AreBytes sends. NEITHER ALONE SUFFICES -- see the width audit: "
                   "without AreBytes the bytes are free field elements and the "
                   "identity is satisfiable for arbitrary byte strings; without "
                   "the identity the bytes are unrelated to the lane.",
        width="THE load-bearing width of this design. Sum of four bytes weighted "
              "by 2^{8k} is < 2^32 << p, so the identity cannot wrap, so `lane` "
              "is forced < 2^32. This is what makes felt -> u32 injective and is "
              "the whole content of obligation O1.",
    ),
    "CarryBit": Contract(
        name="mu * c * (1 - c) = 0",
        guarantee="c in {0, 1} as a field element.",
        obligation="one mu-gated degree-3 eval constraint per carry column.",
        width="licenses treating a carry column as a bit in the add identities. "
              "Dropping it is a FIELD-level forgery invisible to BV.",
    ),
    "ShiftRemainderBound": Contract(
        name="AreBytes on the two bytes of SLL",
        guarantee="SLL in [0, 2^16).",
        obligation="AreBytes sends on SLL's byte pair (per halfword, per rotation).",
        width="the TIGHT remainder bound. With 2^16 invertible mod p it pins "
              "SLL = (x * 2^r) mod 2^16 uniquely. The quotient SLLC needs only a "
              "loose 16-bit bound. Dropping the SLL bound makes the rotation "
              "forgeable -- demonstrable ONLY in the field model.",
    ),
    "NoWrapSideCondition": Contract(
        name="every field-lifted expression < 2^35 << p",
        guarantee="`expr == 0 mod p` implies `expr == 0` over the integers, so "
                  "the BV model's arithmetic is faithful to the field's.",
        obligation="a static bound argument on each identity, discharged by the "
                   "width audit table in ORACLE.md -- NOT by any solver run.",
        width="the bridge between the BV model and the field. If any identity "
              "could reach p, the BV theorems say nothing about the real chip.",
    ),
}


# ---------------------------------------------------------------------------
# BV domain
# ---------------------------------------------------------------------------

class BvContracts:
    """Contracts as BV constructions.  A byte IS an 8-bit BitVec: that is the
    AreBytes guarantee, structurally enforced and therefore un-droppable here.
    That structural enforcement is exactly why bound-necessity must be argued in
    the FIELD domain instead."""

    def __init__(self, tag: str):
        self.tag = tag
        self.assertions: list = []
        self._n = 0
        self.sends = 0            # bus-send accounting, for the cost model

    def fresh(self, width: int = 8):
        v = BitVec(f"{self.tag}_{self._n}", width)
        self._n += 1
        return v

    def fresh_byte(self):
        return self.fresh(8)

    def fresh_word(self) -> list:
        """A 32-bit word as 4 little-endian byte columns."""
        return [self.fresh_byte() for _ in range(4)]

    @staticmethod
    def const_word(val: int) -> list:
        return [BitVecVal((val >> (8 * i)) & 0xFF, 8) for i in range(4)]

    # -- value lifts ------------------------------------------------------
    @staticmethod
    def wide(x):
        return ZeroExt(WIDE - x.size(), x)

    def wval(self, word: list):
        """The field-lifted word value: sum of bytes * 2^{8k}.  < 2^32."""
        acc = BitVecVal(0, WIDE)
        for i in range(4):
            acc = acc + self.wide(word[i]) * BitVecVal(1 << (8 * i), WIDE)
        return acc

    def hwval(self, blo, bhi):
        """Field-lifted halfword value.  < 2^16."""
        return self.wide(blo) + self.wide(bhi) * BitVecVal(256, WIDE)

    # -- contracts --------------------------------------------------------
    def are_bytes(self, *bytes_):
        """AreBytes. Structural in BV (8-bit width). Counted for the cost model:
        one send per PAIR."""
        self.sends += (len(bytes_) + 1) // 2

    def byte_xor(self, x, y):
        """ByteAlu[XOR]: fresh output byte, pinned to x ^ y."""
        z = self.fresh_byte()
        self.assertions.append(z == (x ^ y))
        self.sends += 1
        return z

    def carry_bit(self, enforce: bool = True):
        """A carry column with (or, for a control, without) its booleanity."""
        c = self.fresh(8)
        if enforce:
            self.assertions.append(Or(c == 0, c == 1))
        return c


# ---------------------------------------------------------------------------
# FIELD domain
# ---------------------------------------------------------------------------

class FieldContracts:
    """Contracts as mod-p Int constraints.  Here a column is a FULL field element
    unless a contract bounds it, so dropping a contract is expressible -- which
    is the entire point of having this second domain."""

    def __init__(self, solver):
        self.s = solver
        self._n = 0

    def fresh_felt(self, name: str | None = None):
        v = Int(name or f"felt_{self._n}")
        self._n += 1
        self.s.add(v >= 0, v < P)          # a committed column: any field element
        return v

    def are_bytes(self, *vals):
        for v in vals:
            self.s.add(v >= 0, v < 256)

    def bounded(self, v, bound: int):
        self.s.add(v >= 0, v < bound)

    def carry_bit(self, v):
        self.s.add(Or(v == 0, v == 1))

    @staticmethod
    def lane_from_bytes(b: list):
        return b[0] + 256 * b[1] + 65536 * b[2] + 16777216 * b[3]


def contract_table_md() -> str:
    lines = ["| contract | guarantee | obligation on the chip | width it licenses |",
             "|---|---|---|---|"]
    for c in CONTRACTS.values():
        g = c.guarantee.replace("\n", " ")
        o = c.obligation.replace("\n", " ")
        w = c.width.replace("\n", " ")
        lines.append(f"| `{c.name}` | {g} | {o} | {w} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print(contract_table_md())
