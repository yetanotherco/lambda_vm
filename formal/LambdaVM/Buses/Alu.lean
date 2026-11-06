module

public import LambdaVM.Field
public import LambdaVM.AluOp
public import LambdaVM.Mu
public import Clean.Air.CountedChannel

@[expose] public section

/-!
# ALU bus (`BusId::Alu`)

`ALU[out; in1, in2, flags]`: operands and result travel as `(lo32, hi32)` pairs, `flags`
carries the `alu_op` descriptor plus per-chip flag bits. Every ALU chip pushes to this
channel; the CPU pulls. Each chip adds the clause of `Guarantees` for its own `opsel`.
-/

namespace LambdaVM.Alu

structure Msg (F : Type) where
  in1Lo : F
  in1Hi : F
  in2Lo : F
  in2Hi : F
  flags : F
  outLo : F
  outHi : F
deriving ProvableStruct

/-- `lhs < rhs` on 64-bit values, signed (two's complement) or unsigned, as a bit. -/
def ltRaw (signed L R : ℕ) : ℕ :=
  if signed = 1 then (if (BitVec.ofNat 64 L).slt (BitVec.ofNat 64 R) then 1 else 0)
  else (if (BitVec.ofNat 64 L).ult (BitVec.ofNat 64 R) then 1 else 0)

/-- The two comparisons on values below `2⁶⁴`, read on naturals. -/
lemma ult_ofNat_iff {L R : ℕ} (hL : L < 2^64) (hR : R < 2^64) :
    (BitVec.ofNat 64 L).ult (BitVec.ofNat 64 R) = true ↔ L < R := by
  simp only [BitVec.ult_eq_decide, decide_eq_true_eq, BitVec.toNat_ofNat, Nat.mod_eq_of_lt hL,
    Nat.mod_eq_of_lt hR]

lemma slt_ofNat_iff {L R : ℕ} (hL : L < 2^64) (hR : R < 2^64) :
    (BitVec.ofNat 64 L).slt (BitVec.ofNat 64 R) = true ↔
      (if L < 2^63 then (L : ℤ) else (L : ℤ) - 2^64) <
        (if R < 2^63 then (R : ℤ) else (R : ℤ) - 2^64) := by
  simp only [BitVec.slt_eq_decide, decide_eq_true_eq, BitVec.toInt_eq_toNat_cond, BitVec.toNat_ofNat,
    Nat.mod_eq_of_lt hL, Nat.mod_eq_of_lt hR]
  split_ifs <;> omega

/-- LT result: `(lhs < rhs) xor invert`. -/
def ltBit (signed invert L R : ℕ) : ℕ :=
  if ltRaw signed L R = invert then 0 else 1

/-- EQ result: `(a = b) xor invert`. -/
def eqBit (invert A B : ℕ) : ℕ :=
  if (if A = B then 1 else 0) = invert then 0 else 1

/-- Arithmetic right shift of a 64-bit value: sign-extend to 128 bits, shift, truncate. -/
def sra64 (x k : ℕ) : ℕ :=
  if x < 2^63 then x >>> k else ((x + (2^128 - 2^64)) >>> k) % 2^64

/-- `SHIFT` result (spec/shift.typ): the shift amount is reduced mod 32 for word instructions
and mod 64 otherwise; `direction` picks left (0) or right (1); `signed` picks the arithmetic
right shift. Word results are not sign-extended here (the CPU does that). -/
def shiftResult (word signed direction x s : ℕ) : ℕ :=
  let k := s % (if word = 1 then 32 else 64)
  if direction = 0 then (x <<< k) % 2^64
  else if signed = 0 then x >>> k
  else sra64 x k

/-- A 64-bit value sign-extended to 128 bits: when the operand is read as signed and its top bit
is set, two's complement fills the upper half with ones. -/
def ext128 (signed x : ℕ) : ℕ := if signed = 1 ∧ 2^63 ≤ x then x + (2^128 - 2^64) else x

/-- `MUL` (`spec/mul.typ`): the low (`high = 0`) or high (`high = 1`) 64 bits of the 128-bit
product of the operands, each sign-extended according to its own flag. The four RV64IM forms
`MUL`, `MULH`, `MULHU` and `MULHSU` are the four combinations of the two flags. -/
def mulResult (lhsSigned rhsSigned high x y : ℕ) : ℕ :=
  if high = 1 then (ext128 lhsSigned x * ext128 rhsSigned y) % 2^128 / 2^64
  else (ext128 lhsSigned x * ext128 rhsSigned y) % 2^128 % 2^64

/-- Two's-complement reading of a 64-bit value. -/
def toSigned64 (x : ℕ) : ℤ := if x < 2^63 then x else (x : ℤ) - 2^64

/-- A 64-bit value read as signed (`signed = 1`) or unsigned. -/
def sgn64 (signed x : ℕ) : ℤ := if signed = 1 then toSigned64 x else x

/-- `DIVREM` (`spec/dvrm.typ`): the quotient (`rem = 0`) or the remainder (`rem = 1`) of `a` by
`b`, rounding towards zero, which is `DIV`, `DIVU`, `REM` and `REMU`. Division by zero gives the
quotient `2^64 - 1` and the remainder `a`. The signed overflow `-2^63 / -1` needs no case of its
own: `Int.tdiv` gives `2^63`, which is `-2^63` modulo `2^64`, with remainder `0`, as RISC-V
requires. -/
def divremResult (signed rem a b : ℕ) : ℕ :=
  if b = 0 then (if rem = 1 then a else 2^64 - 1)
  else ((if rem = 1 then (sgn64 signed a).tmod (sgn64 signed b)
    else (sgn64 signed a).tdiv (sgn64 signed b)) % 2^64).toNat

/-- What a puller learns. The caller must have range-checked both words of both operands
(the chips' `IS_WORD` assumptions). One clause per `opsel`, selected by the flag byte read
as a natural. -/
def Guarantees (m : Msg (F p)) : Prop :=
  m.in1Lo.val < 2^32 → m.in1Hi.val < 2^32 → m.in2Lo.val < 2^32 → m.in2Hi.val < 2^32 →
    -- LT (spec/lt.typ): flags = opsel(LT) + 32·signed + 64·invert, out = [res, 0]
    (∀ s i : ℕ, s < 2 → i < 2 → m.flags.val = opselLT + 32 * s + 64 * i →
      m.outHi = 0 ∧
      m.outLo = ((ltBit s i (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi) : ℕ) : F p)) ∧
    -- EQ (spec/eq.typ): flags = opsel(EQ) + 64·invert, out = [res, 0]
    (∀ i : ℕ, i < 2 → m.flags.val = opselEQ + 64 * i →
      m.outHi = 0 ∧
      m.outLo = ((eqBit i (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi) : ℕ) : F p)) ∧
    -- SHIFT (spec/shift.typ): flags = opsel(SHIFT) + word_instr + 32·signed + 64·direction,
    -- only the low byte of the shift amount matters
    (∀ w s d : ℕ, w < 2 → s < 2 → d < 2 →
      m.flags.val = opselSHIFT + w + 32 * s + 64 * d →
      m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
      dword m.outLo m.outHi = shiftResult w s d (dword m.in1Lo m.in1Hi) (m.in2Lo.val % 256)) ∧
    -- BYTEWISE (spec/bytewise.typ): flags = opsel(AND | OR | XOR), with no flag bits above it;
    -- out is the operation applied to the whole 64-bit values
    (m.flags.val = opselAND →
      m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
      dword m.outLo m.outHi = dword m.in1Lo m.in1Hi &&& dword m.in2Lo m.in2Hi) ∧
    (m.flags.val = opselOR →
      m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
      dword m.outLo m.outHi = dword m.in1Lo m.in1Hi ||| dword m.in2Lo m.in2Hi) ∧
    (m.flags.val = opselXOR →
      m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
      dword m.outLo m.outHi = dword m.in1Lo m.in1Hi ^^^ dword m.in2Lo m.in2Hi) ∧
    -- MUL (spec/mul.typ): flags = opsel(MUL) + 32·lhs_signed + 64·rhs_signed + 128·high
    (∀ ls rs h : ℕ, ls < 2 → rs < 2 → h < 2 →
      m.flags.val = opselMUL + 32 * ls + 64 * rs + 128 * h →
      m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
      dword m.outLo m.outHi = mulResult ls rs h (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi)) ∧
    -- DIVREM (spec/dvrm.typ): flags = opsel(DIVREM) + 32·signed + 128·rem
    (∀ s rm : ℕ, s < 2 → rm < 2 →
      m.flags.val = opselDIVREM + 32 * s + 128 * rm →
      m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
      dword m.outLo m.outHi
        = divremResult s rm (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi))

def channel : Channel (F p) Msg where
  name := "ALU"
  Guarantees m _ := Guarantees m

/-- The ALU bus as the one `RawChannel` every chip here uses, pushers and pullers alike.

`Channel.counted N` is a different `RawChannel` for each `N`, and `Channel.toRaw` is a
different one again, all three carrying the name `"ALU"`. A chip that pushes could use
`toRaw`, since a push is unrestricted there, but DVRM pulls this bus with a multiplicity
that reaches `muMax`, and a pull on `toRaw` has to carry `0` or `1`. Two `RawChannel`s
under one name is not a soundness hole, because `unique_channel_names` rejects it, but it
means no ensemble can hold DVRM together with another ALU chip. So the whole model agrees
on this one, the same way it agrees on one `muMax`. The AIR does not see the difference:
`emitUpTo` and `pushIf` build the same interaction but for the channel they name, so
`export/` is unchanged by the choice. -/
abbrev channelC : RawChannel (F p) := channel.counted muMax

end LambdaVM.Alu
