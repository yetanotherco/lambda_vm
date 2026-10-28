module

public import LambdaVM.Field

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

/-- `alu_op` descriptors (`prover/src/tables/types.rs`). -/
def opselEQ : ℕ := 3
def opselLT : ℕ := 4
def opselSHIFT : ℕ := 5

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
      dword m.outLo m.outHi = shiftResult w s d (dword m.in1Lo m.in1Hi) (m.in2Lo.val % 256))

def channel : Channel (F p) Msg where
  name := "ALU"
  Guarantees m _ := Guarantees m

end LambdaVM.Alu
