module

public import LambdaVM.Field
public import LambdaVM.AluOp

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

/-- `SHIFT` result (spec/src/shift.toml): the shift amount is reduced mod 32 for word instructions
and mod 64 otherwise; `direction` picks left (0) or right (1); `signed` picks the arithmetic
right shift. Word results are not sign-extended here (the CPU does that). For `word = 1` the
shift still acts on all 64 bits of `x`, so for a right shift the caller must pass `x` with its
high word already zero-extended (logical) or sign-extended (arithmetic); otherwise bits of the
high word shift into the low 32 bits of the result. -/
def shiftResult (word signed direction x s : ℕ) : ℕ :=
  let k := s % (if word = 1 then 32 else 64)
  if direction = 0 then (x <<< k) % 2^64
  else if signed = 0 then x >>> k
  else sra64 x k

/-- `opsel` of MIN: a toy operation for the counted-pulls PoC, not an RV64IM op. It has the shape
of DIVREM: its chip pulls LT on this bus and pushes its own result on the same bus.

The value is deliberately outside the prover's `alu_op` enum (`prover/src/tables/types.rs`, which
ends at `MUL = 7`, `DIVREM = 8`), so the toy cannot collide with a real opsel once MUL and DIVREM
are modelled on this bus. -/
def opselMIN : ℕ := 9

/-- What a puller learns. One clause per `opsel`, selected by the flag byte read as a natural.
Each clause takes as hypotheses exactly the `[[assumptions]]` of its chip's spec: the words the
caller must have range-checked. The words a chip range-checks itself need no hypothesis. -/
def Guarantees (m : Msg (F p)) : Prop :=
  -- LT (spec/src/lt.toml): flags = opsel(LT) + 32·signed + 64·invert, out = [res, 0];
  -- assumes `IS_WORD[lhs[0]]`, `IS_WORD[rhs[0]]` (LT range-checks the high halves)
  (∀ s i : ℕ, s < 2 → i < 2 → m.flags.val = opselLT + 32 * s + 64 * i →
    m.in1Lo.val < 2^32 → m.in2Lo.val < 2^32 →
    m.outHi = 0 ∧
    m.outLo = ((ltBit s i (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi) : ℕ) : F p)) ∧
  -- EQ (spec/src/eq.toml): flags = opsel(EQ) + 64·invert, out = [res, 0];
  -- assumes `IS_WORD[a[i]]`, `IS_WORD[b[i]]`
  (∀ i : ℕ, i < 2 → m.flags.val = opselEQ + 64 * i →
    m.in1Lo.val < 2^32 → m.in1Hi.val < 2^32 → m.in2Lo.val < 2^32 → m.in2Hi.val < 2^32 →
    m.outHi = 0 ∧
    m.outLo = ((eqBit i (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi) : ℕ) : F p)) ∧
  -- SHIFT (spec/src/shift.toml): flags = opsel(SHIFT) + word_instr + 32·signed + 64·direction;
  -- no assumptions (SHIFT range-checks `in` and the low bytes of the shift amount itself),
  -- only the low byte of the shift amount matters
  (∀ w s d : ℕ, w < 2 → s < 2 → d < 2 →
    m.flags.val = opselSHIFT + w + 32 * s + 64 * d →
    m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
    dword m.outLo m.outHi = shiftResult w s d (dword m.in1Lo m.in1Hi) (m.in2Lo.val % 256)) ∧
  -- MIN (toy, see `opselMIN`): flags = opsel(MIN), out = the unsigned minimum;
  -- assumes `IS_WORD` on both words of both operands, as the global antecedent did before the
  -- per-opsel refactor
  (m.flags.val = opselMIN →
    m.in1Lo.val < 2^32 → m.in1Hi.val < 2^32 → m.in2Lo.val < 2^32 → m.in2Hi.val < 2^32 →
    m.outLo.val < 2^32 ∧ m.outHi.val < 2^32 ∧
    dword m.outLo m.outHi = min (dword m.in1Lo m.in1Hi) (dword m.in2Lo m.in2Hi))

def channel : Channel (F p) Msg where
  name := "ALU"
  Guarantees m _ := Guarantees m

end LambdaVM.Alu
