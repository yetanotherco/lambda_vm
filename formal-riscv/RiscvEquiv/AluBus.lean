import LambdaVM.Buses.Alu

/-!
# Reading ALU bus messages as naturals

The CPU does not read the tables' rows: it pulls a message from the ALU bus and learns
`Alu.Guarantees` (`formal/LambdaVM/Buses/Alu.lean`), which each table's soundness establishes
for every message it pushes. The guarantees are stated on field elements and for generic flags.
This file has one lemma per clause (SHIFT, SHIFT with `word = 1`, LT, EQ) that reads the result
of a message as a natural in terms of its 64-bit operands `a` and `b`. Each instruction's
`Alu.lean` instantiates one of them at its flags and rewrites with its bridge.
-/

namespace RiscvEquiv

open LambdaVM

/-- `ltBit` is a bit. -/
lemma ltBit_lt_two (s i L R : ℕ) : Alu.ltBit s i L R < 2 := by
  unfold Alu.ltBit
  split <;> omega

/-- `eqBit` is a bit. -/
lemma eqBit_lt_two (i A B : ℕ) : Alu.eqBit i A B < 2 := by
  unfold Alu.eqBit
  split_ifs <;> omega

/-- With `word = 0` only the shift amount mod 64 matters. -/
lemma shiftResult_amt (s d x s₁ s₂ : ℕ) (h : s₁ % 64 = s₂ % 64) :
    Alu.shiftResult 0 s d x s₁ = Alu.shiftResult 0 s d x s₂ := by
  simp [Alu.shiftResult, h]

/-- With `word = 1` only the shift amount mod 32 matters. -/
lemma shiftResult_word_amt (s d x s₁ s₂ : ℕ) (h : s₁ % 32 = s₂ % 32) :
    Alu.shiftResult 1 s d x s₁ = Alu.shiftResult 1 s d x s₂ := by
  simp [Alu.shiftResult, h]

/-- A SHIFT message with `word = 0` and flags `signed = s`, `direction = d` carries
`shiftResult` of its operands. SHIFT reads only the low byte of the shift amount; mod 64 it is
the low bits of `b`. -/
lemma shift_alu (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (s d : ℕ) (hs : s < 2) (hd : d < 2) (hflags : m.flags.val = opselSHIFT + 32 * s + 64 * d)
    (x : ℕ) (b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = x)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = Alu.shiftResult 0 s d x b.toNat := by
  obtain ⟨-, -, hshift⟩ := hg h1 h2 h3 h4
  obtain ⟨-, -, hout⟩ := hshift 0 s d (by norm_num) hs hd (by simpa using hflags)
  have hamt : m.in2Lo.val % 256 % 64 = b.toNat % 64 := by
    rw [← hb, dword]
    omega
  rw [hout, ha, shiftResult_amt _ _ _ _ _ hamt]

/-- A SHIFT message with `word = 1` and flags `signed = s`, `direction = d` carries, in `outLo`,
the low word of `shiftResult` of its operands. Mod 32 the low byte of the shift amount is the
low bits of `b`. -/
lemma shift_word_alu_low (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (s d : ℕ) (hs : s < 2) (hd : d < 2) (hflags : m.flags.val = opselSHIFT + 1 + 32 * s + 64 * d)
    (x : ℕ) (b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = x)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    m.outLo.val = Alu.shiftResult 1 s d x b.toNat % 2^32 := by
  obtain ⟨-, -, hshift⟩ := hg h1 h2 h3 h4
  obtain ⟨hlo, -, hout⟩ := hshift 1 s d (by norm_num) hs hd hflags
  have hamt : m.in2Lo.val % 256 % 32 = b.toNat % 32 := by
    rw [← hb, dword]
    omega
  have hlow : m.outLo.val = dword m.outLo m.outHi % 2^32 := by
    rw [dword]
    omega
  rw [hlow, hout, ha, shiftResult_word_amt _ _ _ _ _ hamt]

/-- An LT message with flags `signed = s`, `invert = i` carries `ltBit s i` of its operands.
The result travels as `(bit, 0)`. -/
lemma lt_alu_bit (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (s i : ℕ) (hs : s < 2) (hi : i < 2) (hflags : m.flags.val = opselLT + 32 * s + 64 * i)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = Alu.ltBit s i a.toNat b.toNat := by
  obtain ⟨hlt, -, -⟩ := hg h1 h2 h3 h4
  obtain ⟨hhi, hlo⟩ := hlt s i hs hi hflags
  have hbit := ltBit_lt_two s i a.toNat b.toNat
  rw [dword, hhi, hlo, ha, hb, ZMod.val_zero, val_natCast_of_lt_two_pow_40 (by omega),
    mul_zero, add_zero]

/-- An EQ message with flag `invert = i` carries `eqBit i` of its operands. The result travels
as `(bit, 0)`. -/
lemma eq_alu_bit (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (i : ℕ) (hi : i < 2) (hflags : m.flags.val = opselEQ + 64 * i)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = Alu.eqBit i a.toNat b.toNat := by
  obtain ⟨-, heq, -⟩ := hg h1 h2 h3 h4
  obtain ⟨hhi, hlo⟩ := heq i hi hflags
  have hbit := eqBit_lt_two i a.toNat b.toNat
  rw [dword, hhi, hlo, ha, hb, ZMod.val_zero, val_natCast_of_lt_two_pow_40 (by omega),
    mul_zero, add_zero]

end RiscvEquiv
