import RiscvEquiv.Slt.Bridge

/-!
# SLT and SLTU: what the CPU learns from the ALU bus is Sail's result

Same shape as `Sll/Alu.lean`, with the LT clause of `Alu.Guarantees`: the flag byte is
`opsel(LT) + 32·signed + 64·invert`, and the result travels as `(bit, 0)`.
-/

namespace RiscvEquiv

open LambdaVM

/-- `ltBit` is a bit. -/
lemma ltBit_lt_two (s i L R : ℕ) : Alu.ltBit s i L R < 2 := by
  unfold Alu.ltBit
  split <;> omega

/-- An ALU message with the flags of `SLT` (`opsel = LT`, `signed = 1`, `invert = 0`) and
range-checked operands carries Sail's `SLT` result. -/
theorem slt_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT + 32)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (sltPure a b).toNat := by
  obtain ⟨hlt, -, -⟩ := hg h1 h2 h3 h4
  obtain ⟨hhi, hlo⟩ := hlt 1 0 (by norm_num) (by norm_num) (by simpa using hflags)
  have hbit := ltBit_lt_two 1 0 a.toNat b.toNat
  rw [dword, hhi, hlo, ha, hb, ZMod.val_zero, val_natCast_of_lt_two_pow_40 (by omega),
    sltPure_toNat, mul_zero, add_zero]

/-- An ALU message with the flags of `SLTU` (`opsel = LT`, `signed = 0`, `invert = 0`) and
range-checked operands carries Sail's `SLTU` result. -/
theorem sltu_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (sltuPure a b).toNat := by
  obtain ⟨hlt, -, -⟩ := hg h1 h2 h3 h4
  obtain ⟨hhi, hlo⟩ := hlt 0 0 (by norm_num) (by norm_num) (by simpa using hflags)
  have hbit := ltBit_lt_two 0 0 a.toNat b.toNat
  rw [dword, hhi, hlo, ha, hb, ZMod.val_zero, val_natCast_of_lt_two_pow_40 (by omega),
    sltuPure_toNat, mul_zero, add_zero]

end RiscvEquiv
