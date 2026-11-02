import RiscvEquiv.Beq.Bridge

/-!
# BEQ and BNE: what the CPU learns from the ALU bus is Sail's condition

Same shape as `Slt/Alu.lean`, with the EQ clause of `Alu.Guarantees`: the flag byte is
`opsel(EQ) + 64·invert`, and the result travels as `(bit, 0)`. The CPU takes the branch when
the bit is `1`.
-/

namespace RiscvEquiv

open LambdaVM

/-- `eqBit` is a bit. -/
lemma eqBit_lt_two (i A B : ℕ) : Alu.eqBit i A B < 2 := by
  unfold Alu.eqBit
  split_ifs <;> omega

/-- An ALU message with the flags of `BEQ` (`opsel = EQ`, `invert = 0`) and range-checked
operands carries Sail's `BEQ` condition. -/
theorem beq_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselEQ)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (beqPure a b).toNat := by
  obtain ⟨-, heq, -⟩ := hg h1 h2 h3 h4
  obtain ⟨hhi, hlo⟩ := heq 0 (by norm_num) (by simpa using hflags)
  have hbit := eqBit_lt_two 0 a.toNat b.toNat
  rw [dword, hhi, hlo, ha, hb, ZMod.val_zero, val_natCast_of_lt_two_pow_40 (by omega),
    beqPure_toNat, mul_zero, add_zero]

/-- An ALU message with the flags of `BNE` (`opsel = EQ`, `invert = 1`) and range-checked
operands carries Sail's `BNE` condition. -/
theorem bne_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselEQ + 64)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (bnePure a b).toNat := by
  obtain ⟨-, heq, -⟩ := hg h1 h2 h3 h4
  obtain ⟨hhi, hlo⟩ := heq 1 (by norm_num) (by simpa using hflags)
  have hbit := eqBit_lt_two 1 a.toNat b.toNat
  rw [dword, hhi, hlo, ha, hb, ZMod.val_zero, val_natCast_of_lt_two_pow_40 (by omega),
    bnePure_toNat, mul_zero, add_zero]

end RiscvEquiv
