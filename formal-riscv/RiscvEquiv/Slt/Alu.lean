import RiscvEquiv.Slt.Bridge
import RiscvEquiv.AluBus

/-!
# SLT and SLTU: what the CPU learns from the ALU bus is Sail's result

Same shape as `Sll/Alu.lean`, with the LT clause of `Alu.Guarantees` (`lt_alu_bit`, in
`AluBus.lean`): the flag byte is `opsel(LT) + 32·signed + 64·invert`, and the result travels as
`(bit, 0)`.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `SLT` (`opsel = LT`, `signed = 1`, `invert = 0`) and
range-checked operands carries Sail's `SLT` result. -/
theorem slt_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT + 32)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (sltPure a b).toNat := by
  rw [sltPure_toNat]
  exact lt_alu_bit m hg h1 h2 h3 h4 1 0 (by norm_num) (by norm_num) (by simpa using hflags)
    a b ha hb

/-- An ALU message with the flags of `SLTU` (`opsel = LT`, `signed = 0`, `invert = 0`) and
range-checked operands carries Sail's `SLTU` result. -/
theorem sltu_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (sltuPure a b).toNat := by
  rw [sltuPure_toNat]
  exact lt_alu_bit m hg h1 h2 h3 h4 0 0 (by norm_num) (by norm_num) (by simpa using hflags)
    a b ha hb

end RiscvEquiv
