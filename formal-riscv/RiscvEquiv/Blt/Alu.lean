import RiscvEquiv.Blt.Bridge
import RiscvEquiv.AluBus

/-!
# BLT, BGE, BLTU and BGEU: what the CPU learns from the ALU bus is Sail's condition

Same as `Slt/Alu.lean`, for the four `(signed, invert)` combinations of the LT clause of
`Alu.Guarantees`: each theorem instantiates `lt_alu_bit` (`AluBus.lean`) at the flags of its
branch.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `BLT` (`opsel = LT`, `signed = 1`, `invert = 0`) and
range-checked operands carries Sail's `BLT` condition. -/
theorem blt_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT + 32)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (bltPure a b).toNat := by
  rw [bltPure_toNat]
  exact lt_alu_bit m hg h1 h2 h3 h4 1 0 (by norm_num) (by norm_num) (by simpa using hflags)
    a b ha hb

/-- An ALU message with the flags of `BGE` (`opsel = LT`, `signed = 1`, `invert = 1`) and
range-checked operands carries Sail's `BGE` condition. -/
theorem bge_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT + 32 + 64)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (bgePure a b).toNat := by
  rw [bgePure_toNat]
  exact lt_alu_bit m hg h1 h2 h3 h4 1 1 (by norm_num) (by norm_num) (by simpa using hflags)
    a b ha hb

/-- An ALU message with the flags of `BLTU` (`opsel = LT`, `signed = 0`, `invert = 0`) and
range-checked operands carries Sail's `BLTU` condition. -/
theorem bltu_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (bltuPure a b).toNat := by
  rw [bltuPure_toNat]
  exact lt_alu_bit m hg h1 h2 h3 h4 0 0 (by norm_num) (by norm_num) (by simpa using hflags)
    a b ha hb

/-- An ALU message with the flags of `BGEU` (`opsel = LT`, `signed = 0`, `invert = 1`) and
range-checked operands carries Sail's `BGEU` condition. -/
theorem bgeu_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT + 64)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (bgeuPure a b).toNat := by
  rw [bgeuPure_toNat]
  exact lt_alu_bit m hg h1 h2 h3 h4 0 1 (by norm_num) (by norm_num) (by simpa using hflags)
    a b ha hb

end RiscvEquiv
