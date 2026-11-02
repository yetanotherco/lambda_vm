import RiscvEquiv.SltImm.Sail
import RiscvEquiv.Slt.Alu

/-!
# SLTI and SLTIU: what the CPU learns from the ALU bus is Sail's result

The immediate comparisons use the same ALU flags as `SLT` and `SLTU`; only the second operand
changes, from `x[rs2]` to the immediate sign-extended to 64 bits. So each theorem is the
register one with `b = imm.signExtend 64`, and no new bridge is needed.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `SLT` whose second operand is `imm` sign-extended carries
Sail's `SLTI` result. -/
theorem slti_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT + 32)
    (a : BitVec 64) (imm : BitVec 12) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = (imm.signExtend 64).toNat) :
    dword m.outLo m.outHi = (sltPure a (imm.signExtend 64)).toNat :=
  slt_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha hb

/-- An ALU message with the flags of `SLTU` whose second operand is `imm` sign-extended carries
Sail's `SLTIU` result. -/
theorem sltiu_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselLT)
    (a : BitVec 64) (imm : BitVec 12) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = (imm.signExtend 64).toNat) :
    dword m.outLo m.outHi = (sltuPure a (imm.signExtend 64)).toNat :=
  sltu_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha hb

end RiscvEquiv
