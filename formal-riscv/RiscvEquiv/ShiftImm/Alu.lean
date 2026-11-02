import RiscvEquiv.ShiftImm.Sail
import RiscvEquiv.Sll.Alu
import RiscvEquiv.Srl.Alu
import RiscvEquiv.Sra.Alu

/-!
# SLLI, SRLI and SRAI: what the CPU learns from the ALU bus is Sail's result

The immediate shifts use the same ALU flags as their register versions; only the second
operand changes, from `x[rs2]` to the immediate `shamt`. So each theorem is the register one
with `b = shamt` zero-extended to 64 bits, and no new bridge is needed.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `SLL` whose second operand is `shamt` carries Sail's
`SLLI` result. -/
theorem slli_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT)
    (a : BitVec 64) (shamt : BitVec 6) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = shamt.toNat) :
    dword m.outLo m.outHi = (sllPure a (shamt.setWidth 64)).toNat :=
  sll_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha (by simpa using hb)

/-- An ALU message with the flags of `SRL` whose second operand is `shamt` carries Sail's
`SRLI` result. -/
theorem srli_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 64)
    (a : BitVec 64) (shamt : BitVec 6) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = shamt.toNat) :
    dword m.outLo m.outHi = (srlPure a (shamt.setWidth 64)).toNat :=
  srl_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha (by simpa using hb)

/-- An ALU message with the flags of `SRA` whose second operand is `shamt` carries Sail's
`SRAI` result. -/
theorem srai_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 32 + 64)
    (a : BitVec 64) (shamt : BitVec 6) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = shamt.toNat) :
    dword m.outLo m.outHi = (sraPure a (shamt.setWidth 64)).toNat :=
  sra_alu_matches_sail m hg h1 h2 h3 h4 hflags a _ ha (by simpa using hb)

end RiscvEquiv
