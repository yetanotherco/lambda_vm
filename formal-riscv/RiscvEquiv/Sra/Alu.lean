import RiscvEquiv.Sra.Bridge

/-!
# SRA: what the CPU learns from the ALU bus is Sail's result

Same shape as `Sll/Alu.lean`, at the flags of `SRA`: `signed = 1` and `direction = 1`, so the
flag byte is `opsel(SHIFT) + 32 + 64`.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `SRA` (`opsel = SHIFT`, `word = 0`, `signed = 1`,
`direction = 1`) and range-checked operands carries Sail's `SRA` result. -/
theorem sra_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT + 32 + 64)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (sraPure a b).toNat := by
  obtain ⟨-, -, hshift⟩ := hg h1 h2 h3 h4
  obtain ⟨-, -, hout⟩ := hshift 0 1 1 (by norm_num) (by norm_num) (by norm_num)
    (by simpa using hflags)
  -- SHIFT reads only the low byte of the shift amount; mod 64 it is the low bits of `b`.
  have hamt : m.in2Lo.val % 256 % 64 = b.toNat % 64 := by
    rw [← hb, dword]
    omega
  rw [hout, sraPure_toNat, ha, shiftResult_sra, shiftResult_sra, hamt]

end RiscvEquiv
