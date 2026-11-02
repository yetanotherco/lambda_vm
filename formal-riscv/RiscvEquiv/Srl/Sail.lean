import LeanRV64D.InstsEnd

/-!
# SRL in the Sail RISC-V model

Same shape as `Sll/Sail.lean`. For `SRL`, Sail's `execute_RTYPE` computes
`shift_bits_right x[rs1] (extractLsb x[rs2] (log2_xlen - 1) 0)`, a logical right shift.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `SRL` on values: `rs1` shifted right by the low 6 bits of `rs2`, filling with zeros. -/
def srlPure (rs1 rs2 : BitVec 64) : BitVec 64 := rs1 >>> (rs2.toNat % 64)

/-- Sail's `SRL` reads `rs1` and `rs2` and writes `srlPure x[rs1] x[rs2]` to `rd`. -/
theorem execute_SRL (rs2 rs1 rd : regidx) :
    execute_RTYPE rs2 rs1 rd rop.SRL = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd (srlPure a b)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPE, shift_bits_right, Sail.BitVec.extractLsb, srlPure,
    BitVec.ushiftRight_eq', BitVec.extractLsb_toNat]
  rfl

end RiscvEquiv
