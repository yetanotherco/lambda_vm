import LeanRV64D.InstsEnd

/-!
# SRA in the Sail RISC-V model

Same shape as `Sll/Sail.lean`. For `SRA`, Sail's `execute_RTYPE` computes
`shift_bits_right_arith x[rs1] (extractLsb x[rs2] (log2_xlen - 1) 0)`, an arithmetic right
shift: the vacated high bits are copies of the sign bit.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `SRA` on values: `rs1` shifted right by the low 6 bits of `rs2`, filling with its sign bit. -/
def sraPure (rs1 rs2 : BitVec 64) : BitVec 64 := rs1.sshiftRight (rs2.toNat % 64)

/-- Sail's `SRA` reads `rs1` and `rs2` and writes `sraPure x[rs1] x[rs2]` to `rd`. -/
theorem execute_SRA (rs2 rs1 rd : regidx) :
    execute_RTYPE rs2 rs1 rd rop.SRA = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd (sraPure a b)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPE, shift_bits_right_arith, Sail.BitVec.extractLsb, sraPure,
    Sail.BitVec.toNatInt, BitVec.extractLsb_toNat]
  rfl

end RiscvEquiv
