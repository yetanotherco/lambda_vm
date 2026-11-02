import RiscvEquiv.Slt.Sail

/-!
# SLTI and SLTIU in the Sail RISC-V model

The immediate comparisons run in Sail's `execute_ITYPE` (`LeanRV64D/InstsEnd.lean`), not in
`execute_RTYPE`: the second operand is the 12-bit immediate `imm` sign-extended to 64 bits,
instead of `x[rs2]`. The comparison is the same, so each one is the register version with
`imm.signExtend 64` in place of `x[rs2]`. `SLTIU` also sign-extends the immediate and then
compares unsigned, as the RISC-V manual specifies.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- Sail's `SLTI` reads `rs1` and writes `sltPure x[rs1] (signExtend imm)` to `rd`. -/
theorem execute_SLTI (imm : BitVec 12) (rs1 rd : regidx) :
    execute_ITYPE imm rs1 rd iop.SLTI = (do
      let a ← rX_bits rs1
      wX_bits rd (sltPure a (imm.signExtend 64))
      pure RETIRE_SUCCESS) := by
  simp only [execute_ITYPE, slt_bit]
  rfl

/-- Sail's `SLTIU` reads `rs1` and writes `sltuPure x[rs1] (signExtend imm)` to `rd`. -/
theorem execute_SLTIU (imm : BitVec 12) (rs1 rd : regidx) :
    execute_ITYPE imm rs1 rd iop.SLTIU = (do
      let a ← rX_bits rs1
      wX_bits rd (sltuPure a (imm.signExtend 64))
      pure RETIRE_SUCCESS) := by
  simp only [execute_ITYPE, sltu_bit]
  rfl

end RiscvEquiv
