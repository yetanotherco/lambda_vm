import LeanRV64D.InstsEnd

/-!
# BEQ and BNE in the Sail RISC-V model

Sail's `execute_BTYPE` (`LeanRV64D/InstsEnd.lean`) runs a branch: it reads `rs1` and `rs2`,
decides whether the branch is taken, and if so jumps to `PC + imm` (the 13-bit offset
sign-extended to 64 bits). Branches write no register.

`beqPure` and `bnePure` are the branch conditions as pure functions on values.
`execute_BEQ` and `execute_BNE` prove that Sail's branch is "jump if the condition holds".
The jump target is not connected to the circuit here: that needs the PC, i.e. the CPU in Clean.

`#print axioms` lists `sys_enable_experimental_extensions` for these theorems. It is not used by
the proofs: it is a platform setting that the Sail model declares as an `axiom`
(`LeanRV64D/RiscvExtras.lean`), and `jump_to`, which the statements mention, depends on it.
`execute_BTYPE` itself already does.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `BEQ` condition: `rs1 = rs2`. -/
def beqPure (rs1 rs2 : BitVec 64) : Bool := rs1 == rs2

/-- `BNE` condition: `rs1 ≠ rs2`. -/
def bnePure (rs1 rs2 : BitVec 64) : Bool := rs1 != rs2

/-- Sail's `BEQ` reads `rs1` and `rs2` and jumps to `PC + imm` iff `beqPure x[rs1] x[rs2]`. -/
theorem execute_BEQ (imm : BitVec 13) (rs2 rs1 : regidx) :
    execute_BTYPE imm rs2 rs1 bop.BEQ = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      if beqPure a b then jump_to ((← readReg Register.PC) + (sign_extend (m := 64) imm))
      else pure RETIRE_SUCCESS) := by
  simp only [execute_BTYPE, beqPure, bind_assoc, pure_bind]
  rfl

/-- Sail's `BNE` reads `rs1` and `rs2` and jumps to `PC + imm` iff `bnePure x[rs1] x[rs2]`. -/
theorem execute_BNE (imm : BitVec 13) (rs2 rs1 : regidx) :
    execute_BTYPE imm rs2 rs1 bop.BNE = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      if bnePure a b then jump_to ((← readReg Register.PC) + (sign_extend (m := 64) imm))
      else pure RETIRE_SUCCESS) := by
  simp only [execute_BTYPE, bnePure, bind_assoc, pure_bind]
  rfl

end RiscvEquiv
