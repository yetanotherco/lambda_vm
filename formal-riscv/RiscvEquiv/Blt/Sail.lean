import RiscvEquiv.Beq.Sail

/-!
# BLT, BGE, BLTU and BGEU in the Sail RISC-V model

Same shape as `Beq/Sail.lean`. Sail's `execute_BTYPE` compares `x[rs1]` and `x[rs2]` as signed
(`BLT`, `BGE`) or unsigned (`BLTU`, `BGEU`) integers and jumps to `PC + imm` if the comparison
holds. Sail writes the comparisons on integers (`toInt`, `toNatInt`); the `*Pure` conditions use
Lean's `BitVec.slt` and `BitVec.ult`, the ones the Clean model uses, and `*_cond` proves they
agree.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `BLT` condition: `rs1 < rs2` as signed integers. -/
def bltPure (rs1 rs2 : BitVec 64) : Bool := rs1.slt rs2

/-- `BGE` condition: `rs1 ≥ rs2` as signed integers. -/
def bgePure (rs1 rs2 : BitVec 64) : Bool := !rs1.slt rs2

/-- `BLTU` condition: `rs1 < rs2` as unsigned integers. -/
def bltuPure (rs1 rs2 : BitVec 64) : Bool := rs1.ult rs2

/-- `BGEU` condition: `rs1 ≥ rs2` as unsigned integers. -/
def bgeuPure (rs1 rs2 : BitVec 64) : Bool := !rs1.ult rs2

/-- Sail's signed `<` is `bltPure`.
`zopz0zI_s` is Sail's `operator <_s`, name-mangled by the Lean backend:
`zop` is "operator", `z0` is the space, `zI` is `<`, and `_s` stays as is (it means "signed"). -/
theorem blt_cond (a b : BitVec 64) : zopz0zI_s a b = bltPure a b := by
  simp [zopz0zI_s, bltPure, BitVec.slt_eq_decide]

/-- Sail's signed `≥` is `bgePure`. -/
theorem bge_cond (a b : BitVec 64) : zopz0zKzJ_s a b = bgePure a b := by
  simp only [zopz0zKzJ_s, bgePure, BitVec.slt_eq_decide]
  by_cases h : a.toInt < b.toInt <;> simp [h] <;> omega

/-- Sail's unsigned `<` is `bltuPure`. -/
theorem bltu_cond (a b : BitVec 64) : zopz0zI_u a b = bltuPure a b := by
  simp [zopz0zI_u, bltuPure, BitVec.ult_eq_decide, Sail.BitVec.toNatInt]

/-- Sail's unsigned `≥` is `bgeuPure`. -/
theorem bgeu_cond (a b : BitVec 64) : zopz0zKzJ_u a b = bgeuPure a b := by
  simp only [zopz0zKzJ_u, bgeuPure, BitVec.ult_eq_decide, Sail.BitVec.toNatInt]
  by_cases h : a.toNat < b.toNat <;> simp [h] <;> omega

/-- Sail's `BLT` reads `rs1` and `rs2` and jumps to `PC + imm` iff `bltPure x[rs1] x[rs2]`. -/
theorem execute_BLT (imm : BitVec 13) (rs2 rs1 : regidx) :
    execute_BTYPE imm rs2 rs1 bop.BLT = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      if bltPure a b then jump_to ((← readReg Register.PC) + (sign_extend (m := 64) imm))
      else pure RETIRE_SUCCESS) := by
  simp only [execute_BTYPE, blt_cond, bind_assoc, pure_bind]

/-- Sail's `BGE` reads `rs1` and `rs2` and jumps to `PC + imm` iff `bgePure x[rs1] x[rs2]`. -/
theorem execute_BGE (imm : BitVec 13) (rs2 rs1 : regidx) :
    execute_BTYPE imm rs2 rs1 bop.BGE = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      if bgePure a b then jump_to ((← readReg Register.PC) + (sign_extend (m := 64) imm))
      else pure RETIRE_SUCCESS) := by
  simp only [execute_BTYPE, bge_cond, bind_assoc, pure_bind]

/-- Sail's `BLTU` reads `rs1` and `rs2` and jumps to `PC + imm` iff `bltuPure x[rs1] x[rs2]`. -/
theorem execute_BLTU (imm : BitVec 13) (rs2 rs1 : regidx) :
    execute_BTYPE imm rs2 rs1 bop.BLTU = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      if bltuPure a b then jump_to ((← readReg Register.PC) + (sign_extend (m := 64) imm))
      else pure RETIRE_SUCCESS) := by
  simp only [execute_BTYPE, bltu_cond, bind_assoc, pure_bind]

/-- Sail's `BGEU` reads `rs1` and `rs2` and jumps to `PC + imm` iff `bgeuPure x[rs1] x[rs2]`. -/
theorem execute_BGEU (imm : BitVec 13) (rs2 rs1 : regidx) :
    execute_BTYPE imm rs2 rs1 bop.BGEU = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      if bgeuPure a b then jump_to ((← readReg Register.PC) + (sign_extend (m := 64) imm))
      else pure RETIRE_SUCCESS) := by
  simp only [execute_BTYPE, bgeu_cond, bind_assoc, pure_bind]

end RiscvEquiv
