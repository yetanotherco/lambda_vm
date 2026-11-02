import LeanRV64D.InstsEnd

/-!
# SLT and SLTU in the Sail RISC-V model

Same shape as `Sll/Sail.lean`. For `SLT`, Sail's `execute_RTYPE` computes
`zero_extend (bool_to_bit (x[rs1] <_s x[rs2]))`: compare the two registers as signed
integers and write `1` if the first is smaller, `0` otherwise. `SLTU` compares them as
unsigned.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `SLT` on values: `1` if `rs1 < rs2` as signed integers, `0` otherwise. -/
def sltPure (rs1 rs2 : BitVec 64) : BitVec 64 := if rs1.slt rs2 then 1 else 0

/-- `SLTU` on values: `1` if `rs1 < rs2` as unsigned integers, `0` otherwise. -/
def sltuPure (rs1 rs2 : BitVec 64) : BitVec 64 := if rs1.ult rs2 then 1 else 0

/-- Sail's signed comparison, turned into a 64-bit `0`/`1`, is `sltPure`. -/
theorem slt_bit (a b : BitVec 64) :
    zero_extend (m := 64) (bool_to_bit (zopz0zI_s a b)) = sltPure a b := by
  unfold zopz0zI_s bool_to_bit zero_extend sltPure
  rw [BitVec.slt_eq_decide]
  by_cases h : a.toInt < b.toInt <;> simp [h, bool_bit_forwards] <;> rfl

/-- Sail's unsigned comparison, turned into a 64-bit `0`/`1`, is `sltuPure`. -/
theorem sltu_bit (a b : BitVec 64) :
    zero_extend (m := 64) (bool_to_bit (zopz0zI_u a b)) = sltuPure a b := by
  unfold zopz0zI_u bool_to_bit zero_extend sltuPure
  rw [BitVec.ult_eq_decide]
  by_cases h : a.toNat < b.toNat <;> simp [h, bool_bit_forwards, Sail.BitVec.toNatInt] <;> rfl

/-- Sail's `SLT` reads `rs1` and `rs2` and writes `sltPure x[rs1] x[rs2]` to `rd`. -/
theorem execute_SLT (rs2 rs1 rd : regidx) :
    execute_RTYPE rs2 rs1 rd rop.SLT = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd (sltPure a b)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPE, slt_bit]
  rfl

/-- Sail's `SLTU` reads `rs1` and `rs2` and writes `sltuPure x[rs1] x[rs2]` to `rd`. -/
theorem execute_SLTU (rs2 rs1 rd : regidx) :
    execute_RTYPE rs2 rs1 rd rop.SLTU = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd (sltuPure a b)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPE, sltu_bit]
  rfl

end RiscvEquiv
