import RiscvEquiv.Sll.Sail
import RiscvEquiv.Srl.Sail
import RiscvEquiv.Sra.Sail

/-!
# SLLI, SRLI and SRAI in the Sail RISC-V model

The immediate shifts run in Sail's `execute_SHIFTIOP` (`LeanRV64D/InstsEnd.lean`), not in
`execute_RTYPE`: the shift amount is the 6-bit immediate `shamt` instead of `x[rs2]`. The
computation is the same, so each one is the register shift with `shamt` (zero-extended to
64 bits) in place of `x[rs2]`.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `shamt` zero-extended to 64 bits, read mod 64, is `shamt` itself. -/
theorem shamt_toNat (shamt : BitVec 6) : (shamt.setWidth 64).toNat % 64 = shamt.toNat := by
  have := shamt.isLt
  simp only [BitVec.toNat_setWidth]
  omega

/-- Sail cuts `shamt` to its low `log2_xlen = 6` bits, which on RV64 is all of it. -/
theorem extract_shamt (shamt : BitVec 6) :
    (Sail.BitVec.extractLsb shamt ((Functions.log2_xlen : Int) - 1).toNat 0).toNat
      = shamt.toNat := by
  have := shamt.isLt
  simp only [Sail.BitVec.extractLsb, BitVec.extractLsb_toNat, Nat.shiftRight_zero]
  exact Nat.mod_eq_of_lt (by simpa [Functions.log2_xlen] using this)

/-- Sail's `SLLI` reads `rs1` and writes `sllPure x[rs1] shamt` to `rd`. -/
theorem execute_SLLI (shamt : BitVec 6) (rs1 rd : regidx) :
    execute_SHIFTIOP shamt rs1 rd sop.SLLI = (do
      let a ← rX_bits rs1
      wX_bits rd (sllPure a (shamt.setWidth 64))
      pure RETIRE_SUCCESS) := by
  simp only [execute_SHIFTIOP, shift_bits_left, BitVec.shiftLeft_eq', extract_shamt, sllPure,
    shamt_toNat]
  rfl

/-- Sail's `SRLI` reads `rs1` and writes `srlPure x[rs1] shamt` to `rd`. -/
theorem execute_SRLI (shamt : BitVec 6) (rs1 rd : regidx) :
    execute_SHIFTIOP shamt rs1 rd sop.SRLI = (do
      let a ← rX_bits rs1
      wX_bits rd (srlPure a (shamt.setWidth 64))
      pure RETIRE_SUCCESS) := by
  simp only [execute_SHIFTIOP, shift_bits_right, BitVec.ushiftRight_eq', extract_shamt, srlPure,
    shamt_toNat]
  rfl

/-- Sail's `SRAI` reads `rs1` and writes `sraPure x[rs1] shamt` to `rd`. -/
theorem execute_SRAI (shamt : BitVec 6) (rs1 rd : regidx) :
    execute_SHIFTIOP shamt rs1 rd sop.SRAI = (do
      let a ← rX_bits rs1
      wX_bits rd (sraPure a (shamt.setWidth 64))
      pure RETIRE_SUCCESS) := by
  simp only [execute_SHIFTIOP, shift_bits_right_arith, Sail.BitVec.toNatInt, extract_shamt,
    sraPure, shamt_toNat]
  rfl

end RiscvEquiv
