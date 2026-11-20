import RiscvEquiv.ShiftImm.Sail

/-!
# SLLW, SRLW, SRAW, SLLIW, SRLIW and SRAIW in the Sail RISC-V model

The word shifts run in Sail's `execute_RTYPEW` (register shift amount) and `execute_SHIFTIWOP`
(5-bit immediate `shamt`). Both take the low 32 bits of `x[rs1]`, shift them by the amount
mod 32, and write the 32-bit result sign-extended to 64 bits to `rd`.

`*wPure` is the 32-bit result, before the sign extension: in the circuit the SHIFT table
computes it and the CPU32 chip sign-extends it, so the two halves are proved separately.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `SLLW` on values: the low word of `rs1` shifted left by the low 5 bits of `rs2`. -/
def sllwPure (rs1 rs2 : BitVec 64) : BitVec 32 := rs1.setWidth 32 <<< (rs2.toNat % 32)

/-- `SRLW` on values: the low word of `rs1` shifted right by the low 5 bits of `rs2`,
filling with zeros. -/
def srlwPure (rs1 rs2 : BitVec 64) : BitVec 32 := rs1.setWidth 32 >>> (rs2.toNat % 32)

/-- `SRAW` on values: the low word of `rs1` shifted right by the low 5 bits of `rs2`,
filling with its sign bit (bit 31). -/
def srawPure (rs1 rs2 : BitVec 64) : BitVec 32 := (rs1.setWidth 32).sshiftRight (rs2.toNat % 32)

/-- Sail's `extractLsb x 31 0` is the low word of `x`. -/
theorem extract_low32 (x : BitVec 64) : Sail.BitVec.extractLsb x 31 0 = x.setWidth 32 := by
  apply BitVec.eq_of_toNat_eq
  simp [Sail.BitVec.extractLsb, BitVec.extractLsb_toNat]

/-- The word shift amount: the low 5 bits of the low word of `x[rs2]`. -/
theorem extract_shamt32 (x : BitVec 64) :
    (Sail.BitVec.extractLsb (x.setWidth 32) 4 0).toNat = x.toNat % 32 := by
  simp [Sail.BitVec.extractLsb, BitVec.extractLsb_toNat]

/-- A 5-bit `shamt` zero-extended to 64 bits, read mod 32, is `shamt` itself. -/
theorem shamt5_toNat (shamt : BitVec 5) : (shamt.setWidth 64).toNat % 32 = shamt.toNat := by
  have := shamt.isLt
  simp only [BitVec.toNat_setWidth]
  omega

/-- Sail's `SLLW` writes `sllwPure x[rs1] x[rs2]`, sign-extended, to `rd`. -/
theorem execute_SLLW (rs2 rs1 rd : regidx) :
    execute_RTYPEW rs2 rs1 rd ropw.SLLW = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd ((sllwPure a b).signExtend 64)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPEW, extract_low32, shift_bits_left, BitVec.shiftLeft_eq',
    extract_shamt32, sllwPure, pure_bind]
  rfl

/-- Sail's `SRLW` writes `srlwPure x[rs1] x[rs2]`, sign-extended, to `rd`. -/
theorem execute_SRLW (rs2 rs1 rd : regidx) :
    execute_RTYPEW rs2 rs1 rd ropw.SRLW = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd ((srlwPure a b).signExtend 64)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPEW, extract_low32, shift_bits_right, BitVec.ushiftRight_eq',
    extract_shamt32, srlwPure, pure_bind]
  rfl

/-- Sail's `SRAW` writes `srawPure x[rs1] x[rs2]`, sign-extended, to `rd`. -/
theorem execute_SRAW (rs2 rs1 rd : regidx) :
    execute_RTYPEW rs2 rs1 rd ropw.SRAW = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd ((srawPure a b).signExtend 64)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPEW, extract_low32, shift_bits_right_arith, Sail.BitVec.toNatInt,
    extract_shamt32, srawPure, pure_bind]
  rfl

/-- Sail's `SLLIW` writes `sllwPure x[rs1] shamt`, sign-extended, to `rd`. -/
theorem execute_SLLIW (shamt : BitVec 5) (rs1 rd : regidx) :
    execute_SHIFTIWOP shamt rs1 rd sopw.SLLIW = (do
      let a ← rX_bits rs1
      wX_bits rd ((sllwPure a (shamt.setWidth 64)).signExtend 64)
      pure RETIRE_SUCCESS) := by
  simp only [execute_SHIFTIWOP, extract_low32, shift_bits_left, BitVec.shiftLeft_eq',
    shamt5_toNat, sllwPure, pure_bind]
  rfl

/-- Sail's `SRLIW` writes `srlwPure x[rs1] shamt`, sign-extended, to `rd`. -/
theorem execute_SRLIW (shamt : BitVec 5) (rs1 rd : regidx) :
    execute_SHIFTIWOP shamt rs1 rd sopw.SRLIW = (do
      let a ← rX_bits rs1
      wX_bits rd ((srlwPure a (shamt.setWidth 64)).signExtend 64)
      pure RETIRE_SUCCESS) := by
  simp only [execute_SHIFTIWOP, extract_low32, shift_bits_right, BitVec.ushiftRight_eq',
    shamt5_toNat, srlwPure, pure_bind]
  rfl

/-- Sail's `SRAIW` writes `srawPure x[rs1] shamt`, sign-extended, to `rd`. -/
theorem execute_SRAIW (shamt : BitVec 5) (rs1 rd : regidx) :
    execute_SHIFTIWOP shamt rs1 rd sopw.SRAIW = (do
      let a ← rX_bits rs1
      wX_bits rd ((srawPure a (shamt.setWidth 64)).signExtend 64)
      pure RETIRE_SUCCESS) := by
  simp only [execute_SHIFTIWOP, extract_low32, shift_bits_right_arith, Sail.BitVec.toNatInt,
    shamt5_toNat, srawPure, pure_bind]
  rfl

end RiscvEquiv
