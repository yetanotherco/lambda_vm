import RiscvEquiv.ShiftW.Sail
import RiscvEquiv.Sra.Bridge

/-!
# Word shifts: the Clean model's shift function is Sail's, on the low word

With `word = 1`, `Alu.shiftResult` shifts the whole 64-bit operand by the amount mod 32 and
does not sign-extend (the CPU32 chip does). So the theorems here are about the low 32 bits of
the result, and about the operand CPU32 sends (`spec/src/cpu32.toml`): the low word of `rs1`
zero-extended for `SLLW`/`SRLW` (`signed = 0`), sign-extended for `SRAW` (`signed = 1`).
-/

namespace RiscvEquiv

open LambdaVM

/-- `sllwPure` as a natural is the low word of `shiftResult` with the flags of `SLLW`
(`word = 1`, `signed = 0`, `direction = 0`) on the zero-extended low word of `a`. -/
theorem sllwPure_toNat (a b : BitVec 64) :
    (sllwPure a b).toNat = Alu.shiftResult 1 0 0 (a.toNat % 2^32) b.toNat % 2^32 := by
  simp only [sllwPure, Alu.shiftResult, BitVec.toNat_shiftLeft, BitVec.toNat_setWidth]
  simp [Nat.mod_mod_of_dvd]

/-- `srlwPure` as a natural is the low word of `shiftResult` with the flags of `SRLW`
(`word = 1`, `signed = 0`, `direction = 1`) on the zero-extended low word of `a`. -/
theorem srlwPure_toNat (a b : BitVec 64) :
    (srlwPure a b).toNat = Alu.shiftResult 1 0 1 (a.toNat % 2^32) b.toNat % 2^32 := by
  have hx : (a.toNat % 2^32) >>> (b.toNat % 32) < 2^32 := by
    rw [Nat.shiftRight_eq_div_pow]
    exact lt_of_le_of_lt (Nat.div_le_self _ _) (Nat.mod_lt _ (by norm_num))
  simp only [srlwPure, Alu.shiftResult, BitVec.toNat_ushiftRight, BitVec.toNat_setWidth]
  simp only [if_pos, one_ne_zero, ite_false]
  exact (Nat.mod_eq_of_lt hx).symm

/-- Shifting a sign-extended word arithmetically and keeping the low word is shifting the
word itself. -/
theorem sshiftRight_signExtend_low (v : BitVec 32) (k : ℕ) (hk : k < 32) :
    ((v.signExtend 64).sshiftRight k).setWidth 32 = v.sshiftRight k := by
  ext i hi
  simp only [BitVec.getElem_setWidth, BitVec.getElem_sshiftRight, BitVec.getLsbD_sshiftRight,
    BitVec.getLsbD_signExtend, BitVec.msb_signExtend]
  have h64 : k + i < 64 := by omega
  by_cases h : k + i < 32 <;> simp [h, h64, show i < 64 by omega, BitVec.getLsbD_eq_getElem]

/-- `srawPure` as a natural is the low word of `shiftResult` with the flags of `SRAW`
(`word = 1`, `signed = 1`, `direction = 1`) on the sign-extended low word of `a`. -/
theorem srawPure_toNat (a b : BitVec 64) :
    (srawPure a b).toNat
      = Alu.shiftResult 1 1 1 ((a.setWidth 32).signExtend 64).toNat b.toNat % 2^32 := by
  have hk : b.toNat % 32 < 32 := Nat.mod_lt _ (by norm_num)
  simp only [srawPure, Alu.shiftResult, ↓reduceIte, one_ne_zero]
  rw [← sshiftRight_toNat _ _ (by omega), ← BitVec.toNat_setWidth,
    sshiftRight_signExtend_low _ _ hk]

end RiscvEquiv
