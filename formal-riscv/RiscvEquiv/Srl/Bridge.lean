import RiscvEquiv.Srl.Sail
import LambdaVM.Buses.Alu

/-!
# SRL: the Clean model's shift function is Sail's

With `word = 0`, `signed = 0` and `direction = 1`, `Alu.shiftResult` is a 64-bit logical right
shift. This file proves it is `srlPure` read as a natural.
-/

namespace RiscvEquiv

open LambdaVM

/-- `shiftResult` with the flags of `SRL`: `x` shifted right by `s mod 64`. -/
lemma shiftResult_srl (x s : ℕ) : Alu.shiftResult 0 0 1 x s = x >>> (s % 64) := by
  simp [Alu.shiftResult]

/-- `srlPure` as a natural is `shiftResult` with the flags of `SRL`. -/
theorem srlPure_toNat (a b : BitVec 64) :
    (srlPure a b).toNat = Alu.shiftResult 0 0 1 a.toNat b.toNat := by
  rw [shiftResult_srl, srlPure, BitVec.toNat_ushiftRight]

end RiscvEquiv
