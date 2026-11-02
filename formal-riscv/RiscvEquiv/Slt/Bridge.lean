import RiscvEquiv.Slt.Sail
import LambdaVM.Buses.Alu

/-!
# SLT and SLTU: the Clean model's comparison is Sail's

With `invert = 0`, `Alu.ltBit` is `Alu.ltRaw`, which the Clean model already defines with
Lean's `BitVec.slt` (`signed = 1`) and `BitVec.ult` (`signed = 0`) on the operands read back
as 64-bit vectors. So the bridge only has to undo `BitVec.ofNat 64 a.toNat = a`.
-/

namespace RiscvEquiv

open LambdaVM

/-- `sltPure` as a natural is `ltBit` with the flags of `SLT` (`signed = 1`, `invert = 0`). -/
theorem sltPure_toNat (a b : BitVec 64) :
    (sltPure a b).toNat = Alu.ltBit 1 0 a.toNat b.toNat := by
  by_cases h : a.slt b <;> simp [sltPure, Alu.ltBit, Alu.ltRaw, h]

/-- `sltuPure` as a natural is `ltBit` with the flags of `SLTU` (`signed = 0`, `invert = 0`). -/
theorem sltuPure_toNat (a b : BitVec 64) :
    (sltuPure a b).toNat = Alu.ltBit 0 0 a.toNat b.toNat := by
  by_cases h : a.ult b <;> simp [sltuPure, Alu.ltBit, Alu.ltRaw, h]

end RiscvEquiv
