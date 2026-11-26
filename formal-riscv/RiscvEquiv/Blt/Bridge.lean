import RiscvEquiv.Blt.Sail
import LambdaVM.Buses.Alu

/-!
# BLT, BGE, BLTU and BGEU: the Clean model's comparison bit is Sail's condition

Same as `Slt/Bridge.lean`: `Alu.ltBit signed invert` is defined with `BitVec.slt`
(`signed = 1`) or `BitVec.ult` (`signed = 0`), xor `invert`. `BLT`/`BLTU` are `invert = 0`
(the `SLT`/`SLTU` comparison), `BGE`/`BGEU` are `invert = 1` (its negation).
-/

namespace RiscvEquiv

open LambdaVM

/-- `bltPure` as a bit is `ltBit` with the flags of `BLT` (`signed = 1`, `invert = 0`). -/
theorem bltPure_toNat (a b : BitVec 64) :
    (bltPure a b).toNat = Alu.ltBit 1 0 a.toNat b.toNat := by
  by_cases h : a.slt b <;> simp [bltPure, Alu.ltBit, Alu.ltRaw, h]

/-- `bgePure` as a bit is `ltBit` with the flags of `BGE` (`signed = 1`, `invert = 1`). -/
theorem bgePure_toNat (a b : BitVec 64) :
    (bgePure a b).toNat = Alu.ltBit 1 1 a.toNat b.toNat := by
  by_cases h : a.slt b <;> simp [bgePure, Alu.ltBit, Alu.ltRaw, h]

/-- `bltuPure` as a bit is `ltBit` with the flags of `BLTU` (`signed = 0`, `invert = 0`). -/
theorem bltuPure_toNat (a b : BitVec 64) :
    (bltuPure a b).toNat = Alu.ltBit 0 0 a.toNat b.toNat := by
  by_cases h : a.ult b <;> simp [bltuPure, Alu.ltBit, Alu.ltRaw, h]

/-- `bgeuPure` as a bit is `ltBit` with the flags of `BGEU` (`signed = 0`, `invert = 1`). -/
theorem bgeuPure_toNat (a b : BitVec 64) :
    (bgeuPure a b).toNat = Alu.ltBit 0 1 a.toNat b.toNat := by
  by_cases h : a.ult b <;> simp [bgeuPure, Alu.ltBit, Alu.ltRaw, h]

end RiscvEquiv
