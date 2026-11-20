import RiscvEquiv.Beq.Sail
import LambdaVM.Buses.Alu

/-!
# BEQ and BNE: the Clean model's equality bit is Sail's condition

The EQ table's result is `Alu.eqBit invert A B`, `(A = B) xor invert` on naturals. With
`invert = 0` it is the `BEQ` condition, with `invert = 1` the `BNE` one. Two 64-bit vectors
are equal exactly when their naturals are, so the bridge is one case split.
-/

namespace RiscvEquiv

open LambdaVM

/-- `beqPure` as a bit is `eqBit` with the flags of `BEQ` (`invert = 0`). -/
theorem beqPure_toNat (a b : BitVec 64) :
    (beqPure a b).toNat = Alu.eqBit 0 a.toNat b.toNat := by
  by_cases h : a = b <;> simp [beqPure, Alu.eqBit, h, BitVec.toNat_inj]

/-- `bnePure` as a bit is `eqBit` with the flags of `BNE` (`invert = 1`). -/
theorem bnePure_toNat (a b : BitVec 64) :
    (bnePure a b).toNat = Alu.eqBit 1 a.toNat b.toNat := by
  by_cases h : a = b <;> simp [bnePure, Alu.eqBit, h, BitVec.toNat_inj]

end RiscvEquiv
