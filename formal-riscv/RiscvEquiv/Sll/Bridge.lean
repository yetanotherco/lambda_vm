import RiscvEquiv.Sll.Sail
import LambdaVM.Buses.Alu

/-!
# SLL: the Clean model's shift function is Sail's

The Clean model (`formal/`) states what SHIFT computes with `Alu.shiftResult`, on naturals.
With `word = 0`, `signed = 0` and `direction = 0` it is a 64-bit left shift. This file
proves it is `sllPure` read as a natural.
-/

namespace RiscvEquiv

open LambdaVM

/-- `shiftResult` with the flags of `SLL`: `x` shifted left by `s mod 64`, modulo `2^64`.

`Alu.shiftResult` (formal/LambdaVM/Buses/Alu.lean) is the Lean function that gives SHIFT's
result; it covers all the shifts with three flags.

(word = 0, signed = 0, direction = 0) are the flags for SLL. -/
lemma shiftResult_sll (x s : ℕ) : Alu.shiftResult 0 0 0 x s = (x <<< (s % 64)) % 2^64 := by
  simp [Alu.shiftResult]

/-- `sllPure` as a natural is `shiftResult` with the flags of `SLL`.

sllPure returns a BitVec 64, a 64-bit vector which is the type Sail uses.
Alu.shiftResult, on the other hand, returns a natural number, which is the type the Clean files use
(because in the circuit everything is a field element).
They cannot be compared directly, so the BitVec is converted to a number with .toNat. -/
theorem sllPure_toNat (a b : BitVec 64) :
    (sllPure a b).toNat = Alu.shiftResult 0 0 0 a.toNat b.toNat := by
  rw [shiftResult_sll, sllPure, BitVec.toNat_shiftLeft]

end RiscvEquiv
