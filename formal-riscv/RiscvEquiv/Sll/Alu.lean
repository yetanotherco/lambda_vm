import RiscvEquiv.Sll.Bridge
import RiscvEquiv.AluBus

/-!
# SLL: what the CPU learns from the ALU bus is Sail's result

The CPU does not read SHIFT rows: it pulls a message from the ALU bus and learns
`Alu.Guarantees` (`formal/LambdaVM/Buses/Alu.lean`), which SHIFT's soundness establishes for
every message it pushes. This file instantiates the SHIFT clause (`shift_alu`, in
`AluBus.lean`) at the flags of `SLL`: if the operands are the 64-bit values `a` and `b`, the
result is Sail's `sllPure a b`.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `SLL` (`opsel = SHIFT`, `word = 0`, `signed = 0`,
`direction = 0`) carries Sail's `SLL` result.

This theorem proves that, in the Clean model, an ALU message (`Alu.Msg`) with the flags of
`SLL` that satisfies `Alu.Guarantees` has `sllPure` as its result (which
`execute_SLL` proves equal to Sail's `SLL`). -/
theorem sll_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (hflags : m.flags.val = opselSHIFT)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi % 64 = b.toNat % 64) :
    dword m.outLo m.outHi = (sllPure a b).toNat := by
  rw [sllPure_toNat]
  exact shift_alu m hg 0 0 (by norm_num) (by norm_num) (by simpa using hflags)
    _ b ha hb

end RiscvEquiv
