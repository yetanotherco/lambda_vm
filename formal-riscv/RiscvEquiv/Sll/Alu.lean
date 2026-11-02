import RiscvEquiv.Sll.Bridge

/-!
# SLL: what the CPU learns from the ALU bus is Sail's result

The CPU does not read SHIFT rows: it pulls a message from the ALU bus and learns
`Alu.Guarantees` (`formal/LambdaVM/Buses/Alu.lean`), which SHIFT's soundness establishes for
every message it pushes. This file instantiates the SHIFT clause at the flags of `SLL`:
if the operands are the 64-bit values `a` and `b`, the result is Sail's `sllPure a b`.
-/

namespace RiscvEquiv

open LambdaVM

/-- An ALU message with the flags of `SLL` (`opsel = SHIFT`, `word = 0`, `signed = 0`,
`direction = 0`) and range-checked operands carries Sail's `SLL` result.

This theorem proves that, in the Clean circuits, an ALU message (Alu.Msg) with the SLL flag that satisfies
Alu.Guarantees and range checks has sllPure as its result (which execute_SLL proves equal to Sail's SLL). -/
theorem sll_alu_matches_sail (m : Alu.Msg (F p)) (hg : Alu.Guarantees m)
    (h1 : m.in1Lo.val < 2^32) (h2 : m.in1Hi.val < 2^32)
    (h3 : m.in2Lo.val < 2^32) (h4 : m.in2Hi.val < 2^32)
    (hflags : m.flags.val = opselSHIFT)
    (a b : BitVec 64) (ha : dword m.in1Lo m.in1Hi = a.toNat)
    (hb : dword m.in2Lo m.in2Hi = b.toNat) :
    dword m.outLo m.outHi = (sllPure a b).toNat := by
  obtain ⟨-, -, hshift⟩ := hg h1 h2 h3 h4
  obtain ⟨-, -, hout⟩ := hshift 0 0 0 (by norm_num) (by norm_num) (by norm_num)
    (by simpa using hflags)
  -- SHIFT reads only the low byte of the shift amount; mod 64 it is the low bits of `b`.
  have hamt : m.in2Lo.val % 256 % 64 = b.toNat % 64 := by
    rw [← hb, dword]
    omega
  rw [hout, sllPure_toNat, ha, shiftResult_sll, shiftResult_sll, hamt]

end RiscvEquiv
