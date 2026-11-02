import LeanRV64D.InstsEnd

/-!
# SLL in the Sail RISC-V model

Sail's `execute_RTYPE` (`LeanRV64D/InstsEnd.lean`) runs an R-type instruction on the machine
state: it reads `rs1` and `rs2`, computes the result and writes it to `rd`. For `SLL` the
result is `shift_bits_left x[rs1] (extractLsb x[rs2] (log2_xlen - 1) 0)`.

`sllPure` is the same computation as a pure function on values, with no state.
`execute_SLL` proves that Sail's `SLL` is "read `rs1` and `rs2`, write `sllPure` to `rd`".
The PC is not touched here: Sail's step function advances it after `RETIRE_SUCCESS`.
-/

open Sail LeanRV64D LeanRV64D.Functions

namespace RiscvEquiv

/-- `SLL` on values: `rs1` shifted left by the low 6 bits of `rs2`, modulo `2^64`. -/
def sllPure (rs1 rs2 : BitVec 64) : BitVec 64 := rs1 <<< (rs2.toNat % 64)

/-- Sail's `SLL` reads `rs1` and `rs2` and writes `sllPure x[rs1] x[rs2]` to `rd`.

execute_RTYPE is the Sail function that executes all R-type instructions.
In RISC-V, R-type is the format of instructions that operate on two registers and store the result in a third one.
There are 10 of them: ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR and AND.

The theorem says that what Sail's SLL does to the state is exactly to write the result of sllPure to rd.

With this theorem, from here on we can work with sllPure, which is just a computation, instead of the state.
-/
theorem execute_SLL (rs2 rs1 rd : regidx) :
    execute_RTYPE rs2 rs1 rd rop.SLL = (do
      let a ← rX_bits rs1
      let b ← rX_bits rs2
      wX_bits rd (sllPure a b)
      pure RETIRE_SUCCESS) := by
  simp only [execute_RTYPE, shift_bits_left, Sail.BitVec.extractLsb, sllPure,
    BitVec.shiftLeft_eq', BitVec.extractLsb_toNat]
  rfl

end RiscvEquiv
