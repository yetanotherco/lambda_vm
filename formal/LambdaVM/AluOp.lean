module

public import LambdaVM.Field

@[expose] public section

/-!
# `alu_op` descriptors (`prover/src/tables/types.rs`)

One enumeration serves two places: the `opsel` of the `BYTE_ALU` lookup (AND, OR, XOR) and the
low bits of the flag byte on the `ALU` bus (EQ, LT, SHIFT, MUL).

`alu_op::SHIFTW` is 6 and has no name here: `shift.toml` encodes it as `opsel(SHIFT) +
word_instr`, so the SHIFT clause of the bus covers both.
-/

namespace LambdaVM

def opselAND : ℕ := 0
def opselOR : ℕ := 1
def opselXOR : ℕ := 2
def opselEQ : ℕ := 3
def opselLT : ℕ := 4
def opselSHIFT : ℕ := 5
def opselMUL : ℕ := 7
def opselDIVREM : ℕ := 8

end LambdaVM
