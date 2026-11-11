module

public import LambdaVM.Field

@[expose] public section

/-!
# `alu_op` descriptors (`prover/src/tables/types.rs`)

One enumeration serves two places: the `opsel` of the `BYTE_ALU` lookup (AND, OR, XOR) and the
low bits of the flag byte on the `ALU` bus (EQ, LT, SHIFT, MUL, DIVREM).

A chip writes the flag byte it pushes or pulls with these names, as `(opselLT : F p) + …`.
The `circuit_norm` lemmas below turn each one back into its numeral, so the expressions a proof
sees, and the export, are the same as if the numeral had been written.

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

@[circuit_norm] lemma opselEQ_cast : ((opselEQ : ℕ) : F p) = 3 := by simp [opselEQ]
@[circuit_norm] lemma opselLT_cast : ((opselLT : ℕ) : F p) = 4 := by simp [opselLT]
@[circuit_norm] lemma opselSHIFT_cast : ((opselSHIFT : ℕ) : F p) = 5 := by simp [opselSHIFT]
@[circuit_norm] lemma opselMUL_cast : ((opselMUL : ℕ) : F p) = 7 := by simp [opselMUL]
@[circuit_norm] lemma opselDIVREM_cast : ((opselDIVREM : ℕ) : F p) = 8 := by simp [opselDIVREM]

end LambdaVM
