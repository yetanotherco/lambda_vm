module

public import LambdaVM.Tables.Bitwise

@[expose] public section

/-!
# IS_BYTE / ARE_BYTES (spec/is_byte.typ)

`IS_BYTE<cond, X>` is one interaction `ARE_BYTES[0, X]` with multiplicity `cond`;
`ARE_BYTES<X, Y>` is the unconditional two-byte form (the CPU sends it on every row).
Both are macros expanded inline in the chips that use them: `cond` is a flag that the chip
itself constrains to be boolean, and a conditional interaction needs that constraint in the
same circuit to be lawful.
-/

namespace LambdaVM

open Bitwise (AreBytes areBytesTable)

/-- `ARE_BYTES<X, Y>`: both `x` and `y` are bytes. -/
@[circuit_norm]
def areBytes (x y : Expression (F p)) : Circuit (F p) Unit :=
  AreBytes.pull (x, y)

/-- `IS_BYTE<cond, X>`: `x` is a byte whenever `cond = 1`. -/
@[circuit_norm]
def isByte (cond x : Expression (F p)) : Circuit (F p) Unit :=
  AreBytes.pullIf cond (0, x)

-- Circuit inference does not unfold user wrappers; bridge each one to its body.
instance (x y : Expression (F p)) : ExplicitCircuit (areBytes x y) :=
  .of_eq (circuit' := AreBytes.pull (x, y)) rfl (by infer_explicit_circuit)
instance (cond x : Expression (F p)) : ExplicitCircuit (isByte cond x) :=
  .of_eq (circuit' := AreBytes.pullIf cond (0, x)) rfl (by infer_explicit_circuit)

/-! The two templates as stand-alone assertions, to pin down their meaning. -/

namespace AreBytesCheck

def circuit : FormalAssertion (F p) fieldPair where
  main input := do areBytes input.1 input.2
  Spec | (x, y) => x.val < 256 ∧ y.val < 256
  soundness := by
    circuit_proof_start [areBytes, areBytesTable]
    exact h_holds
  completeness := by
    circuit_proof_start [areBytes, areBytesTable]
    exact h_spec

end AreBytesCheck

namespace IsByteCheck

structure Inputs (F : Type) where
  cond : F
  x : F
deriving ProvableStruct

def circuit : FormalAssertion (F p) Inputs where
  main inputs := do
    -- IS_BIT<1, cond>: in a chip this is one of the chip's own flag constraints
    assertZero (inputs.cond * (1 - inputs.cond))
    isByte inputs.cond inputs.x

  Assumptions := fun { cond, .. } => IsBool cond
  Spec := fun { cond, x } => cond = 1 → x.val < 256

  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨cond, x⟩ := input_var
    simp only [circuit_norm, isByte, areBytesTable]
    grind

  soundness := by
    circuit_proof_start [isByte, areBytesTable]
    obtain ⟨h_bool, h_pull⟩ := h_holds
    rw [mul_eq_zero, sub_eq_zero] at h_bool
    rcases h_bool with rfl | h1
    · simp
    · subst h1
      simp_all

  completeness := by
    circuit_proof_start [isByte, areBytesTable]
    rcases h_assumptions with rfl | rfl <;> simp_all

end IsByteCheck

end LambdaVM
