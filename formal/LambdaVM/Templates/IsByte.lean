module

public import LambdaVM.Tables.Bitwise
public import LambdaVM.Templates.IsBit

@[expose] public section

/-!
# IS_BYTE / ARE_BYTES (spec/src/is_byte.toml)

`IS_BYTE<cond, X>` is one interaction `ARE_BYTES[X, 0]` with multiplicity `cond` (the spec
writes `[0, X]`; the prover's single-byte checks send `[X, 0]`, and this follows the prover);
`ARE_BYTES<X, Y>` is the unconditional two-byte form (the CPU sends it on every row).
Both are helpers a chip expands inline: `cond` is a flag that the chip itself constrains to be
boolean, and a conditional interaction needs that constraint in the same circuit to be lawful.
No chip of this model calls them yet: SHIFT pulls `ARE_BYTES` directly.
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
  AreBytes.pullIf cond (x, 0)

-- Circuit inference does not unfold user wrappers; bridge each one to its body.
instance (x y : Expression (F p)) : ExplicitCircuit (areBytes x y) :=
  .of_eq (circuit' := AreBytes.pull (x, y)) rfl (by infer_explicit_circuit)
instance (cond x : Expression (F p)) : ExplicitCircuit (isByte cond x) :=
  .of_eq (circuit' := AreBytes.pullIf cond (x, 0)) rfl (by infer_explicit_circuit)

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
    isBit inputs.cond
    isByte inputs.cond inputs.x

  Assumptions := fun { cond, .. } => IsBool cond
  Spec := fun { cond, x } => cond = 1 → x.val < 256

  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨cond, x⟩ := input_var
    simp only [circuit_norm, isByte, isBit, areBytesTable]
    intro _ h
    rcases mul_eq_zero.mp h with h0 | h1
    · exact Or.inl h0
    · exact Or.inr (sub_eq_zero.mp h1).symm

  soundness := by
    circuit_proof_start [isByte, isBit, areBytesTable]
    obtain ⟨h_bool, h_pull⟩ := h_holds
    rw [mul_eq_zero, sub_eq_zero] at h_bool
    rcases h_bool with rfl | h1
    · simp
    · subst h1
      simp_all

  completeness := by
    circuit_proof_start [isByte, isBit, areBytesTable]
    rcases h_assumptions with rfl | rfl <;> simp_all

end IsByteCheck

end LambdaVM
