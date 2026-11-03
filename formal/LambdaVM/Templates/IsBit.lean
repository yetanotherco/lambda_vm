module

public import LambdaVM.Field
public import Clean.Gadgets.Boolean

@[expose] public section

/-!
# IS_BIT (spec/is_bit.typ)

Constraint template: when `cond ≠ 0`, `X` must lie in `{0, 1}`.
Single constraint: `cond · X · (1 − X) = 0`. Chips use the unconditional form `isBit`
on their flags; `IsBit.circuit` is the general template with its meaning proven.
-/

namespace LambdaVM

/-- `IS_BIT<1, X>`: `x` is a bit. -/
@[circuit_norm]
def isBit (x : Expression (F p)) : Circuit (F p) Unit :=
  assertZero (x * (1 - x))

-- Circuit inference does not unfold user wrappers; bridge it to its body.
instance (x : Expression (F p)) : ExplicitCircuit (isBit x) :=
  .of_eq (circuit' := assertZero (x * (1 - x))) rfl (by infer_explicit_circuit)

namespace IsBit

structure Inputs (F : Type) where
  cond : F
  x : F
deriving ProvableStruct

def main (inputs : Var Inputs (F p)) : Circuit (F p) Unit :=
  assertZero (inputs.cond * inputs.x * (1 - inputs.x))

instance (inputs : Var Inputs (F p)) : ExplicitCircuit (main inputs) :=
  .of_eq (circuit' := assertZero (inputs.cond * inputs.x * (1 - inputs.x))) rfl
    (by infer_explicit_circuit)

def circuit : FormalAssertion (F p) Inputs where
  main

  Spec := fun { cond, x } => cond ≠ 0 → IsBool x

  soundness := by
    circuit_proof_start
    intro h_cond
    rw [mul_assoc, mul_eq_zero] at h_holds
    exact mul_one_sub_eq_zero_iff.mp (h_holds.resolve_left h_cond)

  completeness := by
    circuit_proof_start
    by_cases h_cond : input_cond = 0
    · simp [h_cond]
    · rw [mul_assoc, mul_one_sub_eq_zero_iff.mpr (h_spec h_cond), mul_zero]

end IsBit

end LambdaVM
