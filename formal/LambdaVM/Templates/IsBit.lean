module

public import LambdaVM.Field
public import Clean.Gadgets.Boolean

@[expose] public section

/-!
# IS_BIT (spec/is_bit.typ)

Constraint template: when `cond ≠ 0`, `X` must lie in `{0, 1}`.
Single constraint: `cond · X · (1 − X) = 0`.
-/

namespace LambdaVM.IsBit

structure Inputs (F : Type) where
  cond : F
  x : F
deriving ProvableStruct

def main (inputs : Var Inputs (F p)) : Circuit (F p) Unit :=
  assertZero (inputs.cond * inputs.x * (1 - inputs.x))

def circuit : FormalAssertion (F p) Inputs where
  main

  Spec := fun { cond, x } => cond ≠ 0 → IsBool x

  soundness := by
    circuit_proof_start
    intro h_cond
    simp only [mul_eq_zero, sub_eq_zero] at h_holds
    rcases h_holds with (h | h) | h
    · exact absurd h h_cond
    · exact Or.inl h
    · exact Or.inr h.symm

  completeness := by
    circuit_proof_start
    by_cases h_cond : input_cond = 0
    · simp [h_cond]
    · rcases h_spec h_cond with h | h <;> simp [h]

end LambdaVM.IsBit
