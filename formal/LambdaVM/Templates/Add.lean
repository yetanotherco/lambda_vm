module

public import LambdaVM.Templates.IsBit
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# ADD and SUB (spec/add.typ)

Constraint templates on `DWordWL` operands. `ADD<cond; lhs, rhs> = sum` has two virtual
carries, `carry[0] = 2⁻³²·(lhs[0] + rhs[0] − sum[0])` and
`carry[1] = 2⁻³²·(lhs[1] + rhs[1] + carry[0] − sum[1])`, and constrains both to be bits when
`cond ≠ 0`. `SUB<cond; lhs, rhs> = diff` is `ADD<cond; rhs, diff> = lhs`. Chips use the
unconditional forms `add` and `sub` (`emit_add_pair` in the prover); `Add.circuit` is the
general template with its meaning proven.
-/

namespace LambdaVM

/-- A 64-bit operand as `(lo, hi)` 32-bit words. -/
structure DWordWL (F : Type) where
  lo : F
  hi : F

namespace Add

variable (lhs rhs sum : DWordWL (Expression (F p)))

/-- `carry[0]` of `lhs + rhs = sum`. -/
def carry0 : Expression (F p) := (lhs.lo + rhs.lo - sum.lo) * inv32

/-- `carry[1]` of `lhs + rhs = sum`. -/
def carry1 : Expression (F p) := (lhs.hi + rhs.hi + carry0 lhs rhs sum - sum.hi) * inv32

end Add

/-- `ADD<1; lhs, rhs> = sum`: both carries are bits. -/
@[circuit_norm]
def add (lhs rhs sum : DWordWL (Expression (F p))) : Circuit (F p) Unit := do
  isBit (Add.carry0 lhs rhs sum)
  isBit (Add.carry1 lhs rhs sum)

/-- `SUB<1; lhs, rhs> = diff`, as `ADD<1; rhs, diff> = lhs`. -/
@[circuit_norm]
def sub (lhs rhs diff : DWordWL (Expression (F p))) : Circuit (F p) Unit := add rhs diff lhs

-- Circuit inference does not unfold user wrappers; bridge them to their bodies.
instance (lhs rhs sum : DWordWL (Expression (F p))) : ExplicitCircuit (add lhs rhs sum) :=
  .of_eq (circuit' := do isBit (Add.carry0 lhs rhs sum); isBit (Add.carry1 lhs rhs sum)) rfl
    (by infer_explicit_circuit)

instance (lhs rhs diff : DWordWL (Expression (F p))) : ExplicitCircuit (sub lhs rhs diff) :=
  inferInstanceAs (ExplicitCircuit (add rhs diff lhs))

namespace Add

structure Inputs (F : Type) where
  cond : F
  lhs0 : F
  lhs1 : F
  rhs0 : F
  rhs1 : F
  sum0 : F
  sum1 : F
deriving ProvableStruct

def main (i : Var Inputs (F p)) : Circuit (F p) Unit := do
  let c0 := carry0 ⟨i.lhs0, i.lhs1⟩ ⟨i.rhs0, i.rhs1⟩ ⟨i.sum0, i.sum1⟩
  let c1 := carry1 ⟨i.lhs0, i.lhs1⟩ ⟨i.rhs0, i.rhs1⟩ ⟨i.sum0, i.sum1⟩
  assertZero (i.cond * c0 * (1 - c0))
  assertZero (i.cond * c1 * (1 - c1))

/-- `ADD<cond; lhs, rhs> = sum` on words in range: when `cond ≠ 0`, `sum = lhs + rhs (mod 2⁶⁴)`. -/
def circuit : FormalAssertion (F p) Inputs where
  main

  Assumptions i :=
    i.lhs0.val < 2^32 ∧ i.lhs1.val < 2^32 ∧ i.rhs0.val < 2^32 ∧ i.rhs1.val < 2^32 ∧
    i.sum0.val < 2^32 ∧ i.sum1.val < 2^32

  Spec i := i.cond ≠ 0 → dword i.sum0 i.sum1 = (dword i.lhs0 i.lhs1 + dword i.rhs0 i.rhs1) % 2^64

  soundness := by
    circuit_proof_start [main, carry0, carry1, dword]
    obtain ⟨hl0, hl1, hr0, hr1, hs0, hs1⟩ := h_assumptions
    obtain ⟨h0, h1⟩ := h_holds
    intro hcond
    rw [mul_assoc, mul_eq_zero] at h0 h1
    have hc0 := mul_one_sub_eq_zero_iff.mp (h0.resolve_left hcond)
    have hc1 := mul_one_sub_eq_zero_iff.mp (h1.resolve_left hcond)
    set c0 := (input_lhs0 + input_rhs0 - input_sum0) * inv32 with hc0def
    set c1 := (input_lhs1 + input_rhs1 + c0 - input_sum1) * inv32 with hc1def
    have hc0_2 := IsBool.val_lt_two hc0
    have hc1_2 := IsBool.val_lt_two hc1
    have hval_lo : (input_lhs0 + input_rhs0).val = input_lhs0.val + input_rhs0.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hval_hi : (input_lhs1 + input_rhs1 + c0).val = input_lhs1.val + input_rhs1.val + c0.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have e0 := carry_lift (by omega) (by omega) hc0 hc0def
    have e1 := carry_lift (by omega) (by omega) hc1 hc1def
    rw [hval_lo] at e0
    rw [hval_hi] at e1
    omega

  completeness := by
    circuit_proof_start [main, carry0, carry1, dword]
    obtain ⟨hl0, hl1, hr0, hr1, hs0, hs1⟩ := h_assumptions
    by_cases hcond : input_cond = 0
    · simp [hcond]
    have hsum := h_spec hcond
    obtain ⟨k0, hk0, e0⟩ : ∃ k0, k0 < 2 ∧
        input_lhs0.val + input_rhs0.val = input_sum0.val + k0 * 2^32 :=
      ⟨(input_lhs0.val + input_rhs0.val) / 2^32, by omega, by omega⟩
    obtain ⟨k1, hk1, e1⟩ : ∃ k1, k1 < 2 ∧
        input_lhs1.val + input_rhs1.val + k0 = input_sum1.val + k1 * 2^32 :=
      ⟨(input_lhs1.val + input_rhs1.val + k0) / 2^32, by omega, by omega⟩
    set c0 := (input_lhs0 + input_rhs0 - input_sum0) * inv32 with hc0
    set c1 := (input_lhs1 + input_rhs1 + c0 - input_sum1) * inv32 with hc1
    have hc0k : c0 = (k0 : F p) := by
      have h := congrArg (fun n : ℕ => (n : F p)) e0
      push_cast at h
      simp only [ZMod.natCast_zmod_val] at h
      have h' : input_lhs0 + input_rhs0 - input_sum0 = (k0 : F p) * 2^32 := by
        linear_combination h
      rw [hc0, h', mul_assoc, two_pow_mul_inv32, mul_one]
    have hc1k : c1 = (k1 : F p) := by
      have h := congrArg (fun n : ℕ => (n : F p)) e1
      push_cast at h
      simp only [ZMod.natCast_zmod_val] at h
      have h' : input_lhs1 + input_rhs1 + c0 - input_sum1 = (k1 : F p) * 2^32 := by
        rw [hc0k]; linear_combination h
      rw [hc1, h', mul_assoc, two_pow_mul_inv32, mul_one]
    refine ⟨?_, ?_⟩
    · rw [hc0k]; rcases (by omega : k0 = 0 ∨ k0 = 1) with rfl | rfl <;> simp
    · rw [hc1k]; rcases (by omega : k1 = 0 ∨ k1 = 1) with rfl | rfl <;> simp

end Add

end LambdaVM
