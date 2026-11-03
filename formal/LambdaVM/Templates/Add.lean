module

public import LambdaVM.Templates.IsBit
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# ADD and SUB (spec/add.typ)

Constraint templates on `DWordWL` operands. `ADD<cond; lhs, rhs> = sum` has two virtual
carries, `carry[0] = 2⁻³²·(lhs[0] + rhs[0] − sum[0])` and
`carry[1] = 2⁻³²·(lhs[1] + rhs[1] + carry[0] − sum[1])`, and applies `IS_BIT<cond, ·>` to both.
`SUB<cond; lhs, rhs> = diff` is `ADD<cond; rhs, diff> = lhs`. Chips use the unconditional forms
`add` and `sub` (`emit_add_pair` in the prover) and read the carries on naturals through
`Add.carries_lift` and `Add.carries_of_sum`; `Add.circuit` is the general template with its
meaning proven.
-/

namespace LambdaVM

/-- A 64-bit operand as `(lo, hi)` 32-bit words. -/
structure DWordWL (F : Type) where
  lo : F
  hi : F
deriving ProvableStruct

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

/-! ## The carries, read on naturals -/

/-- Both carry constraints lifted: with every word in range and both carries bits,
`lhs + rhs = sum` limb by limb. `c₀`, `c₁` are the carry expressions evaluated on a row. -/
lemma carries_lift {l0 l1 r0 r1 s0 s1 c0 c1 : F p}
    (hl0 : l0.val < 2^32) (hl1 : l1.val < 2^32) (hr0 : r0.val < 2^32) (hr1 : r1.val < 2^32)
    (hs0 : s0.val < 2^32) (hs1 : s1.val < 2^32)
    (hc0 : c0 = (l0 + r0 - s0) * inv32) (hc1 : c1 = (l1 + r1 + c0 - s1) * inv32)
    (hb0 : IsBool c0) (hb1 : IsBool c1) :
    l0.val + r0.val = s0.val + c0.val * 2^32 ∧
    l1.val + r1.val + c0.val = s1.val + c1.val * 2^32 := by
  have hc0_2 := IsBool.val_lt_two hb0
  have hval_lo : (l0 + r0).val = l0.val + r0.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hval_hi : (l1 + r1 + c0).val = l1.val + r1.val + c0.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have e0 := carry_lift (by omega) (by omega) hb0 hc0
  have e1 := carry_lift (by omega) (by omega) hb1 hc1
  rw [hval_lo] at e0
  rw [hval_hi] at e1
  exact ⟨e0, e1⟩

/-- The honest carries: when `lhs + rhs = sum (mod 2⁶⁴)` on words in range, the carry
expressions are the bits `k₀`, `k₁` of the limb-wise addition. -/
lemma carries_of_sum {l0 l1 r0 r1 s0 s1 : F p}
    (hl0 : l0.val < 2^32) (hl1 : l1.val < 2^32) (hr0 : r0.val < 2^32) (hr1 : r1.val < 2^32)
    (hs0 : s0.val < 2^32) (hs1 : s1.val < 2^32)
    (hsum : (dword l0 l1 + dword r0 r1) % 2^64 = dword s0 s1) :
    ∃ k0 k1 : ℕ, k0 < 2 ∧ k1 < 2 ∧
      l0.val + r0.val = s0.val + k0 * 2^32 ∧ l1.val + r1.val + k0 = s1.val + k1 * 2^32 ∧
      (l0 + r0 - s0) * inv32 = (k0 : F p) ∧
      (l1 + r1 + (l0 + r0 - s0) * inv32 - s1) * inv32 = (k1 : F p) := by
  simp only [dword] at hsum
  obtain ⟨k0, hk0, e0⟩ : ∃ k0, k0 < 2 ∧ l0.val + r0.val = s0.val + k0 * 2^32 :=
    ⟨(l0.val + r0.val) / 2^32, by omega, by omega⟩
  obtain ⟨k1, hk1, e1⟩ : ∃ k1, k1 < 2 ∧ l1.val + r1.val + k0 = s1.val + k1 * 2^32 :=
    ⟨(l1.val + r1.val + k0) / 2^32, by omega, by omega⟩
  have hc0k : (l0 + r0 - s0) * inv32 = (k0 : F p) := by
    have h := congrArg (fun n : ℕ => (n : F p)) e0
    push_cast at h
    simp only [ZMod.natCast_zmod_val] at h
    have h' : l0 + r0 - s0 = (k0 : F p) * 2^32 := by linear_combination h
    rw [h', mul_assoc, two_pow_mul_inv32, mul_one]
  have hc1k : (l1 + r1 + (l0 + r0 - s0) * inv32 - s1) * inv32 = (k1 : F p) := by
    have h := congrArg (fun n : ℕ => (n : F p)) e1
    push_cast at h
    simp only [ZMod.natCast_zmod_val] at h
    have h' : l1 + r1 + (l0 + r0 - s0) * inv32 - s1 = (k1 : F p) * 2^32 := by
      rw [hc0k]; linear_combination h
    rw [h', mul_assoc, two_pow_mul_inv32, mul_one]
  exact ⟨k0, k1, hk0, hk1, e0, e1, hc0k, hc1k⟩

/-! ## The template with its condition -/

structure Inputs (F : Type) where
  cond : F
  lhs : DWordWL F
  rhs : DWordWL F
  sum : DWordWL F
deriving ProvableStruct

/-- `IS_BIT<cond, carry[i]>` for both carries. -/
def main (i : Var Inputs (F p)) : Circuit (F p) Unit := do
  IsBit.main ⟨i.cond, carry0 i.lhs i.rhs i.sum⟩
  IsBit.main ⟨i.cond, carry1 i.lhs i.rhs i.sum⟩

/-- `ADD<cond; lhs, rhs> = sum` on words in range: when `cond ≠ 0`, `sum = lhs + rhs (mod 2⁶⁴)`. -/
def circuit : FormalAssertion (F p) Inputs where
  main

  Assumptions i :=
    i.lhs.lo.val < 2^32 ∧ i.lhs.hi.val < 2^32 ∧ i.rhs.lo.val < 2^32 ∧ i.rhs.hi.val < 2^32 ∧
    i.sum.lo.val < 2^32 ∧ i.sum.hi.val < 2^32

  Spec i :=
    i.cond ≠ 0 → dword i.sum.lo i.sum.hi = (dword i.lhs.lo i.lhs.hi + dword i.rhs.lo i.rhs.hi) % 2^64

  soundness := by
    circuit_proof_start [main, IsBit.main, carry0, carry1, dword]
    obtain ⟨hl0, hl1, hr0, hr1, hs0, hs1⟩ := h_assumptions
    obtain ⟨h0, h1⟩ := h_holds
    intro hcond
    rw [mul_assoc, mul_eq_zero] at h0 h1
    have hb0 := mul_one_sub_eq_zero_iff.mp (h0.resolve_left hcond)
    have hb1 := mul_one_sub_eq_zero_iff.mp (h1.resolve_left hcond)
    obtain ⟨e0, e1⟩ := carries_lift hl0 hl1 hr0 hr1 hs0 hs1 rfl rfl hb0 hb1
    have := IsBool.val_lt_two hb1
    omega

  completeness := by
    circuit_proof_start [main, IsBit.main, carry0, carry1, dword]
    obtain ⟨hl0, hl1, hr0, hr1, hs0, hs1⟩ := h_assumptions
    by_cases hcond : input_cond = 0
    · simp [hcond]
    obtain ⟨k0, k1, hk0, hk1, -, -, hc0k, hc1k⟩ :=
      carries_of_sum hl0 hl1 hr0 hr1 hs0 hs1 (h_spec hcond).symm
    refine ⟨?_, ?_⟩
    · rw [hc0k]; rcases (by omega : k0 = 0 ∨ k0 = 1) with rfl | rfl <;> simp
    · rw [hc1k]; rcases (by omega : k1 = 0 ∨ k1 = 1) with rfl | rfl <;> simp

end Add

end LambdaVM
