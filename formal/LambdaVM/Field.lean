module

public import Clean.Circuit
public import CompPoly.Fields.Goldilocks

@[expose] public section

namespace LambdaVM

/-- Goldilocks, the base field of the Lambda VM prover: `2^64 - 2^32 + 1`. -/
abbrev p : ℕ := Goldilocks.fieldSize

instance : Fact p.Prime := ⟨Goldilocks.is_prime⟩

-- Size facts that Clean's gadgets ask for.
instance : Fact (p > 512) := by decide
instance : Fact (p > 2^16 + 2^8) := by decide
instance : Fact (p > 2^32) := by decide

example : FiniteField (F p) := inferInstance

/-! ## Lifting field identities on small naturals back to `ℕ` -/

lemma lt_p_of_lt_two_pow_40 {n : ℕ} (h : n < 2^40) : n < p := lt_trans h (by decide)

lemma val_natCast_of_lt_two_pow_40 {n : ℕ} (h : n < 2^40) : ((n : F p)).val = n :=
  ZMod.val_natCast_of_lt (lt_p_of_lt_two_pow_40 h)

/-- Casting naturals below `p` into the field is injective. -/
lemma natCast_inj_of_lt {a b : ℕ} (ha : a < p) (hb : b < p) (h : (a : F p) = (b : F p)) : a = b := by
  rw [ZMod.natCast_eq_natCast_iff', Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hb] at h
  exact h

/-- `x.val = n` once `x` is known to be the cast of a small `n`. -/
lemma val_eq_of_eq_natCast {x : F p} {n : ℕ} (h : x = (n : F p)) (hn : n < 2^40) : x.val = n := by
  rw [h]; exact val_natCast_of_lt_two_pow_40 hn

/-- 64-bit value of a `(lo32, hi32)` pair. -/
def dword (lo hi : F p) : ℕ := lo.val + 2^32 * hi.val

lemma mul_one_sub_eq_zero_iff {x : F p} : x * (1 - x) = 0 ↔ IsBool x := by
  rw [mul_eq_zero, sub_eq_zero, IsBool, eq_comm (a := (1 : F p))]

lemma isBool_of_val_lt_two {x : F p} (h : x.val < 2) : IsBool x := by
  have hx : x = (x.val : F p) := (ZMod.natCast_zmod_val x).symm
  rcases Nat.lt_succ_iff.mp h |>.lt_or_eq with h0 | h1
  · left; rw [hx, Nat.lt_one_iff.mp h0]; simp
  · right; rw [hx, h1]; simp

/-- `2⁻³²`, the constant `INV_SHIFT_32` of the prover's constraint templates. -/
def inv32 : F p := ((2 : F p)^32)⁻¹

lemma two_pow_32_ne_zero : (2 : F p)^32 ≠ 0 := by
  have h : ((2^32 : ℕ) : F p) ≠ 0 := by
    intro h0
    have := congrArg ZMod.val h0
    rw [val_natCast_of_lt_two_pow_40 (by norm_num), ZMod.val_zero] at this
    norm_num at this
  push_cast at h
  exact h

lemma inv32_mul_two_pow : inv32 * (2 : F p)^32 = 1 := inv_mul_cancel₀ two_pow_32_ne_zero
lemma two_pow_mul_inv32 : (2 : F p)^32 * inv32 = 1 := mul_inv_cancel₀ two_pow_32_ne_zero

/-- A carry constraint `c = (a − b)·2⁻³²` with `c` boolean and `a, b` small lifts to
`a = b + c·2³²` on naturals. -/
lemma carry_lift {a b c : F p} (ha : a.val < 2^36) (hb : b.val < 2^36) (hc : IsBool c)
    (hdef : c = (a - b) * inv32) : a.val = b.val + c.val * 2^32 := by
  have hab : a = b + c * 2^32 := by
    rw [hdef, mul_assoc, inv32_mul_two_pow, mul_one]; ring
  have hcast : ((a.val : ℕ) : F p) = ((b.val + c.val * 2^32 : ℕ) : F p) := by
    push_cast
    simp only [ZMod.natCast_zmod_val]
    exact hab
  have hc2 := IsBool.val_lt_two hc
  exact natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega)) (lt_p_of_lt_two_pow_40 (by omega)) hcast

/-- A field element with value below 2 is `0` or `1`, together with its value. -/
lemma eq_zero_or_one_of_val {x : F p} (h : x.val < 2) :
    (x = 0 ∧ x.val = 0) ∨ (x = 1 ∧ x.val = 1) := by
  rcases isBool_of_val_lt_two h with rfl | rfl
  · left; exact ⟨rfl, ZMod.val_zero⟩
  · right; refine ⟨rfl, ?_⟩
    rw [← Nat.cast_one]; exact val_natCast_of_lt_two_pow_40 (by norm_num)

lemma val_one_eq : (1 : F p).val = 1 := by
  rw [← Nat.cast_one]; exact val_natCast_of_lt_two_pow_40 (by norm_num)

lemma val_ofNat (n : ℕ) [n.AtLeastTwo] (h : n < 2^40) : (OfNat.ofNat n : F p).val = n := by
  rw [← Nat.cast_ofNat]; exact val_natCast_of_lt_two_pow_40 h

lemma three_lt_p : 3 < p := by decide

/-- A field difference of two small naturals is small only if it does not wrap. -/
lemma val_sub_of_val_lt {a b : ℕ} (ha : a < 2^40) (hb : b < 2^40)
    (h : ((a : F p) - (b : F p)).val < 2^40) : b ≤ a ∧ ((a : F p) - (b : F p)).val = a - b := by
  by_cases hab : b ≤ a
  · refine ⟨hab, ?_⟩
    rw [← Nat.cast_sub hab]
    exact val_natCast_of_lt_two_pow_40 (by omega)
  · exfalso
    have hlt : a < b := by omega
    have h1 : ((a : F p) - (b : F p)) = -(((b - a : ℕ) : F p)) := by
      rw [Nat.cast_sub hlt.le]; ring
    rw [h1, ZMod.neg_val] at h
    have hne : ((b - a : ℕ) : F p) ≠ 0 := by
      intro h0
      have := congrArg ZMod.val h0
      rw [val_natCast_of_lt_two_pow_40 (by omega), ZMod.val_zero] at this
      omega
    rw [if_neg hne, val_natCast_of_lt_two_pow_40 (by omega)] at h
    have hp : 2^41 < p := by decide
    omega

/-- A requirement `¬ -e = -1 → ¬ e = 0 → P` is vacuous for a boolean `e`. -/
lemma req_vacuous {e : F p} (h : IsBool e) {P : Prop} : ¬ -e = -1 → ¬ e = 0 → P := by
  intro h1 h0
  rcases h with rfl | rfl
  · exact absurd rfl h0
  · exact absurd rfl h1

lemma val_neg_two : (-2 : F p).val = p - 2 := by
  have h2 : (2 : F p) ≠ 0 := by
    intro h
    have := congrArg ZMod.val h
    rw [val_ofNat 2 (by norm_num), ZMod.val_zero] at this
    omega
  rw [ZMod.neg_val, if_neg h2, val_ofNat 2 (by norm_num)]

/-- Three bits whose complement `1 − l0 − l1 − l2` is also a bit are at most one-hot. -/
lemma onehot_of_isBool {l0 l1 l2 : F p} (h0 : IsBool l0) (h1 : IsBool l1) (h2 : IsBool l2)
    (h3 : IsBool (1 - l0 - l1 - l2)) : l0.val + l1.val + l2.val ≤ 1 := by
  have hp := three_lt_p
  have h3' := IsBool.val_lt_two h3
  rcases h0 with rfl | rfl <;> rcases h1 with rfl | rfl <;> rcases h2 with rfl | rfl
  all_goals simp only [ZMod.val_zero, val_one_eq]
  all_goals try omega
  all_goals first
    | (norm_num at h3'; done)
    | (norm_num at h3'; simp only [val_neg_two] at h3'; omega)

lemma eq_natCast_of_val {x : F p} {n : ℕ} (h : x.val = n) : x = (n : F p) := by
  rw [← h, ZMod.natCast_zmod_val]

/-- A conditional interaction whose multiplicity is a bit (`μ·(1 − μ) = 0`) never has an
active requirement. -/
lemma requirement_vacuous {mu : F p} (h : mu * (1 - mu) = 0) (hne1 : ¬ -mu = -1) (hne0 : ¬ mu = 0) :
    False := by
  rcases mul_one_sub_eq_zero_iff.mp h with h0 | h1
  · exact hne0 h0
  · exact hne1 (by rw [h1])

end LambdaVM
