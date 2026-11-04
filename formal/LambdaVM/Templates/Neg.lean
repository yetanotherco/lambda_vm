module

public import LambdaVM.Templates.Add
public import LambdaVM.Tables.Bitwise

@[expose] public section

/-!
# NEG (spec/src/neg.toml)

`NEG<neg; x>` under `cond`: `neg` is the two's complement of the `DWordHL` operand `x`, i.e.
`neg = (2⁶⁴ − x) mod 2⁶⁴`, written as two 32-bit words.

The spec gives the two carries of `x + neg = 0 (mod 2⁶⁴)` as virtual columns,
`carry[0] = 2⁻³²·(x[0] + 2¹⁶·x[1] + neg[0])` and
`carry[1] = 2⁻³²·(x[2] + 2¹⁶·x[3] + neg[1] + carry[0])`, which are exactly `ADD`'s carries of
`x + neg = 0`, so this file reuses `Add.carry0`, `Add.carry1`, `Add.carries_lift` and
`Add.carries_of_sum` with `sum = 0`.

Unlike `ADD`, the carries are not asserted to be bits by `IS_BIT`: each one is read off a
`ZERO` lookup, whose output flag is boolean by the table's own guarantee. `carry[0] = 1` says
the low word of `x` is nonzero and `carry[1] = 1` says `x` itself is nonzero, which is what
makes the result the two's complement rather than `2⁶⁴ − x` unconditionally.

`IS_HALF[x[i]]` and `IS_BIT<cond>` are assumptions of the template, not constraints, so a chip
that inlines `neg` range-checks the halves and constrains the flag itself. `NegCheck.circuit`
adds the flag constraint to pin the meaning down on its own, the way `IsByteCheck` does.

The only chip that uses this template is `dvrm.toml`.
-/

namespace LambdaVM

open Bitwise (Zero zeroTable)

namespace Neg

/-- The `DWordHL` operand `x` as `(lo, hi)` 32-bit words. -/
@[circuit_norm]
def toWL (x0 x1 x2 x3 : Expression (F p)) : DWordWL (Expression (F p)) :=
  ⟨x0 + 65536 * x1, x2 + 65536 * x3⟩

/-- The zero `DWordWL`, the sum that `x + neg` has to reach. -/
@[circuit_norm]
def zeroWL : DWordWL (Expression (F p)) := ⟨0, 0⟩

/-- 64-bit value of a `DWordHL` operand, four 16-bit halves. -/
def dwordHL (x0 x1 x2 x3 : F p) : ℕ :=
  x0.val + 2^16 * x1.val + 2^32 * x2.val + 2^48 * x3.val

/-! ## The field sums of the lookups, read on naturals

Each `ZERO` lookup takes a sum of halves, small enough never to wrap, so its field value is
zero exactly when every half is. -/

/-- Two halves sum without wrapping. -/
lemma val_sum2 {a b : F p} (ha : a.val < 2^16) (hb : b.val < 2^16) :
    (a + b).val = a.val + b.val :=
  val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)

/-- Four halves sum without wrapping. -/
lemma val_sum4 {a b c d : F p} (ha : a.val < 2^16) (hb : b.val < 2^16) (hc : c.val < 2^16)
    (hd : d.val < 2^16) : (a + b + c + d).val = a.val + b.val + c.val + d.val :=
  val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)

/-- A pair of halves as a 32-bit word. -/
lemma val_wl {a b : F p} (ha : a.val < 2^16) (hb : b.val < 2^16) :
    (a + 65536 * b).val = a.val + 65536 * b.val :=
  val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)

end Neg

/-- `NEG<neg; x>` under `cond`: two `ZERO` lookups reading the carries of `x + neg = 0`. -/
@[circuit_norm]
def neg (cond x0 x1 x2 x3 : Expression (F p)) (n : DWordWL (Expression (F p))) :
    Circuit (F p) Unit := do
  Zero.pullIf cond (x0 + x1, 1 - Add.carry0 (Neg.toWL x0 x1 x2 x3) n Neg.zeroWL)
  Zero.pullIf cond (x0 + x1 + x2 + x3, 1 - Add.carry1 (Neg.toWL x0 x1 x2 x3) n Neg.zeroWL)

-- Circuit inference does not unfold user wrappers; bridge it to its body.
instance (cond x0 x1 x2 x3 : Expression (F p)) (n : DWordWL (Expression (F p))) :
    ExplicitCircuit (neg cond x0 x1 x2 x3 n) :=
  .of_eq (circuit' := do
      Zero.pullIf cond (x0 + x1, 1 - Add.carry0 (Neg.toWL x0 x1 x2 x3) n Neg.zeroWL)
      Zero.pullIf cond (x0 + x1 + x2 + x3, 1 - Add.carry1 (Neg.toWL x0 x1 x2 x3) n Neg.zeroWL))
    rfl (by infer_explicit_circuit)

/-! The template as a stand-alone assertion, to pin down its meaning. -/

namespace NegCheck

structure Inputs (F : Type) where
  cond : F
  x0 : F
  x1 : F
  x2 : F
  x3 : F
  neg : DWordWL F
deriving ProvableStruct

def circuit : FormalAssertion (F p) Inputs where
  main i := do
    -- IS_BIT<1, cond>: in a chip this is one of the chip's own flag constraints
    isBit i.cond
    neg i.cond i.x0 i.x1 i.x2 i.x3 i.neg

  Assumptions i :=
    IsBool i.cond ∧
    i.x0.val < 2^16 ∧ i.x1.val < 2^16 ∧ i.x2.val < 2^16 ∧ i.x3.val < 2^16 ∧
    i.neg.lo.val < 2^32 ∧ i.neg.hi.val < 2^32

  Spec i :=
    i.cond ≠ 0 →
      dword i.neg.lo i.neg.hi = (2^64 - Neg.dwordHL i.x0 i.x1 i.x2 i.x3) % 2^64

  -- the conditional pulls are lawful because the circuit constrains `cond` to be a bit
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨cond, x0, x1, x2, x3, nlo, nhi⟩ := input_var
    simp only [circuit_norm, neg, isBit, zeroTable]
    intro env hc
    exact mul_one_sub_eq_zero_iff.mp hc

  soundness := by
    circuit_proof_start [neg, isBit, zeroTable, Neg.toWL, Neg.zeroWL, Neg.dwordHL, dword,
      Add.carry0, Add.carry1]
    obtain ⟨hcond, hx0, hx1, hx2, hx3, hnlo, hnhi⟩ := h_assumptions
    obtain ⟨h_bool, h_p0, h_p1⟩ := h_holds
    refine ⟨?_, mul_one_sub_eq_zero_iff.mp h_bool⟩
    intro hne
    have h1 : input_cond = 1 := (mul_one_sub_eq_zero_iff.mp h_bool).resolve_left hne
    obtain ⟨-, hlo⟩ := h_p0 h1
    obtain ⟨-, hhi⟩ := h_p1 h1
    -- each lookup flag is `1 − carry`, so each carry is `0` or `1` by the table's guarantee
    obtain ⟨c0, hc0def, hc0val⟩ : ∃ c0 : F p,
        c0 = (input_x0 + 65536 * input_x1 + input_neg_lo - 0) * inv32 ∧
          c0 = 1 - (if input_x0 + input_x1 = 0 then 1 else 0) :=
      ⟨_, rfl, by linear_combination -hlo⟩
    obtain ⟨c1, hc1def, hc1val⟩ : ∃ c1 : F p,
        c1 = (input_x2 + 65536 * input_x3 + input_neg_hi + c0 - 0) * inv32 ∧
          c1 = 1 - (if input_x0 + input_x1 + input_x2 + input_x3 = 0 then 1 else 0) :=
      ⟨_, rfl, by rw [hc0def]; linear_combination -hhi⟩
    have hb0 : IsBool c0 := by rw [hc0val]; split <;> simp [IsBool]
    have hb1 : IsBool c1 := by rw [hc1val]; split <;> simp [IsBool]
    -- the carry equations, read on naturals, through ADD's lemma with `sum = 0`
    have hwlo := Neg.val_wl hx0 hx1
    have hwhi := Neg.val_wl hx2 hx3
    obtain ⟨e0, e1⟩ := Add.carries_lift (l0 := input_x0 + 65536 * input_x1)
      (l1 := input_x2 + 65536 * input_x3) (r0 := input_neg_lo) (r1 := input_neg_hi)
      (s0 := 0) (s1 := 0) (by omega) (by omega) hnlo hnhi (by simp) (by simp)
      hc0def hc1def hb0 hb1
    simp only [ZMod.val_zero] at e0 e1
    -- `carry[1] = 1` exactly when `x ≠ 0`, which is what makes the result two's complement
    have hsum4 := Neg.val_sum4 hx0 hx1 hx2 hx3
    have hzero : input_x0 + input_x1 + input_x2 + input_x3 = 0 ↔
        input_x0.val + 2^16 * input_x1.val + 2^32 * input_x2.val + 2^48 * input_x3.val = 0 := by
      rw [← ZMod.val_eq_zero, hsum4]; omega
    have hv0 := IsBool.val_lt_two hb0
    rcases eq_or_ne (input_x0 + input_x1 + input_x2 + input_x3) 0 with h | h
    · simp only [h, ↓reduceIte, sub_self] at hc1val
      have hc1z : c1.val = 0 := by rw [hc1val]; simp
      have hx := hzero.mp h
      omega
    · simp only [h, ↓reduceIte, sub_zero] at hc1val
      have hc1o : c1.val = 1 := by rw [hc1val]; exact val_one_eq
      have hx : ¬ (input_x0.val + 2^16 * input_x1.val + 2^32 * input_x2.val
          + 2^48 * input_x3.val = 0) := fun hx => h (hzero.mpr hx)
      omega

  completeness := by
    circuit_proof_start [neg, isBit, zeroTable, Neg.toWL, Neg.zeroWL, Neg.dwordHL, dword,
      Add.carry0, Add.carry1]
    obtain ⟨hcond, hx0, hx1, hx2, hx3, hnlo, hnhi⟩ := h_assumptions
    have hwlo := Neg.val_wl hx0 hx1
    have hwhi := Neg.val_wl hx2 hx3
    have hsum2 := Neg.val_sum2 hx0 hx1
    have hsum4 := Neg.val_sum4 hx0 hx1 hx2 hx3
    rcases hcond with rfl | rfl
    · -- an inactive row owes nothing: both lookups are disabled
      refine ⟨by simp, ?_, ?_⟩ <;> intro h1 <;> simp at h1
    -- an active row: the spec fixes `neg`, so `ADD`'s converse gives both carries
    have hspec := h_spec one_ne_zero
    have hsum : (dword (input_x0 + 65536 * input_x1) (input_x2 + 65536 * input_x3)
        + dword input_neg_lo input_neg_hi) % 2^64 = dword 0 0 := by
      simp only [dword, hwlo, hwhi, ZMod.val_zero]
      omega
    obtain ⟨k0, k1, hk0, hk1, e0, e1, hc0, hc1⟩ :=
      Add.carries_of_sum (l0 := input_x0 + 65536 * input_x1)
        (l1 := input_x2 + 65536 * input_x3) (r0 := input_neg_lo) (r1 := input_neg_hi)
        (s0 := 0) (s1 := 0) (by omega) (by omega) hnlo hnhi (by simp) (by simp) hsum
    simp only [ZMod.val_zero] at e0 e1
    refine ⟨by simp, ?_, ?_⟩ <;> intro _
    · -- `carry[0] = 1` exactly when the low word of `x` is nonzero
      refine ⟨by omega, ?_⟩
      rw [hc0]
      rcases eq_or_ne (input_x0 + input_x1) 0 with h | h
      · have hv : (input_x0 + input_x1).val = 0 := by rw [h]; simp
        have hk : k0 = 0 := by omega
        simp [h, hk]
      · have hv : (input_x0 + input_x1).val ≠ 0 := by rw [Ne, ZMod.val_eq_zero]; exact h
        have hk : k0 = 1 := by omega
        simp [h, hk]
    · -- `carry[1] = 1` exactly when `x` itself is nonzero
      refine ⟨by omega, ?_⟩
      rw [hc1]
      rcases eq_or_ne (input_x0 + input_x1 + input_x2 + input_x3) 0 with h | h
      · have hv : (input_x0 + input_x1 + input_x2 + input_x3).val = 0 := by rw [h]; simp
        have hk : k1 = 0 := by omega
        simp [h, hk]
      · have hv : (input_x0 + input_x1 + input_x2 + input_x3).val ≠ 0 := by
          rw [Ne, ZMod.val_eq_zero]; exact h
        have hk : k1 = 1 := by omega
        simp [h, hk]

end NegCheck

end LambdaVM
