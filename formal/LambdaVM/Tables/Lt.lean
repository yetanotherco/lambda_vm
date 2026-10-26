module

public import LambdaVM.Tables.Bitwise
public import LambdaVM.Buses.Alu
public import LambdaVM.Templates.IsBit
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# LT (spec/lt.typ)

One row per `(lhs, rhs, signed, invert)` comparison. Columns follow `prover/src/tables/lt.rs`:
`lhs`/`rhs` as `DWordHHW` limbs (word, half, half), `lhs_sub_rhs` as four halves, the two
sign bits, `lt`, `res = lt xor invert` and the multiplicity `μ`.

The carries of `rhs + lhs_sub_rhs = lhs (mod 2⁶⁴)` are virtual: `carry = (…)·2⁻³²`.
-/

namespace LambdaVM.LT
open Alu (toInt64 ltRaw ltBit dword)

structure Row (F : Type) where
  lhs0 : F
  lhs1 : F
  lhs2 : F
  rhs0 : F
  rhs1 : F
  rhs2 : F
  signed : F
  lt : F
  sub0 : F
  sub1 : F
  sub2 : F
  sub3 : F
  lhsMsb : F
  rhsMsb : F
  invert : F
  res : F
  mu : F
deriving ProvableStruct

/-- The LT chip's arithmetic on naturals: the carries of the limb-wise identity
`rhs + (lhs − rhs mod 2⁶⁴) = lhs + c₁·2⁶⁴` decide `lhs < rhs`, and the top halves decide the signs. -/
theorem core
    {l0 l1 l2 r0 r1 r2 s0 s1 s2 s3 c0 c1 A B : ℕ}
    (hl0 : l0 < 2^32) (hl1 : l1 < 2^16) (hl2 : l2 < 2^16)
    (hr0 : r0 < 2^32) (hr1 : r1 < 2^16) (hr2 : r2 < 2^16)
    (hs0 : s0 < 2^16) (hs1 : s1 < 2^16) (hs2 : s2 < 2^16) (hs3 : s3 < 2^16)
    (hc0 : c0 < 2) (hc1 : c1 < 2)
    (hA : A = l2 / 32768) (hB : B = r2 / 32768)
    (e0 : r0 + (s0 + 65536 * s1) = l0 + c0 * 2^32)
    (e1 : r1 + 65536 * r2 + (s2 + 65536 * s3) + c0 = l1 + 65536 * l2 + c1 * 2^32) :
    let L := l0 + 2^32 * (l1 + 65536 * l2)
    let R := r0 + 2^32 * (r1 + 65536 * r2)
    (c1 = 1 ↔ L < R) ∧ (A = 1 ↔ 2^63 ≤ L) ∧ (B = 1 ↔ 2^63 ≤ R) ∧ A < 2 ∧ B < 2 ∧
    (toInt64 L < toInt64 R ↔ ((A = 1 ∧ B = 0) ∨ (A = B ∧ L < R))) := by
  intro L R
  subst hA hB
  refine ⟨by omega, by omega, by omega, by omega, by omega, ?_⟩
  simp only [toInt64]
  split_ifs <;> omega

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  -- μ : Bit. The spec types it so; the Rust chip does not constrain it (see README).
  isBit r.mu
  -- range: IS_HALF[lhs[1]], IS_HALF[rhs[1]], IS_BIT<signed>, IS_BIT<invert>
  Bitwise.IsHalf.pullIf r.mu r.lhs1
  Bitwise.IsHalf.pullIf r.mu r.rhs1
  isBit r.signed
  isBit r.invert
  -- defs: MSB16[lhs[2]] -> lhs_msb, MSB16[rhs[2]] -> rhs_msb
  Bitwise.Msb16.pullIf r.mu (r.lhs2, r.lhsMsb)
  Bitwise.Msb16.pullIf r.mu (r.rhs2, r.rhsMsb)
  let lhsHi := r.lhs1 + 65536 * r.lhs2
  let rhsHi := r.rhs1 + 65536 * r.rhs2
  let subLo := r.sub0 + 65536 * r.sub1
  let subHi := r.sub2 + 65536 * r.sub3
  let c0 := (r.rhs0 + subLo - r.lhs0) * inv32
  let c1 := (rhsHi + subHi + c0 - lhsHi) * inv32
  -- lt = signed·(A(1−B) + A·C + (1−B)·C) + (1−signed)·C, with C = carry[1]
  assertZero (r.lt - (r.signed * (r.lhsMsb * (1 - r.rhsMsb) + r.lhsMsb * c1 + (1 - r.rhsMsb) * c1)
    + (1 - r.signed) * c1))
  -- res = lt xor invert
  assertZero (r.res + 2 * r.lt * r.invert - r.lt - r.invert)
  -- sub: IS_BIT<carry[i]>, IS_HALF[lhs_sub_rhs[i]]
  isBit c0
  isBit c1
  Bitwise.IsHalf.pullIf r.mu r.sub0
  Bitwise.IsHalf.pullIf r.mu r.sub1
  Bitwise.IsHalf.pullIf r.mu r.sub2
  Bitwise.IsHalf.pullIf r.mu r.sub3
  -- output: ALU[lhs, rhs, opsel(LT) + 32·signed + 64·invert] -> [res, 0]
  Alu.channel.pushIf r.mu
    ⟨r.lhs0, lhsHi, r.rhs0, rhsHi, 4 + 32 * r.signed + 64 * r.invert, r.res, 0⟩

lemma ltRaw_lt_two {s L R : ℕ} : ltRaw s L R < 2 := by
  unfold ltRaw; split_ifs <;> omega

/-- On an honest row the `lt` bit is the chip's polynomial `s·Q(A, B, C) + (1 − s)·C`. -/
lemma ltRaw_eq_poly {s A B k1 L R : ℕ} (hs : s < 2) (hA2 : A < 2) (hB2 : B < 2) (hk : k1 < 2)
    (hC : k1 = 1 ↔ L < R) (hA : A = 1 ↔ 2^63 ≤ L) (hB : B = 1 ↔ 2^63 ≤ R)
    (hS : toInt64 L < toInt64 R ↔ ((A = 1 ∧ B = 0) ∨ (A = B ∧ L < R))) :
    ltRaw s L R = s * (A * (1 - B) + A * k1 + (1 - B) * k1) + (1 - s) * k1 := by
  unfold ltRaw
  rcases (by omega : s = 0 ∨ s = 1) with rfl | rfl <;>
  rcases (by omega : A = 0 ∨ A = 1) with rfl | rfl <;>
  rcases (by omega : B = 0 ∨ B = 1) with rfl | rfl <;>
  rcases (by omega : k1 = 0 ∨ k1 = 1) with rfl | rfl <;>
  norm_num at hC hA hB hS ⊢ <;>
  first
  | (exfalso; omega)
  | (rw [← not_lt, hS]; omega)
  | simp [hS, hC]

lemma ltBit_eq {s i L R : ℕ} (hi : i < 2) :
    ltBit s i L R = ltRaw s L R + i - 2 * ltRaw s L R * i := by
  unfold ltBit
  have := ltRaw_lt_two (s := s) (L := L) (R := R)
  rcases (by omega : ltRaw s L R = 0 ∨ ltRaw s L R = 1) with h | h <;>
  rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl <;> simp [h]

/-- The `lt` constraint, from the natural-number identity. -/
lemma lt_constraint {signed A B k1 lt : F p} {sv Av Bv kv : ℕ}
    (hs : signed = (sv : F p)) (hA : A = (Av : F p)) (hB : B = (Bv : F p)) (hk : k1 = (kv : F p))
    (hs2 : sv < 2) (hB2 : Bv < 2)
    (hlt : lt = ((sv * (Av * (1 - Bv) + Av * kv + (1 - Bv) * kv) + (1 - sv) * kv : ℕ) : F p)) :
    lt - (signed * (A * (1 - B) + A * k1 + (1 - B) * k1) + (1 - signed) * k1) = 0 := by
  subst hs hA hB hk hlt
  push_cast [Nat.cast_sub (by omega : Bv ≤ 1), Nat.cast_sub (by omega : sv ≤ 1)]
  ring

/-- The `res = lt xor invert` constraint, from the natural-number identity. -/
lemma res_constraint {lt invert res : F p} {ltv iv : ℕ}
    (hlt : lt = (ltv : F p)) (hi : invert = (iv : F p)) (hltv : ltv < 2) (hiv : iv < 2)
    (hres : res = ((ltv + iv - 2 * ltv * iv : ℕ) : F p)) :
    res + 2 * lt * invert - lt - invert = 0 := by
  subst hlt hi hres
  rcases (by omega : ltv = 0 ∨ ltv = 1) with rfl | rfl <;>
  rcases (by omega : iv = 0 ∨ iv = 1) with rfl | rfl <;> norm_num

/-- The honest prover's row, as `generate_lt_trace` fills it. -/
def Honest (r : Row (F p)) : Prop :=
  IsBool r.mu ∧ IsBool r.signed ∧ IsBool r.invert ∧
  r.lhs0.val < 2^32 ∧ r.lhs1.val < 2^16 ∧ r.lhs2.val < 2^16 ∧
  r.rhs0.val < 2^32 ∧ r.rhs1.val < 2^16 ∧ r.rhs2.val < 2^16 ∧
  r.sub0.val < 2^16 ∧ r.sub1.val < 2^16 ∧ r.sub2.val < 2^16 ∧ r.sub3.val < 2^16 ∧
  -- lhs_sub_rhs = lhs − rhs (mod 2⁶⁴)
  (r.rhs0.val + 2^32 * (r.rhs1.val + 65536 * r.rhs2.val)
      + (r.sub0.val + 65536 * r.sub1.val + 2^32 * (r.sub2.val + 65536 * r.sub3.val))) % 2^64
    = r.lhs0.val + 2^32 * (r.lhs1.val + 65536 * r.lhs2.val) ∧
  r.lhsMsb.val = r.lhs2.val / 32768 ∧ r.rhsMsb.val = r.rhs2.val / 32768 ∧
  r.lt.val = ltRaw r.signed.val (r.lhs0.val + 2^32 * (r.lhs1.val + 65536 * r.lhs2.val))
    (r.rhs0.val + 2^32 * (r.rhs1.val + 65536 * r.rhs2.val)) ∧
  r.res.val = ltBit r.signed.val r.invert.val (r.lhs0.val + 2^32 * (r.lhs1.val + 65536 * r.lhs2.val))
    (r.rhs0.val + 2^32 * (r.rhs1.val + 65536 * r.rhs2.val))

def circuit : GeneralFormalCircuit (F p) Row unit where
  main
  Spec _ _ _ := True
  channelsWithRequirements := [Alu.channel.toRaw]
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨lhs0, lhs1, lhs2, rhs0, rhs1, rhs2, signed, lt, sub0, sub1, sub2, sub3,
      lhsMsb, rhsMsb, invert, res, mu⟩ := input_var
    simp only [circuit_norm, main, isBit, Alu.channel, Bitwise.isHalfTable, Bitwise.msb16Table]
    constructor
    · intro channel h; tauto
    · intro env h
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      · intro hne1 hne0
        exact (requirement_vacuous' h.1 hne1 hne0).elim

  soundness := by
    circuit_proof_start [main, isBit, Alu.channel, Alu.Guarantees, Bitwise.isHalfTable, Bitwise.msb16Table]
    obtain ⟨h_mu, h_l1, h_r1, h_s, h_i, h_lm, h_rm, h_lt, h_res, h_c0, h_c1, h_s0, h_s1, h_s2, h_s3⟩ :=
      h_holds
    -- μ ∈ {0, 1}; with μ = 0 every requirement is vacuous
    rcases mul_one_sub_eq_zero_iff.mp h_mu with h0 | h1
    · subst h0; simp
    subst h1
    refine ⟨by simp, by simp, by simp, by simp, by simp, by simp, by simp, by simp, ?_⟩
    intro _ _ hl0 hlhi hr0 hrhi
    refine ⟨?_, ?_, ?_⟩
    rotate_left
    · -- the EQ clause does not apply: opsel(LT) + 32·s + 64·i ≠ opsel(EQ) + 64·i'
      intro i hi hflags
      exfalso
      have hsb := mul_one_sub_eq_zero_iff.mp h_s
      have hib := mul_one_sub_eq_zero_iff.mp h_i
      have hs_2 := IsBool.val_lt_two hsb
      have hi_2 := IsBool.val_lt_two hib
      have h1 : ((4 + 32 * input_signed.val + 64 * input_invert.val : ℕ) : F p)
          = ((Alu.opselEQ + 64 * i : ℕ) : F p) := by
        rw [← hflags]; push_cast; simp only [ZMod.natCast_zmod_val]
      have h2 := natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega))
        (lt_p_of_lt_two_pow_40 (by simp only [Alu.opselEQ]; omega)) h1
      simp only [Alu.opselEQ] at h2
      omega
    · -- nor the SHIFT clause: opsel(SHIFT) + w + 32·s + 64·d ≡ 5 or 6 (mod 32)
      intro w s d hw hs hd hflags
      exfalso
      have hsb := mul_one_sub_eq_zero_iff.mp h_s
      have hib := mul_one_sub_eq_zero_iff.mp h_i
      have hs_2 := IsBool.val_lt_two hsb
      have hi_2 := IsBool.val_lt_two hib
      have h1 : ((4 + 32 * input_signed.val + 64 * input_invert.val : ℕ) : F p)
          = ((Alu.opselSHIFT + w + 32 * s + 64 * d : ℕ) : F p) := by
        rw [← hflags]; push_cast; simp only [ZMod.natCast_zmod_val]
      have h2 := natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega))
        (lt_p_of_lt_two_pow_40 (by simp only [Alu.opselSHIFT]; omega)) h1
      simp only [Alu.opselSHIFT] at h2
      omega
    intro s i hs hi hflags
    -- the pulled guarantees are now unconditional
    have hl1 := h_l1 rfl
    have hr1 := h_r1 rfl
    obtain ⟨hl2, hlm⟩ := h_lm rfl
    obtain ⟨hr2, hrm⟩ := h_rm rfl
    have hs0 := h_s0 rfl
    have hs1 := h_s1 rfl
    have hs2 := h_s2 rfl
    have hs3 := h_s3 rfl
    -- the virtual carries
    set c0 := (input_rhs0 + (input_sub0 + 65536 * input_sub1) - input_lhs0) * inv32 with hc0
    set c1 := (input_rhs1 + 65536 * input_rhs2 + (input_sub2 + 65536 * input_sub3) + c0
      - (input_lhs1 + 65536 * input_lhs2)) * inv32 with hc1
    have hc0b : IsBool c0 := mul_one_sub_eq_zero_iff.mp h_c0
    have hc1b : IsBool c1 := mul_one_sub_eq_zero_iff.mp h_c1
    have hsb : IsBool input_signed := mul_one_sub_eq_zero_iff.mp h_s
    have hib : IsBool input_invert := mul_one_sub_eq_zero_iff.mp h_i
    have hc0_2 := IsBool.val_lt_two hc0b
    have hc1_2 := IsBool.val_lt_two hc1b
    have hs_2 := IsBool.val_lt_two hsb
    have hi_2 := IsBool.val_lt_two hib
    -- values of the composite limbs
    have hval_lhsHi : (input_lhs1 + 65536 * input_lhs2).val = input_lhs1.val + 65536 * input_lhs2.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hval_rhsHi : (input_rhs1 + 65536 * input_rhs2).val = input_rhs1.val + 65536 * input_rhs2.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hval_lo : (input_rhs0 + (input_sub0 + 65536 * input_sub1)).val
        = input_rhs0.val + (input_sub0.val + 65536 * input_sub1.val) :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hval_hi : (input_rhs1 + 65536 * input_rhs2 + (input_sub2 + 65536 * input_sub3) + c0).val
        = input_rhs1.val + 65536 * input_rhs2.val + (input_sub2.val + 65536 * input_sub3.val) + c0.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    -- the carry identities, on naturals
    have e0 := carry_lift (by omega) (by omega) hc0b hc0
    rw [hval_lo] at e0
    have e1 := carry_lift (by omega) (by omega) hc1b hc1
    rw [hval_hi, hval_lhsHi] at e1
    obtain ⟨hC, hA, hB, hA2, hB2, hS⟩ :=
      core hl0 hl1 hl2 hr0 hr1 hr2 hs0 hs1 hs2 hs3 hc0_2 hc1_2 hlm hrm e0 e1
    -- decode the flags
    have hsi : input_signed.val = s ∧ input_invert.val = i := by
      have h1 : ((4 + 32 * input_signed.val + 64 * input_invert.val : ℕ) : F p)
          = ((Alu.opselLT + 32 * s + 64 * i : ℕ) : F p) := by
        rw [← hflags]; push_cast; simp only [ZMod.natCast_zmod_val]
      have h2 := natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega))
        (lt_p_of_lt_two_pow_40 (by simp only [Alu.opselLT]; omega)) h1
      simp only [Alu.opselLT] at h2
      omega
    -- lt and res as field identities in the boolean values
    have hlt' : input_lt = input_signed * (input_lhsMsb * (1 - input_rhsMsb) + input_lhsMsb * c1
        + (1 - input_rhsMsb) * c1) + (1 - input_signed) * c1 := by
      rw [← sub_eq_zero]; exact h_lt
    have hres' : input_res = input_lt + input_invert - 2 * input_lt * input_invert := by
      linear_combination h_res
    rw [← ZMod.natCast_zmod_val input_signed, ← ZMod.natCast_zmod_val input_lhsMsb,
      ← ZMod.natCast_zmod_val input_rhsMsb, ← ZMod.natCast_zmod_val c1, hsi.1] at hlt'
    rw [← ZMod.natCast_zmod_val input_invert, hsi.2] at hres'
    simp only [dword, hval_lhsHi, hval_rhsHi]
    rcases (by omega : s = 0 ∨ s = 1) with rfl | rfl <;>
    rcases (by omega : i = 0 ∨ i = 1) with rfl | rfl <;>
    rcases (by omega : input_lhsMsb.val = 0 ∨ input_lhsMsb.val = 1) with hAv | hAv <;>
    rcases (by omega : input_rhsMsb.val = 0 ∨ input_rhsMsb.val = 1) with hBv | hBv <;>
    rcases (by omega : c1.val = 0 ∨ c1.val = 1) with hcv | hcv <;>
    simp only [hAv, hBv, hcv, Nat.cast_zero, Nat.cast_one] at hlt' hA hB hC hS <;>
    norm_num at hlt' hA hB hC hS <;>
    first
    | (exfalso; omega)
    | (subst hlt'; norm_num at hres'; simp [ltBit, ltRaw, hS, hC, hres'])


  ProverAssumptions r _ _ := Honest r

  completeness := by
    circuit_proof_start [main, isBit, Alu.channel, Alu.Guarantees, Bitwise.isHalfTable, Bitwise.msb16Table,
      Honest]
    obtain ⟨hmu, hsb, hib, hl0, hl1, hl2, hr0, hr1, hr2, hs0, hs1, hs2, hs3, hsum, hlm, hrm, hlt, hres⟩ :=
      h_assumptions
    set L := input_lhs0.val + 2^32 * (input_lhs1.val + 65536 * input_lhs2.val) with hL
    set R := input_rhs0.val + 2^32 * (input_rhs1.val + 65536 * input_rhs2.val) with hR
    -- the honest carries
    obtain ⟨k0, hk0, e0⟩ : ∃ k0, k0 < 2 ∧
        input_rhs0.val + (input_sub0.val + 65536 * input_sub1.val) = input_lhs0.val + k0 * 2^32 :=
      ⟨(input_rhs0.val + (input_sub0.val + 65536 * input_sub1.val)) / 2^32, by omega, by omega⟩
    obtain ⟨k1, hk1, e1⟩ : ∃ k1, k1 < 2 ∧
        input_rhs1.val + 65536 * input_rhs2.val + (input_sub2.val + 65536 * input_sub3.val) + k0
          = input_lhs1.val + 65536 * input_lhs2.val + k1 * 2^32 :=
      ⟨(input_rhs1.val + 65536 * input_rhs2.val + (input_sub2.val + 65536 * input_sub3.val) + k0) / 2^32,
        by omega, by omega⟩
    set c0 := (input_rhs0 + (input_sub0 + 65536 * input_sub1) - input_lhs0) * inv32 with hc0
    set c1 := (input_rhs1 + 65536 * input_rhs2 + (input_sub2 + 65536 * input_sub3) + c0
      - (input_lhs1 + 65536 * input_lhs2)) * inv32 with hc1
    have hc0k : c0 = (k0 : F p) := by
      have h := congrArg (fun n : ℕ => (n : F p)) e0
      push_cast at h
      simp only [ZMod.natCast_zmod_val] at h
      have h' : input_rhs0 + (input_sub0 + 65536 * input_sub1) - input_lhs0 = (k0 : F p) * 2^32 := by
        linear_combination h
      rw [hc0, h', mul_assoc, two_pow_mul_inv32, mul_one]
    have hc1k : c1 = (k1 : F p) := by
      have h := congrArg (fun n : ℕ => (n : F p)) e1
      push_cast at h
      simp only [ZMod.natCast_zmod_val] at h
      have h' : input_rhs1 + 65536 * input_rhs2 + (input_sub2 + 65536 * input_sub3) + c0
          - (input_lhs1 + 65536 * input_lhs2) = (k1 : F p) * 2^32 := by
        rw [hc0k]; linear_combination h
      rw [hc1, h', mul_assoc, two_pow_mul_inv32, mul_one]
    have hA2 : input_lhsMsb.val < 2 := by omega
    have hB2 : input_rhsMsb.val < 2 := by omega
    obtain ⟨hC, hA, hB, -, -, hS⟩ :=
      core hl0 hl1 hl2 hr0 hr1 hr2 hs0 hs1 hs2 hs3 hk0 hk1 hlm hrm e0 e1
    have hpoly := ltRaw_eq_poly (IsBool.val_lt_two hsb) hA2 hB2 hk1 hC hA hB hS
    have hlt_raw : input_lt = ((ltRaw input_signed.val L R : ℕ) : F p) := by
      rw [← hlt, ZMod.natCast_zmod_val]
    have hres_cast : input_res = ((ltRaw input_signed.val L R + input_invert.val
        - 2 * ltRaw input_signed.val L R * input_invert.val : ℕ) : F p) := by
      rw [← ltBit_eq (IsBool.val_lt_two hib), ← hres, ZMod.natCast_zmod_val]
    refine ⟨mul_one_sub_eq_zero_iff.mpr hmu, fun _ => by omega, fun _ => by omega,
      mul_one_sub_eq_zero_iff.mpr hsb, mul_one_sub_eq_zero_iff.mpr hib,
      fun _ => ⟨by omega, hlm⟩, fun _ => ⟨by omega, hrm⟩, ?_, ?_, ?_, ?_,
      fun _ => by omega, fun _ => by omega, fun _ => by omega, fun _ => by omega⟩
    · exact lt_constraint (ZMod.natCast_zmod_val input_signed).symm
        (ZMod.natCast_zmod_val input_lhsMsb).symm (ZMod.natCast_zmod_val input_rhsMsb).symm hc1k
        (IsBool.val_lt_two hsb) hB2 (by rw [hlt_raw, hpoly])
    · exact res_constraint hlt_raw (ZMod.natCast_zmod_val input_invert).symm ltRaw_lt_two
        (IsBool.val_lt_two hib) hres_cast
    · rw [hc0k]; rcases (by omega : k0 = 0 ∨ k0 = 1) with rfl | rfl <;> simp
    · rw [hc1k]; rcases (by omega : k1 = 0 ∨ k1 = 1) with rfl | rfl <;> simp


end LambdaVM.LT
