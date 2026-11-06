module

public import LambdaVM.Buses.Alu
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.Linarith
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# The arithmetic of the DVRM chip, on naturals and integers

`spec/dvrm.typ` computes the four RISC-V truncated-division instructions `DIV`, `DIVU`, `REM`
and `REMU` from a row holding the operands `n`, `d`, the quotient `q`, the remainder `r`, the
absolute values `abs_r`, `abs_d`, the difference `n_sub_r`, and the bits `signed`,
`div_by_zero`, `overflow`, `sign_n`, `sign_d`, `sign_q`, `sign_r`, `sign_n_sub_r`. The chip
never divides: it witnesses `q` and `r` and checks them with

* `d · q = n - r`, through the `MUL` bus on the 128-bit product of the sign-extended operands,
* `|r| < |d|`, through the `LT` bus on `abs_r` and `abs_d`, unless `d = 0`,
* `r` of the sign of `n`, as `sign_r = sign_n` whenever `r ≠ 0`,
* `q = 2^64 - 1` and `r = n` when `d = 0`.

Those four facts pin `q` and `r` down, because truncated division is the unique quotient and
remainder with that sign convention. Everything here is pure `ℕ`/`ℤ` arithmetic: no circuit, no
interaction, no constraint. The chip reads the file in two directions.

## (a) Soundness: from what a row gives, to `Alu.divremResult`

* `tdiv_unique` — a quotient and remainder with `D·Q = N - R`, `|R| < |D|` and `R` of the sign
  of `N` *are* `N.tdiv D` and `N.tmod D`.
* `ext128_eq` — `Alu.ext128` as the value plus a sign bit times `2^128 - 2^64`.
* `sgn64_eq` — `Alu.sgn64` as the value minus a sign bit times `2^64`.
* `prod_bound` — two 64-bit values read signed or unsigned have a product below `(2^64-1)^2`,
  so the 128-bit truncation the `MUL` bus returns loses nothing.
* `dvrm_core` — the row's limb bounds, sign bits, `MUL` push, `LT` push and `div_by_zero` flag
  give `q = Alu.divremResult signed 0 n d` and `r = Alu.divremResult signed 1 n d`.

## (b) Completeness: from an honest row, to the facts the constraints need

* `sgn64_emod` / `sgn64_zero` — reading back the 64-bit residue of a value in range.
* `tdiv_range` — `tdiv` stays in the signed range except for `-2^63 / -1`.
* `dvrm_honest` — the honest `q` and `r`, over the integers: `d·q = n - r` exactly, `|r| < |d|`,
  `r` of the sign of `n`, and `n - r` within 65 bits.
* `ext128_int` — `Alu.ext128` is the signed value plus a multiple of `2^128`.
* `honest_product` — the honest `n_sub_r` with its sign extension is the 128-bit product.
* `add128_limbs` — the honest 128-bit addition, limb by limb, with each carry a bit.
* `honest_nat` — every arithmetic fact about an honest row, over the naturals: the witness
  values the chip must fill in, and the hypotheses `dvrm_core` asks for.
-/

namespace LambdaVM.DvrmCore
open Alu (ext128 sgn64 toSigned64 divremResult)

/-! ## Soundness: reading a row back as truncated division -/

/-- **Truncated division is unique.** -/
theorem tdiv_unique {N D Q R : ℤ} (hD : D ≠ 0) (h : D * Q = N - R) (hR : R.natAbs < D.natAbs)
    (hpos : 0 ≤ N → 0 ≤ R) (hneg : N < 0 → R ≤ 0) : Q = N.tdiv D ∧ R = N.tmod D := by
  have h' := Int.tmod_add_tdiv_mul N D
  set R' := N.tmod D
  set Q' := N.tdiv D
  have hR' : R'.natAbs < D.natAbs := by
    rw [Int.natAbs_tmod]; exact Nat.mod_lt _ (Int.natAbs_pos.mpr hD)
  have hpos' : 0 ≤ N → 0 ≤ R' := fun hN => Int.tmod_nonneg _ hN
  have hneg' : N < 0 → R' ≤ 0 := by
    intro hN
    have : R' = -((-N).tmod D) := by simp [R', Int.neg_tmod]
    rw [this]
    have := Int.tmod_nonneg D (show 0 ≤ -N by omega)
    omega
  -- `R' - R` is a multiple of `D` smaller than `|D|`
  have hdvd : D ∣ R' - R := ⟨Q - Q', by linarith⟩
  have hsmall : (R' - R).natAbs < D.natAbs := by
    rcases (by omega : 0 ≤ N ∨ N < 0) with hN | hN
    · have := hpos hN; have := hpos' hN; omega
    · have := hneg hN; have := hneg' hN; omega
  have hzero : R' - R = 0 := by
    rcases hdvd with ⟨k, hk⟩
    rw [hk, Int.natAbs_mul] at hsmall
    have : k.natAbs = 0 := by
      by_contra hk0
      have := Nat.le_mul_of_pos_right D.natAbs (Nat.pos_of_ne_zero hk0)
      omega
    rw [hk, Int.natAbs_eq_zero.mp this, mul_zero]
  have hRR : R = R' := by omega
  refine ⟨?_, hRR⟩
  have : D * (Q - Q') = 0 := by rw [mul_sub]; linarith
  rcases mul_eq_zero.mp this with h0 | h0
  · exact absurd h0 hD
  · linarith

/-- `Alu.ext128` as a sign bit times the extension `2^128 - 2^64`. -/
theorem ext128_eq (sg x : ℕ) :
    ext128 sg x = x + (if sg = 1 ∧ 2^63 ≤ x then 1 else 0) * (2^128 - 2^64) := by
  unfold ext128
  split_ifs <;> omega

/-- `Alu.sgn64` is `x - 2^64·(sign bit)`. -/
theorem sgn64_eq (sg x : ℕ) :
    sgn64 sg x = (x : ℤ) - (if sg = 1 ∧ 2^63 ≤ x then 1 else 0) * 2^64 := by
  unfold sgn64 toSigned64
  by_cases h1 : sg = 1
  · by_cases h2 : x < 2^63
    · rw [ite_eq_left h1, ite_eq_left h2, ite_eq_right (by omega)]; push_cast; ring
    · rw [ite_eq_left h1, ite_eq_right h2, ite_eq_left ⟨h1, by omega⟩]; push_cast; ring
  · rw [ite_eq_right h1, ite_eq_right (by omega)]; push_cast; ring

/-- Two 64-bit values read as signed or unsigned: their product stays below `2^128 - 2^64`. -/
theorem prod_bound {D Q : ℤ} (hD0 : -2^63 ≤ D) (hD1 : D ≤ 2^64 - 1) (hQ0 : -2^63 ≤ Q)
    (hQ1 : Q ≤ 2^64 - 1) : -(2^64 - 1)^2 ≤ D * Q ∧ D * Q ≤ (2^64 - 1)^2 := by
  have h1 := mul_nonneg (sub_nonneg.2 hD1) (sub_nonneg.2 hQ1)
  have h2 := mul_nonneg (sub_nonneg.2 hD0) (sub_nonneg.2 hQ0)
  have h3 := mul_nonneg (sub_nonneg.2 hD1) (sub_nonneg.2 hQ0)
  have h4 := mul_nonneg (sub_nonneg.2 hD0) (sub_nonneg.2 hQ1)
  constructor
  · rcases (by omega : 0 ≤ D + Q ∨ D + Q < 0) with h | h <;> nlinarith
  · nlinarith

/-- **`DVRM` over the naturals.** The chip's constraints, read as facts about the values of its
columns, give `q` and `r` as `Alu.divremResult`. `sq` is whether `MUL` reads `q` as signed: any
bit works, which is why the `overflow` column is only needed for completeness. Neither `sq` nor
`sign_n_sub_r` (`s`) even needs to be a bit here. -/
theorem dvrm_core {n d q r nsr s signed sq dbz : ℕ}
    (hn : n < 2^64) (hd : d < 2^64) (hq : q < 2^64) (hr : r < 2^64) (hnsr : nsr < 2^64)
    (hdbz : dbz = if d = 0 then 1 else 0)
    (hq0 : dbz = 1 → q = 2^64 - 1)
    (hrs : r ≠ 0 → (signed = 1 ∧ 2^63 ≤ r ↔ signed = 1 ∧ 2^63 ≤ n))
    (hlt : dbz = 0 → (if signed = 1 ∧ 2^63 ≤ r then (2^64 - r) % 2^64 else r)
      < (if signed = 1 ∧ 2^63 ≤ d then (2^64 - d) % 2^64 else d))
    (hadd : ∃ c, nsr + s * (2^128 - 2^64) + ext128 signed r
      = ext128 signed n + c * 2^128)
    (hmul : nsr + 2^64 * ((2^64 - 1) * s)
      = (ext128 signed d * ext128 sq q) % 2^128) :
    q = divremResult signed 0 n d ∧ r = divremResult signed 1 n d := by
  -- the signed readings
  set sn := if signed = 1 ∧ 2^63 ≤ n then 1 else 0 with hsn
  set sd := if signed = 1 ∧ 2^63 ≤ d then 1 else 0 with hsd
  set sr := if signed = 1 ∧ 2^63 ≤ r then 1 else 0 with hsr
  set sq' := if sq = 1 ∧ 2^63 ≤ q then 1 else 0 with hsq'
  have hsnb : sn ≤ 1 ∧ (sn = 1 → 2^63 ≤ n) := by
    by_cases h : signed = 1 ∧ 2^63 ≤ n
    · simp only [hsn, ite_eq_left h]; omega
    · simp only [hsn, ite_eq_right h]; omega
  have hsdb : sd ≤ 1 ∧ (sd = 1 → 2^63 ≤ d) := by
    by_cases h : signed = 1 ∧ 2^63 ≤ d
    · simp only [hsd, ite_eq_left h]; omega
    · simp only [hsd, ite_eq_right h]; omega
  have hsrb : sr ≤ 1 ∧ (sr = 1 → 2^63 ≤ r) := by
    by_cases h : signed = 1 ∧ 2^63 ≤ r
    · simp only [hsr, ite_eq_left h]; omega
    · simp only [hsr, ite_eq_right h]; omega
  have hsqb : sq' ≤ 1 ∧ (sq' = 1 → 2^63 ≤ q) := by
    by_cases h : sq = 1 ∧ 2^63 ≤ q
    · simp only [hsq', ite_eq_left h]; omega
    · simp only [hsq', ite_eq_right h]; omega
  set N : ℤ := (n : ℤ) - sn * 2^64 with hN
  set D : ℤ := (d : ℤ) - sd * 2^64 with hD
  set Q : ℤ := (q : ℤ) - sq' * 2^64 with hQ
  set R : ℤ := (r : ℤ) - sr * 2^64 with hR
  clear_value sn sd sr sq' N D Q R
  -- `n - r` as a 128-bit addition: exact, by size
  obtain ⟨c, hc⟩ := hadd
  rw [ext128_eq signed r, ext128_eq signed n, ← hsr, ← hsn] at hc
  have hsub : (nsr : ℤ) - s * 2^64 + R = N := by
    have hc' : ((nsr + s * (2^128 - 2^64) + (r + sr * (2^128 - 2^64)) : ℕ) : ℤ)
        = ((n + sn * (2^128 - 2^64) + c * 2^128 : ℕ) : ℤ) := by rw [hc]
    push_cast [Nat.cast_sub (show 2^64 ≤ 2^128 by norm_num)] at hc'
    have : c ≤ 2 := by omega
    rw [hN, hR]
    omega
  -- `d·q` as a 128-bit product: exact, by size
  rw [ext128_eq signed d, ext128_eq sq q, ← hsd, ← hsq'] at hmul
  have hprod : D * Q = (nsr : ℤ) - s * 2^64 := by
    have hk := Nat.mod_add_div ((d + sd * (2^128 - 2^64)) * (q + sq' * (2^128 - 2^64))) (2^128)
    rw [← hmul] at hk
    set k := (d + sd * (2^128 - 2^64)) * (q + sq' * (2^128 - 2^64)) / 2^128
    have hk' : (((nsr + 2^64 * ((2^64 - 1) * s) + 2^128 * k) : ℕ) : ℤ)
        = (((d + sd * (2^128 - 2^64)) * (q + sq' * (2^128 - 2^64)) : ℕ) : ℤ) := by rw [hk]
    push_cast [Nat.cast_sub (show 2^64 ≤ 2^128 by norm_num),
      Nat.cast_sub (show 1 ≤ 2^64 by norm_num)] at hk'
    have hb := prod_bound (D := D) (Q := Q) (by rw [hD]; omega) (by rw [hD]; omega)
      (by rw [hQ]; omega) (by rw [hQ]; omega)
    -- the product of the extended values differs from `D·Q` by a multiple of `2^128`
    have key : D * Q - ((nsr : ℤ) - s * 2^64)
        = 2^128 * ((s : ℤ) + k - (sd * Q + sq' * D + sd * sq' * 2^128)) := by
      rw [hD, hQ]; linear_combination -hk'
    generalize (s : ℤ) + k - (sd * Q + sq' * D + sd * sq' * 2^128) = M at key
    generalize D * Q = P at hb key ⊢
    omega
  have heq : D * Q = N - R := by rw [hprod]; linarith
  clear hmul hc hprod hsub
  -- the readings of the inputs as `sgn64`
  have hsgn_n : sgn64 signed n = N := by rw [sgn64_eq signed n, hN, hsn]; push_cast; ring
  have hsgn_d : sgn64 signed d = D := by rw [sgn64_eq signed d, hD, hsd]; push_cast; ring
  by_cases hd0 : d = 0
  · -- division by zero: `q = 2^64 - 1`, `r = n`
    have hdz : dbz = 1 := by rw [hdbz, ite_eq_left hd0]
    have hsd0 : sd = 0 := by rw [hsd, ite_eq_right (by omega)]
    have hD0 : D = 0 := by rw [hD, hd0, hsd0]; simp
    rw [hD0, zero_mul] at heq
    simp only [divremResult, hd0, ite_true]
    refine ⟨hq0 hdz, ?_⟩
    rw [hN, hR] at heq
    clear hlt hrs hq0 hdbz hsgn_n hsgn_d hsqb
    omega
  · have hdz : dbz = 0 := by rw [hdbz, ite_eq_right hd0]
    have hlt' := hlt hdz
    have habs : R.natAbs < D.natAbs := by
      have hRa : R.natAbs = if signed = 1 ∧ 2^63 ≤ r then (2^64 - r) % 2^64 else r := by
        clear hlt hlt' hrs hq0 hdbz hsgn_n hsgn_d heq hsn hsd hsq'
        rw [hR, hsr]
        by_cases h : signed = 1 ∧ 2^63 ≤ r
        · simp only [ite_eq_left h]; omega
        · simp only [ite_eq_right h]; omega
      have hDa : D.natAbs = if signed = 1 ∧ 2^63 ≤ d then (2^64 - d) % 2^64 else d := by
        clear hlt hlt' hrs hq0 hdbz hsgn_n hsgn_d heq hsn hsr hsq'
        rw [hD, hsd]
        by_cases h : signed = 1 ∧ 2^63 ≤ d
        · simp only [ite_eq_left h]; omega
        · simp only [ite_eq_right h]; omega
      rw [hRa, hDa]; exact hlt'
    have hDne : D ≠ 0 := by
      clear hlt hlt' hrs hq0 hdbz hsgn_n hsgn_d heq habs hsn hsd hsr hsq'
      rw [hD]; omega
    -- `r` has the sign of `n`
    have hsrsn : r ≠ 0 → sr = sn := by
      intro hr0
      have h := hrs hr0
      by_cases h1 : signed = 1 ∧ 2^63 ≤ n
      · simp only [hsr, hsn, ite_eq_left h1, ite_eq_left (h.mpr h1)]
      · simp only [hsr, hsn, ite_eq_right h1, ite_eq_right (fun h' => h1 (h.mp h'))]
    have hr0s : r = 0 → sr = 0 := fun h => by rw [hsr, ite_eq_right (by omega)]
    have hpos : 0 ≤ N → 0 ≤ R := by
      clear hlt hlt' hrs hq0 hdbz hsgn_n hsgn_d heq habs hDne hsd hsq' hsn hsr
      intro hN0
      rw [hN] at hN0; rw [hR]
      by_cases hr0 : r = 0
      · rw [hr0s hr0]; omega
      · rw [hsrsn hr0]; omega
    have hneg : N < 0 → R ≤ 0 := by
      clear hlt hlt' hrs hq0 hdbz hsgn_n hsgn_d heq habs hpos hDne hsd hsq' hsn hsr
      intro hN0
      rw [hN] at hN0; rw [hR]
      by_cases hr0 : r = 0
      · rw [hr0s hr0, hr0]; omega
      · rw [hsrsn hr0]; omega
    clear hlt hrs
    have hqmod : (Q % 2^64).toNat = q := by
      rw [hQ, show (q : ℤ) - sq' * 2^64 = q + 2^64 * (-(sq' : ℤ)) by ring,
        Int.add_mul_emod_self_left, Int.emod_eq_of_lt (by positivity) (by exact_mod_cast hq)]
      simp
    have hrmod : (R % 2^64).toNat = r := by
      rw [hR, show (r : ℤ) - sr * 2^64 = r + 2^64 * (-(sr : ℤ)) by ring,
        Int.add_mul_emod_self_left, Int.emod_eq_of_lt (by positivity) (by exact_mod_cast hr)]
      simp
    obtain ⟨hQq, hRr⟩ := tdiv_unique hDne heq habs hpos hneg
    simp only [divremResult, hd0, ite_false, zero_ne_one, ite_true, hsgn_n, hsgn_d]
    rw [← hQq, ← hRr, hqmod, hrmod]
    exact ⟨rfl, rfl⟩

/-! ## Completeness: the honest quotient and remainder -/

/-- A signed reading of the 64-bit residue of an integer in `[-2^63, 2^63)` gives it back. -/
theorem sgn64_emod {x : ℤ} (h0 : -2^63 ≤ x) (h1 : x < 2^63) :
    sgn64 1 ((x % 2^64).toNat) = x := by
  have hm : (x % 2^64).toNat < 2^64 := by omega
  rw [sgn64_eq 1 _]
  by_cases hx : 0 ≤ x
  · rw [ite_eq_right (by omega)]; omega
  · rw [ite_eq_left ⟨rfl, by omega⟩]; omega

/-- An unsigned reading of a natural below `2^64`. -/
theorem sgn64_zero {x : ℕ} : sgn64 0 x = x := by
  simp [sgn64]

/-- `tdiv` stays in the signed range except for `-2^63 / -1`. -/
theorem tdiv_range {N D : ℤ} (hN0 : -2^63 ≤ N) (hN1 : N < 2^63) (hD0 : -2^63 ≤ D) (hD1 : D < 2^63)
    (hD : D ≠ 0) (hov : ¬ (N = -2^63 ∧ D = -1)) : -2^63 ≤ N.tdiv D ∧ N.tdiv D < 2^63 := by
  have h := Int.tmod_add_tdiv_mul N D
  have hr : (N.tmod D).natAbs < D.natAbs := by
    rw [Int.natAbs_tmod]; exact Nat.mod_lt _ (Int.natAbs_pos.mpr hD)
  have e : (N.tdiv D).natAbs = N.natAbs / D.natAbs := by rw [Int.natAbs_tdiv]; rfl
  have habs : (N.tdiv D).natAbs ≤ N.natAbs := by rw [e]; exact Nat.div_le_self _ _
  constructor
  · by_contra hlt
    have : (N.tdiv D).natAbs > 2^63 := by omega
    omega
  · by_contra hge
    have h2 : (N.tdiv D).natAbs ≥ 2^63 := by omega
    have hNabs : N.natAbs = 2^63 := by omega
    have hN : N = -2^63 := by omega
    -- `|N / D| = |N|` forces `|D| = 1`, and `D = 1` gives `-2^63`
    have hD1' : D.natAbs = 1 := by
      rw [e] at h2
      by_contra hne
      have : 2 ≤ D.natAbs := by have := Int.natAbs_pos.mpr hD; omega
      have := Nat.div_le_div_left (a := N.natAbs) this (by norm_num : 0 < 2)
      omega
    rcases Int.natAbs_eq D with hD' | hD'
    · rw [hD1'] at hD'; subst hD'; rw [hN] at hge; simp at hge
    · rw [hD1'] at hD'; exact hov ⟨hN, by omega⟩

/-- **The honest `DVRM` row, over the integers.** With `q`, `r` the RISC-V quotient and remainder,
and `q` read signed or not as `MUL` does (`sign_q = signed · (1 - overflow)`): `d·q = n - r`
exactly, `|r| < |d|` unless `d = 0`, `r` has the sign of `n`, and `n - r` fits in 65 bits. -/
theorem dvrm_honest {n d s : ℕ} (hn : n < 2^64) (hd : d < 2^64) (hs : s < 2) :
    let q := divremResult s 0 n d
    let r := divremResult s 1 n d
    let sq := s * (1 - (if s = 1 ∧ n = 2^63 ∧ d = 2^64 - 1 then 1 else 0))
    q < 2^64 ∧ r < 2^64 ∧
      sgn64 s d * sgn64 sq q = sgn64 s n - sgn64 s r ∧
      (d ≠ 0 → (sgn64 s r).natAbs < (sgn64 s d).natAbs) ∧
      (r ≠ 0 → (s = 1 ∧ 2^63 ≤ r ↔ s = 1 ∧ 2^63 ≤ n)) ∧
      (d = 0 → q = 2^64 - 1 ∧ r = n) ∧
      -2^64 < sgn64 s n - sgn64 s r ∧
      sgn64 s n - sgn64 s r < 2^64 := by
  intro q r sq
  by_cases hd0 : d = 0
  · -- division by zero
    have hq : q = 2^64 - 1 := by simp [q, divremResult, hd0]
    have hr : r = n := by simp [r, divremResult, hd0]
    have hD : sgn64 s d = 0 := by rw [hd0]; simp [sgn64, toSigned64]
    rw [hD, zero_mul, hr, sub_self]
    exact ⟨by omega, hn, rfl, fun h => absurd hd0 h, fun _ => Iff.rfl, fun _ => ⟨hq, rfl⟩,
      by norm_num, by norm_num⟩
  rcases (by omega : s = 0 ∨ s = 1) with rfl | rfl
  · -- unsigned: `n / d` and `n % d`
    have hq : q = n / d := by
      simp only [q, divremResult, ite_eq_right hd0, zero_ne_one, ite_false, sgn64_zero]
      rw [← Int.ofNat_tdiv]
      have h1 : ((n / d : ℕ) : ℤ) % 2^64 = ((n / d : ℕ) : ℤ) :=
        Int.emod_eq_of_lt (by positivity) (by have := Nat.div_le_self n d; omega)
      rw [h1, Int.toNat_natCast]
    have hr : r = n % d := by
      simp only [r, divremResult, ite_eq_right hd0, ite_true, sgn64_zero]
      rw [← Int.ofNat_tmod]; have := Nat.mod_le n d; omega
    have hsq : sq = 0 := by simp [sq]
    rw [hsq, sgn64_zero, sgn64_zero, sgn64_zero, sgn64_zero, hq, hr]
    have hdm := Nat.div_add_mod n d
    have hml := Nat.mod_lt n (Nat.pos_of_ne_zero hd0)
    have hdl : n / d < 2^64 := lt_of_le_of_lt (Nat.div_le_self _ _) hn
    refine ⟨hdl, by omega, ?_, ?_, fun _ => by simp, fun h => absurd h hd0, by omega, by omega⟩
    · push_cast; have := Int.emod_add_ediv_mul (n : ℤ) d; linarith
    · intro _; simp only [Int.natAbs_natCast]; exact hml
  · -- signed
    set N := sgn64 1 n with hN
    set D := sgn64 1 d with hD
    have hN' : N = (n : ℤ) - (if 2^63 ≤ n then 1 else 0) * 2^64 := by
      rw [hN, sgn64_eq]; simp
    have hD' : D = (d : ℤ) - (if 2^63 ≤ d then 1 else 0) * 2^64 := by
      rw [hD, sgn64_eq]; simp
    have hNr : -2^63 ≤ N ∧ N < 2^63 := by rw [hN']; split_ifs <;> constructor <;> omega
    have hDr : -2^63 ≤ D ∧ D < 2^63 := by rw [hD']; split_ifs <;> constructor <;> omega
    have hDne : D ≠ 0 := by rw [hD']; split_ifs <;> omega
    have hq : q = ((N.tdiv D) % 2^64).toNat := by
      simp only [q, divremResult, ite_eq_right hd0, zero_ne_one, ite_false, hN, hD]
    have hr : r = ((N.tmod D) % 2^64).toNat := by
      simp only [r, divremResult, ite_eq_right hd0, ite_true, hN, hD]
    have hmod := Int.tmod_add_tdiv_mul N D
    have hrabs : (N.tmod D).natAbs < D.natAbs := by
      rw [Int.natAbs_tmod]; exact Nat.mod_lt _ (Int.natAbs_pos.mpr hDne)
    have hRr : -2^63 ≤ N.tmod D ∧ N.tmod D < 2^63 := by constructor <;> omega
    have hR : sgn64 1 r = N.tmod D := by rw [hr]; exact sgn64_emod hRr.1 hRr.2
    -- the remainder has the sign of `n`
    have hsign : (0 ≤ N → 0 ≤ N.tmod D) ∧ (N < 0 → N.tmod D ≤ 0) := by
      refine ⟨fun h => Int.tmod_nonneg _ h, fun h => ?_⟩
      have : N.tmod D = -((-N).tmod D) := by simp [Int.neg_tmod]
      rw [this]; have := Int.tmod_nonneg D (show 0 ≤ -N by omega); omega
    have hrb : r < 2^64 := by rw [hr]; omega
    have hNneg : (2^63 ≤ n ↔ N < 0) := by rw [hN']; split_ifs <;> constructor <;> intro <;> omega
    have hRneg : (2^63 ≤ r ↔ N.tmod D < 0) := by rw [hr]; constructor <;> intro <;> omega
    have hr0 : (r = 0 ↔ N.tmod D = 0) := by rw [hr]; constructor <;> intro <;> omega
    have hcore : (r ≠ 0 → ((1 = 1 ∧ 2^63 ≤ r) ↔ (1 = 1 ∧ 2^63 ≤ n))) := by
      intro hr0'
      have : N.tmod D ≠ 0 := fun h => hr0' (hr0.mpr h)
      simp only [true_and]
      rw [hRneg, hNneg]
      constructor
      · intro h; by_contra h'; have := hsign.1 (by omega); omega
      · intro h; have := hsign.2 h; omega
    by_cases hov : n = 2^63 ∧ d = 2^64 - 1
    · -- the overflow `-2^63 / -1`: `q = 2^63`, read unsigned
      have hNv : N = -2^63 := by rw [hN', hov.1]; norm_num
      have hDv : D = -1 := by rw [hD', hov.2]; norm_num
      have hsq : sq = 0 := by simp [sq, hov]
      have htd : N.tdiv D = 2^63 := by rw [hNv, hDv, Int.neg_tdiv_neg, Int.tdiv_one]
      have htm : N.tmod D = 0 := by rw [hNv, hDv, Int.tmod_neg, Int.tmod_one]
      have hqv : q = 2^63 := by rw [hq, htd]; rfl
      rw [hsq, sgn64_zero, hR, htm, hqv, hDv, hNv]
      refine ⟨by norm_num, hrb, by norm_num, fun _ => by simp, hcore,
        fun h => absurd h hd0, by norm_num, by norm_num⟩
    · have hsq : sq = 1 := by simp only [sq]; rw [ite_eq_right (fun h => hov ⟨by omega, by omega⟩)]
      have hNDov : ¬ (N = -2^63 ∧ D = -1) := by
        rintro ⟨h1, h2⟩
        apply hov
        rw [hN'] at h1; rw [hD'] at h2
        constructor
        · split_ifs at h1 <;> omega
        · split_ifs at h2 <;> omega
      have hT := tdiv_range hNr.1 hNr.2 hDr.1 hDr.2 hDne hNDov
      have hQ : sgn64 1 q = N.tdiv D := by rw [hq]; exact sgn64_emod hT.1 hT.2
      rw [hsq, hQ, hR]
      refine ⟨by rw [hq]; omega, hrb, by linarith, fun _ => hrabs, hcore, fun h => absurd h hd0,
        by omega, by omega⟩

/-- `Alu.ext128` reads a 64-bit value as its signed value plus a multiple of `2^128`. -/
theorem ext128_int (sg x : ℕ) (_hx : x < 2^64) (_hsg : sg < 2) :
    ((ext128 sg x : ℕ) : ℤ)
      = sgn64 sg x + (if sg = 1 ∧ 2^63 ≤ x then 1 else 0) * 2^128 := by
  rw [ext128_eq, sgn64_eq]
  split_ifs <;> push_cast <;> omega

/-- **The honest `n - r` against the 128-bit product.** If `d·q = n - r` (signed readings) and
`m = n - r` fits in 65 bits, then `m mod 2^64` plus the extension `(2^64 - 1)·[m < 0]` in the high
half is the 128-bit product `MUL` returns. -/
theorem honest_product {n d q r s sq : ℕ} (hd : d < 2^64) (hq : q < 2^64) (hs : s < 2) (hsq : sq < 2)
    (heq : sgn64 s d * sgn64 sq q = sgn64 s n - sgn64 s r)
    (hlo : -2^64 < sgn64 s n - sgn64 s r)
    (hhi : sgn64 s n - sgn64 s r < 2^64) :
    let m := sgn64 s n - sgn64 s r
    (m % 2^64).toNat + 2^64 * ((2^64 - 1) * (if m < 0 then 1 else 0))
      = (ext128 s d * ext128 sq q) % 2^128 := by
  intro m
  have ed := ext128_int s d hd hs
  have eq := ext128_int sq q hq hsq
  set a := (if s = 1 ∧ 2^63 ≤ d then 1 else 0 : ℤ)
  set b := (if sq = 1 ∧ 2^63 ≤ q then 1 else 0 : ℤ)
  have hprod : ((ext128 s d * ext128 sq q : ℕ) : ℤ)
      = m + 2^128 * (a * sgn64 sq q + b * sgn64 s d + a * b * 2^128) := by
    push_cast; rw [ed, eq]; simp only [m, ← heq]; ring
  have key : ((((m % 2^64).toNat + 2^64 * ((2^64 - 1) * (if m < 0 then 1 else 0)) : ℕ)) : ℤ)
      = (((ext128 s d * ext128 sq q % 2^128 : ℕ)) : ℤ) := by
    rw [Int.natCast_mod, hprod]
    push_cast
    rw [Int.add_mul_emod_self_left]
    split_ifs with hm <;> push_cast <;> omega
  exact_mod_cast key

/-- **The honest 128-bit subtraction, limb by limb.** If `a + b ≡ c (mod 2^128)` for numbers
written in four 32-bit limbs each, every limb of `c` is the sum of the limbs of `a` and `b` plus
the carry, and every carry is a bit. -/
theorem add128_limbs {a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 : ℕ} (ha0 : a0 < 2^32) (ha1 : a1 < 2^32)
    (ha2 : a2 < 2^32) (ha3 : a3 < 2^32) (hb0 : b0 < 2^32) (hb1 : b1 < 2^32) (hb2 : b2 < 2^32)
    (hb3 : b3 < 2^32) (hc0 : c0 < 2^32) (hc1 : c1 < 2^32) (hc2 : c2 < 2^32) (hc3 : c3 < 2^32)
    (h : ∃ k, a0 + 2^32 * a1 + 2^64 * a2 + 2^96 * a3 + (b0 + 2^32 * b1 + 2^64 * b2 + 2^96 * b3)
      = c0 + 2^32 * c1 + 2^64 * c2 + 2^96 * c3 + k * 2^128) :
    let k0 := (a0 + b0) / 2^32
    let k1 := (a1 + b1 + k0) / 2^32
    let k2 := (a2 + b2 + k1) / 2^32
    c0 = (a0 + b0) % 2^32 ∧ c1 = (a1 + b1 + k0) % 2^32 ∧ c2 = (a2 + b2 + k1) % 2^32 ∧
      c3 = (a3 + b3 + k2) % 2^32 ∧ k0 < 2 ∧ k1 < 2 ∧ k2 < 2 ∧ (a3 + b3 + k2) / 2^32 < 2 := by
  intro k0 k1 k2
  obtain ⟨k, hk⟩ := h
  omega

/-- **Every arithmetic fact about an honest `DVRM` row**, over the naturals. -/
theorem honest_nat {n d s : ℕ} (hn : n < 2^64) (hd : d < 2^64) (hs : s < 2) :
    let q := divremResult s 0 n d
    let r := divremResult s 1 n d
    let ov := if s = 1 ∧ n = 2^63 ∧ d = 2^64 - 1 then 1 else 0
    let sq := s * (1 - ov)
    let M := sgn64 s n - sgn64 s r
    let nsr := (M % 2^64).toNat
    let sm := if M < 0 then 1 else 0
    let sn := if s = 1 ∧ 2^63 ≤ n then 1 else 0
    let sd := if s = 1 ∧ 2^63 ≤ d then 1 else 0
    let sr := if s = 1 ∧ 2^63 ≤ r then 1 else 0
    let ar := (sgn64 s r).natAbs
    let ad := (sgn64 s d).natAbs
    let dz := if d = 0 then 1 else 0
    q < 2^64 ∧ r < 2^64 ∧ nsr < 2^64 ∧ sm ≤ 1 ∧ ar < 2^64 ∧ ad < 2^64 ∧ sq ≤ 1 ∧
      (dz = 1 → q = 2^64 - 1) ∧ (r ≠ 0 → sr = sn) ∧
      (sr = 1 → ar = (2^64 - r) % 2^64) ∧ (sr = 0 → ar = r) ∧
      (sd = 1 → ad = (2^64 - d) % 2^64) ∧ (sd = 0 → ad = d) ∧
      (if ar < ad then 1 else 0) = 1 - dz ∧
      nsr + 2^64 * ((2^64 - 1) * sm) = (ext128 s d * ext128 sq q) % 2^128 ∧
      (∃ k, nsr + sm * (2^128 - 2^64) + ext128 s r = ext128 s n + k * 2^128) := by
  intro q r ov sq M nsr sm sn sd sr ar ad dz
  obtain ⟨hq, hr, heq, hlt, hsign, hz, hlo, hhi⟩ := dvrm_honest (n := n) (d := d) (s := s) hn hd hs
  have hsq : sq < 2 := by simp only [sq, ov]; split_ifs <;> omega
  have hsv := fun x (hx : x < 2^64) => sgn64_eq s x
  have eN := hsv n hn
  have eD := hsv d hd
  have eR := hsv r hr
  have hprod := honest_product (n := n) (r := r) hd hq hs hsq heq hlo hhi
  refine ⟨hq, hr, by simp only [nsr]; omega, by simp only [sm]; split_ifs <;> omega, ?_, ?_,
    by omega, ?_, ?_, ?_, ?_, ?_, ?_, ?_, hprod, ?_⟩
  · simp only [ar]; rw [eR]; split_ifs <;> omega
  · simp only [ad]; rw [eD]; split_ifs <;> omega
  · intro h; exact (hz (by simp only [dz] at h; split_ifs at h with h'; omega)).1
  · intro h; simp only [sr, sn]; rw [if_congr (hsign h) rfl rfl]
  · intro h; simp only [ar, sr] at h ⊢; rw [eR]; split_ifs at h ⊢; omega
  · intro h; simp only [ar, sr] at h ⊢; rw [eR]; split_ifs at h ⊢; omega
  · intro h; simp only [ad, sd] at h ⊢; rw [eD]; split_ifs at h ⊢; omega
  · intro h; simp only [ad, sd] at h ⊢; rw [eD]; split_ifs at h ⊢; omega
  · -- `LT`: `|r| < |d|`, unless dividing by zero
    simp only [dz, ar, ad]
    by_cases hd0 : d = 0
    · have h0 : sgn64 s 0 = 0 := by simp [sgn64, toSigned64]
      simp [hd0, h0]
    · rw [ite_eq_right hd0, ite_eq_left (hlt hd0)]
  · -- the 128-bit addition `(n - r) + r = n`: the carry out is `sm + sr - sn`
    refine ⟨(if M < 0 then 1 else 0) + (if s = 1 ∧ 2^63 ≤ r then 1 else 0)
      - (if s = 1 ∧ 2^63 ≤ n then 1 else 0), ?_⟩
    simp only [nsr, sm, M, ext128_eq, eN, eR]
    split_ifs <;> omega

end LambdaVM.DvrmCore
