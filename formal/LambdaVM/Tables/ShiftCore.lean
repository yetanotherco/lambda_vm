module

public import LambdaVM.Buses.Alu

@[expose] public section

/-!
# The arithmetic of the SHIFT chip, on naturals

`spec/shift.typ` shifts in two phases: by `bit_shift < 16` bits inside each 16-bit limb using
the `HWSL` lookup, then by whole limbs using the one-hot `limb_shift`. These lemmas prove that
the limb recombination the chip constrains equals the shift of the 64-bit value.
-/

namespace LambdaVM.ShiftCore
open Alu (shiftResult sra64)

/-- `HWSL` as in BITWISE: `(h << z) mod 2^16` and `h >> (16 − z)`, with the `z = 0` convention. -/
def sll (h z : ℕ) : ℕ := if z = 0 then h else (h <<< z) % 65536
def sllc (h z : ℕ) : ℕ := if z = 0 then 0 else h >>> (16 - z)

def limbs (a0 a1 a2 a3 : ℕ) : ℕ := a0 + 65536 * a1 + 2^32 * a2 + 2^48 * a3

theorem and15 (s : ℕ) : s &&& 15 = s % 16 := by
  have := Nat.and_two_pow_sub_one_eq_mod s 4
  norm_num at this
  exact this

theorem and48 {s : ℕ} (hs : s < 256) : s &&& 48 = 16 * (s / 16 % 4) := by
  interval_cases s <;> decide

theorem and16 {s : ℕ} (hs : s < 256) : s &&& 16 = 16 * (s / 16 % 2) := by
  interval_cases s <;> decide

/-- Left shift: phase 1 (`bit_shift = b`) and phase 2 (`limb_shift = L`) together. -/
theorem leftShift {i0 i1 i2 i3 b L : ℕ}
    (h0 : i0 < 65536) (h1 : i1 < 65536) (h2 : i2 < 65536) (h3 : i3 < 65536)
    (hb : b < 16) (hL : L < 4) :
    let il0 := sll i0 b
    let il1 := sll i1 b + sllc i0 b
    let il2 := sll i2 b + sllc i1 b
    let il3 := sll i3 b + sllc i2 b
    let sh0 := if L = 0 then il0 else 0
    let sh1 := if L = 0 then il1 else if L = 1 then il0 else 0
    let sh2 := if L = 0 then il2 else if L = 1 then il1 else if L = 2 then il0 else 0
    let sh3 := if L = 0 then il3 else if L = 1 then il2 else if L = 2 then il1 else il0
    sh0 < 65536 ∧ sh1 < 65536 ∧ sh2 < 65536 ∧ sh3 < 65536 ∧
    limbs sh0 sh1 sh2 sh3 = (limbs i0 i1 i2 i3 <<< (16 * L + b)) % 2^64 := by
  intro il0 il1 il2 il3 sh0 sh1 sh2 sh3
  simp only [il0, il1, il2, il3, sh0, sh1, sh2, sh3, limbs, sll, sllc, Nat.shiftLeft_eq,
    Nat.shiftRight_eq_div_pow]
  interval_cases b <;> interval_cases L <;> norm_num <;> omega

/-- Right shift, phase 1: `bit_shift = b = (16 − r) mod 16` turns the limbs of `in` (plus one
extension limb `e` above) into the limbs of `in >> r`. -/
theorem rightIntra {i0 i1 i2 i3 r b e : ℕ}
    (h0 : i0 < 65536) (h1 : i1 < 65536) (h2 : i2 < 65536) (h3 : i3 < 65536)
    (hr : r < 16) (hb : b = (16 - r) % 16) (he : e = 0 ∨ e = 65535) :
    let xi (h : ℕ) := if b = 0 then 0 else sll h b
    let yi (h : ℕ) := if b = 0 then h else sllc h b
    let x4 := if b = 0 then 0 else sll e b
    let ir0 := yi i0 + xi i1
    let ir1 := yi i1 + xi i2
    let ir2 := yi i2 + xi i3
    let ir3 := yi i3 + x4
    ir0 < 65536 ∧ ir1 < 65536 ∧ ir2 < 65536 ∧ ir3 < 65536 ∧
    limbs ir0 ir1 ir2 ir3 = ((limbs i0 i1 i2 i3 + e * 2^64) >>> r) % 2^64 := by
  intro xi yi x4 ir0 ir1 ir2 ir3
  simp only [xi, yi, x4, ir0, ir1, ir2, ir3, limbs, sll, sllc, Nat.shiftLeft_eq,
    Nat.shiftRight_eq_div_pow]
  rcases he with rfl | rfl <;> interval_cases r <;> norm_num at hb <;> subst hb <;> norm_num <;> omega

/-- Right shift, phase 2: `limb_shift = L` moves whole limbs, filling with `e`. -/
theorem rightLimb {a0 a1 a2 a3 L e : ℕ}
    (h0 : a0 < 65536) (h1 : a1 < 65536) (h2 : a2 < 65536) (h3 : a3 < 65536)
    (hL : L < 4) (he : e = 0 ∨ e = 65535) :
    let sh0 := if L = 0 then a0 else if L = 1 then a1 else if L = 2 then a2 else a3
    let sh1 := if L = 0 then a1 else if L = 1 then a2 else if L = 2 then a3 else e
    let sh2 := if L = 0 then a2 else if L = 1 then a3 else e
    let sh3 := if L = 0 then a3 else e
    sh0 < 65536 ∧ sh1 < 65536 ∧ sh2 < 65536 ∧ sh3 < 65536 ∧
    limbs sh0 sh1 sh2 sh3
      = ((limbs a0 a1 a2 a3 + e * (2^64 + 2^80 + 2^96)) >>> (16 * L)) % 2^64 := by
  intro sh0 sh1 sh2 sh3
  simp only [sh0, sh1, sh2, sh3, limbs, Nat.shiftRight_eq_div_pow]
  rcases he with rfl | rfl <;> interval_cases L <;> norm_num <;> omega

/-- The two phases compose into one shift of a 128-bit sign-extended value. -/
theorem rightCompose {X r L e : ℕ} (hX : X < 2^64) (hr : r < 16) (hL : L < 4)
    (he : e = 0 ∨ e = 65535) :
    ((((X + e * 2^64) >>> r) % 2^64 + e * (2^64 + 2^80 + 2^96)) >>> (16 * L)) % 2^64
      = ((X + e * (2^64 + 2^80 + 2^96 + 2^112)) >>> (16 * L + r)) % 2^64 := by
  simp only [Nat.shiftRight_eq_div_pow]
  rcases he with rfl | rfl <;> interval_cases r <;> interval_cases L <;> norm_num <;> omega

/-- The `shifted` limbs exactly as the row polynomial reads them, on naturals:
`d` is `direction`, `l0 l1 l2` the stored one-hot limbs, `e` the extension limb. -/
def shifted (d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : ℕ) : ℕ × ℕ × ℕ × ℕ :=
  let l3 := 1 - l0 - l1 - l2
  let left := 1 - d
  let right := d
  let il0 := x0
  let il1 := x1 + y0
  let il2 := x2 + y1
  let il3 := x3 + y2
  let ir0 := y0 + x1
  let ir1 := y1 + x2
  let ir2 := y2 + x3
  let ir3 := y3 + x4
  (left * (l0 * il0) + right * (l0 * ir0 + l1 * ir1 + l2 * ir2 + l3 * ir3),
   left * (l0 * il1 + l1 * il0) + right * (l0 * ir1 + l1 * ir2 + l2 * ir3 + e * l3),
   left * (l0 * il2 + l1 * il1 + l2 * il0) + right * (l0 * ir2 + l1 * ir3 + e * (l2 + l3)),
   left * (l0 * il3 + l1 * il2 + l2 * il1 + l3 * il0) + right * (l0 * ir3 + e * (l1 + l2 + l3)))

/-- The chip's arithmetic: from the decoded `bit_shift`, the one-hot `limb_shift`, the sign bit
and the HWSL outputs (or their `zbs` overrides), the recombined limbs are the shift result. -/
theorem main {d sg w s i0 i1 i2 i3 l0 l1 l2 b n x0 x1 x2 x3 x4 y0 y1 y2 y3 : ℕ}
    (hd : d < 2) (hsg : sg < 2) (hw : w < 2) (hs : s < 256)
    (h0 : i0 < 65536) (h1 : i1 < 65536) (h2 : i2 < 65536) (h3 : i3 < 65536)
    (hl0 : l0 < 2) (hl1 : l1 < 2) (hl2 : l2 < 2) (hl : l0 + l1 + l2 ≤ 1)
    (hL : 16 * (l1 + 2 * l2 + 3 * (1 - l0 - l1 - l2)) = s &&& (48 - 32 * w))
    (hb : b = if d = 0 then s % 16 else (16 - s % 16) % 16)
    (hn : n = if sg = 1 then i3 / 32768 else 0)
    (hx0 : x0 = if b = 0 then (if d = 0 then i0 else 0) else sll i0 b)
    (hx1 : x1 = if b = 0 then (if d = 0 then i1 else 0) else sll i1 b)
    (hx2 : x2 = if b = 0 then (if d = 0 then i2 else 0) else sll i2 b)
    (hx3 : x3 = if b = 0 then (if d = 0 then i3 else 0) else sll i3 b)
    (hx4 : x4 = if b = 0 then 0 else sll (65535 * n) b)
    (hy0 : y0 = if b = 0 then (if d = 0 then 0 else i0) else sllc i0 b)
    (hy1 : y1 = if b = 0 then (if d = 0 then 0 else i1) else sllc i1 b)
    (hy2 : y2 = if b = 0 then (if d = 0 then 0 else i2) else sllc i2 b)
    (hy3 : y3 = if b = 0 then (if d = 0 then 0 else i3) else sllc i3 b) :
    let sh := shifted d l0 l1 l2 (65535 * n) x0 x1 x2 x3 x4 y0 y1 y2 y3
    sh.1 < 65536 ∧ sh.2.1 < 65536 ∧ sh.2.2.1 < 65536 ∧ sh.2.2.2 < 65536 ∧
    limbs sh.1 sh.2.1 sh.2.2.1 sh.2.2.2 = shiftResult w sg d (limbs i0 i1 i2 i3) s := by
  intro sh
  -- the one-hot index and the shift amount
  have hmask : s &&& (48 - 32 * w) = 16 * (s / 16 % (4 - 2 * w)) := by
    rcases (by omega : w = 0 ∨ w = 1) with rfl | rfl
    · simpa using and48 hs
    · simpa using and16 hs
  rw [hmask] at hL
  have hn2 : n < 2 := by rw [hn]; split_ifs <;> omega
  have he : 65535 * n = 0 ∨ 65535 * n = 65535 := by omega
  have hX : limbs i0 i1 i2 i3 < 2^64 := by unfold limbs; omega
  rcases (by omega : d = 0 ∨ d = 1) with rfl | rfl
  · -- left shift: `shifted` is the if-form of `leftShift` with `L` the one-hot index
    simp only [if_true] at hb
    subst hb
    have hb16 : s % 16 < 16 := Nat.mod_lt _ (by norm_num)
    have core := leftShift h0 h1 h2 h3 hb16 (L := l1 + 2 * l2 + 3 * (1 - l0 - l1 - l2)) (by omega)
    simp only at core
    have hk : s % (if w = 1 then 32 else 64) = 16 * (s / 16 % (4 - 2 * w)) + s % 16 := by
      rcases (by omega : w = 0 ∨ w = 1) with rfl | rfl <;> simp <;> omega
    have hLv : s / 16 % (4 - 2 * w) = l1 + 2 * l2 + 3 * (1 - l0 - l1 - l2) := by omega
    rcases (by omega : (l0 = 1 ∧ l1 = 0 ∧ l2 = 0) ∨ (l0 = 0 ∧ l1 = 1 ∧ l2 = 0) ∨
        (l0 = 0 ∧ l1 = 0 ∧ l2 = 1) ∨ (l0 = 0 ∧ l1 = 0 ∧ l2 = 0)) with
      ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;>
    · simp only [shifted, sh, hx0, hx1, hx2, hx3, hx4, hy0, hy1, hy2, hy3]
      norm_num at core hLv ⊢
      simp only [shiftResult, if_true]
      rw [hk, hLv]
      norm_num at core ⊢
      by_cases hz : s % 16 = 0
      · simpa [sll, sllc, hz] using core
      · simpa [sll, sllc, hz] using core
  · -- right shift: phase 1, phase 2, and their composition
    simp only [one_ne_zero, if_false] at hb
    have hr : s % 16 < 16 := Nat.mod_lt _ (by norm_num)
    have p1 := rightIntra h0 h1 h2 h3 hr hb he
    simp only at p1
    obtain ⟨hr0, hr1, hr2, hr3, hir⟩ := p1
    have p2 := rightLimb hr0 hr1 hr2 hr3 (L := l1 + 2 * l2 + 3 * (1 - l0 - l1 - l2)) (by omega) he
    simp only at p2
    obtain ⟨hs0, hs1, hs2, hs3, hsh⟩ := p2
    rw [hir, rightCompose hX hr (by omega) he] at hsh
    have hk : s % (if w = 1 then 32 else 64) = 16 * (s / 16 % (4 - 2 * w)) + s % 16 := by
      rcases (by omega : w = 0 ∨ w = 1) with rfl | rfl <;> simp <;> omega
    -- the result as the shift of the sign-extended value
    have hres : shiftResult w sg 1 (limbs i0 i1 i2 i3) s
        = ((limbs i0 i1 i2 i3 + 65535 * n * (2^64 + 2^80 + 2^96 + 2^112))
            >>> (16 * (s / 16 % (4 - 2 * w)) + s % 16)) % 2^64 := by
      simp only [shiftResult, one_ne_zero, if_false]
      rw [hk]
      have hX63 : limbs i0 i1 i2 i3 < 2^63 ↔ i3 < 32768 := by unfold limbs; omega
      rcases (by omega : sg = 0 ∨ sg = 1) with rfl | rfl
      · simp only [if_true, zero_ne_one, if_false] at hn ⊢
        subst hn
        simp only [mul_zero, zero_mul, add_zero]
        rw [Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.shiftRight_le _ _) hX)]
      · simp only [if_true, one_ne_zero, if_false] at hn ⊢
        subst hn
        rcases (by omega : i3 / 32768 = 0 ∨ i3 / 32768 = 1) with h | h
        · have hlt : limbs i0 i1 i2 i3 < 2^63 := hX63.mpr (by omega)
          rw [h]; simp only [mul_zero, zero_mul, add_zero, sra64, if_pos hlt]
          rw [Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.shiftRight_le _ _) hX)]
        · have hge : ¬ limbs i0 i1 i2 i3 < 2^63 := by rw [hX63]; omega
          rw [h]; simp only [sra64, if_neg hge]
          norm_num
    rw [hres]
    have hLv : s / 16 % (4 - 2 * w) = l1 + 2 * l2 + 3 * (1 - l0 - l1 - l2) := by omega
    rw [hLv]
    rcases (by omega : (l0 = 1 ∧ l1 = 0 ∧ l2 = 0) ∨ (l0 = 0 ∧ l1 = 1 ∧ l2 = 0) ∨
        (l0 = 0 ∧ l1 = 0 ∧ l2 = 1) ∨ (l0 = 0 ∧ l1 = 0 ∧ l2 = 0)) with
      ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ | ⟨rfl, rfl, rfl⟩ <;>
    · simp only [shifted, sh, hx0, hx1, hx2, hx3, hx4, hy0, hy1, hy2, hy3]
      norm_num at hs0 hs1 hs2 hs3 hsh
      refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
      first | (norm_num; done) | (norm_num; assumption)

end LambdaVM.ShiftCore
