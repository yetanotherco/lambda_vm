module

public import LambdaVM.Buses.Alu
public import Mathlib.Tactic.Ring
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# The arithmetic of the MUL chip, on naturals

`spec/mul.typ` multiplies by the schoolbook method on 16-bit halves. Both operands are extended
to eight halves, the upper four being `65535 * is_negative`, so that the extension is two's
complement. The products are collected by 32-bit position into four limbs

  `raw_product[i] = Σ_(k ≤ 1) 2^(16k) Σ_(j ≤ 2i+k) l_j · r_(2i+k-j)`

and the terms at or above `2^128` are dropped, which is exactly the truncation the 128-bit
result asks for. A four-step carry chain then turns the four `raw` limbs into the four 32-bit
words of `lo ++ hi`.

* `product_identity` — the schoolbook identity, with the discarded terms named.
* `raw_lt` — each `raw` limb is below `2^51`, the `B51` type of the column.
* `product_mod` — the carry chain makes the four words the 128-bit product (soundness).
* `limbs_of_mod` — the honest carries satisfy the chain (completeness).
* `ext_eq_ext128` — the upper four halves `65535 * neg` are `Alu.ext128`.
* `carry_step` / `carry_cast` — the field carry `2⁻³²·(r + cin − res)` and the natural one agree.
-/

namespace LambdaVM.MulCore
open Alu (ext128 mulResult)

/-- Four 16-bit halves as a 64-bit natural. -/
def ofHalves4 (x0 x1 x2 x3 : ℕ) : ℕ := x0 + 2^16 * x1 + 2^32 * x2 + 2^48 * x3

/-- Eight 16-bit halves as a 128-bit natural: an operand after sign extension. -/
def ofHalves8 (x0 x1 x2 x3 x4 x5 x6 x7 : ℕ) : ℕ :=
  x0 + 2^16 * x1 + 2^32 * x2 + 2^48 * x3
    + 2^64 * x4 + 2^80 * x5 + 2^96 * x6 + 2^112 * x7

/-- `raw_product[i]`, as `spec/mul.typ` defines it. -/
def raw (i : ℕ) (l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 : ℕ) : ℕ :=
  match i with
  | 0 => (l0 * r0)
         + 2^16 * (l0 * r1 + l1 * r0)
  | 1 => (l0 * r2 + l1 * r1 + l2 * r0)
         + 2^16 * (l0 * r3 + l1 * r2 + l2 * r1 + l3 * r0)
  | 2 => (l0 * r4 + l1 * r3 + l2 * r2 + l3 * r1 + l4 * r0)
         + 2^16 * (l0 * r5 + l1 * r4 + l2 * r3 + l3 * r2 + l4 * r1 + l5 * r0)
  | _ => (l0 * r6 + l1 * r5 + l2 * r4 + l3 * r3 + l4 * r2 + l5 * r1 + l6 * r0)
         + 2^16 * (l0 * r7 + l1 * r6 + l2 * r5 + l3 * r4 + l4 * r3 + l5 * r2 + l6 * r1 + l7 * r0)

/-- The terms of the product at or above `2^128`, which the result discards. -/
def rawHigh (l1 l2 l3 l4 l5 l6 l7 r1 r2 r3 r4 r5 r6 r7 : ℕ) : ℕ :=
  l1 * r7
  + l2 * r6 + 2^16 * (l2 * r7)
  + l3 * r5 + 2^16 * (l3 * r6) + 2^32 * (l3 * r7)
  + l4 * r4 + 2^16 * (l4 * r5) + 2^32 * (l4 * r6) + 2^48 * (l4 * r7)
  + l5 * r3 + 2^16 * (l5 * r4) + 2^32 * (l5 * r5) + 2^48 * (l5 * r6) + 2^64 * (l5 * r7)
  + l6 * r2 + 2^16 * (l6 * r3) + 2^32 * (l6 * r4) + 2^48 * (l6 * r5) + 2^64 * (l6 * r6)
    + 2^80 * (l6 * r7)
  + l7 * r1 + 2^16 * (l7 * r2) + 2^32 * (l7 * r3) + 2^48 * (l7 * r4) + 2^64 * (l7 * r5)
    + 2^80 * (l7 * r6) + 2^96 * (l7 * r7)

/-- The schoolbook identity: the product of the two extended operands is the four `raw` limbs at
their 32-bit positions, plus the terms the truncation drops. -/
theorem product_identity (l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 : ℕ) :
    ofHalves8 l0 l1 l2 l3 l4 l5 l6 l7 * ofHalves8 r0 r1 r2 r3 r4 r5 r6 r7
      = raw 0 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
        + 2^32 * raw 1 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
        + 2^64 * raw 2 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
        + 2^96 * raw 3 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
        + 2^128 * rawHigh l1 l2 l3 l4 l5 l6 l7 r1 r2 r3 r4 r5 r6 r7 := by
  simp only [ofHalves8, raw, rawHigh]
  ring

/-- Each `raw` limb stays below `2^51`, the `B51` type of the column: it is a sum of at most eight
products of halves, plus `2^16` times another such sum. -/
theorem raw_lt {l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 : ℕ}
    (hl0 : l0 < 2^16) (hl1 : l1 < 2^16) (hl2 : l2 < 2^16) (hl3 : l3 < 2^16)
    (hl4 : l4 < 2^16) (hl5 : l5 < 2^16) (hl6 : l6 < 2^16) (hl7 : l7 < 2^16)
    (hr0 : r0 < 2^16) (hr1 : r1 < 2^16) (hr2 : r2 < 2^16) (hr3 : r3 < 2^16)
    (hr4 : r4 < 2^16) (hr5 : r5 < 2^16) (hr6 : r6 < 2^16) (hr7 : r7 < 2^16)
    (i : ℕ) : raw i l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 < 2^51 := by
  have key : ∀ a b : ℕ, a < 2^16 → b < 2^16 → a * b ≤ 65535 * 65535 :=
    fun _ _ ha hb => Nat.mul_le_mul (by omega) (by omega)
  have h00 := key _ _ hl0 hr0; have h01 := key _ _ hl0 hr1; have h02 := key _ _ hl0 hr2
  have h03 := key _ _ hl0 hr3; have h04 := key _ _ hl0 hr4; have h05 := key _ _ hl0 hr5
  have h06 := key _ _ hl0 hr6; have h07 := key _ _ hl0 hr7
  have h10 := key _ _ hl1 hr0; have h11 := key _ _ hl1 hr1; have h12 := key _ _ hl1 hr2
  have h13 := key _ _ hl1 hr3; have h14 := key _ _ hl1 hr4; have h15 := key _ _ hl1 hr5
  have h16 := key _ _ hl1 hr6
  have h20 := key _ _ hl2 hr0; have h21 := key _ _ hl2 hr1; have h22 := key _ _ hl2 hr2
  have h23 := key _ _ hl2 hr3; have h24 := key _ _ hl2 hr4; have h25 := key _ _ hl2 hr5
  have h30 := key _ _ hl3 hr0; have h31 := key _ _ hl3 hr1; have h32 := key _ _ hl3 hr2
  have h33 := key _ _ hl3 hr3; have h34 := key _ _ hl3 hr4
  have h40 := key _ _ hl4 hr0; have h41 := key _ _ hl4 hr1; have h42 := key _ _ hl4 hr2
  have h43 := key _ _ hl4 hr3
  have h50 := key _ _ hl5 hr0; have h51 := key _ _ hl5 hr1; have h52 := key _ _ hl5 hr2
  have h60 := key _ _ hl6 hr0; have h61 := key _ _ hl6 hr1
  have h70 := key _ _ hl7 hr0
  unfold raw
  split <;> omega

/-- **Soundness of the reduction.** If the carries link the `raw` limbs to four 32-bit words,
those words are the 128-bit product. -/
theorem product_mod (l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 : ℕ)
    {res0 res1 res2 res3 c0 c1 c2 c3 : ℕ}
    (hres0 : res0 < 2^32) (hres1 : res1 < 2^32) (hres2 : res2 < 2^32) (hres3 : res3 < 2^32)
    (e0 : raw 0 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 = res0 + 2^32 * c0)
    (e1 : raw 1 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 + c0 = res1 + 2^32 * c1)
    (e2 : raw 2 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 + c1 = res2 + 2^32 * c2)
    (e3 : raw 3 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 + c2 = res3 + 2^32 * c3) :
    res0 + 2^32 * res1 + 2^64 * res2 + 2^96 * res3
      = (ofHalves8 l0 l1 l2 l3 l4 l5 l6 l7 * ofHalves8 r0 r1 r2 r3 r4 r5 r6 r7) % 2^128 := by
  rw [product_identity]
  generalize raw 0 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 = R0 at *
  generalize raw 1 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 = R1 at *
  generalize raw 2 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 = R2 at *
  generalize raw 3 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 = R3 at *
  omega

/-- The product modulo `2^128` depends only on the four `raw` limbs. -/
theorem product_mod_raw (l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 : ℕ) :
    (ofHalves8 l0 l1 l2 l3 l4 l5 l6 l7 * ofHalves8 r0 r1 r2 r3 r4 r5 r6 r7) % 2^128
      = (raw 0 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
          + 2^32 * raw 1 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
          + 2^64 * raw 2 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7
          + 2^96 * raw 3 l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7) % 2^128 := by
  rw [product_identity, Nat.add_mul_mod_self_left]

/-- **Completeness of the reduction.** For the honest result, the honest carries
`c_i = (R_i + c_(i-1)) / 2^32` satisfy the chain. -/
theorem limbs_of_mod {R0 R1 R2 R3 res0 res1 res2 res3 : ℕ}
    (hres0 : res0 < 2^32) (hres1 : res1 < 2^32) (hres2 : res2 < 2^32) (hres3 : res3 < 2^32)
    (h : res0 + 2^32 * res1 + 2^64 * res2 + 2^96 * res3
      = (R0 + 2^32 * R1 + 2^64 * R2 + 2^96 * R3) % 2^128) :
    R0 = res0 + 2^32 * (R0 / 2^32) ∧
      R1 + R0 / 2^32 = res1 + 2^32 * ((R1 + R0 / 2^32) / 2^32) ∧
      R2 + (R1 + R0 / 2^32) / 2^32 = res2 + 2^32 * ((R2 + (R1 + R0 / 2^32) / 2^32) / 2^32) ∧
      R3 + (R2 + (R1 + R0 / 2^32) / 2^32) / 2^32
        = res3 + 2^32 * ((R3 + (R2 + (R1 + R0 / 2^32) / 2^32) / 2^32) / 2^32) := by
  omega

/-- The upper four halves `65535 * n` are two's complement sign extension. -/
theorem ext_limbs (x0 x1 x2 x3 n : ℕ) :
    ofHalves8 x0 x1 x2 x3 (65535 * n) (65535 * n) (65535 * n) (65535 * n)
      = ofHalves4 x0 x1 x2 x3 + n * (2^128 - 2^64) := by
  simp only [ofHalves8, ofHalves4]
  norm_num
  ring

/-- `is_negative` as `SIGN` computes it makes that extension `Alu.ext128`. -/
theorem ext_eq_ext128 {x n s : ℕ} (_hx : x < 2^64) (_hs : s < 2)
    (hn : n = if s = 1 ∧ 2^63 ≤ x then 1 else 0) :
    x + n * (2^128 - 2^64) = ext128 s x := by
  unfold ext128
  split_ifs with h <;> simp_all

/-- The two halves of the 128-bit product are what the `ALU` bus promises for the two `MUL`
messages a row pushes, the `high = 0` one and the `high = 1` one. -/
theorem mulResult_of_product {ls rs x y lo hi : ℕ} (hlo : lo < 2^64)
    (h : lo + 2^64 * hi = (ext128 ls x * ext128 rs y) % 2^128) :
    mulResult ls rs 0 x y = lo ∧ mulResult ls rs 1 x y = hi := by
  -- `omega` will not look inside the product, so the split is stated on an opaque `P`
  have key : ∀ P : ℕ, lo + 2^64 * hi = P → P % 2^64 = lo ∧ P / 2^64 = hi :=
    fun _ hp => by omega
  obtain ⟨h0, h1⟩ := key _ h
  refine ⟨?_, ?_⟩ <;> simp only [mulResult, reduceIte]
  · exact h0
  · exact h1

/-! ## The carry chain, between the field and the naturals -/

/-- One step of the carry chain lifted to `ℕ`: with `c = 2⁻³²·(r + cin − res)` and every value
small, `r + cin = res + 2^32·c` holds over the naturals. -/
theorem carry_step {r cin res c : F p} (hr : r.val < 2^52) (hcin : cin.val < 2^20)
    (hres : res.val < 2^32) (hc : c.val < 2^20)
    (h : c = (r + cin - res) * inv32) :
    r.val + cin.val = res.val + 2^32 * c.val := by
  have hp : (2 : ℕ)^53 < p := by decide
  have h' : c * 2^32 = r + cin - res := by
    rw [h, mul_assoc, inv32_mul_two_pow, mul_one]
  refine natCast_inj_of_lt (by omega) (by omega) ?_
  push_cast
  simp only [ZMod.natCast_zmod_val]
  linear_combination -h'

/-- The field carry is the honest natural carry: `2⁻³²·(r + cin − res) = k` when the naturals
behind `r`, `cin` and `res` satisfy `R + Cin = S + 2^32·k`. -/
theorem carry_cast {r cin res : F p} {R Cin S k : ℕ} (hr : r = (R : F p))
    (hcin : cin = (Cin : F p)) (hres : res = (S : F p)) (h : R + Cin = S + 2^32 * k) :
    (r + cin - res) * inv32 = (k : F p) := by
  have h' : (R : F p) + Cin - S = 2^32 * k := by
    have hc := congrArg (Nat.cast : ℕ → F p) h
    push_cast at hc
    linear_combination hc
  rw [hr, hcin, hres, h', mul_right_comm, two_pow_mul_inv32, one_mul]

end LambdaVM.MulCore
