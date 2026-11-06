module

public import Clean.Air.CountedChannel
public import LambdaVM.Buses.Alu
public import LambdaVM.Mu
public import LambdaVM.Tables.Bitwise
public import LambdaVM.Tables.MulCore
public import LambdaVM.Templates.IsBit
public import LambdaVM.Templates.IsByte
public import LambdaVM.Templates.Sign

@[expose] public section

/-!
# MUL (`spec/src/mul.toml`)

The 128-bit product of two 64-bit operands, each read as signed or unsigned by its own flag, so
one chip serves the four RV64IM forms `MUL`, `MULH`, `MULHU` and `MULHSU`. A row holds the
operands and both halves of the result as 16-bit limbs, the two sign bits `SIGN` produces, and
the four `raw_product` limbs of the schoolbook convolution. The carries are virtual, read off
`raw_product` and the result by division by `2^32`.

The row pushes two `ALU` messages for one product, `high = 0` for `lo` and `high = 1` for `hi`,
with multiplicities `μ_lo` and `μ_hi`. Every range check fires with their sum, which is why the
model's counted-pull bound is `2 * 255` and not `255` (see `LambdaVM/Mu.lean`).

The arithmetic lives in `MulCore.lean`; this file is the translation between a row and those
statements.

Two readings differ from `spec/src/mul.toml` as vendored under `spec/`, and follow the newer
spec and `prover/src/tables/mul.rs` instead:

* `μ_lo` and `μ_hi` are bytes, range-checked. The vendored TOML types them `BaseField` and
  declares no check. `mul.rs` explains what the bound buys: without it a row with
  `μ_lo = 1, μ_hi = −1` receives a lookup while sending none of its range checks, and an honest
  `μ_hi = 1` copy of the row cancels the stray `−1`.
* The two `IS_BYTE` templates are the single `ARE_BYTES[μ_lo, μ_hi]` that `mul.rs` emits, which
  is the same statement and keeps the interaction count at parity.
-/

namespace LambdaVM.MUL
open Bitwise (IsHalf IsB20)

/-- `IS_HALF` and `IS_B20` with pulls of multiplicity up to `muMax`. A MUL row pulls them
`μ_lo + μ_hi` times, and each multiplicity is a byte. -/
abbrev IsHalfC : RawChannel (F p) := Bitwise.IsHalf.counted muMax
abbrev IsB20C : RawChannel (F p) := Bitwise.IsB20.counted muMax

/-- A MUL row, in the column order of `cols` in `prover/src/tables/mul.rs`. -/
structure Row (F : Type) where
  lhs0 : F
  lhs1 : F
  lhs2 : F
  lhs3 : F
  lhsSigned : F
  rhs0 : F
  rhs1 : F
  rhs2 : F
  rhs3 : F
  rhsSigned : F
  lo0 : F
  lo1 : F
  lo2 : F
  lo3 : F
  hi0 : F
  hi1 : F
  hi2 : F
  hi3 : F
  lhsNeg : F
  rhsNeg : F
  raw0 : F
  raw1 : F
  raw2 : F
  raw3 : F
  muLo : F
  muHi : F
deriving ProvableStruct

/-- Two 16-bit halves as the 32-bit word the `ALU` bus carries. -/
@[circuit_norm]
def word (x0 x1 : Expression (F p)) : Expression (F p) := x0 + 65536 * x1

/-- `raw_product[i]` as an expression in the eight extended halves of each operand, the same
convolution `MulCore.raw` states over the naturals. -/
@[circuit_norm]
def rawE (i : ℕ) (l0 l1 l2 l3 l4 l5 l6 l7 r0 r1 r2 r3 r4 r5 r6 r7 : Expression (F p)) :
    Expression (F p) :=
  match i with
  | 0 => (l0 * r0)
         + 65536 * (l0 * r1 + l1 * r0)
  | 1 => (l0 * r2 + l1 * r1 + l2 * r0)
         + 65536 * (l0 * r3 + l1 * r2 + l2 * r1 + l3 * r0)
  | 2 => (l0 * r4 + l1 * r3 + l2 * r2 + l3 * r1 + l4 * r0)
         + 65536 * (l0 * r5 + l1 * r4 + l2 * r3 + l3 * r2 + l4 * r1 + l5 * r0)
  | _ => (l0 * r6 + l1 * r5 + l2 * r4 + l3 * r3 + l4 * r2 + l5 * r1 + l6 * r0)
         + 65536 * (l0 * r7 + l1 * r6 + l2 * r5 + l3 * r4 + l4 * r3 + l5 * r2 + l6 * r1 + l7 * r0)

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  -- IS_BYTE<1, μ_lo> and IS_BYTE<1, μ_hi>, batched as `mul.rs` emits them
  areBytes r.muLo r.muHi
  -- IS_BIT<lhs_signed>, IS_BIT<rhs_signed>
  isBit r.lhsSigned
  isBit r.rhsSigned
  -- every range check below fires with the sum of the two multiplicities
  let mu := r.muLo + r.muHi
  -- range: IS_HALF on both operands and both halves of the result
  IsHalf.pullUpTo muMax mu r.lhs0
  IsHalf.pullUpTo muMax mu r.lhs1
  IsHalf.pullUpTo muMax mu r.lhs2
  IsHalf.pullUpTo muMax mu r.lhs3
  IsHalf.pullUpTo muMax mu r.rhs0
  IsHalf.pullUpTo muMax mu r.rhs1
  IsHalf.pullUpTo muMax mu r.rhs2
  IsHalf.pullUpTo muMax mu r.rhs3
  -- defs: SIGN<lhs_is_negative; lhs[3], lhs_signed> and the same for rhs
  sign r.lhsSigned r.lhs3 r.lhsNeg
  sign r.rhsSigned r.rhs3 r.rhsNeg
  IsHalf.pullUpTo muMax mu r.lo0
  IsHalf.pullUpTo muMax mu r.lo1
  IsHalf.pullUpTo muMax mu r.lo2
  IsHalf.pullUpTo muMax mu r.lo3
  IsHalf.pullUpTo muMax mu r.hi0
  IsHalf.pullUpTo muMax mu r.hi1
  IsHalf.pullUpTo muMax mu r.hi2
  IsHalf.pullUpTo muMax mu r.hi3
  -- virtual: the four 32-bit words of the result, and the carries that link them to `raw_product`
  let res0 := word r.lo0 r.lo1
  let res1 := word r.lo2 r.lo3
  let res2 := word r.hi0 r.hi1
  let res3 := word r.hi2 r.hi3
  let c0 := (r.raw0 - res0) * inv32
  let c1 := (r.raw1 + c0 - res1) * inv32
  let c2 := (r.raw2 + c1 - res2) * inv32
  let c3 := (r.raw3 + c2 - res3) * inv32
  IsB20.pullUpTo muMax mu c0
  IsB20.pullUpTo muMax mu c1
  IsB20.pullUpTo muMax mu c2
  IsB20.pullUpTo muMax mu c3
  -- prod: raw_product[i] is the convolution of the sign-extended operands. Written
  -- `raw_product - convolution`, the way `mul.rs` emits it and the way the `constraint` field of
  -- `spec/src/mul.toml` reads it. Its `poly` field there is the negation of its own `constraint`,
  -- the one place in the spec that breaks the left-hand-side-first convention EQ and SHIFT follow.
  let le4 := 65535 * r.lhsNeg
  let re4 := 65535 * r.rhsNeg
  assertZero (r.raw0 - rawE 0 r.lhs0 r.lhs1 r.lhs2 r.lhs3 le4 le4 le4 le4
    r.rhs0 r.rhs1 r.rhs2 r.rhs3 re4 re4 re4 re4)
  assertZero (r.raw1 - rawE 1 r.lhs0 r.lhs1 r.lhs2 r.lhs3 le4 le4 le4 le4
    r.rhs0 r.rhs1 r.rhs2 r.rhs3 re4 re4 re4 re4)
  assertZero (r.raw2 - rawE 2 r.lhs0 r.lhs1 r.lhs2 r.lhs3 le4 le4 le4 le4
    r.rhs0 r.rhs1 r.rhs2 r.rhs3 re4 re4 re4 re4)
  assertZero (r.raw3 - rawE 3 r.lhs0 r.lhs1 r.lhs2 r.lhs3 le4 le4 le4 le4
    r.rhs0 r.rhs1 r.rhs2 r.rhs3 re4 re4 re4 re4)
  -- output: ALU[lhs, rhs, opsel(MUL) + 32·lhs_signed + 64·rhs_signed (+128)] -> lo, hi
  let flags := 7 + 32 * r.lhsSigned + 64 * r.rhsSigned
  Alu.channel.pushIf r.muLo
    ⟨word r.lhs0 r.lhs1, word r.lhs2 r.lhs3, word r.rhs0 r.rhs1, word r.rhs2 r.rhs3,
     flags, res0, res1⟩
  Alu.channel.pushIf r.muHi
    ⟨word r.lhs0 r.lhs1, word r.lhs2 r.lhs3, word r.rhs0 r.rhs1, word r.rhs2 r.rhs3,
     flags + 128, res2, res3⟩

/-! ## What a row means -/

/-- The left operand as a 64-bit value. -/
def Row.lhsVal (r : Row (F p)) : ℕ :=
  MulCore.ofHalves4 r.lhs0.val r.lhs1.val r.lhs2.val r.lhs3.val
/-- The right operand as a 64-bit value. -/
def Row.rhsVal (r : Row (F p)) : ℕ :=
  MulCore.ofHalves4 r.rhs0.val r.rhs1.val r.rhs2.val r.rhs3.val
/-- The low half of the result as a 64-bit value. -/
def Row.loVal (r : Row (F p)) : ℕ :=
  MulCore.ofHalves4 r.lo0.val r.lo1.val r.lo2.val r.lo3.val
/-- The high half of the result as a 64-bit value. -/
def Row.hiVal (r : Row (F p)) : ℕ :=
  MulCore.ofHalves4 r.hi0.val r.hi1.val r.hi2.val r.hi3.val

/-- An active row holds the 128-bit product of its two operands, each read as signed or unsigned
by its own flag. This is the whole meaning of the chip; everything else in the row is the
apparatus that computes it. -/
def RowSpec (r : Row (F p)) : Prop :=
  r.muLo + r.muHi ≠ 0 →
    r.loVal + 2^64 * r.hiVal
      = (Alu.ext128 r.lhsSigned.val r.lhsVal * Alu.ext128 r.rhsSigned.val r.rhsVal) % 2^128

/-- The honest prover's row, as `dedup_mul_rows` and the trace generator of `mul.rs` fill it:
byte multiplicities, bit sign flags, half limbs throughout, the two sign bits `SIGN` computes,
the four convolution limbs, and a result that is the product. -/
def Honest (r : Row (F p)) : Prop :=
  r.muLo.val < 2^8 ∧ r.muHi.val < 2^8 ∧
  IsBool r.lhsSigned ∧ IsBool r.rhsSigned ∧
  r.lhs0.val < 2^16 ∧ r.lhs1.val < 2^16 ∧ r.lhs2.val < 2^16 ∧ r.lhs3.val < 2^16 ∧
  r.rhs0.val < 2^16 ∧ r.rhs1.val < 2^16 ∧ r.rhs2.val < 2^16 ∧ r.rhs3.val < 2^16 ∧
  r.lo0.val < 2^16 ∧ r.lo1.val < 2^16 ∧ r.lo2.val < 2^16 ∧ r.lo3.val < 2^16 ∧
  r.hi0.val < 2^16 ∧ r.hi1.val < 2^16 ∧ r.hi2.val < 2^16 ∧ r.hi3.val < 2^16 ∧
  r.lhsNeg.val = (if r.lhsSigned.val = 1 ∧ 2^63 ≤ r.lhsVal then 1 else 0) ∧
  r.rhsNeg.val = (if r.rhsSigned.val = 1 ∧ 2^63 ≤ r.rhsVal then 1 else 0) ∧
  r.raw0.val = MulCore.raw 0 r.lhs0.val r.lhs1.val r.lhs2.val r.lhs3.val
    (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val)
    r.rhs0.val r.rhs1.val r.rhs2.val r.rhs3.val
    (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) ∧
  r.raw1.val = MulCore.raw 1 r.lhs0.val r.lhs1.val r.lhs2.val r.lhs3.val
    (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val)
    r.rhs0.val r.rhs1.val r.rhs2.val r.rhs3.val
    (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) ∧
  r.raw2.val = MulCore.raw 2 r.lhs0.val r.lhs1.val r.lhs2.val r.lhs3.val
    (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val)
    r.rhs0.val r.rhs1.val r.rhs2.val r.rhs3.val
    (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) ∧
  r.raw3.val = MulCore.raw 3 r.lhs0.val r.lhs1.val r.lhs2.val r.lhs3.val
    (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val) (65535 * r.lhsNeg.val)
    r.rhs0.val r.rhs1.val r.rhs2.val r.rhs3.val
    (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) (65535 * r.rhsNeg.val) ∧
  r.loVal + 2^64 * r.hiVal
    = (Alu.ext128 r.lhsSigned.val r.lhsVal * Alu.ext128 r.rhsSigned.val r.rhsVal) % 2^128

-- A 25-interaction circuit: without the precomputed instance every proof below elaborates with
-- metavariables and times out. See "Lessons for big chips" in formal/README.md.
@[reducible]
instance elaborated : ElaboratedCircuit (F p) Row unit main := by elaborate_circuit

/-- The `SIGN` bit of a 64-bit operand held as four halves: `0` when the operand is read
unsigned, and the top bit of its top half otherwise. This is the sign bit `Alu.ext128` reads. -/
theorem neg_eq_sign {s n h0 h1 h2 h3 : ℕ} (hh0 : h0 < 2^16) (hh1 : h1 < 2^16) (hh2 : h2 < 2^16)
    (hh3 : h3 < 2^16) (h : (s = 0 ∧ n = 0) ∨ (s = 1 ∧ n = h3 / 2^15)) :
    n = if s = 1 ∧ 2^63 ≤ MulCore.ofHalves4 h0 h1 h2 h3 then 1 else 0 := by
  have hx : (2^63 ≤ MulCore.ofHalves4 h0 h1 h2 h3) ↔ 2^15 ≤ h3 := by
    simp only [MulCore.ofHalves4]; omega
  simp only [hx]
  rcases h with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩
  · rw [ite_eq_right (by simp)]
  · by_cases hc : 2^15 ≤ h3
    · rw [ite_eq_left ⟨rfl, hc⟩]; omega
    · rw [ite_eq_right (fun hn => hc hn.2)]; omega

theorem soundness :
    GeneralFormalCircuit.Soundness (F p) (Input := Row) (Output := unit) main (fun _ _ => True)
      (fun r _ _ => RowSpec r) := by
  circuit_proof_start [main, areBytes, isBit, sign, word, rawE, RowSpec, Row.lhsVal, Row.rhsVal,
    Row.loVal, Row.hiVal, Alu.channel, Alu.Guarantees, dword, Bitwise.areBytesTable,
    Bitwise.isHalfTable, Bitwise.isB20Table, Bitwise.msb16Table]
  obtain ⟨⟨hmuLo, hmuHi⟩, h_ls, h_rs, h_l0, h_l1, h_l2, h_l3, h_r0, h_r1, h_r2, h_r3,
    h_sgnL, h_negL, h_sgnR, h_negR, h_lo0, h_lo1, h_lo2, h_lo3, h_hi0, h_hi1, h_hi2, h_hi3,
    h_cc0, h_cc1, h_cc2, h_cc3, e0, e1, e2, e3⟩ := h_holds
  have hlsb : IsBool input_lhsSigned := mul_one_sub_eq_zero_iff.mp h_ls
  have hrsb : IsBool input_rhsSigned := mul_one_sub_eq_zero_iff.mp h_rs
  have hls2 := IsBool.val_lt_two hlsb
  have hrs2 := IsBool.val_lt_two hrsb
  -- `ARE_BYTES[μ_lo, μ_hi]` makes the sum of the two multiplicities a small natural
  have hsumval : (input_muLo + input_muHi).val = input_muLo.val + input_muHi.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  by_cases h0 : input_muLo + input_muHi = 0
  · -- an inactive row owes nothing: no range check fires and neither message is pushed
    have hz : input_muLo.val + input_muHi.val = 0 := by rw [← hsumval, h0, ZMod.val_zero]
    have hmu0 : input_muLo = 0 := by rw [← ZMod.val_eq_zero]; omega
    have hmu1 : input_muHi = 0 := by rw [← ZMod.val_eq_zero]; omega
    constructorm* _ ∧ _
    all_goals first
      | exact fun h => absurd h0 h
      | exact Or.inl h0
      | exact hlsb
      | exact hrsb
      | exact fun h => absurd hmu0 h
      | exact fun h => absurd hmu1 h
  -- an active row: the sum is a count of at most `muMax`, so every range check delivers
  have hcount : Channel.CountUpTo muMax (input_muLo + input_muHi) :=
    Channel.countUpTo_of_val h0 (by simp only [hsumval, muMax]; omega)
  have hl0 := h_l0 hcount
  have hl1 := h_l1 hcount
  have hl2 := h_l2 hcount
  have hl3 := h_l3 hcount
  have hr0 := h_r0 hcount
  have hr1 := h_r1 hcount
  have hr2 := h_r2 hcount
  have hr3 := h_r3 hcount
  have hlo0 := h_lo0 hcount
  have hlo1 := h_lo1 hcount
  have hlo2 := h_lo2 hcount
  have hlo3 := h_lo3 hcount
  have hhi0 := h_hi0 hcount
  have hhi1 := h_hi1 hcount
  have hhi2 := h_hi2 hcount
  have hhi3 := h_hi3 hcount
  have hb0 := h_cc0 hcount
  have hb1 := h_cc1 hcount
  have hb2 := h_cc2 hcount
  have hb3 := h_cc3 hcount
  -- the two `SIGN` outputs are the sign bits of the operands
  have hnegL : input_lhsNeg.val = if input_lhsSigned.val = 1 ∧
      2^63 ≤ MulCore.ofHalves4 input_lhs0.val input_lhs1.val input_lhs2.val input_lhs3.val
      then 1 else 0 := by
    refine neg_eq_sign hl0 hl1 hl2 hl3 ?_
    rcases hlsb with h | h
    · rw [h] at h_negL
      simp only [sub_zero, one_mul] at h_negL
      exact Or.inl ⟨by rw [h, ZMod.val_zero], by rw [h_negL, ZMod.val_zero]⟩
    · exact Or.inr ⟨by rw [h, val_one_eq], (h_sgnL h).2⟩
  have hnegR : input_rhsNeg.val = if input_rhsSigned.val = 1 ∧
      2^63 ≤ MulCore.ofHalves4 input_rhs0.val input_rhs1.val input_rhs2.val input_rhs3.val
      then 1 else 0 := by
    refine neg_eq_sign hr0 hr1 hr2 hr3 ?_
    rcases hrsb with h | h
    · rw [h] at h_negR
      simp only [sub_zero, one_mul] at h_negR
      exact Or.inl ⟨by rw [h, ZMod.val_zero], by rw [h_negR, ZMod.val_zero]⟩
    · exact Or.inr ⟨by rw [h, val_one_eq], (h_sgnR h).2⟩
  have hLN : input_lhsNeg.val ≤ 1 := by rw [hnegL]; split_ifs <;> omega
  have hRN : input_rhsNeg.val ≤ 1 := by rw [hnegR]; split_ifs <;> omega
  -- the four `raw_product` limbs, as the convolution of `MulCore` over the naturals
  obtain ⟨RAW, hRAW⟩ : ∃ f : ℕ → ℕ, f = fun i => MulCore.raw i input_lhs0.val input_lhs1.val
      input_lhs2.val input_lhs3.val (65535 * input_lhsNeg.val) (65535 * input_lhsNeg.val)
      (65535 * input_lhsNeg.val) (65535 * input_lhsNeg.val) input_rhs0.val input_rhs1.val
      input_rhs2.val input_rhs3.val (65535 * input_rhsNeg.val) (65535 * input_rhsNeg.val)
      (65535 * input_rhsNeg.val) (65535 * input_rhsNeg.val) := ⟨_, rfl⟩
  have hrawlt : ∀ i : ℕ, RAW i < 2^51 := by
    intro i
    simp only [hRAW]
    exact MulCore.raw_lt hl0 hl1 hl2 hl3 (by omega) (by omega) (by omega) (by omega)
      hr0 hr1 hr2 hr3 (by omega) (by omega) (by omega) (by omega) i
  -- a limb below `2^51` is pinned down by its field value
  have hlift : ∀ (i : ℕ) (x : F p), x = ((RAW i : ℕ) : F p) → x.val = RAW i := by
    intro i x hx
    rw [hx]
    exact ZMod.val_natCast_of_lt (lt_trans (hrawlt i) (by decide))
  have hraw0 : input_raw0.val = RAW 0 := hlift 0 _ (by
    simp only [hRAW, MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]
    linear_combination e0)
  have hraw1 : input_raw1.val = RAW 1 := hlift 1 _ (by
    simp only [hRAW, MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]
    linear_combination e1)
  have hraw2 : input_raw2.val = RAW 2 := hlift 2 _ (by
    simp only [hRAW, MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]
    linear_combination e2)
  have hraw3 : input_raw3.val = RAW 3 := hlift 3 _ (by
    simp only [hRAW, MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]
    linear_combination e3)
  have hrl0 := hrawlt 0
  have hrl1 := hrawlt 1
  have hrl2 := hrawlt 2
  have hrl3 := hrawlt 3
  simp only [hRAW] at hraw0 hraw1 hraw2 hraw3 hrl0 hrl1 hrl2 hrl3
  -- the four 32-bit words of the result, and the virtual carries that link them to `raw_product`
  set w0 := input_lo0 + 65536 * input_lo1 with hw0
  set w1 := input_lo2 + 65536 * input_lo3 with hw1
  set w2 := input_hi0 + 65536 * input_hi1 with hw2
  set w3 := input_hi2 + 65536 * input_hi3 with hw3
  set c0 := (input_raw0 - w0) * inv32 with hc0
  set c1 := (input_raw1 + c0 - w1) * inv32 with hc1
  set c2 := (input_raw2 + c1 - w2) * inv32 with hc2
  set c3 := (input_raw3 + c2 - w3) * inv32 with hc3
  have hv0 : w0.val = input_lo0.val + 65536 * input_lo1.val := by
    rw [hw0]
    exact val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hv1 : w1.val = input_lo2.val + 65536 * input_lo3.val := by
    rw [hw1]
    exact val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hv2 : w2.val = input_hi0.val + 65536 * input_hi1.val := by
    rw [hw2]
    exact val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hv3 : w3.val = input_hi2.val + 65536 * input_hi3.val := by
    rw [hw3]
    exact val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  -- the carry chain, lifted to the naturals one step at a time
  have k0 := MulCore.carry_step (r := input_raw0) (cin := 0) (res := w0) (c := c0)
    (by omega) (by rw [ZMod.val_zero]; norm_num) (by omega) hb0 (by rw [hc0]; ring)
  have k1 := MulCore.carry_step (r := input_raw1) (cin := c0) (res := w1) (c := c1)
    (by omega) hb0 (by omega) hb1 hc1
  have k2 := MulCore.carry_step (r := input_raw2) (cin := c1) (res := w2) (c := c2)
    (by omega) hb1 (by omega) hb2 hc2
  have k3 := MulCore.carry_step (r := input_raw3) (cin := c2) (res := w3) (c := c3)
    (by omega) hb2 (by omega) hb3 hc3
  rw [ZMod.val_zero, Nat.add_zero] at k0
  -- the extended operands are `Alu.ext128` of their 64-bit values
  have hextL : MulCore.ofHalves8 input_lhs0.val input_lhs1.val input_lhs2.val input_lhs3.val
      (65535 * input_lhsNeg.val) (65535 * input_lhsNeg.val) (65535 * input_lhsNeg.val)
      (65535 * input_lhsNeg.val)
      = Alu.ext128 input_lhsSigned.val (MulCore.ofHalves4 input_lhs0.val input_lhs1.val
          input_lhs2.val input_lhs3.val) := by
    rw [MulCore.ext_limbs]
    exact MulCore.ext_eq_ext128 (by simp only [MulCore.ofHalves4]; omega) hls2 hnegL
  have hextR : MulCore.ofHalves8 input_rhs0.val input_rhs1.val input_rhs2.val input_rhs3.val
      (65535 * input_rhsNeg.val) (65535 * input_rhsNeg.val) (65535 * input_rhsNeg.val)
      (65535 * input_rhsNeg.val)
      = Alu.ext128 input_rhsSigned.val (MulCore.ofHalves4 input_rhs0.val input_rhs1.val
          input_rhs2.val input_rhs3.val) := by
    rw [MulCore.ext_limbs]
    exact MulCore.ext_eq_ext128 (by simp only [MulCore.ofHalves4]; omega) hrs2 hnegR
  -- the carry chain makes the four words the 128-bit product
  have hprod := MulCore.product_mod (res0 := w0.val) (res1 := w1.val) (res2 := w2.val)
    (res3 := w3.val) (c0 := c0.val) (c1 := c1.val) (c2 := c2.val) (c3 := c3.val)
    input_lhs0.val input_lhs1.val input_lhs2.val input_lhs3.val
    (65535 * input_lhsNeg.val) (65535 * input_lhsNeg.val) (65535 * input_lhsNeg.val)
    (65535 * input_lhsNeg.val) input_rhs0.val input_rhs1.val input_rhs2.val input_rhs3.val
    (65535 * input_rhsNeg.val) (65535 * input_rhsNeg.val) (65535 * input_rhsNeg.val)
    (65535 * input_rhsNeg.val)
    (by omega) (by omega) (by omega) (by omega)
    (by rw [← hraw0]; omega) (by rw [← hraw1]; omega) (by rw [← hraw2]; omega)
    (by rw [← hraw3]; omega)
  rw [hextL, hextR] at hprod
  -- the whole meaning of the row
  have hkey : MulCore.ofHalves4 input_lo0.val input_lo1.val input_lo2.val input_lo3.val
      + 2^64 * MulCore.ofHalves4 input_hi0.val input_hi1.val input_hi2.val input_hi3.val
      = (Alu.ext128 input_lhsSigned.val (MulCore.ofHalves4 input_lhs0.val input_lhs1.val
            input_lhs2.val input_lhs3.val)
          * Alu.ext128 input_rhsSigned.val (MulCore.ofHalves4 input_rhs0.val input_rhs1.val
            input_rhs2.val input_rhs3.val)) % 2^128 := by
    rw [← hprod, hv0, hv1, hv2, hv3]
    simp only [MulCore.ofHalves4]
    ring
  -- the two halves, as the `ALU` bus reads them
  obtain ⟨hmul0, hmul1⟩ := MulCore.mulResult_of_product
    (by simp only [MulCore.ofHalves4]; omega) hkey
  -- the operand words, read on naturals
  have hu0 : (input_lhs0 + 65536 * input_lhs1).val
      = input_lhs0.val + 65536 * input_lhs1.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hu1 : (input_lhs2 + 65536 * input_lhs3).val
      = input_lhs2.val + 65536 * input_lhs3.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hu2 : (input_rhs0 + 65536 * input_rhs1).val
      = input_rhs0.val + 65536 * input_rhs1.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hu3 : (input_rhs2 + 65536 * input_rhs3).val
      = input_rhs2.val + 65536 * input_rhs3.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hLval : (input_lhs0 + 65536 * input_lhs1).val
      + 2^32 * (input_lhs2 + 65536 * input_lhs3).val
      = MulCore.ofHalves4 input_lhs0.val input_lhs1.val input_lhs2.val input_lhs3.val := by
    simp only [MulCore.ofHalves4, hu0, hu1]; ring
  have hRval : (input_rhs0 + 65536 * input_rhs1).val
      + 2^32 * (input_rhs2 + 65536 * input_rhs3).val
      = MulCore.ofHalves4 input_rhs0.val input_rhs1.val input_rhs2.val input_rhs3.val := by
    simp only [MulCore.ofHalves4, hu2, hu3]; ring
  refine ⟨fun _ => hkey, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount,
    .inr hcount, .inr hcount, .inr hcount, hlsb, hrsb, .inr hcount, .inr hcount, .inr hcount,
    .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount, .inr hcount,
    .inr hcount, .inr hcount, ?_, ?_⟩
  · -- the `high = 0` message: the low 64 bits of the product
    intro _ _ _ _ _
    have hflags : (7 + 32 * input_lhsSigned + 64 * input_rhsSigned : F p).val
        = 7 + 32 * input_lhsSigned.val + 64 * input_rhsSigned.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    refine ⟨fun s i hs hi h => by
        exfalso; rw [hflags] at h; simp only [opselLT] at h; omega,
      fun i hi h => by exfalso; rw [hflags] at h; simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by
        exfalso; rw [hflags] at h; simp only [opselSHIFT] at h; omega,
      fun h => by exfalso; rw [hflags] at h; simp only [opselAND] at h; omega,
      fun h => by exfalso; rw [hflags] at h; simp only [opselOR] at h; omega,
      fun h => by exfalso; rw [hflags] at h; simp only [opselXOR] at h; omega, ?_⟩
    intro ls rs hgh hls hrs hh h
    rw [hflags] at h
    simp only [opselMUL] at h
    obtain ⟨rfl, rfl, rfl⟩ : ls = input_lhsSigned.val ∧ rs = input_rhsSigned.val ∧ hgh = 0 := by
      omega
    refine ⟨by omega, by omega, ?_⟩
    rw [hLval, hRval, hmul0, hv0, hv1]
    simp only [MulCore.ofHalves4]
    ring
  · -- the `high = 1` message: the high 64 bits of the product
    intro _ _ _ _ _
    have hflags : (7 + 32 * input_lhsSigned + 64 * input_rhsSigned + 128 : F p).val
        = 7 + 32 * input_lhsSigned.val + 64 * input_rhsSigned.val + 128 :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    refine ⟨fun s i hs hi h => by
        exfalso; rw [hflags] at h; simp only [opselLT] at h; omega,
      fun i hi h => by exfalso; rw [hflags] at h; simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by
        exfalso; rw [hflags] at h; simp only [opselSHIFT] at h; omega,
      fun h => by exfalso; rw [hflags] at h; simp only [opselAND] at h; omega,
      fun h => by exfalso; rw [hflags] at h; simp only [opselOR] at h; omega,
      fun h => by exfalso; rw [hflags] at h; simp only [opselXOR] at h; omega, ?_⟩
    intro ls rs hgh hls hrs hh h
    rw [hflags] at h
    simp only [opselMUL] at h
    obtain ⟨rfl, rfl, rfl⟩ : ls = input_lhsSigned.val ∧ rs = input_rhsSigned.val ∧ hgh = 1 := by
      omega
    refine ⟨by omega, by omega, ?_⟩
    rw [hLval, hRval, hmul1, hv2, hv3]
    simp only [MulCore.ofHalves4]
    ring

/-- **The honest carries, on naturals.** From the limb bounds, the two sign bits and the product
equation an honest row satisfies, the four carries of the chain exist, each below `2^20` — the
range `IS_B20` checks. The four `raw` limbs enter abstractly, as the values `R0 … R3` of the
`raw_product` columns, which is the form `Honest` hands over. -/
theorem carries_of_product {a0 a1 a2 a3 b0 b1 b2 b3 A B ls rs o0 o1 o2 o3 q0 q1 q2 q3
    R0 R1 R2 R3 : ℕ}
    (ha0 : a0 < 2^16) (ha1 : a1 < 2^16) (ha2 : a2 < 2^16) (ha3 : a3 < 2^16)
    (hb0 : b0 < 2^16) (hb1 : b1 < 2^16) (hb2 : b2 < 2^16) (hb3 : b3 < 2^16)
    (ho0 : o0 < 2^16) (ho1 : o1 < 2^16) (ho2 : o2 < 2^16) (ho3 : o3 < 2^16)
    (hq0 : q0 < 2^16) (hq1 : q1 < 2^16) (_hq2 : q2 < 2^16) (_hq3 : q3 < 2^16)
    (hls : ls < 2) (hrs : rs < 2)
    (hA : A = if ls = 1 ∧ 2^63 ≤ a0 + 2^16 * a1 + 2^32 * a2 + 2^48 * a3 then 1 else 0)
    (hB : B = if rs = 1 ∧ 2^63 ≤ b0 + 2^16 * b1 + 2^32 * b2 + 2^48 * b3 then 1 else 0)
    (hR0 : R0 = MulCore.raw 0 a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
      b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B))
    (hR1 : R1 = MulCore.raw 1 a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
      b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B))
    (hR2 : R2 = MulCore.raw 2 a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
      b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B))
    (hR3 : R3 = MulCore.raw 3 a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
      b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B))
    (hprod : o0 + 2^16 * o1 + 2^32 * o2 + 2^48 * o3
          + 2^64 * (q0 + 2^16 * q1 + 2^32 * q2 + 2^48 * q3)
        = (Alu.ext128 ls (a0 + 2^16 * a1 + 2^32 * a2 + 2^48 * a3)
            * Alu.ext128 rs (b0 + 2^16 * b1 + 2^32 * b2 + 2^48 * b3)) % 2^128) :
    ∃ k0 k1 k2 k3 : ℕ, k0 < 2^20 ∧ k1 < 2^20 ∧ k2 < 2^20 ∧ k3 < 2^20 ∧
      R0 = o0 + 65536 * o1 + 2^32 * k0 ∧
      R1 + k0 = o2 + 65536 * o3 + 2^32 * k1 ∧
      R2 + k1 = q0 + 65536 * q1 + 2^32 * k2 ∧
      R3 + k2 = q2 + 65536 * q3 + 2^32 * k3 := by
  have hAb : A < 2 := by rw [hA]; split_ifs <;> omega
  have hBb : B < 2 := by rw [hB]; split_ifs <;> omega
  -- the sign-extended operands are the `ext128` of the `ALU` bus
  have hext_l : MulCore.ofHalves8 a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
      = Alu.ext128 ls (a0 + 2^16 * a1 + 2^32 * a2 + 2^48 * a3) := by
    have h := MulCore.ext_eq_ext128 (x := a0 + 2^16 * a1 + 2^32 * a2 + 2^48 * a3) (n := A)
      (s := ls) (by omega) hls hA
    rw [MulCore.ext_limbs]; exact h
  have hext_r : MulCore.ofHalves8 b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B)
      = Alu.ext128 rs (b0 + 2^16 * b1 + 2^32 * b2 + 2^48 * b3) := by
    have h := MulCore.ext_eq_ext128 (x := b0 + 2^16 * b1 + 2^32 * b2 + 2^48 * b3) (n := B)
      (s := rs) (by omega) hrs hB
    rw [MulCore.ext_limbs]; exact h
  -- so the product modulo `2^128` is the four `raw` limbs at their 32-bit positions
  have hmod := MulCore.product_mod_raw a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
    b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B)
  rw [hext_l, hext_r, ← hR0, ← hR1, ← hR2, ← hR3] at hmod
  rw [hmod] at hprod
  -- each limb is below `2^51`, which bounds every carry by `2^19`
  have bnd : ∀ i, MulCore.raw i a0 a1 a2 a3 (65535 * A) (65535 * A) (65535 * A) (65535 * A)
      b0 b1 b2 b3 (65535 * B) (65535 * B) (65535 * B) (65535 * B) < 2^51 :=
    fun i => MulCore.raw_lt ha0 ha1 ha2 ha3 (by omega) (by omega) (by omega) (by omega)
      hb0 hb1 hb2 hb3 (by omega) (by omega) (by omega) (by omega) i
  have g0 : R0 < 2^51 := by rw [hR0]; exact bnd 0
  have g1 : R1 < 2^51 := by rw [hR1]; exact bnd 1
  have g2 : R2 < 2^51 := by rw [hR2]; exact bnd 2
  have g3 : R3 < 2^51 := by rw [hR3]; exact bnd 3
  obtain ⟨e0, e1, e2, e3⟩ := MulCore.limbs_of_mod (R0 := R0) (R1 := R1) (R2 := R2) (R3 := R3)
    (res0 := o0 + 65536 * o1) (res1 := o2 + 65536 * o3)
    (res2 := q0 + 65536 * q1) (res3 := q2 + 65536 * q3)
    (by omega) (by omega) (by omega) (by omega) (by omega)
  exact ⟨R0 / 2^32, (R1 + R0 / 2^32) / 2^32, (R2 + (R1 + R0 / 2^32) / 2^32) / 2^32,
    (R3 + (R2 + (R1 + R0 / 2^32) / 2^32) / 2^32) / 2^32,
    by omega, by omega, by omega, by omega, by omega, by omega, by omega, by omega⟩

theorem completeness :
    GeneralFormalCircuit.Completeness (F p) (Input := Row) (Output := unit) main
      (fun r _ _ => Honest r) (fun _ _ _ => True) := by
  circuit_proof_start [main, Honest, Row.lhsVal, Row.rhsVal, Row.loVal, Row.hiVal,
    MulCore.ofHalves4, Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.isB20Table,
    Bitwise.msb16Table]
  obtain ⟨hmuLo, hmuHi, hlsb, hrsb, ha0, ha1, ha2, ha3, hb0, hb1, hb2, hb3,
    ho0, ho1, ho2, ho3, hq0, hq1, hq2, hq3, hA, hB, hraw0, hraw1, hraw2, hraw3,
    hprod⟩ := h_assumptions
  have hls2 := IsBool.val_lt_two hlsb
  have hrs2 := IsBool.val_lt_two hrsb
  -- the honest carries, on naturals
  obtain ⟨k0, k1, k2, k3, hk0, hk1, hk2, hk3, e0, e1, e2, e3⟩ :=
    carries_of_product ha0 ha1 ha2 ha3 hb0 hb1 hb2 hb3 ho0 ho1 ho2 ho3 hq0 hq1 hq2 hq3
      hls2 hrs2 hA hB hraw0 hraw1 hraw2 hraw3 hprod
  -- the same values in the field
  have hw0c : input_raw0 = ((input_raw0.val : ℕ) : F p) := (ZMod.natCast_zmod_val _).symm
  have hw1c : input_raw1 = ((input_raw1.val : ℕ) : F p) := (ZMod.natCast_zmod_val _).symm
  have hw2c : input_raw2 = ((input_raw2.val : ℕ) : F p) := (ZMod.natCast_zmod_val _).symm
  have hw3c : input_raw3 = ((input_raw3.val : ℕ) : F p) := (ZMod.natCast_zmod_val _).symm
  have hpair : ∀ a b : F p, a + 65536 * b = ((a.val + 65536 * b.val : ℕ) : F p) := by
    intro a b; push_cast; simp only [ZMod.natCast_zmod_val]
  -- the four virtual carries of the circuit are those naturals
  have hc0 := MulCore.carry_cast (r := input_raw0) (cin := 0) (Cin := 0) (k := k0)
    hw0c (by simp) (hpair input_lo0 input_lo1) (by omega)
  rw [add_zero] at hc0
  have hc1 := MulCore.carry_cast (k := k1) hw1c hc0 (hpair input_lo2 input_lo3) (by omega)
  have hc2 := MulCore.carry_cast (k := k2) hw2c hc1 (hpair input_hi0 input_hi1) (by omega)
  have hc3 := MulCore.carry_cast (k := k3) hw3c hc2 (hpair input_hi2 input_hi3) (by omega)
  -- `raw_product` as a cast of the natural convolution
  have hn0 := eq_natCast_of_val hraw0
  have hn1 := eq_natCast_of_val hraw1
  have hn2 := eq_natCast_of_val hraw2
  have hn3 := eq_natCast_of_val hraw3
  refine ⟨⟨hmuLo, hmuHi⟩,
    mul_one_sub_eq_zero_iff.mpr hlsb, mul_one_sub_eq_zero_iff.mpr hrsb,
    fun _ => ha0, fun _ => ha1, fun _ => ha2, fun _ => ha3,
    fun _ => hb0, fun _ => hb1, fun _ => hb2, fun _ => hb3,
    ?_, ?_, ?_, ?_,
    fun _ => ho0, fun _ => ho1, fun _ => ho2, fun _ => ho3,
    fun _ => hq0, fun _ => hq1, fun _ => hq2, fun _ => hq3,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · -- MSB16[lhs[3]] -> lhs_is_negative, when the operand is read as signed
    intro h1
    have hv : input_lhsSigned.val = 1 := by rw [h1, val_one_eq]
    have key : (if ZMod.val input_lhsSigned = 1 ∧ 2^63 ≤ ZMod.val input_lhs0
          + 2^16 * ZMod.val input_lhs1 + 2^32 * ZMod.val input_lhs2
          + 2^48 * ZMod.val input_lhs3 then 1 else 0) = ZMod.val input_lhs3 / 2^15 := by
      by_cases hge : 2^63 ≤ ZMod.val input_lhs0 + 2^16 * ZMod.val input_lhs1
          + 2^32 * ZMod.val input_lhs2 + 2^48 * ZMod.val input_lhs3
      · rw [ite_eq_left ⟨hv, hge⟩]; omega
      · rw [ite_eq_right fun h => hge h.2]; omega
    refine ⟨ha3, ?_⟩
    rw [hA]
    exact key
  · -- (1 − lhs_signed) · lhs_is_negative = 0
    rcases hlsb with h | h
    · have hz : input_lhsNeg.val = 0 := by rw [hA, h, ZMod.val_zero]; simp
      rw [(ZMod.val_eq_zero _).mp hz, mul_zero]
    · rw [h]; ring
  · -- MSB16[rhs[3]] -> rhs_is_negative
    intro h1
    have hv : input_rhsSigned.val = 1 := by rw [h1, val_one_eq]
    have key : (if ZMod.val input_rhsSigned = 1 ∧ 2^63 ≤ ZMod.val input_rhs0
          + 2^16 * ZMod.val input_rhs1 + 2^32 * ZMod.val input_rhs2
          + 2^48 * ZMod.val input_rhs3 then 1 else 0) = ZMod.val input_rhs3 / 2^15 := by
      by_cases hge : 2^63 ≤ ZMod.val input_rhs0 + 2^16 * ZMod.val input_rhs1
          + 2^32 * ZMod.val input_rhs2 + 2^48 * ZMod.val input_rhs3
      · rw [ite_eq_left ⟨hv, hge⟩]; omega
      · rw [ite_eq_right fun h => hge h.2]; omega
    refine ⟨hb3, ?_⟩
    rw [hB]
    exact key
  · -- (1 − rhs_signed) · rhs_is_negative = 0
    rcases hrsb with h | h
    · have hz : input_rhsNeg.val = 0 := by rw [hB, h, ZMod.val_zero]; simp
      rw [(ZMod.val_eq_zero _).mp hz, mul_zero]
    · rw [h]; ring
  · -- IS_B20 on the four carries
    intro _; rw [hc0, Bitwise.val_natCast_of_lt_two_pow_20 hk0]; exact hk0
  · intro _; rw [hc1, Bitwise.val_natCast_of_lt_two_pow_20 hk1]; exact hk1
  · intro _; rw [hc2, Bitwise.val_natCast_of_lt_two_pow_20 hk2]; exact hk2
  · intro _; rw [hc3, Bitwise.val_natCast_of_lt_two_pow_20 hk3]; exact hk3
  -- the four `raw_product` constraints
  · rw [hn0]; simp only [MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]; ring
  · rw [hn1]; simp only [MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]; ring
  · rw [hn2]; simp only [MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]; ring
  · rw [hn3]; simp only [MulCore.raw]; push_cast; simp only [ZMod.natCast_zmod_val]; ring

def circuit : GeneralFormalCircuit (F p) Row unit where
  name := "MUL"
  main
  elaborated
  Spec r _ _ := RowSpec r
  channelsWithRequirements :=
    [Alu.channel.toRaw, IsHalfC, IsB20C, Bitwise.Msb16.toRaw, Bitwise.AreBytes.toRaw]

  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨lhs0, lhs1, lhs2, lhs3, lhsSigned, rhs0, rhs1, rhs2, rhs3, rhsSigned,
      lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3, lhsNeg, rhsNeg,
      raw0, raw1, raw2, raw3, muLo, muHi⟩ := input_var
    simp only [circuit_norm, main, areBytes, isBit, sign, word, rawE, IsHalfC, IsB20C,
      Bitwise.IsHalf, Bitwise.IsB20, Bitwise.Msb16, Bitwise.AreBytes, Alu.channel]
    intro channel h
    tauto

  soundness := soundness
  ProverAssumptions r _ _ := Honest r
  completeness := completeness

end LambdaVM.MUL
