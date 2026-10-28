module

public import LambdaVM.Tables.Bitwise
public import LambdaVM.Buses.Alu
public import LambdaVM.Templates.IsBit
public import LambdaVM.Templates.Add
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# EQ (spec/eq.typ)

One row per equality test `res = (a = b) xor invert`. Columns follow `prover/src/tables/eq.rs`:
`a`, `b` as `DWordWL` (two words), `diff = a − b` as four halves, the `eq` bit, `res`, `μ`.

`diff` is enforced with the SUB template (the ADD carries of `b + diff = a`), its halves are
range-checked, and `eq` is the ZERO lookup of their sum: four halves sum to zero iff all are
zero iff `a = b`.
-/

namespace LambdaVM.EQ
open Alu (eqBit)

/-- One EQ row, fields in the column order of `eq.rs` and named as in `spec/src/eq.toml`:
`a`, `b` as `DWordWL` words, `diff = a − b` as `DWordHL` halves. -/
structure Row (F : Type) where
  a0 : F
  a1 : F
  b0 : F
  b1 : F
  invert : F
  res : F
  diff0 : F
  diff1 : F
  diff2 : F
  diff3 : F
  eq : F
  mu : F
deriving ProvableStruct

/-- The chip's arithmetic on naturals: with `b + diff = a (mod 2⁶⁴)` and everything in range,
the halves of `diff` sum to zero exactly when `a = b`. -/
theorem core {a0 a1 b0 b1 d0 d1 d2 d3 c0 c1 : ℕ}
    (ha0 : a0 < 2^32) (ha1 : a1 < 2^32) (hb0 : b0 < 2^32) (hb1 : b1 < 2^32)
    (hd0 : d0 < 2^16) (hd1 : d1 < 2^16) (hd2 : d2 < 2^16) (hd3 : d3 < 2^16)
    (hc0 : c0 < 2) (hc1 : c1 < 2)
    (e0 : b0 + (d0 + 65536 * d1) = a0 + c0 * 2^32)
    (e1 : b1 + (d2 + 65536 * d3) + c0 = a1 + c1 * 2^32) :
    d0 + d1 + d2 + d3 = 0 ↔ a0 + 2^32 * a1 = b0 + 2^32 * b1 := by
  have h64 : b0 + 2^32 * b1 + (d0 + 65536 * d1 + 2^32 * (d2 + 65536 * d3))
      = a0 + 2^32 * a1 + c1 * 2^64 := by omega
  rcases (by omega : c1 = 0 ∨ c1 = 1) with rfl | rfl
  · constructor <;> intro h <;> omega
  · constructor <;> intro h <;> omega

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  -- IS_HALF[diff[i]], IS_BIT<invert>
  Bitwise.IsHalf.pullIf r.mu r.diff0
  Bitwise.IsHalf.pullIf r.mu r.diff1
  Bitwise.IsHalf.pullIf r.mu r.diff2
  Bitwise.IsHalf.pullIf r.mu r.diff3
  isBit r.invert
  -- SUB<diff; a, b>: the carries of b + diff = a are bits
  let diffLo := r.diff0 + 65536 * r.diff1
  let diffHi := r.diff2 + 65536 * r.diff3
  sub ⟨r.a0, r.a1⟩ ⟨r.b0, r.b1⟩ ⟨diffLo, diffHi⟩
  -- ZERO[Σ diff[i]] -> eq
  Bitwise.Zero.pullIf r.mu (r.diff0 + r.diff1 + r.diff2 + r.diff3, r.eq)
  -- res = eq xor invert
  assertZero (r.res + 2 * r.eq * r.invert - r.eq - r.invert)
  -- μ : Bit (no deduplication)
  isBit r.mu
  -- output: ALU[a, b, opsel(EQ) + 64·invert] -> [res, 0]
  Alu.channel.pushIf r.mu ⟨r.a0, r.a1, r.b0, r.b1, 3 + 64 * r.invert, r.res, 0⟩

/-- The honest prover's row, as `generate_eq_trace` fills it. -/
def Honest (r : Row (F p)) : Prop :=
  IsBool r.mu ∧ IsBool r.invert ∧
  r.a0.val < 2^32 ∧ r.a1.val < 2^32 ∧ r.b0.val < 2^32 ∧ r.b1.val < 2^32 ∧
  r.diff0.val < 2^16 ∧ r.diff1.val < 2^16 ∧ r.diff2.val < 2^16 ∧ r.diff3.val < 2^16 ∧
  -- diff = a − b (mod 2⁶⁴)
  (dword r.b0 r.b1
      + (r.diff0.val + 65536 * r.diff1.val + 2^32 * (r.diff2.val + 65536 * r.diff3.val))) % 2^64
    = dword r.a0 r.a1 ∧
  r.eq.val = (if dword r.a0 r.a1 = dword r.b0 r.b1 then 1 else 0) ∧
  r.res.val = eqBit r.invert.val (dword r.a0 r.a1) (dword r.b0 r.b1)

/-- What an active row establishes: with range-checked operands, `res` is the equality bit. -/
def RowSpec (r : Row (F p)) : Prop :=
  r.mu = 1 → r.a0.val < 2^32 → r.a1.val < 2^32 → r.b0.val < 2^32 → r.b1.val < 2^32 →
    r.res = ((eqBit r.invert.val (dword r.a0 r.a1) (dword r.b0 r.b1) : ℕ) : F p)

def circuit : GeneralFormalCircuit (F p) Row unit where
  main
  Spec r _ _ := RowSpec r
  channelsWithRequirements := [Alu.channel.toRaw]
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨a0, a1, b0, b1, invert, res, diff0, diff1, diff2, diff3, eq, mu⟩ := input_var
    simp only [circuit_norm, main, isBit, sub, add, Add.carry0, Add.carry1, Alu.channel,
      Bitwise.isHalfTable, Bitwise.zeroTable]
    constructor
    · intro channel h; tauto
    · intro env h
      refine ⟨?_, ?_, ?_, ?_, ?_⟩ <;>
      · intro hne1 hne0
        exact (requirement_vacuous h.2.2.2.2 hne1 hne0).elim

  soundness := by
    circuit_proof_start [main, isBit, sub, add, Add.carry0, Add.carry1, RowSpec, dword, Alu.channel,
      Alu.Guarantees, Bitwise.isHalfTable, Bitwise.zeroTable]
    obtain ⟨h_d0, h_d1, h_d2, h_d3, h_i, h_c0, h_c1, h_z, h_res, h_mu⟩ := h_holds
    rcases mul_one_sub_eq_zero_iff.mp h_mu with h0 | h1
    · subst h0; simp
    subst h1
    have hib : IsBool input_invert := mul_one_sub_eq_zero_iff.mp h_i
    have hi_2 := IsBool.val_lt_two hib
    have hd0 := h_d0 rfl
    have hd1 := h_d1 rfl
    have hd2 := h_d2 rfl
    have hd3 := h_d3 rfl
    obtain ⟨-, heq⟩ := h_z rfl
    -- the equality bit, from the pulled guarantees and the SUB carries
    have key : input_a0.val < 2^32 → input_a1.val < 2^32 → input_b0.val < 2^32 →
        input_b1.val < 2^32 →
        input_res = ((eqBit input_invert.val (input_a0.val + 2^32 * input_a1.val)
          (input_b0.val + 2^32 * input_b1.val) : ℕ) : F p) := by
      intro ha0 ha1 hb0 hb1
      set c0 := (input_b0 + (input_diff0 + 65536 * input_diff1) - input_a0) * inv32 with hc0
      set c1 := (input_b1 + (input_diff2 + 65536 * input_diff3) + c0 - input_a1) * inv32 with hc1
      have hc0b : IsBool c0 := mul_one_sub_eq_zero_iff.mp h_c0
      have hc1b : IsBool c1 := mul_one_sub_eq_zero_iff.mp h_c1
      -- the halves of diff as words
      have hval_lo : (input_diff0 + 65536 * input_diff1).val
          = input_diff0.val + 65536 * input_diff1.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      have hval_hi : (input_diff2 + 65536 * input_diff3).val
          = input_diff2.val + 65536 * input_diff3.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      -- the carry identities, on naturals
      obtain ⟨e0, e1⟩ := Add.carries_lift hb0 hb1 (by omega) (by omega) ha0 ha1 hc0 hc1 hc0b hc1b
      rw [hval_lo] at e0
      rw [hval_hi] at e1
      have hcore := core ha0 ha1 hb0 hb1 hd0 hd1 hd2 hd3 (IsBool.val_lt_two hc0b)
        (IsBool.val_lt_two hc1b) e0 e1
      have hval_sum : (input_diff0 + input_diff1 + input_diff2 + input_diff3).val
          = input_diff0.val + input_diff1.val + input_diff2.val + input_diff3.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      have hz : input_diff0 + input_diff1 + input_diff2 + input_diff3 = 0 ↔
          input_a0.val + 2^32 * input_a1.val = input_b0.val + 2^32 * input_b1.val := by
        rw [← ZMod.val_eq_zero, hval_sum]; exact hcore
      have heq_cast : input_eq = ((if input_a0.val + 2^32 * input_a1.val
          = input_b0.val + 2^32 * input_b1.val then 1 else 0 : ℕ) : F p) := by
        rw [heq]
        simp only [hz]
        split_ifs <;> simp
      have hres' : input_res = input_eq + input_invert - 2 * input_eq * input_invert := by
        linear_combination h_res
      rcases eq_zero_or_one_of_val hi_2 with ⟨rfl, hiv⟩ | ⟨rfl, hiv⟩ <;>
      rw [hres', heq_cast, hiv] <;>
      simp only [eqBit] <;>
      split_ifs <;> (try norm_num) <;> omega
    refine ⟨fun _ => key, by simp, by simp, by simp, by simp, by simp, ?_⟩
    intro _ _ ha0 ha1 hb0 hb1
    -- the flag byte, read as a natural
    have hflagsV : (3 + 64 * input_invert : F p).val = 3 + 64 * input_invert.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    refine ⟨?_, ?_, ?_⟩
    · -- the LT clause does not apply: opsel(EQ) + 64·i ≠ opsel(LT) + 32·s + 64·i'
      intro s i hs hi hflags
      exfalso; rw [hflagsV] at hflags; simp only [opselLT] at hflags; omega
    swap
    · -- nor the SHIFT clause
      intro w s d hw hs hd hflags
      exfalso; rw [hflagsV] at hflags; simp only [opselSHIFT] at hflags; omega
    intro i hi hflags
    rw [hflagsV] at hflags; simp only [opselEQ] at hflags
    rw [key ha0 ha1 hb0 hb1, show input_invert.val = i by omega]

  ProverAssumptions r _ _ := Honest r

  completeness := by
    circuit_proof_start [main, isBit, sub, add, Add.carry0, Add.carry1, Alu.channel, Alu.Guarantees,
      Bitwise.isHalfTable, Bitwise.zeroTable, Honest, dword]
    obtain ⟨hmu, hib, ha0, ha1, hb0, hb1, hd0, hd1, hd2, hd3, hsum, heq, hres⟩ := h_assumptions
    -- the halves of diff as words
    have hval_lo : (input_diff0 + 65536 * input_diff1).val
        = input_diff0.val + 65536 * input_diff1.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hval_hi : (input_diff2 + 65536 * input_diff3).val
        = input_diff2.val + 65536 * input_diff3.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    -- the honest carries
    have hr0 : (input_diff0 + 65536 * input_diff1).val < 2^32 := by omega
    have hr1 : (input_diff2 + 65536 * input_diff3).val < 2^32 := by omega
    obtain ⟨k0, k1, hk0, hk1, e0, e1, hc0k, hc1k⟩ :=
      Add.carries_of_sum hb0 hb1 hr0 hr1 ha0 ha1 (by simp only [dword, hval_lo, hval_hi]; omega)
    rw [hval_lo] at e0
    rw [hval_hi] at e1
    have hval_sum : (input_diff0 + input_diff1 + input_diff2 + input_diff3).val
        = input_diff0.val + input_diff1.val + input_diff2.val + input_diff3.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    have hcore := core ha0 ha1 hb0 hb1 hd0 hd1 hd2 hd3 hk0 hk1 e0 e1
    have hz : input_diff0 + input_diff1 + input_diff2 + input_diff3 = 0 ↔
        input_a0.val + 2^32 * input_a1.val = input_b0.val + 2^32 * input_b1.val := by
      rw [← ZMod.val_eq_zero, hval_sum]; exact hcore
    have heq_cast : input_eq = ((if input_a0.val + 2^32 * input_a1.val
        = input_b0.val + 2^32 * input_b1.val then 1 else 0 : ℕ) : F p) :=
      (ZMod.natCast_zmod_val input_eq).symm.trans (congrArg (Nat.cast : ℕ → F p) heq)
    have hres_cast : input_res = ((eqBit input_invert.val (input_a0.val + 2^32 * input_a1.val)
        (input_b0.val + 2^32 * input_b1.val) : ℕ) : F p) := by
      rw [← hres, ZMod.natCast_zmod_val]
    refine ⟨fun _ => by omega, fun _ => by omega, fun _ => by omega, fun _ => by omega,
      mul_one_sub_eq_zero_iff.mpr hib, ?_, ?_, fun _ => ⟨by rw [hval_sum]; omega, ?_⟩, ?_,
      mul_one_sub_eq_zero_iff.mpr hmu⟩
    · rw [hc0k]; rcases (by omega : k0 = 0 ∨ k0 = 1) with rfl | rfl <;> simp
    · rw [hc1k]; rcases (by omega : k1 = 0 ∨ k1 = 1) with rfl | rfl <;> simp
    · rw [heq_cast]
      simp only [hz]
      split_ifs <;> simp
    · rw [hres_cast, heq_cast]
      rcases eq_zero_or_one_of_val (IsBool.val_lt_two hib) with ⟨rfl, hiv⟩ | ⟨rfl, hiv⟩ <;>
      rw [hiv] <;>
      simp only [eqBit] <;>
      split_ifs <;> (try norm_num) <;> omega

end LambdaVM.EQ
