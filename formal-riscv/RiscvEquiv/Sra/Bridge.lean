import RiscvEquiv.Sra.Sail
import LambdaVM.Buses.Alu

/-!
# SRA: the Clean model's shift function is Sail's

With `word = 0`, `signed = 1` and `direction = 1`, `Alu.shiftResult` is `Alu.sra64`, which the
Clean model defines on naturals: a non-negative `x` (`x < 2^63`) is shifted as is, a negative
one is first sign-extended to 128 bits (`x + 2^128 − 2^64`) and the result is cut back to
64 bits. Lean's `BitVec.sshiftRight` instead complements, shifts and complements back. This file
proves the two agree, so `sraPure` read as a natural is `shiftResult`.
-/

namespace RiscvEquiv

open LambdaVM

/-- Shifting the 128-bit complement of `y` right by `k` bits and cutting to 64 bits gives the
64-bit complement of `y` shifted by `k`. This is the arithmetic behind the negative case. -/
lemma complement_div_two_pow (y k : ℕ) (hy : y < 2^64) (hk : k ≤ 64) :
    ((2^128 - 1 - y) / 2^k) % 2^64 = 2^64 - 1 - y / 2^k := by
  have hD : 0 < 2^k := by positivity
  have hDle : 2^k ≤ 2^64 := Nat.pow_le_pow_right (by norm_num) hk
  have hDE : 2^k * 2^(64 - k) = 2^64 := by rw [← pow_add]; congr 1; omega
  have hE : 1 ≤ 2^(64 - k) := Nat.one_le_two_pow
  have hyqr := Nat.div_add_mod y (2^k)
  have hr := Nat.mod_lt y hD
  have hq : y / 2^k ≤ y := Nat.div_le_self _ _
  -- The quotient is `2^128 / 2^k − 1 − y / 2^k`.
  have hdiv : (2^128 - 1 - y) / 2^k = 2^64 * 2^(64 - k) - 1 - y / 2^k := by
    have e : (2^64 * 2^(64 - k) - 1 - y / 2^k) * 2^k
        = 2^64 * 2^64 - 2^k - 2^k * (y / 2^k) := by
      rw [Nat.sub_mul, Nat.sub_mul, one_mul, mul_assoc, mul_comm (2^(64-k)) (2^k), hDE,
        mul_comm (y / 2^k)]
    apply Nat.div_eq_of_lt_le
    · rw [e]
      generalize 2^k * (y / 2^k) = P at *
      generalize y % 2^k = r at *
      generalize 2^k = D at *
      omega
    · rw [Nat.add_mul, one_mul, e]
      generalize 2^k * (y / 2^k) = P at *
      generalize y % 2^k = r at *
      generalize 2^k = D at *
      omega
  rw [hdiv]
  -- Modulo `2^64` only the low part survives.
  have hsplit : 2^64 * 2^(64 - k) - 1 - y / 2^k
      = (2^64 - 1 - y / 2^k) + 2^64 * (2^(64 - k) - 1) := by
    rw [Nat.mul_sub_one]
    generalize 2^(64 - k) = E at *
    have : 2^64 ≤ 2^64 * E := Nat.le_mul_of_pos_right _ hE
    omega
  rw [hsplit, Nat.add_mul_mod_self_left,
    Nat.mod_eq_of_lt (lt_of_le_of_lt (Nat.sub_le _ _) (by norm_num))]

/-- Lean's arithmetic right shift, read as a natural, is the Clean model's `sra64`. -/
theorem sshiftRight_toNat (a : BitVec 64) (k : ℕ) (hk : k < 64) :
    (a.sshiftRight k).toNat = Alu.sra64 a.toNat k := by
  have ha := a.isLt
  rw [BitVec.toNat_sshiftRight, BitVec.msb_eq_decide, Alu.sra64]
  by_cases h : a.toNat < 2^63
  · have h' : ¬ (2^(64-1) ≤ a.toNat) := by omega
    simp only [decide_eq_true_eq]
    rw [if_neg h', if_pos h]
  · have h' : 2^(64-1) ≤ a.toNat := by omega
    simp only [decide_eq_true_eq]
    rw [if_pos h', if_neg h]
    have e : a.toNat + (2^128 - 2^64) = 2^128 - 1 - (2^64 - 1 - a.toNat) := by omega
    rw [e, Nat.shiftRight_eq_div_pow, Nat.shiftRight_eq_div_pow,
      complement_div_two_pow _ _ (by omega) (by omega)]

/-- `shiftResult` with the flags of `SRA`: `sra64` of `x` by `s mod 64`. -/
lemma shiftResult_sra (x s : ℕ) : Alu.shiftResult 0 1 1 x s = Alu.sra64 x (s % 64) := by
  simp [Alu.shiftResult]

/-- `sraPure` as a natural is `shiftResult` with the flags of `SRA`. -/
theorem sraPure_toNat (a b : BitVec 64) :
    (sraPure a b).toNat = Alu.shiftResult 0 1 1 a.toNat b.toNat := by
  rw [shiftResult_sra, sraPure, sshiftRight_toNat _ _ (Nat.mod_lt _ (by norm_num))]

end RiscvEquiv
