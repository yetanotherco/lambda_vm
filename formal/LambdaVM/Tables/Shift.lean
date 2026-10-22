module

public import LambdaVM.Tables.Bitwise
public import LambdaVM.Buses.Alu
public import LambdaVM.Tables.ShiftCore
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# SHIFT (spec/shift.typ)

One row per shift. Columns follow `prover/src/tables/shift.rs`: `in` as four halves, the low
byte of the shift amount plus its upper limbs for the ALU bus, the three flags, `out` as two
words, and the scratch of the two-phase design: `bit_shift = shift mod 16` (or its complement
when shifting right), `zbs = [bit_shift = 0]`, the HWSL outputs `X[i], Y[i]`, the one-hot
`limb_shift`, and the sign bit.
-/

namespace LambdaVM.SHIFT
open Alu (dword shiftResult)
open ShiftCore (sll sllc limbs)

structure Row (F : Type) where
  in0 : F
  in1 : F
  in2 : F
  in3 : F
  shiftAmount : F
  direction : F
  signed : F
  wordInstr : F
  out0 : F
  out1 : F
  isNegative : F
  bitShift : F
  zbs : F
  x0 : F
  x1 : F
  x2 : F
  x3 : F
  x4 : F
  y0 : F
  y1 : F
  y2 : F
  y3 : F
  ls0 : F
  ls1 : F
  ls2 : F
  mu : F
  shiftB1 : F
  shiftH1 : F
  shiftHigh : F
deriving ProvableStruct

/-- The `shifted` limbs of the spec as one polynomial, over expressions or over field values:
`mu`, `d` (direction), the stored one-hot `l0 l1 l2`, the extension limb `e`, and the HWSL
outputs `x0..x4`, `y0..y3`. Kept folded during circuit proofs; `eval_shiftedPoly` pushes
evaluation inside. -/
def shiftedPoly {R : Type} [Add R] [Mul R] [Sub R] [OfNat R 1]
    (mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : R) : R × R × R × R :=
  let l3 := 1 - l0 - l1 - l2
  let left := mu - d
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

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  let left := r.mu - r.direction
  let right := r.direction
  let extension := 65535 * r.isNegative
  let ls3 := 1 - r.ls0 - r.ls1 - r.ls2
  let sh := shiftedPoly r.mu r.direction r.ls0 r.ls1 r.ls2 extension
    r.x0 r.x1 r.x2 r.x3 r.x4 r.y0 r.y1 r.y2 r.y3
  -- constraints, in the order of `ShiftConstraints::eval`
  assertZero (r.direction * (1 - r.mu))
  assertZero (r.zbs * (r.x0 - r.in0 * left))
  assertZero (r.zbs * (r.x1 - r.in1 * left))
  assertZero (r.zbs * (r.x2 - r.in2 * left))
  assertZero (r.zbs * (r.x3 - r.in3 * left))
  assertZero (r.zbs * r.x4)
  assertZero (r.zbs * (r.y0 - r.in0 * right))
  assertZero (r.zbs * (r.y1 - r.in1 * right))
  assertZero (r.zbs * (r.y2 - r.in2 * right))
  assertZero (r.zbs * (r.y3 - r.in3 * right))
  assertZero (r.ls0 * (1 - r.ls0))
  assertZero (r.ls1 * (1 - r.ls1))
  assertZero (r.ls2 * (1 - r.ls2))
  assertZero (ls3 * (1 - ls3))
  assertZero (r.out0 - sh.1 - sh.2.1 * 65536)
  assertZero (r.out1 - sh.2.2.1 - sh.2.2.2 * 65536)
  assertZero (r.direction * (1 - r.direction))
  assertZero (r.signed * (1 - r.signed))
  assertZero (r.wordInstr * (1 - r.wordInstr))
  assertZero (r.zbs * (1 - r.zbs))
  assertZero (r.mu * (1 - r.mu))
  assertZero ((1 - r.signed) * r.isNegative)
  -- interactions, in the order of `bus_interactions`
  Bitwise.Msb16.pullIf r.signed (r.in3, r.isNegative)
  Bitwise.ByteAlu.pullIf left ⟨0, r.shiftAmount, 15, r.bitShift⟩
  Bitwise.ByteAlu.pullIf right ⟨0, 256 - 16 * r.zbs - r.shiftAmount, 15, r.bitShift⟩
  Bitwise.Zero.pullIf r.mu (r.bitShift, r.zbs)
  Bitwise.Hwsl.pullIf (1 - r.zbs) ⟨r.in0, r.bitShift, r.x0, r.y0⟩
  Bitwise.Hwsl.pullIf (1 - r.zbs) ⟨r.in1, r.bitShift, r.x1, r.y1⟩
  Bitwise.Hwsl.pullIf (1 - r.zbs) ⟨r.in2, r.bitShift, r.x2, r.y2⟩
  Bitwise.Hwsl.pullIf (1 - r.zbs) ⟨r.in3, r.bitShift, r.x3, r.y3⟩
  Bitwise.Hwsl.pullIf (1 - r.zbs) ⟨extension, r.bitShift, r.x4, extension - r.x4⟩
  Bitwise.ByteAlu.pullIf r.mu
    ⟨0, r.shiftAmount, 48 - 32 * r.wordInstr, 48 - 48 * r.ls0 - 32 * r.ls1 - 16 * r.ls2⟩
  Alu.channel.pushIf r.mu
    ⟨r.in0 + 65536 * r.in1, r.in2 + 65536 * r.in3,
      r.shiftAmount + 256 * r.shiftB1 + 65536 * r.shiftH1, r.shiftHigh,
      5 + r.wordInstr + 32 * r.signed + 64 * r.direction, r.out0, r.out1⟩
  Bitwise.AreBytes.pull (r.shiftAmount, r.shiftB1)
  Bitwise.IsHalf.pullIf r.mu r.shiftH1
  Bitwise.IsHalf.pullIf r.mu r.in0
  Bitwise.IsHalf.pullIf r.mu r.in1
  Bitwise.IsHalf.pullIf r.mu r.in2
  Bitwise.IsHalf.pullIf r.mu r.in3

lemma eval_shiftedPoly_1 (env : Environment (F p))
    (mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : Expression (F p)) :
    Expression.eval env (shiftedPoly mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3).1
    = (shiftedPoly (env mu) (env d) (env l0) (env l1) (env l2) (env e) (env x0) (env x1) (env x2)
        (env x3) (env x4) (env y0) (env y1) (env y2) (env y3)).1 := by
  simp only [shiftedPoly, circuit_norm]
lemma eval_shiftedPoly_2 (env : Environment (F p))
    (mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : Expression (F p)) :
    Expression.eval env (shiftedPoly mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3).2.1
    = (shiftedPoly (env mu) (env d) (env l0) (env l1) (env l2) (env e) (env x0) (env x1) (env x2)
        (env x3) (env x4) (env y0) (env y1) (env y2) (env y3)).2.1 := by
  simp only [shiftedPoly, circuit_norm]
lemma eval_shiftedPoly_3 (env : Environment (F p))
    (mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : Expression (F p)) :
    Expression.eval env (shiftedPoly mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3).2.2.1
    = (shiftedPoly (env mu) (env d) (env l0) (env l1) (env l2) (env e) (env x0) (env x1) (env x2)
        (env x3) (env x4) (env y0) (env y1) (env y2) (env y3)).2.2.1 := by
  simp only [shiftedPoly, circuit_norm]
lemma eval_shiftedPoly_4 (env : Environment (F p))
    (mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : Expression (F p)) :
    Expression.eval env (shiftedPoly mu d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3).2.2.2
    = (shiftedPoly (env mu) (env d) (env l0) (env l1) (env l2) (env e) (env x0) (env x1) (env x2)
        (env x3) (env x4) (env y0) (env y1) (env y2) (env y3)).2.2.2 := by
  simp only [shiftedPoly, circuit_norm]

/-- Over field values that are casts of naturals (with `μ = 1`), `shiftedPoly` is the cast of
`ShiftCore.shifted`. -/
lemma shiftedPoly_cast {d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3 : ℕ} (hd : d ≤ 1)
    (hl : l0 + l1 + l2 ≤ 1) :
    let shF := shiftedPoly (1 : F p) (d : F p) (l0 : F p) (l1 : F p) (l2 : F p) (e : F p)
      (x0 : F p) (x1 : F p) (x2 : F p) (x3 : F p) (x4 : F p) (y0 : F p) (y1 : F p) (y2 : F p) (y3 : F p)
    let sh := ShiftCore.shifted d l0 l1 l2 e x0 x1 x2 x3 x4 y0 y1 y2 y3
    shF.1 = (sh.1 : F p) ∧ shF.2.1 = (sh.2.1 : F p) ∧ shF.2.2.1 = (sh.2.2.1 : F p) ∧
      shF.2.2.2 = (sh.2.2.2 : F p) := by
  intro shF sh
  have h1 : l0 ≤ 1 := by omega
  have h2 : l1 ≤ 1 - l0 := by omega
  have h3 : l2 ≤ 1 - l0 - l1 := by omega
  simp only [shF, sh, shiftedPoly, ShiftCore.shifted]
  push_cast [Nat.cast_sub hd, Nat.cast_sub h1, Nat.cast_sub h2, Nat.cast_sub h3]
  refine ⟨by ring, by ring, by ring, by ring⟩

/-- What an active row establishes: `out` is `in` shifted by the low byte of the amount. -/
def RowSpec (r : Row (F p)) : Prop :=
  r.mu = 1 →
    r.out0.val < 2^32 ∧ r.out1.val < 2^32 ∧
    r.out0.val + 2^32 * r.out1.val
      = shiftResult r.wordInstr.val r.signed.val r.direction.val
          (ShiftCore.limbs r.in0.val r.in1.val r.in2.val r.in3.val) r.shiftAmount.val

/-- The `shifted` limbs of a row, on naturals. -/
def shiftedOf (r : Row (F p)) : ℕ × ℕ × ℕ × ℕ :=
  ShiftCore.shifted r.direction.val r.ls0.val r.ls1.val r.ls2.val (65535 * r.isNegative.val)
    r.x0.val r.x1.val r.x2.val r.x3.val r.x4.val r.y0.val r.y1.val r.y2.val r.y3.val

/-- The honest prover's active row, as `generate_shift_trace` fills it (`compute_aux`). -/
def Honest (r : Row (F p)) : Prop :=
  r.mu = 1 ∧ IsBool r.direction ∧ IsBool r.signed ∧ IsBool r.wordInstr ∧
  r.in0.val < 65536 ∧ r.in1.val < 65536 ∧ r.in2.val < 65536 ∧ r.in3.val < 65536 ∧
  r.shiftAmount.val < 256 ∧ r.shiftB1.val < 256 ∧ r.shiftH1.val < 65536 ∧
  r.isNegative.val = (if r.signed.val = 1 then r.in3.val / 32768 else 0) ∧
  r.bitShift.val = (if r.direction.val = 0 then r.shiftAmount.val % 16
    else (16 - r.shiftAmount.val % 16) % 16) ∧
  r.zbs = (if r.bitShift = 0 then 1 else 0) ∧
  (r.x0.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then r.in0.val else 0)
    else sll r.in0.val r.bitShift.val) ∧
  (r.x1.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then r.in1.val else 0)
    else sll r.in1.val r.bitShift.val) ∧
  (r.x2.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then r.in2.val else 0)
    else sll r.in2.val r.bitShift.val) ∧
  (r.x3.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then r.in3.val else 0)
    else sll r.in3.val r.bitShift.val) ∧
  (r.x4.val = if r.bitShift.val = 0 then 0 else sll (65535 * r.isNegative.val) r.bitShift.val) ∧
  (r.y0.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then 0 else r.in0.val)
    else sllc r.in0.val r.bitShift.val) ∧
  (r.y1.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then 0 else r.in1.val)
    else sllc r.in1.val r.bitShift.val) ∧
  (r.y2.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then 0 else r.in2.val)
    else sllc r.in2.val r.bitShift.val) ∧
  (r.y3.val = if r.bitShift.val = 0 then (if r.direction.val = 0 then 0 else r.in3.val)
    else sllc r.in3.val r.bitShift.val) ∧
  -- limb_shift: one-hot at (shift / 16) mod (4 or 2)
  (r.ls0.val = if r.shiftAmount.val / 16 % (4 - 2 * r.wordInstr.val) = 0 then 1 else 0) ∧
  (r.ls1.val = if r.shiftAmount.val / 16 % (4 - 2 * r.wordInstr.val) = 1 then 1 else 0) ∧
  (r.ls2.val = if r.shiftAmount.val / 16 % (4 - 2 * r.wordInstr.val) = 2 then 1 else 0) ∧
  -- out = shifted, as two words
  r.out0.val = (shiftedOf r).1 + 65536 * (shiftedOf r).2.1 ∧
  r.out1.val = (shiftedOf r).2.2.1 + 65536 * (shiftedOf r).2.2.2

/-- A padding row: everything zero except `zbs = 1`. -/
def Padding (r : Row (F p)) : Prop :=
  r.mu = 0 ∧ r.direction = 0 ∧ r.signed = 0 ∧ r.wordInstr = 0 ∧ r.zbs = 1 ∧
  r.isNegative = 0 ∧ r.x0 = 0 ∧ r.x1 = 0 ∧ r.x2 = 0 ∧ r.x3 = 0 ∧ r.x4 = 0 ∧
  r.y0 = 0 ∧ r.y1 = 0 ∧ r.y2 = 0 ∧ r.y3 = 0 ∧ r.ls0 = 0 ∧ r.ls1 = 0 ∧ r.ls2 = 0 ∧
  r.out0 = 0 ∧ r.out1 = 0 ∧ r.shiftAmount = 0 ∧ r.shiftB1 = 0

/-- `HWSL[e, b]`'s second output for an extension limb: `e − sll e b` is `sllc e b`. -/
lemma ext_sllc {e b : ℕ} (he : e = 0 ∨ e = 65535) (hb : b < 16) :
    sll e b ≤ e ∧ e - sll e b = sllc e b := by
  rcases he with rfl | rfl <;> interval_cases b <;> simp [sll, sllc]

/-- The `zbs` override of an `X` scratch cell holds on an honest row. -/
lemma override_x {zbs x i d : F p} {b : ℕ} (hzb : IsBool zbs) (hdb : IsBool d)
    (hiff : zbs = 1 → b = 0)
    (hx : x.val = if b = 0 then (if d.val = 0 then i.val else 0) else sll i.val b) :
    zbs * (x - i * (1 - d)) = 0 := by
  rcases hzb with hz | hz
  · rw [hz, zero_mul]
  · rw [if_pos (hiff hz)] at hx
    rw [hz, one_mul, sub_eq_zero]
    rcases hdb with hd | hd
    · rw [hd] at hx ⊢
      simp only [ZMod.val_zero, if_true] at hx
      rw [ZMod.val_injective _ hx]; ring
    · rw [hd] at hx ⊢
      simp only [val_one_eq, one_ne_zero, if_false] at hx
      rw [(ZMod.val_eq_zero _).mp hx]; ring

/-- The `zbs` override of a `Y` scratch cell holds on an honest row. -/
lemma override_y {zbs y i d : F p} {b : ℕ} (hzb : IsBool zbs) (hdb : IsBool d)
    (hiff : zbs = 1 → b = 0)
    (hy : y.val = if b = 0 then (if d.val = 0 then 0 else i.val) else sllc i.val b) :
    zbs * (y - i * d) = 0 := by
  rcases hzb with hz | hz
  · rw [hz, zero_mul]
  · rw [if_pos (hiff hz)] at hy
    rw [hz, one_mul, sub_eq_zero]
    rcases hdb with hd | hd
    · rw [hd] at hy ⊢
      simp only [ZMod.val_zero, if_true] at hy
      rw [(ZMod.val_eq_zero _).mp hy]; ring
    · rw [hd] at hy ⊢
      simp only [val_one_eq, one_ne_zero, if_false] at hy
      rw [ZMod.val_injective _ hy]; ring

@[reducible]
instance elaborated : ElaboratedCircuit (F p) Row unit main := by elaborate_circuit

theorem soundness :
    GeneralFormalCircuit.Soundness (F p) (Input := Row) (Output := unit) main (fun _ _ => True)
      (fun r _ _ => RowSpec r) := by
    circuit_proof_start [main, RowSpec, Alu.channel, Alu.Guarantees, Bitwise.isHalfTable,
      Bitwise.msb16Table, Bitwise.byteAluTable, Bitwise.zeroTable, Bitwise.hwslTable,
      Bitwise.areBytesTable]
    obtain ⟨h_dm, h_zx0, h_zx1, h_zx2, h_zx3, h_zx4, h_zy0, h_zy1, h_zy2, h_zy3,
      h_ls0, h_ls1, h_ls2, h_ls3, h_out0, h_out1, h_dir, h_sg, h_w, h_zbs, h_mu, h_neg,
      h_msb, h_balL, h_balR, h_zero, h_hw0, h_hw1, h_hw2, h_hw3, h_hw4, h_lsb,
      h_ab, h_h1, h_in0, h_in1, h_in2, h_in3⟩ := h_holds
    have hdb := isBool_of_mul_one_sub h_dir
    have hsb := isBool_of_mul_one_sub h_sg
    have hwb := isBool_of_mul_one_sub h_w
    have hzb := isBool_of_mul_one_sub h_zbs
    have hmb := isBool_of_mul_one_sub h_mu
    have hl0b := isBool_of_mul_one_sub h_ls0
    have hl1b := isBool_of_mul_one_sub h_ls1
    have hl2b := isBool_of_mul_one_sub h_ls2
    have hl3b := isBool_of_mul_one_sub h_ls3
    have hleft : IsBool (input_mu - input_direction) := by
      rcases hmb with h | h <;> rcases hdb with h' | h' <;> simp_all [IsBool]
    have hzb' : IsBool (1 - input_zbs) := by
      rcases hzb with h | h <;> simp [h, IsBool]
    rcases hmb with hmu0 | hmu1
    · -- μ = 0: nothing is required of the row
      rw [hmu0] at hleft ⊢
      refine ⟨fun h => absurd h (by norm_num), ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
        ?_, ?_⟩
      all_goals first
        | exact req_vacuous hsb
        | exact req_vacuous hleft
        | exact req_vacuous hdb
        | exact req_vacuous hzb'
        | (intro _ h0; exact absurd rfl h0)
    subst hmu1
    -- the facts every active row has
    have hd2 := IsBool.val_lt_two hdb
    have hs2 := IsBool.val_lt_two hsb
    have hw2 := IsBool.val_lt_two hwb
    have hz2 := IsBool.val_lt_two hzb
    have hl0 := IsBool.val_lt_two hl0b
    have hl1 := IsBool.val_lt_two hl1b
    have hl2 := IsBool.val_lt_two hl2b
    have honehot := onehot_of_isBool hl0b hl1b hl2b hl3b
    obtain ⟨hsA, hB1⟩ := h_ab
    obtain ⟨-, hzbs_eq⟩ := h_zero rfl
    have hH1 := h_h1 rfl
    have hi0 := h_in0 rfl
    have hi1 := h_in1 rfl
    have hi2 := h_in2 rfl
    have hi3 := h_in3 rfl
    have h15 : (15 : F p).val = 15 := val_ofNat 15 (by norm_num)
    have h02ne : (0 : F p) ≠ 2 := by
      intro h
      have := congrArg ZMod.val h
      rw [ZMod.val_zero, val_ofNat 2 (by norm_num)] at this
      omega
    -- zbs is the zero flag of bit_shift
    have hzb_iff : input_zbs.val = 1 ↔ input_bitShift.val = 0 := by
      by_cases hb0 : input_bitShift = 0
      · rw [hzbs_eq, if_pos hb0, val_one_eq, hb0, ZMod.val_zero]; simp
      · rw [hzbs_eq, if_neg hb0, ZMod.val_zero]
        have : input_bitShift.val ≠ 0 := by rwa [Ne, ZMod.val_eq_zero]
        omega
    -- bit_shift from the BYTE_ALU lookups
    have hb : input_bitShift.val = if input_direction.val = 0 then input_shiftAmount.val % 16
        else (16 - input_shiftAmount.val % 16) % 16 := by
      rcases hdb with hd0 | hd1
      · rw [hd0]; simp only [ZMod.val_zero, if_true]
        obtain ⟨-, -, hand | ⟨h02, -⟩⟩ := h_balL (by rw [hd0]; norm_num)
        · rw [hand, h15, ShiftCore.and15]
        · exact absurd h02 h02ne
      · rw [hd1]; simp only [val_one_eq, one_ne_zero, if_false]
        obtain ⟨hx, -, hand | ⟨h02, -⟩⟩ := h_balR (by rw [hd1])
        · rw [h15, ShiftCore.and15] at hand
          rw [← ZMod.natCast_zmod_val input_shiftAmount] at hx hand
          rcases hzb with hz0 | hz1
          · rw [hz0] at hx hand
            simp only [mul_zero, sub_zero] at hx hand
            rw [show (256 : F p) = ((256 : ℕ) : F p) by norm_num] at hx hand
            obtain ⟨hle, hval⟩ := val_sub_of_val_lt (by norm_num) (by omega) (lt_trans hx (by norm_num))
            rw [hval] at hand
            omega
          · rw [hz1] at hx hand
            rw [show (256 : F p) - 16 * 1 = ((240 : ℕ) : F p) by norm_num] at hx hand
            obtain ⟨hle, hval⟩ := val_sub_of_val_lt (by norm_num) (by omega) (lt_trans hx (by norm_num))
            rw [hval] at hand
            omega
        · exact absurd h02 h02ne
    -- the sign bit
    have hn : input_isNegative.val = if input_signed.val = 1 then input_in3.val / 32768 else 0 := by
      rcases hsb with hs0 | hs1
      · rw [hs0] at h_neg ⊢
        simp only [ZMod.val_zero, zero_ne_one, if_false]
        have : input_isNegative = 0 := by simpa using h_neg
        rw [this, ZMod.val_zero]
      · rw [hs1]; simp only [val_one_eq, if_true]
        exact (h_msb (by rw [hs1])).2
    have hn2 : input_isNegative.val < 2 := by rw [hn]; split_ifs <;> omega
    have hext : (65535 * input_isNegative).val = 65535 * input_isNegative.val :=
      val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
    -- the HWSL outputs, or their overrides when bit_shift = 0
    have hxy : (input_x0.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then input_in0.val else 0)
          else sll input_in0.val input_bitShift.val) ∧
        (input_x1.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then input_in1.val else 0)
          else sll input_in1.val input_bitShift.val) ∧
        (input_x2.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then input_in2.val else 0)
          else sll input_in2.val input_bitShift.val) ∧
        (input_x3.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then input_in3.val else 0)
          else sll input_in3.val input_bitShift.val) ∧
        (input_x4.val = if input_bitShift.val = 0 then 0
          else sll (65535 * input_isNegative.val) input_bitShift.val) ∧
        (input_y0.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then 0 else input_in0.val)
          else sllc input_in0.val input_bitShift.val) ∧
        (input_y1.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then 0 else input_in1.val)
          else sllc input_in1.val input_bitShift.val) ∧
        (input_y2.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then 0 else input_in2.val)
          else sllc input_in2.val input_bitShift.val) ∧
        (input_y3.val = if input_bitShift.val = 0 then (if input_direction.val = 0 then 0 else input_in3.val)
          else sllc input_in3.val input_bitShift.val) := by
      rcases hzb with hz0 | hz1
      · -- bit_shift ≠ 0: the lookups fix X and Y
        have hbne : input_bitShift.val ≠ 0 := by
          intro h; have := hzb_iff.mpr h; rw [hz0, ZMod.val_zero] at this; omega
        obtain ⟨-, -, hx0v, hy0v⟩ := h_hw0 (by rw [hz0]; norm_num)
        obtain ⟨-, -, hx1v, hy1v⟩ := h_hw1 (by rw [hz0]; norm_num)
        obtain ⟨-, -, hx2v, hy2v⟩ := h_hw2 (by rw [hz0]; norm_num)
        obtain ⟨-, -, hx3v, hy3v⟩ := h_hw3 (by rw [hz0]; norm_num)
        obtain ⟨-, -, hx4v, -⟩ := h_hw4 (by rw [hz0]; norm_num)
        rw [hext] at hx4v
        simp only [if_neg hbne]
        exact ⟨hx0v, hx1v, hx2v, hx3v, hx4v, hy0v, hy1v, hy2v, hy3v⟩
      · -- bit_shift = 0: the overrides
        have hb0 : input_bitShift.val = 0 := hzb_iff.mp (by rw [hz1, val_one_eq])
        rw [hz1] at h_zx0 h_zx1 h_zx2 h_zx3 h_zx4 h_zy0 h_zy1 h_zy2 h_zy3
        simp only [one_mul, sub_eq_zero] at h_zx0 h_zx1 h_zx2 h_zx3 h_zx4 h_zy0 h_zy1 h_zy2 h_zy3
        simp only [if_pos hb0]
        rcases hdb with hd0 | hd1
        · rw [hd0] at h_zx0 h_zx1 h_zx2 h_zx3 h_zy0 h_zy1 h_zy2 h_zy3 ⊢
          simp only [sub_zero, mul_one, mul_zero] at h_zx0 h_zx1 h_zx2 h_zx3 h_zy0 h_zy1 h_zy2 h_zy3
          simp only [ZMod.val_zero, if_true, h_zx0, h_zx1, h_zx2, h_zx3, h_zy0, h_zy1, h_zy2, h_zy3,
            h_zx4, and_self]
        · rw [hd1] at h_zx0 h_zx1 h_zx2 h_zx3 h_zy0 h_zy1 h_zy2 h_zy3 ⊢
          simp only [sub_self, mul_one, mul_zero] at h_zx0 h_zx1 h_zx2 h_zx3 h_zy0 h_zy1 h_zy2 h_zy3
          simp only [val_one_eq, one_ne_zero, if_false, h_zx0, h_zx1, h_zx2, h_zx3, h_zy0, h_zy1,
            h_zy2, h_zy3, h_zx4, ZMod.val_zero, and_self]
    obtain ⟨hx0, hx1, hx2, hx3, hx4, hy0, hy1, hy2, hy3⟩ := hxy
    -- the one-hot limb index from the BYTE_ALU lookup
    have hL : 16 * (input_ls1.val + 2 * input_ls2.val
          + 3 * (1 - input_ls0.val - input_ls1.val - input_ls2.val))
        = input_shiftAmount.val &&& (48 - 32 * input_wordInstr.val) := by
      obtain ⟨-, -, hand | ⟨h02, -⟩⟩ := h_lsb rfl
      · have hmask : (48 - 32 * input_wordInstr).val = 48 - 32 * input_wordInstr.val := by
          rcases hwb with hw0 | hw1
          · rw [hw0]; simp only [mul_zero, sub_zero, ZMod.val_zero]; exact val_ofNat 48 (by norm_num)
          · rw [hw1]; norm_num; exact val_ofNat 16 (by norm_num)
        have henc : (48 - 48 * input_ls0 - 32 * input_ls1 - 16 * input_ls2).val
            = 48 - 48 * input_ls0.val - 32 * input_ls1.val - 16 * input_ls2.val := by
          rcases hl0b with h0 | h0 <;> rcases hl1b with h1 | h1 <;> rcases hl2b with h2 | h2 <;>
          · rw [h0, h1, h2]
            simp only [h0, h1, h2, ZMod.val_zero, val_one_eq] at honehot ⊢
            first
            | omega
            | (norm_num; done)
            | (norm_num; exact val_ofNat _ (by norm_num))
        rw [henc, hmask] at hand
        omega
      · exact absurd h02 h02ne
    -- the arithmetic
    have hcore := ShiftCore.main hd2 hs2 hw2 hsA hi0 hi1 hi2 hi3 hl0 hl1 hl2 honehot hL hb hn
      hx0 hx1 hx2 hx3 hx4 hy0 hy1 hy2 hy3
    simp only at hcore
    obtain ⟨hsh0, hsh1, hsh2, hsh3, hres⟩ := hcore
    -- the output words
    have hcast := shiftedPoly_cast (e := 65535 * input_isNegative.val)
      (x0 := input_x0.val) (x1 := input_x1.val) (x2 := input_x2.val) (x3 := input_x3.val)
      (x4 := input_x4.val) (y0 := input_y0.val) (y1 := input_y1.val) (y2 := input_y2.val)
      (y3 := input_y3.val) (by omega : input_direction.val ≤ 1) honehot
    simp only at hcast
    obtain ⟨hc1, hc2, hc3, hc4⟩ := hcast
    rw [eval_shiftedPoly_1, eval_shiftedPoly_2] at h_out0
    rw [eval_shiftedPoly_3, eval_shiftedPoly_4] at h_out1
    simp only [circuit_norm, h_input] at h_out0 h_out1
    rw [← ZMod.natCast_zmod_val input_direction, ← ZMod.natCast_zmod_val input_ls0,
      ← ZMod.natCast_zmod_val input_ls1, ← ZMod.natCast_zmod_val input_ls2,
      ← ZMod.natCast_zmod_val input_isNegative, ← ZMod.natCast_zmod_val input_x0,
      ← ZMod.natCast_zmod_val input_x1, ← ZMod.natCast_zmod_val input_x2,
      ← ZMod.natCast_zmod_val input_x3, ← ZMod.natCast_zmod_val input_x4,
      ← ZMod.natCast_zmod_val input_y0, ← ZMod.natCast_zmod_val input_y1,
      ← ZMod.natCast_zmod_val input_y2, ← ZMod.natCast_zmod_val input_y3,
      show (65535 : F p) * ((input_isNegative.val : ℕ) : F p)
        = ((65535 * input_isNegative.val : ℕ) : F p) by push_cast; ring] at h_out0 h_out1
    rw [hc1, hc2] at h_out0
    rw [hc3, hc4] at h_out1
    have hout0 : input_out0.val = (ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
        input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
        input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).1
        + 65536 * (ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
        input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
        input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).2.1 :=
      val_eq_of_eq_natCast (by push_cast; linear_combination h_out0) (by omega)
    have hout1 : input_out1.val = (ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
        input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
        input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).2.2.1
        + 65536 * (ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
        input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
        input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).2.2.2 :=
      val_eq_of_eq_natCast (by push_cast; linear_combination h_out1) (by omega)
    have key : input_out0.val < 2^32 ∧ input_out1.val < 2^32 ∧
        input_out0.val + 2^32 * input_out1.val
          = shiftResult input_wordInstr.val input_signed.val input_direction.val
              (limbs input_in0.val input_in1.val input_in2.val input_in3.val)
              input_shiftAmount.val := by
      refine ⟨by omega, by omega, ?_⟩
      rw [← hres, hout0, hout1]
      simp only [limbs]
      omega
    -- the row spec and the bus requirements
    refine ⟨fun _ => key, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    all_goals first
      | exact req_vacuous hsb
      | exact req_vacuous hleft
      | exact req_vacuous hdb
      | exact req_vacuous hzb'
      | exact (fun h => absurd rfl h)
      | skip
    -- what remains is the ALU push
    intro _ _ hin1lo hin1hi hin2lo hin2hi
    have hflags_ne : ∀ (w s d : ℕ), w < 2 → s < 2 → d < 2 →
        5 + input_wordInstr + 32 * input_signed + 64 * input_direction
          = ((Alu.opselSHIFT + w + 32 * s + 64 * d : ℕ) : F p) →
        input_wordInstr.val = w ∧ input_signed.val = s ∧ input_direction.val = d := by
      intro w s d hw hs hd hflags
      have h1 : ((5 + input_wordInstr.val + 32 * input_signed.val + 64 * input_direction.val : ℕ) : F p)
          = ((Alu.opselSHIFT + w + 32 * s + 64 * d : ℕ) : F p) := by
        rw [← hflags]; push_cast; simp only [ZMod.natCast_zmod_val]
      have h2 := natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega))
        (lt_p_of_lt_two_pow_40 (by simp only [Alu.opselSHIFT]; omega)) h1
      simp only [Alu.opselSHIFT] at h2
      omega
    refine ⟨?_, ?_, ?_⟩
    · -- not LT: 5 + w + 32 s + 64 d ≠ 4 + 32 s' + 64 i (mod 32)
      intro s i hs hi hflags
      exfalso
      have h1 : ((5 + input_wordInstr.val + 32 * input_signed.val + 64 * input_direction.val : ℕ) : F p)
          = ((Alu.opselLT + 32 * s + 64 * i : ℕ) : F p) := by
        rw [← hflags]; push_cast; simp only [ZMod.natCast_zmod_val]
      have h2 := natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega))
        (lt_p_of_lt_two_pow_40 (by simp only [Alu.opselLT]; omega)) h1
      simp only [Alu.opselLT] at h2
      omega
    · -- not EQ
      intro i hi hflags
      exfalso
      have h1 : ((5 + input_wordInstr.val + 32 * input_signed.val + 64 * input_direction.val : ℕ) : F p)
          = ((Alu.opselEQ + 64 * i : ℕ) : F p) := by
        rw [← hflags]; push_cast; simp only [ZMod.natCast_zmod_val]
      have h2 := natCast_inj_of_lt (lt_p_of_lt_two_pow_40 (by omega))
        (lt_p_of_lt_two_pow_40 (by simp only [Alu.opselEQ]; omega)) h1
      simp only [Alu.opselEQ] at h2
      omega
    · intro w s d hw hs hd hflags
      obtain ⟨rfl, rfl, rfl⟩ := hflags_ne w s d hw hs hd hflags
      have hv1 : (input_in0 + 65536 * input_in1).val = input_in0.val + 65536 * input_in1.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      have hv2 : (input_in2 + 65536 * input_in3).val = input_in2.val + 65536 * input_in3.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      have hin1 : dword (input_in0 + 65536 * input_in1) (input_in2 + 65536 * input_in3)
          = limbs input_in0.val input_in1.val input_in2.val input_in3.val := by
        simp only [dword, limbs, hv1, hv2]; omega
      have hv3 : (input_shiftAmount + 256 * input_shiftB1 + 65536 * input_shiftH1).val
          = input_shiftAmount.val + 256 * input_shiftB1.val + 65536 * input_shiftH1.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      have hin2 : (input_shiftAmount + 256 * input_shiftB1 + 65536 * input_shiftH1).val % 256
          = input_shiftAmount.val := by
        rw [hv3]; omega
      rw [hin1, hin2]
      exact ⟨key.1, key.2.1, key.2.2⟩


theorem completeness :
    GeneralFormalCircuit.Completeness (F p) (Input := Row) (Output := unit) main
      (fun r _ _ => Honest r ∨ Padding r) (fun _ _ _ => True) := by
    circuit_proof_start [main, Alu.channel, Alu.Guarantees, Bitwise.isHalfTable, Bitwise.msb16Table,
      Bitwise.byteAluTable, Bitwise.zeroTable, Bitwise.hwslTable, Bitwise.areBytesTable, Honest,
      Padding, shiftedOf]
    rcases h_assumptions with
      ⟨hmu, hdb, hsb, hwb, hi0, hi1, hi2, hi3, hsA, hB1, hH1, hn, hb, hzbs, hx0, hx1, hx2, hx3, hx4,
        hy0, hy1, hy2, hy3, hl0, hl1, hl2, hout0, hout1⟩ |
      ⟨hmu, hd, hs, hw, hz, hneg, hx0, hx1, hx2, hx3, hx4, hy0, hy1, hy2, hy3, hl0, hl1, hl2,
        hout0, hout1, hsA, hB1⟩
    · -- an active row
      subst hmu
      have hd2 := IsBool.val_lt_two hdb
      have hs2 := IsBool.val_lt_two hsb
      have hw2 := IsBool.val_lt_two hwb
      have h15 : (15 : F p).val = 15 := val_ofNat 15 (by norm_num)
      have hn2 : input_isNegative.val < 2 := by rw [hn]; split_ifs <;> omega
      have hb16 : input_bitShift.val < 16 := by rw [hb]; split_ifs <;> omega
      have hL4 : input_shiftAmount.val / 16 % (4 - 2 * input_wordInstr.val) < 4 := by
        rcases (by omega : input_wordInstr.val = 0 ∨ input_wordInstr.val = 1) with h | h <;>
          rw [h] <;> omega
      -- the bits as field values
      have hl0b : IsBool input_ls0 := isBool_of_val_lt_two (by rw [hl0]; split_ifs <;> omega)
      have hl1b : IsBool input_ls1 := isBool_of_val_lt_two (by rw [hl1]; split_ifs <;> omega)
      have hl2b : IsBool input_ls2 := isBool_of_val_lt_two (by rw [hl2]; split_ifs <;> omega)
      have honehot : input_ls0.val + input_ls1.val + input_ls2.val ≤ 1 := by
        rw [hl0, hl1, hl2]; split_ifs <;> omega
      have hnb : IsBool input_isNegative := isBool_of_val_lt_two hn2
      have hzb : IsBool input_zbs := by rw [hzbs]; split_ifs <;> simp [IsBool]
      have hzb_iff : input_zbs = 1 ↔ input_bitShift.val = 0 := by
        rw [hzbs]
        by_cases hb0 : input_bitShift = 0
        · rw [if_pos hb0, hb0, ZMod.val_zero]; simp
        · rw [if_neg hb0]
          have hne : input_bitShift.val ≠ 0 := by rwa [Ne, ZMod.val_eq_zero]
          constructor
          · intro h; exact absurd h zero_ne_one
          · intro h; exact absurd h hne
      have hext : (65535 * input_isNegative).val = 65535 * input_isNegative.val :=
        val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
      -- the arithmetic core and the output words
      have hL : 16 * (input_ls1.val + 2 * input_ls2.val
            + 3 * (1 - input_ls0.val - input_ls1.val - input_ls2.val))
          = input_shiftAmount.val &&& (48 - 32 * input_wordInstr.val) := by
        rcases (by omega : input_wordInstr.val = 0 ∨ input_wordInstr.val = 1) with h | h
        · rw [h] at hl0 hl1 hl2 ⊢; rw [ShiftCore.and48 hsA, hl0, hl1, hl2]; split_ifs <;> omega
        · rw [h] at hl0 hl1 hl2 ⊢; rw [ShiftCore.and16 hsA, hl0, hl1, hl2]; split_ifs <;> omega
      have hcore := ShiftCore.main hd2 hs2 hw2 hsA hi0 hi1 hi2 hi3
        (by rw [hl0]; split_ifs <;> omega) (by rw [hl1]; split_ifs <;> omega)
        (by rw [hl2]; split_ifs <;> omega) honehot hL hb hn hx0 hx1 hx2 hx3 hx4 hy0 hy1 hy2 hy3
      simp only at hcore
      obtain ⟨hsh0, hsh1, hsh2, hsh3, -⟩ := hcore
      have hcast := shiftedPoly_cast (e := 65535 * input_isNegative.val)
        (x0 := input_x0.val) (x1 := input_x1.val) (x2 := input_x2.val) (x3 := input_x3.val)
        (x4 := input_x4.val) (y0 := input_y0.val) (y1 := input_y1.val) (y2 := input_y2.val)
        (y3 := input_y3.val) (by omega : input_direction.val ≤ 1) honehot
      simp only at hcast
      obtain ⟨hc1, hc2, hc3, hc4⟩ := hcast
      have hout0' : input_out0 = (((ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
          input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
          input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).1
          + 65536 * (ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
          input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
          input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).2.1 : ℕ) : F p) := by
        rw [← hout0, ZMod.natCast_zmod_val]
      have hout1' : input_out1 = (((ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
          input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
          input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).2.2.1
          + 65536 * (ShiftCore.shifted input_direction.val input_ls0.val input_ls1.val
          input_ls2.val (65535 * input_isNegative.val) input_x0.val input_x1.val input_x2.val
          input_x3.val input_x4.val input_y0.val input_y1.val input_y2.val input_y3.val).2.2.2 : ℕ) : F p) := by
        rw [← hout1, ZMod.natCast_zmod_val]
      -- field forms of the naturals
      have hdc := (ZMod.natCast_zmod_val input_direction).symm
      have hl0c := (ZMod.natCast_zmod_val input_ls0).symm
      have hl1c := (ZMod.natCast_zmod_val input_ls1).symm
      have hl2c := (ZMod.natCast_zmod_val input_ls2).symm
      have hnc := (ZMod.natCast_zmod_val input_isNegative).symm
      have hx0c := (ZMod.natCast_zmod_val input_x0).symm
      have hx1c := (ZMod.natCast_zmod_val input_x1).symm
      have hx2c := (ZMod.natCast_zmod_val input_x2).symm
      have hx3c := (ZMod.natCast_zmod_val input_x3).symm
      have hx4c := (ZMod.natCast_zmod_val input_x4).symm
      have hy0c := (ZMod.natCast_zmod_val input_y0).symm
      have hy1c := (ZMod.natCast_zmod_val input_y1).symm
      have hy2c := (ZMod.natCast_zmod_val input_y2).symm
      have hy3c := (ZMod.natCast_zmod_val input_y3).symm
      have hiff : input_zbs = 1 → input_bitShift.val = 0 := hzb_iff.mp
      have hl3b : IsBool (1 - input_ls0 - input_ls1 - input_ls2) := by
        rcases hl0b with h0 | h0 <;> rcases hl1b with h1 | h1 <;> rcases hl2b with h2 | h2 <;>
          simp only [h0, h1, h2, ZMod.val_zero, val_one_eq] at honehot ⊢ <;>
          first | omega | norm_num [IsBool]
      refine ⟨by simp, override_x hzb hdb hiff hx0, override_x hzb hdb hiff hx1,
        override_x hzb hdb hiff hx2, override_x hzb hdb hiff hx3, ?_,
        override_y hzb hdb hiff hy0, override_y hzb hdb hiff hy1, override_y hzb hdb hiff hy2,
        override_y hzb hdb hiff hy3, isBool_mul_one_sub hl0b,
        isBool_mul_one_sub hl1b, isBool_mul_one_sub hl2b, isBool_mul_one_sub hl3b, ?_, ?_,
        isBool_mul_one_sub hdb, isBool_mul_one_sub hsb, isBool_mul_one_sub hwb,
        isBool_mul_one_sub hzb, by simp, ?_,
        ?_, ?_, ?_, fun _ => ⟨by omega, hzbs⟩, ?_, ?_, ?_, ?_, ?_, ?_, ⟨hsA, hB1⟩, fun _ => hH1,
        fun _ => hi0, fun _ => hi1, fun _ => hi2, fun _ => hi3⟩
      · -- zbs · X[4] = 0
        rcases hzb with hz | hz
        · rw [hz, zero_mul]
        · rw [if_pos (hiff hz)] at hx4
          rw [hz, one_mul, (ZMod.val_eq_zero _).mp hx4]
      · -- out[0]
        rw [eval_shiftedPoly_1, eval_shiftedPoly_2]
        simp only [circuit_norm, h_input]
        rw [hdc, hl0c, hl1c, hl2c, hnc, hx0c, hx1c, hx2c, hx3c, hx4c, hy0c, hy1c, hy2c, hy3c,
          show (65535 : F p) * ((input_isNegative.val : ℕ) : F p)
            = ((65535 * input_isNegative.val : ℕ) : F p) by push_cast; ring, hc1, hc2, hout0']
        push_cast; ring
      · -- out[1]
        rw [eval_shiftedPoly_3, eval_shiftedPoly_4]
        simp only [circuit_norm, h_input]
        rw [hdc, hl0c, hl1c, hl2c, hnc, hx0c, hx1c, hx2c, hx3c, hx4c, hy0c, hy1c, hy2c, hy3c,
          show (65535 : F p) * ((input_isNegative.val : ℕ) : F p)
            = ((65535 * input_isNegative.val : ℕ) : F p) by push_cast; ring, hc3, hc4, hout1']
        push_cast; ring
      · -- (1 − signed) · is_negative = 0
        rcases hsb with hs | hs
        · rw [hs] at hn ⊢
          simp only [ZMod.val_zero, zero_ne_one, if_false] at hn
          rw [(ZMod.val_eq_zero _).mp hn]; ring
        · rw [hs]; ring
      · -- MSB16[in[3]] when signed
        intro h
        have hs1 : input_signed = 1 := neg_inj.mp h
        rw [hs1, val_one_eq, if_pos rfl] at hn
        exact ⟨hi3, hn⟩
      · -- BYTE_ALU[AND, shift, 15] when left
        intro h
        have hd0 : input_direction = 0 := by
          have := neg_inj.mp h
          linear_combination -this
        rw [hd0, ZMod.val_zero, if_pos rfl] at hb
        refine ⟨hsA, by rw [h15]; norm_num, Or.inl ?_⟩
        rw [h15, ShiftCore.and15, hb]
      · -- BYTE_ALU[AND, 256 − 16·zbs − shift, 15] when right
        intro h
        have hd1 : input_direction = 1 := neg_inj.mp h
        rw [hd1, val_one_eq, if_neg one_ne_zero] at hb
        rcases hzb with hz | hz
        · have hbne : input_bitShift.val ≠ 0 := fun h0 => by
            rw [hzbs, if_pos ((ZMod.val_eq_zero _).mp h0)] at hz; exact one_ne_zero hz
          have hs1 : 1 ≤ input_shiftAmount.val := by omega
          have hcast : (256 : F p) - 16 * input_zbs - input_shiftAmount
              = ((256 - input_shiftAmount.val : ℕ) : F p) := by
            rw [hz, Nat.cast_sub (by omega)]; push_cast
            rw [ZMod.natCast_zmod_val input_shiftAmount]; ring
          rw [hcast, val_natCast_of_lt_two_pow_40 (by omega), h15, ShiftCore.and15, hb]
          exact ⟨by omega, by norm_num, Or.inl (by omega)⟩
        · have hb0 : input_bitShift.val = 0 := hiff hz
          have hle : input_shiftAmount.val ≤ 240 := by omega
          have hcast : (256 : F p) - 16 * input_zbs - input_shiftAmount
              = ((240 - input_shiftAmount.val : ℕ) : F p) := by
            rw [hz, Nat.cast_sub hle]; push_cast
            rw [ZMod.natCast_zmod_val input_shiftAmount]; ring
          rw [hcast, val_natCast_of_lt_two_pow_40 (by omega), h15, ShiftCore.and15, hb0]
          exact ⟨by omega, by norm_num, Or.inl (by omega)⟩
      · -- HWSL[in[0], bit_shift]
        intro h
        have hbne : input_bitShift.val ≠ 0 := fun h0 => by
          have := hzb_iff.mpr h0
          rw [this] at h; norm_num at h
        rw [if_neg hbne] at hx0 hy0
        exact ⟨hi0, hb16, by rw [hx0]; simp [sll, hbne], by rw [hy0]; simp [sllc, hbne]⟩
      · intro h
        have hbne : input_bitShift.val ≠ 0 := fun h0 => by
          have := hzb_iff.mpr h0
          rw [this] at h; norm_num at h
        rw [if_neg hbne] at hx1 hy1
        exact ⟨hi1, hb16, by rw [hx1]; simp [sll, hbne], by rw [hy1]; simp [sllc, hbne]⟩
      · intro h
        have hbne : input_bitShift.val ≠ 0 := fun h0 => by
          have := hzb_iff.mpr h0
          rw [this] at h; norm_num at h
        rw [if_neg hbne] at hx2 hy2
        exact ⟨hi2, hb16, by rw [hx2]; simp [sll, hbne], by rw [hy2]; simp [sllc, hbne]⟩
      · intro h
        have hbne : input_bitShift.val ≠ 0 := fun h0 => by
          have := hzb_iff.mpr h0
          rw [this] at h; norm_num at h
        rw [if_neg hbne] at hx3 hy3
        exact ⟨hi3, hb16, by rw [hx3]; simp [sll, hbne], by rw [hy3]; simp [sllc, hbne]⟩
      · -- HWSL[extension, bit_shift]
        intro h
        have hbne : input_bitShift.val ≠ 0 := fun h0 => by
          have := hzb_iff.mpr h0
          rw [this] at h; norm_num at h
        rw [if_neg hbne] at hx4
        have he : 65535 * input_isNegative.val = 0 ∨ 65535 * input_isNegative.val = 65535 := by omega
        obtain ⟨hle, hsub⟩ := ext_sllc he hb16
        have hdiff : (65535 * input_isNegative - input_x4).val
            = 65535 * input_isNegative.val - input_x4.val := by
          rw [← hx4] at hle
          rw [show (65535 : F p) * input_isNegative - input_x4
              = ((65535 * input_isNegative.val - input_x4.val : ℕ) : F p) by
            rw [Nat.cast_sub hle]; push_cast
            rw [ZMod.natCast_zmod_val input_isNegative, ZMod.natCast_zmod_val input_x4]; try ring]
          exact val_natCast_of_lt_two_pow_40 (by omega)
        refine ⟨by rw [hext]; omega, hb16, ?_, ?_⟩
        · rw [hext, hx4]; try simp [sll, hbne]
        · rw [hdiff, hext, hx4, hsub]; try simp [sllc, hbne]
      · -- BYTE_ALU[AND, shift, mask]
        intro _
        have hmask : (48 - 32 * input_wordInstr).val = 48 - 32 * input_wordInstr.val := by
          rcases hwb with hw0 | hw0
          · rw [hw0]; simp only [mul_zero, sub_zero, ZMod.val_zero]; exact val_ofNat 48 (by norm_num)
          · rw [hw0]; norm_num; exact val_ofNat 16 (by norm_num)
        refine ⟨hsA, by rw [hmask]; omega, Or.inl ?_⟩
        rw [hmask]
        have hand : input_shiftAmount.val &&& (48 - 32 * input_wordInstr.val)
            = 16 * (input_shiftAmount.val / 16 % (4 - 2 * input_wordInstr.val)) := by
          rcases (by omega : input_wordInstr.val = 0 ∨ input_wordInstr.val = 1) with h | h
          · rw [h]; simpa using ShiftCore.and48 hsA
          · rw [h]; simpa using ShiftCore.and16 hsA
        rw [hand]
        rw [eq_natCast_of_val hl0, eq_natCast_of_val hl1, eq_natCast_of_val hl2]
        rcases (by omega : input_shiftAmount.val / 16 % (4 - 2 * input_wordInstr.val) = 0 ∨
            input_shiftAmount.val / 16 % (4 - 2 * input_wordInstr.val) = 1 ∨
            input_shiftAmount.val / 16 % (4 - 2 * input_wordInstr.val) = 2 ∨
            input_shiftAmount.val / 16 % (4 - 2 * input_wordInstr.val) = 3) with hLv | hLv | hLv | hLv
        · rw [hLv]; norm_num
        · rw [hLv]; norm_num; exact val_ofNat 16 (by norm_num)
        · rw [hLv]; norm_num; exact val_ofNat 32 (by norm_num)
        · rw [hLv]; norm_num; exact val_ofNat 48 (by norm_num)
    · -- a padding row
      subst hmu hd hs hw hz hneg hx0 hx1 hx2 hx3 hx4 hy0 hy1 hy2 hy3 hl0 hl1 hl2 hout0 hout1 hsA hB1
      refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
      · try simp only [eval_shiftedPoly_1, eval_shiftedPoly_2, eval_shiftedPoly_3, eval_shiftedPoly_4]
        simp [circuit_norm, h_input, shiftedPoly]


def circuit : GeneralFormalCircuit (F p) Row unit where
  main
  elaborated
  Spec r _ _ := RowSpec r
  channelsWithRequirements := [Alu.channel.toRaw]
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨in0, in1, in2, in3, shiftAmount, direction, signed, wordInstr, out0, out1, isNegative,
      bitShift, zbs, x0, x1, x2, x3, x4, y0, y1, y2, y3, ls0, ls1, ls2, mu, shiftB1, shiftH1,
      shiftHigh⟩ := input_var
    simp only [circuit_norm, main, Alu.channel, Bitwise.isHalfTable, Bitwise.msb16Table,
      Bitwise.byteAluTable, Bitwise.zeroTable, Bitwise.hwslTable, Bitwise.areBytesTable]
    constructor
    · intro channel h; tauto
    · intro env h
      obtain ⟨h_dm, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h_dir, h_sg, -, h_zbs, h_mu, -⟩ := h
      have hdb := isBool_of_mul_one_sub h_dir
      have hsb := isBool_of_mul_one_sub h_sg
      have hzb := isBool_of_mul_one_sub h_zbs
      have hmb := isBool_of_mul_one_sub h_mu
      constructorm* _ ∧ _ <;>
      · intro h1 h0
        rcases hmb with hm | hm <;> rcases hdb with hd | hd <;> rcases hzb with hz | hz <;>
          rcases hsb with hs | hs <;> simp_all

  soundness := soundness
  ProverAssumptions r _ _ := Honest r ∨ Padding r

  completeness := completeness

end LambdaVM.SHIFT
