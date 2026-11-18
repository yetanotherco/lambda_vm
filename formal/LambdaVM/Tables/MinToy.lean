module

public import LambdaVM.Buses.Alu
public import LambdaVM.Templates.IsBit
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# MIN (toy, for the counted-pulls PoC)

A chip with the shape of DVRM: it pulls an LT result on the ALU bus and pushes its own result on
the same bus. `out = lt ? a : b` limb by limb, where `lt = (a < b)` comes from the pulled LT
message. One table, one bus, no cycle: it receives `flags = opsel(LT)` and owes
`flags = opsel(MIN)`.
-/

namespace LambdaVM.MinToy

structure Row (F : Type) where
  a0 : F
  a1 : F
  b0 : F
  b1 : F
  lt : F
  out0 : F
  out1 : F
  mu : F
deriving ProvableStruct

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  isBit r.mu
  -- ALU[a, b, opsel(LT)] -> [lt, 0]: unsigned, no invert. Pulled on every row, padding included.
  Alu.channel.pull ⟨r.a0, r.a1, r.b0, r.b1, 4, r.lt, 0⟩
  -- out = lt ? a : b
  assertZero (r.out0 - (r.lt * r.a0 + (1 - r.lt) * r.b0))
  assertZero (r.out1 - (r.lt * r.a1 + (1 - r.lt) * r.b1))
  -- ALU[a, b, opsel(MIN) = 9] -> out
  Alu.channel.pushIf r.mu ⟨r.a0, r.a1, r.b0, r.b1, 9, r.out0, r.out1⟩

/-- The flag bytes in `main` and in `CountedMin.main` are written as the literals `4` and `9`,
not as `opselLT` and `Alu.opselMIN`: the proofs normalise the evaluated flag to `(4 : F p)`, and
a `((opselLT : ℕ) : F p)` elaborates to `Nat.cast`, which those rewrites no longer match. This
pins the literals to the names, so the two cannot drift apart unnoticed. -/
theorem opsel_literals : opselLT = 4 ∧ Alu.opselMIN = 9 := ⟨rfl, rfl⟩

lemma val_lit (n : ℕ) (h : n < 2^40) : ((n : F p)).val = n :=
  ZMod.val_natCast_of_lt (lt_p_of_lt_two_pow_40 h)

/-- The content of a MIN row, shared by MIN and CMIN: given the LT answer the row pulled and the
two selection constraints, `out` is the unsigned minimum. The two chips differ only in how they
come by `hlt` — MIN pulls once, CMIN pulls `μ` times — so everything after that is this lemma. -/
lemma min_of_lt_answer {a0 a1 b0 b1 lt out0 out1 : F p}
    (ha0 : a0.val < 2^32) (ha1 : a1.val < 2^32) (hb0 : b0.val < 2^32) (hb1 : b1.val < 2^32)
    (hlt : lt = ((Alu.ltBit 0 0 (dword a0 a1) (dword b0 b1) : ℕ) : F p))
    (h_o0 : out0 - (lt * a0 + (1 - lt) * b0) = 0)
    (h_o1 : out1 - (lt * a1 + (1 - lt) * b1) = 0) :
    out0.val < 2^32 ∧ out1.val < 2^32 ∧ dword out0 out1 = min (dword a0 a1) (dword b0 b1) := by
  have hL : dword a0 a1 < 2^64 := by unfold dword; omega
  have hR : dword b0 b1 < 2^64 := by unfold dword; omega
  by_cases hLR : dword a0 a1 < dword b0 b1
  · have h1 : lt = 1 := by
      rw [hlt]; simp [Alu.ltBit, Alu.ltRaw, Alu.ult_ofNat_iff hL hR, hLR]
    have e0 : out0 = a0 := by rw [h1] at h_o0; linear_combination h_o0
    have e1 : out1 = a1 := by rw [h1] at h_o1; linear_combination h_o1
    rw [e0, e1, min_eq_left hLR.le]
    exact ⟨ha0, ha1, rfl⟩
  · have h1 : lt = 0 := by
      rw [hlt]; simp [Alu.ltBit, Alu.ltRaw, Alu.ult_ofNat_iff hL hR, hLR]
    have e0 : out0 = b0 := by rw [h1] at h_o0; linear_combination h_o0
    have e1 : out1 = b1 := by rw [h1] at h_o1; linear_combination h_o1
    rw [e0, e1, min_eq_right (not_lt.mp hLR)]
    exact ⟨hb0, hb1, rfl⟩

/-- The honest prover's row: `lt` is the comparison and `out` the selected operand. -/
def Honest (r : Row (F p)) : Prop :=
  IsBool r.mu ∧ r.lt = ((Alu.ltBit 0 0 (dword r.a0 r.a1) (dword r.b0 r.b1) : ℕ) : F p) ∧
  r.out0 = r.lt * r.a0 + (1 - r.lt) * r.b0 ∧ r.out1 = r.lt * r.a1 + (1 - r.lt) * r.b1

/-- What a row establishes: with range-checked operand limbs, `out` is the unsigned minimum of
the two operands. The LT answer is pulled unconditionally, so this holds on every row, padding
included — unlike `CountedMin.RowSpec`, it needs no `μ ≠ 0` guard. -/
def RowSpec (r : Row (F p)) : Prop :=
  r.a0.val < 2^32 → r.a1.val < 2^32 → r.b0.val < 2^32 → r.b1.val < 2^32 →
    r.out0.val < 2^32 ∧ r.out1.val < 2^32 ∧
    dword r.out0 r.out1 = min (dword r.a0 r.a1) (dword r.b0 r.b1)

def circuit : GeneralFormalCircuit (F p) Row unit where
  name := "MIN"
  main
  Spec r _ _ := RowSpec r
  channelsWithRequirements := [Alu.channel.toRaw]
  soundness := by
    circuit_proof_start [main, isBit, Alu.channel, Alu.Guarantees, RowSpec]
    obtain ⟨-, h_pull, h_o0, h_o1⟩ := h_holds
    have h4 : (4 : F p).val = 4 := by simpa using val_lit 4 (by norm_num)
    have h9 : (9 : F p).val = 9 := by simpa using val_lit 9 (by norm_num)
    obtain ⟨hLT, -⟩ := h_pull
    -- The content of the row, proven once and used twice: as the circuit's own spec, and as the
    -- MIN clause of what this row owes the ALU bus.
    have key : input_a0.val < 2^32 → input_a1.val < 2^32 → input_b0.val < 2^32 →
        input_b1.val < 2^32 → input_out0.val < 2^32 ∧ input_out1.val < 2^32 ∧
        dword input_out0 input_out1 =
          min (dword input_a0 input_a1) (dword input_b0 input_b1) := by
      intro ha0 ha1 hb0 hb1
      -- the pulled LT result, with μ = 1
      exact min_of_lt_answer ha0 ha1 hb0 hb1
        (hLT 0 0 (by norm_num) (by norm_num) (by rw [h4]; simp [opselLT]) ha0 hb0) h_o0 h_o1
    refine ⟨key, ?_⟩
    intro _
    rw [h9]
    -- only the MIN clause applies to flags = opsel(MIN); it carries its own `IS_WORD` hypotheses
    exact ⟨fun s i hs hi h => by simp only [opselLT] at h; omega,
      fun i hi h => by simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by simp only [opselSHIFT] at h; omega,
      fun _ ha0 ha1 hb0 hb1 => key ha0 ha1 hb0 hb1⟩

  ProverAssumptions r _ _ := Honest r
  completeness := by
    circuit_proof_start [main, isBit, Alu.channel, Alu.Guarantees, Honest]
    obtain ⟨hmu, hlt, ho0, ho1⟩ := h_assumptions
    have h4 : (4 : F p).val = 4 := by simpa using val_lit 4 (by norm_num)
    refine ⟨mul_one_sub_eq_zero_iff.mpr hmu, ?_, by rw [ho0]; ring, by rw [ho1]; ring⟩
    rw [h4]
    refine ⟨fun s i hs hi h _ _ => ?_, fun i hi h => by simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by simp only [opselSHIFT] at h; omega,
      fun h => by simp only [Alu.opselMIN] at h; omega⟩
    simp only [opselLT] at h
    obtain ⟨rfl, rfl⟩ : s = 0 ∧ i = 0 := by omega
    exact hlt

end LambdaVM.MinToy
