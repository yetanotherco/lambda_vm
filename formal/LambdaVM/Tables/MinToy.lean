module

public import LambdaVM.Buses.Alu
public import LambdaVM.Templates.IsBit

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
  -- ALU[a, b, opsel(MIN)] -> out
  Alu.channel.pushIf r.mu ⟨r.a0, r.a1, r.b0, r.b1, 7, r.out0, r.out1⟩

lemma val_lit (n : ℕ) (h : n < 2^40) : ((n : F p)).val = n :=
  ZMod.val_natCast_of_lt (lt_p_of_lt_two_pow_40 h)

/-- The honest prover's row: `lt` is the comparison and `out` the selected operand. -/
def Honest (r : Row (F p)) : Prop :=
  IsBool r.mu ∧ r.lt = ((Alu.ltBit 0 0 (dword r.a0 r.a1) (dword r.b0 r.b1) : ℕ) : F p) ∧
  r.out0 = r.lt * r.a0 + (1 - r.lt) * r.b0 ∧ r.out1 = r.lt * r.a1 + (1 - r.lt) * r.b1

def circuit : GeneralFormalCircuit (F p) Row unit where
  name := "MIN"
  main
  Spec _ _ _ := True
  channelsWithRequirements := [Alu.channel.toRaw]
  soundness := by
    circuit_proof_start [main, isBit, Alu.channel, Alu.Guarantees]
    obtain ⟨-, h_pull, h_o0, h_o1⟩ := h_holds
    intro _
    have h4 : (4 : F p).val = 4 := by simpa using val_lit 4 (by norm_num)
    have h7 : (7 : F p).val = 7 := by simpa using val_lit 7 (by norm_num)
    obtain ⟨hLT, -⟩ := h_pull
    rw [h7]
    -- only the MIN clause applies to flags = 7; it carries its own `IS_WORD` hypotheses
    refine ⟨fun s i hs hi h => by simp only [opselLT] at h; omega,
      fun i hi h => by simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by simp only [opselSHIFT] at h; omega,
      fun _ ha0 ha1 hb0 hb1 => ?_⟩
    -- the pulled LT result, with μ = 1
    have hlt := hLT 0 0 (by norm_num) (by norm_num) (by rw [h4]; simp [opselLT]) ha0 hb0
    have hL : dword input_a0 input_a1 < 2^64 := by unfold dword; omega
    have hR : dword input_b0 input_b1 < 2^64 := by unfold dword; omega
    by_cases hLR : dword input_a0 input_a1 < dword input_b0 input_b1
    · have h1 : input_lt = 1 := by
        rw [hlt]; simp [Alu.ltBit, Alu.ltRaw, Alu.ult_ofNat_iff hL hR, hLR]
      have e0 : input_out0 = input_a0 := by rw [h1] at h_o0; linear_combination h_o0
      have e1 : input_out1 = input_a1 := by rw [h1] at h_o1; linear_combination h_o1
      rw [e0, e1, min_eq_left hLR.le]
      exact ⟨ha0, ha1, rfl⟩
    · have h1 : input_lt = 0 := by
        rw [hlt]; simp [Alu.ltBit, Alu.ltRaw, Alu.ult_ofNat_iff hL hR, hLR]
      have e0 : input_out0 = input_b0 := by rw [h1] at h_o0; linear_combination h_o0
      have e1 : input_out1 = input_b1 := by rw [h1] at h_o1; linear_combination h_o1
      rw [e0, e1, min_eq_right (not_lt.mp hLR)]
      exact ⟨hb0, hb1, rfl⟩

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
