module

public import LambdaVM.Ensembles.LtStaged

@[expose] public section

/-!
# Counted pulls and a response on the same bus (PoC)

The shape of DVRM with everything at once. The ALU bus is a counted channel. One table, CMIN,
answers `μ ≤ 255` identical MIN requests per row: it pulls an LT result on the ALU bus with
multiplicity `μ`, bounded by its own pull of `ARE_BYTES[μ, 0]`, and pushes the MIN result `μ`
times on the same bus. The LT answers come from a provider that stands for the LT table.

Stages, oldest first: `ARE_BYTES`, the ALU messages that are not MIN, the MIN messages. CMIN's
requirement on its counted pull (`μ = 0 ∨ 1 ≤ μ ≤ 255`) uses its guarantee on `ARE_BYTES`; its
requirement on the MIN push uses its guarantee on the LT pull, in the earlier ALU class.
-/

namespace LambdaVM.CountedMin
open LambdaVM.LT (muMax)

/-- The ALU bus with pulls of multiplicity up to `muMax`. -/
abbrev AluC : RawChannel (F p) := Alu.channel.counted muMax

def main (r : Var MinToy.Row (F p)) : Circuit (F p) Unit := do
  -- μ : Byte: ARE_BYTES[μ, 0]
  Bitwise.AreBytes.pull (r.mu, 0)
  -- ALU[a, b, opsel(LT)] -> [lt, 0], μ times
  Alu.channel.pullUpTo muMax r.mu ⟨r.a0, r.a1, r.b0, r.b1, 4, r.lt, 0⟩
  -- out = lt ? a : b
  assertZero (r.out0 - (r.lt * r.a0 + (1 - r.lt) * r.b0))
  assertZero (r.out1 - (r.lt * r.a1 + (1 - r.lt) * r.b1))
  -- ALU[a, b, opsel(MIN)] -> out, μ times
  Alu.channel.emitUpTo muMax r.mu ⟨r.a0, r.a1, r.b0, r.b1, 7, r.out0, r.out1⟩

/-- The honest prover's row: `μ` a byte, `lt` the comparison and `out` the selected operand. -/
def Honest (r : MinToy.Row (F p)) : Prop :=
  r.mu.val < 2^8 ∧ r.lt = ((Alu.ltBit 0 0 (dword r.a0 r.a1) (dword r.b0 r.b1) : ℕ) : F p) ∧
  r.out0 = r.lt * r.a0 + (1 - r.lt) * r.b0 ∧ r.out1 = r.lt * r.a1 + (1 - r.lt) * r.b1

def circuit : GeneralFormalCircuit (F p) MinToy.Row unit where
  name := "CMIN"
  main
  Spec _ _ _ := True
  channelsWithRequirements := [AluC]
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨a0, a1, b0, b1, lt, out0, out1, mu⟩ := input_var
    simp only [circuit_norm, main, AluC, Bitwise.AreBytes]
    intro channel h
    tauto
  soundness := by
    circuit_proof_start [main, Alu.channel, Alu.Guarantees, Bitwise.areBytesTable]
    obtain ⟨⟨h_mu, -⟩, h_pull, h_o0, h_o1⟩ := h_holds
    -- with μ = 0 nothing is owed
    by_cases h0 : input_mu = 0
    · simp [h0]
    -- otherwise ARE_BYTES[μ, 0] makes μ a count of at most 255
    have hcount : Channel.CountUpTo muMax input_mu :=
      Channel.countUpTo_of_val h0 (by simp only [muMax]; omega)
    refine ⟨.inr hcount, fun _ ha0 ha1 hb0 hb1 => ?_⟩
    have h4 : (4 : F p).val = 4 := by simpa using MinToy.val_lit 4 (by norm_num)
    have h7 : (7 : F p).val = 7 := by simpa using MinToy.val_lit 7 (by norm_num)
    -- the pulled LT result
    obtain ⟨hLT, -⟩ := h_pull hcount ha0 ha1 hb0 hb1
    have hlt := hLT 0 0 (by norm_num) (by norm_num) (by rw [h4]; simp [opselLT])
    have hL : dword input_a0 input_a1 < 2^64 := by unfold dword; omega
    have hR : dword input_b0 input_b1 < 2^64 := by unfold dword; omega
    rw [h7]
    refine ⟨fun s i hs hi h => by simp only [opselLT] at h; omega,
      fun i hi h => by simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by simp only [opselSHIFT] at h; omega, fun _ => ?_⟩
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
    circuit_proof_start [main, Alu.channel, Alu.Guarantees, Bitwise.areBytesTable, Honest]
    obtain ⟨hmu, hlt, ho0, ho1⟩ := h_assumptions
    have h4 : (4 : F p).val = 4 := by simpa using MinToy.val_lit 4 (by norm_num)
    refine ⟨⟨hmu, by simp⟩, ?_, by rw [ho0]; ring, by rw [ho1]; ring⟩
    intro _ _ _ _ _
    rw [h4]
    refine ⟨fun s i hs hi h => ?_, fun i hi h => by simp only [opselEQ] at h; omega,
      fun w s d hw hs hd h => by simp only [opselSHIFT] at h; omega,
      fun h => by simp only [Alu.opselMIN] at h; omega⟩
    simp only [opselLT] at h
    obtain ⟨rfl, rfl⟩ : s = 0 ∧ i = 0 := by omega
    exact hlt

/-! ## The ensemble: ARE_BYTES, the LT answers, CMIN -/

open Air.Flat
open Ensembles.LtStaged (providerComponent countedProviderComponent demandOn isMinMsg
  isMinMsg_toElements msg_eval mu_count providerComponent_guarantees
  providerComponent_requirements countedProviderComponent_guarantees
  countedProviderComponent_requirements)

def cminComponent : Component (F p) where
  circuit := circuit
  generation := demandOn Alu.channel _

@[circuit_norm] lemma cminComponent_guarantees :
    cminComponent.circuit.channelsWithGuarantees = [Bitwise.AreBytes.toRaw, AluC] := rfl
@[circuit_norm] lemma cminComponent_requirements :
    cminComponent.circuit.channelsWithRequirements = [AluC] := rfl

/-- Stages, newest first: the MIN messages, the other ALU messages, ARE_BYTES. -/
def stages : List (Stage (F p)) :=
  [{ channel := AluC, cls := isMinMsg }, { channel := AluC, cls := fun m => !isMinMsg m },
   { channel := Bitwise.AreBytes.toRaw }]

/-- The LT answers stand for the LT table: a provider of ALU messages on the counted bus. -/
def ensemble : Ensemble (F p) unit where
  tables := [providerComponent Bitwise.AreBytes, countedProviderComponent muMax Alu.channel,
    cminComponent]
  unique_names := by
    simp [providerComponent, countedProviderComponent, cminComponent, Ensembles.LtStaged.provider,
      Ensembles.LtStaged.countedProvider, circuit, Alu.channel, Bitwise.AreBytes,
      Bitwise.areBytesTable, StaticLookupChannel.toChannel]
  channels := [AluC, Bitwise.AreBytes.toRaw]
  unique_channel_names := by
    simp [AluC, circuit_norm, Alu.channel, Bitwise.AreBytes, Bitwise.areBytesTable]
  tables_channels := by
    intro table h
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl <;>
      simp only [GeneralFormalCircuit.channels, providerComponent_guarantees,
        providerComponent_requirements, countedProviderComponent_guarantees,
        countedProviderComponent_requirements, cminComponent_guarantees,
        cminComponent_requirements] <;>
      simp [AluC]
  verifier_channels := by simp [circuit_norm]

/-- `msg_eval`, in the form `simp` leaves an evaluated message in. -/
lemma msg_eval_array {M : TypeMap} [ProvableType M] (env : Environment (F p))
    (x : M (Expression (F p))) :
    Array.map (Expression.eval env) (toElements x).toArray = (toElements (eval env x)).toArray := by
  rw [← Vector.toArray_map]
  exact msg_eval env x

lemma isMinMsg_eval (env : Environment (F p)) (m : Alu.Msg (Expression (F p))) :
    isMinMsg (Array.map (fun x => Expression.eval env x) (toElements m).toArray) =
      decide ((eval env m).flags = 7) := by
  rw [show (fun x => Expression.eval env x) = Expression.eval env from rfl, msg_eval_array,
    isMinMsg_toElements]

lemma areBytesTable_guarantees (x y : F p) :
    Bitwise.areBytesTable.Guarantees (x, y) ↔ x.val < 2^8 ∧ y.val < 2^8 := Iff.rfl

lemma bytes_ne_alu : Bitwise.AreBytes.toRaw ≠ Channel.counted muMax Alu.channel := by
  intro h
  have := congrArg RawChannel.name h
  simp [circuit_norm, Alu.channel, Bitwise.AreBytes, Bitwise.areBytesTable, StaticLookupChannel.toChannel,
    Channel.counted, Channel.toRaw] at this

/-- Each row of CMIN has two interactions on the ALU bus: the LT pull, which is not a MIN message,
and the MIN push. -/
lemma cmin_alu_shape {table : Table (F p)} (h_comp : table.component = cminComponent)
    {i : Interaction (F p)} (h_i : i ∈ table.interactionsWith AluC) :
    (i.direction = .pull ∧ isMinMsg i.msg = false) ∨
      (i.direction = .push ∧ isMinMsg i.msg = true) := by
  cases table
  simp only at h_comp
  subst h_comp
  simp [Table.interactionsWith, List.mem_flatMap, Operations.interactionValuesWith_eq_map,
    Component.interactionsWith_eq, Component.rowOperations, cminComponent, circuit, main, AluC,
    circuit_norm] at h_i
  obtain ⟨row, -, a, ha, rfl⟩ := h_i
  split_ifs at ha with h_eq
  · exfalso
    have := congrArg RawChannel.name h_eq
    simp [Bitwise.areBytesTable, Channel.counted, Alu.channel, circuit_norm] at this
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ha
  rcases ha with rfl | rfl
  · left
    simp only [AbstractInteraction.eval, Channel.pulledUpTo]
    erw [msg_eval, isMinMsg_toElements]
    simp (config := { decide := true }) [circuit_norm]
  · right
    simp only [AbstractInteraction.eval, Channel.emittedUpTo]
    erw [msg_eval, isMinMsg_toElements]
    simp (config := { decide := true }) [circuit_norm]

/-- CMIN's counted pull owes `μ = 0 ∨ 1 ≤ μ ≤ 255`, which its guarantee on ARE_BYTES gives. -/
lemma cmin_pull_requirement {table : Table (F p)} (h_comp : table.component = cminComponent)
    (g_bytes : ∀ i ∈ table.interactionsWith Bitwise.AreBytes.toRaw, i.Guarantees table.data)
    {i : Interaction (F p)} (h_i : i ∈ table.interactionsWith AluC)
    (h_pull : i.direction = .pull) : i.Requirements table.data := by
  cases table
  simp only at h_comp
  subst h_comp
  simp [Table.interactionsWith, List.mem_flatMap, Operations.interactionValuesWith_eq_map,
    Component.interactionsWith_eq, Component.rowOperations, cminComponent, circuit, main, AluC,
    circuit_norm, forall_exists_index, and_imp] at h_i g_bytes
  obtain ⟨row, hrow, a, ha, rfl⟩ := h_i
  split_ifs at ha with h_eq
  · exfalso
    have := congrArg RawChannel.name h_eq
    simp [Bitwise.areBytesTable, Channel.counted, Alu.channel, circuit_norm] at this
  simp only [List.mem_cons, List.not_mem_nil, or_false] at ha
  rcases ha with rfl | rfl
  · have g := g_bytes _ row hrow (Or.inl rfl)
    simp only [AbstractInteraction.eval, ChannelInteraction.toRaw, pulled, Interaction.Guarantees,
      Channel.toRaw, Interaction.msgVector] at g
    have g1 := g trivial (by simp [circuit_norm])
    simp only [Vector.mk_toArray, ProvableType.fromElements_eval_toElements] at g1
    simp only [circuit_norm, explicit_provable_type, areBytesTable_guarantees] at g1
    simp only [AbstractInteraction.eval, Channel.pulledUpTo, Interaction.Requirements,
      Channel.counted]
    exact mu_count g1.1
  · simp [AbstractInteraction.eval, Channel.emittedUpTo] at h_pull

theorem tableSoundness : ensemble.TableSoundness (Balance.logUpUpTo muMax) := by
  apply Ensemble.tableSoundness_of_stagedOn stages rfl
  · intro s hs
    simp only [stages, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl <;> simp [ensemble]
  · intro component h c hc msg
    simp only [ensemble, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl
    iterate 2 simp [providerComponent_guarantees, countedProviderComponent_guarantees] at hc
    · rw [cminComponent_guarantees] at hc
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      rcases hc with rfl | rfl
      · exact ⟨{ channel := Bitwise.AreBytes.toRaw }, by simp [stages], rfl, rfl⟩
      · cases h : isMinMsg msg
        · exact ⟨{ channel := AluC, cls := fun m => !isMinMsg m }, by simp [stages], rfl,
            by simp [h]⟩
        · exact ⟨{ channel := AluC, cls := isMinMsg }, by simp [stages], rfl, h⟩
  · intro s hs
    show s.channel.ConsistentUpTo muMax
    simp only [stages, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl <;> infer_instance
  · intro component h
    simp only [ensemble, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl
    -- the providers pull nothing
    iterate 2
      refine ⟨?_, ?_, ?_, trivial⟩ <;>
        exact Component.requirementsAfterOn_of_guarantees_subset (by
          simp [providerComponent_guarantees, countedProviderComponent_guarantees])
    refine ⟨?_, ?_, ?_, trivial⟩
    -- MIN messages: after the LT pull's class and ARE_BYTES
    · apply Component.requirementsAfterOn_of_covered
      intro table h_comp c hc i h_i h_pull
      rw [cminComponent_guarantees] at hc
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      rcases hc with rfl | rfl
      · exact ⟨{ channel := Bitwise.AreBytes.toRaw }, by simp, rfl, rfl⟩
      · rcases cmin_alu_shape h_comp h_i with ⟨-, h_cls⟩ | ⟨h_push, -⟩
        · exact ⟨{ channel := AluC, cls := fun m => !isMinMsg m }, by simp, rfl, by simp [h_cls]⟩
        · simp [h_push] at h_pull
    -- the other ALU messages: only the counted pull, bounded through ARE_BYTES
    · rintro table h_comp - - guarantees i h_i h_cls
      rcases cmin_alu_shape h_comp h_i with ⟨h_pull, -⟩ | ⟨-, h_min⟩
      · exact cmin_pull_requirement h_comp
          (fun j hj => guarantees { channel := Bitwise.AreBytes.toRaw } (by simp) j hj rfl)
          h_i h_pull
      · simp [h_min] at h_cls
    -- nothing owed on ARE_BYTES
    · apply Component.requirementsAfterOn_of_not_mem
      rw [cminComponent_requirements, List.mem_singleton]
      intro h
      have := congrArg RawChannel.name h
      simp [circuit_norm, AluC, Alu.channel, Bitwise.AreBytes, Bitwise.areBytesTable,
        StaticLookupChannel.toChannel, Channel.counted, Channel.toRaw] at this

end LambdaVM.CountedMin
