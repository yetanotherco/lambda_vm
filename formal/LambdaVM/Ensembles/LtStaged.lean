module

public import LambdaVM.Tables.Lt
public import Clean.Air.StagedChannels

@[expose] public section

/-!
# LT with counted pulls (PoC)

`ARE_BYTES`, `IS_HALF` and `MSB16` providers and the LT chip, with `μ` a byte as in `lt.rs`.
LT pulls `IS_HALF` and `MSB16` with multiplicity `μ`, so they are counted channels, and the bound
`μ ≤ 255` comes from LT's own pull of `ARE_BYTES[μ, 0]`. Clean's `OrderedChannel` cannot express
that (LT pulls from and owes requirements on `IS_HALF`); soundness by channel finishing order can:

  ARE_BYTES → IS_HALF (counted) → MSB16 (counted) → ALU

LT's requirements on `IS_HALF` and `MSB16` (`μ = 0 ∨ 1 ≤ μ ≤ 255`) follow from its guarantee on
`ARE_BYTES`, which is finished before them.
-/

namespace LambdaVM.Ensembles.LtStaged
open Air.Flat LambdaVM.LT

variable {Message : TypeMap} [ProvableType Message]

/-- A provider row: pushes `value` `multiplicity` times, assuming `value` satisfies the channel's
guarantee. Stands for the BITWISE table, whose rows are fixed. -/
def provider (channel : Channel (F p) Message) :
    GeneralFormalCircuit (F p) (ProvablePair Message field) unit where
  name := channel.name
  main | (value, multiplicity) => channel.emit multiplicity value
  channelsWithRequirements := [channel.toRaw]
  Assumptions input data := channel.Guarantees input.1 data
  Spec _ _ _ := True
  soundness := by
    circuit_proof_start
    rcases input with ⟨value, multiplicity⟩
    simp_all
  completeness := by circuit_proof_start

/-- The same, on `channel.counted N`. -/
def countedProvider (N : ℕ) (channel : Channel (F p) Message) :
    GeneralFormalCircuit (F p) (ProvablePair Message field) unit where
  name := channel.name
  main | (value, multiplicity) => channel.emitUpTo N multiplicity value
  channelsWithRequirements := [channel.counted N]
  Assumptions input data := channel.Guarantees input.1 data
  Spec _ _ _ := True
  soundness := by
    circuit_proof_start
    rcases input with ⟨value, multiplicity⟩
    simp_all
  completeness := by circuit_proof_start

@[circuit_norm] lemma provider_channelsWithGuarantees (channel : Channel (F p) Message) :
    (provider channel).channelsWithGuarantees = [] := rfl
@[circuit_norm] lemma countedProvider_channelsWithGuarantees (N : ℕ) (channel : Channel (F p) Message) :
    (countedProvider N channel).channelsWithGuarantees = [] := rfl
@[circuit_norm] lemma provider_channelsWithRequirements (channel : Channel (F p) Message) :
    (provider channel).channelsWithRequirements = [channel.toRaw] := rfl
@[circuit_norm] lemma countedProvider_channelsWithRequirements (N : ℕ)
    (channel : Channel (F p) Message) :
    (countedProvider N channel).channelsWithRequirements = [channel.counted N] := rfl
@[circuit_norm] lemma lt_channelsWithRequirements :
    LT.circuit.channelsWithRequirements = [Alu.channel.toRaw, IsHalfC, Msb16C] := rfl
@[circuit_norm] lemma lt_channelsWithGuarantees :
    LT.circuit.channelsWithGuarantees = [Bitwise.AreBytes.toRaw, IsHalfC, IsHalfC, Msb16C, Msb16C,
      IsHalfC, IsHalfC, IsHalfC, IsHalfC] := rfl

/-- Witness generation is not part of this PoC: every component answers demand on its channel. -/
def demandOn (channel : Channel (F p) Message) (Input : TypeMap) [ProvableType Input] :
    WitnessGeneration.Generation (F p) Input :=
  .demand {
    channel
    trigger := .pull
    input _ := pure (fromElements (Vector.replicate (size Input) .multiplicity)) }

def providerComponent (channel : Channel (F p) Message) : Component (F p) where
  circuit := provider channel
  generation := demandOn channel _
  Assumptions input data := channel.Guarantees input.1 data
  circuit_assumptions_of_assumptions := by intro _ _ _ _ _ h; exact h

def countedProviderComponent (N : ℕ) (channel : Channel (F p) Message) : Component (F p) where
  circuit := countedProvider N channel
  generation := demandOn channel _
  Assumptions input data := channel.Guarantees input.1 data
  circuit_assumptions_of_assumptions := by intro _ _ _ _ _ h; exact h

def ltComponent : Component (F p) where
  circuit := LT.circuit
  generation := demandOn Alu.channel _

@[circuit_norm] lemma providerComponent_guarantees (channel : Channel (F p) Message) :
    (providerComponent channel).circuit.channelsWithGuarantees = [] := rfl
@[circuit_norm] lemma providerComponent_requirements (channel : Channel (F p) Message) :
    (providerComponent channel).circuit.channelsWithRequirements = [channel.toRaw] := rfl
@[circuit_norm] lemma countedProviderComponent_guarantees (N : ℕ) (channel : Channel (F p) Message) :
    (countedProviderComponent N channel).circuit.channelsWithGuarantees = [] := rfl
@[circuit_norm] lemma countedProviderComponent_requirements (N : ℕ)
    (channel : Channel (F p) Message) :
    (countedProviderComponent N channel).circuit.channelsWithRequirements =
      [channel.counted N] := rfl
@[circuit_norm] lemma ltComponent_guarantees :
    ltComponent.circuit.channelsWithGuarantees = [Bitwise.AreBytes.toRaw, IsHalfC, IsHalfC, Msb16C,
      Msb16C, IsHalfC, IsHalfC, IsHalfC, IsHalfC] := rfl
@[circuit_norm] lemma ltComponent_requirements :
    ltComponent.circuit.channelsWithRequirements = [Alu.channel.toRaw, IsHalfC, Msb16C] := rfl

/-- Finishing order, newest first. -/
def finished : List (RawChannel (F p)) :=
  [Alu.channel.toRaw, Msb16C, IsHalfC, Bitwise.AreBytes.toRaw]

def ensemble : Ensemble (F p) unit where
  tables := [providerComponent Bitwise.AreBytes, countedProviderComponent muMax Bitwise.IsHalf,
    countedProviderComponent muMax Bitwise.Msb16, ltComponent]
  unique_names := by
    simp [providerComponent, countedProviderComponent, ltComponent, provider, countedProvider,
      LT.circuit, Bitwise.AreBytes, Bitwise.IsHalf, Bitwise.Msb16, Bitwise.areBytesTable,
      Bitwise.isHalfTable, Bitwise.msb16Table, StaticLookupChannel.toChannel]
  channels := finished
  unique_channel_names := by
    simp [finished, circuit_norm, Alu.channel, Bitwise.AreBytes, Bitwise.IsHalf, Bitwise.Msb16,
      Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table]
  tables_channels := by
    intro table h
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;>
      simp only [GeneralFormalCircuit.channels, providerComponent_guarantees, providerComponent_requirements, countedProviderComponent_guarantees, countedProviderComponent_requirements, ltComponent_guarantees, ltComponent_requirements] <;>
      simp [finished, IsHalfC, Msb16C]
  verifier_channels := by simp [circuit_norm]

/-- `μ` from `ARE_BYTES[μ, 0]` is an inactive or counted pull. -/
lemma mu_count {x : F p} (h : x.val < 256) : x = 0 ∨ Channel.CountUpTo muMax x := by
  by_cases h0 : x = 0
  · exact .inl h0
  · exact .inr (Channel.countUpTo_of_val h0 (by simp only [muMax]; omega))

/-- LT's requirements on its counted pulls follow from its guarantee on `ARE_BYTES`. -/
theorem lt_requirementsAfter_counted {c : RawChannel (F p)} (hc : c = IsHalfC ∨ c = Msb16C)
    {earlier : List (RawChannel (F p))} (h_earlier : Bitwise.AreBytes.toRaw ∈ earlier) :
    ltComponent.RequirementsAfter c earlier := by
  rintro table h_comp - constraints guarantees row hrow
  have g := guarantees _ h_earlier row hrow
  cases table
  simp only at h_comp
  subst h_comp
  simp only [Component.channelGuarantees_iff, Component.channelRequirements_iff] at g ⊢
  simp (config := { decide := true }) [circuit_norm, Component.rowOperations, ltComponent,
    LT.circuit, LT.main, isBit, sub, add, Add.carry0, Add.carry1, IsHalfC, Msb16C, Alu.channel,
    Bitwise.AreBytes, Bitwise.IsHalf, Bitwise.Msb16, Bitwise.areBytesTable, Bitwise.isHalfTable,
    Bitwise.msb16Table, StaticLookupChannel.toChannel] at g ⊢
  have hm := mu_count g.1
  rcases hc with rfl | rfl <;>
    simp (config := { decide := true }) [hm, IsHalfC, Msb16C, Bitwise.IsHalf, Bitwise.Msb16,
      Bitwise.isHalfTable, Bitwise.msb16Table, StaticLookupChannel.toChannel]
  -- the ALU push is on another channel
  all_goals
    intro h
    have := congrArg RawChannel.name h
    simp [Channel.counted, Channel.toRaw] at this

theorem tableSoundness : ensemble.TableSoundness (Balance.logUpUpTo muMax) := by
  apply Ensemble.tableSoundness_of_staged finished rfl (List.Subset.refl _)
  · intro component h
    simp only [ensemble, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl <;>
      simp only [providerComponent_guarantees, countedProviderComponent_guarantees,
        ltComponent_guarantees] <;>
      simp [finished, IsHalfC, Msb16C]
  · intro c hc
    show c.ConsistentUpTo muMax
    simp only [finished, List.mem_cons, List.not_mem_nil, or_false] at hc
    rcases hc with rfl | rfl | rfl | rfl <;> infer_instance
  · intro component h
    simp only [ensemble, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl
    -- the providers pull nothing
    iterate 3
      exact Component.stagedRequirements_of_forall fun _ _ _ =>
        Component.requirementsAfter_of_guarantees_subset (by
          simp only [providerComponent_guarantees, countedProviderComponent_guarantees,
            List.nil_subset])
    -- LT: ALU after its pulls, the counted pulls after ARE_BYTES, nothing owed on ARE_BYTES
    refine ⟨?_, lt_requirementsAfter_counted (.inr rfl) (by simp),
      lt_requirementsAfter_counted (.inl rfl) (by simp), ?_, trivial⟩
    · apply Component.requirementsAfter_of_guarantees_subset
      simp only [ltComponent_guarantees]
      simp [IsHalfC, Msb16C]
    · apply Component.requirementsAfter_of_not_mem
      simp only [ltComponent_requirements, List.mem_cons, List.not_mem_nil, or_false, not_or]
      refine ⟨fun h => ?_, fun h => ?_, fun h => ?_⟩ <;> have := congrArg RawChannel.name h <;>
        simp [circuit_norm, IsHalfC, Msb16C, Alu.channel, Bitwise.AreBytes, Bitwise.IsHalf,
          Bitwise.Msb16, Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table,
          StaticLookupChannel.toChannel] at this

end LambdaVM.Ensembles.LtStaged
