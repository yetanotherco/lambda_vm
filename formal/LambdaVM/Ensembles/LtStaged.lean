module

public import LambdaVM.Tables.Lt
public import LambdaVM.Tables.MinToy
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

/-!
## Adding MIN: one table pulls and pushes on the same bus

MIN pulls an LT result from the ALU bus and pushes its own result on the same bus. Ordering by
whole channels can't place it: ALU would have to be finished before MIN (for its pull) and after
it (for its push). Ordering by stages can: the ALU messages that are not MIN are finished first,
then the MIN messages. MIN is one table; only the bus is read in two classes.
-/

namespace LambdaVM.Ensembles.LtStaged
open Air.Flat LambdaVM.LT

def minComponent : Component (F p) where
  circuit := MinToy.circuit
  generation := demandOn Alu.channel _

@[circuit_norm] lemma minComponent_guarantees :
    minComponent.circuit.channelsWithGuarantees = [Alu.channel.toRaw] := rfl
@[circuit_norm] lemma minComponent_requirements :
    minComponent.circuit.channelsWithRequirements = [Alu.channel.toRaw] := rfl

/-- The ALU messages of MIN: read as an ALU message, the flag field is `opsel(MIN)`. -/
def isMinMsg (msg : Array (F p)) : Bool :=
  if h : msg.size = size Alu.Msg then
    decide ((fromElements (⟨msg, h⟩ : Vector (F p) (size Alu.Msg)) : Alu.Msg (F p)).flags = 9)
  else false

lemma isMinMsg_toElements (m : Alu.Msg (F p)) :
    isMinMsg (toElements m).toArray = decide (m.flags = 9) := by
  simp [isMinMsg, ProvableType.fromElements_toElements]

/-- The message of an evaluated interaction, as the evaluated typed message. -/
lemma msg_eval {M : TypeMap} [ProvableType M] (env : Environment (F p)) (x : M (Expression (F p))) :
    ((toElements x).map (Expression.eval env)).toArray = (toElements (eval env x)).toArray := by
  rw [← ProvableType.fromElements_eval_toElements (env := env) x,
    ProvableType.toElements_fromElements, Vector.toArray_map]

/-- Each row of MIN has two interactions on ALU: the LT pull, which is not a MIN message, and
the MIN push. -/
lemma min_alu_shape {table : Table (F p)} (h_comp : table.component = minComponent)
    {i : Interaction (F p)} (h_i : i ∈ table.interactionsWith Alu.channel.toRaw) :
    (i.direction = .pull ∧ isMinMsg i.msg = false) ∨
      (i.direction = .push ∧ isMinMsg i.msg = true) := by
  cases table
  simp only at h_comp
  subst h_comp
  simp [Table.interactionsWith, List.mem_flatMap, Operations.interactionValuesWith_eq_map,
    Component.interactionsWith_eq, Component.rowOperations, minComponent, MinToy.circuit,
    MinToy.main, isBit, circuit_norm] at h_i
  obtain ⟨row, -, rfl | rfl⟩ := h_i
  · left
    simp only [AbstractInteraction.eval, ChannelInteraction.toRaw, pulled, msg_eval,
      isMinMsg_toElements]
    simp (config := { decide := true }) [circuit_norm]
  · right
    simp only [AbstractInteraction.eval, ChannelInteraction.toRaw, pushedIf, msg_eval,
      isMinMsg_toElements]
    simp (config := { decide := true }) [circuit_norm]

/-- MIN's pull, of multiplicity `1`, owes nothing beyond its multiplicity. -/
lemma min_pull_requirements {table : Table (F p)} (h_comp : table.component = minComponent)
    (constraints : table.Constraints) {i : Interaction (F p)}
    (h_i : i ∈ table.interactionsWith Alu.channel.toRaw) (h_pull : i.direction = .pull) :
    i.Requirements table.data := by
  cases table
  simp only at h_comp
  subst h_comp
  simp [Table.interactionsWith, List.mem_flatMap, Operations.interactionValuesWith_eq_map,
    Component.interactionsWith_eq, Component.rowOperations, minComponent, MinToy.circuit,
    MinToy.main, isBit, circuit_norm] at h_i
  obtain ⟨row, -, rfl | rfl⟩ := h_i
  · simp [AbstractInteraction.eval, ChannelInteraction.toRaw, pulled, Interaction.Requirements,
      Channel.toRaw]
    exact .inr (by simp [circuit_norm])
  · simp [AbstractInteraction.eval, ChannelInteraction.toRaw, pushedIf] at h_pull

/-- Stages, newest first. The ALU bus is finished in two classes: first every message that is
not MIN, then the MIN messages. -/
def stages : List (Stage (F p)) :=
  [{ channel := Alu.channel.toRaw, cls := isMinMsg },
   { channel := Alu.channel.toRaw, cls := fun m => !isMinMsg m },
   { channel := Msb16C }, { channel := IsHalfC }, { channel := Bitwise.AreBytes.toRaw }]

def ensembleMin : Ensemble (F p) unit where
  tables := [providerComponent Bitwise.AreBytes, countedProviderComponent muMax Bitwise.IsHalf,
    countedProviderComponent muMax Bitwise.Msb16, ltComponent, minComponent]
  unique_names := by
    simp [providerComponent, countedProviderComponent, ltComponent, minComponent, provider,
      countedProvider, LT.circuit, MinToy.circuit, Bitwise.AreBytes, Bitwise.IsHalf, Bitwise.Msb16,
      Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table, StaticLookupChannel.toChannel]
  channels := finished
  unique_channel_names := ensemble.unique_channel_names
  tables_channels := by
    intro table h
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl <;>
      simp only [GeneralFormalCircuit.channels, providerComponent_guarantees,
        providerComponent_requirements, countedProviderComponent_guarantees,
        countedProviderComponent_requirements, ltComponent_guarantees, ltComponent_requirements,
        minComponent_guarantees, minComponent_requirements] <;>
      simp [finished, IsHalfC, Msb16C]
  verifier_channels := by simp [circuit_norm]

/-- A channel other than ALU, by name. -/
lemma ne_alu {c : RawChannel (F p)}
    (hc : c = IsHalfC ∨ c = Msb16C ∨ c = Bitwise.AreBytes.toRaw) : c ∉ [Alu.channel.toRaw] := by
  rw [List.mem_singleton]
  intro h
  have := congrArg RawChannel.name h
  rcases hc with rfl | rfl | rfl <;>
    simp [circuit_norm, IsHalfC, Msb16C, Alu.channel, Bitwise.AreBytes, Bitwise.IsHalf,
      Bitwise.Msb16, Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table,
      StaticLookupChannel.toChannel, Channel.counted, Channel.toRaw] at this

theorem tableSoundness_withMin : ensembleMin.TableSoundness (Balance.logUpUpTo muMax) := by
  apply Ensemble.tableSoundness_of_stagedOn stages rfl
  · intro s hs
    simp only [stages, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl <;> simp [ensembleMin, finished]
  · intro component h c hc msg
    simp only [ensembleMin, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl
    iterate 3 simp [providerComponent_guarantees, countedProviderComponent_guarantees] at hc
    · rw [ltComponent_guarantees] at hc
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
      rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp [stages]
    · rw [minComponent_guarantees, List.mem_singleton] at hc
      subst hc
      cases h : isMinMsg msg
      · exact ⟨{ channel := Alu.channel.toRaw, cls := fun m => !isMinMsg m }, by simp [stages],
          rfl, by simp [h]⟩
      · exact ⟨{ channel := Alu.channel.toRaw, cls := isMinMsg }, by simp [stages], rfl, h⟩
  · intro s hs
    show s.channel.ConsistentUpTo muMax
    simp only [stages, List.mem_cons, List.not_mem_nil, or_false] at hs
    rcases hs with rfl | rfl | rfl | rfl | rfl <;> infer_instance
  · intro component h
    simp only [ensembleMin, List.mem_cons, List.not_mem_nil, or_false] at h
    rcases h with rfl | rfl | rfl | rfl | rfl
    -- the providers pull nothing
    iterate 3
      refine ⟨?_, ?_, ?_, ?_, ?_, trivial⟩ <;>
        exact Component.requirementsAfterOn_of_guarantees_subset (by
          simp [providerComponent_guarantees, countedProviderComponent_guarantees])
    -- LT: both ALU classes after its pulls, the counted pulls after ARE_BYTES
    · refine ⟨?_, ?_, ?_, ?_, ?_, trivial⟩
      iterate 2
        apply Component.requirementsAfterOn_of_guarantees_subset
        rw [ltComponent_guarantees]
        intro c hc msg
        simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
        rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> simp
      · exact Component.requirementsAfterOn_of_requirementsAfter
          (lt_requirementsAfter_counted (earlier := [IsHalfC, Bitwise.AreBytes.toRaw]) (.inr rfl)
            (by simp))
          (by
            intro c hc
            simp only [List.mem_cons, List.not_mem_nil, or_false] at hc
            rcases hc with rfl | rfl
            · exact ⟨{ channel := IsHalfC }, by simp, rfl, fun _ => rfl⟩
            · exact ⟨{ channel := Bitwise.AreBytes.toRaw }, by simp, rfl, fun _ => rfl⟩)
      · exact Component.requirementsAfterOn_of_requirementsAfter
          (lt_requirementsAfter_counted (earlier := [Bitwise.AreBytes.toRaw]) (.inl rfl) (by simp))
          (by
            intro c hc
            rw [List.mem_singleton] at hc
            subst hc
            exact ⟨{ channel := Bitwise.AreBytes.toRaw }, by simp, rfl, fun _ => rfl⟩)
      · apply Component.requirementsAfterOn_of_not_mem
        simp only [ltComponent_requirements, List.mem_cons, List.not_mem_nil, or_false, not_or]
        refine ⟨fun h => ?_, fun h => ?_, fun h => ?_⟩ <;> have := congrArg RawChannel.name h <;>
          simp [circuit_norm, IsHalfC, Msb16C, Alu.channel, Bitwise.AreBytes, Bitwise.IsHalf,
            Bitwise.Msb16, Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table,
            StaticLookupChannel.toChannel] at this
    -- MIN: the MIN class after the other ALU class, which holds its LT pull
    · refine ⟨?_, ?_, ?_, ?_, ?_, trivial⟩
      · apply Component.requirementsAfterOn_of_covered
        intro table h_comp c hc i h_i h_pull
        rw [minComponent_guarantees, List.mem_singleton] at hc
        subst hc
        rcases min_alu_shape h_comp h_i with ⟨-, h_cls⟩ | ⟨h_push, -⟩
        · exact ⟨{ channel := Alu.channel.toRaw, cls := fun m => !isMinMsg m }, by simp,
            rfl, by simp [h_cls]⟩
        · simp [h_push] at h_pull
      · -- on the other class MIN owes only its pull's multiplicity, from its constraints
        rintro table h_comp - constraints - i h_i h_cls
        rcases min_alu_shape h_comp h_i with ⟨h_pull, -⟩ | ⟨-, h_min⟩
        · exact min_pull_requirements h_comp constraints h_i h_pull
        · simp [h_min] at h_cls
      all_goals
        apply Component.requirementsAfterOn_of_not_mem
        rw [minComponent_requirements]
        exact ne_alu (by simp)

end LambdaVM.Ensembles.LtStaged
