module

public import LambdaVM.Tables.Eq
public import Clean.Air.OrderedChannel

@[expose] public section

/-!
# The first ensemble: BITWISE lookups feeding the EQ chip

Three components and three channels, in dependency order: the provider of `IS_HALF`, the
provider of `ZERO`, the EQ chip. Every channel is finished before the next table that pulls
from it, so Clean's ordered-channel soundness applies: in any witness whose constraints hold
and whose channels balance, every EQ row with `μ = 1` computes the equality bit.
-/

namespace LambdaVM.Ensembles
open Air.Flat

/-- The provider side of a static lookup channel: one circuit that emits every entry of the
table with the prover's multiplicity for it. Its row is the vector of multiplicities, the
role the `μ` columns of BITWISE play. -/
def pushStatic {Message : TypeMap} [ProvableType Message] (slc : StaticLookupChannel (F p) Message)
    [NeZero slc.table.length] : GeneralFormalCircuit (F p) (fields slc.table.length) unit where
  name := slc.name
  main mults := do
    let _ ← Circuit.mapFinRange slc.table.length fun i =>
      slc.toChannel.emit mults[i] (const slc.table[i])
  channelsWithRequirements := [slc.toChannel.toRaw]
  Spec _ _ _ := True
  soundness := by
    circuit_proof_start
    intro i _
    exact (slc.guarantees_iff _).mpr (List.getElem_mem _)
  completeness := by circuit_proof_start

/-- The same provider on `slc.toChannel.counted muMax`. A chip that pulls a bus with a
multiplicity needs the counted channel, and `unique_channel_names` forbids the provider from
serving the other spelling of the same bus, so the provider has to follow. The proof does not
change at all: a push owes `mult ≠ 0 → Guarantees` on the counted channel exactly as on `toRaw`,
and only a pull is treated differently. -/
def countedPushStatic {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    GeneralFormalCircuit (F p) (fields slc.table.length) unit where
  name := slc.name
  main mults := do
    let _ ← Circuit.mapFinRange slc.table.length fun i =>
      slc.toChannel.emitUpTo muMax mults[i] (const slc.table[i])
  channelsWithRequirements := [slc.toChannel.counted muMax]
  Spec _ _ _ := True
  soundness := by
    circuit_proof_start
    intro i _
    exact (slc.guarantees_iff _).mpr (List.getElem_mem _)
  completeness := by circuit_proof_start

@[circuit_norm]
lemma countedPushStatic_channelsWithGuarantees {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (countedPushStatic slc).channelsWithGuarantees = [] := by
  simp [circuit_norm, countedPushStatic]

@[circuit_norm]
lemma countedPushStatic_channelsWithRequirements {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (countedPushStatic slc).channelsWithRequirements = [slc.toChannel.counted muMax] := rfl

/-- The counted provider as a component. -/
def countedPushStaticComponent {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] : Component (F p) where
  circuit := countedPushStatic slc
  generation := .demand {
    channel := slc.toChannel
    trigger := .pull
    input _ := pure (fromElements (Vector.replicate slc.table.length .multiplicity)) }

@[circuit_norm] lemma countedPushStaticComponent_channelsWithGuarantees {Message : TypeMap}
    [ProvableType Message] (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (countedPushStaticComponent slc).circuit.channelsWithGuarantees = [] :=
  countedPushStatic_channelsWithGuarantees slc
@[circuit_norm] lemma countedPushStaticComponent_channelsWithRequirements {Message : TypeMap}
    [ProvableType Message] (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (countedPushStaticComponent slc).circuit.channelsWithRequirements =
      [slc.toChannel.counted muMax] := rfl
@[circuit_norm] lemma countedPushStaticComponent_circuit {Message : TypeMap}
    [ProvableType Message] (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (countedPushStaticComponent slc).circuit = countedPushStatic slc := rfl
@[circuit_norm] lemma countedPushStatic_name {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (countedPushStatic slc).name = slc.name := rfl

/-- The provider has no pulls, and requires only its own channel. -/
@[circuit_norm]
lemma pushStatic_channelsWithGuarantees {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (pushStatic slc).channelsWithGuarantees = [] := by
  simp [circuit_norm, pushStatic]

@[circuit_norm]
lemma pushStatic_channelsWithRequirements {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (pushStatic slc).channelsWithRequirements = [slc.toChannel.toRaw] := rfl

/-- The provider as a component. Clean main asks every component for its witness generation;
the multiplicities are counted from the demand on the channel. -/
def pushStaticComponent {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] : Component (F p) where
  circuit := pushStatic slc
  generation := .demand {
    channel := slc.toChannel
    trigger := .pull
    input _ := pure (fromElements (Vector.replicate slc.table.length .multiplicity)) }

/-- The EQ chip as a component, generated from the demand on the ALU bus. EQ pushes its result,
so the demand it answers is a pull: `Demand.direction` is `.pull` when a message was pulled more
often than pushed. -/
def eqComponent : Component (F p) where
  circuit := EQ.circuit
  generation := .demand {
    channel := Alu.channel
    trigger := .pull
    input _ := pure (fromElements (Vector.replicate (size EQ.Row) .multiplicity)) }

@[circuit_norm] lemma pushStaticComponent_circuit {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (pushStaticComponent slc).circuit = pushStatic slc := rfl
@[circuit_norm] lemma pushStatic_name {Message : TypeMap} [ProvableType Message]
    (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (pushStatic slc).name = slc.name := rfl
@[circuit_norm] lemma eqComponent_circuit : eqComponent.circuit = EQ.circuit := rfl
@[circuit_norm] lemma pushStaticComponent_channelsWithGuarantees {Message : TypeMap}
    [ProvableType Message] (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (pushStaticComponent slc).circuit.channelsWithGuarantees = [] :=
  pushStatic_channelsWithGuarantees slc
@[circuit_norm] lemma pushStaticComponent_channelsWithRequirements {Message : TypeMap}
    [ProvableType Message] (slc : StaticLookupChannel (F p) Message) [NeZero slc.table.length] :
    (pushStaticComponent slc).circuit.channelsWithRequirements = [slc.toChannel.toRaw] := rfl
@[circuit_norm] lemma eq_name : EQ.circuit.name = "EQ" := rfl
@[circuit_norm] lemma isHalfTable_name : Bitwise.isHalfTable.name = "IS_HALF" := rfl
@[circuit_norm] lemma zeroTable_name : Bitwise.zeroTable.name = "ZERO" := rfl
@[circuit_norm] lemma alu_name : Alu.channel.name = "ALU" := rfl
@[circuit_norm] lemma eqComponent_channelsWithGuarantees :
    eqComponent.circuit.channelsWithGuarantees =
      [Bitwise.IsHalf.counted muMax, Bitwise.IsHalf.counted muMax, Bitwise.IsHalf.counted muMax,
        Bitwise.IsHalf.counted muMax, Bitwise.Zero.counted muMax] := rfl
@[circuit_norm] lemma eqComponent_channelsWithRequirements :
    eqComponent.circuit.channelsWithRequirements = [Alu.channelC] := rfl

instance : NeZero Bitwise.isHalfTable.table.length :=
  ⟨by simp [Bitwise.isHalfTable, Bitwise.rows, Bitwise.numRows]⟩
instance : NeZero Bitwise.zeroTable.table.length :=
  ⟨by simp [Bitwise.zeroTable, Bitwise.rows, Bitwise.numRows]⟩

/-- The ensemble is stated for `Balance.logUpUpTo muMax` rather than for any balance, because
the ALU bus is the counted channel: finishing it asks for `ConsistentUpTo balance.pullBound`,
and a counted channel is consistent only up to its own `N`. Every ensemble that finishes this
bus inherits that, which is the price of the model agreeing on one ALU channel. It costs
nothing here, since `logUpUpTo` is the balance counted pulls were introduced for and the one
`LtStaged` already uses. -/
def eqLookup : SoundEnsemble (F p) unit (Balance.logUpUpTo muMax) :=
  SoundEnsemble.empty (F p) unit (Balance.logUpUpTo muMax)
  |>.addChannel (Bitwise.IsHalf.counted muMax)
  |>.addTable (countedPushStaticComponent Bitwise.isHalfTable)
      (by rw [countedPushStaticComponent_channelsWithGuarantees]; exact List.nil_subset _)
      (by rw [countedPushStaticComponent_channelsWithRequirements]; simp [circuit_norm])
  |>.markFinished (Bitwise.IsHalf.counted muMax)
  |>.addChannel (Bitwise.Zero.counted muMax)
  |>.addTable (countedPushStaticComponent Bitwise.zeroTable)
      (by rw [countedPushStaticComponent_channelsWithGuarantees]; exact List.nil_subset _)
      (by
        rw [countedPushStaticComponent_channelsWithRequirements]
        simp [circuit_norm]
        -- both buses are counted now, so the names are what tells them apart
        intro h
        have := congrArg RawChannel.name h
        simp [Channel.counted] at this)
  |>.markFinished (Bitwise.Zero.counted muMax)
  |>.addChannel Alu.channelC
  |>.addTable eqComponent
      (by rw [eqComponent_channelsWithGuarantees]; simp [circuit_norm])
      (by
        -- every bus here is counted now, so all three sides are `Channel.counted` and the
        -- mismatch is read off the names alone
        rw [eqComponent_channelsWithRequirements]
        simp only [circuit_norm, Alu.channelC]
        all_goals (and_intros <;>
          (intro h
           have := congrArg RawChannel.name h
           simp [Channel.counted, alu_name] at this)))
  |>.markFinished Alu.channelC

/-- Lookup soundness, concretely: if the constraints of every table hold, every channel has few
enough interactions for the balance check, and every channel balances, then every EQ row with
`μ = 1` and range-checked operands carries the equality bit. -/
theorem eq_rows_correct
    (witness : EnsembleWitness (eqLookup).ensemble)
    (hc : witness.Constraints) (hfit : witness.InteractionsFit (Balance.logUpUpTo muMax))
    (hb : witness.Balanced (Balance.logUpUpTo muMax)) :
    ∀ table ∈ witness.tables, table.component = eqComponent →
      ∀ row ∈ table.table, EQ.RowSpec (valueFromOffset EQ.Row 0 (table.environment row)) := by
  have hsound : (eqLookup).ensemble.TableSoundness (Balance.logUpUpTo muMax) :=
    Ensemble.tableSoundness_of_soundChannels
      ⟨(eqLookup).finished, (eqLookup).finished_subset,
        (eqLookup).soundChannels⟩
  -- no component assumes anything
  have hassume : witness.Assumptions := by
    intro table htable row _
    have hmem := witness.component_mem_tables_of_mem htable
    simp only [eqLookup, circuit_norm, List.mem_cons, List.not_mem_nil, or_false] at hmem
    obtain ⟨comp, tbl, data, uw, ll, frm, dc⟩ := table
    simp only at hmem
    rcases hmem with rfl | rfl | rfl <;> trivial
  have hspec := (hsound witness hassume hc hfit hb).2
  intro table htable hcomp row hrow
  have := hspec table htable row hrow
  obtain ⟨comp, tbl, data, uw, ll, frm, dc⟩ := table
  simp only at hcomp
  subst hcomp
  exact this

end LambdaVM.Ensembles
