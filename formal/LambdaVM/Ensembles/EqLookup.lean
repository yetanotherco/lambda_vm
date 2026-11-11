module

public import LambdaVM.Tables.Eq
public import Clean.Air.OrderedChannel

@[expose] public section

/-!
# The first ensemble: BITWISE lookups feeding the EQ chip

Three components and three channels, in dependency order: the provider of `IS_HALF`, the
provider of `ZERO`, the EQ chip. Every channel is finished before the next table that pulls
from it, so Clean's ordered-channel soundness applies: in any witness whose constraints hold
and whose channels balance, every EQ row with `μ = 1` and range-checked operands computes the
equality bit.

BITWISE is idealized by `pushStatic`: one provider row whose columns are the multiplicities of
every table entry. Nothing pulls from ALU yet (the CPU is not modelled), so balance on the ALU
channel forces `μ = 0` on every EQ row and `eq_rows_correct` is trivially true. Today it shows
that the chain composes; it constrains active rows once a component pulls from ALU.
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
    eqComponent.circuit.channelsWithGuarantees = [Bitwise.IsHalf.toRaw, Bitwise.IsHalf.toRaw,
      Bitwise.IsHalf.toRaw, Bitwise.IsHalf.toRaw, Bitwise.Zero.toRaw] := rfl
@[circuit_norm] lemma eqComponent_channelsWithRequirements :
    eqComponent.circuit.channelsWithRequirements = [Alu.channel.toRaw] := rfl

instance : NeZero Bitwise.isHalfTable.table.length :=
  ⟨by simp [Bitwise.isHalfTable, Bitwise.rows, Bitwise.numRows]⟩
instance : NeZero Bitwise.zeroTable.table.length :=
  ⟨by simp [Bitwise.zeroTable, Bitwise.rows, Bitwise.numRows]⟩

def eqLookup (balance : Balance (F p)) : SoundEnsemble (F p) unit balance :=
  SoundEnsemble.empty (F p) unit balance
  |>.addChannel Bitwise.IsHalf.toRaw
  |>.addTable (pushStaticComponent Bitwise.isHalfTable)
      (by rw [pushStaticComponent_channelsWithGuarantees]; exact List.nil_subset _)
      (by rw [pushStaticComponent_channelsWithRequirements]; simp [circuit_norm])
  |>.markFinished Bitwise.IsHalf.toRaw
  |>.addChannel Bitwise.Zero.toRaw
  |>.addTable (pushStaticComponent Bitwise.zeroTable)
      (by rw [pushStaticComponent_channelsWithGuarantees]; exact List.nil_subset _)
      (by rw [pushStaticComponent_channelsWithRequirements]; simp [circuit_norm])
  |>.markFinished Bitwise.Zero.toRaw
  |>.addChannel Alu.channel.toRaw
  |>.addTable eqComponent
      (by rw [eqComponent_channelsWithGuarantees]; simp [circuit_norm])
      (by rw [eqComponent_channelsWithRequirements]; simp [circuit_norm])
  |>.markFinished Alu.channel.toRaw

/-- Lookup soundness, concretely: if the constraints of every table hold, every channel has few
enough interactions for the balance check, and every channel balances, then every EQ row with
`μ = 1` and range-checked operands carries the equality bit. Without an ALU consumer in the
ensemble no such row exists (see the module header), so the statement is trivially true for now. -/
theorem eq_rows_correct {balance : Balance (F p)}
    (witness : EnsembleWitness (eqLookup balance).ensemble)
    (hc : witness.Constraints) (hfit : witness.InteractionsFit balance)
    (hb : witness.Balanced balance) :
    ∀ table ∈ witness.tables, table.component = eqComponent →
      ∀ row ∈ table.table, EQ.RowSpec (valueFromOffset EQ.Row 0 (table.environment row)) := by
  have hsound : (eqLookup balance).ensemble.TableSoundness balance :=
    Ensemble.tableSoundness_of_soundChannels
      ⟨(eqLookup balance).finished, (eqLookup balance).finished_subset,
        (eqLookup balance).soundChannels⟩
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
