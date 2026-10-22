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
  main mults := do
    let _ ← Circuit.mapFinRange slc.table.length fun i =>
      (Channel.fromStatic (F p) Message slc).emit mults[i] (const slc.table[i])
  channelsWithRequirements := [(Channel.fromStatic (F p) Message slc).toRaw]
  Spec _ _ _ := True
  soundness := by
    circuit_proof_start
    intro i _ _
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
    (pushStatic slc).channelsWithRequirements = [(Channel.fromStatic (F p) Message slc).toRaw] := rfl

lemma RawChannel.ne_of_name_ne {c1 c2 : RawChannel (F p)} (h : c1.name ≠ c2.name) : c1 ≠ c2 :=
  fun e => h (congrArg RawChannel.name e)

instance : NeZero Bitwise.isHalfTable.table.length :=
  ⟨by simp [Bitwise.isHalfTable, Bitwise.rows, Bitwise.numRows]⟩
instance : NeZero Bitwise.zeroTable.table.length :=
  ⟨by simp [Bitwise.zeroTable, Bitwise.rows, Bitwise.numRows]⟩

def eqLookup : SoundEnsemble (F p) unit :=
  SoundEnsemble.empty (F p) unit
  |>.addTable ⟨pushStatic Bitwise.isHalfTable⟩
      (by rw [pushStatic_channelsWithGuarantees]; simp)
      (by simp only [circuit_norm, List.not_mem_nil, false_implies, implies_true])
  |>.addFinishedChannel Bitwise.IsHalf.toRaw
  |>.addTable ⟨pushStatic Bitwise.zeroTable⟩
      (by rw [pushStatic_channelsWithGuarantees]; simp)
      (by
        intro channel hc
        simp only [circuit_norm, List.mem_singleton] at hc
        subst hc
        rw [pushStatic_channelsWithRequirements, List.mem_singleton]
        exact RawChannel.ne_of_name_ne (by decide))
  |>.addFinishedChannel Bitwise.Zero.toRaw
  |>.addTable ⟨EQ.circuit⟩
      (by simp [circuit_norm, EQ.circuit, Bitwise.isHalfTable, Bitwise.zeroTable])
      (by simp [circuit_norm, EQ.circuit, Alu.channel, Bitwise.isHalfTable, Bitwise.zeroTable])
  |>.addFinishedChannel Alu.channel.toRaw

/-- Lookup soundness, concretely: if the constraints of every table hold and every channel
balances, then every EQ row with `μ = 1` and range-checked operands carries the equality bit. -/
theorem eq_rows_correct (witness : EnsembleWitness eqLookup.ensemble)
    (hc : witness.Constraints) (hb : witness.BalancedChannels) :
    ∀ table ∈ witness.tables, table.component = ⟨EQ.circuit⟩ →
      ∀ row ∈ table.table, EQ.RowSpec (valueFromOffset EQ.Row 0 (table.environment row)) := by
  have hsound : eqLookup.ensemble.TableSoundness :=
    Ensemble.tableSoundness_of_soundChannels
      ⟨eqLookup.finished, eqLookup.finished_subset, eqLookup.soundChannels⟩
  have hassume : witness.Assumptions := by
    intro table htable
    have hmem := EnsembleWitness.mem_allTables_component_of_mem_allTables htable
    simp only [circuit_norm, eqLookup, Ensemble.allTables, List.mem_cons, List.not_mem_nil,
      or_false] at hmem
    obtain ⟨comp, width, tbl, data, uw⟩ := table
    simp only at hmem
    intro row _
    rcases hmem with h | h | h | h <;> subst h <;> exact trivial
  have hspec := hsound witness hassume hc hb
  intro table htable hcomp row hrow
  have := hspec table (witness.mem_allTables_of_mem_tables htable) row hrow
  obtain ⟨comp, width, tbl, data, uw⟩ := table
  simp only at hcomp
  subst hcomp
  exact this

end LambdaVM.Ensembles
