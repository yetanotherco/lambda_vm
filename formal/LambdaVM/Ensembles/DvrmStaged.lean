module

public import LambdaVM.Tables.Lt
public import LambdaVM.Tables.Mul
public import LambdaVM.Tables.Dvrm
public import LambdaVM.Ensembles.EqLookup
public import LambdaVM.Buses.Canonical
public import LambdaVM.Ensembles.LtStaged
public import Clean.Air.StagedChannels

@[expose] public section

/-!
# The DVRM ensemble

Eight components: the five providers of the static lookups DVRM's dependency cone needs, and
the three chips LT, MUL and DVRM.

DVRM is the first chip that both receives and owes on the same bus. It pulls `ALU` three times,
once for a `LT` comparison (`|r| < |d|`) and twice for `MUL` products (`d·q = n − r`), and pushes
`ALU` twice with its own `DIVREM` results. Ordering whole channels cannot express that: `ALU`
would have to be finished both before and after DVRM. Ordering *stages* can, a stage being a
channel together with a class of its messages, so the `DIVREM` messages are finished after the
`LT` and `MUL` ones. That is what `Ensemble.tableSoundness_of_stagedOn` is for, and this is its
only user in either repo.

The two classes are read off the `flags` field of the message, which carries the `alu_op`
descriptor: `opselDIVREM = 8` for what DVRM pushes, `opselLT = 4` and `opselMUL = 7` for what it
pulls.
-/

namespace LambdaVM.Ensembles.DvrmStaged
open Air.Flat

instance : NeZero Bitwise.areBytesTable.table.length :=
  ⟨by simp [Bitwise.areBytesTable, Bitwise.rows, Bitwise.numRows]⟩
instance : NeZero Bitwise.msb16Table.table.length :=
  ⟨by simp [Bitwise.msb16Table, Bitwise.rows, Bitwise.numRows]⟩
instance : NeZero Bitwise.isB20Table.table.length :=
  ⟨by simp [Bitwise.isB20Table, Bitwise.rows, Bitwise.numRows]⟩

/-- Witness generation is not what this ensemble is about; every chip answers the demand on the
bus it pushes. -/
def demandOnAlu (Input : TypeMap) [ProvableType Input] : WitnessGeneration.Generation (F p) Input :=
  .demand {
    channel := Alu.channel
    trigger := .pull
    input _ := pure (fromElements (Vector.replicate (size Input) .multiplicity)) }

/-- LT's component is the one `LtStaged` already built, so that its counted-pull argument --
that `μ` is a count because the chip's own `ARE_BYTES[μ, 0]` pull says so -- applies here
instead of being written again. The two agree: both answer the demand on the ALU bus. -/
abbrev ltComponent : Component (F p) := LtStaged.ltComponent

def mulComponent : Component (F p) where
  circuit := MUL.circuit
  generation := demandOnAlu _

def dvrmComponent : Component (F p) where
  circuit := DVRM.circuit
  generation := demandOnAlu _

@[circuit_norm] lemma ltComponent_circuit : ltComponent.circuit = LT.circuit := rfl
@[circuit_norm] lemma mulComponent_circuit : mulComponent.circuit = MUL.circuit := rfl
@[circuit_norm] lemma dvrmComponent_circuit : dvrmComponent.circuit = DVRM.circuit := rfl

@[circuit_norm] lemma lt_name : LT.circuit.name = "LT" := rfl
@[circuit_norm] lemma mul_name : MUL.circuit.name = "MUL" := rfl
@[circuit_norm] lemma dvrm_name : DVRM.circuit.name = "DVRM" := rfl
@[circuit_norm] lemma msb16Table_name : Bitwise.msb16Table.name = "MSB16" := rfl
@[circuit_norm] lemma isB20Table_name : Bitwise.isB20Table.name = "IS_B20" := rfl
@[circuit_norm] lemma areBytesTable_name : Bitwise.areBytesTable.name = "ARE_BYTES" := rfl
@[circuit_norm] lemma byteAluTable_name : Bitwise.byteAluTable.name = "BYTE_ALU" := rfl
@[circuit_norm] lemma hwslTable_name : Bitwise.hwslTable.name = "HWSL" := rfl

/-- Two buses are different channels when their names differ, which is the only way to tell
`counted` apart from `counted`: the other two fields are closures into `Prop`. -/
macro "channel_ne" : tactic =>
  `(tactic| (intro h
             have := congrArg RawChannel.name h
             -- the list covers every bus of the model, so at any one site most of it is
             -- unused; that is the point of having it in one place
             set_option linter.unusedSimpArgs false in
             simp [Channel.counted, Channel.toRaw, Alu.channel, StaticLookupChannel.toChannel,
               Bitwise.AreBytes, Bitwise.IsHalf, Bitwise.Msb16, Bitwise.IsB20, Bitwise.Zero,
               Bitwise.ByteAlu, Bitwise.Hwsl, Bitwise.areBytesTable, Bitwise.isHalfTable,
               Bitwise.msb16Table, Bitwise.isB20Table, Bitwise.zeroTable, Bitwise.byteAluTable,
               Bitwise.hwslTable] at this))

@[circuit_norm] lemma ltComponent_guarantees :
    ltComponent.circuit.channelsWithGuarantees = [Bitwise.AreBytes.toRaw, LT.IsHalfC, LT.IsHalfC,
      LT.Msb16C, LT.Msb16C, LT.IsHalfC, LT.IsHalfC, LT.IsHalfC, LT.IsHalfC] := rfl
@[circuit_norm] lemma ltComponent_requirements :
    ltComponent.circuit.channelsWithRequirements = [Alu.channelC, LT.IsHalfC, LT.Msb16C] := rfl
/-- MUL's twenty-three pulls, in the order `MUL.main` emits them: the `ARE_BYTES` that bounds
its two multiplicities, the eight halves of its operands, the two `MSB16` of the `SIGN`
templates, the eight halves of its result, and the four `IS_B20` carries. -/
@[circuit_norm] lemma mulComponent_guarantees :
    mulComponent.circuit.channelsWithGuarantees =
      [Bitwise.AreBytes.toRaw,
       MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC,
       MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC,
       MUL.Msb16C, MUL.Msb16C,
       MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC,
       MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC, MUL.IsHalfC,
       MUL.IsB20C, MUL.IsB20C, MUL.IsB20C, MUL.IsB20C] := rfl

@[circuit_norm] lemma mulComponent_requirements :
    mulComponent.circuit.channelsWithRequirements =
      [Alu.channelC, MUL.IsHalfC, MUL.IsB20C] := rfl
@[circuit_norm] lemma dvrmComponent_requirements :
    dvrmComponent.circuit.channelsWithRequirements =
      [DVRM.AluC, DVRM.IsHalfC, DVRM.ZeroC] := rfl

def ensemble : Ensemble (F p) unit where
  tables :=
    [ pushStaticComponent Bitwise.areBytesTable
    , countedPushStaticComponent Bitwise.isHalfTable
    , countedPushStaticComponent Bitwise.msb16Table
    , countedPushStaticComponent Bitwise.isB20Table
    , countedPushStaticComponent Bitwise.zeroTable
    , ltComponent, mulComponent, dvrmComponent ]
  unique_names := by
    simp [circuit_norm, pushStaticComponent, countedPushStaticComponent, pushStatic,
      countedPushStatic, ltComponent, mulComponent, dvrmComponent, LT.circuit, MUL.circuit,
      DVRM.circuit, Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table,
      Bitwise.isB20Table, Bitwise.zeroTable]
  -- the model's bus list, which is what `Buses/Canonical.lean` checks the chips against. Two
  -- of its eight, `BYTE_ALU` and `HWSL`, are not touched here; a channel with no interaction
  -- balances for free, and `finished` only has to be a subset.
  channels := Buses.modelChannels
  unique_channel_names := Buses.modelChannels_unique_names
  -- the three chips are exactly what `model_channels_agree` says; the five providers push one
  -- bus each and pull none
  tables_channels := by
    intro table h
    simp only [List.mem_cons, List.not_mem_nil, or_false] at h
    obtain ⟨hlt, -, -, -, hmul, hdvrm⟩ := Buses.model_channels_agree
    rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
    -- the five providers push one bus each and pull none
    iterate 5
      simp only [GeneralFormalCircuit.channels, pushStaticComponent_channelsWithGuarantees,
        pushStaticComponent_channelsWithRequirements,
        countedPushStaticComponent_channelsWithGuarantees,
        countedPushStaticComponent_channelsWithRequirements]
      simp [Buses.modelChannels]
    -- the three chips are what the guard already checked
    · exact hlt
    · exact hmul
    · exact hdvrm
  verifier_channels := by simp [circuit_norm]

/-! ## The two classes of `ALU` messages

An `Alu.Msg` is seven field elements and `flags` is the fifth, so the class is read off
`msg[4]`. DVRM pushes `opselDIVREM + 32 * signed`, and the same again `+ 128` for the
remainder half, which is four values. LT pushes `opselLT + 32 * signed + 64 * invert` and MUL
`opselMUL + 32 * lhsSigned + 64 * rhsSigned (+ 128)`, none of which are among them. -/

/-- The four flag bytes DVRM writes. -/
def divremFlags : List (F p) := [8, 40, 136, 168]

/-- The `DIVREM` class: what DVRM pushes. -/
def IsDivrem (msg : Array (F p)) : Bool := divremFlags.contains (msg.getD 4 0)

/-- What DVRM pushes, finished last. -/
def stageDivrem : Stage (F p) := { channel := Alu.channelC, cls := IsDivrem }

/-- What DVRM pulls: the `LT` and `MUL` results, finished before it. -/
def stageQuery : Stage (F p) := { channel := Alu.channelC, cls := fun msg => !IsDivrem msg }

/-- Finishing order, newest first. The `ALU` bus is split in two, which is the whole point:
DVRM receives on `stageQuery` and owes on `stageDivrem`, so one has to come after the other
and a whole-channel order cannot say that. -/
def finished : List (Stage (F p)) :=
  [ stageDivrem
  , stageQuery
  , { channel := Bitwise.Zero.counted muMax }
  , { channel := Bitwise.IsB20.counted muMax }
  , { channel := Bitwise.Msb16.counted muMax }
  , { channel := Bitwise.IsHalf.counted muMax }
  , { channel := Bitwise.ByteAlu.counted muMax }
  , { channel := Bitwise.Hwsl.toRaw }
  , { channel := Bitwise.AreBytes.toRaw } ]

theorem finished_channels : ∀ s ∈ finished, s.channel ∈ ensemble.channels := by
  intro s hs
  simp only [finished, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    simp [ensemble, Buses.modelChannels, stageDivrem, stageQuery]

theorem consistent : ∀ s ∈ finished,
    s.channel.ConsistentUpTo (Balance.logUpUpTo (F := F p) muMax).pullBound := by
  intro s hs
  simp only [finished, List.mem_cons, List.not_mem_nil, or_false] at hs
  rcases hs with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;>
    first
      | exact inferInstance
      | (show (Alu.channelC : RawChannel (F p)).ConsistentUpTo _; exact inferInstance)

/-- Every bus of the model is the channel of a finished stage, and `ALU` of two whose classes
between them hold every message. `BYTE_ALU` and `HWSL` are here only so that this reads off
`modelChannels` rather than off a hand-written list of what each chip receives on: the guard
already proved the chips stay inside that list, and those two are the entries no chip of this
ensemble touches. -/
theorem modelChannel_covered (c : RawChannel (F p)) (hc : c ∈ Buses.modelChannels)
    (msg : Array (F p)) : ∃ s ∈ finished, s.channel = c ∧ s.cls msg := by
  simp only [Buses.modelChannels, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · -- `ALU`: a message is `DIVREM` or it is not, and there is a stage for each
    by_cases h : IsDivrem msg
    · exact ⟨stageDivrem, by simp [finished], rfl, h⟩
    · exact ⟨stageQuery, by simp [finished], rfl, by simp [stageQuery, h]⟩
  -- the rest are whole-channel stages, in the order `modelChannels` lists them
  · exact ⟨{ channel := Bitwise.AreBytes.toRaw }, by simp [finished], rfl, rfl⟩
  · exact ⟨{ channel := Bitwise.ByteAlu.counted muMax }, by simp [finished], rfl, rfl⟩
  · exact ⟨{ channel := Bitwise.Hwsl.toRaw }, by simp [finished], rfl, rfl⟩
  · exact ⟨{ channel := Bitwise.IsB20.counted muMax }, by simp [finished], rfl, rfl⟩
  · exact ⟨{ channel := Bitwise.IsHalf.counted muMax }, by simp [finished], rfl, rfl⟩
  · exact ⟨{ channel := Bitwise.Msb16.counted muMax }, by simp [finished], rfl, rfl⟩
  · exact ⟨{ channel := Bitwise.Zero.counted muMax }, by simp [finished], rfl, rfl⟩

/-- Every pull of every table lands in a finished stage. This is `tables_channels` composed
with the covering above, so it costs nothing beyond what the guard already proved: no list of
what each chip receives on is written out here. -/
theorem covered : ∀ component ∈ ensemble.tables,
    ∀ c ∈ component.circuit.channelsWithGuarantees, ∀ msg,
      ∃ s ∈ finished, s.channel = c ∧ s.cls msg := by
  intro component h c hc msg
  refine modelChannel_covered c ?_ msg
  exact ensemble.tables_channels component h (List.mem_append_left _ hc)

/-! ## Staged requirements

`Component.StagedRequirementsOn` walks the finishing order and asks, at each stage, that the
component's requirements there follow from its guarantees on everything finished before. -/

/-- A component that receives no guarantee owes nothing anywhere: its requirements are already
unconditional. That is the five providers, which push their table and pull nothing. -/
theorem stagedRequirementsOn_of_no_guarantees {component : Component (F p)}
    (h : component.circuit.channelsWithGuarantees = []) :
    ∀ stages : List (Stage (F p)), component.StagedRequirementsOn stages
  | [] => trivial
  | _ :: rest =>
    ⟨Component.requirementsAfterOn_of_guarantees_subset (by simp [h]),
     stagedRequirementsOn_of_no_guarantees h rest⟩

/-- LT receives on `ARE_BYTES`, `IS_HALF` and `MSB16` only, so once those three are finished it
owes whatever soundness gives it. -/
theorem lt_guarantees_in {earlier : List (Stage (F p))}
    (hab : ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier)
    (hih : ({ channel := Bitwise.IsHalf.counted muMax } : Stage (F p)) ∈ earlier)
    (hms : ({ channel := Bitwise.Msb16.counted muMax } : Stage (F p)) ∈ earlier) :
    ∀ c ∈ ltComponent.circuit.channelsWithGuarantees, ∀ msg : Array (F p),
      ∃ s ∈ earlier, s.channel = c ∧ s.cls msg := by
  intro c hc msg
  simp only [ltComponent_guarantees, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨_, hab, rfl, rfl⟩
  all_goals first
    | exact ⟨_, hih, rfl, rfl⟩
    | exact ⟨_, hms, rfl, rfl⟩

/-- LT's nine obligations. Five stages are buses it does not require, so it owes nothing there.
On the two `ALU` stages everything it receives is already finished. The two that carry weight are
`IS_HALF` and `MSB16`, where the counted pull's bound comes from `ARE_BYTES`, finished last. -/
theorem lt_staged : ltComponent.StagedRequirementsOn finished := by
  have easy : ∀ {earlier : List (Stage (F p))} (stage : Stage (F p)),
      ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier →
      ({ channel := Bitwise.IsHalf.counted muMax } : Stage (F p)) ∈ earlier →
      ({ channel := Bitwise.Msb16.counted muMax } : Stage (F p)) ∈ earlier →
      ltComponent.RequirementsAfterOn stage earlier :=
    fun _ hab hih hms =>
      Component.requirementsAfterOn_of_guarantees_subset (lt_guarantees_in hab hih hms)
  have counted : ∀ {c : RawChannel (F p)}, (c = LT.IsHalfC ∨ c = LT.Msb16C) →
      ∀ {earlier : List (Stage (F p))},
      ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier →
      ltComponent.RequirementsAfterOn { channel := c } earlier := by
    intro c hc earlier hab
    refine Component.requirementsAfterOn_of_requirementsAfter
      (LtStaged.lt_requirementsAfter_counted hc (List.mem_singleton_self _)) ?_
    intro c' hc'
    simp only [List.mem_singleton] at hc'
    exact ⟨_, hc' ▸ hab, rfl, fun _ => rfl⟩
  refine ⟨easy _ (by simp) (by simp) (by simp),
    easy _ (by simp) (by simp) (by simp),
    ?_, ?_, counted (.inr rfl) (by simp), counted (.inl rfl) (by simp),
    ?_, ?_, ?_, trivial⟩
  all_goals
    refine Component.requirementsAfterOn_of_not_mem _ ?_
    simp only [LtStaged.ltComponent_requirements, Alu.channelC, LT.IsHalfC, LT.Msb16C,
      List.mem_cons, List.not_mem_nil, or_false, not_or]
    and_intros <;> channel_ne

/-- MUL's multiplicity is the sum of two bytes, so `ARE_BYTES[μ_lo, μ_hi]` bounds it by `510`,
which is `muMax`. LT's is a single byte and its lemma says `< 256`; this is the two-byte form. -/
lemma mul_mu_count {a b : F p} (ha : a.val < 256) (hb : b.val < 256) :
    a + b = 0 ∨ Channel.CountUpTo muMax (a + b) := by
  by_cases h0 : a + b = 0
  · exact .inl h0
  refine .inr (Channel.countUpTo_of_val h0 ?_)
  have hsum : (a + b).val = a.val + b.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  simp only [hsum, muMax]
  omega

/-- MUL's requirements on its two counted buses follow from its guarantee on `ARE_BYTES`.

Only two, because MUL's `MSB16` pulls carry a sign flag rather than the multiplicity, and a flag
is a bit by the chip's own constraint, so that obligation is met in `Mul.lean` and never reaches
an ensemble. `IS_HALF` and `IS_B20` both carry `μ_lo + μ_hi`, which is what `ARE_BYTES` bounds.

`MUL.circuit` carries an explicit `elaborated` where `LT.circuit` takes the default, so the simp
set needs `MUL.elaborated` to reach the operations at all. The guarantee then comes out as the
pair of byte bounds followed by one vacuous implication per other channel, which is why the two
bounds are `g.1.1` and `g.1.2`. -/
theorem mul_requirementsAfter_counted {c : RawChannel (F p)}
    (hc : c = MUL.IsHalfC ∨ c = MUL.IsB20C)
    {earlier : List (RawChannel (F p))} (h_earlier : Bitwise.AreBytes.toRaw ∈ earlier) :
    mulComponent.RequirementsAfter c earlier := by
  rintro table h_comp - constraints guarantees row hrow
  have g := guarantees _ h_earlier row hrow
  cases table
  simp only at h_comp
  subst h_comp
  simp only [Component.channelGuarantees_iff, Component.channelRequirements_iff] at g ⊢
  simp (config := { decide := true }) [circuit_norm, Component.rowOperations, mulComponent,
    MUL.circuit, MUL.elaborated, MUL.main, areBytes, isBit, sign, MUL.word, MUL.rawE,
    MUL.IsHalfC, MUL.IsB20C, Alu.channel, Bitwise.AreBytes, Bitwise.IsHalf,
    Bitwise.Msb16, Bitwise.IsB20, Bitwise.areBytesTable, Bitwise.isHalfTable,
    Bitwise.msb16Table, Bitwise.isB20Table, StaticLookupChannel.toChannel] at g ⊢
  have hm := mul_mu_count g.1.1 g.1.2
  rcases hc with rfl | rfl <;>
    simp (config := { decide := true }) [hm, MUL.IsHalfC, MUL.IsB20C, Bitwise.IsHalf,
      Bitwise.IsB20, Bitwise.isHalfTable, Bitwise.isB20Table, StaticLookupChannel.toChannel]
  -- the two ALU pushes are on another channel
  all_goals (and_intros <;> channel_ne)

/-- MUL receives on `ARE_BYTES`, `IS_HALF`, `MSB16` and `IS_B20`, all whole-channel stages. -/
theorem mul_guarantees_in {earlier : List (Stage (F p))}
    (hab : ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier)
    (hih : ({ channel := Bitwise.IsHalf.counted muMax } : Stage (F p)) ∈ earlier)
    (hms : ({ channel := Bitwise.Msb16.counted muMax } : Stage (F p)) ∈ earlier)
    (hb2 : ({ channel := Bitwise.IsB20.counted muMax } : Stage (F p)) ∈ earlier) :
    ∀ c ∈ mulComponent.circuit.channelsWithGuarantees, ∀ msg : Array (F p),
      ∃ s ∈ earlier, s.channel = c ∧ s.cls msg := by
  intro c hc msg
  simp only [mulComponent_guarantees, List.mem_cons, List.not_mem_nil, or_false] at hc
  rcases hc with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl |
    rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨_, hab, rfl, rfl⟩
  all_goals first
    | exact ⟨_, hih, rfl, rfl⟩
    | exact ⟨_, hms, rfl, rfl⟩
    | exact ⟨_, hb2, rfl, rfl⟩

/-- MUL's nine obligations. Five stages are buses it does not require -- `MSB16` among them now,
since its sign flags are bits by its own constraints -- so it owes nothing there. On the two
`ALU` stages everything it receives is already finished. `IS_B20` and `IS_HALF` are the two that
carry weight, and both get their bound from `ARE_BYTES`. -/
theorem mul_staged : mulComponent.StagedRequirementsOn finished := by
  have easy : ∀ {earlier : List (Stage (F p))} (stage : Stage (F p)),
      ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier →
      ({ channel := Bitwise.IsHalf.counted muMax } : Stage (F p)) ∈ earlier →
      ({ channel := Bitwise.Msb16.counted muMax } : Stage (F p)) ∈ earlier →
      ({ channel := Bitwise.IsB20.counted muMax } : Stage (F p)) ∈ earlier →
      mulComponent.RequirementsAfterOn stage earlier :=
    fun _ hab hih hms hb2 =>
      Component.requirementsAfterOn_of_guarantees_subset (mul_guarantees_in hab hih hms hb2)
  have counted : ∀ {c : RawChannel (F p)}, (c = MUL.IsHalfC ∨ c = MUL.IsB20C) →
      ∀ {earlier : List (Stage (F p))},
      ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier →
      mulComponent.RequirementsAfterOn { channel := c } earlier := by
    intro c hc earlier hab
    refine Component.requirementsAfterOn_of_requirementsAfter
      (mul_requirementsAfter_counted hc (List.mem_singleton_self _)) ?_
    intro c' hc'
    simp only [List.mem_singleton] at hc'
    exact ⟨_, hc' ▸ hab, rfl, fun _ => rfl⟩
  refine ⟨easy _ (by simp) (by simp) (by simp) (by simp),
    easy _ (by simp) (by simp) (by simp) (by simp),
    ?_, counted (.inr rfl) (by simp), ?_, counted (.inl rfl) (by simp), ?_, ?_, ?_, trivial⟩
  all_goals
    refine Component.requirementsAfterOn_of_not_mem _ ?_
    simp only [mulComponent_requirements, Alu.channelC, MUL.IsHalfC, MUL.IsB20C,
      List.mem_cons, List.not_mem_nil, or_false, not_or]
    and_intros <;> channel_ne

/-! ## DVRM

The chip at the top of the ensemble: it pulls `LT` and `MUL` answers off the `ALU` bus and pushes
`DIVREM` answers back onto it, which is the whole reason the bus is staged by message class.

Its multiplicities come in three kinds, and only one of them is the row's `mu = mu_q + mu_r` that
`ARE_BYTES` bounds. The `ARE_BYTES` pull itself fires with a literal `1`, and the four `ZERO`
pulls that `NEG` inlines fire with a sign flag, which is a bit only because `MSB16` says so. The
first kind is `Mul.lean`'s argument again; the second is met in `Dvrm.lean`; the third is why
`MSB16` has to be finished before `ZERO`. -/

/-- DVRM's thirty-three pulls, in the order `DVRM.main` emits them: the `ARE_BYTES` that bounds
`mu`, the eight halves of the operands, the borrowed `LT` answer, the four `ZERO` of the two `NEG`
templates and the one that decides overflow, the eight halves of the remainder and of `n - r`, the
two borrowed `MUL` answers, the four halves of the quotient, the three `MSB16` of the `SIGN`
templates, and the `ZERO` that decides division by zero. The two `DIVREM` pushes are not here: a
pull guarantee is vacuous on a push. -/
@[circuit_norm] lemma dvrmComponent_guarantees :
    dvrmComponent.circuit.channelsWithGuarantees =
      [Bitwise.AreBytes.toRaw,
       DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC,
       DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC,
       DVRM.AluC,
       DVRM.ZeroC, DVRM.ZeroC, DVRM.ZeroC, DVRM.ZeroC, DVRM.ZeroC,
       DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC,
       DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC,
       DVRM.AluC, DVRM.AluC,
       DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC, DVRM.IsHalfC,
       DVRM.Msb16C, DVRM.Msb16C, DVRM.Msb16C,
       DVRM.ZeroC] := rfl

/-- The one normalization that opens a `DVRM` row: the component down to its interactions, the
chip down to its templates, and every bus down to its table. Written once because the three
obligations below each need all of it, and leaving any of it out leaves a goal that looks like a
missing lemma and is really a missing `simp` argument. -/
macro "dvrm_norm" loc:(Lean.Parser.Tactic.location)? : tactic =>
  `(tactic| (set_option linter.unusedSimpArgs false in
             simp (config := { decide := true }) [circuit_norm, Component.rowOperations,
               dvrmComponent, DVRM.circuit, DVRM.elaborated, DVRM.main, areBytes, isBit, sign,
               neg, DVRM.word, DVRM.fill, DVRM.IsHalfC, DVRM.ZeroC, DVRM.AluC, Alu.channel,
               Alu.channelC, Bitwise.AreBytes, Bitwise.IsHalf, Bitwise.Msb16, Bitwise.Zero,
               Bitwise.areBytesTable, Bitwise.isHalfTable, Bitwise.msb16Table, Bitwise.zeroTable,
               StaticLookupChannel.toChannel] $(loc)?))

/-- DVRM's sixteen `IS_HALF` pulls all carry `mu`, so `ARE_BYTES` bounds them, exactly as it does
for `MUL`. -/
theorem dvrm_requirementsAfter_isHalf {earlier : List (RawChannel (F p))}
    (h_ab : Bitwise.AreBytes.toRaw ∈ earlier) :
    dvrmComponent.RequirementsAfter DVRM.IsHalfC earlier := by
  rintro table h_comp - constraints guarantees row hrow
  have gab := guarantees _ h_ab row hrow
  cases table
  simp only at h_comp
  subst h_comp
  simp only [Component.channelGuarantees_iff, Component.channelRequirements_iff] at gab ⊢
  dvrm_norm at gab ⊢
  have hm := mul_mu_count gab.1.1 gab.1.2
  simp (config := { decide := true }) [hm]
  and_intros <;> channel_ne

/-- DVRM's six `ZERO` pulls do not all carry `mu`: the two `NEG` templates pull twice each under a
sign flag. The flag is a bit because `MSB16` returned it, which is why this obligation needs
`MSB16` finished as well as `ARE_BYTES`, and why `MSB16` sits below `ZERO` in `finished`. -/
theorem dvrm_requirementsAfter_zero {earlier : List (RawChannel (F p))}
    (h_ab : Bitwise.AreBytes.toRaw ∈ earlier) (h_ms : DVRM.Msb16C ∈ earlier) :
    dvrmComponent.RequirementsAfter DVRM.ZeroC earlier := by
  rintro table h_comp - constraints guarantees row hrow
  have gab := guarantees _ h_ab row hrow
  have gms := guarantees _ h_ms row hrow
  have hc := constraints row hrow
  cases table
  simp only at h_comp
  subst h_comp
  rw [Component.constraintsHold_iff] at hc
  simp only [Component.channelGuarantees_iff] at gms
  simp only [Component.rowOperations, dvrmComponent, DVRM.circuit] at hc gms
  have hsigns := DVRM.isBool_signs hc gms
  clear hc gms
  simp only [Component.channelGuarantees_iff, Component.channelRequirements_iff] at gab ⊢
  dvrm_norm at gab hsigns ⊢
  have hm := mul_mu_count gab.1.1 gab.1.2
  simp (config := { decide := true }) [hm]
  and_intros
  all_goals try channel_ne
  -- what is left is the two flags `NEG` pulls under, `sign_r` and `sign_d`
  all_goals first
    | exact eq_zero_or_countUpTo hsigns.2.1
    | exact eq_zero_or_countUpTo hsigns.2.2

/-! ### The `ALU` bus, by message class

`DVRM` is the chip the split exists for. It pulls `LT` and `MUL` answers off `ALU` and pushes
`DIVREM` answers back onto it, so a finishing order over whole channels cannot hold it: the bus
would have to be finished before itself. With the two classes, `DIVREM` finishes last and the
queries it borrows are already done by then. -/

/-- Reading a stage's obligation one row and one interaction at a time, which is the form the
chip's own `circuit_norm` normalizes to. `Table.RequirementsOn` quantifies over the interactions
of the whole table; `Table.forall_interactionsWith_iff` turns that into rows times the component's
own operations, and `AbstractInteraction.eval_requirements` carries the requirement back. -/
theorem requirementsAfterOn_of_rows {component : Component (F p)} {stage : Stage (F p)}
    {earlier : List (Stage (F p))}
    (h : ∀ table : Table (F p), table.component = component → table.Assumptions →
      table.Constraints →
      (∀ s ∈ earlier, table.GuaranteesOn s) →
      ∀ row ∈ table.table, ∀ i ∈ component.rowOperations.interactions,
        i.channel = stage.channel →
        stage.cls ((i.eval (table.environment row)).msg) = true →
        i.Requirements (table.environment row)) :
    component.RequirementsAfterOn stage earlier := by
  rintro table rfl assumptions constraints guarantees
  show ∀ i ∈ table.interactionsWith stage.channel, _
  rw [Table.forall_interactionsWith_iff]
  intro row hrow i hi hchan hcls
  rw [Component.interactions_eq] at hi
  exact AbstractInteraction.eval_requirements.mpr
    (h table rfl assumptions constraints guarantees row hrow i hi hchan hcls)

/-- The class of an `ALU` message is read off its fifth element, and `Vector` append does not
reduce on its own: `simp` cannot even rewrite under the index, because flattening the append
changes the type. The five `rw`s walk the `ProvableStruct` one field at a time, which is the
only step here that is not bookkeeping. -/
lemma isDivrem_emitted (env : Environment (F p)) (N : ℕ) (ch : Channel (F p) Alu.Msg)
    (mult a b c d fl e g : Expression (F p)) :
    IsDivrem (AbstractInteraction.eval env
      (Channel.emittedUpTo N ch mult ⟨a, b, c, d, fl, e, g⟩)).msg
      = divremFlags.contains (Expression.eval env fl) := by
  simp [IsDivrem, AbstractInteraction.eval, Channel.emittedUpTo, circuit_norm,
    explicit_provable_type]
  rw [Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_left (by omega)]
  simp

/-- DVRM owes nothing on the queries it borrows beyond the bound every counted pull owes. Its
three `ALU` pulls carry `mu`, so `ARE_BYTES` covers them; its two pushes are not in this class at
all, because their opcode is `8 + 32 * signed (+ 128)` and the chip constrains `signed` to a bit,
which puts all four possible opcodes in `DIVREM`'s class. That is the whole content: the pushes
are deferred to the stage where everything they need is finished. -/
theorem dvrm_stageQuery {earlier : List (Stage (F p))}
    (h_ab : ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier) :
    dvrmComponent.RequirementsAfterOn stageQuery earlier := by
  apply requirementsAfterOn_of_rows
  intro table h_comp _ constraints guarantees row hrow i hi hchan hcls
  have gab : table.ChannelGuarantees Bitwise.AreBytes.toRaw :=
    (Table.channelGuarantees_iff_forall _ _).mpr fun j hj => guarantees _ h_ab j hj rfl
  have g := gab row hrow
  have hc := constraints row hrow
  rw [h_comp] at g hc
  rw [Component.constraintsHold_iff] at hc
  simp only [Component.channelGuarantees_iff] at g
  simp only [Component.rowOperations, dvrmComponent, DVRM.circuit] at hc g
  have hflags := DVRM.flags_of_constraints hc
  clear hc
  simp only [stageQuery] at hchan hcls
  dvrm_norm at g
  have hm := mul_mu_count g.1.1 g.1.2
  dvrm_norm at hi
  repeat' rcases hi with rfl | hi
  -- the twenty-nine interactions on other buses
  all_goals try (simp only [circuit_norm] at hchan
                 refine absurd hchan ?_
                 channel_ne
                 done)
  -- the three pulls, bounded by `ARE_BYTES`
  all_goals try (simp only [circuit_norm]; exact hm)
  -- the two pushes, which this stage does not see
  all_goals (exfalso
             rw [isDivrem_emitted] at hcls
             rcases hflags.1 with h | h <;>
               simp (config := { decide := true }) [circuit_norm, divremFlags, h] at hcls)

/-- The same projection for a pull, which is the direction that carries the weight: `DVRM`'s
three borrowed answers have to be shown *not* to be `DIVREM` messages, since that is what puts
them in the stage finished before this one. -/
lemma isDivrem_pulled (env : Environment (F p)) (N : ℕ) (ch : Channel (F p) Alu.Msg)
    (mult a b c d fl e g : Expression (F p)) :
    IsDivrem (AbstractInteraction.eval env
      (Channel.pulledUpTo N ch mult ⟨a, b, c, d, fl, e, g⟩)).msg
      = divremFlags.contains (Expression.eval env fl) := by
  simp [IsDivrem, AbstractInteraction.eval, Channel.pulledUpTo, circuit_norm,
    explicit_provable_type]
  rw [Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_right (by omega) (by omega),
      Vector.getElem_append_left (by omega)]
  simp

/-- A push of nonzero multiplicity makes the row's `mu` nonzero, and so makes every counted pull
of that row active. Both multiplicities are bytes, so their sum vanishes only when both do; this
is the step that turns "this push is not suppressed" into "this row's lookups answered". -/
lemma mu_count_of_ne_zero {a b : F p} (ha : a.val < 256) (hb : b.val < 256)
    (h : a ≠ 0 ∨ b ≠ 0) : Channel.CountUpTo muMax (a + b) := by
  have hsum : (a + b).val = a.val + b.val :=
    val_eq_of_eq_natCast (by push_cast; simp only [ZMod.natCast_zmod_val]) (by omega)
  have hne : a + b ≠ 0 := by
    intro hz
    rw [hz, ZMod.val_zero] at hsum
    rcases h with h | h
    · exact h ((ZMod.val_eq_zero a).mp (by omega))
    · exact h ((ZMod.val_eq_zero b).mp (by omega))
  exact Channel.countUpTo_of_val hne (by simp only [hsum, muMax]; omega)

/-! The opcodes `DVRM` borrows are never `DIVREM`'s. `LT`'s is the literal `4`; `MUL`'s is
`7 + 32 * signed + 64 * sign_q` and the same plus `128` for the high half, which with both flags
bits gives `{7, 39, 71, 103}` and `{135, 167, 199, 231}` -- close to `{8, 40, 136, 168}` and never
equal to it. The flags stay abstract here on purpose: inside these three lemmas `s` and `q` are
variables, so a case split rewrites them directly, where at the use site the same terms come
wrapped in a `ProvableStruct` projection that `simp` normalizes the other way. -/
lemma not_isDivrem_lt {env : Environment (F p)} :
    (!divremFlags.contains (Expression.eval env (4 : Expression (F p)))) = true := by
  simp (config := { decide := true }) [circuit_norm, divremFlags]

lemma not_isDivrem_mul {env : Environment (F p)} {s q : Expression (F p)}
    (hs : IsBool (Expression.eval env s)) (hq : IsBool (Expression.eval env q)) :
    (!divremFlags.contains (Expression.eval env (7 + 32 * s + 64 * q))) = true := by
  rcases hs with h1 | h1 <;> rcases hq with h2 | h2 <;>
    simp (config := { decide := true }) [circuit_norm, divremFlags, h1, h2]

lemma not_isDivrem_mulHi {env : Environment (F p)} {s q : Expression (F p)}
    (hs : IsBool (Expression.eval env s)) (hq : IsBool (Expression.eval env q)) :
    (!divremFlags.contains (Expression.eval env (7 + 32 * s + 64 * q + 128))) = true := by
  rcases hs with h1 | h1 <;> rcases hq with h2 | h2 <;>
    simp (config := { decide := true }) [circuit_norm, divremFlags, h1, h2]

/-- A whole-channel stage, read at one row: what the ensemble hands over as a fact about every
interaction of the table becomes a fact about the operations of this row. -/
theorem dvrm_rowChannelGuarantees {earlier : List (Stage (F p))}
    {table : Table (F p)} (h_comp : table.component = dvrmComponent)
    (guarantees : ∀ s ∈ earlier, table.GuaranteesOn s)
    {row : Array (F p)} (hrow : row ∈ table.table)
    {c : RawChannel (F p)} (hc : ({ channel := c } : Stage (F p)) ∈ earlier) :
    dvrmComponent.rowOperations.ChannelGuarantees c (table.environment row) := by
  have h1 : table.ChannelGuarantees c :=
    (Table.channelGuarantees_iff_forall _ _).mpr fun j hj => guarantees _ hc j hj rfl
  have h2 := h1 row hrow
  rw [h_comp] at h2
  simp only [Component.channelGuarantees_iff] at h2
  exact h2

/-- Every guarantee of one row, once its two sign flags are known to be bits.

This is the step the `DIVREM` stage needs and the only place the message classes do real work.
Twenty-nine of the thirty-three pulls are on whole-channel stages and come straight from the
ensemble. The two pushes receive nothing, a pull guarantee being vacuous on a push. The three
left are the `ALU` answers the chip borrows, and they are covered by the query stage exactly
because their opcodes are not `DIVREM`'s -- which is what `hs` and `hq` are for.

The three are dispatched by position rather than by `first`, because a failed `exact` on a goal
of the form `(!divremFlags.contains e) = true` sends the unifier into `ZMod`'s decidable equality
and costs more than the proof it was looking for. -/
theorem dvrm_fullGuarantees {earlier : List (Stage (F p))}
    (h_ab : ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier)
    (h_ih : ({ channel := DVRM.IsHalfC } : Stage (F p)) ∈ earlier)
    (h_ze : ({ channel := DVRM.ZeroC } : Stage (F p)) ∈ earlier)
    (h_ms : ({ channel := DVRM.Msb16C } : Stage (F p)) ∈ earlier)
    (h_q : stageQuery ∈ earlier)
    {table : Table (F p)} (h_comp : table.component = dvrmComponent)
    (guarantees : ∀ s ∈ earlier, table.GuaranteesOn s)
    {row : Array (F p)} (hrow : row ∈ table.table)
    (hs : IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).signed))
    (hq : IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).signQ)) :
    dvrmComponent.rowOperations.FullGuarantees (table.environment row) := by
  have whole : ∀ {c : RawChannel (F p)}, ({ channel := c } : Stage (F p)) ∈ earlier →
      dvrmComponent.rowOperations.ChannelGuarantees c (table.environment row) :=
    fun hc => dvrm_rowChannelGuarantees h_comp guarantees hrow hc
  have query : ∀ j ∈ dvrmComponent.rowOperations.interactions, j.channel = Alu.channelC →
      (!IsDivrem ((j.eval (table.environment row)).msg)) = true →
      j.Guarantees (table.environment row) := by
    intro j hj hc1 hc2
    have h1 := guarantees _ h_q
    rw [Table.GuaranteesOn, Table.forall_interactionsWith_iff] at h1
    rw [← h_comp, ← Component.interactions_eq] at hj
    exact AbstractInteraction.eval_guarantees.mp (h1 row hrow j hj hc1 hc2)
  intro j hj
  have hmemj := hj
  simp only [circuit_norm, Component.rowOperations, dvrmComponent, DVRM.circuit,
    DVRM.elaborated, DVRM.main, areBytes, isBit, sign, neg, DVRM.word, DVRM.fill] at hj
  repeat' rcases hj with rfl | hj
  all_goals try (first
    | exact whole h_ih _ hmemj rfl
    | exact whole h_ze _ hmemj rfl
    | exact whole h_ms _ hmemj rfl
    | exact whole h_ab _ hmemj rfl
    | (simp only [circuit_norm]; done))
  -- the three borrowed answers, in the order `main` pulls them: `LT`, then `MUL` low and high
  · exact query _ hmemj rfl (by rw [isDivrem_pulled]; exact not_isDivrem_lt)
  · exact query _ hmemj rfl (by rw [isDivrem_pulled]; exact not_isDivrem_mul hs hq)
  · exact query _ hmemj rfl (by rw [isDivrem_pulled]; exact not_isDivrem_mulHi hs hq)

/-- Soundness at one row: its guarantees give its requirements. The two sign flags are the only
thing the caller has to supply, and it supplies them from the chip's own constraints plus
`ZERO`'s answer. -/
theorem dvrm_rowRequirements {earlier : List (Stage (F p))}
    (h_ab : ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier)
    (h_ih : ({ channel := DVRM.IsHalfC } : Stage (F p)) ∈ earlier)
    (h_ze : ({ channel := DVRM.ZeroC } : Stage (F p)) ∈ earlier)
    (h_ms : ({ channel := DVRM.Msb16C } : Stage (F p)) ∈ earlier)
    (h_q : stageQuery ∈ earlier)
    {table : Table (F p)} (h_comp : table.component = dvrmComponent)
    (assumptions : table.Assumptions) (constraints : table.Constraints)
    (guarantees : ∀ s ∈ earlier, table.GuaranteesOn s)
    {row : Array (F p)} (hrow : row ∈ table.table)
    (hs : IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).signed))
    (hq : IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).signQ)) :
    dvrmComponent.rowOperations.FullRequirements (table.environment row) := by
  have hc := constraints row hrow
  rw [h_comp] at hc
  have hca := Table.circuitAssumptions_of_assumptions _ assumptions row hrow
  rw [h_comp] at hca
  exact (Component.requirements_iff _).mp
    (Component.weakSoundness hca hc
      ((Component.guarantees_iff _).mpr
        (dvrm_fullGuarantees h_ab h_ih h_ze h_ms h_q h_comp guarantees hrow hs hq))).2

/-- `DVRM`'s obligation on the `DIVREM` class, the last stage to finish and the one its two
pushes live in. A push owes the `ALU` guarantee whenever its multiplicity is nonzero, which is
soundness, which needs every guarantee the row has -- including the three borrowed `ALU` answers,
which only the query stage can supply, and only because the row's opcodes are not `DIVREM`'s.

That chain is why the stage needs `mu_q` or `mu_r` nonzero to begin with. It gives `mu` nonzero,
which makes the counted pulls active, which makes `ZERO` answer, which makes `overflow` a bit,
which with `IS_BIT` on `signed` pins `sign_q` -- and only then are the borrowed opcodes known.
When both multiplicities are zero the push is suppressed and there is nothing to prove. -/
theorem dvrm_stageDivrem {earlier : List (Stage (F p))}
    (h_ab : ({ channel := Bitwise.AreBytes.toRaw } : Stage (F p)) ∈ earlier)
    (h_ih : ({ channel := DVRM.IsHalfC } : Stage (F p)) ∈ earlier)
    (h_ze : ({ channel := DVRM.ZeroC } : Stage (F p)) ∈ earlier)
    (h_ms : ({ channel := DVRM.Msb16C } : Stage (F p)) ∈ earlier)
    (h_q : stageQuery ∈ earlier) :
    dvrmComponent.RequirementsAfterOn stageDivrem earlier := by
  apply requirementsAfterOn_of_rows
  intro table h_comp assumptions constraints guarantees row hrow i hi hchan hcls
  have gn := dvrm_rowChannelGuarantees h_comp guarantees hrow h_ab
  have hbytes := DVRM.bytes_mu gn
  dvrm_norm at gn
  have hc := constraints row hrow
  rw [h_comp, Component.constraintsHold_iff] at hc
  simp only [Component.rowOperations, dvrmComponent, DVRM.circuit] at hc
  have hflags := DVRM.flags_of_constraints hc
  clear hc
  have ovf : Channel.CountUpTo muMax
      (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).muQ +
        Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).muR) →
      IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).overflow) :=
    fun h => DVRM.isBool_overflow h (dvrm_rowChannelGuarantees h_comp guarantees hrow h_ze)
  have reqOf : IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).signed) →
      IsBool (Expression.eval (table.environment row) (varFromOffset DVRM.Row 0).signQ) →
      dvrmComponent.rowOperations.FullRequirements (table.environment row) :=
    fun hs hq => dvrm_rowRequirements h_ab h_ih h_ze h_ms h_q h_comp assumptions constraints
      guarantees hrow hs hq
  have hmem := hi
  simp only [stageDivrem] at hchan hcls
  dvrm_norm at hi
  repeat' rcases hi with rfl | hi
  all_goals try (simp only [circuit_norm] at hchan
                 refine absurd hchan ?_
                 channel_ne
                 done)
  all_goals try (simp only [circuit_norm]
                 exact mul_mu_count gn.1.1 gn.1.2)
  all_goals (
    intro hmult
    exact (reqOf hflags.1 (DVRM.isBool_signQ hflags.1
      (ovf (mu_count_of_ne_zero hbytes.1 hbytes.2
        (by first | exact Or.inl hmult | exact Or.inr hmult))) hflags.2) _ hmem) hmult)

/-- DVRM's nine obligations. Five stages are buses it does not require. `IS_HALF` and `ZERO` are
counted pulls, bounded by `ARE_BYTES` and, for the sign flags `NEG` pulls under, by `MSB16`. The
two that carry the weight are the `ALU` stages, and they are why the bus is split: the queries
DVRM borrows finish first, and the `DIVREM` answers it pushes finish last. -/
theorem dvrm_staged : dvrmComponent.StagedRequirementsOn finished := by
  have counted : ∀ {c : RawChannel (F p)} {ecs : List (RawChannel (F p))},
      dvrmComponent.RequirementsAfter c ecs →
      ∀ {earlier : List (Stage (F p))},
      (∀ c' ∈ ecs, ({ channel := c' } : Stage (F p)) ∈ earlier) →
      dvrmComponent.RequirementsAfterOn { channel := c } earlier := by
    intro c ecs h earlier hmem
    exact Component.requirementsAfterOn_of_requirementsAfter h
      fun c' hc' => ⟨_, hmem c' hc', rfl, fun _ => rfl⟩
  refine ⟨dvrm_stageDivrem (by simp) (by simp) (by simp) (by simp) (by simp),
    dvrm_stageQuery (by simp),
    counted (dvrm_requirementsAfter_zero
      (earlier := [Bitwise.AreBytes.toRaw, DVRM.Msb16C]) (by simp) (by simp)) (by simp),
    ?_, ?_,
    counted (dvrm_requirementsAfter_isHalf
      (earlier := [Bitwise.AreBytes.toRaw]) (by simp)) (by simp),
    ?_, ?_, ?_, trivial⟩
  all_goals
    refine Component.requirementsAfterOn_of_not_mem _ ?_
    simp only [dvrmComponent_requirements, DVRM.AluC, DVRM.IsHalfC, DVRM.ZeroC,
      List.mem_cons, List.not_mem_nil, or_false, not_or]
    and_intros <;> channel_ne

/-- Every table of the ensemble meets its spec.

This is the first use of `Ensemble.tableSoundness_of_stagedOn` anywhere: ordering whole channels
cannot hold this ensemble, because `DVRM` both pulls from `ALU` and pushes to it, so the bus
would have to be finished before itself. Splitting it by message class breaks the cycle, and
nothing else in the model needed that until `DVRM` arrived. -/
theorem tableSoundness : ensemble.TableSoundness (Balance.logUpUpTo muMax) := by
  apply Ensemble.tableSoundness_of_stagedOn finished rfl finished_channels covered consistent
  intro component h
  simp only [ensemble, List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact stagedRequirementsOn_of_no_guarantees
      (pushStaticComponent_channelsWithGuarantees _) _
  iterate 4
    exact stagedRequirementsOn_of_no_guarantees
      (countedPushStaticComponent_channelsWithGuarantees _) _
  · exact lt_staged
  · exact mul_staged
  · exact dvrm_staged

/-- Table soundness, concretely: if the constraints of every table hold, every channel has few
enough interactions for the balance check, and every channel balances, then every `DVRM` row
means what the chip says it means -- the quotient and remainder of its two operands, signed or
unsigned, with the division-by-zero and overflow cases the spec names.

The eight components assume nothing, so the ensemble's `Assumptions` are free. -/
theorem dvrm_rows_correct
    (witness : EnsembleWitness ensemble)
    (hc : witness.Constraints) (hfit : witness.InteractionsFit (Balance.logUpUpTo muMax))
    (hb : witness.Balanced (Balance.logUpUpTo muMax)) :
    ∀ table ∈ witness.tables, table.component = dvrmComponent →
      ∀ row ∈ table.table, DVRM.RowSpec (valueFromOffset DVRM.Row 0 (table.environment row)) := by
  have hassume : witness.Assumptions := by
    intro table htable row _
    have hmem := witness.component_mem_tables_of_mem htable
    simp only [ensemble, circuit_norm, List.mem_cons, List.not_mem_nil, or_false] at hmem
    obtain ⟨comp, tbl, data, uw, ll, frm, dc⟩ := table
    simp only at hmem
    rcases hmem with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> trivial
  have hspec := (tableSoundness witness hassume hc hfit hb).2
  intro table htable hcomp row hrow
  have := hspec table htable row hrow
  obtain ⟨comp, tbl, data, uw, ll, frm, dc⟩ := table
  simp only at hcomp
  subst hcomp
  exact this

end LambdaVM.Ensembles.DvrmStaged
