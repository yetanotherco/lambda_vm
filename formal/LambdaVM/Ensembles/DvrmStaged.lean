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

end LambdaVM.Ensembles.DvrmStaged
