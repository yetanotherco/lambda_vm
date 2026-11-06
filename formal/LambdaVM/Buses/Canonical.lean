module

public import LambdaVM.Tables.Lt
public import LambdaVM.Tables.Eq
public import LambdaVM.Tables.Shift
public import LambdaVM.Tables.Bytewise
public import LambdaVM.Tables.Mul
public import LambdaVM.Tables.Dvrm

@[expose] public section

/-!
# One `RawChannel` per bus name

`Channel.toRaw` and `Channel.counted N` build different `RawChannel`s that carry the same
`name`, and `counted N` and `counted M` are different again. `Ensemble.unique_channel_names`
asks for `(channels.map (·.name)).Nodup`, so a bus the model spells two ways is two channels
under one name and no ensemble can hold the chips that disagree about it.

Nothing said so until somebody wrote the ensemble that crossed them. This file says it here.
`modelChannels` is the one spelling the model agrees on for each bus; `Canonical` says a
channel list only holds those. The six chips are checked below, so a chip that moves a pull
between `pullIf` and `pullUpTo` without the rest of the model moving with it fails `lake build`
here, at the chip, instead of at the next ensemble somebody writes.

Adding a bus means adding a line to `modelChannels`: a channel whose name is not listed is not
canonical, so a new bus arrives with its spelling decided rather than defaulted.
-/

namespace LambdaVM.Buses
open Bitwise

/-- The one `RawChannel` the model uses under each bus name. One entry per bus. -/
def modelChannels : List (RawChannel (F p)) :=
  [ Alu.channelC            -- "ALU"
  , AreBytes.toRaw          -- "ARE_BYTES"
  , ByteAlu.counted muMax   -- "BYTE_ALU"
  , Hwsl.toRaw              -- "HWSL"
  , IsB20.counted muMax     -- "IS_B20"
  , IsHalf.counted muMax    -- "IS_HALF"
  , Msb16.counted muMax     -- "MSB16"
  , Zero.counted muMax ]    -- "ZERO"

/-- What an ensemble over all six chips would have to discharge as `unique_channel_names`.
Decided on the names, which are strings; no channel is ever compared to another. -/
theorem modelChannels_unique_names : (modelChannels.map (·.name)).Nodup := by
  decide   -- fallback, if a name does not reduce in the kernel:
           -- `simp only [modelChannels, List.map_cons, List.map_nil, circuit_norm]; decide`

/-- The canonical channel for a bus name, if the model has one. -/
def busOf (name : String) : Option (RawChannel (F p)) :=
  modelChannels.find? (·.name == name)

/-- Every channel of `l` is the canonical one for its name. -/
def Canonical (l : List (RawChannel (F p))) : Prop :=
  ∀ c ∈ l, busOf c.name = some c

theorem canonical_nil : Canonical [] := by simp [Canonical]

theorem canonical_cons {c : RawChannel (F p)} {l : List (RawChannel (F p))}
    (hc : busOf c.name = some c) (hl : Canonical l) : Canonical (c :: l) := by
  intro x hx
  rcases List.mem_cons.mp hx with rfl | hx
  · exact hc
  · exact hl x hx

/-- Peel the channel list one entry at a time, closing each by evaluating `busOf` on a string
literal. No two channels are ever compared, so no `Guarantees` closure is ever unfolded: that
is what keeps this affordable on a chip with 35 interactions. -/
macro "canonical_channels" : tactic =>
  `(tactic| repeat first
      | exact canonical_nil
      | refine canonical_cons rfl ?_)

/-! ## The six chips -/

theorem lt_canonical       : Canonical LT.circuit.channels       := by canonical_channels
theorem eq_canonical       : Canonical EQ.circuit.channels       := by canonical_channels
theorem shift_canonical    : Canonical SHIFT.circuit.channels    := by canonical_channels
theorem bytewise_canonical : Canonical BYTEWISE.circuit.channels := by canonical_channels
theorem mul_canonical      : Canonical MUL.circuit.channels      := by canonical_channels
theorem dvrm_canonical     : Canonical DVRM.circuit.channels     := by canonical_channels

/-! ## What the six checks buy -/

/-- On a canonical list the name determines the channel. This is the fact every ensemble
needs and never gets to use: `unique_channel_names` can only be met by a list that holds one
channel per name, and `tables_channels` can only be met if each chip's channels are in it. -/
theorem eq_of_name_eq {l : List (RawChannel (F p))} (h : Canonical l)
    {c₁ c₂ : RawChannel (F p)} (h₁ : c₁ ∈ l) (h₂ : c₂ ∈ l) (hn : c₁.name = c₂.name) : c₁ = c₂ :=
  Option.some.inj (by rw [← h c₁ h₁, ← h c₂ h₂, hn])

/-- A canonical list is covered by `modelChannels`, which is `tables_channels` for the
ensemble `modelChannels_unique_names` makes available. -/
theorem subset_modelChannels {l : List (RawChannel (F p))} (h : Canonical l) :
    l ⊆ modelChannels :=
  fun c hc => List.mem_of_find?_eq_some (h c hc)

/-- The headline: the six chips agree on every bus, so an ensemble holding all six has a
channel list with unique names that covers each of them. -/
theorem model_channels_agree :
    LT.circuit.channels       ⊆ modelChannels ∧
    EQ.circuit.channels       ⊆ modelChannels ∧
    SHIFT.circuit.channels    ⊆ modelChannels ∧
    BYTEWISE.circuit.channels ⊆ modelChannels ∧
    MUL.circuit.channels      ⊆ modelChannels ∧
    DVRM.circuit.channels     ⊆ modelChannels :=
  ⟨subset_modelChannels lt_canonical, subset_modelChannels eq_canonical,
   subset_modelChannels shift_canonical, subset_modelChannels bytewise_canonical,
   subset_modelChannels mul_canonical, subset_modelChannels dvrm_canonical⟩

end LambdaVM.Buses
