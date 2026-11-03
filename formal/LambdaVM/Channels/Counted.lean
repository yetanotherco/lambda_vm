module

public import Clean.Air.Balance
public import LambdaVM.Tables.Bitwise
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# Counted pulls

A deduplicated row pulls its range checks with multiplicity `μ ∈ [0, 255]` (`MU_MAX`, PR #1016).
Clean's typed channels give the guarantee only to a pull of multiplicity `1`, and require every
pull to have multiplicity `0` or `1`. `counted N channel` is the raw channel that gives the
guarantee to a pull of multiplicity `1, …, N` and charges every other nonzero interaction with
the requirement, as a push.

The price is in the balance premise. `exists_push_of_pull` needs `N · length < char F`. Clean's
`BalancedInteractions` only matches pulls of multiplicity `1`, and counts a pull of any other
nonzero multiplicity as a push; `not_consistent` shows the gap is real: for `N ≥ 2` the counted
channel is not `Consistent`, so it cannot be finished in a `SoundEnsemble` until that premise
carries the bound. With `N = 1` everything coincides (`counted 1 channel` is `Normal`).
-/

namespace LambdaVM.Counted

variable {F : Type} [FiniteField F] [DecidableEq F]
variable {Message : TypeMap} [ProvableType Message]

/-- `mult = k` for a count `1 ≤ k ≤ N`. -/
def IsPull (N : ℕ) (mult : F) : Prop := ∃ k : ℕ, 1 ≤ k ∧ k ≤ N ∧ mult = (k : F)

/-- `channel` with pulls allowed to carry a count up to `N`. -/
def counted (N : ℕ) (channel : Channel F Message) : RawChannel F where
  name := channel.name
  arity := size Message
  Guarantees mult message data :=
    IsPull N mult → channel.Guarantees (fromElements message) data
  Requirements direction mult message data :=
    match direction with
    | .push => mult ≠ 0 → channel.Guarantees (fromElements message) data
    | .pull => mult ≠ 0 → ¬ IsPull N mult → channel.Guarantees (fromElements message) data

/-- Pull `msg` `count` times. -/
@[circuit_norm]
def pullIf (N : ℕ) (channel : Channel F Message) (count : Expression F)
    (msg : Message (Expression F)) : Circuit F Unit := fun _ =>
  ((), [.interact ⟨counted N channel, .pull, count, toElements msg⟩])

/-- The provider side: emit `msg` with multiplicity `mult`. -/
@[circuit_norm]
def emit (N : ℕ) (channel : Channel F Message) (mult : Expression F)
    (msg : Message (Expression F)) : Circuit F Unit := fun _ =>
  ((), [.interact ⟨counted N channel, .push, mult, toElements msg⟩])

instance {N : ℕ} {channel : Channel F Message} {count : Expression F} :
    ExplicitCircuits (F := F) (pullIf N channel count) where
  output _ _ := ()
  localLength _ _ := 0
  operations msg _ := [.interact ⟨counted N channel, .pull, count, toElements msg⟩]
  channelsWithGuarantees _ _ := [counted N channel]

instance {N : ℕ} {channel : Channel F Message} {mult : Expression F} :
    ExplicitCircuits (F := F) (emit N channel mult) where
  output _ _ := ()
  localLength _ _ := 0
  operations msg _ := [.interact ⟨counted N channel, .push, mult, toElements msg⟩]
  channelsWithGuarantees _ _ := []

attribute [explicit_circuit_no_unfold] pullIf emit

/-- An interaction that acts as a push of its message: an active push, or a pull whose
multiplicity is not a count. -/
def PushLike (N : ℕ) (i : Interaction F) : Prop :=
  i.mult ≠ 0 ∧ (i.direction = .pull → ¬ IsPull N i.mult)

/-- `RawChannel.Normal` with the pull multiplicity `1` widened to `1, …, N`. -/
class Normal (N : ℕ) (channel : RawChannel F) : Prop where
  grts_of_reqs : ∀ (msg : Vector F channel.arity) (direction : Channel.Direction) (mult : F) data,
    mult ≠ 0 → (direction = .pull → ¬ IsPull N mult) →
    channel.Requirements direction mult msg data →
    ∀ mult', IsPull N mult' → channel.Guarantees mult' msg data
  grts_of_not_pull : ∀ (msg : Vector F channel.arity) (mult : F) data, ¬ IsPull N mult →
    channel.Guarantees mult msg data

instance (N : ℕ) (channel : Channel F Message) : Normal N (counted N channel) where
  grts_of_reqs := by
    intro msg direction mult data h0 hnp reqs mult' hp
    cases direction <;> simp only [counted] at reqs ⊢
    · exact fun _ => reqs h0
    · exact fun _ => reqs h0 (hnp rfl)
  grts_of_not_pull := by
    intro msg mult data hnp
    simp only [counted]
    exact fun hp => absurd hp hnp

/-- On a message whose nonzero multiplicities are all counted pulls, the balance is minus a
natural number bounded by `N` per interaction, and positive as soon as one pull is active. -/
lemma balanceOf_counted (N : ℕ) (msg : Array F) :
    ∀ (l : List (Interaction F)),
    (∀ i ∈ l, i.msg = msg → i.mult ≠ 0 → i.direction = .pull ∧ IsPull N i.mult) →
    ∃ s : ℕ, s ≤ N * l.length ∧ balanceOf l msg = -(s : F) ∧
      (∀ a ∈ l, a.msg = msg → a.mult ≠ 0 → 1 ≤ s)
  | [], _ => ⟨0, by simp, by simp [balanceOf], by simp⟩
  | i :: l, h => by
    obtain ⟨s, hs, hb, ha⟩ :=
      balanceOf_counted N msg l (fun j hj => h j (List.mem_cons_of_mem _ hj))
    rw [balanceOf_cons]
    simp only [List.length_cons, Nat.mul_succ]
    by_cases hm : i.msg = msg
    · by_cases h0 : i.mult = 0
      · refine ⟨s, by omega, ?_, ?_⟩
        · have hsigned : i.signedMult = 0 := by
            cases hd : i.direction <;> simp [Interaction.signedMult, hd, h0]
          rw [ite_eq_left hm, hsigned, hb, zero_add]
        · intro a ha' ham ha0
          rcases List.mem_cons.mp ha' with rfl | ha'
          · exact absurd h0 ha0
          · exact ha a ha' ham ha0
      · obtain ⟨hd, k, hk1, hkN, hk⟩ := h i (by simp) hm h0
        refine ⟨k + s, by omega, ?_, ?_⟩
        · rw [ite_eq_left hm, hb]
          simp only [Interaction.signedMult, hd, hk]
          push_cast; ring
        · intro a _ _ _; omega
    · refine ⟨s, by omega, ?_, ?_⟩
      · rw [ite_eq_right hm, hb, zero_add]
      · intro a ha' ham ha0
        rcases List.mem_cons.mp ha' with rfl | ha'
        · exact absurd ham hm
        · exact ha a ha' ham ha0

/-- `exists_push_of_pull` for counted pulls. The balance premise must bound the total count,
`N * length`, by the characteristic; Clean's `BalancedInteractions` only matches pulls of
multiplicity `1`. -/
theorem exists_push_of_pull (N : ℕ) (interactions : List (Interaction F))
    (bound : N * interactions.length < ringChar F ∨ ringChar F = 0)
    (balance : ∀ msg, balanceOf interactions msg = 0) :
    ∀ a ∈ interactions, a.direction = .pull → IsPull N a.mult →
      ∃ b ∈ interactions, b.msg = a.msg ∧ PushLike N b := by
  intro a ha hdir hpull
  by_contra! hno
  have hno' : ∀ i ∈ interactions, i.msg = a.msg → i.mult ≠ 0 →
      i.direction = .pull ∧ IsPull N i.mult := by
    intro i hi hm h0
    by_contra hc
    exact hno i hi hm ⟨h0, fun hd hp => hc ⟨hd, hp⟩⟩
  obtain ⟨s, hs, hb, h1⟩ := balanceOf_counted N a.msg interactions hno'
  have hlen : 0 < interactions.length := List.length_pos_of_mem ha
  have hNlen : N ≤ N * interactions.length := Nat.le_mul_of_pos_right N hlen
  have ha0 : a.mult ≠ 0 := by
    obtain ⟨k, hk1, hkN, hk⟩ := hpull
    rw [hk]
    rcases bound with bound | hchar
    · intro hk0
      rw [Lean.Grind.IsCharP.natCast_eq_zero_iff_of_lt _ (by omega)] at hk0
      omega
    · rw [CharP.ringChar_zero_iff_CharZero] at hchar
      exact Nat.cast_ne_zero.mpr (by omega)
  have hs1 := h1 a ha rfl ha0
  have hzero := balance a.msg
  rw [hb, neg_eq_zero] at hzero
  rcases bound with bound | hchar
  · rw [Lean.Grind.IsCharP.natCast_eq_zero_iff_of_lt _ (by omega)] at hzero
    omega
  · rw [CharP.ringChar_zero_iff_CharZero] at hchar
    rw [Nat.cast_eq_zero] at hzero
    omega

/-- `RawChannel.Consistent` for a counted channel, under the stronger balance premise. -/
theorem consistent (N : ℕ) (channel : RawChannel F) [Normal N channel]
    (interactions : List (Interaction F)) (data : ProverData F)
    (bound : N * interactions.length < ringChar F ∨ ringChar F = 0)
    (balance : ∀ msg, balanceOf interactions msg = 0)
    (reqs : ∀ i ∈ interactions, i.channel = channel ∧ i.Requirements data) :
    ∀ i ∈ interactions, i.Guarantees data := by
  intro a a_mem
  simp only [Interaction.Guarantees, Interaction.Requirements, Interaction.msgVector] at reqs ⊢
  intro hdir
  have a_channel_eq := (reqs a a_mem).1
  have a_msg_size : a.msg.size = channel.arity := by rw [a.same_size, a_channel_eq]
  suffices channel.Guarantees a.mult ⟨a.msg, a_msg_size⟩ data by convert this
  by_cases a_pull : IsPull N a.mult
  case neg => exact Normal.grts_of_not_pull ⟨a.msg, a_msg_size⟩ a.mult data a_pull
  obtain ⟨b, b_mem, b_msg_eq, b_ne_zero, b_not_pull⟩ :=
    exists_push_of_pull N interactions bound balance a a_mem hdir a_pull
  apply Normal.grts_of_reqs ⟨a.msg, a_msg_size⟩ b.direction b.mult data b_ne_zero b_not_pull _
    a.mult a_pull
  have ⟨b_channel_eq, b_reqs⟩ := reqs _ b_mem
  symm at b_channel_eq
  simp only [b_msg_eq] at b_reqs
  convert b_reqs

/-! ## `N = 1` is Clean's normal channel -/

omit [DecidableEq F] in
lemma isPull_one_iff (mult : F) : IsPull 1 mult ↔ mult = 1 := by
  constructor
  · rintro ⟨k, hk1, hk, rfl⟩
    have : k = 1 := by omega
    subst this; simp
  · rintro rfl; exact ⟨1, le_rfl, le_rfl, by simp⟩

instance (channel : Channel F Message) : (counted 1 channel).Normal where
  guarantees_of_requirements := by
    rintro ⟨_, a_direction, a_mult, a_msg, a_size⟩ ⟨_, b_direction, b_mult, b_msg, b_size⟩
      data rfl rfl (rfl : a_msg = b_msg) a_pull ⟨b_ne_zero, b_not_one⟩ b_reqs
    cases b_direction <;>
      simp only [Interaction.Guarantees, Interaction.Requirements, Interaction.msgVector, counted,
        isPull_one_iff] at b_reqs ⊢
    · exact fun _ _ => b_reqs b_ne_zero
    · exact fun _ _ => b_reqs b_ne_zero (b_not_one rfl)
  guarantees_of_not_pull := by
    rintro ⟨_, direction, mult, msg, size⟩ data rfl not_pull
    simp only [Interaction.Pull, not_and] at not_pull
    simp only [Interaction.Guarantees, counted, isPull_one_iff]
    exact fun h_direction h_mult => absurd h_mult (not_pull h_direction)

example (channel : Channel F Message) : (counted 1 channel).Consistent := inferInstance

end LambdaVM.Counted

/-! ## `N ≥ 2` is not `Consistent` under Clean's balance premise -/

namespace LambdaVM.Counted
variable {Message : TypeMap} [ProvableType Message]

/-- The witness `BalancedInteractions` admits but the counted channel cannot survive: one pull of
`N` and one pull of `1` on a message that is not in the table, nothing pushed. Clean counts the
pull of `N` as a push of the message, so the pull of `1` is matched; the counted channel reads
both as pulls, and the guarantee of the second is false. -/
theorem not_consistent (N : ℕ) (hN : 2 ≤ N) (hN' : N ≤ 255)
    (channel : Channel (F p) Message) (m : Message (F p)) (data : ProverData (F p))
    (hm : ¬ channel.Guarantees m data) : ¬ (counted N channel).Consistent := by
  rintro ⟨hc⟩
  let mk : F p → Interaction (F p) := fun mult =>
    ⟨counted N channel, .pull, mult, (toElements m).toArray, by simp [counted]⟩
  let l := [mk (N : F p), mk 1]
  have hval : ((N : ℕ) : F p).val = N := ZMod.val_natCast_of_lt (lt_p_of_lt_two_pow_40 (by omega))
  have hN0 : (N : F p) ≠ 0 := by
    intro h; have := congrArg ZMod.val h; rw [hval, ZMod.val_zero] at this; omega
  have hN1 : (N : F p) ≠ 1 := by
    intro h; have := congrArg ZMod.val h; rw [hval, ZMod.val_one] at this; omega
  have hbal : BalancedInteractions l := by
    constructor
    · -- the pull of `1` is matched by the pull of `N`, which Clean reads as a push
      intro a ha _
      refine ⟨mk (N : F p), by simp [l], ?_, hN0, fun _ => hN1⟩
      simp only [l, List.mem_cons, List.not_mem_nil, or_false] at ha
      rcases ha with rfl | rfl <;> rfl
    · -- the multiplicity `N` is not `0` or `1`, so only messages with no interaction qualify
      intro msg is_bool
      by_cases hmsg : (toElements m).toArray = msg
      · exfalso
        rcases is_bool (mk (N : F p)) (by simp [l]) hmsg with h | h
        · exact hN0 h
        · exact hN1 h
      · simp [activeCount, l, mk, hmsg]
  have hreqs : ∀ i ∈ l, i.channel = counted N channel ∧ i.Requirements data := by
    intro i hi
    simp only [l, List.mem_cons, List.not_mem_nil, or_false] at hi
    rcases hi with rfl | rfl
    · refine ⟨rfl, ?_⟩
      simp only [Interaction.Requirements, mk, counted, Interaction.msgVector]
      intro _ hnp
      exact absurd ⟨N, by omega, le_rfl, rfl⟩ hnp
    · refine ⟨rfl, ?_⟩
      simp only [Interaction.Requirements, mk, counted, Interaction.msgVector]
      intro _ hnp
      exact absurd ⟨1, le_rfl, by omega, by simp⟩ hnp
  have hg := hc l data hbal hreqs (mk 1) (by simp [l])
  simp only [Interaction.Guarantees, mk, counted, Interaction.msgVector] at hg
  apply hm
  have h := hg trivial ⟨1, le_rfl, by omega, by simp⟩
  convert h
  simp [ProvableType.fromElements_toElements]

end LambdaVM.Counted

/-! ## A deduplicated row

`μ` is range-checked as a byte through `ARE_BYTES` and `x` is pulled from `IS_HALF` with
multiplicity `μ`: the shape of every dedup chip after PR #1016. The counted channel is listed
under `channelsWithRequirements` because the requirement on an out-of-range `μ` is only
discharged by the byte guarantee, which lawfulness cannot see; soundness can. -/

namespace LambdaVM.Counted.Demo
open LambdaVM Bitwise

structure Row (F : Type) where
  mu : F
  x : F
deriving ProvableStruct

def main (r : Var Row (F p)) : Circuit (F p) Unit := do
  AreBytes.pull (r.mu, 0)
  pullIf 255 IsHalf r.mu r.x

def circuit : GeneralFormalCircuit (F p) Row unit where
  main
  Spec r _ _ := r.mu ≠ 0 → r.x.val < 65536
  ProverAssumptions r _ _ := r.mu.val < 256 ∧ r.x.val < 65536
  channelsWithRequirements := [counted 255 IsHalf]
  requirementsChannelsLawful input_var i₀ := by
    obtain ⟨mu, x⟩ := input_var
    simp only [circuit_norm, main, pullIf, areBytesTable]
    grind
  soundness := by
    circuit_proof_start [main, pullIf, counted, areBytesTable, isHalfTable]
    simp only [AbstractInteraction.Guarantees, AbstractInteraction.Requirements, circuit_norm,
      h_input] at h_holds ⊢
    obtain ⟨⟨h_mu, _⟩, h_x⟩ := h_holds
    -- an active row pulls with the count `μ.val ∈ [1, 255]`
    have key : input_mu ≠ 0 → IsPull 255 input_mu := fun h0 =>
      ⟨input_mu.val, by have := (ZMod.val_ne_zero _).mpr h0; omega, by omega,
        by rw [ZMod.natCast_zmod_val]⟩
    refine ⟨fun h0 => h_x (key h0), fun h0 hnp => absurd (key ?_) hnp⟩
    exact h0
  completeness := by
    circuit_proof_start [main, pullIf, counted, areBytesTable, isHalfTable]
    simp only [AbstractInteraction.Guarantees, circuit_norm, h_input]
    exact ⟨⟨h_assumptions.1, by rw [ZMod.val_zero]; norm_num⟩, fun _ => h_assumptions.2⟩

end LambdaVM.Counted.Demo
