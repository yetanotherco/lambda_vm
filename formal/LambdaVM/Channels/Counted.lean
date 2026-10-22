module

public import Clean.Air.Balance
public import LambdaVM.Tables.Bitwise
public import Mathlib.Tactic.LinearCombination

@[expose] public section

/-!
# Counted pulls

A deduplicated row pulls its range checks with multiplicity `-μ`, `μ ∈ [0, 255]`
(`MU_MAX`, PR #1016). Clean grants a channel guarantee at multiplicity `-1` only.
`counted N channel` is the raw channel that grants it at `-1, …, -N` and charges every other
nonzero multiplicity with the requirement, exactly as `Channel.toRaw` does for `-1`.

The price is in the balance premise. `exists_push_of_pull` needs `N · length < char F`;
Clean's `BalancedInteractions` only provides `length < char F`, and `not_consistent` shows the
gap is real: for `N ≥ 2` the counted channel is not `Consistent`, so it cannot be finished in a
`SoundEnsemble` until that premise carries the bound. With `N = 1` everything coincides
(`counted 1 channel` is `Normal`).
-/

namespace LambdaVM.Counted

variable {F : Type} [FiniteField F] [DecidableEq F]
variable {Message : TypeMap} [ProvableType Message]

/-- `mult = -k` for a count `1 ≤ k ≤ N`. -/
def IsPull (N : ℕ) (mult : F) : Prop := ∃ k : ℕ, 1 ≤ k ∧ k ≤ N ∧ mult = -(k : F)

/-- `channel` with pulls allowed to carry a count up to `N`. -/
def counted (N : ℕ) (channel : Channel F Message) : RawChannel F where
  name := channel.name
  arity := size Message
  Guarantees mult message data :=
    IsPull N mult → channel.Guarantees (fromElements message) data
  Requirements mult message data :=
    mult ≠ 0 → ¬ IsPull N mult → channel.Guarantees (fromElements message) data

/-- Pull `msg` with multiplicity `-count`. -/
@[circuit_norm]
def pullIf (N : ℕ) (channel : Channel F Message) (count : Expression F)
    (msg : Message (Expression F)) : Circuit F Unit := fun _ =>
  ((), [.interact ⟨counted N channel, -count, toElements msg, true⟩])

/-- The provider side: emit `msg` with multiplicity `mult`. -/
@[circuit_norm]
def emit (N : ℕ) (channel : Channel F Message) (mult : Expression F)
    (msg : Message (Expression F)) : Circuit F Unit := fun _ =>
  ((), [.interact ⟨counted N channel, mult, toElements msg, false⟩])

instance {N : ℕ} {channel : Channel F Message} {count : Expression F} :
    ExplicitCircuits (F := F) (pullIf N channel count) where
  output _ _ := ()
  localLength _ _ := 0
  operations msg _ := [.interact ⟨counted N channel, -count, toElements msg, true⟩]
  channelsWithGuarantees _ _ := [counted N channel]

instance {N : ℕ} {channel : Channel F Message} {mult : Expression F} :
    ExplicitCircuits (F := F) (emit N channel mult) where
  output _ _ := ()
  localLength _ _ := 0
  operations msg _ := [.interact ⟨counted N channel, mult, toElements msg, false⟩]
  channelsWithGuarantees _ _ := []

attribute [explicit_circuit_no_unfold] pullIf emit

/-- `RawChannel.Normal` with the pull multiplicity `-1` widened to `-1, …, -N`. -/
class Normal (N : ℕ) (channel : RawChannel F) : Prop where
  grts_of_reqs : ∀ (msg : Vector F channel.arity) (mult : F) data, mult ≠ 0 → ¬ IsPull N mult →
    channel.Requirements mult msg data → ∀ mult', IsPull N mult' → channel.Guarantees mult' msg data
  grts_of_not_pull : ∀ (msg : Vector F channel.arity) (mult : F) data, ¬ IsPull N mult →
    channel.Guarantees mult msg data

instance (N : ℕ) (channel : Channel F Message) : Normal N (counted N channel) where
  grts_of_reqs := by
    intro msg mult data h0 hnp reqs mult' hp
    simp only [counted] at reqs ⊢
    exact fun _ => reqs h0 hnp
  grts_of_not_pull := by
    intro msg mult data hnp
    simp only [counted]
    exact fun hp => absurd hp hnp

/-- On a message whose nonzero multiplicities are all counted pulls, the balance is minus a
natural number bounded by `N` per interaction, and positive as soon as one pull is active. -/
lemma balanceOf_counted (N : ℕ) (msg : Array F) :
    ∀ (l : List (Interaction F)),
    (∀ i ∈ l, i.msg = msg → i.mult ≠ 0 → IsPull N i.mult) →
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
        · rw [if_pos hm, h0, hb, zero_add]
        · intro a ha' ham ha0
          rcases List.mem_cons.mp ha' with rfl | ha'
          · exact absurd h0 ha0
          · exact ha a ha' ham ha0
      · obtain ⟨k, hk1, hkN, hk⟩ := h i (by simp) hm h0
        refine ⟨k + s, by omega, ?_, ?_⟩
        · rw [if_pos hm, hk, hb]; push_cast; ring
        · intro a _ _ _; omega
    · refine ⟨s, by omega, ?_, ?_⟩
      · rw [if_neg hm, hb, zero_add]
      · intro a ha' ham ha0
        rcases List.mem_cons.mp ha' with rfl | ha'
        · exact absurd ham hm
        · exact ha a ha' ham ha0

/-- `exists_push_of_pull` for counted pulls. The balance premise must bound the total count,
`N * length`, by the characteristic; Clean's `BalancedInteractions` bounds `length` only. -/
theorem exists_push_of_pull (N : ℕ) (interactions : List (Interaction F))
    (bound : N * interactions.length < ringChar F ∨ ringChar F = 0)
    (balance : ∀ msg, balanceOf interactions msg = 0) :
    ∀ a ∈ interactions, IsPull N a.mult →
      ∃ b ∈ interactions, b.msg = a.msg ∧ b.mult ≠ 0 ∧ ¬ IsPull N b.mult := by
  intro a ha hpull
  by_contra! hno
  obtain ⟨s, hs, hb, h1⟩ := balanceOf_counted N a.msg interactions hno
  have hlen : 0 < interactions.length := List.length_pos_of_mem ha
  have hNlen : N ≤ N * interactions.length := Nat.le_mul_of_pos_right N hlen
  have ha0 : a.mult ≠ 0 := by
    obtain ⟨k, hk1, hkN, hk⟩ := hpull
    rw [hk, neg_ne_zero]
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
  intro _
  have a_channel_eq := (reqs a a_mem).1
  have a_msg_size : a.msg.size = channel.arity := by rw [a.same_size, a_channel_eq]
  suffices channel.Guarantees a.mult ⟨a.msg, a_msg_size⟩ data by convert this
  by_cases a_pull : IsPull N a.mult
  case neg => exact Normal.grts_of_not_pull ⟨a.msg, a_msg_size⟩ a.mult data a_pull
  obtain ⟨b, b_mem, b_msg_eq, b_ne_zero, b_not_pull⟩ :=
    exists_push_of_pull N interactions bound balance a a_mem a_pull
  apply Normal.grts_of_reqs ⟨a.msg, a_msg_size⟩ b.mult data b_ne_zero b_not_pull _ a.mult a_pull
  have ⟨b_channel_eq, b_reqs⟩ := reqs _ b_mem
  symm at b_channel_eq
  simp only [b_msg_eq] at b_reqs
  convert b_reqs

/-! ## `N = 1` is Clean's normal channel -/

omit [DecidableEq F] in
lemma isPull_one_iff (mult : F) : IsPull 1 mult ↔ mult = -1 := by
  constructor
  · rintro ⟨k, hk1, hk, rfl⟩
    have : k = 1 := by omega
    subst this; simp
  · rintro rfl; exact ⟨1, le_rfl, le_rfl, by simp⟩

instance (channel : Channel F Message) : (counted 1 channel).Normal where
  grts_of_reqs := by
    intro msg mult data h0 h1 reqs
    simp only [counted, isPull_one_iff] at reqs ⊢
    exact fun _ => reqs h0 h1
  grts_of_ne_neg_one := by
    intro msg mult data h1
    simp only [counted, isPull_one_iff]
    exact fun h => absurd h h1

example (channel : Channel F Message) : (counted 1 channel).Consistent := inferInstance

end LambdaVM.Counted

/-! ## `N ≥ 2` is not `Consistent` under Clean's balance premise -/

namespace LambdaVM.Counted
variable {Message : TypeMap} [ProvableType Message]

/-- The witness `BalancedInteractions` admits but the counted channel cannot survive:
`p / N` pulls of `-N` and `p % N` pulls of `-1` on one message, nothing pushed. It has fewer
than `p` interactions, balances to `-p = 0`, and claims the guarantee on a message that is
not in the table. -/
theorem not_consistent (N : ℕ) (hN : 2 ≤ N) (hN' : N ≤ 255)
    (channel : Channel (F p) Message) (m : Message (F p)) (data : ProverData (F p))
    (hm : ¬ channel.Guarantees m data) : ¬ (counted N channel).Consistent := by
  rintro ⟨hc⟩
  let mk : F p → Interaction (F p) := fun mult =>
    ⟨counted N channel, mult, (toElements m).toArray, by simp [counted], true⟩
  let l := List.replicate (p / N) (mk (-(N : F p))) ++ List.replicate (p % N) (mk (-1))
  have hdvd : ¬ N ∣ p := by
    intro h
    rcases (Fact.out : p.Prime).eq_one_or_self_of_dvd N h with h1 | h2
    · omega
    · have : (255 : ℕ) < p := by decide
      omega
  have hmod : 0 < p % N := Nat.pos_of_ne_zero (fun h => hdvd (Nat.dvd_of_mod_eq_zero h))
  have hmem : mk (-1) ∈ l := by
    simp only [l, List.mem_append, List.mem_replicate]
    exact Or.inr ⟨by omega, trivial⟩
  have hlen : l.length < ringChar (F p) ∨ ringChar (F p) = 0 := by
    left
    rw [ZMod.ringChar_zmod_n]
    simp only [l, List.length_append, List.length_replicate]
    have h1 : p / N ≤ p / 2 := Nat.div_le_div_left hN (by norm_num)
    have h2 : p % N < N := Nat.mod_lt _ (by omega)
    have h3 : p / 2 + 255 < p := by decide
    omega
  have hbal : ∀ msg, balanceOf l msg = 0 := by
    intro msg
    simp only [l, balanceOf_append]
    rw [balanceOf_eq_of_const_mult' (mult := -(N : F p)) (by intro i hi; rw [(List.mem_replicate.mp hi).2]),
      balanceOf_eq_of_const_mult' (mult := -1) (by intro i hi; rw [(List.mem_replicate.mp hi).2])]
    simp only [List.countP_replicate]
    by_cases hmsg : (toElements m).toArray = msg
    · simp only [mk, hmsg, decide_true, if_true]
      have : ((N * (p / N) + p % N : ℕ) : F p) = 0 := by
        rw [Nat.div_add_mod]; exact ZMod.natCast_self p
      push_cast at this
      linear_combination -this
    · simp [mk, hmsg]
  have hreqs : ∀ i ∈ l, i.channel = counted N channel ∧ i.Requirements data := by
    intro i hi
    simp only [l, List.mem_append, List.mem_replicate] at hi
    rcases hi with ⟨_, rfl⟩ | ⟨_, rfl⟩
    · refine ⟨rfl, ?_⟩
      simp only [Interaction.Requirements, mk, counted, Interaction.msgVector]
      intro _ hnp
      exact absurd ⟨N, by omega, le_rfl, rfl⟩ hnp
    · refine ⟨rfl, ?_⟩
      simp only [Interaction.Requirements, mk, counted, Interaction.msgVector]
      intro _ hnp
      exact absurd ⟨1, le_rfl, by omega, by simp⟩ hnp
  have hg := hc l data ⟨hlen, hbal⟩ hreqs (mk (-1)) hmem
  simp only [Interaction.Guarantees, mk, counted, Interaction.msgVector] at hg
  apply hm
  have h := hg trivial ⟨1, le_rfl, by omega, by simp⟩
  convert h
  simp [ProvableType.fromElements_toElements]

end LambdaVM.Counted

/-! ## A deduplicated row

`μ` is range-checked as a byte through `ARE_BYTES` and `x` is pulled from `IS_HALF` with
multiplicity `-μ`: the shape of every dedup chip after PR #1016. The counted channel is listed
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
      ProvableType.fromElements_eval_toElements, h_input] at h_holds ⊢
    obtain ⟨⟨h_mu, _⟩, h_x⟩ := h_holds
    -- an active row pulls with the count `μ.val ∈ [1, 255]`
    have key : input_mu ≠ 0 → IsPull 255 (-input_mu) := fun h0 =>
      ⟨input_mu.val, by have := (ZMod.val_ne_zero _).mpr h0; omega, by omega,
        by rw [ZMod.natCast_zmod_val]⟩
    refine ⟨fun h0 => h_x (key h0), fun h0 hnp => absurd (key ?_) hnp⟩
    exact h0
  completeness := by
    circuit_proof_start [main, pullIf, counted, areBytesTable, isHalfTable]
    simp only [AbstractInteraction.Guarantees, circuit_norm,
      ProvableType.fromElements_eval_toElements, h_input]
    exact ⟨⟨h_assumptions.1, by rw [ZMod.val_zero]; norm_num⟩, fun _ => h_assumptions.2⟩

end LambdaVM.Counted.Demo
