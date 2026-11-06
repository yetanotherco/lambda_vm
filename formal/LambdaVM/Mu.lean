module

public import LambdaVM.Field
public import Clean.Air.CountedChannel

@[expose] public section

/-!
# The bound on a counted pull

`Channel.counted N` is a different channel for each `N`: the name is the same, but both the
`Requirements` a puller owes and the `Guarantees` it receives mention `N`. Two chips pulling the
same bus therefore have to agree on it, or an ensemble holding both carries two channels under
one name and no provider can serve them.

So the bound is one constant for the whole model rather than one per chip, and it is the largest
any chip needs. That is MUL and DVRM: each of their two multiplicities is a byte, and their
range checks fire with the sum, which reaches `2 * 255`. LT and BYTEWISE pull with a single byte
multiplicity and would be content with `255`. They use this constant anyway; their own
constraints still bound their own `μ` by `255`, and the looser channel only widens what the
provider must be ready to serve.

Raising it costs no proof work: `Balance.MatchedUpTo.mono` and
`Channel.counted_consistentUpTo_of_le` carry a channel counted at `N` into any balance check
with a larger bound. The one price is the length bound of `Balance.logUpUpTo`, whose `fits` asks
`count * N < char F`. At `510` a trace may still hold around `3.6 · 10^16` interactions on a
channel, which no prover reaches.
-/

namespace LambdaVM

/-- The bound every counted pull in this model uses: twice `u8::MAX`, the largest multiplicity a
chip adds up, from MUL's `μ_lo + μ_hi` and DVRM's `μ_q + μ_r`. -/
abbrev muMax : ℕ := 510

/-! ## What a bit owes a channel counted at `muMax`

A chip that pulls a bus under a one-bit flag owes `IsBool` on `Channel.toRaw` and the weaker
`mult = 0 ∨ CountUpTo muMax mult` on `Channel.counted muMax`. The two are interchangeable for a
bit, and these five lemmas are the translation, kept here next to `muMax` because every chip
that moves a conditional pull onto a counted channel needs them. -/

/-- A zero multiplicity is not a count, so an inactive counted pull owes nothing. -/
lemma not_countUpTo_zero : ¬ Channel.CountUpTo muMax (0 : F p) := by
  rintro ⟨k, hk1, hk2, hk⟩
  have hv : ((k : ℕ) : F p).val = k :=
    val_natCast_of_lt_two_pow_40 (by simp only [muMax] at hk2; omega)
  rw [← hk, ZMod.val_zero] at hv
  omega

/-- A multiplicity of one is a count of one. -/
lemma countUpTo_one : Channel.CountUpTo muMax (1 : F p) :=
  Channel.countUpTo_of_val one_ne_zero (by simp only [muMax, ZMod.val_one]; omega)

/-- An active bit is a counted pull of one. -/
lemma countUpTo_of_eq_one {x : F p} (h : x = 1) : Channel.CountUpTo muMax x := by
  rw [h]; exact countUpTo_one

/-- A bit is an inactive pull or a counted pull of one: what the counted channel asks for
where `toRaw` asked only for the bit. -/
lemma eq_zero_or_countUpTo {x : F p} (h : IsBool x) : x = 0 ∨ Channel.CountUpTo muMax x := by
  rcases h with rfl | rfl
  · exact .inl rfl
  · exact .inr countUpTo_one

/-- The converse for a bit: a count rules out zero, so the bit is one. This is what lets a proof
that used to read `h : cond = 1` off a `pullIf` keep reading it off a `pullUpTo`. -/
lemma eq_one_of_countUpTo {x : F p} (hb : IsBool x) (h : Channel.CountUpTo muMax x) : x = 1 :=
  Or.resolve_left hb fun h0 => absurd (h0 ▸ h) not_countUpTo_zero

end LambdaVM
