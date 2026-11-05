module

public import LambdaVM.Field

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

end LambdaVM
