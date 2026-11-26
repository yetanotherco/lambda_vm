# Lambda VM formal model (Clean / Lean 4)

Lean 4 project that models the Lambda VM arithmetization in
[Clean](https://github.com/Verified-zkEVM/clean) and proves it against `spec/`.

What it covers: the AIR tables, the LogUp buses (Clean "channels") and the
semantic specification of each table. What it does not cover: the STARK /
FRI / WHIR prover, the LogUp argument itself, the executor, GPU code.

## Layout

- `LambdaVM/Field.lean` — Goldilocks as the base field (`F p`), with the size
  facts Clean's gadgets ask for. Primality comes from CompPoly's Pratt certificate.
- `LambdaVM/Templates/` — every constraint template the vendored spec declares,
  as helpers the chips call (`isBit`, `isByte`, `areBytes`, `add`, `sub`, `sign`,
  `neg`), each with a `FormalAssertion` whose `Spec` is the template's meaning
  taken from `spec/src/*.toml`: `IsBit.circuit`, `IsByteCheck.circuit`,
  `AreBytesCheck.circuit`, `Add.circuit`, `SignCheck.circuit`, `NegCheck.circuit`.
  The `kind` of a component lives in `spec/src/signatures.toml`, not in its own
  file, and six tags carry `kind = "template"` there: `IS_BIT`, `IS_BYTE`, `ADD`,
  `SUB`, `NEG` and `SIGN`. A newer spec than the one under `spec/` declares three
  more, `REG`, `REGW` and `IS_WHH`, which cannot be modelled until their
  dependencies are: `REG` and `REGW` read `MEMW`, and `IS_WHH` reads `IS_WH`.
  The chips call `isBit`, `add` and `sub`; no chip calls `isByte` or `areBytes`
  yet (SHIFT pulls `ARE_BYTES` directly). `sign` has its first caller in MUL;
  `neg` has its only one in DVRM, which calls it twice, for the absolute values
  of the remainder and the divisor. `shift` reads `MSB16` directly rather than
  through `SIGN`, which is what `shift.toml` asks for.
- `LambdaVM/AluOp.lean` — the `alu_op` descriptors of `types.rs`, one enumeration
  for the `opsel` of BYTE_ALU and the low bits of the ALU flag byte.
- `LambdaVM/Mu.lean` — the bound every counted pull uses, one constant for the
  model rather than one per chip, because `Channel.counted N` is a different
  channel for each `N`. Its value is twice `u8::MAX`, the largest multiplicity a
  chip adds up: MUL's `μ_lo + μ_hi` and DVRM's `μ_q + μ_r`.
- `LambdaVM/StaticLookup.lean` — `StaticLookupChannel`: a table the verifier
  knows, given as a list of rows, with `guarantees_iff` relating the list to the
  semantic reading of a row, and `toChannel` for the components that read it.
  Clean main replaced its own `StaticLookupChannel` by `Air.Flat.Lookup`, whose
  rows are witness programs of the row index, so the structure lives here.
- `LambdaVM/Buses/` — the buses that are not static tables. `Alu.lean` is the
  unified ALU bus: message layout as in `BusId::Alu` and one `Guarantees` clause
  per `opsel` (AND, OR and XOR for BYTEWISE, then LT, EQ, SHIFT, MUL and
  DIVREM), which is now every value `alu_op` takes in `types.rs`. Each clause
  takes as hypotheses the `IS_WORD` assumptions of its chip's spec (BYTEWISE:
  none; LT: the low words; EQ: all four words; SHIFT, MUL and DIVREM: none). The
  flag byte is read as a natural, so a chip pushing with one `opsel` discharges
  the other clauses by `omega`.
- `LambdaVM/Tables/` — one file per chip. `Bitwise.lean` transcribes the
  precomputed row function of `prover/src/tables/bitwise.rs` and defines each bus
  the table serves as a `StaticLookupChannel` (read through `toChannel`), with
  the semantic `Guarantees` proven equivalent to table membership. `Lt.lean` is
  the LT chip: the row circuit has the constraints and interactions of `lt.rs`
  (in the spec's order, except the `μ` delta below), `core` is the
  arithmetic argument of `spec/src/lt.toml` on naturals, `Honest` is the row
  `generate_lt_trace` produces, `RowSpec` is what an active row establishes, and
  the `GeneralFormalCircuit` proves that a row that pulls its range checks
  satisfies `RowSpec` and pushes a correct ALU result (soundness) and that an
  honest row satisfies everything (completeness), for byte multiplicities
  `μ ∈ [0, 255]`, deduplicated rows included. `Eq.lean` is the EQ chip in the
  same shape: the `sub` template, four IS_HALF pulls, a ZERO pull on the sum of
  the halves, `res = eq xor invert`; its `μ` is a bit, as in the spec and `eq.rs`. Completeness covers
  the honest row and the all-zero padding row.

- `LambdaVM/Ensembles/` — tables wired together through their channels.
  `EqLookup.lean` is the first one: `pushStatic` is the provider side of any
  static lookup channel (its row is the vector of multiplicities, the role of
  the `μ` columns of BITWISE), and `eqLookup` chains the IS_HALF provider, the
  ZERO provider and the EQ chip in dependency order. `eq_rows_correct` states
  that, for any witness whose constraints hold and whose channels balance,
  every EQ row with `μ = 1` and range-checked operands carries the equality
  bit. The ensemble has no ALU consumer yet (the CPU is not modelled), so
  balance on the ALU channel forces `μ = 0` on every EQ row and the theorem is
  trivially true. That IS_HALF → ZERO → EQ compose is shown by `eqLookup`
  itself, which only type-checks as a `SoundEnsemble` because every channel is
  finished before the tables that pull from it; the theorem says something
  about active rows once a component pulls from ALU. `LtStaged.lean` is the
  counted-pulls ensemble (see below).
- `LambdaVM/Tables/ShiftCore.lean` and `Shift.lean` — the SHIFT chip. `ShiftCore`
  is the two-phase shift argument of `spec/src/shift.toml` on naturals (HWSL
  per limb, then whole limbs, left and right, logical and arithmetic), ending in
  `ShiftCore.main`, which takes exactly the facts the row gives (decoded
  `bit_shift`, one-hot `limb_shift`, sign bit, HWSL outputs or their `zbs`
  overrides) and concludes `shiftResult`. `Shift.lean` mirrors the 22 constraints
  and 17 interactions of `shift.rs` and proves soundness and completeness, the
  latter for both the honest active row and the padding row.
- `LambdaVM/Tables/Bytewise.lean` — the BYTEWISE chip: `AND`, `OR` and `XOR` of
  two 64-bit operands, looked up one byte triple at a time in `BYTE_ALU` with a
  common `op` and pushed on the `ALU` bus with the operands re-limbed to
  `DWordWL`. `Bits.compose` is the arithmetic behind the re-limbing: an
  operation that acts on each bit independently commutes with reading bytes as a
  little-endian number, so the eight byte results assemble into the operation on
  the whole values. `op` is never constrained — `BYTE_ALU` only has rows for the
  three operations, so the lookup is the range check — and the operand bytes are
  not assumed either, since that same guarantee supplies them, which is why
  soundness needs no assumption at all. `IS_BYTE[μ]` is a constraint of the chip,
  which is what makes the eight pulls counted pulls of multiplicity at most 255.
- `LambdaVM/Tables/MulCore.lean` and `Mul.lean` — the MUL chip, the 128-bit
  product of two 64-bit operands, each read as signed or unsigned by its own
  flag, so one chip serves `MUL`, `MULH`, `MULHU` and `MULHSU`. `MulCore` is the
  arithmetic on naturals, the way `ShiftCore` is for SHIFT: the schoolbook
  convolution of the two operands over sixteen 16-bit halves after sign
  extension, with the terms at or above `2^128` named rather than left implicit,
  the `2^51` bound on each `raw_product` limb, and both directions of the
  four-step carry chain. `Mul.lean` is the translation between a row and those
  statements. The two `SIGN` templates give the sign bits that make the upper
  four halves two's complement; the row pushes two `ALU` messages for one
  product, `high = 0` for `lo` and `high = 1` for `hi`, and every range check
  fires with `μ_lo + μ_hi`, which is what sets the model's counted-pull bound.
- `LambdaVM/Tables/DvrmCore.lean` and `Dvrm.lean` — the DVRM chip, truncated
  division of two 64-bit operands, both read signed or unsigned by one flag, so
  one chip serves `DIV`, `DIVU`, `REM` and `REMU`. The chip never divides: it
  witnesses the quotient and the remainder and checks them. `DvrmCore` is the
  arithmetic on naturals, the way `MulCore` is for MUL, and `tdiv_unique`, the
  uniqueness of truncated division, is its core; `dvrm_core` and `honest_nat`
  are the two directions the chip consumes. `Dvrm.lean` is the translation
  between a row and those statements: `d · q = n − r` through two `ALU` messages
  with `opsel = MUL`, `|r| < |d|` through one with `opsel = LT`, two `NEG`
  templates for the absolute values, the rule that a nonzero remainder carries
  the sign of the numerator, and division by zero forcing `q` to `2^64 − 1`. The
  signed overflow `−2^63 / −1` needs no case of its own, since `Int.tdiv`
  already wraps. `ProverAssumptions` is `Honest r ∨ Padding r`, as in SHIFT:
  `dvrm.rs` builds its trace from `zeroed_fe_vec`, so a padding row is all zero,
  and an all-zero row is not honest, since an honest row with `d = 0` carries
  `div_by_zero = 1` and `q = 2^64 − 1`. The row groups its four-half operands
  rather than holding 34 flat fields, because `deriving ProvableStruct` does not
  reach that many components; the flattened order is unchanged, so it is still
  the column order of `dvrm.rs`.

## Lessons for big chips (from SHIFT, MUL and DVRM)

- Precompute the `ElaboratedCircuit` instance (`instance elaborated := by
  elaborate_circuit`) and pass it explicitly; without it every proof of a
  39-operation circuit times out at elaboration.
- Keep large row polynomials folded behind a definition (`shiftedPoly`) with a
  lemma that pushes `Expression.eval` inside, and bridge to the natural-number
  version with one cast lemma.
- State `soundness` and `completeness` as standalone theorems and reference
  them from the structure; the structure itself otherwise exceeds the heartbeat
  budget while abstracting the proof terms.
- `rw ... at h` closes goals by `rfl`; a following `simp`/`ring` then fails with
  "no goals". Guard with `try` where the rewrite may already close the goal.
- Clean's `doc/performance-problems.md` is the catalogue for elaboration and
  kernel blowups, and DVRM hit two of its items hard enough to be worth naming.
  Item 7, `have` over `obtain ⟨..⟩ :=` on a big conjunction, is a *kernel*
  failure here, not only a slow one: destructuring the result of a large
  application gave `(kernel) deep recursion detected` with no failing tactic,
  while `have h := …` and then `h.1` / `h.2` went through unchanged.
- Item 9, keeping powers opaque to `omega`, costs more than it looks. The four
  limb equations of a DVRM row are 32-bit facts, but with the `% 2^128` equation
  still in scope one declaration went from 15 seconds to over twenty minutes.
  `clear` the wide hypotheses before calling `omega`, which is why the soundness
  proof clears so aggressively.
- `simp` can build a proof term the elaborator accepts and the kernel rejects.
  On DVRM's nested `if`s — the division-by-zero and remainder cases — it reported
  the failure at the declaration header with no line to blame; rewriting the
  same steps with the primitive `ite_eq_left` and `ite_eq_right` removed it.
  When a kernel error has no obvious source, bisect by moving a closing `sorry`
  through the proof: the elaborator reports nothing.

## Differential check against the Rust prover

The Lean circuits are hand transcriptions of `prover/src/tables/*.rs`. The check
that they are the same polynomials and the same bus interactions is mechanical:

1. `lake env lean --run LambdaVM/Export.lean` writes
   `export/{lt,eq,shift,bytewise,mul,dvrm}.json`:
   every `assert` as an expression tree and every `interact` with its channel,
   direction, multiplicity and message, variables numbered in Rust column order.
   Multiplicities are nonnegative; the direction carries the sign. It also writes
   `export/bitwise.json`: the size of the BITWISE table and of every static
   lookup channel, and a sample of the table (every `z` with edge bytes, which
   includes the last row, and 256 pseudo-random rows) with the precomputed
   columns of each row and the message every channel serves there.
2. `prover/src/tests/lean_diff.rs` is the prover test that reads them:
   `cargo test -p lambda-vm-prover lean_diff -- --nocapture`.

For each table the test evaluates the Lean asserts and the Rust `ConstraintSet`
on 64 random rows and matches them as multisets of functions, does the same for
the interactions, and checks that every Lean constraint vanishes on honest
traces from `generate_*_trace`, padding rows included. Each test asserts that
all four unmatched lists are empty: no table has a Lean-only or a Rust-only
constraint or interaction left. The Rust side of an interaction is what the
verifier enforces: the multiplicity and the LogUp fingerprint
`z − (bus + α·e₀ + α²·e₁ + …)` as `emit_multiplicity` and `emit_fingerprint`
emit them, captured like the constraints and evaluated at fixed `z` and `α`;
the Lean side folds the exported message the same way. Direction (a Rust sender
is Clean's pull), multiplicity and fingerprint are matched separately. For
BITWISE the test checks the table and channel sizes, compares the sampled rows
with `generate_bitwise_row` and each channel's messages with the fingerprints of
the receivers of
`bitwise::bus_interactions()` on the same bus, and checks all 2^20 rows of
`generate_bitwise_row` against a Rust transcription of `Bitwise.row` that the
sampled rows pin to the Lean export.

The models follow the multiplicity constraints of PR #1016. What `lean_diff`
enforces, on every PR:

- EQ: 5 constraints and 6 interactions, exact match.
- SHIFT: 22 constraints and 17 interactions, exact match.
- LT: 6 constraints and 10 interactions, exact match, with `μ` a byte as in
  `lt.rs`.
- BITWISE: same size, all 2^20 rows, 656 sampled rows and 8 channels (10 Rust
  receivers), exact match.
- BYTEWISE: 0 constraints and 10 interactions, over 26 columns, exact match.
  `lean_diff_bytewise` feeds repeated operations so the rows carry `μ > 1` and
  one of them crosses `MU_MAX` into two rows, which is the only way to exercise
  the counted pull this chip is built on.
- MUL: 8 constraints and 25 interactions, over 26 columns. The column count, the
  eight constraints in the order `mul.rs` emits them (`IS_BIT` on each sign flag,
  `(1 − signed)·is_negative` for each, then the four `raw_product` convolutions)
  and the interaction split (`ARE_BYTES` once, `IS_HALF` sixteen times, `MSB16`
  twice, `IS_B20` four times, `ALU` twice) all agree with `mul.rs`, and
  `lean_diff_mul` checks it, asking for both halves of one repeated product so
  the row carries `μ_lo > 1` and `μ_hi > 1` and the range checks fire with
  their sum.
- DVRM: 19 constraints and 35 interactions, over 34 columns. The column count
  (`cols::NUM_COLUMNS = 34`), the nineteen constraints in the order `dvrm.rs`
  emits them and the interaction split (`ARE_BYTES` once, `IS_HALF` twenty
  times, `MSB16` three times, `ZERO` six times, `ALU` five) all agree with
  `dvrm.rs`. That `ZERO` count follows the Rust code and not its module comment,
  which says the bus is sent five times while the code emits six: two per `NEG`
  template, plus `overflow` and `div_by_zero`. `lean_diff_dvrm` checks all of
  it, with division by zero and the signed overflow `−2^63 / −1` named among
  the cases rather than left to the random ones, and one repeated operation
  asked for both results so the row carries `μ_q > 1` and `μ_r > 1`.

Every modelled chip now has a test here. The three that landed without one,
BYTEWISE, MUL and DVRM, were at parity by inspection until the tests were
written, and writing them found two real differences: BYTEWISE sent the
`ARE_BYTES` pair in the other order, and MUL's four convolutions were the
negation of what `mul.rs` emits.

The check compares polynomials, not zero sets: it caught `μ·(μ − 1)` versus
`μ·(1 − μ)` in EQ, which is why the Lean bit constraints use the spec's form.

`export/` is a committed artifact; `.github/workflows/formal.yml` builds
`formal/`, re-runs the export and fails if it differs, on every PR that touches
`formal/`.

## Counted pulls

After PR #1016 a deduplicated row pulls its range checks with multiplicity `μ`,
`μ` a byte. Clean's typed channel grants the guarantee to a pull of multiplicity
`1` only, and requires every pull to carry `0` or `1`; any other nonzero
multiplicity is a requirement the row cannot meet.
`LambdaVM/Channels/Counted.lean` is this model's own generalisation. It is **not**
the channel LT pulls on — that one is the fork's `Channel.counted`, whose
semantics differ; see the comparison at the end of this section.

- `countedCharged N channel` grants the guarantee to a pull of `1, …, N` and
  charges every other nonzero interaction, mirroring `Channel.toRaw`.
  `Counted.pullIf`/`Counted.emit` are its circuit operations, with
  `ExplicitCircuits` instances so elaboration works. The name keeps it apart from
  the fork's `Channel.counted`, which this model also pulls on and which treats an
  out-of-range pull differently; the table at the end of this section is the
  comparison. Clean has its own `Channel.pullIf`, named below, and they are not
  the same function.
- `exists_push_of_pull` and `consistent` are Clean's lemmas for this channel.
  Their balance premise is `N · length < char F`. Clean's `BalancedInteractions`
  carries no length bound of its own: the bound is the `fits` of the `Balance`
  the ensemble is checked against, and reaches soundness as the
  `InteractionsFit` premise of `TableSoundness`. For `Balance.logUp`, `fits` is
  `count < char F`.
- `not_consistent` shows that Clean's `BalancedInteractions` is too weak a
  premise for a counted channel (the field-sum premise of `consistent` cannot be
  replaced by it): for `2 ≤ N ≤ 255`
  over Goldilocks, one pull of `N` and one pull of `1` on a message outside the
  table satisfy `BalancedInteractions` — Clean counts the pull of `N` as a push
  of the message, so the pull of `1` is matched — while the counted channel
  reads both as pulls and the guarantee of the second is false. So
  `RawChannel.Consistent` is false and the channel cannot be finished in a
  `SoundEnsemble`. With `N = 1` the channel is `Normal`, as in Clean.
- `Demo.circuit` is the shape of a dedup row: `ARE_BYTES[μ, 0]` then a counted
  `IS_HALF` pull with multiplicity `μ`. The counted channel sits in
  `channelsWithRequirements`: an out-of-range `μ` is only ruled out by the byte
  guarantee, which lawfulness does not see but soundness does.

Counted pulls in the LT model need two things Clean main lacks:

1. A `Balance` for counted pulls: `BalancedInteractions` matching a pull of
   `1, …, N`, and a `fits` that bounds `N * count` rather than `count` by the
   characteristic.
2. A dependency-aware finishing order: a table may hold guarantees and
   requirements on the same channel when the requirements depend only on
   channels finished earlier.

The model takes both from the `yetanotherco/clean` fork, pinned in
`lakefile.toml` by sha at
[`8552c7a1`](https://github.com/yetanotherco/clean/commit/8552c7a10dac340de34bfff9e19a6967ed3b1bba)
— by sha, not by branch name, so the pin survives the branch moving, being
renamed or deleted once the upstream PRs land. That sha is the merge putting the
counted-pulls work onto the fork's `pr-base-upstream-main`, where the fork's PRs
land; its two parents are the upstream mirror `a985c76c` and `e30c5c44`, the work
itself: `Balance.logUpUpTo N` with `RawChannel.ConsistentUpTo N`, the typed
`Channel.counted N`, and `Ensemble.tableSoundness_of_stagedRequirements` /
`_stagedRequirementsOn` (soundness by finishing order of channels, and of
message classes within a channel). On top of it:

- `Tables/Lt.lean` pulls `IS_HALF` and `MSB16` with multiplicity `μ`, bounded by
  its own pull of `ARE_BYTES[μ, 0]`, exactly as `lt.rs`.
- `Tables/Bytewise.lean` pulls `BYTE_ALU` eight times with multiplicity `μ`,
  bounded by its own `IS_BYTE[μ]` constraint.
- `Tables/Mul.lean` pulls `IS_HALF` sixteen times and `IS_B20` four times with
  `μ_lo + μ_hi`, bounded by its own `ARE_BYTES[μ_lo, μ_hi]`. This is the chip
  that fixes `LambdaVM.muMax` at `2 * 255`: the bound is shared, so LT and
  BYTEWISE pull through the same channel even though their own `μ` is a byte.
- `Tables/Dvrm.lean` pulls `IS_HALF` twenty times with `μ_q + μ_r`, bounded by
  its own `ARE_BYTES[μ_q, μ_r]`, and is the one chip that pulls the `ALU` bus.
- `Ensembles/LtStaged.lean` proves `tableSoundness`, that is
  `Ensemble.TableSoundness (Balance.logUpUpTo muMax)` for the LT ensemble,
  finishing `ARE_BYTES` before the counted `IS_HALF` and `MSB16` and those
  before ALU.

DVRM is the chip that pulls and pushes the *same* bus. A toy MIN table and
`Ensembles/CountedSameBus.lean` carried that shape while counted pulls were
being designed, and were removed once BYTEWISE exercised the mechanism on a real
chip; DVRM replaces them with the real thing. The table proof is unaffected,
since the chip receives on its pulls and owes on its pushes, so ordering the
`ALU` messages is an ensemble concern:
`Ensemble.tableSoundness_of_stagedRequirementsOn` finishes one bus in two classes of
messages, which is how a puller and a pusher of one bus are ordered.

That ensemble is not written yet, and it needs the two names for the `ALU` bus
resolved first. The six chips that push, DVRM's own two `DIVREM` pushes included,
use `Alu.channel.toRaw`, which requires every pull to carry `0` or `1`; DVRM's
multiplicity reaches 510, so it has to pull on `Alu.channel.counted muMax`. Those
are two different `RawChannel`s under one name, which `unique_channel_names`
rejects, so no ensemble can hold DVRM at all, not even on its own: its
`channelsWithRequirements` lists both, and `ZERO` likewise (counted for its own
pulls, plain for the ones inlined from `neg`). Moving the pushers to the counted
channel leaves `export/` byte-identical — the channel is a choice in the Lean
model, invisible in the AIR — but it does change proofs.

The toy MIN table and the `CountedSameBus` ensemble that stood in for DVRM are gone:
the real DVRM is here, so the stand-ins have nothing left to stand in for.
The provider and consumer stand-ins of `LtStaged.lean` conclude `True`, which is what the
next paragraphs are about.

That is not an end-to-end LT theorem. `TableSoundness` is one premise of a
sound ensemble, not the conclusion: it reads `witness.Assumptions →
witness.Constraints → witness.InteractionsFit balance → witness.Balanced
balance → witness.Spec`, and leaves `witness.Assumptions` for the caller to
discharge. In `LtStaged` that premise is not free, because the `ARE_BYTES`,
`IS_HALF` and `MSB16` providers are not tables. `provider` and
`countedProvider` are rows that push whatever value they are handed, under
`Assumptions input data := channel.Guarantees input.1 data`, and
`providerComponent` and `countedProviderComponent` carry the same assumption
into the ensemble. Each provider assumes its own answers are correct; nothing
in the file derives that from a row of BITWISE.

The LT ensemble also carries a consumer of the ALU bus (`consumer` in
`LtStaged.lean`), and that stand-in is not a concession. A
pull gains the channel's guarantee instead of owing it, so a consumer assumes
nothing on the soundness side; what the honest prover must know — that it pulls
a message the bus really answers — is a `ProverAssumptions`, which only
completeness may use. Without a consumer the bus would carry pushes and no
pulls at all, a balanced witness would have to send every `μ` on it to zero, and
every row spec guarded by `μ ≠ 0` would hold vacuously, starting with `LT.RowSpec` in
`LtStaged.tableSoundness`.

`EqLookup`, already in the model (PR #1027), is the contrast. Its `pushStatic` emits the
entries of a real `StaticLookupChannel` and proves the guarantee from
membership in the table, so `eq_rows_correct` discharges `witness.Assumptions`
outright — "no component assumes anything" — and states a conclusion about
rows: every EQ row with `μ = 1` and range-checked operands carries the
equality bit. There is no `lt_rows_correct`. Writing one needs `pushStatic`,
or a counted analogue of it, behind LT's three range checks, and this branch
has neither. What `LtStaged` does establish is that the staged finishing order
admits counted pulls, which is what it was written for.

`Channels/Counted.lean` above predates the fork, and its `countedCharged` is a
different channel from the fork's, not a copy. They part on what a pull of an
out-of-range multiplicity costs:

| | out-of-range pull |
|---|---|
| this file (`countedCharged`) | `mult ≠ 0 → ¬ IsPull N mult → Guarantees` — charged like a push: the row must justify the message |
| the fork (`Channel.counted`) | `mult = 0 ∨ CountUpTo N mult` — forbidden: a range obligation the row must discharge |

LT is on the fork's. This file's channel has one consumer,
`Demo.circuit`, and keeping it is deliberate: its `not_consistent` is the reason
the fork's balance carries the `N * count` bound, and its `consistentUpTo` shows
the channel is consistent under the fork's `BalancedUpTo N`. Nothing else uses it.

Moving LT onto the counted channel has one structural cost: **LT cannot share an
ensemble with EQ or SHIFT.** `Channel.counted` keeps the base channel's name
(`name := channel.name`), so `Bitwise.IsHalf.counted muMax`, which `Tables/Lt.lean`
pulls, and `Bitwise.IsHalf.toRaw`, which `Tables/Eq.lean` and `Tables/Shift.lean`
pull through Clean's `Channel.pullIf`, are two different `RawChannel`s both
called `IS_HALF`, and `Ensemble.unique_channel_names` rejects a `tables` list
holding both. `MSB16` is
the same. A combined ensemble would have to put EQ and SHIFT on the *same*
counted channel LT uses, at `N = 255`. Dropping them onto `Channel.counted 1`
would not do: it keeps the base name too, so it collides with `Channel.counted
255` for exactly the reason above, and the pinned fork has no lemma collapsing it
to the plain channel either. EQ and SHIFT reach the channel through Clean's
`Channel.pullIf`, so moving them is a proof obligation about their rows, not a
rename. No ensemble
here needs it yet: `LtStaged` holds LT only.

## What is intentionally not proven

Scope limits, not bugs. Each is also noted where it applies.

- **The lookups a row makes are assumed, not proven.** Every chip theorem is
  about one row, given that each range check or lookup the row pulls is
  answered correctly. Nothing discharges that for the whole system:
  `eq_rows_correct` is trivially true because no ALU consumer is modelled, the
  ensembles this branch adds prove `Ensemble.TableSoundness` from providers
  that assume their own answers, and SHIFT has no ensemble. A negative
  multiplicity in an unmodelled table that shares `IS_HALF` (the bug PR #1016
  fixed) leaves every theorem and `lean_diff` green.
- **There is no `lt_rows_correct`.** LT's soundness and completeness do cover
  the deduplicated rows `lt.rs` produces (`μ = 2…255`), but the `ARE_BYTES`,
  `IS_HALF` and `MSB16` providers assume their own answers, so nothing
  concludes about the LT rows of a real trace. See "Counted pulls".
- **The Lean BITWISE table is compared with Rust on 656 sampled rows.** The
  loop over all 2^20 rows compares `generate_bitwise_row` with a Rust
  transcription of `Bitwise.row`, not with Lean.
- **The meaning side is not checked.** The ALU guarantee has no consumer,
  since the CPU is not modelled. "Proved against the spec" means
  `spec/main@2eaf1f080`, not the `spec/` copy in this repository.

## Known differences from the Rust prover

- Every chip whose `μ` the spec types as `Byte` needs the counted treatment LT
  got above. BYTEWISE, MUL and DVRM have it; BRANCH is not modelled yet.
- `isByte` sends `ARE_BYTES[X, 0]`, as every single-byte check in the prover
  does; `spec/src/is_byte.toml` writes `[0, X]`.
- `bytewise.toml` is one of the files where the `spec/` copy in this repository
  and `spec/main` disagree in substance, not only in wording: here `μ` is still
  typed `BaseField` with no range check, while the branch types it `Byte` and
  adds `IS_BYTE[μ]`. The model follows the branch, because an unbounded `μ` on
  the pull side admits no sound bound for counted pulls at all.
- `dvrm.toml` lags in the same way, in two readings the model takes from
  `dvrm.rs` and the newer spec instead: the multiplicities are bytes,
  range-checked through `ARE_BYTES`, and `div_by_zero` pads to `0`, not `1`.
- The `spec/` copy in this repository lags the `spec/main` branch; the spec the
  models follow is that branch at `2eaf1f080` (EQ's `μ` is `Bit` with
  `IS_BIT[μ]` there). Constraints are cited from `spec/src/*.toml`, which the
  `.typ` pages render.

## Building

```sh
# once: Lean toolchain manager
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

cd formal
lake exe cache get     # clones the pinned dependencies, fetches Mathlib's prebuilt binaries
lake build             # also compiles the export script
```

Clean itself is compiled from source; only the modules we import are built.

## Conventions

- Files use Clean's module header: `module` / `public import ...` /
  `@[expose] public section`.
- One table or template per file. `Spec` states the meaning of the rows in
  terms of the spec's variables; `Assumptions` are the preconditions the
  caller table must establish.
- Proofs start with `circuit_proof_start` (or `circuit_proof_all` when it
  closes the goal). See Clean's `AGENTS.md` and `doc/proving-guide.md`.
- Clean's circuit inference does not unfold user-defined `Circuit` helpers. A
  helper used inside a `main` needs an `ExplicitCircuit` instance, bridged to
  its body with `ExplicitCircuit.of_eq rfl (by infer_explicit_circuit)`.
- A conditional interaction (Clean's `Channel.pullIf cond`) is only lawful when
  the same circuit constrains `cond` to be boolean: Clean requires every pull to
  carry
  multiplicity `0` or `1`, and a pull outside `{0, 1}` is a requirement the
  circuit cannot discharge at lawfulness.
