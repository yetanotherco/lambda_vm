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
  taken from its `spec/*.typ` page: `IsBit.circuit`, `IsByteCheck.circuit`,
  `Add.circuit`, `SignCheck.circuit`, `NegCheck.circuit`. The `kind` of a
  component lives in `spec/src/signatures.toml`, not in its own file, and six
  tags carry `kind = "template"` there: `IS_BIT`, `IS_BYTE`, `ADD`, `SUB`, `NEG`
  and `SIGN`. A newer spec than the one under `spec/` declares three more,
  `REG`, `REGW` and `IS_WHH`, which cannot be modelled until their dependencies
  are: `REG` and `REGW` read `MEMW`, and `IS_WHH` reads `IS_WH`. `sign` and
  `neg`: `sign` has its first caller in MUL, while `neg` still has none, since
  `dvrm` is the only modelled-adjacent chip that uses it and DVRM is not written
  yet. `shift` reads `MSB16` directly rather than through `SIGN`, which is what
  `shift.toml` asks for.
- `LambdaVM/AluOp.lean` — the `alu_op` descriptors of `types.rs`, one enumeration
  for the `opsel` of BYTE_ALU and the low bits of the ALU flag byte.
- `LambdaVM/Mu.lean` — the bound every counted pull uses, one constant for the
  model rather than one per chip, because `Channel.counted N` is a different
  channel for each `N`. Its value is twice `u8::MAX`, the largest multiplicity a
  chip adds up, which is MUL's `μ_lo + μ_hi`.
- `LambdaVM/StaticLookup.lean` — `StaticLookupChannel`: a table the verifier
  knows, given as a list of rows, with `guarantees_iff` relating the list to the
  semantic reading of a row, and `toChannel` for the components that read it.
  Clean main replaced its own `StaticLookupChannel` by `Air.Flat.Lookup`, whose
  rows are witness programs of the row index, so the structure lives here.
- `LambdaVM/Buses/` — the buses that are not static tables. `Alu.lean` is the
  unified ALU bus: message layout as in `BusId::Alu`, the caller's `IS_WORD`
  assumptions on both operands as hypotheses, and one `Guarantees` clause per
  `opsel` (AND, OR and XOR for BYTEWISE, then EQ, LT, SHIFT and MUL). The flag byte
  is read as a natural, so a chip pushing with one `opsel` discharges the other
  clauses by `omega`.
- `LambdaVM/Tables/` — one file per chip. `Bitwise.lean` transcribes the
  precomputed row function of `prover/src/tables/bitwise.rs` and defines each bus
  the table serves as a `StaticLookupChannel` (read through `toChannel`), with
  the semantic `Guarantees` proven equivalent to table membership. `Lt.lean` is
  the LT chip: the row circuit mirrors `lt.rs` (constraints and interactions, in
  order), `core` is the
  arithmetic argument of `spec/lt.typ` on naturals, `Honest` is the row
  `generate_lt_trace` produces, `RowSpec` is what an active row establishes, and
  the `GeneralFormalCircuit` proves that a row that pulls its range checks
  satisfies `RowSpec` and pushes a correct ALU result (soundness) and that an
  honest row satisfies everything (completeness). `Eq.lean` is the EQ chip in the
  same shape: the `sub` template, four IS_HALF pulls, a ZERO pull on the sum of
  the halves, `res = eq xor invert`; its `μ` is a bit, as in the spec and `eq.rs`.

- `LambdaVM/Ensembles/` — tables wired together through their channels.
  `EqLookup.lean` is the first one: `pushStatic` is the provider side of any
  static lookup channel (its row is the vector of multiplicities, the role of
  the `μ` columns of BITWISE), and `eqLookup` chains the IS_HALF provider, the
  ZERO provider and the EQ chip in dependency order. `eq_rows_correct` is the
  first global theorem: for any witness whose constraints hold and whose
  channels balance, every EQ row with `μ = 1` and range-checked operands
  carries the equality bit. The lookup argument is what turns the pulled range
  checks into facts. `LtStaged.lean` is the counted-pulls ensemble (see below).
- `LambdaVM/Tables/ShiftCore.lean` and `Shift.lean` — the SHIFT chip. `ShiftCore`
  is the two-phase shift argument of `spec/shift.typ` on naturals (HWSL
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

## Lessons for big chips (from SHIFT)

- Precompute the `ElaboratedCircuit` instance (`instance elaborated := by
  elaborate_circuit`) and pass it explicitly; without it every proof of a
  38-operation circuit times out at elaboration.
- Keep large row polynomials folded behind a definition (`shiftedPoly`) with a
  lemma that pushes `Expression.eval` inside, and bridge to the natural-number
  version with one cast lemma.
- State `soundness` and `completeness` as standalone theorems and reference
  them from the structure; the structure itself otherwise exceeds the heartbeat
  budget while abstracting the proof terms.
- `rw ... at h` closes goals by `rfl`; a following `simp`/`ring` then fails with
  "no goals". Guard with `try` where the rewrite may already close the goal.

## Differential check against the Rust prover

The Lean circuits are hand transcriptions of `prover/src/tables/*.rs`. The check
that they are the same polynomials and the same bus interactions is mechanical:

1. `lake env lean --run LambdaVM/Export.lean` writes
   `export/{lt,eq,shift,bytewise,mul}.json`:
   every `assert` as an expression tree and every `interact` with its channel,
   direction, multiplicity and message, variables numbered in Rust column order.
   Multiplicities are nonnegative; the direction carries the sign.
2. `prover/src/tests/lean_diff.rs` is the prover test that reads them:
   `cargo test -p lambda-vm-prover lean_diff -- --nocapture`.

For each table the test evaluates the Lean asserts and the Rust `ConstraintSet`
on 64 random rows and matches them as multisets of functions, does the same for
the interactions (bus id, signed multiplicity, message elements), and checks
that every Lean constraint vanishes on honest traces from `generate_*_trace`,
padding rows included. Each test asserts that all four unmatched lists are
empty: no table has a Lean-only or a Rust-only constraint or interaction left.
The sign of an interaction comes from its direction: a Rust sender is Clean's
pull and counts `−m`, a receiver is Clean's push and counts `+m`.

The models follow the multiplicity constraints of PR #1016, merged 2026-10-23.
Result on `main` after that merge, 2026-10-28:

- EQ: 5 constraints and 6 interactions, exact match.
- SHIFT: 22 constraints and 17 interactions, exact match.
- LT: 6 constraints and 10 interactions, exact match, with `μ` a byte as in
  `lt.rs`.
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
`LambdaVM/Channels/Counted.lean` is the generalisation:

- `counted N channel` grants the guarantee to a pull of `1, …, N` and charges
  every other nonzero interaction, mirroring `Channel.toRaw`. `pullIf`/`emit`
  are the circuit operations, with `ExplicitCircuits` instances so elaboration
  works.
- `exists_push_of_pull` and `consistent` are Clean's lemmas for this channel.
  Their balance premise is `N · length < char F`. Clean's `BalancedInteractions`
  carries no length bound of its own: the bound is the `fits` of the `Balance`
  the ensemble is checked against, and reaches soundness as the
  `InteractionsFit` premise of `TableSoundness`. For `Balance.logUp`, `fits` is
  `count < char F`.
- `not_consistent` proves the premise cannot be weakened: for `2 ≤ N ≤ 255`
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

The model takes both from the `counted-pulls` branch of
[yetanotherco/clean](https://github.com/yetanotherco/clean/tree/counted-pulls),
Clean main `f3abad7c` plus three commits: `Balance.logUpUpTo N` with
`RawChannel.ConsistentUpTo N`, the typed `Channel.counted N`, and
`Ensemble.tableSoundness_of_staged` / `_stagedOn` (soundness by finishing
order of channels, and of message classes within a channel). On top of it:

- `Tables/Lt.lean` pulls `IS_HALF` and `MSB16` with multiplicity `μ`, bounded by
  its own pull of `ARE_BYTES[μ, 0]`, exactly as `lt.rs`.
- `Tables/Bytewise.lean` pulls `BYTE_ALU` eight times with multiplicity `μ`,
  bounded by its own `IS_BYTE[μ]` constraint.
- `Tables/Mul.lean` pulls `IS_HALF` sixteen times and `IS_B20` four times with
  `μ_lo + μ_hi`, bounded by its own `ARE_BYTES[μ_lo, μ_hi]`. This is the chip
  that fixes `LambdaVM.muMax` at `2 * 255`: the bound is shared, so LT and
  BYTEWISE pull through the same channel even though their own `μ` is a byte.
- `Ensembles/LtStaged.lean` proves `tableSoundness`, that is
  `Ensemble.TableSoundness (Balance.logUpUpTo 255)` for the LT ensemble,
  finishing `ARE_BYTES` before the counted `IS_HALF` and `MSB16` and those
  before ALU.

A chip that pulls and pushes the *same* bus — the shape of DVRM — has no model
here. A toy MIN table and `Ensembles/CountedSameBus.lean` carried that shape
while counted pulls were being designed, and were removed once BYTEWISE
exercised the mechanism on a real chip. What they used is in Clean and
unchanged: `Ensemble.tableSoundness_of_stagedOn` finishes one bus in two classes
of messages, which is how a puller and pusher of the same bus is ordered.

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

`EqLookup`, in this same PR, is the contrast. Its `pushStatic` emits the
entries of a real `StaticLookupChannel` and proves the guarantee from
membership in the table, so `eq_rows_correct` discharges `witness.Assumptions`
outright — "no component assumes anything" — and states a conclusion about
rows: every EQ row with `μ = 1` and range-checked operands carries the
equality bit. There is no `lt_rows_correct`. Writing one needs `pushStatic`,
or a counted analogue of it, behind LT's three range checks, and this branch
has neither. What `LtStaged` does establish is that the staged finishing order
admits counted pulls, which is what it was written for.

`Channels/Counted.lean` above predates the fork. Its `not_consistent` is the
reason the fork's balance carries the `N * count` bound; its `consistentUpTo`
shows the counted channel is consistent under the fork's `BalancedUpTo N`.
Nothing else uses it.

## Known differences from the Rust prover

- Every chip whose `μ` the spec types as `Byte` needs the counted treatment LT
  got above. BYTEWISE has it; MUL, DVRM and BRANCH are not modelled yet.
- `bytewise.toml` is one of the files where the `spec/` copy in this repository
  and `spec/main` disagree in substance, not only in wording: here `μ` is still
  typed `BaseField` with no range check, while the branch types it `Byte` and
  adds `IS_BYTE[μ]`. The model follows the branch, because an unbounded `μ` on
  the pull side admits no sound bound for counted pulls at all.
- The `spec/` copy in this repository lags the `spec/main` branch; the spec the
  models follow is the branch (EQ's `μ` is `Bit` with `IS_BIT[μ]` there).

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
- A conditional interaction (`pullIf cond`) is only lawful when the same circuit
  constrains `cond` to be boolean: Clean requires every pull to carry
  multiplicity `0` or `1`, and a pull outside `{0, 1}` is a requirement the
  circuit cannot discharge at lawfulness.
