# Lambda VM formal model (Clean / Lean 4)

Lean 4 project that models the Lambda VM arithmetization in
[Clean](https://github.com/Verified-zkEVM/clean) and proves it against `spec/`.

What it covers: the AIR tables, the LogUp buses (Clean "channels") and the
semantic specification of each table. What it does not cover: the STARK /
FRI / WHIR prover, the LogUp argument itself, the executor, GPU code.

## Layout

- `LambdaVM/Field.lean` — Goldilocks as the base field (`F p`), with the size
  facts Clean's gadgets ask for. Primality comes from CompPoly's Pratt certificate.
- `LambdaVM/Templates/` — the constraint templates of the spec as helpers the
  chips call (`isBit`, `isByte`, `areBytes`, `add`, `sub`), each with a
  `FormalAssertion` whose `Spec` is the template's meaning taken from its
  `spec/*.typ` page: `IsBit.circuit`, `IsByteCheck.circuit`, `Add.circuit`.
- `LambdaVM/AluOp.lean` — the `alu_op` descriptors of `types.rs`, one enumeration
  for the `opsel` of BYTE_ALU and the low bits of the ALU flag byte.
- `LambdaVM/StaticLookup.lean` — `StaticLookupChannel`: a table the verifier
  knows, given as a list of rows, with `guarantees_iff` relating the list to the
  semantic reading of a row, and `toChannel` for the components that read it.
  Clean main replaced its own `StaticLookupChannel` by `Air.Flat.Lookup`, whose
  rows are witness programs of the row index, so the structure lives here.
- `LambdaVM/Buses/` — the buses that are not static tables. `Alu.lean` is the
  unified ALU bus: message layout as in `BusId::Alu`, the caller's `IS_WORD`
  assumptions on both operands as hypotheses, and one `Guarantees` clause per
  `opsel` (LT, EQ, SHIFT and the toy MIN of the counted-pulls PoC). The flag
  byte is read as a natural, so a chip pushing with one `opsel` discharges the
  other clauses by `omega`.
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
  checks into facts. `LtStaged.lean` and `CountedSameBus.lean` are the
  counted-pulls ensembles (see below).
- `LambdaVM/Tables/ShiftCore.lean` and `Shift.lean` — the SHIFT chip. `ShiftCore`
  is the two-phase shift argument of `spec/shift.typ` on naturals (HWSL
  per limb, then whole limbs, left and right, logical and arithmetic), ending in
  `ShiftCore.main`, which takes exactly the facts the row gives (decoded
  `bit_shift`, one-hot `limb_shift`, sign bit, HWSL outputs or their `zbs`
  overrides) and concludes `shiftResult`. `Shift.lean` mirrors the 22 constraints
  and 17 interactions of `shift.rs` and proves soundness and completeness, the
  latter for both the honest active row and the padding row.
- `LambdaVM/Tables/MinToy.lean` — MIN, a toy ALU operation with the shape of
  DVRM: it pulls an LT result on the ALU bus and pushes its own result on the
  same bus. Not an RV64IM operation; it stands in for DVRM until DVRM is
  modelled.

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

1. `lake env lean --run LambdaVM/Export.lean` writes `export/{lt,eq,shift}.json`:
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

The check compares polynomials, not zero sets: it caught `μ·(μ − 1)` versus
`μ·(1 − μ)` in EQ, which is why the Lean bit constraints use the spec's form.

`export/` is a committed artifact; `.github/workflows/formal.yml` builds
`formal/`, re-runs the export and fails if it differs, on every PR that touches
`formal/`.

## Counted pulls (`μ ∈ [0, 255]`)

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

- `Tables/Lt.lean` pulls `IS_HALF` and `MSB16` with multiplicity `μ` through
  `Channel.counted 255`, bounded by its own pull of `ARE_BYTES[μ, 0]`, exactly
  as `lt.rs`.
- `Ensembles/LtStaged.lean` proves `tableSoundness`, that is
  `Ensemble.TableSoundness (Balance.logUpUpTo 255)` for the LT ensemble,
  finishing `ARE_BYTES` before the counted `IS_HALF` and `MSB16` and those
  before ALU. Its second half adds MIN, which pulls LT and pushes MIN on the
  ALU bus: ALU is finished in two stages, the non-MIN messages first and the
  MIN messages after.
- `Ensembles/CountedSameBus.lean` (CMIN) does the same with the ALU bus itself
  counted: a row answers `μ` identical MIN requests, the full shape of DVRM.

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
in the file derives that from a row of BITWISE. `CountedSameBus` reuses those
two components, so the LT answers it feeds to CMIN are assumed in the same way.

`EqLookup`, in this same PR, is the contrast. Its `pushStatic` emits the
entries of a real `StaticLookupChannel` and proves the guarantee from
membership in the table, so `eq_rows_correct` discharges `witness.Assumptions`
outright — "no component assumes anything" — and states a conclusion about
rows: every EQ row with `μ = 1` and range-checked operands carries the
equality bit. There is no `lt_rows_correct`. Writing one needs `pushStatic`,
or a counted analogue of it, behind LT's three range checks, and this branch
has neither. What `LtStaged` and `CountedSameBus` do establish is that the
staged finishing order admits counted pulls, which is what they were written
for.

`Channels/Counted.lean` above predates the fork. Its `not_consistent` is the
reason the fork's balance carries the `N * count` bound; its `consistentUpTo`
shows the counted channel is consistent under the fork's `BalancedUpTo N`.
Nothing else uses it.

## Known differences from the Rust prover

- Every chip whose `μ` the spec types as `Byte` (MUL, DVRM, BRANCH, BYTEWISE)
  needs the counted treatment LT got above; none of them is modelled yet.
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
