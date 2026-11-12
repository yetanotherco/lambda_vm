# Lambda VM formal model (Clean / Lean 4)

Lean 4 project that models the Lambda VM arithmetization in
[Clean](https://github.com/Verified-zkEVM/clean) and proves it against `spec/`.

What it covers: the AIR tables, the LogUp buses (Clean "channels") and the
semantic specification of each table. What it does not cover: the STARK /
FRI / WHIR prover, the LogUp argument itself, the executor, GPU code.

## Layout

- `LambdaVM/Field.lean` — Goldilocks as the base field (`F p`), with the size
  facts Clean's gadgets ask for. Primality comes from CompPoly's Pratt certificate.
- `LambdaVM/Templates/` — the constraint templates of the spec as helpers
  (`isBit`, `isByte`, `areBytes`, `add`, `sub`), each with a `FormalAssertion`
  whose `Spec` is the template's meaning taken from `spec/src/*.toml`:
  `IsBit.circuit`, `IsByteCheck.circuit`, `AreBytesCheck.circuit`, `Add.circuit`.
  The chips call `isBit`, `add` and `sub`; no chip calls `isByte` or `areBytes`
  yet (SHIFT pulls `ARE_BYTES` directly).
- `LambdaVM/AluOp.lean` — the `alu_op` descriptors of `types.rs`, one enumeration
  for the `opsel` of BYTE_ALU and the low bits of the ALU flag byte.
- `LambdaVM/StaticLookup.lean` — `StaticLookupChannel`: a table the verifier
  knows, given as a list of rows, with `guarantees_iff` relating the list to the
  semantic reading of a row, and `toChannel` for the components that read it.
  Clean main replaced its own `StaticLookupChannel` by `Air.Flat.Lookup`, whose
  rows are witness programs of the row index, so the structure lives here.
- `LambdaVM/Buses/` — the buses that are not static tables. `Alu.lean` is the
  unified ALU bus: message layout as in `BusId::Alu` and one `Guarantees` clause
  per `opsel` (LT, EQ and SHIFT), each taking as hypotheses the `IS_WORD`
  assumptions of its chip's spec (LT: the low words; EQ: all four words; SHIFT:
  none). The flag byte is read as a natural, so a chip
  pushing with one `opsel` discharges the other clauses by `omega`.
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
  honest row satisfies everything (completeness), for rows with `μ ∈ {0, 1}`
  only (see "Known differences"). `Eq.lean` is the EQ chip in the
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
  about active rows once a component pulls from ALU.
- `LambdaVM/Tables/ShiftCore.lean` and `Shift.lean` — the SHIFT chip. `ShiftCore`
  is the two-phase shift argument of `spec/src/shift.toml` on naturals (HWSL
  per limb, then whole limbs, left and right, logical and arithmetic), ending in
  `ShiftCore.main`, which takes exactly the facts the row gives (decoded
  `bit_shift`, one-hot `limb_shift`, sign bit, HWSL outputs or their `zbs`
  overrides) and concludes `shiftResult`. `Shift.lean` mirrors the 22 constraints
  and 17 interactions of `shift.rs` and proves soundness and completeness, the
  latter for both the honest active row and the padding row.

## Lessons for big chips (from SHIFT)

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

## Differential check against the Rust prover

The Lean circuits are hand transcriptions of `prover/src/tables/*.rs`. The check
that they are the same polynomials and the same bus interactions is mechanical:

1. `lake env lean --run LambdaVM/Export.lean` writes `export/{lt,eq,shift}.json`:
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
the interactions (bus id, signed multiplicity, message elements), and checks
that every Lean constraint vanishes on honest traces from `generate_*_trace`,
padding rows included; the Lean-only constraints of the documented deltas are
skipped there. The sign of an interaction comes from its direction: a Rust
sender is Clean's pull and counts `−m`, a receiver is Clean's push and counts
`+m`. For BITWISE the test checks the table and channel sizes, compares the
sampled rows with `generate_bitwise_row` and each channel's messages with the
receivers of `bitwise::bus_interactions()` on the same bus, and checks all 2^20
rows of `generate_bitwise_row` against a Rust transcription of `Bitwise.row`
that the sampled rows pin to the Lean export.

The models follow the multiplicity constraints of PR #1016, merged 2026-10-23.
Result on `main` after that merge, 2026-10-28:

- EQ: 5 constraints and 6 interactions, exact match.
- SHIFT: 22 constraints and 17 interactions, exact match.
- LT: exact match except the two documented deltas: `μ·(1 − μ) = 0` exists
  only in Lean, `ARE_BYTES[μ, 0]` with multiplicity one exists only in Rust.
  The test pins both by content, and its trace has deduplicated rows (`μ ≥ 2`),
  where the Lean-only constraint fails as expected.
- BITWISE: same size, all 2^20 rows, 656 sampled rows and 8 channels (10 Rust
  receivers), exact match.

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

What upstream Clean needs for an end-to-end LT theorem with byte `μ`:

1. A `Balance` for counted pulls: `BalancedInteractions` matching a pull of
   `1, …, N`, and a `fits` that bounds `N * count` rather than `count` by the
   characteristic.
2. Either a dependency-aware `OrderedChannel` (a table may hold guarantees and
   requirements on the same channel when the requirements depend only on
   channels finished earlier), or the range check of `μ` as a static `lookup`,
   which is a local constraint and makes the requirement vacuous at lawfulness.

Until then LT keeps `μ ∈ {0, 1}` (see below).

## Known differences from the Rust prover

- `LT` constrains `μ·(1 − μ) = 0`. The spec types `μ` as `Byte` with
  `IS_BYTE[μ]` and `lt.rs` sends `ARE_BYTES[μ, 0]`, because LT deduplicates
  rows and `μ` is a count. Clean only grants a channel guarantee to a pull with
  multiplicity exactly 1, so until the counted pulls above land upstream the
  model keeps `μ` a bit. Every chip whose `μ` the spec types as `Byte` (MUL,
  DVRM, BRANCH, BYTEWISE) will need the same treatment.
- `isByte` sends `ARE_BYTES[X, 0]`, as every single-byte check in the prover
  does; `spec/src/is_byte.toml` writes `[0, X]`.
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
- A conditional interaction (`pullIf cond`) is only lawful when the same circuit
  constrains `cond` to be boolean: Clean requires every pull to carry
  multiplicity `0` or `1`, and a pull outside `{0, 1}` is a requirement the
  circuit cannot discharge at lawfulness.
