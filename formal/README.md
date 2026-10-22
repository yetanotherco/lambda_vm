# Lambda VM formal model (Clean / Lean 4)

Lean 4 project that models the Lambda VM arithmetization in
[Clean](https://github.com/Verified-zkEVM/clean) and proves it against `spec/`.

What it covers: the AIR tables, the LogUp buses (Clean "channels") and the
semantic specification of each table. What it does not cover: the STARK /
FRI / WHIR prover, the LogUp argument itself, the executor, GPU code.

## Layout

- `LambdaVM/Field.lean` — Goldilocks as the base field (`F p`), with the size
  facts Clean's gadgets ask for. Primality comes from CompPoly's Pratt certificate.
- `LambdaVM/Buses/` — the buses that are not static tables. `Alu.lean` is the
  unified ALU bus: message layout as in `BusId::Alu`, the caller's `IS_WORD`
  assumptions on both operands as hypotheses, and one `Guarantees` clause per
  `opsel` (LT and EQ so far). A chip pushing with one `opsel` discharges the
  other clauses by showing its flags cannot decode as them.
- `LambdaVM/Tables/` — one file per chip. `Bitwise.lean` transcribes the
  precomputed row function of `prover/src/tables/bitwise.rs` and defines each bus
  the table serves as a `StaticLookupChannel`, with the semantic `Guarantees`
  proven equivalent to table membership. `Lt.lean` is the LT chip: the row
  circuit mirrors `lt.rs` (constraints and interactions, in order), `core` is the
  arithmetic argument of `spec/lt.typ` on naturals, `Honest` is the row
  `generate_lt_trace` produces, and the `GeneralFormalCircuit` proves that a row
  that pulls its range checks pushes a correct ALU result (soundness) and that an
  honest row satisfies everything (completeness). `Eq.lean` is the EQ chip in the
  same shape: SUB carries, four IS_HALF pulls, a ZERO pull on the sum of the
  halves, `res = eq xor invert`; its `μ` is a bit, as in PR #1016.

- `LambdaVM/Ensembles/` — tables wired together through their channels.
  `EqLookup.lean` is the first one: `pushStatic` is the provider side of any
  static lookup channel (its row is the vector of multiplicities, the role of
  the `μ` columns of BITWISE), and `eqLookup` chains the IS_HALF provider, the
  ZERO provider and the EQ chip in dependency order. `eq_rows_correct` is the
  first global theorem: for any witness whose constraints hold and whose
  channels balance, every EQ row with `μ = 1` and range-checked operands
  carries the equality bit. The lookup argument is what turns the pulled range
  checks into facts.
- `LambdaVM/Tables/ShiftCore.lean` and `Shift.lean` — the SHIFT chip. `ShiftCore`
  is the two-phase shift argument of `spec/shift.typ` on naturals (HWSL
  per limb, then whole limbs, left and right, logical and arithmetic), ending in
  `ShiftCore.main`, which takes exactly the facts the row gives (decoded
  `bit_shift`, one-hot `limb_shift`, sign bit, HWSL outputs or their `zbs`
  overrides) and concludes `shiftResult`. `Shift.lean` mirrors the 22 constraints
  and 17 interactions of `shift.rs` and proves soundness and completeness, the
  latter for both the honest active row and the padding row.

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
   multiplicity and message, variables numbered in Rust column order.
2. `diff/lean_diff.rs` is a prover test. Install it as
   `prover/src/tests/lean_diff.rs`, add `pub mod lean_diff;` to
   `prover/src/tests/mod.rs` and `serde_json = "1"` to the prover's
   dev-dependencies, then run
   `cargo test -p lambda-vm-prover lean_diff -- --nocapture`.

For each table the test evaluates the Lean asserts and the Rust `ConstraintSet`
on 64 random rows and matches them as multisets of functions, does the same for
the interactions (bus id, signed multiplicity, message elements), and checks
that every Lean constraint vanishes on honest traces from `generate_*_trace`,
padding rows included.

The models follow PR #1016 (`fix/multiplicity-soundness`), so the test is meant
to run on that branch (a git worktree of it works). Result there, 2026-10-20:

- EQ: 5 constraints and 6 interactions, exact match.
- SHIFT: 22 constraints and 17 interactions, exact match.
- LT: exact match except the two documented deltas: `μ·(1 − μ) = 0` exists
  only in Lean, `ARE_BYTES[μ, 0]` exists only in Rust.

The check compares polynomials, not zero sets: it caught `μ·(μ − 1)` versus
`μ·(1 − μ)` in EQ, which is why the Lean bit constraints use the spec's form.

## Counted pulls (`μ ∈ [0, 255]`)

After PR #1016 a deduplicated row pulls its range checks with multiplicity `-μ`,
`μ` a byte. Clean's typed channel grants the guarantee at `-1` only; any other
nonzero multiplicity is a requirement the row cannot meet.
`LambdaVM/Channels/Counted.lean` is the generalisation:

- `counted N channel` grants the guarantee at `-1, …, -N` and charges every
  other nonzero multiplicity, mirroring `Channel.toRaw`. `pullIf`/`emit` are the
  circuit operations, with `ExplicitCircuits` instances so elaboration works.
- `exists_push_of_pull` and `consistent` are Clean's lemmas for this channel.
  Their balance premise is `N · length < char F`, not Clean's `length < char F`.
- `not_consistent` proves the premise cannot be weakened: for `2 ≤ N ≤ 255`
  over Goldilocks, `p / N` pulls of `-N` plus `p % N` pulls of `-1` on a message
  outside the table balance to `-p = 0` with fewer than `p` interactions, so
  `RawChannel.Consistent` is false and the channel cannot be finished in a
  `SoundEnsemble`. With `N = 1` the channel is `Normal`, as in Clean.
- `Demo.circuit` is the shape of a dedup row: `ARE_BYTES[μ, 0]` then a counted
  `IS_HALF` pull with multiplicity `-μ`. The counted channel sits in
  `channelsWithRequirements`: an out-of-range `μ` is only ruled out by the byte
  guarantee, which lawfulness does not see but soundness does.

What upstream Clean needs for an end-to-end LT theorem with byte `μ`:

1. A per-channel multiplicity bound in `BalancedInteractions`
   (`bound * length < ringChar F`), threaded through `BalancedChannels`.
2. Either a dependency-aware `OrderedChannel` (a table may hold guarantees and
   requirements on the same channel when the requirements depend only on
   channels finished earlier), or the range check of `μ` as a static `lookup`,
   which is a local constraint and makes the requirement vacuous at lawfulness.

Until then LT keeps `μ ∈ {0, 1}` (see below).

## Known differences from the Rust prover

- `LT` constrains `μ·(1 − μ) = 0`. The spec types `μ` as `Bit`; `lt.rs` has no
  constraint on it and uses it as a deduplication count. Clean only grants a
  channel guarantee to a pull with multiplicity exactly 1, which is the
  faithful reading: a sender multiplicity outside `{0, 1}` is not protected by
  the LogUp balance (two rows with `μ = 1` and `μ = −1` cancel each other's
  lookups). The same pattern exists in MUL, DVRM, SHIFT, EQ and BYTEWISE.
- `LambdaVM/Templates/` — the constraint templates of the spec (`IS_BIT`,
  `IS_BYTE`, `ARE_BYTES`, ...), with `Spec` taken from the corresponding
  `spec/*.typ` page, not from the constraints. Templates that are only
  interactions are plain `Circuit` helpers meant to be expanded inline in a chip;
  each comes with a `*Check` assertion that pins down its meaning.

## Building

```sh
# once: Lean toolchain manager
curl https://elan.lean-lang.org/elan-init.sh -sSf | sh

cd formal
lake update            # clones Clean, Mathlib, CompPoly at the pinned revisions
lake exe cache get     # Mathlib prebuilt binaries (without this, ~1 h of compile)
lake build
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
  constrains `cond` to be boolean; multiplicities outside `{0, 1}` turn the
  channel guarantee into a proof obligation.
