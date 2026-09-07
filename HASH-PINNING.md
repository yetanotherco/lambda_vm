# HASH PINNING — `per-table-gpu` under RPX256

The integration branch's BLOCK PATH is pinned to RPX256. The workspace default
stays BLAKE3, with its `const` assertions intact; nothing in `crypto/stark`
moves. This file records what the pin is, what enforces it, and what a person
running a box under it must not get wrong.

**This branch's hash: RPX256** — Rescue-Prime eXtended (XHash12), state 12 /
rate 8 / capacity 4, a **one-cell four-felt digest** against the byte hashes'
two-cell 32-byte one. RPO's geometry with RPO's constants and a different round
schedule, `FB E FB E FB E M`: three of the seven rounds trade the ~2^63-dense
inverse S-box for a seventh power in the degree-3 EXTENSION field, and the last
round is linear. Cheaper on the host and narrower in the AIR than RPO
(`prover/src/lfm/hash.rs`, `HasherKind::Rpx`).

| pin item | value |
|---|---|
| `BlockStarkHash` | `algebraic_commit::RpxStarkHash` |
| `BlockTranscript` / `block_transcript()` | `algebraic_transcript::AlgebraicTranscript`, built `with_seed(BLOCK_HASHER, seed)` |
| `BLOCK_HASHER` | `hash::HasherKind::Rpx` |

⚖ **Provenance, stated plainly.** Miden publishes no known-answer table for
RPX — the opposite of RPO's nineteen published vectors — so this hash has a
weaker external anchor than the RPO pin had. What anchors it is the repo's own
host known-answer harness (`make test-rpx-host-kat`;
`crypto/math-cuda/tests/host_kat/rpx_host_kat.cpp` against
`prover/tests/rpx_host_kat_vectors.rs`), which checks the C reference and the
Rust permutation against each other. That is a **self-built oracle**, not an
external one, and it must be described as such. Domain separation is through
the capacity, exactly as for RPO: the three algebraic candidates share one
leaf and parent construction (`prover/src/lfm/algebraic_commit.rs`) and differ
only in the permutation.

## The pin, mechanically — `prover/src/hash_pin.rs`, and the workspace default does NOT move

`crypto/stark`'s `DefaultStarkHash` is the *workspace's* default. It names the
hash behind `Commitment`, `BatchedMerkleTree` and every blessed constant in the
repo, and a `const` assertion in `config.rs` makes re-pointing it a compile
error so those artifacts cannot drift. The pin therefore lives one layer up:
`IsStarkProver<Field, FieldExtension, PI, H: StarkHash>` is generic over the
configuration, and `prover` names the configuration explicitly at every prove
and verify call site. Collecting those names behind `hash_pin.rs` turns "which
hash does the block path use" into a property of one module.

**THREE ORTHOGONAL AXES, all named in `hash_pin.rs` and nowhere else:**

| axis | name | this pin |
|---|---|---|
| what the HOST commits under | `BlockStarkHash` | `algebraic_commit::RpxStarkHash` |
| the Fiat–Shamir transcript OBJECT | `BlockTranscript` / `block_transcript` | `algebraic_transcript::AlgebraicTranscript` |
| the `LFM_HASH` socket permutation | `BLOCK_HASHER` | `hash::HasherKind::Rpx` |

⚠ **Axis 2 is the dangerous one.** `StarkHash::Transcript` names a *digest*
configuration, which is what GRINDING computes over; the Fiat–Shamir transcript
*object* is built by the caller and handed to `multi_prove`, so the type system
does not force it to match. For the byte hashes the two coincide. For an
algebraic hash they do not, and a branch that pinned only `BlockStarkHash`
would commit under RPX while sponging Fiat–Shamir through bytes — self-consistent
between prover and verifier, and therefore **silent**.

⚠ **Axis 3 is consulted only by algebraic programs.** Under a byte hash the
emitter's Merkle work lowers to the dedicated KECCAK / `LFM_BLAKE3` chips and
emits no `Instr::Hash` at all, so the socket hasher handed to `execute` is never
consulted and a toy permutation is free and correct. The algebraic arm goes
through `compress` / `permute`, which ARE `Instr::Hash`, executed by whatever is
passed.

### `REGISTRY_HASHER` and the classification rule

`registry::build_artifacts` defaults to `REGISTRY_HASHER = HasherKind::Test`,
the permutation `LFM_REGISTRY` is blessed under, and the generator
(`compute_lfm_registry`) imports that same constant, so the blessed value and
the builder default are one definition. The hasher is part of program IDENTITY
(`HasherKind::as_tag` is folded into `lfm_program_id`), so the block path names
its hasher AT THE CALL SITE instead:

> **A program built at `WrapHash::production()` emits `Instr::Hash` and must be
> proved under `BLOCK_HASHER`. A program that pins a byte hash on its own
> builder emits none, never consults the socket, and is correct at the
> registry's blessed default under every pin.**

It is *checkable*, not a judgement — read which program the site builds. The 17
block-path sites (`wrap_tests` 8, `aggregator_tests` 7, `fri_tests` 1,
`join_tests` 1) name `BLOCK_HASHER` through `build_artifacts_with_hasher`;
`wrap_tests`' keccak-chain census site keeps the default because
`keccak_chain_program` pins keccak on its own builder. The registry-identity
suites (`machine_tests`, the chip suites) keep the default too — that is what
`registry_drift_*` compares against.

⛔ **Do not "fix" a `registry_drift_*` failure by re-blessing the registry under
the pin.** Beyond violating the table's own doctrine — *a second hasher becomes
additional ROWS, never a silent replacement* — it would move registry
identities on the BLAKE3 control, converting "control drifted → STOP and
investigate" into a self-inflicted alarm on the one measurement the comparison
turns on. `build_artifacts` was briefly made to name `BLOCK_HASHER` itself;
that fixed a real aggregator defect at the wrong scope, and every registry
identity moved.

### Enforcement in-tree

`prover/src/tests/hash_pin_enumeration.rs` scans the crate for any code line
reaching `DefaultStarkHash`, `DefaultStarkTranscript` or `HasherKind::default()`,
any `Prover::multi_prove` / `Verifier::multi_verify` call that is not the
`BlockProver::` / `BlockVerifier::` spelling, and any item taken from
`stark::config` outside the hash-agnostic allowlist (`Commitment`,
`CommitmentHash`, `StarkHash`, `DeviceTreeBackend`) — an allowlist over a
namespace, because a name list always lags one spelling behind the newest way
to denote the default. A new site fails the test and names itself. What the
gate cannot see is a site that names a hash explicitly and names the *wrong*
one; the instruments for that are `hash_pin::tests` and the differentials in
`algebraic_commit` / `algebraic_transcript`.

### The arena stride is the BUILDER's digest width

`SubProofShape::{query_words, opening_words}`, `FriShape::query_words` and
`TableVerifyShape::{opening_words, fri_words}` take the digest width as an
argument. The machine side passes `edsl::digest_words(b)` — the builder's width,
the one every emitter advances its cursor by — and the host side passes
`proof_arena::words_per_root()`, the width it serialises roots at. A shape that
read the configuration instead agreed with a builder at `WrapHash::production()`
and disagreed with any other, and the executor's arena-length check is strict:
a program declaring a roots arena at a literal two words per digest is an
`ArenaLenMismatch` under this pin, not a slow path.

## `cuda` on an algebraic pin — COMPILES, and still cannot prove under the wrong hash

`--features cuda` builds on this branch. The algebraic backends are
`DeviceTreeBackend`s carrying their own `CommitmentHash` as the device dispatch
key, so the type system pairs a device tree with the permutation it was named
for and cannot produce a keccak tree *labelled* RPX. `math-cuda` has no kernel
for the RPX permutation wired into its dispatch yet, so a GPU run under this pin
aborts at its first device commit with `unimplemented!` naming the hash — loud,
at launch, naming the cause.

⛔ **Neither a `compile_error!` nor a byte-hash fallback belongs here.** The
first hides the cuda lint arm from the branch, which is how a dispatch
regression would reach main unseen; the second is exactly the silent wrong-hash
build this pin exists to make impossible. A build that aborts is safe; a build
that quietly proves under the wrong hash is not.

**Lint standard on this branch: BOTH passes gate.** `make lint`'s cuda
combination is a real signal here, unlike on the pre-dispatch `hash-rpo` cut
where it was expected to fail.

**Consequence for box work:** proving a block under this pin means CPU until the
RPX kernels land in the dispatch. GPU boxes remain useful for the byte-hash lanes
only.

## ⚠ TWO REGENERATIONS — a pin change is not complete without both

Every root blessed under BLAKE3 has to be regenerated, and there are two
families of them. This is why the pin PR is large and mostly generated tables —
that is EXPECTED, not a mistake.

1. **`LFM_REGISTRY`** — the hasher and the commitment hash are both folded into
   every `program_id`. `cargo run --bin compute_lfm_registry --release`. Per
   entry the `roots`, `program_id` and `prep_root` move; `log_heights`,
   `prep_widths`, `chip_set` and `keccak_rnd_chunks` are shape and must not.
2. **The static preprocessed commitments** — FOUR families, not three:
   `bitwise`, `keccak_rc`, and `page`'s zero-page AND private-page (OFFSET-only)
   constants, each at blowup 2/4/8. Each returns a BLESSED CONSTANT from
   `preprocessed_commitment` rather than recomputing, so under a new pin the
   prover recomputes an RPX root, compares it against a BLAKE3 constant and
   fails with `ProvingError::PrecomputedCommitmentMismatch`.
   `cargo run --bin compute_static_commitments --release`.

That failure is the **trial flip** this pin's PR performs on purpose: flip the
four `hash_pin.rs` lines without regenerating, run the crate's own prove/verify
legs, and expect the mismatch — loud, at prove time, naming the cause. A trial
flip that is GREEN before regeneration means the static-commitment path was not
exercised; treat that as a coverage hole, not as good news.

★ **Regenerate control-first.** Run each regenerator under the outgoing pin and
confirm it reproduces the existing table byte for byte (rustfmt's trailing
commas are the only expected textual difference) BEFORE trusting it on RPX.

⛔ **AND THAT IS ALL THE CONTROL PROVES.** `compute_lfm_registry` names
`REGISTRY_HASHER` explicitly and never reads `build_artifacts`, so re-running it
validates the generator **against itself**. When `build_artifacts` was briefly
changed to name `BLOCK_HASHER`, every registry `program_id` moved and this
control reproduced byte-for-byte anyway — it could not have fired. **The check
that fires is `machine_tests::registry_drift_*`**, because it recomputes from
the changed path and compares against the blessed table. A self-consistency
check and an independent check are not substitutes, and quoting the first for a
claim only the second can support is how a green number gets trusted for
something it never examined. A drift failure is investigated, never re-blessed
to silence the test, and neither table is ever hand-edited.

## ⚠ THE ONE WIDTH DEFECT THAT COULD HAVE PASSED

Every digest-width defect on the algebraic migration failed loudly, and there
is a reason rather than luck: the machine reconstructs a root matching nothing,
and nothing downstream can proceed. **One shape sidestepped reconstruction
entirely.** `fri_tests`' leaf gate published its digest as two cells and
compared them pairwise. An algebraic `WrapDigest` is ONE cell whose second slot
**repeats the first** (`WrapDigest::from_cell`), so the comparison read one lane
twice and would have **passed on a duplicated value** — a green test asserting
nothing, in the one place whose entire claim is that the machine's leaf IS the
verifier's leaf. It now publishes the digest's own cells. Re-audit any new
comparison that could pass on a repeated cell; `edsl::keccak256` and the BLAKE3
chain return `[Cell; 2]` because those digests genuinely are two cells.

## RUNNING UNDER THIS PIN

- **Every proving gate is a CPU run** (see the cuda section). The gates the pin
  PR ran: `machine_tests::registry_drift_*` unchanged, `hash_pin`,
  `tests::hash_pin_enumeration`, `fri_tests::the_fri_leg_proves_and_verifies`,
  `join_tests::the_join_proves_and_verifies`,
  `wrap_tests::the_fixture_epoch_wraps`, the four grinding differentials in
  `algebraic_commit.rs`, then the full `--lib` suite and `make lint` on both
  arms.
- **`P3_ARTIFACT_DIR` must be a FRESH directory for any block run.** The block
  driver *loads* cached artifacts when it finds them, so a directory carrying a
  BLAKE3 run's bundle and wraps would feed byte-hash proofs to an RPX verifier.
  A fresh directory still persists artifacts, so an aggregation OOM does not
  cost the first hour again.
- The fixture cache is separate and IS keyed on the pin:
  `proof_fixture::cache_format_key()` reads `BLOCK_COMMITMENT_HASH`, so this pin
  gets its own blob for free.
- **Proof BYTES do not reproduce run to run** (grinding draws a nonce
  non-deterministically); roots do. Never `sha256`-compare proofs.
- **Numbers:** memory and time are separate verdicts on separate lines, block
  level only. The BLAKE3 batched record the hash comparison is measured against
  (`hash-blake3`, `HASH-PINNING.md` there: 104.2 min wall, 358.2 GiB peak RSS,
  36.9 MB block proof) is the control, and this branch's per-table prover is a
  different aggregator from the one that set it — a per-table number and a
  batched number are not comparable, and no projected RPX line is carried over
  from the RPO pin.

Poseidon is **UNSHIPPABLE** (broken family, eprint 2026/306 and 2026/1692) and
remains a reference column only; XHash8 is flagged and **not adopted**.
