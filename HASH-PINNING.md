# HASH PINNING — `hash-rpo`

One of four sibling branches (`hash-blake3`, `hash-poseidon`, `hash-rpo`,
`hash-rpx`), all cut from the same shared base so the comparison is
like-for-like. `blake3-full-mmcs` remains the campaign trunk and is not one of
these four.

**Shared base: `046da129`** — frozen. All four branches are cut from it, so the
comparison is like-for-like. ⚠ This supersedes the earlier `73ee2a64` and
`25855f70` cuts; a comparison across different bases is not a comparison.

**This branch's hash: RPO256** — Rescue-Prime Optimized, state 12 / rate 8 /
capacity 4, a **one-cell four-felt digest** against the byte hashes' two-cell
32-byte one.

| pin item | value |
|---|---|
| `BlockStarkHash` | `algebraic_commit::RpoStarkHash` |
| `BlockTranscript` | `algebraic_transcript::AlgebraicTranscript` |
| `BLOCK_HASHER` | `hash::HasherKind::Rpo` |

⚖ Security status: unbroken since 2019. Domain separation ✓ through the
capacity, AIR-enforced. External anchoring ✓ — nineteen published miden
known-answer vectors, the strongest provenance of the three algebraic
candidates.

## ★ What changed since the first cut of this file's BLAKE3 sibling

`hash-blake3`'s recipe carried a warning that read, in full: *"the other three
branches CANNOT run a block at this base"* — there was no `impl StarkHash` for
any algebraic member, `transcript_replay.rs` had no algebraic arm, and
`edsl::algebraic_byte_hash_unreachable()` was a live interim panic. **All three
are closed.** This branch runs.

## The pin, mechanically — `prover/src/hash_pin.rs`, and the workspace default does NOT move

⚠ **This is the structural difference from `hash-blake3`.** That branch's pin is
`crypto/stark`'s `DefaultStarkHash` alias, which is the *workspace's* default and
is guarded by a `const` assertion at `crypto/stark/src/config.rs:456`. Re-pointing
it would invalidate every blessed constant in the repo.

So the algebraic branches pin the **block path** instead, leaving the workspace
default at BLAKE3 and its assertion intact. ✓ VERIFIED that is expressible:
`IsStarkProver<Field, FieldExtension, PI, H: StarkHash>` is generic over the
configuration, and `prover` was already naming `DefaultStarkHash` *explicitly* at
each of its prove and verify call sites — those are type parameters, not a global.

**THREE ORTHOGONAL AXES, all named in `hash_pin.rs` and nowhere else:**

| axis | name | this branch |
|---|---|---|
| what the HOST commits under | `BlockStarkHash` | `algebraic_commit::RpoStarkHash` |
| the Fiat–Shamir transcript OBJECT | `BlockTranscript` / `block_transcript` | `algebraic_transcript::AlgebraicTranscript` |
| the `LFM_HASH` socket permutation | `BLOCK_HASHER` | `hash::HasherKind::Rpo` |

⚠ **Axis 2 is the dangerous one.** `StarkHash::Transcript` names a *digest*
configuration, which is what GRINDING computes over; the Fiat–Shamir transcript
*object* is built by the caller and handed to `multi_prove`, so the type system
does not force it to match. For the byte hashes the two coincide. For an
algebraic hash they do not, and a branch that pinned only `BlockStarkHash` would
commit under RPO while sponging Fiat–Shamir through bytes — self-consistent
between prover and verifier, and therefore **silent**.

⚠ **Axis 3 stayed unpinned until it bit.** Under a byte hash the emitter's Merkle
work lowers to the dedicated KECCAK / `LFM_BLAKE3` chips and emits no
`Instr::Hash` at all, so the socket hasher handed to `execute` is never consulted
and a toy permutation is free and correct. The algebraic arm goes through
`compress` / `permute`, which ARE `Instr::Hash`. `registry::build_artifacts` was
defaulting it to `HasherKind::Test`, a one-round toy — and `build_artifacts` is
what the **P3 block driver itself calls**, so on this branch the defect sat
directly in the path that produces block numbers. Fixed at `13453ef9`.

Enforcement in-tree: `prover/src/tests/hash_pin_enumeration.rs` scans the crate
for any call site reaching `DefaultStarkHash`, `DefaultStarkTranscript` or
`HasherKind::default()` outside the pin, against a blessed inventory with a
reason per entry. A new site fails the test and names itself.

## ⛔ CUDA — expected to FAIL TO COMPILE, deliberately

`--features cuda` on this branch hits a `compile_error!` in `hash_pin.rs`. The
cuda batch commitment path is written against `KeccakTreeBackend`, a byte-hash
trait the algebraic backend does not implement and could not without a device
kernel for the permutation, which does not exist.

★ **A cuda fallback to a byte hash was considered and REJECTED.** It would give a
build that compiles, runs, and proves under a different hash than the branch is
named for. A build that does not exist is safe; one that quietly proves under the
wrong hash is not.

**Lint standard on this branch: the non-cuda pass gates; the cuda pass is
expected to fail to compile.** `make lint`'s cuda combination is not a
regression here.

## ⚠ TWO REGENERATIONS — a pin change is not complete without both

Every root blessed under BLAKE3 has to be regenerated, and there are two
families. This is why an algebraic branch's pin commit is large and mostly
generated tables — that is EXPECTED, not a mistake.

1. **`LFM_REGISTRY`** — the hasher is folded into every `program_id`.
   `cargo run --bin compute_lfm_registry --release`
2. **The static preprocessed commitments** — `bitwise`, `keccak_rc` and `page`
   each return a BLESSED CONSTANT from `preprocessed_commitment` rather than
   recomputing. Under a new pin the prover recomputes an RPO root, compares it
   against a BLAKE3 constant and fails with
   `ProvingError::PrecomputedCommitmentMismatch`.
   `cargo run --bin compute_static_commitments --release`

✓ VERIFIED (2) empirically — it is exactly how the trial flip failed, and it is
the correct failure: loud, at prove time, naming the cause.

★ **Regenerate control-first.** Run each regenerator under the BLAKE3 pin and
confirm it reproduces the existing table byte for byte BEFORE trusting it on RPO.

✓ **VERIFIED on this cut, in the other direction too.** Both regenerators were
re-run at the frozen base and their output compared against the tables carried
forward from the previous cut: the four static-commitment families match, and the
registry matches on **all 13,056 hex bytes** — the only textual difference is
rustfmt's trailing commas. So the regeneration is deterministic, and the base's
`programs.rs` change (the R1f Merkle instrument) moved no registered program
identity, exactly as the six-kind `LfmProgramKind` list predicts.
A regenerator that silently produced the wrong table would bless the wrong roots
and every drift test would agree with it. `registry.rs` governs both: a drift
failure is investigated, never re-blessed to silence the test, and neither table
is ever hand-edited.

## THE RECIPE — how to run this branch's block

**Test:** `lfm::aggregator_tests::the_real_block_aggregates_end_to_end`

```sh
export LFM_CENSUS_ELF=/root/ethrex.elf
export LFM_CENSUS_INPUT=/root/ethrex_mainnet_25368371.bin   # 1,110,156 B, sha 61eba49b
export LFM_CENSUS_EPOCH_LOG2=24
export LAMBDA_VM_MAX_ROWS_LOG2=24
export P3_AGG_TERMINAL_BLOWUP=2
export P3_AGG_RESIDENCY=recompute
export P3_ARTIFACT_DIR=/root/p3_rpo_pin_artifacts          # ★ FRESH — see below

cargo test --release -p lambda-vm-prover --lib \
  lfm::aggregator_tests::the_real_block_aggregates_end_to_end \
  -- --ignored --exact --nocapture --test-threads=1
```

★ **`P3_ARTIFACT_DIR` must be a FRESH directory, and on this branch that is not
merely methodology — it is correctness.** The driver *loads* cached artifacts
when it finds them, and any directory carrying a BLAKE3 run's bundle and wraps
would feed byte-hash proofs to an RPO verifier. A fresh directory still persists
artifacts, so an aggregation OOM does not cost the first hour again.

⚠ The fixture cache is separate and IS keyed on the pin —
`proof_fixture::cache_format_key()` reads `BLOCK_COMMITMENT_HASH`, so the
algebraic branches get their own blob for free. See RPO-LANE.md's follow-up on
what that key still does not cover.

The two `P3_AGG_*` values are the BLAKE3 record's, kept so the comparison is
like-for-like, and both are load-bearing:
- `P3_AGG_TERMINAL_BLOWUP=2` — blowup 4 OOM-killed the box three straight times.
  The query count re-derives from the same Johnson target by construction
  (`with_blowup`), so 2 → 219 q; the FRI terminal stays at the preset's fp8.
- `P3_AGG_RESIDENCY=recompute` — trades ~2× aggregation prove time for the LDE
  peak. `Retain` at real scale OOM-killed a 483 GiB box.

⚠ Idle guard first. The box's own noise study puts CV at 0.06–0.25% idle but ≥2%
on a single contended pair, and this is a one-shot measurement.

## The bar, and what is compared

| | BLAKE3 record |
|---|---|
| wall | **104.2 min** |
| peak RSS | **358.2 GiB** |
| block proof | **36.9 MB** (38,675,976 bytes) |

★ Separate lines deliberately: time and memory are separate verdicts and are
never averaged into one.

⚠ **The comparison is those three numbers plus a passing verify and consumer
ritual — NOT proof bytes.** Grinding draws a nonce non-deterministically, so the
proof BYTES do not reproduce run to run even at a fixed commit; the roots do.

⚠ **The four branches are NOT four equally-grounded numbers.** BLAKE3 has a
measured block record; the other three have projections from measured chips and a
measured census. Keep marking which cells are measured and which are
extrapolated. Projected peak RSS is **affine**, not proportional, and must be
quoted with its unmeasured fixed term `C` named: **RPO 76.2 + 0.774·C** GiB.

⚠ Counterweight, measured: BLAKE3 is the **cheapest hash per cell in time**
(39.7 ns/cell vs RPO 46.1) — the algebraic case is a MEMORY win that reduces time
by reducing cells, not a per-cell speedup.

Poseidon is **UNSHIPPABLE** (broken family, eprint 2026/306 and 2026/1692) and
appears in the comparison only because the numbers were asked for. XHash8 is
flagged and **not adopted**.

Full working: `thoughts/shared/block-compression/RPO-LANE.md` and
`HASH-SWAP-DESIGN.md`.
