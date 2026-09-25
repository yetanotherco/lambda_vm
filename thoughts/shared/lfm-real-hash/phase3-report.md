# Phase 3 — bind the hasher into the program digest and the registry

**Status:** BUILT, gates green, **uncommitted** in
`/Users/maurofab/workspace/lambda_vm-blake3-impl` (branch `blake3-real-hash`).
Nothing pushed, nothing committed — the lead reviews and commits.
**Date:** 2026-08-10. Closes review finding **F3-2 / F3.3**; also closes **F3.2**
(see §7, flagged as an extra).

Claims below are marked ✓ VERIFIED (I read the code and cite `file:line`, or I
ran the thing) or ? INFERRED. Line numbers are post-change unless stated.

---

## 1. The gap, confirmed before fixing

✓ VERIFIED by reading the pre-change tree, not by trusting the brief:

- `lfm_program_id`'s preimage was tag ‖ machine version ‖ preset ‖ per-slot
  `(index, root, log_height)` ‖ chunk count — **no hasher**
  (`statement.rs:40-56`, pre-change).
- `LfmRegistryEntry` had `kind, blowup_factor, roots, log_heights,
  keccak_rnd_chunks, program_id` — **no hasher** (`registry.rs:52-60`).
- `LfmArtifacts` likewise (`registry.rs:63-68`).
- `lfm_verify` → `verify_against` → `verify_against_with_hasher(...,
  HasherKind::default())` (`proof.rs:135-181`). The registry path could not
  reach Poseidon at all.

So a Test-backed and a Poseidon-backed machine of the same program had
byte-identical roots **and** byte-identical `program_id`. The only separator was
`hash::num_columns` differing (39 vs 623), caught by the framework as a width
mismatch — a layout coincidence, not a binding.

**The one fact that makes this load-bearing rather than belt-and-braces**, which
I confirmed rather than assumed: no preprocessed root moves with the hasher.
`layout::hash::PREP_WIDTH = 11` under both candidates, so `build_artifacts`
commits identical groups either way. I proved this by measurement, not by
reading — see §5, where the regenerated table shows **all 84 root literals
bit-identical and only the 6 digests moved**. The commitments therefore *cannot*
carry the hasher; a tag in the digest is the only place it can live.

---

## 2. The diff — 12 files, +337 / −87

### Source (5 files)

| File | Change |
|---|---|
| `prover/src/lfm/hash.rs` | `HasherKind` gains `#[repr(u8)]` with written-out discriminants (`Test = 0`, `Poseidon = 1`) and `pub const fn as_tag(self) -> u8`. Doc says why the wire value must not follow declaration order. |
| `prover/src/lfm/statement.rs` | `lfm_program_id` takes `hasher: HasherKind` and folds `h.update([hasher.as_tag()])` in **after `LFM_PRESET_TAG`, before the per-slot loop** — the position §4 of the plan specifies. Everything else in the preimage is untouched and in the same order. |
| `prover/src/lfm/registry.rs` | `hasher: HasherKind` field added to **both** `LfmRegistryEntry` and `LfmArtifacts`. `build_artifacts(program, options)` now delegates to new `build_artifacts_with_hasher(program, options, hasher)`, which derives `program_id` from the hasher and stores it. Registry constants regenerated. |
| `prover/src/lfm/proof.rs` | See §3 — the verify and prove paths. |
| `prover/src/lfm/mod.rs` | Re-exports `HasherKind` and `build_artifacts_with_hasher`. |

### Generator (1 file)

`prover/src/bin/compute_lfm_registry.rs` — emits the `hasher:` line, builds via
`build_artifacts_with_hasher` under a named `REGISTRY_HASHER: HasherKind =
HasherKind::Test` constant (so the table's hasher is a stated decision, not an
implicit default), and calls `validate(program)` per program (§7).

### Tests (6 files)

`machine_tests.rs` (+129), `poseidon_chip_tests.rs`, `constraint_tests.rs`,
`fri_tests.rs`, `join_tests.rs`, `wrap_tests.rs` — the new tests (§4) plus the
33 mechanical `verify_against` call-site updates (§3).

---

## 3. `lfm_verify` reads the hasher; there is no defaulting path left

**The fix asked for** (`proof.rs:150-167`): `lfm_verify` resolves the entry and
passes `entry.hasher` into the AIR-set build. ✓ VERIFIED by reading the final
file; the `HasherKind::default()` call is gone from the verify path entirely.

**One design decision worth the lead's attention.** I **merged**
`verify_against` and `verify_against_with_hasher` into a single
`verify_against(roots, program_id, keccak_rnd_chunks, proof, claimed_public,
options, hasher)` rather than leaving the defaulting wrapper in place.
`grep verify_against_with_hasher` now returns **0** hits.

Reasoning: the brief asked that `verify_against` "also take/carry the hasher",
and adding a `hasher` parameter to the wrapper would have made it identical to
the function it wrapped. Keeping the wrapper would also have left a live hazard
that this phase creates: `artifacts.program_id` is now derived from the hasher,
so a test that switches to `build_artifacts_with_hasher(..., Poseidon)` and
calls a defaulting `verify_against` would be silently pairing one hasher's
digest with another hasher's AIR set. Passing `artifacts.hasher` explicitly
keeps the two locked together at every call site.

Cost: **33 call sites** updated, all inside `prover/src/lfm/*`
(`verify_against` was `pub` but never re-exported from `mod.rs`, so there are no
callers outside the module — ✓ VERIFIED by grep across the workspace). They were
patched by a script that reads the receiver off each call's **own** first
argument (`&NAME.roots,`) rather than assuming the binding is called
`artifacts`, and reports anything it cannot parse instead of guessing. The one
"unhandled" report was the function definition itself.

**What I deliberately did NOT change:** `verify_against` keeps its
piece-by-piece parameter list rather than taking `&LfmArtifacts`. I checked
whether the artifacts-struct signature was viable and it is not:
`wrap_tests.rs:505-512` passes a **deliberately mutated** `program_id` (`other`,
with `other[0] ^= 1`) alongside the real roots, and that falsification is the
point of the test. The loose form has to survive.

### The prove side, which the brief did not ask about but this change forces

Adding `hasher` to `LfmArtifacts` creates a new way to be wrong: artifacts built
for one hasher, proved under another, produce a proof whose statement names a
permutation the trace does not use. I closed it rather than leaving it:

- `lfm_prove` now uses `artifacts.hasher` instead of `HasherKind::default()`
  (`proof.rs:51-58`). Same for the test-only `prove_traces`.
- `lfm_prove_with_hasher` **asserts** `artifacts.hasher == hasher`
  (`proof.rs:83-88`), documented under a `# Panics` section. It is a caller bug,
  not a proof outcome, so it panics rather than returning `Err`.

This is why `poseidon_chip_tests.rs` needed real changes and not just a rename:
two of its tests previously built default (Test) artifacts and proved under
Poseidon. They now build hasher-matched artifacts.

---

## 4. The soundness property, and the test that proves it

**The property:** two programs identical except for `HasherKind` now have
**distinct** `program_id`s.

### The test the lead asked for

`poseidon_chip_tests::the_hasher_choice_moves_the_program_digest_and_no_root`
✓ VERIFIED PASSING. It replaces the pre-change test
`the_hasher_choice_does_not_move_any_program_digest`, whose name asserted the
exact property this phase inverts. (That old test in fact only checked
`build_artifacts` determinism — it called the same no-hasher function twice — so
its name and doc comment had been describing something it did not test. Worth
noting as a stale-doc finding in its own right.)

For both `trivial_program` and `fri_toy_program`, over Test vs Poseidon:

- `assert_eq!(test.roots, pos.roots)` — no preprocessed root moves;
- `assert_eq!` on `log_heights` and `keccak_rnd_chunks` — nothing else moves;
- `assert_ne!(test.program_id, pos.program_id)` — **the digest moves.**

The first assertion is what makes the third meaningful: with every other input
to `lfm_program_id` held bit-identical, the inequality can only come from the
tag.

### Three more tests, all ✓ VERIFIED PASSING

- `machine_tests::every_registry_entry_binds_its_hasher_into_its_digest` — for
  each of the six entries: the stored `program_id` **is** what the stored
  `(roots, log_heights, chunks, hasher)` derive (honest control, the table is
  self-consistent), and recomputing with any *other* `HasherKind` gives a
  different digest (the property, at registry level).
- `machine_tests::the_registry_hasher_is_what_verify_builds` — the honest-path
  control the standing rule requires. An honest `TrivialV0` proof **verifies**
  through `lfm_verify` (which now builds from `entry.hasher`), and the same
  proof against the same entry's roots and digest under any other hasher
  **rejects**. The accept half is not decoration: a fix that rejected everything
  would pass the reject half on its own.
- `poseidon_chip_tests::the_hasher_tags_are_stable_and_distinct` — pins
  `Test.as_tag() == 0`, `Poseidon.as_tag() == 1`, `default() == Test`. The tag
  is the mechanism, so it is pinned directly and not only through a digest.

A `const ALL_HASHERS` in `machine_tests.rs` lists every variant by hand, so
adding BLAKE3 in a later phase forces a deliberate edit here rather than
silently narrowing the coverage.

The six `registry_drift_*` tests also gained
`assert_eq!(entry.hasher, artifacts.hasher, "hasher drifted")`.

---

## 5. Registry regeneration — done, and it moved exactly what it should

`cargo run --bin compute_lfm_registry --release` ran clean (exit 0). Not
blocked; it needs no ELF or fixture. Output spliced into `registry.rs`,
`cargo fmt` applied.

**All six `program_id`s moved. No root moved.** ✓ VERIFIED mechanically, not by
eye: I extracted every 32-byte literal from the table before and after (90 per
version = 6 entries × (14 roots + 1 digest)) and compared. Exactly six differ,
at indices 14, 29, 44, 59, 74, 89 — the 15th literal of each entry, i.e. the
`program_id`, and nothing else.

This is the deliberate re-blessing §4 of the plan calls for, and it needs
calling out in the PR body. New digests (first 4 bytes):

| kind | new `program_id` |
|---|---|
| `TrivialV0` | `9f 05 37 f5 …` |
| `FriToyV0` | `3b 4e 71 8c …` |
| `KeccakChainV0` | `eb 59 1d e1 …` |
| `KeccakSpongeV0` | `1d 90 d7 b5 …` |
| `TranscriptReplayV0` | `26 03 3a 9e …` |
| `StatementReplayV0` | `af 84 2f d9 …` |

All six drift tests recompute and match ✓ VERIFIED PASSING — the table is
self-consistent.

---

## 6. Gates

| Gate | Result |
|---|---|
| `cargo build -p lambda-vm-prover` | ✓ clean |
| `cargo check -p lambda-vm-prover --tests` | ✓ clean |
| `cargo check -p lambda-vm-prover --bin compute_lfm_registry` | ✓ clean |
| `cargo test --release --lib -- registry_drift is_admissible hasher registered_programs` | ✓ **20 passed, 0 failed** |
| `cargo test --release --lib -- wrap_tests constraint_tests fri_tests join_tests poseidon_chip_tests` | ✓ **56 passed, 0 failed**, 4 ignored |
| `make fmt` | ✓ clean |
| `make lint` | ✓ clean — all four clippy feature combos under `-D warnings` |

`make lint` and `make fmt` were both run from the worktree root; neither was
skipped.

### The 19 pre-existing failures, checked rather than assumed

`cargo test --release -p lambda-vm-prover --lib lfm::` reports **233 passed, 19
failed, 7 ignored**. I did **not** assume those 19 were pre-existing. I stashed
the entire change (`git stash push -- prover/`, after saving a backup patch),
re-ran the same 19 on the pristine tree, and got the **identical failing set**;
then popped the stash and confirmed the diff restored intact.

They are `lfm::epoch_tests` (7), `lfm::epoch_verify_tests` (6),
`lfm::logup_tests::a_zero_row_fixed_table_carries_some_zero_not_none`,
`arena_filler_reads_real_committed_roots`,
`continuation_fixture_generates_two_epochs`, and three `l2g_binding*` tests.
The visible cause is `Exec(ArenaLenMismatch { arena: 0, expected: 4, found: 2 })`
— fixture-shaped, unrelated to hashing. ✓ VERIFIED pre-existing on
`blake3-real-hash` head `ef13e746`.

---

## 7. One thing I did beyond the four IMPLEMENT items — flag for the lead

Plan §4 step 6 says to take **F3.2** while in the same file, and I did:
`compute_lfm_registry` now calls `validate(program)` per program before building
artifacts, so the admission gate `validator.rs` declares is mechanically wired
into registry generation instead of resting on the convention that every
registered kind also has a hand-written admissibility test.

It is 3 lines in one file, and it passed for all six programs on the first run
(the generator would have panicked otherwise) — so this is confirmation, not a
change in what is admitted. It is **not** in the lead's four-item IMPLEMENT
list, so drop it if you want the phase kept to exactly that scope.

---

## 8. State and what is next

- Worktree `/Users/maurofab/workspace/lambda_vm-blake3-impl`, branch
  `blake3-real-hash`, **12 files modified, uncommitted, unpushed.**
- Backup of the diff: `phase3.patch` in this session's scratchpad (insurance for
  the stash cycle in §6; the working tree is authoritative).
- `HasherKind` still has exactly two variants. No BLAKE3 arm was added — that is
  a later phase, and the binding is hasher-generic so it does not need one.
- The ordering constraint in the plan is now satisfied: the tag is in place
  **before** a third candidate exists, so BLAKE3 cannot land on a colliding
  identity. The next hasher needs: a variant with the next unused discriminant,
  an `ALL_HASHERS` entry in `machine_tests.rs`, and a registry row — the digest
  binding itself needs no further work.
