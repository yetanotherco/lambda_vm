# Batched-MMCS primitives port (M-1 / M-2) — resume note

**Branch** `mmcs-primitives` (worktree `/Users/maurofab/workspace/lambda_vm-mmcs`), based on
`blake3-real-hash` @ `3a0b8485`. Signed, **unpushed**, working tree clean. This is a
complete milestone, not a checkpoint of half-done work.

> **Why this file is not `RESUME.md`.** The worktree root already holds `RESUME.md`, the
> **RATE-4 lane's** resume note, inherited from the base branch (`38c89d86`). The lead's
> standing order said "write RESUME.md at your worktree root"; taken literally that
> clobbers a sibling lane's handoff the moment `mmcs-primitives` is merged into
> `blake3-real-hash` (which is the plan — see the campaign's task list). Renaming costs
> nothing and loses nothing. **Do not "fix" this by overwriting `RESUME.md`.**

Source of the port: `origin/feat/batched-fri-per-epoch` (PR #768), files
`crypto/stark/src/fri/mmcs.rs` (~1,015 lines), `crypto/stark/src/fri/batched.rs` (~499),
`crypto/stark/src/tests/bus_tests/batched_soundness_tests.rs` (~237). Scoping document:
`thoughts/shared/block-compression/MMCS-PLAN.md` (§2 the port verdict, §3.3 the streaming
constraint, §3.6 the item table, §M-10-RESULT the index-convention analysis).

| commit | what |
|---|---|
| `13aac0fe` | `feat(fri)`: the two primitives, re-parameterized over `StarkHash`, with M-12 / M-13a / M-14 corrected on port |
| `472e7efd` | `test(fri)`: soundness negatives for the batched commitment primitives |
| this note | `docs(mmcs)` |

```
 crypto/stark/src/fri/batched.rs                     | 921 +++++  NEW
 crypto/stark/src/fri/mmcs.rs                        |1334 +++++  NEW
 crypto/stark/src/fri/mod.rs                         |   2 +      two `mod` lines only
 crypto/stark/src/tests/batched_mmcs_soundness_tests.rs | 339 +++  NEW
 crypto/stark/src/tests/mod.rs                       |   1 +      one `mod` line only
```

## Scope guards — all honoured

* **No prover/verifier integration.** M-3+ waits for P-a Stage 2. `prover.rs`,
  `verifier.rs`, `continuation.rs`, `prover/src/lib.rs`, anything cuda: untouched.
* **No wire-type change.** `git diff 3a0b8485..HEAD -- crypto/stark/src/proof/ prover/src/`
  is **empty**. `MixedOpening` lives inside `fri/mmcs.rs`, so `StarkProof` / `MultiProof`
  rkyv layouts are byte-identical by construction, not by test. The wire types
  `BatchedQueryOpening` / `BatchedTableData` / `BatchedMultiProof` were **not** needed and
  did not come along.
* **#845 zero-copy view layer intact** (the silent deletion §2.1 warns a rebase would
  cause): `EpochProofView` 10 hits, `ContinuationProofView` 7, `verify_continuation_view`
  5, `access_recursion_archive` 3, `verify_l2g_commitment_binding_view` 15.
* `fri/mod.rs`'s existing paths untouched beyond the two `mod` declarations.

## Drift vs the June code, and how it was adapted

**The hash path — this is M-1, and it is resolved differently from MMCS-PLAN §2.3.**
The June files hard-code `BatchedMerkleTreeBackend` in three places (`hash_group_leaf`,
`hash_group_openings`, `compress`). Both files are now generic over `H: StarkHash`,
reaching `<H::Batched<E>>::hash_data` and `::hash_new_parent` — the same two functions the
existing per-table row-pair tree commits with. Types are `MixedMmcs<E, H>`, matching the
`TableCommit<F, H>` convention already in `prover.rs`.

**I did not add a third `type Mmcs<F>` member to `StarkHash`, and §2.3's recommendation to
add one should be treated as superseded.** Its stated obstacle — "a mixed-height MMCS leaf
is not a `Vec`; the tree builder must know the injection schedule; `IsStreamingLeafBackend`
has no vocabulary for it" — is about `MerkleTree::build`, which `MixedMmcs` never calls. It
builds its own layers and needs exactly two things, a leaf hash over a `Vec<FieldElement<E>>`
and a 2-input compression, both of which `Batched<F>` already has at the right shapes. A
third member whose keccak instance is literally `BatchKeccak256Backend<F>` would be a second
encoding of the same leaf — precisely what PA-PLAN §1.4 forbids ("do not prove that two
independently-written encodings coincide; make them one function"). Going through `Batched`
makes "a single-matrix MMCS equals the per-table tree" true **by construction**;
`single_matrix_root_matches_existing_row_pair_tree` and
`single_matrix_fp3_root_matches_existing_row_pair_tree` pin that no second encoding crept
in, which is the whole backward-compatibility argument §2.3 wanted a test for.

Everything else still exists with compatible signatures: `crate::par::par_map_collect`,
`commitment::commit_bit_reversed`, `proof::stark::PolynomialOpenings`,
`grinding::generate_nonce`, and `fri_functions::{fold_evaluations_in_place,
compute_coset_twiddles_inv, update_twiddles_in_place}`. The module docs' "Task 1 / Task 2 /
Task 7" scaffolding and the two stale doc comments §2.4 flagged (`batched.rs:87-89`
claiming termination mirrors the unbatched phase; `:186-188` claiming #729 is absent) are
gone — present tense, no migration references.

## M-12 — the terminal-polynomial gap: FIXED, with a consequence the plan did not price

`batched_commit_phase` used `num_committed_layers = h_max - 1`, folded to a scalar and
appended it. It now derives the fold count through the shared
`crate::fri::terminal::FriFoldLayout` and appends the terminal polynomial's coefficients,
exactly as `commit_phase_from_evaluations` does. **The saving is exactly `blowup_log + k`
committed layers** (8 at blowup 2 / k=7; the plan's "~9" is its own `k + blowup_log`
estimate rounded up).

**★ The consequence, and M-3/M-4 must not re-derive the terminal from `h_max` alone.**
A bucket whose height is below the terminal would never be folded into the running
codeword — it would be silently dropped from the FRI, which is a soundness hole, not a
perf question. So `BatchedFriLayout::new(h_max, h_min, blowup_log, k)` floors the stop at
the **shortest** bucket:

```
terminal_log = min(blowup_log + k, h_min)
total_folds  = h_max - terminal_log
num_committed = total_folds - 1        (saturating)
```

and the **final fold now injects the bucket at the terminal height** — #768's loop injects
only after the committed folds, so a bucket sitting exactly at the terminal would have been
missed. Rate is preserved by the injection (MMCS-PLAN §M-10.1), so the sum is still degree
`< 2^effective_k` and `coeffs_from_terminal_codeword` applies unchanged. At a real epoch
the shortest table sits well above `blowup_log + k`, so the floor is **inert** and the
layout is exactly the unbatched one — it costs nothing in the common case.

Oracles: `single_bucket_terminal_matches_the_unbatched_commit_phase` asserts same layer
count, same coefficients, same layer roots and identical final transcript state against
`commit_phase_from_evaluations`. It is **non-vacuous** — under #768's construction that
input commits 9 layers where the unbatched one commits 3.
`terminal_is_floored_at_the_shortest_codeword` covers both the inert and the active branch.

## M-13a — shape binding: DONE

`absorb_height_histogram(transcript, heights)` → `absorb_shape_histogram(transcript,
heights, widths)`, binding `(height, width)` pairs; length-prefixed, fixed-width,
order-preserving, no sort or dedup. Renamed because "height histogram" would now be a stale
name and there was no existing call site to migrate. `derive_batched_fri_challenges` threads
widths through. Controls: `absorb_shape_histogram_binds_heights_and_widths_into_alpha`
(height, width and table-order changes each move α) and
`the_shape_encoding_separates_distinct_epochs` (the encoding is injective).

**M-13b is NOT answered and is not mine.** MMCS-PLAN §M-10.4 splits M-13 into (a) add
widths to the round-4 histogram — done here — and (b) *answer whether any rounds-1-3
challenge is shape-exploitable*, which gates §3.4's addendum ratification. That remains
open and needs the integration to be meaningful.

## M-14 — the index convention: DOCUMENTED, CONTROLLED, and HARDENED

`verify_batch` walks the path with `(iota >> level) & 1`, i.e. it consumes the **low**
`h_max - 1` bits, while a shorter matrix inside the tree is located by
`iota >> (h_max - h_m)`, i.e. the **high** bits. Consistent only when this MMCS's `h_max`
equals the FRI's — which §3.1/M-6's batched *preprocessed* round breaks (round h_max 21 vs
FRI h_max 23 at the real 2^21 epoch).

Three dispositions:

1. The module header states the caller's obligation as a hard precondition:
   `iota_round = iota_fri >> (h_max_fri - h_max_round)`, with the reason it fails silently
   (prover and verifier share the routine, so a wrong convention is self-consistent —
   honest proofs verify and the short matrices end up authenticated at positions the
   DEEP/FRI join never checks).
2. **Beyond the plan: `verify_batch` now rejects an `iota` outside `[0, 2^(h_max-1))`.**
   A global index from a taller domain exceeds the round's leaf count most of the time, so
   this converts most of the misuse class into a loud rejection at zero cost — honest
   callers already pass in-range indices. It is a backstop, **not** a substitute for the
   reduction: an index that happens to land in range is still accepted at the wrong leaf,
   and the header says so.
3. The control the analysis demands: `short_round_low_bit_convention_is_exercised` — a
   round with `h_max` 4 under a hypothetical FRI `h_max` 6, asserting (a) the honest
   reduced index verifies [honest-path control, house rule], (b) a tampered row of the
   **SHORT** (injected) matrix is rejected, (c) the un-reduced FRI index is rejected, and
   (d) an in-range-but-wrong leaf is also rejected, so the guard is not the only thing
   standing between the two conventions. A tamper control on the tallest matrix alone
   passes under either convention and catches none of this.

## ★ Two findings the integration must absorb

**1. #768's soundness tests could not be ported, and the reason is structural.**
All 16 tests in `batched_soundness_tests.rs` build a `BatchedMultiProof` via
`multi_prove_batched_ram` and call `Verifier::batched_multi_verify`. Both are integration
surfaces that do not exist here, and porting them would mean porting the integration —
explicitly out of scope. What landed instead is
`crypto/stark/src/tests/batched_mmcs_soundness_tests.rs`, covering what the primitives can
actually decide:

* *Reaching down from #768*: a tampered row in **every** height group (not only the tall
  one — short matrices are bound through injection, and a tall-only control would miss a
  wrong injection level entirely), a tampered or mis-sized authentication path, widths
  disagreeing with the opening.
* *New here, not in #768's file*: an opening replayed at any other index is rejected
  (asserted over the **whole** leaf range, not one sample); two same-shape matrices'
  openings swapped inside a height group is rejected, so INPUT ORDER is part of the
  commitment; a relabelled injection height is rejected; a root from another epoch shape is
  rejected; tampering the FRI transcript (layer root, terminal coefficient, height, width)
  moves the query indices.
* *Deferred with the integration (M-5)*: per-query FRI layer evaluations, OOD values, bus
  balance, query count, grinding nonce. `batched_mmcs_soundness_tests.rs` is the named home
  for those to grow into.

**2. Streaming is per height GROUP, not per matrix — MMCS-PLAN §3.3's pseudocode does not
describe what this leaf layout does.**
`MixedMmcs::commit` takes a `LeafSource` and owns no evaluations; that property is
preserved and is made falsifiable rather than asserted in prose by
`commit_reads_each_height_group_in_one_contiguous_phase`, which traces access windows and
proves each matrix is read inside **one contiguous phase, in descending height order** — so
a caller may produce a height group's LDEs, commit, and drop them before the next group is
needed.

But §3.3 assumes a per-matrix chained absorb (`acc[leaf] = absorb(acc[leaf], m's rows)`).
This layout does **not** do that: the group leaf is a single `hash_data` over the
concatenation, so **every matrix at a given height must be readable simultaneously**. Since
the tallest group is most of an epoch's tables, a caller serving rows from full in-RAM LDE
buffers still holds `O(N)` at the base layer. A `LeafSource` may serve from disk, device
memory or recomputation instead — that is the escape hatch — but true streaming *within* a
height group needs an incremental leaf hasher (absorb matrix by matrix into one sponge per
leaf, ~200 B of state per leaf, ≈200 MB at 2^20 leaves), and `IsStreamingLeafBackend`
exposes only `hash_bytes` and `hash_data_from_slices`, neither of which is a multi-update
API. This is documented in the `fri/mmcs.rs` module header under "Memory: what the caller
may drop, and when".

**Consequence: M-4's peak-anon acceptance test will fail if the integration assumes §3.3's
chain is what the primitive provides.** Either serve the base group's rows without
materializing them, or extend the backend trait with an incremental hasher.

On the FRI side the analogous concern *was* fixed: `HeightCombiner` absorbs codewords one
at a time (`combine_by_height` is now a thin materialized wrapper over it), so a prover
never has to hold every table's quotient at once —
`streaming_absorption_matches_materialized_combine` pins the equivalence.

## Public API the integration will consume

```rust
// crypto/stark/src/fri/mmcs.rs
pub trait LeafSource<E: IsField> {
    fn num_matrices(&self) -> usize;
    fn log_height(&self, m: usize) -> usize;
    fn width(&self, m: usize) -> usize;
    fn append_row(&self, m: usize, bitrev_row: usize, out: &mut Vec<FieldElement<E>>);
}
pub enum BorrowedMatrix<'a, E> { RowMajorNatural {..}, ColMajorNatural {..} }
pub struct MixedMmcs<E: IsField, H: StarkHash>;
impl<E, H> MixedMmcs<E, H> {
    pub fn commit<S: LeafSource<E> + Sync>(source: &S) -> Self;
    pub fn root(&self) -> Commitment;
    pub fn h_max(&self) -> usize;          // added: the round's index space
    pub fn dims(&self) -> &[(usize, usize)];// added: the shape actually committed
    pub fn open_batch<S: LeafSource<E>>(&self, iota: usize, source: &S) -> MixedOpening<E>;
    pub fn verify_batch(root, iota, opening, heights, widths) -> bool;  // never panics
}

// crypto/stark/src/fri/batched.rs
pub struct HeightCombiner<E>;              // new: streaming absorption
pub fn combine_by_height(inputs, alpha) -> Vec<Option<Vec<FieldElement<E>>>>;
pub struct BatchedFriLayout { total_folds, num_committed, terminal_len, effective_k }
pub fn batched_commit_phase(combined, transcript, coset_offset, blowup_log, k)
    -> (Vec<FieldElement<E>>, Vec<FriLayer<..>>);          // coeffs, not last_value
pub fn absorb_shape_histogram(transcript, heights, widths);
pub fn derive_batched_fri_challenges(..) -> Option<BatchedFriChallenges<E>>;  // None = reject
```

Two signature notes for whoever wires this up. `batched_commit_phase` returns the terminal
**coefficients**, not a `last_value` — the round-4 transcript sequence is
`shape histogram → α → (β, root)* → β_final → coeffs → grinding → iotas`, and
`derive_batched_fri_challenges` is the single routine both sides must call so they provably
agree (`batched_round4_prover_inline_matches_verifier_replay` checks the by-hand prover
sequence against it). `derive_batched_fri_challenges` returns `Option` rather than panicking:
`None` when the proof's layer-root count or coefficient count contradicts the layout the
epoch's shape implies, or when a height is out of range — heights come from proof-supplied
trace lengths, so a bogus one is a rejection, never a panic on the verifier's path.

## Suite counts

| | result |
|---|---|
| `stark` lib | **274 passed, 0 failed** (245 baseline + **29 new**) |
| `stark` other test binaries | 0 / 0 / 3 ignored — unchanged |
| `crypto` | **52 passed, 0 failed** |
| `lfm::` | **310 passed / 19 failed / 9 ignored** — exactly the `blake3-real-hash` baseline, untouched |
| `make lint` | clean on **all four** combos (default, no-default+debug-checks, disk-spill, cuda) |
| `make fmt` | applied; `--check` clean |

The 29 new tests also pass in a **debug** build, which is what actually exercises the
`debug_assert`s — the terminal-length check and "every bucket was injected before the
terminal", the two invariants that would catch a wrong fold count.

Reproduce:

```sh
cd /Users/maurofab/workspace/lambda_vm-mmcs
cargo test --release -p stark
cargo test --release -p crypto
cargo test --release -p lambda-vm-prover --lib lfm::     # 310/19 is baseline, not a regression
cargo test -p stark --lib -- fri::batched fri::mmcs batched_mmcs_soundness   # debug: fires the debug_asserts
make lint
```

## Open items

* **M-13b** — is any rounds-1-3 challenge shape-exploitable? Gates §3.4's addendum. Not
  answerable without the integration.
* **M-4's peak-anon test** — see finding 2. The base-layer group must not be served from
  `O(N)` resident LDE buffers, or the win is given back in the same commit.
* **M-6 / the batched preprocessed round** — the one case where the round's `h_max` is
  below the FRI's. The reduction is now documented and range-guarded, but M-6 must apply it
  and keep a per-matrix tamper control on the `prep_root` comparison (MMCS-PLAN §3.3's
  closing warning: consolidating a per-table soundness check into one comparison is exactly
  where coverage quietly goes missing).
* **Leaf allocation** — `hash_group_leaf` builds one `Vec<FieldElement<E>>` per leaf, as
  #768 did. `IsStreamingLeafBackend` exists to avoid exactly that, but `LeafSource::append_row`
  bakes the `Vec` into the trait. Left alone deliberately: it is a micro-opt that has not
  earned a measured win, and changing it touches the trait every caller implements.
* **Not a batching blocker, for Mauro** — MMCS-PLAN §M-10.3's aside: under the most
  conservative proximity-gaps form the batching term is already 2^−108 *today*, at parity
  with the query term. Which theorem/constant the system claims is unstated anywhere.
  Batching does not change the answer; it is the natural moment to write it down.

## Do not

Push. Merge. Wire any of this into `prover.rs` / `verifier.rs` — that is M-3+ and it waits
for P-a Stage 2. Overwrite the RATE-4 `RESUME.md`.
