# Fix plan: LFM recursion-machine bus-balance divergence after the main merge

**Context.** Merging `origin/main` into `blake3-real-hash` (worktree `lambda_vm-blake3-merge`,
branch `blake3-real-hash-mainmerge`). The artifact-feature reconciliation is done and green
(round-trip 11/11). The remaining breakage: **20 LFM machine proof/verify tests fail**, all on
one root cause, plus **2 trivial HINT bookkeeping tests**. This plan fixes both. Nothing is
committed; the pristine campaign tip is tagged `blake3-campaign-preMerge`.

## Diagnosis (evidence, not hypothesis)

Instrumented `Verifier::multi_verify_views` and `verify_against` (throwaway `eprintln`s, to be
removed). Findings for `machine_proves_the_sample_replay`:

1. The proof **proves** fine; only **verify** fails.
2. Every failure is the **cross-table LogUp bus-balance** check (`total != expected_bus_balance`
   at `verifier.rs:1442`). No other check fires — composition-parts, `ood_blocks_well_formed`,
   preprocessed-commitment match, per-table `verify_rounds_2_to_4` (incl. #909's width check) all
   **pass**. So per-table STARK verification is correct; only the cross-table binding is off.
3. Per-table contributions (14 tables, all `has_interaction`): sum = `5597…836`, expected =
   `16884…021` — different. Not a sign flip, not a missing table, not zero.
4. Ruled out as the cause (branch vs `origin/main`, byte-identical or unchanged):
   `LOGUP_NUM_CHALLENGES` (2), `compute_alpha_powers`, `build_accumulated_column_from_terms`
   (the L computation), the Phase-A transcript absorption order (the LFM machine's hand-rolled
   `replay_transcript_phase_a_view` matches `multi_verify_views` Phase A exactly), and the
   fiat-shamir/transcript module (untouched by the merge).

**Conclusion.** The LFM machine hand-rolls its cross-table binding — `expected_public_balance`
(`prover/src/lfm/proof.rs`) and `replay_transcript_phase_a_view` (`prover/src/lib.rs`) — to mirror
crypto/stark's LogUp convention. The merge's large crypto/stark batch (prover rewrite #877/#875/#863
et al.) shifted that convention in a way the obvious diffs don't reveal, so the branch's hand-rolled
mirror disagrees. **main's crypto/stark stays authoritative; the hand-rolled LFM binding adapts** —
exactly as with the artifact feature.

This is **soundness-critical**: `expected_public_balance` is the recursion verifier's cross-table
check. A wrong fix could make the machine accept invalid proofs. The fix must be validated by BOTH
positive (valid proofs verify) AND negative (tampered proofs still rejected) controls.

## Step 1 — PIN the exact convention (decisive, before any fix)

Compare the SAME `sample()` proof's internals on the pristine branch vs the merge:
- In `lambda_vm-blake3-impl` @ `ed1b7785` (branch, test passes) and in `lambda_vm-blake3-merge`
  (merge, test fails), print: `z`, `alpha`, each table's `bus_table_contribution`, and `expected`.
- **Outcome A:** `z`/`alpha` differ ⇒ challenge derivation changed (unlikely — transcript module
  untouched). Fix targets the replay.
- **Outcome B:** `z`/`alpha` identical but per-table contributions differ ⇒ main's aux/LogUp
  column construction changed the L values ⇒ the fix is either in how the LFM machine reads/sums
  contributions or in `expected_public_balance`'s target formula.
- **Outcome C:** contributions identical, only `expected` differs ⇒ the target formula in
  `expected_public_balance` is stale ⇒ fix it directly.
- Also inspect the LfmPublic **send token layout** (how the LFM chips send `(index, v0..v3)` to the
  LfmPublic bus) vs `expected_public_balance`'s hard-coded fingerprint `busid + index·α + Σ v_l·α^{2+l}`
  vs main's actual bus-interaction fingerprint alpha-power assignment. A shifted alpha-power offset
  is the leading suspect.

Deliverable: the exact convention that shifted, named with file:line on both sides.

## Step 2 — FIX the hand-rolled binding

Scope is confined to the **branch's** hand-rolled binding — NOT crypto/stark:
- `prover/src/lfm/proof.rs::expected_public_balance` (the fingerprint/target formula), and/or
- `prover/src/lib.rs::replay_transcript_phase_a_view` (the challenge replay),
- and any sibling that mirrors the same convention (`compute_expected_commit_bus_balance_view`,
  `absorb_lfm_statement`).
Update them to main's pinned convention. No edits under `crypto/stark/` (main's IR/verifier remain).

## Step 3 — HINT bookkeeping (independent, trivial)

Add the `HINT` design-table entry to the LFM design census and update the one epoch-budget constant
(`lfm::constraint_tests::constraint_leg_instruction_census`, `continuation_epoch_constraint_leg_cost`).
These are unrelated to the bus-balance fix; done in the same pass because they're the last 2 of the 22.

## Step 4 — VALIDATE

- Remove ALL throwaway diagnostics (verifier.rs, proof.rs, lib.rs). Confirm `git diff` under
  `crypto/stark/` is only the intended merge content (no diagnostics, no logic changes).
- **Positive:** all 20 machine proof/verify tests pass; full `lfm::` returns to a clean baseline
  (the 19 pre-existing `fibonacci.elf` failures only, modulo the HINT tables now passing).
- **Negative controls (mandatory, soundness):** the existing tamper/rejection tests
  (`tampered_l2g_binding_rejects`, the output-swap-hazard tests, any "must NOT verify" tests) still
  REJECT. A fix that makes the balance always pass is as wrong as the bug.
- Artifact round-trip suite (`constraint_artifact`) stays 11/11.
- Chip gate `artifact_pin.py --check` still green (BLAKE3 chip untouched by any of this).
- Cross-version / whole-suite sanity: full lib suite failure set vs the `blake3-campaign-preMerge`
  baseline shows only pre-existing fixture/env failures — zero new.

## Step 5 — REVIEW + FINALIZE

- Adversarial review of the binding change (it is the recursion verifier's cross-table soundness
  check): confirm the new formula matches main's convention AND that negative controls hold.
- Commit the merge; fast-forward `blake3-real-hash` to the merged branch; push → PR #930 up to date.
- Keep `blake3-campaign-preMerge` as the recoverable pristine point.

## Rollback

Merge is uncommitted in a dedicated worktree; the pristine tip is tagged and pushed. Any failure ⇒
`git reset --hard blake3-campaign-preMerge` (or discard the worktree). Zero risk to PR #930 until the
final fast-forward.

## Risk register

- **R1 (high impact):** wrong binding formula → machine accepts invalid proofs. Mitigation: negative
  controls in Step 4 are mandatory and gate the commit.
- **R2:** the convention shift is in main's TRACE/aux construction (Outcome B), not the formula —
  fix might need to touch how contributions are read, not just `expected`. Mitigation: Step 1 pins
  which, before any edit.
- **R3:** more than one convention shifted at once. Mitigation: Step 1 compares ALL of z/alpha/
  per-table-contrib/expected, catching multiple divergences together.
- **R4:** the fix passes the sample test but not other programs (join, splice, keccak variants).
  Mitigation: Step 4 runs the full 20, not one.
