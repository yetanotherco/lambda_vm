# Adversarial attack on FIX-PLAN.md — attacker's brief

**Verdict: the plan should be sent back for revision.**

Its central conclusion — *"main's crypto/stark batch shifted the LogUp convention, so the
branch's hand-rolled binding must adapt"* — is **falsified by the diff**. I also found what
is almost certainly the real root cause, in a file the plan places out of scope. Executing
Step 2 as written would most likely bake an unbalanced-bus residual into a verifier
soundness check, and **none of the plan's named negative controls would catch it**.

All paths are in the merge worktree `/Users/maurofab/workspace/lambda_vm-blake3-merge`
unless stated otherwise. I was read-only (no cargo), so execution-dependent claims are
marked; everything marked ✓ VERIFIED was established by reading the code or the diff.

---

## 1. CRITICAL — The likely real root cause is in `prover/src/lfm/keccak_adapter.rs`, which the plan never mentions

✓ VERIFIED by reading both sides of the diff.

Main deleted **120 `BusId::Hwsl` sender interactions per keccak round** from the
**production** KECCAK_RND chip, replacing them with inline μ-gated linear identities.

`git diff ed1b7785 origin/main -- prover/src/tables/keccak_rnd.rs`:

- `Vec::with_capacity(1371)` → `Vec::with_capacity(1031)`
- the `--- Theta: HWSL for rotated C (20) ---` block is removed, replaced by
  `--- Theta: rotate-C-by-1 shift is enforced by an inline μ-gated linear identity
  (see KeccakRndConstraints), not an HWSL lookup. ---`
- the `--- Rho: HWSL (100) ---` block is removed
- the new module comment states: *"The matching HWSL multiplicities are likewise dropped
  on the BITWISE side (`collect_bitwise_from_keccak`)."*

Main updated the production receiver side accordingly:
`prover/src/tables/trace_builder.rs`, +418/−41.

**The LFM machine embeds those production chips.** `prover/src/lfm/airs.rs:21`:

```rust
use crate::tables::{bitwise, keccak_rc, keccak_rnd};
```

and `airs.rs:239` (`let rnd_interactions = keccak_rnd::bus_interactions().len();`) and
`airs.rs:488-494` build the LFM KECCAK_RND AIRs from `keccak_rnd::bus_interactions()` and
`keccak_rnd::KeccakRndConstraints`. So the LFM AIR set picked up main's deletion
automatically, at merge time, silently.

**But the LFM machine has its own forked copy of the receiver-side multiplicity
collection, and it is branch-only code the merge never touched.**
`prover/src/lfm/keccak_adapter.rs:306-318`:

```rust
/// BITWISE lookups the `KECCAK_RND` rows of `ops` send: exactly `24 * 1148` per
/// permutation.
///
/// This is the per-round half of `trace_builder::collect_bitwise_from_keccak`,
/// forked rather than called: ...
pub fn bitwise_ops_for(ops: &[KeccakAdapterOperation]) -> Vec<BitwiseOperation> {
    let mut out = Vec::with_capacity(ops.len() * 24 * 1148);
```

It still pushes `BitwiseOperationType::Hwsl` in the Theta loop
(`keccak_adapter.rs:361-366`, 20 per round) and in the Rho loop
(`keccak_adapter.rs:441-446`, 100 per round) — **exactly the 120 sends main deleted**.
The per-round count `1148` is pinned in the capacity and is now stale.

**Consequence.** The LFM BITWISE chip receives 120 HWSL lookups per round that KECCAK_RND
no longer sends. With circular LogUp constraints there are no boundary constraints on the
accumulator, so **proving still succeeds** and the imbalance surfaces only as a nonzero
residual in `total` at `crypto/stark/src/verifier.rs:1448`. That is precisely the reported
symptom, and it predicts the observed failing/passing split exactly: every keccak-touching
program fails; `trivial_program_*` (no keccak) passes.

**Why this destroys the plan.** This is the plan's own **Outcome B**, and the fix lives in
`prover/src/lfm/keccak_adapter.rs` — not in `expected_public_balance`, not in
`replay_transcript_phase_a_view`. Step 2's enumerated scope cannot reach it.

**To close:** before any edit, dump the **per-bus** residual, not just the per-table total.
`crypto/stark/src/lookup.rs` already has `compute_debug_bus_sums_batched`, and
`DEBUG_BUS_TRACKER=1` enables per-bus balance reporting in release. Run it on
`machine_proves_the_sample_replay` and confirm whether the residual lands on
`BusId::Hwsl`/`Bitwise` or on `LfmPublic`. That single measurement decides the whole plan
and should have preceded it.

---

## 2. CRITICAL — Step 2 as written is a genuine soundness regression, and R1's mitigation is blind to it

If the residual is an unmatched-HWSL constant (Finding 1), then "adjust
`expected_public_balance` to main's convention" means folding that residual into the
verifier's expected target at `prover/src/lfm/proof.rs:247-276`. The LfmPublic target would
then absorb an arbitrary unbalanced-bus remainder, and the cross-table check at
`crypto/stark/src/verifier.rs:1448` would **permanently stop detecting unmatched HWSL
lookups in the recursion machine**. That is the recursion verifier's only cross-table
binding.

Critically, **every negative control the plan names would stay green.** The residual is a
constant independent of the claimed public words, so tampering a value lane still moves
`expected` and still rejects. R1 says "negative controls in Step 4 are mandatory and gate
the commit" — but the controls are structurally blind to this exact failure mode. The
plan's headline mitigation does not mitigate its headline risk.

**To close:** add a control sensitive to a *constant offset* in `expected`, not just to word
tampering — cheapest is asserting the per-bus residual is zero on every bus except
LfmPublic, which is the same measurement Finding 1 requires anyway.

---

## 3. CRITICAL — "20 failures, one root cause" is falsified; at least three cannot be the hand-rolled binding

✓ VERIFIED by reading the tests. The diagnosis is **n = 1**
(`machine_proves_the_sample_replay`) generalized to 20 without checking.

### (a) Both `keccak_probe` failures never touch the hand-rolled binding at all

`prover/src/lfm/keccak_probe.rs:126-143`:

```rust
fn verify_proof(opts: &ProofOptions, adapter: &AdapterAir,
                proof: &stark::proof::stark::MultiProof<F, E, ()>) -> bool {
    let rnd_air = create_keccak_rnd_air(opts);
    let rc_air  = create_keccak_rc_air(opts).with_preprocessed(...);
    let bw_air  = create_bitwise_air(opts).with_preprocessed(...);
    let refs: Vec<DynAir> = vec![adapter, &rnd_air, &rc_air, &bw_air];
    let mut vt = transcript();
    Verifier::multi_verify_views(&refs, MultiProofView::Owned(proof), &mut vt, &FEE::zero())
}
```

Expected bus balance is a **hard-coded `FEE::zero()`**. This path calls neither
`verify_against`, nor `lfm_verify`, nor `expected_public_balance`, nor
`replay_transcript_phase_a_view`. Its AIR set is the **production** VM AIRs plus a local
adapter. Yet `keccak_probe::adapter_probe_proves_real_permutations` and
`keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard` are both in the
new-failure list (`reconcile-report.md:281-282`). **No change to `proof.rs` or `lib.rs` can
fix them.** They *are* explained by Finding 1.

(The reconcile-report itself noticed this — §5.3, *"keccak_probe.rs contains zero
occurrences of `artifact`, yet two of its tests are in the new-failure list"* — and used it
to exonerate the artifact reconciliation. The FIX-PLAN then folded them into the LFM
binding, which is equally impossible.)

### (b) `preprocessed_tags_close_the_output_swap_hazard` cannot fail from a bus-balance mismatch

`prover/src/lfm/machine_tests.rs:415-460`. Its **only** verify assertion is negative:

```rust
    assert!(
        !lfm_verify(LfmProgramKind::KeccakChainV0, &proof, &public, &opts).expect("registered"),
        "with distinct tags the swapped outputs must no longer balance"
    );                                                  // :441-444
```

A universal verify failure **satisfies** that. Its remaining failure modes are all
prove-side or compile-side:

- `assert_ne!(tag(0), tag(1), "keccak tags must be distinct")` — `:429`
- `.expect("locally consistent")` on `prove_keccak_chain_with_tamper` — `:440`, proving
  must **succeed**
- `expect_err(...)` + `matches!(err, ProvingError::PrecomputedCommitmentMismatch)` —
  `:455-459`

Main rewrote the prover's preprocessed / split-tree commit path (`crypto/stark/src/prover.rs`,
+1725 lines; the diff comment reads *"Preprocessed tables also carry a handle with
`trace_dev` (the split-tree path)"*), a plausible independent cause. **Neither the plan's
theory nor mine explains this test** — which is the point: nobody has read its actual
failure message.

**To close:** capture the actual assertion/panic message for all 20 and bucket them. This is
one `cargo test ... 2>&1 | grep -B2 -A5 panicked` away and should have preceded the plan.

---

## 4. HIGH — "The convention shifted" is contradicted by the diff *and* by the tests that still pass

✓ VERIFIED.

**From the crypto/stark side.** `git diff ed1b7785 origin/main -- crypto/stark/src/lookup.rs`
is 246 lines and touches only:

- `Arc`-wrapping `constraint_program` (perf)
- a new `Clone` impl for `AirWithBuses` and `#[derive(Clone)]` on `AuxiliaryTraceBuildData`
- removal of `with_precaptured` / `precaptured_program` (the branch's own feature, also
  removed from `crypto/stark/src/traits.rs`)
- lazy host `main_cols_cell` for the GPU-resident aux path

**`compute_alpha_powers` (`lookup.rs:73`), `add_combined_terms` (`:265-370`), and the entire
alpha-offset assignment (`:624-800`) are untouched.** There is no shifted fingerprint
convention in crypto/stark to adapt to.

Phase A is likewise preserved on the prover side: the `prover.rs` diff shows the same
absorption sequence (precomputed root if preprocessed, then main root, in index order),
followed by the same two-challenge sample. `crypto/stark/src/verifier.rs:1305-1350` matches
`prover/src/lib.rs:994-1002` exactly. Outcome A is dead.

**From the test side.** `prover/src/lfm/machine_tests.rs:36
trivial_program_proves_and_verifies` asserts a **positive** verify through
`lfm_verify` → `verify_against` → `expected_public_balance`, with non-empty public words
(proved by `:52 tampered_claimed_public_word_rejects` indexing `claimed[0].1[0]`, and by
`:81-87` which asserts both a positive verify and a cross-claim reject). **None of those
three is in the new-failure list.** So `expected_public_balance` computes the correct value
today for at least one program — and a formula edit would break them.

**This also retires Step 1's stated "leading suspect"** (a shifted alpha-power offset). The
LFM_PUBLIC sender is `direct(cols::INDEX)` plus `word(cols::V0)`
(`prover/src/lfm/chips.rs:1340-1349`) — five single-alpha elements — and
`proof.rs:253-265` maps them to α¹…α⁵ over `bus = BusId::LfmPublic = 34`
(`prover/src/tables/types.rs:373`), consistent with its own doc comment at `proof.rs:245`
and with the byte-identical `add_combined_terms`. Step 1 is pointed at the wrong thing.

---

## 5. HIGH — The named negative control is the wrong test, and one "control" is inverted

✓ VERIFIED.

**`tampered_l2g_binding_rejects` — the plan's headline soundness gate — contains no
negative verify assertion at all.** `prover/src/lfm/machine_tests.rs:3593-3660`: its tamper
vectors go through

```rust
        let err = super::executor::execute(&program, &arenas, &super::hash::TestPermutation)
            .err()
            .unwrap_or_else(|| panic!("{what}: must not execute"));
```

— an **executor-level** reject, no proof and no verifier. Its coherent branch then asserts
the proof **proves** and compares `published_root(&proved.public_words, 0)` in host Rust.
It cannot detect an `expected_public_balance` that accepts everything.

**Worse, Step 4 lists "the output-swap-hazard tests" among controls that must "still
REJECT".** `prover/src/lfm/keccak_probe.rs:261-290
duplicate_tag_output_swap_accepts_demonstrating_hazard` asserts the **opposite**:

```rust
    let proof = prove_traces(&opts, &adapter, &mut traces).expect("locally consistent");
    assert!(
        verify_proof(&opts, &adapter, &proof),
        "documents the tag-uniqueness obligation: with duplicate tags the swapped \
         outputs still balance the bus, so the verifier cannot catch the forgery"
    );
```

It must **ACCEPT** — it documents an open hazard. Under the plan's framing, that test
staying red would be misread as "control holding".

**The controls that actually guard this code go unnamed:**
`machine_tests.rs:52` (`tampered_claimed_public_word_rejects`), `machine_tests.rs:84-87`
(cross-claim), `wrap_tests.rs:490` and `:510`, `constraint_tests.rs:1365`,
`blake3_socket_tests.rs:1417`.

Two caveats the plan must state and does not:

1. They currently pass **vacuously** while everything rejects. The gate must be "positive
   AND negative green in the same run", not "the reject tests still reject".
2. Every one of them perturbs a **value** lane. Nothing covers the `index·α` term or a
   permutation of words, so a formula edit that dropped or mis-weighted `index·α` would
   keep every control green.

**To close:** name the right tests, require positive+negative in one run, add an
index-permutation vector.

---

## 6. HIGH — Scope is understated: the convention is also hand-rolled *in-circuit*, and other consumers share the code

✓ VERIFIED.

**In-circuit mirror.** `prover/src/lfm/statement_replay.rs:164-190` (`replay_phase_a`) is an
**LfmBuilder / in-circuit** mirror of `replay_transcript_phase_a_view`, doc-linked as such
at `:164-167` (*"Mirrors `crate::replay_transcript_phase_a_view` — for each air, the
preprocessed commitment when it has one, then the main trace root, and finally `z` and `α`
…"*). `machine_tests.rs:2101` mirrors it a third time.

If Step 1 landed on Outcome A, the fix would have to change the **compiled LFM program**,
which moves `artifacts.roots` and `artifacts.program_id`, which breaks every
`registry_drift_*` test (`machine_tests.rs:118`, `:230`, `:511`, `:754`) — whose own doc
says *"A failure here means the trivial program, a chip layout, the commit pipeline or the
digest changed: investigate, never re-bless"* (`machine_tests.rs:113-117`). Step 2's file
list omits `statement_replay.rs` entirely, and the plan never mentions registry
re-blessing.

**Non-LFM consumers.** `replay_transcript_phase_a_view` is not LFM-only. Reached via
`compute_expected_commit_bus_balance_view` (`prover/src/lib.rs:1007-1016`) from:

- `prover/src/lib.rs:1442` — the **VM** verifier
- `prover/src/continuation.rs:896` — the **continuation** verifier
- `prover/src/lfm/logup_tests.rs:1201`, `prover/src/lfm/epoch_tests.rs:743`

Editing it changes the VM and continuation verifiers. The plan's validation runs neither
suite.

---

## 7. HIGH — Step 2's scope contradicts the plan's own R2

R2 concedes the shift may be in main's trace/aux construction (Outcome B). But Step 2 offers
only two edit targets, both verifier-side. Under Outcome B the correct fix is on the LFM
trace/chip side (`prover/src/lfm/keccak_adapter.rs`, `trace.rs`, `chips.rs`) — none of which
is in scope. "Patch `expected` until it matches the contributions" is precisely the move R1
warns against, and Step 2 provides no exit from it.

R2's mitigation ("Step 1 pins which") is not a mitigation: pinning *which* outcome holds
does nothing if Step 2 only has verifier-side edits available. **Outcome B needs an explicit
third branch: if the contributions changed, do NOT touch `expected` — find why the trace
changed.**

---

## 8. MEDIUM — Validation omits the one suite that gates the merge's actual risk

Step 4 gates on `lfm::`, `constraint_artifact` (11/11), `artifact_pin.py --check`, and a
full-lib baseline diff. **It never runs `cargo test -p stark`.**

Main added `crypto/stark/src/tests/opening_width_tests.rs` (+532) and
`crypto/stark/src/tests/aux_opening_width_tests.rs` (+715) — the executable soundness tests
for #909, the very check the diagnosis claims to have cleared — and the merged
`verifier.rs` has been hand-edited (Finding 10). Those tests are the gate and they are not
in the plan.

Separately, `constraint_artifact` 11/11 and `artifact_pin.py` do not exercise `proof.rs` or
`lib.rs` at all. Listing them as gates for **this** change creates false assurance.

---

## 9. MEDIUM — The tree is not in the state the plan describes, and Step 4's verification command will not do what it says

✓ VERIFIED. `git diff --name-only --diff-filter=U` shows five paths still in **unmerged
index state**:

```
crypto/stark/src/lookup.rs
prover/src/continuation.rs
prover/src/tests/constraint_program_device_tests.rs
prover/src/tests/constraint_program_tests.rs
prover/src/tests/ood_window_ir_tests.rs
```

The working files are resolved (`grep -c '^<<<<<<<' crypto/stark/src/lookup.rs` → 0) but
were never `git add`ed. Consequently `git diff --stat -- crypto/stark/` literally prints

```
 crypto/stark/src/lookup.rs | Unmerged
```

so Step 4's *"Confirm `git diff` under `crypto/stark/` is only the intended merge content"*
would **silently skip `lookup.rs`** — the very file whose convention the entire plan is
about.

**To close:** `git add` the resolved paths first, and state the check as
`git diff origin/main -- crypto/stark/` returning **empty**.

---

## 10. MEDIUM — The in-tree diagnostics already changed crypto/stark semantics

✓ VERIFIED. `crypto/stark/src/verifier.rs:1448-1454` in the merged tree:

```rust
            if total != *expected_bus_balance {
                #[cfg(not(feature = "test_fiat_shamir"))]
                eprintln!("DBG909 FAIL: BUS BALANCE total={total:?} expected={expected_bus_balance:?}");
error!(
                    "LogUp bus does not balance: ...",
                    total, expected_bus_balance
                );
```

On clean main (`git show origin/main:crypto/stark/src/verifier.rs`, the
`LogUp bus does not balance` block, ~:1413-1418) that `#[cfg]` guards `error!`. The
inserted `eprintln!` **stole the attribute**, leaving `error!` unguarded under
`test_fiat_shamir`. The same de-indent-to-column-0 damage is at `:1315`, `:1323`, `:1363`.

Also present: a `W909_DEBUG` block at `verifier.rs:263-292` that restructured
`trace_opening_widths_well_formed`'s `(0..num_queries).all(...)` into
`let ok = ...; ...; ok`, and a `std::env::var("LFM_BUS_DEBUG")` lookup on **every verify**
at `prover/src/lfm/proof.rs:224-240`.

Step 4's "remove ALL" is right; the acceptance criterion should be an **empty** diff, not
"only the intended merge content".

---

## 11. LOW-MEDIUM — Baseline comparability is not established

`reconcile-report.md` §3 says `executor/program_artifacts/asm/` was copied into the merge
worktree, while the `ed1b7785` baseline worktree had *"neither `asm/` nor `recursion/`
present"* (§5). It claims both-ways measurement, but the FIX-PLAN Step 4 gate — full lib
suite vs `blake3-campaign-preMerge` — **has not been run at all** (PLAN.md step 15 is still
open).

Several failing tests are fixture-driven: `machine_tests.rs:3706`
(`program_id_matches_production_on_the_real_fixture`, via
`proof_fixture::load_or_generate(&fixture_cache())` at `:3690`) and `machine_tests.rs:4282`
(`the_register_derivation_proves_and_verifies`, via `proof_fixture::fixture_options()`), so
fixture asymmetry can move tests between the "pre-existing 19" and "new 22" buckets and
invalidate the "22 new, 0 fixed" diff.

Additionally `machine_tests.rs:3709-3713` hard-asserts `pages.is_empty()` about that
fixture, and main's private-page OFFSET change (reconcile-report §4, `continuation.rs:240`
now calls `with_preprocessed(page::private_page_preprocessed_commitment(opts),
page::NUM_PREPROCESSED_COLS_PRIVATE)`) touches page preprocessing. That deserves an explicit
check rather than an assumption.

---

## 12. LOW — Note for the judge on what the plan gets right

To avoid a strawman reading: the plan's *process* instincts are sound. Step 1 (pin before
fixing), R1 (this is soundness-critical), the rollback story, and the insistence on both
positive and negative controls are all correct and should survive revision. The defects are
that Step 1's three outcomes are already decidable from the diff and were not decided; that
Step 2's scope was fixed *before* Step 1 ran, so it cannot express the outcome the evidence
actually supports; and that the controls named in Step 4 are the wrong tests.

---

## What must change before this plan is implementable

1. **Measure the per-bus residual** on one failing test (`DEBUG_BUS_TRACKER=1` /
   `compute_debug_bus_sums_batched`), to confirm or refute that the imbalance is on
   HWSL/Bitwise rather than LfmPublic. Decisive, cheap, and it settles Findings 1, 2, 4, 7
   at once.
2. **Get the actual failure message for all 20** and bucket them — not one test generalized
   to twenty.
3. **Rewrite Step 2** with `prover/src/lfm/keccak_adapter.rs` (and the trace/chip side
   generally) in scope, plus an explicit prohibition on editing `expected_public_balance`
   unless Step 1 proves the formula itself is stale.
4. **Name the correct negative controls**, state the positive-and-negative-in-one-run rule,
   and add an index-permutation vector.
5. **Add `cargo test -p stark`** (incl. `opening_width_tests` / `aux_opening_width_tests`)
   to the gate; drop `constraint_artifact` / `artifact_pin.py` as gates for this change or
   label them as regression-only.
6. **`git add` the five unmerged paths** so the crypto/stark diff check actually inspects
   `lookup.rs`; restate the criterion as an empty diff vs `origin/main`.
