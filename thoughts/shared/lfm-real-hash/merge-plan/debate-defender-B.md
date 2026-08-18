# Defender B — execution safety, reversibility, and process risk management

**Position:** the plan in `FIX-PLAN.md` is SOUND and READY to execute.

**Angle:** not the correctness of the diagnosis (that is Defender A's). This is an
argument about *process*: whether pin→fix→validate→review is the right risk-managed
shape for a soundness-critical merge reconciliation, whether the operation is safely
reversible, and whether the validation actually discriminates a real fix from a
weakened check.

**Method note.** Everything below marked ✓ VERIFIED was read out of the tree or the
git refs in `/Users/maurofab/workspace/lambda_vm-blake3-merge` during this review.
Claims marked ? INFERRED are derived from `reconcile-report.md`'s recorded
measurements rather than re-measured by me, and I say so at each site. I made no
edits and ran no cargo.

---

## Verdict

Execute it. The central process choice — pin the convention with a decisive
measurement *before* touching a line — is the correct defense against this bug's
specific temptation, and the isolation makes the operation genuinely reversible.

I found four concrete strengthenings. One of them (§S1) is a real defect in the
rollback story that must be fixed before Step 5 runs.

---

## Claim 1 — Pin-before-edit is the right order, and this tree shows why

The "made it pass by weakening a check" failure mode is not hypothetical here.
`expected_public_balance` (`prover/src/lfm/proof.rs:247`) is the only thing standing
between the LFM machine and accepting a proof against a public output it did not
produce. Every degenerate repair to it turns all 20 tests green in one step:

- return a constant;
- drop the dependence on `claimed_public`;
- derive the target from the proof's own `bus_table_contribution()` values.

The bug *presents* as "a number doesn't match" (`total = 5597…836` vs
`expected = 16884…021`), which is precisely the presentation that invites a tuned
constant. Nothing about the symptom distinguishes "the formula is stale" from "the
formula is right and the inputs moved."

Step 1 defuses this by making its **deliverable a named divergence with file:line on
both sides**, before any edit is permitted. That converts the task from "make the
numbers agree" (which has infinitely many answers, almost all wrong) into "restore a
stated convention" (which has one). Only the second framing has a *detectable* wrong
answer.

**Evidence that this discipline is already load-bearing in this tree.** The diagnosis
pass left live instrumentation inside a soundness-critical file. ✓ VERIFIED:

- `crypto/stark/src/verifier.rs` currently differs from `origin/main` by 50 changed
  lines, and **every one of them is diagnostic** — a `W909_DEBUG` env-gated block, a
  set of `DBG909 FAIL:` eprintlns, and a `let ok = …` binding that exists only to hold
  the debug block.
- I checked whether any of it altered control flow. It does not: every insertion is
  print-then-fall-through, and the `if !ok && std::env::var("W909_DEBUG").is_ok()`
  block still returns `ok`. No check was removed or weakened during diagnosis.

So the plan's Step 4 requirement that `crypto/stark/` contain only intended merge
content is a necessary check on real residue, not ceremony. §S6 below makes it
mechanical.

---

## Claim 2 — Isolation and reversibility check out (with one defect)

I verified the refs directly rather than taking the plan's word for them.

✓ VERIFIED:

| fact | value |
|---|---|
| `blake3-campaign-preMerge` | `ed1b7785964568d237567dd0ee83162e9db87d58` |
| `blake3-real-hash` (local) | same commit |
| `blake3-real-hash-mainmerge` HEAD | same commit |
| `refs/heads/blake3-real-hash` on origin | same commit |
| `MERGE_HEAD` in the merge worktree | present (`58160b6f…`) — merge genuinely uncommitted |
| merge worktree `.git` | an 82-byte link file, i.e. a linked worktree |

Consequences, each of which is what "safe and reversible" has to mean concretely:

1. **Nothing is committed.** `git merge --abort` or `git reset --hard
   blake3-campaign-preMerge` restores the pristine tree with no history to rewrite.
2. **Discarding the worktree is free.** Because `.git` is a link file, `git worktree
   remove` drops the working copy without touching the shared object store. There are
   16 worktrees on this machine (`git worktree list`); none of them can be corrupted
   by this operation.
3. **PR #930 is untouched.** `origin/blake3-real-hash` is still the pristine campaign
   tip, so the PR shows pre-merge content and there is zero external exposure until
   the deliberate final fast-forward in Step 5.

### ⚠️ S1 — The rollback claim is currently false, and Step 5 is what makes it matter

`FIX-PLAN.md:96` states the pristine tip is "tagged **and pushed**."

✓ VERIFIED: it is tagged but **not pushed**. `git ls-remote --tags origin` returns no
match for `blake3-campaign-preMerge` (exit 1, empty).

Today this is harmless only by coincidence: the sole remote copy of `ed1b7785` is
`refs/heads/blake3-real-hash`, which happens to point at it. **Step 5's
fast-forward-and-push is the exact moment that stops being true.** After that push,
the only remote record of the pristine campaign tip is gone and the tag meant to
replace it exists on one laptop.

**Fix: push the tag before the final push, not after.** This is a one-command change
to the sequencing and it is the difference between "recoverable from anywhere" and
"recoverable until this disk fails."

---

## Claim 3 — The validation is sufficient

### 3a. Set-equality is the strongest element, and it is already baselined

Running all 20 rather than the one instrumented test is the plan's R4 mitigation and
it matters. But the more powerful criterion is Step 4's last bullet: the full `lfm::`
failure set must equal the pristine baseline's *set*.

? INFERRED (from `reconcile-report.md` §5, a recorded measurement I did not re-run):
the baseline at `ed1b7785` is 306 passed / 19 failed, the 19 being the
`recursion/fibonacci.elf` fixture set, measured in the branch's own worktree with
identical fixture state.

Set-equality is strictly stronger than a pass count, because it flags a test that
newly *passes* for the wrong reason as loudly as one that newly fails. Given R1
(a fix that makes the balance always pass), that direction is the one that matters.

### 3b. Orthogonal guards on what must not move

- Artifact round-trip `constraint_artifact` at 11/11 — pins that `program()` still
  reproduces `air.constraint_program()` bit-for-bit, i.e. that the LFM machine's input
  is unchanged.
- Chip pin `artifact_pin.py --check` — ✓ VERIFIED the script exists at
  `thoughts/shared/lfm-real-hash/gate-oracle/artifact_pin.py`.

Both are already green and neither is downstream of the binding fix, so they are
genuine independent guards rather than restatements of the same signal.

### 3c. ⭐ S2 — The single most important negative control (the plan names the wrong one)

The plan lists `tampered_l2g_binding_rejects` first. That is the **wrong primary
tripwire**: it exercises epoch-root binding, and ✓ VERIFIED at
`machine_tests.rs:3617-3629` its first three tamper vectors reject inside
`super::executor::execute` — guest-side asserts that never reach the balance check at
all. Only its final coherent-swap leg (`:3647`) touches `verify_against`.

**The control that must hold is `tampered_claimed_public_word_rejects` —
`prover/src/lfm/machine_tests.rs:52`, asserting at `:62`.**

✓ VERIFIED, its body:

```rust
let mut claimed = proved.public_words.clone();
claimed[0].1[0] = &claimed[0].1[0] + FE::from(1u64);      // :59
let ok = lfm_verify(LfmProgramKind::TrivialV0, &proved.proof, &claimed, &opts)
    .expect("registry entry exists");
assert!(!ok, "a tampered claimed public word must reject");  // :62
```

It holds the proof **fixed** and perturbs exactly one lane of `claimed_public`. It
therefore fails if and only if `expected_public_balance` stops depending injectively
on the claimed words — which is R1, stated exactly. No other test in the suite
isolates that variable.

**Why it does not fall into the "attack rejected" trap.** A negative control is
worthless if the fix rejects everything, because then it passes for free. This one is
paired with a live positive control on the same program and the same code path:
`trivial_program_proves_and_verifies` at `:36`, asserting `ok` at `:48`. **Run the
pair, and treat either half failing as a stop.** (This is the user's own
`feedback-honest-control-catches-overbroad-fix` rule applied to the specific test.)

**Best single test, if only one gates the commit:**
`different_arena_values_change_the_public_output_not_the_program` at `:66`. ✓ VERIFIED
it carries both directions in one body over two genuinely different proofs:

- `:80` — `assert_ne!(a.public_words, b.public_words)`
- `:81-83` — proof `b` against `b`'s words must **verify**
- `:85-87` — proof `b` against `a`'s words must **reject**

That is positive control, negative control, and a proof that the two statements are
actually distinct, in one test.

### 3d. The tripwire set, with file:line

All ✓ VERIFIED present and ? INFERRED currently-passing (none appears in
`reconcile-report.md` §5's enumerated 22 new failures):

| test | file:line | what it pins | assertion site |
|---|---|---|---|
| `tampered_claimed_public_word_rejects` | `machine_tests.rs:52` | claimed-word injectivity — **primary** | `:62` |
| `different_arena_values_change_the_public_output_not_the_program` | `machine_tests.rs:66` | positive + cross-claim negative in one body | `:81`, `:85` |
| `trivial_program_proves_and_verifies` | `machine_tests.rs:36` | the honest-path pair for the primary | `:48` |
| `tampered_statement_or_root_rejects` | `machine_tests.rs:2248` | statement byte and Phase-A root each move z/alpha; claiming honest words rejects | `:2261`, `:2265-2276` |
| `tampered_l2g_binding_rejects` | `machine_tests.rs:3593` | coherent epoch-root reorder rejects | `:3647` |

**Category error to avoid.** Step 4 says "the output-swap-hazard tests" as if they
were negative controls. They are not one thing:

- `preprocessed_tags_close_the_output_swap_hazard` (`machine_tests.rs:415`) asserts
  **rejection** at `:441-444` — and see §S3, it is currently *failing*, so it is not
  available as a tripwire until the fix lands.
- `keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard`
  (`prover/src/lfm/keccak_probe.rs:261`) asserts **acceptance** — it documents a known
  hazard. ✓ VERIFIED from its body (`ops[1].tag = ops[0].tag; // the whole point:
  duplicate tag`) and its single `assert!` at `:285`. Treating it as a negative
  control would invert its meaning.

---

## Claim 4 — HINT (Step 3) in the same pass is fine, with one condition

The two HINT items are **data, not logic**, and they touch no verification path, so
they cannot mask or be masked by the binding fix:

- `lfm::constraint_tests::constraint_leg_instruction_census` — a missing design-table
  row. ? INFERRED from `reconcile-report.md` §5: the census machinery "ran fine and
  printed a full, sane per-table node/leaf/fused/emitted table — the node-index walk
  over `artifact.nodes` works; it is the *design table* that lacks the new row." That
  is direct evidence the mechanism is healthy and only the pinned data is stale.
- `lfm::constraint_tests::continuation_epoch_constraint_leg_cost` — one pinned
  constant, `left: 62375 right: 63393`.

**Condition (S4).** That second item re-blesses a pinned constant, which is precisely
the anti-pattern the drift tests exist to catch ("investigate, never re-bless" —
`machine_tests.rs:113-116`, the registry drift doc comment, states the house rule).
The 1018-cell delta should be **attributed** to HINT's constraint legs before it is
pasted in. If it cannot be attributed, it is a second finding, not bookkeeping.

**Sequencing improvement.** Put the two HINT changes in a **separate commit** from the
binding fix. You keep the plan's efficiency of doing both in one pass, while leaving
the soundness-critical diff reviewable on its own in Step 5. This removes the only
credible objection to combining them at zero cost.

---

## Concrete strengthenings

### S1 — Push the tag before the final push
See Claim 2. `FIX-PLAN.md:96`'s "tagged and pushed" is ✓ VERIFIED false; the tag is
local-only. Step 5's fast-forward destroys the only remote copy of `ed1b7785`.
**Move `git push origin blake3-campaign-preMerge` to before the branch push.**

### S2 — Primary negative control is `tampered_claimed_public_word_rejects`
See §3c. `machine_tests.rs:52`/`:62`, run paired with `trivial_program_proves_and_verifies`
(`:36`/`:48`). If one test gates the commit, make it
`different_arena_values_change_the_public_output_not_the_program` (`:66`), which
carries both directions.

### ⭐ S3 — "All 20 share one root cause" is not established, and one test contradicts it

Only `machine_proves_the_sample_replay` was instrumented. The plan generalizes from
n=1 to 20.

✓ VERIFIED counterexample: `preprocessed_tags_close_the_output_swap_hazard`
(`machine_tests.rs:415`) is in the failing set (? INFERRED from `reconcile-report.md`
§5's list), but its **only** verify-side assertion is a *negative* one:

```rust
assert!(
    !lfm_verify(LfmProgramKind::KeccakChainV0, &proof, &public, &opts).expect("registered"),
    "with distinct tags the swapped outputs must no longer balance"
);   // :441-444
```

A globally-broken verify **satisfies** that assertion. So this test's failure cannot
be explained by the bus-balance theory. It must come from one of:

- leg 1 — `assert_ne!(tag(0), tag(1), "keccak tags must be distinct")` at `:429`
  (compiler-side);
- leg 2's prove — `prove_keccak_chain_with_tamper(…).expect("locally consistent")` at
  `:440` (prover-side);
- leg 3 — `.expect_err("preprocessed tags cannot be rewritten")` at `:455` plus
  `matches!(err, ProvingError::PrecomputedCommitmentMismatch)` at `:457`
  (prover/commitment-side).

All three are prover- or compiler-side, contradicting the diagnosis's "the proof
proves fine; only verify fails."

This is the plan's own **R3** ("more than one convention shifted at once") showing up
with a name attached. **Recommendation: classify all 22 failures by their actual panic
message before Step 2 concludes single-root-cause.** It costs one test run and it
either confirms the theory across the cluster or saves a wasted fix. This *supports*
the plan's structure — R3 is already in the risk register — it just supplies the
evidence that R3 has materialized.

Related note: `program_id_matches_production_on_the_real_fixture` and
`program_id_folds_pages_in_the_production_layout` each contain a digest `assert_eq!`
*before* their positive `verify_against` (✓ VERIFIED at `:3723` then `:3728-3738`, and
`:3780` then `:3785-3795`). Either assertion could be the failing one. Same
classification argument applies.

### ⭐ S5 — A sharper and cheaper Step 1 than the cross-worktree diff

Step 1 as written compares the same program across two trees
(`lambda_vm-blake3-impl` @ `ed1b7785` vs `lambda_vm-blake3-merge`). There is a better
controlled experiment available **inside the merge tree alone**.

? INFERRED from `reconcile-report.md` §5 (no `trivial_*` test appears among the
enumerated 22 new failures, and the 19 pre-existing are the `fibonacci.elf` set):
**TrivialV0 proves and verifies today, while `machine_proves_the_sample_replay`
fails.** Confirm this first — it costs two test names.

✓ VERIFIED that both run the identical path: `verify_against` (`proof.rs:185`) →
`crate::replay_transcript_phase_a_view` (`proof.rs:219`) → `expected_public_balance`
(`proof.rs:220`). ✓ VERIFIED TrivialV0 has non-empty public words, because
`machine_tests.rs:59` indexes `claimed[0].1[0]`.

Therefore a stale fingerprint layout — the plan's **Outcome C**, and its stated
leading suspect ("a shifted alpha-power offset") — would break **both** programs. It
does not. So:

- Running the same-tree A/B **falsifies Outcome C in a single run**, without needing
  the second worktree at all.
- It localizes the divergence to whatever *differs between the programs* — chip/table
  set, `keccak_rnd_chunks`, preprocessed tag rows — which the cross-tree diff does not
  isolate, because it varies the tree instead of the program.

**Recommendation: run the same-tree passing-vs-failing A/B first; keep the
cross-worktree diff as confirmation, not as the opening move.** Same deliverable,
fewer moving parts, and it discriminates the plan's own Outcome A/B/C trichotomy
faster.

Supporting ✓ VERIFIED facts that narrow this further, all of which back the plan's
"ruled out" list:

- `crypto/stark/src/traits.rs` — main changed **nothing** in the interaction/
  preprocessed surface. `git diff HEAD origin/main -- crypto/stark/src/traits.rs`
  filtered to `fn |interaction|preprocess|num_aux|trace_layout|bus` yields exactly one
  line, `- fn precaptured_constraint_program(`, which is the branch's own feature.
  So `has_trace_interaction` / `is_preprocessed` / `num_auxiliary_rap_columns`
  semantics are unchanged by the merge.
- `crypto/stark/src/lookup.rs` — main's change is the `Arc` wrap of
  `constraint_program` plus a hand-written `Clone` impl. `max_bus_elements` exists on
  both sides (7 occurrences at branch HEAD, 8 on main — the extra is the new `Clone`
  body), so it is not a new bus-layout field.
- All LFM chips are preprocessed through a single site,
  `prover/src/lfm/airs.rs:348` (`.with_preprocessed(root, num_prep)`), so Phase-A
  absorption of `precomputed_commitment()` is uniform across LFM programs and does
  **not** discriminate TrivialV0 from the failing set.

### ⭐ S6 — Blast radius: `replay_transcript_phase_a_view` is NOT LFM-local

Step 2 says "Scope is confined to the **branch's** hand-rolled binding — NOT
crypto/stark," then lists `prover/src/lib.rs::replay_transcript_phase_a_view` as a fix
target. Those two statements are in tension, and the second is the dangerous one.

✓ VERIFIED call graph:

- `replay_transcript_phase_a_view` is defined at `prover/src/lib.rs:989`.
- It is called at `prover/src/lib.rs:1014`, inside
  `compute_expected_commit_bus_balance_view`.
- That function is called at **`prover/src/lib.rs:1442` — the production VM verify
  path** — and at `prover/src/continuation.rs:896`, plus `lfm/epoch_tests.rs:743`,
  `lfm/logup_tests.rs:1201`, and roughly a dozen sites in
  `prover/src/tests/prove_elfs_tests.rs`.
- `replay_transcript_phase_a_view` is *also* called directly at `lfm/proof.rs:219` and
  `lfm/logup_tests.rs:1364`.

**Editing it changes the main VM's verifier**, not just the LFM machine's. The plan's
scope sentence would not catch that, because the file is under `prover/` rather than
`crypto/stark/`.

By contrast, ✓ VERIFIED `expected_public_balance` (`prover/src/lfm/proof.rs:247`) is a
private `fn` with **exactly one caller**, `proof.rs:220`. It is genuinely LFM-local.

**Recommendation:**
1. Strongly prefer landing the fix in `expected_public_balance`.
2. If it must land in the shared replay, the acceptance gate has to include the main
   VM's own verify tests (`prover/src/tests/prove_elfs_tests.rs`) and the continuation
   path, not just `lfm::`.
3. State the rule in Step 2 as "no edit whose blast radius reaches
   `lib.rs:1442`," which is the property that actually matters, rather than a
   directory boundary.

### S7 — Make "crypto/stark is clean" mechanical instead of eyeball

Step 4 asks to "Confirm `git diff` under `crypto/stark/` is only the intended merge
content." That diff is 4435 insertions across 23 files. Reviewing it by eye for stray
diagnostics is not a check. Three binary tests replace it:

**Test 1 (decisive).** `git diff origin/main -- crypto/stark/src/verifier.rs` must be
**exactly empty**. ✓ VERIFIED that all 50 of its currently-changed lines are
diagnostic — including the `let ok = (0..num_queries).all(…)` refactor, which exists
only to hold the `W909_DEBUG` block, and which must revert to the direct
`(0..num_queries).all(…)` return.

**Test 2.** `git diff origin/main -- crypto/stark/` must reduce to **exactly seven
files**, all of them the branch's artifact feature. ✓ VERIFIED the current residual
set is:

```
crypto/stark/src/constraint_ir/artifact.rs        (+771)
crypto/stark/src/constraint_ir/artifact_tests.rs  (+394)
crypto/stark/src/constraint_ir/device.rs          (rkyv derives)
crypto/stark/src/constraint_ir/mod.rs             (ArtifactNode re-export)
crypto/stark/src/constraints/builder.rs           (PartialEq/Eq derive)
crypto/stark/src/lookup.rs                        (with_precaptured + precaptured_program)
crypto/stark/src/traits.rs                        (precaptured_constraint_program)
crypto/stark/src/verifier.rs                      ← MUST DISAPPEAR from this list
```

**Test 3 (free).** `cargo fmt --check`. ✓ VERIFIED that inserting the prints
de-indented four `error!(` call sites to column zero (the diff shows
`-                        error!(` / `+error!(`). Incomplete diagnostic removal is
therefore also a formatting failure. Per the user's global convention, `make fmt` and
`make lint` from the repo root are the right invocations, not per-package clippy.

**Two corrections to Step 4's diagnostic inventory** (✓ VERIFIED by grepping the
working-tree diff for `eprintln|env::var|dbg!|println!`):

1. `prover/src/lib.rs` contains **zero** campaign diagnostics. Step 4 over-names it.
   The actual removal set is `crypto/stark/src/verifier.rs` plus the `LFM_BUS_DEBUG`
   block at `prover/src/lfm/proof.rs:224-240`.
2. **Do not strip main's own instrumentation.** `LAMBDA_VM_TIMELINE_JSON` and
   `LAMBDA_VM_TRACE_BUILDERS` arrived with the merge alongside
   `crypto/stark/src/instruments.rs` (+79 lines) and are legitimate merge content.
   Deleting them while "removing diagnostics" would be its own regression. The
   campaign diagnostics are identifiable by their markers: `W909_DEBUG`, `DBG909`,
   `LFM_BUS_DEBUG`.

---

## Summary of recommended plan amendments

| # | Amendment | Where | Cost |
|---|---|---|---|
| S1 | Push `blake3-campaign-preMerge` **before** the Step 5 branch push | Step 5 / Rollback | one command |
| S2 | Name `tampered_claimed_public_word_rejects` (`machine_tests.rs:52`) the primary negative control, run paired with `trivial_program_proves_and_verifies` (`:36`) | Step 4 | none |
| S3 | Classify all 22 failures by actual panic message before concluding one root cause; `preprocessed_tags_close_the_output_swap_hazard` (`:415`) already contradicts it | Step 1 | one test run |
| S4 | Attribute the 62375→63393 delta to HINT before re-blessing; separate commit from the binding fix | Step 3 / Step 5 | small |
| S5 | Run the same-tree TrivialV0-vs-sample A/B first; cross-worktree diff becomes confirmation | Step 1 | negative (cheaper) |
| S6 | Scope rule = "no edit whose blast radius reaches `lib.rs:1442`"; prefer `expected_public_balance` (one caller) over `replay_transcript_phase_a_view` (production VM verify) | Step 2 | none |
| S7 | Replace the eyeball diff check with: `git diff origin/main -- crypto/stark/src/verifier.rs` empty, residual = the 7 artifact files, `make fmt` clean | Step 4 | none |

None of these changes the plan's shape. S1 is a correctness fix to the rollback
story; S3 and S5 sharpen Step 1 within its own stated Outcome A/B/C frame; S6 and S7
replace prose scope boundaries with mechanical ones. The plan's process —
pin, then fix, then validate positively *and* negatively, then review — is the right
one, and I recommend executing it with these seven amendments.
