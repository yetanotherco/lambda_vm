# Defender A — the FIX-PLAN is sound and ready

Adversarial review of `thoughts/shared/lfm-real-hash/merge-plan/FIX-PLAN.md`.
Position: **the plan should be implemented**, with the strengthenings in §6.

All code citations are from the merge worktree
`/Users/maurofab/workspace/lambda_vm-blake3-merge` (branch
`blake3-real-hash-mainmerge`, `git merge origin/main` uncommitted, `MERGE_HEAD`
present). Revisions referenced: `ed1b7785` = pre-merge campaign tip = tag
`blake3-campaign-preMerge` = `blake3-real-hash`; `origin/main` = `528a8411`.

Confidence markers per the house rule: ✓ VERIFIED = I read the file or ran the
comparison; ? INFERRED = rests on a measurement in `reconcile-report.md` that I
did not re-run.

---

## Verdict

The plan's **structure** — pin before fix, confine the fix to the branch's
hand-rolled mirror, gate on controls in *both* directions, keep rollback bounded
— is correct and ready to implement. Its **diagnosis narrative** is the weak
part, and the most useful thing I can do for it is show that two of its three
Step-1 outcomes are already dead by diff, and that a discriminating control it
never names is sitting in the same test file. Neither finding changes the plan's
direction; both make Step 1 cheaper and sharper.

---

## 1. The diagnosis is correct *in kind*

**Claim.** The failure is the cross-table LogUp bus balance, per-table STARK
verification is fine, and the cause lives in the LFM's hand-rolled binding
rather than in main's `crypto/stark`.

**Evidence that the observation is real, not inferred.** The instrumentation
that produced the plan's §2 findings is still in the worktree, and it prints a
*distinct* message for every rejection path upstream of the balance check:

| site | print |
|---|---|
| `crypto/stark/src/verifier.rs:1290` | `DBG909 FAIL: composition parts, table {idx}` |
| `crypto/stark/src/verifier.rs:1302` | `DBG909 FAIL: ood_blocks_well_formed, table {idx}` |
| `crypto/stark/src/verifier.rs:1314` | `DBG909 FAIL: preprocessed commitment MISMATCH table {idx}` |
| `crypto/stark/src/verifier.rs:1322` | `DBG909 FAIL: preprocessed commitment MISSING table {idx}` |
| `crypto/stark/src/verifier.rs:1362` | `DBG909 FAIL: missing bus_public_inputs table {idx}` |
| `crypto/stark/src/verifier.rs:1417` | `DBG909 FAIL: verify_rounds_2_to_4 table {idx}` |
| `crypto/stark/src/verifier.rs:1450` | `DBG909 FAIL: BUS BALANCE total=… expected=…` |

plus a `W909_DEBUG` block inside `trace_opening_widths_well_formed` (#909's
width pin) that dumps the expected/actual precomputed/main/aux split. ✓ VERIFIED
by reading the diff of `crypto/stark/src/verifier.rs` against `origin/main`. So
"only the balance fired, everything else passed" is an *observed* fact with
per-check granularity, not a deduction.

**Evidence the plan's eliminations hold.**

- ✓ VERIFIED `LOGUP_NUM_CHALLENGES = 2` and `LOGUP_CHALLENGE_ALPHA = 1`
  (`crypto/stark/src/lookup.rs:102,105`), so the replay's `(z, alpha)` ordering
  at `prover/src/lib.rs:1000-1002` matches the consumer's
  `challenges[0]` / `challenges[LOGUP_CHALLENGE_ALPHA]` at `lookup.rs:1736-1737`.
- ✓ VERIFIED the fingerprint the doc comment at `prover/src/lfm/proof.rs:245-246`
  advertises is what the code computes: `powers[i] = α^{i+1}`
  (`proof.rs:253-257`), so `acc = BusId::LfmPublic + index·α + Σ_l v_l·α^{2+l}`
  (`proof.rs:261-265`).
- ✓ VERIFIED `BusId::LfmPublic = 34` survives the merge with no collision from
  main's new tables (`prover/src/tables/types.rs:368-373`; the branch→main diff
  shows 32/33/34 are branch-only additions and main added no id in that range).

**A structural argument the plan does not make, stronger than the ones it does.**
`verify_against` forks the replay transcript *before* handing the same object to
the verifier:

```
prover/src/lfm/proof.rs:218   let mut replay = transcript.clone();
prover/src/lfm/proof.rs:219   let (z, alpha) = crate::replay_transcript_phase_a_view(&refs, view, &mut replay);
prover/src/lfm/proof.rs:242   Verifier::multi_verify_views(&refs, view, &mut transcript, &expected)
```

`multi_verify_views` re-runs the identical Phase A absorption
(`crypto/stark/src/verifier.rs:1279-1337`) on `transcript` and samples its own
`lookup_challenges` (`:1344-1350`), which flow into `verify_rounds_2_to_4`
(`:1410-1416`). Since those per-table checks **pass**, the challenges the
verifier used must equal the ones the prover used — a wrong `alpha` would break
the OOD composition check. Therefore the surviving degrees of freedom are the
target formula and its inputs, which is exactly where the plan aims Step 2. This
argument does not require trusting any diff.

---

## 2. The scope is right: main authoritative, the LFM binding adapts

**Claim.** Fixing only `prover/src/lfm/proof.rs` + `prover/src/lib.rs` and leaving
`crypto/stark` as main's is the correct direction.

**Evidence — a counting argument, not an inspection.**

1. ✓ VERIFIED the merge touched exactly **two** files under `prover/src/lfm/`
   (`git diff --stat ed1b7785 -- prover/src/lfm/`):
   `constraint_tests.rs` (6 lines — the `ArtifactNode` import move from the
   artifact reconciliation) and `proof.rs` (+18 — the `LFM_BUS_DEBUG` diagnostic
   block at `proof.rs:224-240`). Every other LFM file is byte-identical to
   `ed1b7785`, where all 20 of these tests passed.
2. ✓ VERIFIED main's LogUp math is byte-identical across all three revisions:
   `crypto/stark/src/lookup.rs` lines 1-833 — which contain
   `compute_alpha_powers` (`:73`) and all four `accumulate_fingerprint` /
   `accumulate_fingerprint_with` / `accumulate_fingerprint_from_step` impls
   (`:274, :377, :626, :742` in branch numbering) plus the packing shifts —
   hash to `29d4849da6bda633b2fe37235319e14a` on `ed1b7785`, on `origin/main`,
   and in the merged worktree. The slice from `fn compute_logup_term_column` to
   EOF (the fingerprint loop, multiplicities, `build_accumulated_column_from_terms`,
   the debug bus sums) hashes to `00bd7a930244cb5817270e5a6bf9f4f8` on all three.
3. ✓ VERIFIED `replay_transcript_phase_a_view` (`prover/src/lib.rs:989-1003`)
   hashes to `422b0f7436f50c57717f860a99d94930` on `ed1b7785`, `origin/main` and
   the worktree; `compute_commit_bus_offset` (`prover/src/lib.rs:947-984`) to
   `05fadc35e83bbdf0576d85af3881728c` on all three. Both are **main's own code on
   the VM's live verify path**, and main is green.
4. ✓ VERIFIED the only `crypto/stark` semantic change main brought to the
   verifier is #909's opening-width pin: the `ed1b7785 → origin/main` diff of
   `verifier.rs` is five hunks, all introducing `trace_opening_widths_well_formed`
   (`verifier.rs:199-263` post-merge) and its call site (`:1628-1640`) plus
   comments. Phase A, Phase B and the balance check carry **no hunk**.

So main's generic machinery is internally consistent and externally validated by
main's own CI; the only code that *mirrors* its convention from outside is the
LFM binding. Fixing the mirror is the only direction that does not fork
`crypto/stark` from main.

**Precedent.** This is the same call the artifact reconciliation already made
and validated: main's `DeviceProgram::lower` stayed authoritative and the
branch's serialization decoupled into `ArtifactNode`, with
`device_program()` re-deriving through main's `lower`
(`reconcile-report.md` §2). That reconciliation is green — round-trip 11/11,
`stark --lib constraint_ir` 39/0.

---

## 3. Step 1 (pin before fix) + Step 4 (negative gate) make this safe

`expected_public_balance` is the recursion verifier's only cross-table check:
the LfmPublic bus has no in-trace receiver, so the target *is* the binding
(`proof.rs:215-222`, and the balance check itself at `verifier.rs:1438-1459` is
the last gate before `return true`). A formula edit that "makes the 20 pass" by
weakening the target is a silent soundness break, and it would be invisible to
the rejection tests: `tampered_claimed_public_word_rejects`
(`prover/src/lfm/machine_tests.rs:52-63`) passes whether the binding is correct
*or* uniformly broken. Demanding both directions is therefore not ceremony, it
is the only gate design that discriminates.

The codebase already uses that idiom by name —
`prover/src/lfm/blake3_probe.rs:521 falsification_control_the_untampered_proof_verifies`
sits directly above five tamper tests. The plan is consistent with the house
rule recorded in memory ("every soundness fix needs a test asserting honest
proofs STILL verify").

✓ VERIFIED the plan's named control exists and is in the right module:
`tampered_l2g_binding_rejects` at `prover/src/lfm/machine_tests.rs:3593`.

Step 1's ordering is also right for a second reason the plan states in R2: if
the shift turns out to be in how contributions are *produced* rather than in the
target, the fix site changes entirely. Editing first and measuring second would
mean editing `expected_public_balance` — the soundness-critical function — on a
guess.

---

## 4. Rollback bounds the downside

✓ VERIFIED:

- `blake3-campaign-preMerge` → `ed1b7785964568d237567dd0ee83162e9db87d58`.
- `blake3-real-hash` still at `ed1b7785` in its own worktree
  (`/Users/maurofab/workspace/lambda_vm-blake3-impl`); the merge lives only in
  `/Users/maurofab/workspace/lambda_vm-blake3-merge` on the throwaway branch
  `blake3-real-hash-mainmerge`.
- `MERGE_HEAD` present — nothing is committed. PR #930 cannot move until the
  deliberate fast-forward in Step 5.

Two corrections to that section are in §6.9.

---

## 5. Summary of the defense

| plan claim | status |
|---|---|
| Failure is the cross-table balance; per-table STARK is correct | ✓ VERIFIED (per-check instrumentation) |
| Challenge derivation unchanged | ✓ VERIFIED (byte-identical replay + no verifier hunk) |
| LogUp L-value math unchanged | ✓ VERIFIED (two region hashes across three revisions) |
| Fix belongs on the LFM side, not in `crypto/stark` | ✓ sound (counting argument, §2) |
| Negative controls are mandatory | ✓ sound, and the named one exists (`machine_tests.rs:3593`) |
| Rollback is bounded | ✓ VERIFIED, minus the "pushed" claim (§6.9) |
| "The convention shifted"; alpha-power offset is the leading suspect | ✗ **ruled out by diff** (§6.1) — redirect Step 1 |
| All 20 share one root cause | ? unverified assumption (§6.4) |
| Step 3 is trivial bookkeeping caused by HINT | ✗ **wrong cause** for the budget half (§6.5) |

---

## 6. Concrete strengthenings

### 6.1 Retire Outcomes A and B — they are dead by diff

Per §2 items 2-4: neither the challenge derivation nor the LogUp column
construction changed between `ed1b7785` and `origin/main`. That includes the
plan's stated **leading suspect** — the alpha-power offset of the LfmPublic
sender token. `accumulate_fingerprint` is byte-identical, so the slot layout
(`bus_id` at α⁰, values from α¹ upward, `lookup.rs:1759-1774`) cannot have
shifted. Step 1 should not spend a build on Outcome A or on the alpha-offset
hypothesis. What remains is Outcome C (the target's *inputs*, not its shape) and
R2 (main's prover changed what goes into L).

### 6.2 The discriminating control the plan is missing — re-aim Step 1 at it

? INFERRED, from `reconcile-report.md` §5's failure list (a measurement I did
not re-run): **`trivial_program_proves_and_verifies`
(`prover/src/lfm/machine_tests.rs:36-49`) passes in the merged tree.** It is not
among the 22 new failures, it needs no ELF fixture (so it cannot be one of the
19 pre-existing `recursion/fibonacci.elf` failures), and ✓ VERIFIED it is not
`#[ignore]`d.

It exercises the *identical* binding end to end — `lfm_prove` → `lfm_verify`
→ `resolve` → `verify_against` → `replay_transcript_phase_a_view` +
`expected_public_balance` + `multi_verify_views` — and its public word vector is
non-empty (✓ VERIFIED: `machine_tests.rs:59` indexes `claimed[0].1[0]`, and
`machine_tests.rs:52-63` asserts a tamper on it rejects).

If that holds, **"the merge shifted the convention and the hand-rolled mirror is
stale" is false as a blanket statement**, and the divergence is
*program-dependent*. Both sides use the same 14 chips — ✓ VERIFIED
`NUM_LFM_CHIPS = 14` (`prover/src/lfm/airs.rs:50`), all registry entries carry
`keccak_rnd_chunks: 1` (`prover/src/lfm/registry.rs:270,354,438,522`), and the
plan itself reports 14 tables for the failing `machine_proves_the_sample_replay`
— so chunk count and table count are *not* the difference. Trace content and
height are, which moves the leading suspect to main's prover rewrite
(#877/#875/#863: padding, aux-build path selection, `resident_aux_ok`,
row counts) or to a program-shape assumption baked into the LFM programs.

**Action.** Replace Step 1's cross-worktree comparison with a *within-tree*
differential: instrument once, run `trivial_program_proves_and_verifies` (passes)
and `machine_proves_the_sample_replay` (`machine_tests.rs:916-949`, fails)
side by side, and diff `z`, `alpha`, the 14 per-table contributions and
`expected`. One build instead of two, no cross-worktree fixture skew, and it
isolates *what about the bigger program* matters — which the cross-tree
comparison cannot tell you.

Confirm the premise first, it is one command:

```
cargo test --release -p lambda-vm-prover --lib \
  lfm::machine_tests::trivial_program_proves_and_verifies -- --nocapture
```

If that test in fact fails, §6.2 collapses and the plan's original Step 1 stands
unchanged — so this costs nothing to check and settles the whole framing.

### 6.3 Falsify the scope claim for free, before editing anything

The tree already ships per-bus attribution: `crypto/stark/src/bus_debug.rs`
(`log_interaction` `:223`, `analyze_mismatches` `:105`, `print_summary` `:264`,
env selector `DEBUG_BUS_ID=`), fed by
`per_bus_sums` / `per_bus_sender_sums` / `per_bus_receiver_sums` populated at
`crypto/stark/src/lookup.rs:1346-1371` and exposed on `BusPublicInputs`
(`lookup.rs:1642-1655`). All of it is behind `--features debug-checks`
(`bus_debug.rs:8-14`).

One run tells you **which bus** is unbalanced: `LfmPublic` (34), `LfmMem` (32) or
`LfmRange` (33). If it is either of the latter two — internal buses that must net
to zero in trace — then `expected_public_balance` is *innocent* and Step 2's
scope is wrong. This is the single cheapest falsification of the plan's core
hypothesis, it requires no new code, and it should run before Step 1's bespoke
`eprintln`s.

### 6.4 Verify "all 20 share one root cause" instead of assuming it

The plan's §2 evidence comes from instrumenting one test
(`machine_proves_the_sample_replay`) but its opening asserts all 20 fail "on one
root cause". The `DBG909` prints are unconditional, so a single run groups all
20 by which check fired:

```
cargo test --release -p lambda-vm-prover --lib lfm:: -- --nocapture 2>&1 | grep -E "DBG909|W909"
```

Two of the 20 look like a *different* cause:
`machine_tests::program_id_matches_production_on_the_real_fixture` and
`machine_tests::program_id_folds_pages_in_the_production_layout` are about the
**production** table layout, which main moved (HINT added,
`NUM_PRODUCTION_AIRS` 28→29 per `PLAN.md` step 5). If those two are a separate
item, R4's mitigation ("run the full 20") would report a partial fix as a
regression and cost a debugging cycle.

### 6.5 Step 3 is not trivial, and it should run FIRST as a diagnostic

The plan calls the two census failures "unrelated bookkeeping" with a single
stale constant. ✓ VERIFIED that is wrong for the budget half:

`continuation_epoch_constraint_leg_cost` computes
`design_intermediate = families_unfused + fixed_unfused + l2g_unfused`
(`prover/src/lfm/constraint_tests.rs:1547`), summing over
`SPLIT_FAMILIES` — 14 labels, `constraint_tests.rs:1491-1494` — plus
`FIXED[..9]` — `constraint_tests.rs:1497-1508` — plus `L2G_MEMORY`.
**HINT appears in neither list.** So the observed −1018 delta (computed 62,375
vs the pinned `63_393` at `constraint_tests.rs:1566-1569`) cannot come from HINT
being added; it comes from one or more of those 24 existing tables' constraint
counts moving on main. Re-pinning the constant to 62,375 without attributing the
delta destroys exactly the signal the test claims for itself:

```
prover/src/lfm/constraint_tests.rs:1563-1565
// The design's §8.2.2 arithmetic, reproduced from the emitter's own unfused
// counts. A mismatch means the epoch composition changed, which is a finding
// about the epoch, not about this pass.
```

Related: `constraint_leg_instruction_census` **panics** at
`constraint_tests.rs:438-442` (`no design entry for {label}`) *before* reaching
its post-loop `mismatches` assertion (accumulated at `:447-452`). So adding the
HINT row to `DESIGN_INSTR` will very likely surface a per-table design-vs-emitter
mismatch list — and that list is precisely the attribution the budget delta
needs, and may name the production tables whose shape changed. Given #4's
suspicion about the `program_id_*` failures, that is a plausible common thread.

**Action.** Move Step 3 ahead of Step 2 (it is a read-only census; it does not
touch the binding), record the per-table deltas in the merge notes, and pin the
new budget with the attribution written down rather than as a bare number.

### 6.6 Name the Step-4 controls, in both directions

Step 4 says "all 20 pass" and "the existing tamper tests still REJECT". Make it
an explicit list so the gate is checkable by someone who did not write it.

Honest-path (must stay green — these catch an over-broad fix):

- `prover/src/lfm/machine_tests.rs:36` `trivial_program_proves_and_verifies`
- `prover/src/lfm/machine_tests.rs:66` `different_arena_values_change_the_public_output_not_the_program` — its `:81-83` assertion is honest-path
- `prover/src/lfm/blake3_probe.rs:521` `falsification_control_the_untampered_proof_verifies`

Rejection (must stay red for the prover):

- `prover/src/lfm/machine_tests.rs:52` `tampered_claimed_public_word_rejects`
- `prover/src/lfm/machine_tests.rs:85-87` the cross-claim assertion inside `different_arena_values_…`
- `prover/src/lfm/machine_tests.rs:3593` `tampered_l2g_binding_rejects`
- `prover/src/lfm/framework_probe.rs:188` `b0_tampered_witness_value_breaks_balance`
- `prover/src/lfm/framework_probe.rs:169` `b0_verifier_rejects_wrong_preprocessed_root`
- `prover/src/lfm/framework_probe.rs:156` `b0_prover_rejects_mismatched_preprocessed_root`
- `prover/src/lfm/keccak_probe.rs:219,228,237` (tampered output byte / input byte / padding-row multiplicity)
- `prover/src/lfm/blake3_probe.rs:529,542,556,564,575`
- `prover/src/lfm/logup_tests.rs:478` `the_closure_cannot_sum_a_contribution_the_constraints_rejected`

### 6.7 Add the one control that does not exist yet

Every current tamper mutates a public **value** (`machine_tests.rs:59`:
`claimed[0].1[0] += 1`). A binding that dropped or shifted the `index·α` term at
`prover/src/lfm/proof.rs:261` would still reject all of those — but would accept
a **permutation of the public words** (same lanes, swapped indices). That is
precisely the error class Step 2 is most likely to introduce, and nothing in the
suite catches it.

Add to `tampered_claimed_public_word_rejects`, or as a sibling: build `claimed`
by swapping the `index` fields of two entries (or reversing the vector) with all
lane values untouched, and assert `lfm_verify` returns `false`. Cheap, and it is
the control that makes an alpha-offset regression impossible to land.

### 6.8 Make the Step-4 cleanup gate mechanical, not a judgement call

Step 4 says "confirm `git diff` under `crypto/stark/` is only the intended merge
content". That is unfalsifiable as written. It can be made exact:

✓ VERIFIED the pre-merge branch made **zero** changes to
`crypto/stark/src/verifier.rs` (the `ed1b7785 → origin/main` diff of that file is
five hunks, all main-side additions). ✓ VERIFIED the worktree's current 48-line
delta vs `origin/main` in that file is **entirely diagnostics** — the `W909_DEBUG`
block plus the `DBG909` `eprintln`s — and it includes a **structural rewrite of a
soundness check**: `trace_opening_widths_well_formed`'s body was changed from
`(0..num_queries).all(…)` into `let ok = (0..num_queries).all(…); …; ok`.

So the gate is: **`git diff origin/main -- crypto/stark/src/verifier.rs` must be
empty.** And `git diff --stat origin/main -- crypto/stark/` must reduce to
exactly this allow-list (current state, minus the verifier line):

```
crypto/stark/src/constraint_ir/artifact.rs        771 ++  (new — artifact feature)
crypto/stark/src/constraint_ir/artifact_tests.rs  394 ++  (new)
crypto/stark/src/constraint_ir/mod.rs               8 +   (re-exports incl. ArtifactNode)
crypto/stark/src/constraint_ir/device.rs            2 +-  (one derive: PartialEq, Eq)
crypto/stark/src/constraints/builder.rs             2 +-  (one derive: rkyv Archive/Serialize/Deserialize)
crypto/stark/src/lookup.rs                         67 +-  (precaptured_program re-addition)
crypto/stark/src/traits.rs                         43 +-  (precaptured_constraint_program re-addition)
crypto/stark/src/verifier.rs                        0     (MUST be empty after cleanup)
```

Note this also corrects the plan's wording: "No edits under `crypto/stark/`" is
already literally false — the merge deliberately re-adds `with_precaptured` /
`precaptured_constraint_program`, which main deleted (`lookup.rs:968-1007` and
`traits.rs:270-317` on `ed1b7785`; removed on main). State the Step-2 boundary as
*this allow-list*, not as a directory.

The `LFM_BUS_DEBUG` block at `prover/src/lfm/proof.rs:224-240` must go too.

### 6.9 Two fixes to the rollback section

- ✗ The tag is **not pushed**. `git ls-remote --tags origin` returns nothing for
  `blake3-campaign-preMerge`; the plan asserts "tagged and pushed". The pristine
  tip is separately recoverable via the `blake3-real-hash` remote branch (PR
  #930), so the exposure is small, but the claim is untrue and the fix is one
  command: `git push origin blake3-campaign-preMerge`.
- "Uncommitted" is presented purely as a safety property, but it is also the
  risk: a large conflict resolution (`prover.rs` alone is 1,725 changed lines
  between the two sides) exists only in one worktree's index, with no recovery
  point. `blake3-real-hash-mainmerge` is a disposable branch — commit the
  resolved merge **now** as a checkpoint and do the fix as follow-up commits;
  Step 5's "commit the merge" becomes a squash before the fast-forward. That
  keeps the bounded downside *and* adds a checkpoint, instead of trading one for
  the other.

---

## 7. What would change my position

If §6.3 shows the unbalanced bus is `LfmMem` or `LfmRange` rather than
`LfmPublic`, Step 2's scope (`expected_public_balance` / the replay) is wrong and
the plan needs a new Step 2 — the internal buses balance in-trace, so a mismatch
there is a trace/prover-side finding, not a target-formula one. Everything else
in the plan (ordering, gates, rollback, the "main stays authoritative" call)
survives that outcome unchanged, which is itself an argument that the plan's
skeleton is the right one.
