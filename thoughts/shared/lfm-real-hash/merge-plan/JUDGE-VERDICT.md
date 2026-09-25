# JUDGE VERDICT — FIX-PLAN.md adversarial review

**Ruling: REVISE, THEN IMPLEMENT.** The plan's process skeleton survives; its diagnosis and its
Step-2 scope do not. The attacker is right on the root cause, and I verified it independently.
The corrected fix is mechanical and fail-closed and needs no further human review pass — but two
carve-outs do, and they are named in §7.

All code citations are from the merge worktree `/Users/maurofab/workspace/lambda_vm-blake3-merge`
(branch `blake3-real-hash-mainmerge`, `HEAD = ed1b7785`, `MERGE_HEAD = 58160b6f`, merge
uncommitted). `origin/main` = `528a8411`. Everything marked ✓ VERIFIED I read out of the tree or
the diff myself; I ran no cargo and made no edits.

---

## 1. Ruling on the central dispute

**The attacker's diagnosis is correct. The plan's is falsified.**

The root cause is a **sender/receiver multiplicity mismatch on `BusId::Hwsl` (id 9)** inside the
LFM machine, created by the merge: main deleted the HWSL sends from the production `KECCAK_RND`
chip *and* from the production receiver-side collector, but the LFM machine's **forked** receiver
-side collector — branch-only code the merge never touched — still emits them.

### Fix location

```
prover/src/lfm/keccak_adapter.rs:361-366   // theta HWSL push  (20 per round)
prover/src/lfm/keccak_adapter.rs:441-446   // rho   HWSL push  (100 per round)
prover/src/lfm/keccak_adapter.rs:306,319   // the stale pinned count `24 * 1148`
prover/src/lfm/keccak_probe.rs:201-205     // the same count, asserted
```

### The verification chain, link by link

1. ✓ **Main removed the sends.** `git diff ed1b7785 origin/main -- prover/src/tables/keccak_rnd.rs`
   filtered to `BusId::` / section comments yields exactly two removals and nothing else:
   `--- Theta: HWSL for rotated C (20) ---` (hunk `@@ -587,48 +596,8 @@`) and
   `--- Rho: HWSL (100) ---` (hunk `@@ -717,53 +686,8 @@`), each replaced by a comment saying the
   shift is now enforced by an inline μ-gated linear identity in `KeccakRndConstraints`. Capacity
   `1371 → 1031` (`keccak_rnd.rs:446`), and the new module comment at `keccak_rnd.rs:439` states
   *"The θ/ρ halfword shifts no longer emit HWSL lookups (120 sends/row removed) … The matching
   HWSL multiplicities are likewise dropped on the BITWISE side (`collect_bitwise_from_keccak`)."*
   The surrounding `AreBytes` blocks are **unchanged** on both sides — the delta is exactly the
   120 HWSL sends per row, nothing more.

2. ✓ **Main removed the matching receives, in the production collector.**
   `git show origin/main:prover/src/tables/trace_builder.rs` has **zero** occurrences of
   `BitwiseOperationType::Hwsl`. `git show ed1b7785:…` has two, at `:2427` (theta) and `:2510`
   (rho), inside `collect_bitwise_from_keccak` (`:2343` branch / `:2447` main). Main changed both
   sides in lockstep. The production path is self-consistent.

3. ✓ **The LFM machine takes main's sender side automatically.**
   `prover/src/lfm/airs.rs:21` imports `crate::tables::{bitwise, keccak_rc, keccak_rnd}`;
   `airs.rs:239` reads `keccak_rnd::bus_interactions().len()`; `airs.rs:488-494` builds the LFM
   `KECCAK_RND` AIRs from `keccak_rnd::bus_interactions()` and `keccak_rnd::KeccakRndConstraints`.
   The trace comes from main's own generator (`prover/src/lfm/trace.rs:166`
   `.map(keccak_rnd::generate_keccak_rnd_trace)`), which is why per-table STARK verification still
   passes — the trace does satisfy main's new inline identities.

4. ✓ **The LFM machine does NOT take main's receiver side.**
   `prover/src/lfm/trace.rs:176` feeds the BITWISE multiplicity histogram from
   `keccak_adapter::bitwise_ops_for(&keccak_ops)` — the branch's fork, documented as such at
   `keccak_adapter.rs:306-315` (*"the per-round half of `trace_builder::collect_bitwise_from_keccak`,
   forked rather than called"*). ✓ `git diff ed1b7785 -- prover/src/lfm/keccak_adapter.rs` is
   **empty**: the merge did not touch it. It still pushes `BitwiseOperationType::Hwsl` at `:362`
   (5×4 = 20/round) and `:442` (5×5×4 = 100/round).

5. ✓ **Those two sites are the only HWSL in the whole LFM module.**
   `grep -rn "Hwsl" prover/src/lfm/` returns exactly `keccak_adapter.rs:362` and `:442`. So in the
   LFM AIR set the Hwsl bus now has **receivers with no senders at all** — a pure one-sided
   imbalance, not a subtle re-weighting.

6. ✓ **The count arithmetic closes.** Hand-counting the pushes per round in `bitwise_ops_for`:
   theta XOR chain 160 + theta (20 HWSL + 20 AreBytes) + theta Dxz 40 + theta final 200 +
   rho (100 HWSL + 200 AreBytes) + chi 400 + iota 8 = **1148**, matching the pin at `:319`.
   Removing the 120 HWSL gives **1028**.

7. ✓ **No other embedded table drifted.** `prover/src/tables/bitwise.rs` and
   `prover/src/tables/keccak_rc.rs` are byte-identical branch↔main; `keccak.rs` differs by one
   `#[derive(Clone, Copy)]`; `types.rs` differs only by the branch's own `LfmMem = 32` /
   `LfmRange = 33` / `LfmPublic = 34` additions, which survive the merge with no collision
   (`prover/src/tables/types.rs:363-373`; `Hwsl = 9` at `:283`).

### The passing/failing split matches this theory and nothing else

✓ `keccak_ops` is derived **only** from `records.keccak` (`prover/src/lfm/trace.rs:145-155`), i.e.
from explicit `Instr::KeccakF` rows — never from the hash chip. ✓ `HasherKind::default() = Test`
(`prover/src/lfm/hash.rs:196-199`), and `build_artifacts` uses the default
(`prover/src/lfm/registry.rs:117-119`). ✓ Every `KECCAK_RND` bus interaction is gated
`Multiplicity::Column(cols::MU)` (`keccak_rnd.rs:446ff`), so padding rows send nothing.

Therefore a program with **zero keccak permutations** feeds `bitwise_ops_for(&[])` → no HWSL
receives → balanced; a program with **any** keccak permutation is unbalanced. That is exactly the
observed split:

- `trivial_program_source` (`prover/src/lfm/programs.rs:31-79`) uses `b.compress` (the hash chip
  under `TestPermutation`) and **no** `keccak_f`/absorb → passes. So do the BLAKE3 suites.
- Every one of the 20 failures is keccak-touching: the keccak_* / sponge / chain / merkle-walk
  tests obviously; `splice`/`append_ext`/`transcript_replay`/`statement_replay` are keccak
  absorbs; `fri_tests` goes through `edsl::keccak_merkle_walk` (`prover/src/lfm/fri.rs:564`);
  `join_tests` through `prover/src/lfm/sub_proof.rs:256,268,289`
  (`edsl::keccak_leaf_hash` / `keccak256` / `keccak_merkle_walk`); `program_id_*` are keccak folds.

---

## 2. Why the plan's diagnosis is dead

✓ The plan's premise — *"the merge's large crypto/stark batch shifted [the LogUp] convention"* —
is refuted by the diff. `git diff ed1b7785 origin/main -- crypto/stark/src/lookup.rs` is 67+/79−
and its **first hunk starts at line 834**. `compute_alpha_powers` (`:73`), every
`accumulate_fingerprint*` impl (`:274`, `:377`, `:626`, `:742`), `add_combined_terms` and the whole
alpha-offset assignment are all **above** the first hunk and therefore untouched. The later hunks
(`@@ -1134`, `-1177`, `-1210`, `-1223`, `-1242`, `-1270`, `-1278`, `-1299`, `-1372`) are: `Arc`-wrap
of `constraint_program`, removal of the branch's `precaptured_program`, a `OnceCell` for lazily
materializing host main columns on the GPU-resident aux path, and a `#[derive(Clone)]`. **No value
semantics.** Defenders A and B reached the same conclusion by region hashing; I confirmed it by
hunk boundaries. Step 1's "leading suspect" (a shifted alpha-power offset) is dead.

---

## 3. Ruling on the soundness-regression claim (attacker Finding 2) — UPHELD, and stronger

The attacker says patching `expected_public_balance` to match would fold an unmatched-bus residual
into the verifier target and permanently blind the cross-table check, with every named control
staying green. **I uphold that, and I find the situation is worse than stated.**

`expected_public_balance` (`prover/src/lfm/proof.rs:247-276`) is a pure function of
`(claimed_public, z, alpha)`. The Hwsl residual is `Σ over the trace's HWSL lookup multiset of
mult/(z − fingerprint)` — a function of the *keccak trace contents*, which the verifier does not
have and which differs per program and per input. So **no formula change to
`expected_public_balance` can compensate for it.** The only edits that would turn the 20 tests
green are the degenerate ones Defender B enumerates: return a constant, drop the dependence on
`claimed_public`, or derive the target from the proof's own `bus_table_contribution()` values. The
last of those is the one that "works," and it is a total soundness break — `expected_public_balance`
is the recursion verifier's only cross-table binding, since LfmPublic has no in-trace receiver
(`proof.rs:215-222`).

So Step 2 is not merely aimed at the wrong file. **Executed as written with "make the 20 pass" as
the acceptance criterion, it has exactly one reachable answer, and that answer is catastrophic.**
R1 names this risk and Step 4's controls cannot see it: the residual is independent of
`claimed_public`, so `tampered_claimed_public_word_rejects` still rejects. The plan's headline
mitigation does not mitigate its headline risk.

The same reasoning kills `replay_transcript_phase_a_view` as a target: changing it moves `z`/`α`,
which would break the per-table OOD composition checks — and those **pass**. Nothing in the
verifier binding can be the cause.

---

## 4. Ruling on "20 failures = one root cause" — FALSE, as both the attacker and Defender B argued

✓ **`keccak_probe::adapter_probe_proves_real_permutations` and
`keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard` cannot be touched by the
plan's fix at all.** `keccak_probe.rs:126-143` verifies with a hardcoded `&FEE::zero()` expected
balance through `Verifier::multi_verify_views` on the **production** AIRs plus a local adapter. It
calls neither `verify_against` nor `lfm_verify` nor `expected_public_balance` nor
`replay_transcript_phase_a_view`. Both tests **are** explained by the HWSL mismatch
(`:211 round_trip(|_|{}) == Ok(true)`; `:285 assert!(verify_proof(...))`).

⚠️ **`keccak_probe::duplicate_tag_output_swap_accepts_demonstrating_hazard` is not a negative
control.** `keccak_probe.rs:284-289` asserts the proof **verifies**, documenting an open
tag-uniqueness hazard. Step 4's instruction that "the output-swap-hazard tests" must "still REJECT"
would invert its meaning. Both defenders flagged this; it is correct and it must be struck.

✓ **`machine_tests::preprocessed_tags_close_the_output_swap_hazard` is not explained by either
theory.** `machine_tests.rs:441-444` is its only verify assertion and it is *negative*
(`assert!(!lfm_verify(...))`) — a universally-rejecting verifier **satisfies** it. Its remaining
failure modes are all prove/compile-side: `assert_ne!(tag(0), tag(1))` at `:429`,
`prove_keccak_chain_with_tamper(...).expect("locally consistent")` at `:440` (proving must
*succeed*), and `.expect_err(...)` + `matches!(err, ProvingError::PrecomputedCommitmentMismatch)`
at `:455-459`. Main rewrote the prover's preprocessed / split-tree commit path, which is the
plausible independent cause. Nobody has read this test's actual failure message.

✓ **The two census failures are a third bucket, prover-side, and one of them must not be
re-blessed.** `constraint_leg_instruction_census` dies at `constraint_tests.rs:438-442`
(`panic!("no design entry for {label}")`) on main's new HINT table — that half is genuinely
bookkeeping. But `continuation_epoch_constraint_leg_cost` is not: ✓ HINT appears in **neither**
`SPLIT_FAMILIES` (`constraint_tests.rs:1491-1494`) nor `FIXED` (`:1497-1508`), so the −1018 delta
(62 375 vs the pinned `63_393` at `:1566-1569`) **cannot** come from HINT. Defender A is right that
the plan misattributes it. I add the likely true attribution: **`KECCAK_RND` is in `FIXED`
(`:1502`), and main's HWSL→inline-identity swap is precisely a change to `KeccakRndConstraints`
(the `@@ -900,26 +824,99 @@` hunk).** That is the same main change as the root cause, and it should
be checked first. The test's own doc comment (`:1563-1565`) says a mismatch "is a finding about the
epoch, not about this pass" — so the number must be attributed, not pasted.

⚠️ **`program_id_matches_production_on_the_real_fixture` and
`program_id_folds_pages_in_the_production_layout` each carry a digest `assert_eq!` before their
`verify_against`** (`machine_tests.rs:3722-3726` then `:3728-3738`). Both are keccak folds so HWSL
explains them, but which assertion fires is unknown. The first also depends on
`proof_fixture::load_or_generate(&fixture_cache())` (`machine_tests.rs:3690`) and hard-asserts
`pages.is_empty()` at `:3709-3713`, so it is fixture-sensitive — the attacker's baseline-
comparability caution (Finding 11) is legitimate.

**Conclusion: at least four buckets, not one.** Bucketing by actual panic message is mandatory and
costs one test run.

---

## 5. Is the corrected fix mechanical enough to implement directly? YES

I rule that the `keccak_adapter` reconciliation may be implemented **directly, without a further
human review pass**, for four reasons I verified:

1. **It is a deletion of two 6-line push blocks plus three number updates.** No new logic.
2. **It is fail-closed.** The edit changes what multiplicities the *prover* claims BITWISE was
   looked up for. Get it wrong in either direction and the bus fails to balance and the proof is
   **rejected**. Unlike the plan's Step 2, there is no way for this edit to make the verifier
   accept more than it should — it cannot weaken a check, because it is not on a check.
3. **Its blast radius does not reach any pinned identity.** ✓ BITWISE is not among the 11 committed
   groups in `LfmArtifacts` (`prover/src/lfm/registry.rs:133-152`: const_, balu, xalu, select,
   bitdec, hash, keccak, lanes, hint, public, range; slot 11 is the KECCAK_RND sentinel), and
   multiplicities live in the main trace, not in `bitwise::NUM_PRECOMPUTED_COLS = 11`
   (`prover/src/tables/bitwise.rs:101`). So `artifacts.roots`, `program_id`, the `registry_drift_*`
   tests and the in-circuit `statement_replay.rs:164-190` mirror are all **untouched**. The
   attacker's Finding 6 (registry drift / in-circuit mirror) is a real risk *for the plan's fix*
   and a non-risk for the correct one.
4. **It restores an invariant that has an external oracle** — main's own
   `collect_bitwise_from_keccak`, which the fork's doc comment already names as its source. The
   correct post-state is not a judgement call; it is "the fork agrees with its documented origin
   again."

Contrast with the plan's proposed fix, which would edit the recursion verifier's only cross-table
soundness check on a false diagnosis. That is the difference between the two verdicts.

---

## 6. Which debate strengthenings are adopted

**Adopted (mandatory).** Per-bus residual measurement before any edit (attacker §1; Defender A
§6.3) — ✓ the machinery exists: `crypto/stark/src/bus_debug.rs` behind the `debug-checks` feature
(`crypto/stark/Cargo.toml`, `prover/Cargo.toml`), runtime filter `DEBUG_BUS_ID` at
`bus_debug.rs:66`, and `BusId::Hwsl = 9`. Note: the `DEBUG_BUS_TRACKER=1` form recorded in project
memory does not appear in `bus_debug.rs`; use the feature + `DEBUG_BUS_ID`. Bucket all 22 by panic
message (attacker §3; Defender B S3; Defender A §6.4). Correct negative controls, run positive and
negative in the same run (Defender B S2; attacker §5). Index-permutation control (Defender A §6.7).
`git add` the five unmerged paths (attacker §9) — ✓ confirmed still unmerged:
`crypto/stark/src/lookup.rs`, `prover/src/continuation.rs`,
`prover/src/tests/constraint_program_{device_,}tests.rs`, `prover/src/tests/ood_window_ir_tests.rs`;
until they are added, `git diff -- crypto/stark/` prints `lookup.rs | Unmerged` and inspects
nothing. Empty-diff cleanup gate on `verifier.rs` (Defender B S7; Defender A §6.8) — ✓ confirmed
all ~50 changed lines are diagnostics, and ✓ the `DBG909` insert at `verifier.rs:1447-1450` **stole
the `#[cfg(not(feature = "test_fiat_shamir"))]` attribute** from `error!`, plus de-indent damage at
`:1311`, `:1315`, `:1363`. Push the tag before the branch push (Defender B S1; Defender A §6.9).
Separate commit for the HINT/census items (Defender B S4). Attribute the −1018 before re-pinning
(Defender A §6.5, Defender B S4).

**Adopted (should).** `cargo test -p stark` including main's new
`opening_width_tests.rs` / `aux_opening_width_tests.rs` (attacker §8). Checkpoint-commit the
resolved merge rather than leaving 1 700+ conflict-resolved lines in one index (Defender A §6.9).
Baseline-comparability check on fixture state (attacker §11).

**Rejected.** Defender A §6.1's conclusion that "what remains is Outcome C (the target's inputs)"
— Outcome C is dead too; the answer is the trace side, which Step 2 could not express. Defender A
§6.2/Defender B S5's same-tree A/B as the *opening* move — it is a good experiment but the per-bus
dump is strictly more decisive and equally cheap, so it goes first. Step 4's framing of the
output-swap-hazard tests as negative controls — struck outright (see §4).

**Both defenders deserve credit for conceding the load-bearing points** (Outcomes A and B dead by
diff; wrong primary control; single-root-cause unverified; `replay_transcript_phase_a_view` reaches
the production VM verifier at `prover/src/lib.rs:1442` and `prover/src/continuation.rs:896`, while
`expected_public_balance` has exactly one caller at `proof.rs:220` — ✓ both verified). Their
defense of the plan's *shape* stands. Their defense of its *content* does not survive the
`keccak_adapter` finding, which neither of them located.

---

## 7. The corrected plan

### Step 0 — CONFIRM the diagnosis (no edits)

1. Per-bus residual on one failing test:
   `DEBUG_BUS_ID=9 cargo test --release -p lambda-vm-prover --features debug-checks --lib
   lfm::machine_tests::machine_proves_the_sample_replay -- --nocapture`.
   **Expect the residual on `Hwsl` (9), receiver-side, zero senders.** If it lands on `LfmPublic`
   (34), `LfmMem` (32) or `LfmRange` (33) instead, **stop** and re-open this verdict.
2. Bucket all 22 by actual panic message:
   `cargo test --release -p lambda-vm-prover --lib lfm:: -- --nocapture 2>&1 | grep -B2 -A5 panicked`.
   Record the bucket for each. Expect ≥4 buckets (§4).

### Step 1 — FIX the fork

Delete the two HWSL pushes: `prover/src/lfm/keccak_adapter.rs:361-366` (theta) and `:441-446`
(rho). Update the pinned per-round count `1148 → 1028` at `keccak_adapter.rs:306` (doc), `:319`
(capacity), and `keccak_probe.rs:201-205` (assertion). Refresh the doc comment at
`keccak_adapter.rs:306-315` to record that main dropped the θ/ρ HWSL lookups in favour of inline
μ-gated identities, so the fork is again the per-round half of main's collector.

**Prohibited without new evidence and an explicit escalation:** any edit to
`expected_public_balance`, `replay_transcript_phase_a_view`, `compute_expected_commit_bus_balance_view`,
`absorb_lfm_statement`, or anything under `crypto/stark/`. The scope rule is Defender B's, and it
is better than a directory boundary: **no edit whose blast radius reaches `prover/src/lib.rs:1442`.**

### Step 2 — CONTROLS (positive and negative in the same run)

Honest-path, must be GREEN: `machine_tests.rs:36 trivial_program_proves_and_verifies`;
`machine_tests.rs:66 different_arena_values_change_the_public_output_not_the_program` (carries
positive `:81-83`, cross-claim negative `:85-87`, and distinctness `:80` in one body — the single
best gate); `blake3_probe.rs:521 falsification_control_the_untampered_proof_verifies`.

Rejection, must stay RED for the prover: `machine_tests.rs:52 tampered_claimed_public_word_rejects`
(primary); `machine_tests.rs:2248 tampered_statement_or_root_rejects`;
`framework_probe.rs:156,169,188`; `keccak_probe.rs:219,228,237`;
`blake3_probe.rs:529,542,556,564,575`;
`logup_tests.rs:478 the_closure_cannot_sum_a_contribution_the_constraints_rejected`.

Must ACCEPT (not a negative control):
`keccak_probe.rs:261 duplicate_tag_output_swap_accepts_demonstrating_hazard`.

Executor-level only, keep but do not treat as the binding's gate:
`machine_tests.rs:3593 tampered_l2g_binding_rejects` (its first vectors reject inside
`super::executor::execute`; only the coherent-swap leg at `:3647` reaches `verify_against`).

New control to add: an **index-permutation** vector — build `claimed` by swapping two entries'
`index` fields with all lane values untouched, assert `lfm_verify` returns `false`. Nothing in the
suite currently covers the `index·α` term (`proof.rs:261`).

### Step 3 — HINT / census, in a SEPARATE commit

Add the HINT row to `DESIGN_INSTR`; expect the census to then print a per-table design-vs-emitter
mismatch list (`constraint_tests.rs:447-452`) — that list is the attribution the budget delta
needs. **Do not re-pin `63_393` until the −1018 is attributed.** First hypothesis to test:
`KECCAK_RND`, whose constraint set main rewrote in the same change as the root cause.

### Step 4 — VALIDATE

- `lfm::` failure set equals the `blake3-campaign-preMerge` baseline set (306/19, the
  `recursion/fibonacci.elf` fixture failures) — set equality, not counts. Confirm the fixture state
  matches the baseline worktree before comparing.
- `cargo test -p stark`, explicitly including `opening_width_tests` and `aux_opening_width_tests`.
- The main VM and continuation verify paths are untouched by this fix; if that ever stops being
  true, `prover/src/tests/prove_elfs_tests.rs` and the continuation suite join the gate.
- `constraint_artifact` 11/11 and `artifact_pin.py --check` remain as **regression-only** guards;
  they do not exercise `keccak_adapter.rs`, `proof.rs` or `lib.rs` and must not be cited as gates
  for this change.

### Step 5 — CLEANUP (mechanical, not eyeball)

1. `git add` the five unmerged paths first, or the crypto/stark diff check inspects nothing.
2. `git diff origin/main -- crypto/stark/src/verifier.rs` must be **exactly empty** — including
   reverting `let ok = (0..num_queries).all(…); …; ok` back to the direct return, and restoring
   `#[cfg(not(feature = "test_fiat_shamir"))]` to its `error!`.
3. `git diff --name-only origin/main -- crypto/stark/` must reduce to exactly the seven artifact-
   feature files (`constraint_ir/artifact.rs`, `constraint_ir/artifact_tests.rs`,
   `constraint_ir/mod.rs`, `constraint_ir/device.rs`, `constraints/builder.rs`, `lookup.rs`,
   `traits.rs`). Note the plan's "No edits under `crypto/stark/`" is already false: the merge
   deliberately re-adds `with_precaptured` / `precaptured_constraint_program`, which main deleted.
4. Remove the `LFM_BUS_DEBUG` block at `prover/src/lfm/proof.rs:224-240`. Campaign diagnostics are
   identifiable by the markers `W909_DEBUG`, `DBG909`, `LFM_BUS_DEBUG`. **Do not** strip main's own
   `LAMBDA_VM_TIMELINE_JSON` / `LAMBDA_VM_TRACE_BUILDERS` instrumentation — that is merge content.
5. `make fmt` and `make lint` from the repo root (the de-indented `error!` sites are also a
   formatting failure).

### Step 6 — FINALIZE

`git push origin blake3-campaign-preMerge` **before** the branch push — ✓ the tag is currently
local-only, and after the fast-forward `refs/heads/blake3-real-hash` stops being the remote copy of
`ed1b7785`. Then commit (binding fix and HINT/census as separate commits), fast-forward, push.

### Escalation gates (the only things that need a further review pass)

- **G1.** If Step 0's residual is not on `Hwsl`, or if bucketing shows failures the HWSL theory
  cannot explain *and* they point at the verifier binding — stop, do not edit, re-open.
- **G2.** Any change to `expected_public_balance` or `replay_transcript_phase_a_view` — human
  review, mandatory. Both defenders and the attacker agree these are the recursion verifier's and
  the production VM verifier's soundness surface.
- **G3.** Re-pinning `continuation_epoch_constraint_leg_cost` — the attribution must be written
  down and read by a human before the constant moves. "Investigate, never re-bless"
  (`machine_tests.rs:113-117`) is the house rule and it applies here.

Everything else: implement directly.

---

## 8. Verification log (what I read myself)

| Claim | Status | Evidence |
|---|---|---|
| Main dropped 120 HWSL sends/round from `KECCAK_RND` | ✓ VERIFIED | `keccak_rnd.rs:439,446`; diff hunks `@@ -587,48 +596,8` / `@@ -717,53 +686,8` |
| Only the HWSL blocks changed in `bus_interactions()` | ✓ VERIFIED | filtered diff shows two `- BusId::Hwsl` and no other `BusId::` line |
| Main dropped the matching receives in production | ✓ VERIFIED | `origin/main:trace_builder.rs` has 0 `BitwiseOperationType::Hwsl`; `ed1b7785` has 2 (`:2427`, `:2510`) |
| LFM AIR built from main's `keccak_rnd::bus_interactions()` | ✓ VERIFIED | `airs.rs:21,239,488-494` |
| LFM receiver side is the branch fork, merge-untouched | ✓ VERIFIED | `trace.rs:176`; `git diff ed1b7785 -- keccak_adapter.rs` empty |
| Fork still emits 120 HWSL/round | ✓ VERIFIED | `keccak_adapter.rs:361-366` (20), `:441-446` (100) |
| Those are the only HWSL in `prover/src/lfm/` | ✓ VERIFIED | `grep -rn Hwsl prover/src/lfm/` → 2 hits |
| Per-round total is 1148; 1148 − 120 = 1028 | ✓ VERIFIED | hand count of `bitwise_ops_for` `:324-509` |
| Interactions μ-gated ⇒ padding sends nothing | ✓ VERIFIED | `Multiplicity::Column(cols::MU)` throughout `keccak_rnd.rs:446ff` |
| `keccak_ops` comes only from `records.keccak` | ✓ VERIFIED | `trace.rs:145-155`; `executor.rs:440,522` |
| `HasherKind::default() = Test`; trivial has no keccak | ✓ VERIFIED | `hash.rs:196-199`; `registry.rs:117-119`; `programs.rs:31-79` |
| crypto/stark LogUp fingerprint math unchanged | ✓ VERIFIED | first `lookup.rs` hunk at line 834; all fingerprint fns above it |
| `keccak_probe` verifies against hardcoded `FEE::zero()` | ✓ VERIFIED | `keccak_probe.rs:126-143` |
| `duplicate_tag_…_hazard` asserts ACCEPT | ✓ VERIFIED | `keccak_probe.rs:284-289` |
| `preprocessed_tags_…` only verify assert is negative | ✓ VERIFIED | `machine_tests.rs:441-444`; prove-side legs `:429`, `:440`, `:455-459` |
| HINT absent from both epoch-budget lists | ✓ VERIFIED | `constraint_tests.rs:1491-1494`, `:1497-1508`; pin at `:1566-1569` |
| `KECCAK_RND` is in `FIXED` (budget delta suspect) | ✓ VERIFIED | `constraint_tests.rs:1502` |
| Census panics before its mismatch assert | ✓ VERIFIED | `constraint_tests.rs:438-442` vs `:447-452` |
| `expected_public_balance` has exactly one caller | ✓ VERIFIED | `proof.rs:220` / `:247` |
| `replay_transcript_phase_a_view` reaches the VM verifier | ✓ VERIFIED | `lib.rs:989,1014,1442`; `continuation.rs:896` |
| In-circuit mirror exists | ✓ VERIFIED | `statement_replay.rs:164-190`; third mirror `machine_tests.rs:2101` |
| BITWISE not in `LfmArtifacts.roots` ⇒ program_id safe | ✓ VERIFIED | `registry.rs:133-152`; `bitwise.rs:101` |
| Five paths still unmerged in the index | ✓ VERIFIED | `git diff --name-only --diff-filter=U` |
| `verifier.rs` diagnostics stole a `#[cfg]` | ✓ VERIFIED | diff at `verifier.rs:1447-1450`; de-indents at `:1311,:1315,:1363` |
| `let ok = …; ok` refactor is semantics-preserving | ✓ VERIFIED | diff at `verifier.rs:240-263` |
| `debug-checks` + `DEBUG_BUS_ID` exist; `Hwsl = 9` | ✓ VERIFIED | `bus_debug.rs:8-14,66`; both Cargo.toml; `types.rs:283` |
| Tag `blake3-campaign-preMerge` not pushed | ? INFERRED | both defenders ran `git ls-remote --tags origin`; I did not re-run |
| `trivial_program_proves_and_verifies` passes in the merge tree | ? INFERRED | absent from `reconcile-report.md` §5's list; not re-run |
