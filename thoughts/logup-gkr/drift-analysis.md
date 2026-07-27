# Drift analysis: `feat/gkr-logup` (PR #485) vs `origin/main`

> Companion to `port-plan.md`. Produced 2026-07-21. Merge-base `7b42e51b` (2026-04-10); main (`a8648320`) has ~148 commits since, including full rewrites of every GKR integration point.

## 1. Per-commit classification (22 commits on the branch)

| Commit | Subject | Class | Verdict |
|---|---|---|---|
| `292fce80` | port logup gkr (the v2 port: gkr.rs +2663, sumcheck.rs +828, lagrange_kernel.rs +316, lookup/prover/verifier wiring) | GKR-core | **port** (via branch-tip state, not cherry-pick) |
| `bbe4dbc4` | docs: scaling improvements spec | docs | optional (unrelated to GKR — see branch-analysis §10) |
| `d5666021` | cap LT/LOAD/BRANCH/MEMW_R max_rows at 2^20 | perf | **already-in-main** — landed as PR #499 (`902e1027`); main's `prover/src/tables/mod.rs` has a `max_rows` module with all four at `1 << 20` |
| `c3f5719b` | dedup LdeTwiddles by domain size in multi_prove | perf | **already-in-main (superseded)** — main has a full shared-cache design (`LdeTwiddles` + `OnceLock` composition twiddles, `twiddle_caches: &[Arc<LdeTwiddles>]` threaded through multi_prove) |
| `3aa03e6c` | rayon-parallelize `fold_evaluations_in_place` | perf | **still-absent on main** — main's fold is sequential (verified). Independent of GKR; resubmit separately if still wanted (signature changed via #591/#598, needs re-port) |
| `ccb75483` | eliminate per-row Vec alloc in `commit_columns_bit_reversed` | perf | **superseded-by-rewrite** — function gone; #735 replaced it with `commitment.rs::commit_bit_reversed` (`rows_per_leaf`) |
| `b2821bf8` | Merge feat/logup_gkr_v2 | merge | n/a |
| `36150b81` | parallelize GKR `fold_table` inner loop | GKR-core (perf) | port (already in branch-tip gkr.rs) |
| `9b038bfb` | wire batch GKR into STARK prover+verifier | GKR-core (wiring) | **re-implement by hand** — targets pre-#764/#823 prover/verifier |
| `7ec0b566` | SVO for eq polynomial in sumcheck | GKR-core (perf) | port |
| `dd6a2147` | eliminate O(N) combined_claims recompute | GKR-core (perf) | port |
| `a9aa266d` | save work (prover.rs wiring tweaks) | GKR-core (wiring) | re-implement |
| `fb65d0f2` | SVO port to batch GKR inner loop | GKR-core (perf) | port |
| `95deda49` | "Fix three verifier panics in batch GKR" | GKR-core | port (note: diffstat is only +1 line in fri_functions.rs — message/content mismatch; real fixes appear in later commits) |
| `54dfa975` | verifier DoS guards (layer_proofs len, child_claims bounds) | GKR-core (soundness) | port — in branch-tip gkr.rs |
| `d04f642a` | Lagrange kernel soundness, 0-layer bus forgery, column_claims FS gap | GKR-core (soundness) | port — spans gkr/lookup/prover/verifier; the lookup/prover/verifier parts need re-implementation |
| `334093fc` / `6ee8c779` | merges of main (spec v0.2 etc.) | merge | n/a |
| `dfd4c53b` | return false on zero GKR root denominator | GKR-core (soundness) | port |
| `e7761ed9` | gate check for trivial layers + single-proof API fix | GKR-core | port |
| `72ad8cf7` | dead-pub cleanup, Result instead of panic, tests | GKR-core | port |
| `78be304f` | fix fmt | chore | n/a |

Key point: **all GKR algorithm work (incl. all soundness fixes) is contained in the branch-tip versions of the three new files**; the wiring commits are the only part that must be redone.

## 2. Sibling branches

- **`feat/logup_gkr_v2`**: fully contained in `feat/gkr-logup` (`git log gkr-logup..logup_gkr_v2` is empty). Ignore.
- **`feat/gkr-logup_opt`**: stale precursor. Unique commits: `a64da708` "add instruments" (+31 lines of timing instrumentation in prover.rs) and a 4-line save-work. It **lacks** the later soundness fixes. Nothing worth salvaging; ignore.
- **PR #489 disk-spill** (branch deleted; fetchable as `refs/pull/489/head`, tip `f2b2bbf2`): stacked on gkr-logup, adds mmap disk-spill of trace tables + Merkle trees (`crypto/stark/src/{table,trace}.rs` +555, `prover/src/tables/trace_builder.rs` +199, `prover/src/tests/disk_spill_tests.rs`, memmap deps). Orthogonal scaling feature, also missing the final soundness fixes. Treat as separate future work, not part of this port.

## 3. Main-side rewrites of touched files (all still exist on main; none renamed/deleted)

| File | Main commits since base | Dominant reshapers |
|---|---|---|
| `lookup.rs` | 10 | **#764 single-source constraints** (rewrote LogUp emission wholesale), **#823 forward accumulation** (acc = sole next-row OOD read), #762 GPU aux-trace build, #696, #769. Branch is 3099 lines vs main 2755; **zero** GKR identifiers survive on main |
| `prover.rs` | 28 | **#799 full GPU trace residency**, #748 GPU-resident LDE/Merkle, #735 commitment-layer unification, #715 row-major LDE, #729 FRI early termination, #762, #823. Branch 2457 vs main 3325 lines |
| `verifier.rs` | 14 | #823, #826 deep-composition fusion, #815 OOD guards, **#769 rkyv in-place verify** (verifier reads archived proofs), #764, #729, #735 |
| `traits.rs` | 4 | #764 (AIR trait: `compute_transition_prover`/`constraint_program()`), #823 |
| `proof/stark.rs` | 4 | #769 (rkyv derive on proof types!), #729, #735, #823 — any new GKR proof fields must be rkyv-compatible |
| `constraints/evaluator.rs` | 6 | #764, **#798 GPU constraint/composition eval**, #799 |
| `fri/fri_functions.rs` | 2 | #591, #598 (signature changes only; branch's rayon fold not present) |
| `debug.rs` | 3 | #764 etc. (branch delta trivial: +2 lines) |
| `lib.rs` | 9 | module-registration churn (trivial to redo: add `pub mod gkr; pub mod sumcheck; pub mod lagrange_kernel;`) |
| `prover/src/tables/mod.rs` | 7 | #499 already covers the branch's change; #685 continuations, #657/#753 ECSM, #644 |
| `tests/bus_tests/*` | many | all three touched files exist; heavily churned by #764/#823/#688 — branch test edits must be re-authored against the new harness |

## 4. Mechanical conflict probe (`git merge-tree --write-tree origin/main origin/feat/gkr-logup`)

**Conflicted (9):** `constraints/evaluator.rs`, `lib.rs`, `lookup.rs`, `proof/stark.rs`, `prover.rs`, `bus_tests/soundness_tests.rs`, `traits.rs`, `verifier.rs`, `prover/src/tables/mod.rs` — i.e., every wiring file.
**Clean:** `gkr.rs`, `sumcheck.rs`, `lagrange_kernel.rs`, the docs spec (pure adds), plus `debug.rs`, `fri_functions.rs`, `completeness_tests.rs`, `packing_tests.rs` — though "auto-merged" here still means semantically wrong (they auto-merge against code #764 deleted).

## 5. Verdict

**Rebase/cherry-pick is not viable; port-plus-rewire is.** Concretely:

1. **Port wholesale (clean adds, branch-tip versions):** `crypto/stark/src/gkr.rs` (3188 lines), `sumcheck.rs` (830), `lagrange_kernel.rs` (311). These carry every algorithmic commit including all five soundness-fix commits. Expect only compile-level fixups (field-trait/API drift from #598 etc.).
2. **Re-implement by hand on main:** all wiring — `lookup.rs` (against #764's single-source LogUp emission + #823's forward accumulation), `prover.rs` (against #748/#799's GPU-resident round structure), `verifier.rs` (against #769's rkyv archived-proof access + #826), `proof/stark.rs` (new GKR fields need rkyv derives), `traits.rs`, `lib.rs` module decls, and re-authored bus_tests. Use the branch's diffs (`git diff 7b42e51b..origin/feat/gkr-logup -- <file>`) as the semantic spec of *what* to wire, not *how*.
3. **Drop:** `d5666021` (in main as #499), `c3f5719b` (superseded by shared twiddle caches), `ccb75483` (function deleted by #735), everything from `_opt` and `logup_gkr_v2`.
4. **Optional separate PR:** re-port `3aa03e6c` (rayon FRI fold) — genuinely absent on main's CPU path.
5. **Defer:** PR #489 disk-spill (orthogonal, stacked, stale).
