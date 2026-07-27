# LogUp-GKR branch (`origin/feat/gkr-logup`, PR #485) — implementation analysis

> Companion to `port-plan.md`. Produced 2026-07-21 by deep-reading the branch tip (`78be304f`).
> "SNAP" refers to full-file snapshots of the branch, "DIFF" to per-file diffs vs the merge-base `7b42e51b`.
> Regenerate them locally with:
> `git fetch origin feat/gkr-logup`
> `git show origin/feat/gkr-logup:crypto/stark/src/gkr.rs` (etc.)
> `git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/lookup.rs` (etc.)
> Line references below are into those snapshots/diffs, NOT into current main.

## 1. Protocol architecture

**Core idea.** Standard LogUp commits per-table aux term columns + an accumulated column and checks Σ contributions via `bus_public_inputs.table_contribution`. The GKR branch instead proves each table's total LogUp sum with a **batch GKR over fractional summation trees**, and replaces the committed aux machinery with exactly **2 aux columns** per interacting table: the **Lagrange kernel `l`** (aux col 0) and a **bridge running sum `σ`** (aux col 1).

Pipeline per table:
1. **Leaf fractions** (`compute_logup_leaf_fractions`, DIFF lookup.rs:837-905): for each row i, all K interactions are folded into one fraction N(i)/D(i) by cross-multiplication (`n' = n·fp_k + sign_k·m_k·d`, `d' = d·fp_k`), where `fp_k = z − (bus_id·α⁰ + Σ vⱼ·αʲ)` and `m_k` from the `Multiplicity` variant. Row-parallel, no batch inversion.
2. **Summation tree** (`gen_layers`/`next_logup_layer`, SNAP gkr.rs:91-135): pairwise fraction addition halves each layer up to a 1-element root; root value n/d = the table's total contribution. `Layer::{LogUpGeneric, LogUpSingles}` (Singles = implicit numerators 1, ~50% fewer muls; production leaves are Generic).
3. **Batch GKR** (`gkr_prove_batch`, gkr.rs:1221-1851): ONE proof for all tables. Per layer (root→leaves): append each active instance's (n,d) claims; sample `sumcheck_alpha` (combines instances) and `lambda` (combines n/d claims: `claim = n + λ·d`); run one shared sumcheck over `max_parent_vars` variables with degree-3 round polys (4 evals each, `RoundPoly` in sumcheck.rs); gate = `nl·dr + nr·dl + λ·dl·dr` weighted by `eq(current_point, ·)`. Mixed-size instances: smaller tables join at later layers, with a `2^n_unused` doubling factor on their combined claim (gkr.rs:1309-1316) and constant `claim/2` round polys while inactive (1534-1550). After rounds, append 4 child claims per instance, sample `eta`, fold; `current_point = [eta] ++ challenges`. Eq handling uses Dao-Thaler halving + a scalar `eq_correction`, and **SVO** (split-value optimization, ePrint 2025/1117 Alg 5, `SVO_THRESHOLD=8`): eq split into prefix×suffix tables for √-memory (gkr.rs:522-534, batch version 1471-1494, 1614-1659).
4. **Output**: shared `random_point` + per-instance `(n_claim, d_claim)` = claimed leaf MLE evals. Per-instance point = `instance_eval_point(shared, n_vars)` = `[eta] ++ last (n_vars−1) challenges` (gkr.rs:1861-1876).
5. **Bridge back to the STARK** (the committed-trace link):
   - `column_claims`: MLE of each distinct main column referenced by any interaction (sorted indices via `extract_column_indices`, DIFF lookup.rs:352-399), evaluated at the instance point via the Lagrange kernel (`finalize_logup_gkr_result`, lookup.rs diff 1336-1376).
   - **Aux col 0 = Lagrange kernel** `l[i] = eq(bits(i), r)` (SNAP lagrange_kernel.rs:21-67). Bound by boundary constraint `l[0] = ∏(1−r_j)` (DIFF lookup.rs:312-330) and by an **l² self-check**: the bridge's batched value includes `γ^K·l[i]`, forcing `Σ l[i]² = ∏(r_k² + (1−r_k)²)` (BUG-004 fix; `compute_bridge_params`, lookup.rs diff 437-458).
   - **Aux col 1 = σ running sum** with circular transition constraint `LookupBridgeSumConstraint` (lookup.rs diff 512-629, degree 2, `end_exemptions=0`): `σ_next − σ_curr − l_curr·batched_curr + Δ = 0`, where `batched = Σⱼ γʲ·colⱼ + γ^K·l` and `Δ = target/N`, `target = Σⱼ γʲ·cⱼ + γ^K·l_mle_claim`. Telescoping over N rows proves `⟨l, colⱼ⟩ = cⱼ` for all j (Schwartz-Zippel over γ).
6. **Bus balance**: verifier sums `root_n/root_d` over `batch_gkr_proof.root_claims` and compares to `expected_bus_balance` (DIFF verifier.rs:386-421). `bus_public_inputs` is gone for GKR tables (`build_auxiliary_trace` returns `None`, only allocates).

**What replaces the acc columns:** aux goes from `⌈K/2⌉ term columns + 1 acc` to a **fixed 2 columns** regardless of K (`AirWithBuses::new`, DIFF lookup.rs:121-127). This is always-on for any AIR with interactions — no mode flag anywhere on the branch.

**Memory motivation** (PR bench: peak heap −70%, 221→66 GB; prove +2.6% on fib_iterative_8M at commit `334093fc`): the committed aux trace collapses from ~⌈K/2⌉+1 extension columns at LDE size to 2 (CPU K=40 → 21→2, DVRM 34 → 18→2, MEMW 26 → 14→2). That shrinks aux LDE buffers, aux Merkle trees, OOD tables, and DEEP openings ~10× for the big tables. GKR's own layer trees (~4N ext elements/table) are transient and freed before the LDE phase. (? INFERRED from code structure — consistent with the aux-column arithmetic; the in-branch design doc doesn't cover GKR.)

## 2. Prover integration map (`multi_prove`, DIFF prover.rs:157-510)

Round-1 structure becomes: **Phase A** (unchanged: per-table main commits appended to main transcript, SNAP prover.rs:1583-1639) → **Phase B** (unchanged: sample z, α; `LOGUP_NUM_CHALLENGES=2`, prover.rs:1648-1654) → **NEW Phase B′**: collect `gkr_table_indices` where `air.has_trace_interaction()`; compute layer trees in parallel (`compute_logup_layers` from `air.bus_interactions()` + `trace.columns_main()`); run `gkr_prove_batch` on the **main** transcript; distribute per-table `LogUpGkrResult` in parallel (`finalize_logup_gkr_result`; `table_contrib = root_n·root_d⁻¹`) → **NEW Phase B″**: append every table's `column_claims` values to main transcript, then sample **γ** (BUG-012) → **Phase C pass 1 changed**: for GKR tables, the aux trace is built **inline in prover.rs** (not in `build_auxiliary_trace`): write kernel to aux col 0; compute `l_mle_claim`, `compute_bridge_params`, precompute `batched[i]` (row-parallel), then sequentially fill `σ` forward with `σ[0]=0` (prover diff 326-401). Non-GKR aux path retained for AIRs without interactions. → **metadata build**: per-table `rap_challenges = [z, α] ++ bridge params` via `extend_rap_challenges_with_bridge` (layout: `[2]=γ, [3]=Δ, [4..4+K+1]=γ⁰..γ^K, then random_point`; lookup.rs diff 460-495); `bus_public_inputs: None`; new `Round1Metadata.logup_gkr_result` field → **Phase C pass 2 / Rounds 2-4 unchanged** (fork per table, aux root appended in fork), except each proof gets `logup_gkr_proof = {gkr_proof: stub-with-claimed_sum, random_point, column_claims}` attached post-hoc (prover diff 483-496); `MultiProof.batch_gkr_proof` carries the real batch proof. Single-table `prove` copies `batch_gkr_proof` into the `StarkProof` (BUG-014, prover diff 513-528). New bound `Field: IsPrimeField` on `multi_prove`/`prove`.

## 3. Verifier integration map (`multi_verify`, DIFF verifier.rs:171-429)

The old monolithic `step_1_replay_rounds_and_recover_challenges` is **deleted** (verifier diff 21-166; rounds 2-4 replay lives in `verify_rounds_2_to_4`). After Phase B challenge sampling: **Phase B′**: require `multi_proof.batch_gkr_proof` if any table has interactions; `gkr_verify_batch(batch_proof, n_layers_by_instance, transcript)` where `n_vars = trace_length.trailing_zeros()` per table; on success, per table: require `proof.logup_gkr_proof`; check `column_claims.len() == extract_column_indices(air.bus_interactions()).len()`; run `reconstruct_and_verify_gkr_claims(n_claim, d_claim, column_claims, interactions, challenges, n_layers)` (see §7); store `gkr_bridge_claims[idx]` and transcript-derived `gkr_random_points[idx] = instance_eval_point(shared_random_point, n_vars)`. **Phase B″**: append column_claims in the same order, sample γ. **Per-table**: fork transcript, append aux root, build `table_rap_challenges` via the same `extend_rap_challenges_with_bridge`, then `verify_rounds_2_to_4` — the bridge is checked as an ordinary transition constraint at the OOD point, the kernel by the boundary constraint. **Bus balance**: Σ over `batch_proof.root_claims` of `root_n·root_d⁻¹` (zero denominator → `return false`) `== expected_bus_balance`. `verify` wraps into a `MultiProof` carrying `proof.batch_gkr_proof`.

`gkr_verify_batch` (gkr.rs:1911-2170) replays exactly: activation → append active claims → sample α, λ → per-round `sum_at_binary` check + append evals + sample challenge → per-instance gate check with `eq(instance_point, challenges[n_unused..])` → append child claims → sample eta → fold claims.

## 4. Fiat-Shamir ordering (main transcript, branch-tip behavior)

1. Per table in index order: [preprocessed root] main root (`append_bytes`).
2. If any interactions: sample z, α (2 × `sample_field_element`).
3. `append_bytes(b"gkr_batch")`; `append_bytes(n_instances as u64 LE)`. Then per batch layer: for each instance active this layer (ascending index): append n, d; sample `sumcheck_alpha`; sample `lambda`; if non-trivial layer, per round: append 4 round-poly evals, sample challenge; then per active instance: append 4 child claims; sample `eta`.
4. Per GKR table (ascending index): append each `column_claims` value (sorted column order); then sample γ (ONE shared γ on the main transcript).
5. Per-table fork: clone; if multi-table append idx domain separator (pre-existing); append aux root; rounds 2-4 unchanged (β, OOD z, trace OOD evals, composition OOD, deep-γ, FRI ζ/roots, last value, nonce, iotas).

Note the (pre-existing) fork-time `append_field_element(bpi.table_contribution)` is dead for GKR tables since `bus_public_inputs` is `None`.

## 5. Proof format changes (DIFF proof_stark.rs)

- New `LogUpGkrProof<E> { gkr_proof: GkrProof<E>, random_point: Vec<E>, column_claims: Vec<(usize, E)> }`.
- `StarkProof` += `logup_gkr_proof: Option<LogUpGkrProof<E>>` and `#[serde(default)] batch_gkr_proof: Option<BatchGkrProof<E>>`.
- `MultiProof` += `#[serde(default)] batch_gkr_proof: Option<BatchGkrProof<E>>`.
- `BatchGkrProof { root_claims: Vec<(E,E)>, layer_proofs: Vec<BatchGkrLayerProof> }`; `BatchGkrLayerProof { sumcheck_proof: SumcheckProof, child_claims_by_instance: Vec<[E;4]> }`. All serde-derived (branch predates rkyv-authoritative wire format).
- **Dead weight**: in batch mode `LogUpGkrProof.gkr_proof` is a stub (empty `layer_proofs`) and `random_point` is written by the prover but **never read by the verifier** (✓ VERIFIED by grep: verifier only reads `column_claims`). The port should drop both fields.
- Proof size: replaces per-table aux OOD/opening data with the batch GKR proof (≈ Σ_layers rounds × 4 evals + 4 claims/instance/layer, extension elements). No in-branch size numbers.

## 6. API/trait changes needed from the (now-rewritten) lookup layer

- `AIR::bus_interactions(&self) -> &[BusInteraction]` (new trait method, default `&[]`; DIFF traits.rs:18-24) — the only trait addition.
- The GKR path consumes: `BusInteraction{bus_id, is_sender, values, multiplicity}`, all `Multiplicity` variants, `BusValue::{Packed, Linear}` + `Packing::combine`/`num_columns` + `accumulate_fingerprint` + `column_indices()`, `LinearTerm`, `num_bus_elements()`, `PackingShifts`, `compute_alpha_powers`. It reads **raw main columns** (`trace.columns_main()`) — it never touches the constraint system.
- New lookup.rs exports: `LOGUP_CHALLENGE_Z/GAMMA`, `LOGUP_BRIDGE_OFFSET_IDX`, `LOGUP_GAMMA_POWERS_START`, `logup_random_point_start`, `extract_column_indices`, `compute_bridge_params`, `extend_rap_challenges_with_bridge`, `compute_logup_leaf_fractions`, `compute_logup_layers`, `finalize_logup_gkr_result`, `reconstruct_and_verify_gkr_claims`, `LogUpGkrResult`, `LookupBridgeSumConstraint` (a boxed `TransitionConstraint` — on current main this must become an emit-style constraint body). Old machinery (`split_interactions`, term-column builders, `LookupBatchedTermConstraint`, `LookupAccumulatedConstraint`, debug bus sums) is `#[allow(dead_code)]`-retired on the branch, not deleted.
- `AirWithBuses::new`: aux layout = 2 columns; appends the bridge constraint; `transition_offsets` stays `[0,1]`.

## 7. Soundness history + review-item status at branch tip

Branch bug-fix commits: `d04f642a` (BUG-004 Lagrange-kernel binding via γ^K·l² self-check; BUG-011 0-layer forgery → `n_layers == 0` direct rational check; BUG-012 column_claims FS gap → Phase B″ binding before γ), `54dfa975` + `95deda49` (verifier DoS/panic bounds), `dfd4c53b` (zero root denominator → reject not panic), `e7761ed9` (trivial-layer gate check in batch verifier + BUG-014 single-proof API), `72ad8cf7` (Result-returning cleanups + tests).

**The Codex review's three items, verified at branch tip:**

1. **Payload `random_point` trust — FIXED.** ✓ VERIFIED: the verifier uses only the transcript-derived point: `gkr_verify_batch(...) → shared_random_point` (SNAP verifier.rs:828), `instance_eval_point(&shared_random_point, n_vars)` → `gkr_random_points[table_idx]` (889-890), fed to `extend_rap_challenges_with_bridge` (956). `proof.logup_gkr_proof.random_point` is never read anywhere in verifier.rs. Covered by the batch-verify rework + `d04f642a`.

2. **`reconstruct_and_verify_gkr_claims` fail-open — STILL FAIL-OPEN for multi-interaction tables; a real open soundness gap.** ✓ VERIFIED at tip (DIFF lookup.rs:1108-1118): `if interactions.len() == 1 || n_layers == 0 { rational cross-check n·d̂ == n̂·d } else { true }` — the `n_layers == 0` arm is BUG-011's fix, but K>1 tables with N>1 rows still pass after structural checks only. The in-code justification ("bridge ensures column_claims consistency") is true but insufficient: the bridge binds **column_claims ↔ trace** (⟨l,colⱼ⟩=cⱼ), while **nothing binds the GKR leaf claims (n_claim,d_claim) to the columns** — the leaf fraction is a nonlinear (product) combination, MLE doesn't commute with products, and no other check touches `per_instance_claims` for these tables. Consequence: a prover can run an honest GKR over **fabricated leaf vectors** (arbitrary root contribution, e.g. to fake bus balance) while supplying honest column_claims so the bridge, kernel checks, and FS all pass. Since every production table is multi-interaction, this needs a protocol fix before production use — e.g. per-interaction leaves (K separate GKR instances, or a wider input layer where numerator/denominator are **linear** in columns, as in Winterfell/Stwo LogUp-GKR) or an extra input-layer sumcheck reducing the leaf claim to column MLE claims.

3. **`gkr_verify_batch` malformed-proof panics — LARGELY FIXED, one residual vector.** ✓ VERIFIED: `layer_proofs.len() == max_layers` (SNAP gkr.rs:1939-1947), `child_claims_by_instance.len() >= active_instances.len()` (1999-2008), `num_rounds >= parent_num_vars_i` (2099-2106), zero root denominator → `return false` (DIFF verifier.rs:397-411). **Residual**: `RoundPoly` derives `Deserialize` with no length invariant; `sum_at_binary` **asserts** `evals.len() >= 2` (SNAP sumcheck.rs:38-44) and `evaluate` computes `evals.len()-1` (57) — a proof containing an empty/1-eval round poly still panics the verifier (called at gkr.rs:2072 before any length validation). Minor: `1u64 << n_unused` (gkr.rs:1989) can overflow-panic in debug if instance size spans ≥64 (needs absurd `trace_length`; `trace_length=0` gives `trailing_zeros()=64` unless validated upstream). The port must add a `num_evals` check (or make `sum_at_binary`/`evaluate` fallible).

## 8. Tests

- **Portable as-is** (self-contained modules): gkr.rs `mod tests` (fraction add, tree building, single + batch prove/verify roundtrips, tamper rejection), sumcheck.rs tests, lagrange_kernel.rs tests (kernel values, partition of unity, MLE evals).
- **lookup.rs tests** (DIFF 1378-1623): leaf-fraction unit tests (single sender / receiver-with-multiplicity / two-interaction cross-multiply / consistency vs `compute_logup_term_column`) — portable modulo the old term-column reference (the consistency test needs a local reimplementation or deletion).
- **bus_tests/soundness_tests.rs** (DIFF, −4/+5 tests): DELETED obsolete `test_tampered_table_contribution`, `test_missing_bus_public_inputs_rejected`, `test_injected_bus_public_inputs_on_non_logup_air_rejected`, `test_zeroed_table_contribution_rejected`; ADDED `generate_valid_multi_proof` helper + `test_tampered_gkr_column_claims_rejected`, `test_tampered_gkr_claimed_sum_rejected` (tampers `batch_proof.root_claims`), `test_missing_gkr_proof_rejected`, `test_tampered_sigma_ood_rejected`, `test_tampered_lagrange_kernel_random_point_rejected` (tampers `child_claims`); kept/adapted `test_tampered_acc_ood_evaluation`. All portable in spirit; they use `AirWithBuses` test AIRs which exist (rewritten) on main.
- **completeness_tests.rs**: +`test_single_table_prove_verify_with_gkr` (BUG-014 regression). packing_tests.rs: 8-line layout-constant tweak.
- No test covers the §7-item-2 gap (a fabricated-leaves forgery test would FAIL-to-reject on this branch — worth adding as an expected-fail/#[ignore] marker test during the port).

## 9. Perf-commit contamination (exclude from the GKR port)

Unrelated hunks (commits `bbe4dbc4`, `d5666021`, `c3f5719b`, `3aa03e6c`, `ccb75483`):
- `crypto/stark/src/fri/fri_functions.rs` — entire diff (parallel `fold_evaluations_in_place`), commit `3aa03e6c`. NOT on main; separable follow-up PR if still wanted.
- `crypto/stark/src/prover.rs` — `commit_columns_bit_reversed` `map_init` row-buffer hunk (DIFF prover.rs:35-79), commit `ccb74483` — function deleted on main (#735); `Arc<LdeTwiddles>` dedup hunks (82-105, 117-147, 152-153, 410-411, 465-466), commit `c3f5719b` — superseded on main.
- `prover/src/tables/mod.rs` — entire diff (max_rows caps), commit `d5666021` — already on main as #499.
- `crypto/stark/src/constraints/evaluator.rs` — entire diff (timing instrumentation only).
- `crypto/stark/src/debug.rs` — 2 comment lines.
- `docs/superpowers/specs/2026-04-09-scaling-improvements-design.md` — commit `bbe4dbc4` (see §10).

GKR-core also includes its own perf commits (`7ec0b566`/`fb65d0f2` SVO, `dd6a2147` O(1) claim updates, `36150b81` parallel fold_table) — these live inside gkr.rs/sumcheck.rs and port with the files.

## 10. The in-branch design doc

`docs/superpowers/specs/2026-04-09-scaling-improvements-design.md` is **not** about GKR — it's the spec for the unrelated Item-9 perf commits (unified 2^20 sizing, twiddle dedup, FRI-fold parallelization, commit-alloc fix) plus future MMCS batched commitments and shared FRI; it explicitly lists "GKR-based LogUp" as a **non-goal**. GKR's actual motivation is the memory profile (peak heap −70% at +2.6% prove time, per PR bench). The disk-spill follow-up (PR #489) is not described in any in-branch doc.
