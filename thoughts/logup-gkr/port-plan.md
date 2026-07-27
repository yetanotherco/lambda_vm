# Plan: bring LogUp-GKR (PR #485) up to date with the current prover

> Written 2026-07-21 against main `a8648320`. Execution target: an implementation agent working phase-by-phase.
> Companion docs in this directory — READ ALL THREE BEFORE STARTING:
> - `branch-analysis.md` — what the GKR branch implements (protocol, wiring semantics, FS order, soundness status). This is the **semantic spec** of what to port.
> - `main-surface-map.md` — the current shape of every integration point on main (verified file:line anchors).
> - `drift-analysis.md` — per-commit classification, what to drop, why rebase is not viable.
>
> Also read the project memory doc `single-source-constraints-architecture.md` (auto-memory) if available — it explains the constraint-emission contract this port must respect.

## 0. Executive summary

PR #485 (`feat/gkr-logup`, tip `78be304f`, closed draft) replaces committed LogUp aux columns (⌈K/2⌉ term columns + 1 acc per table) with a **batch GKR proof over fractional summation trees** plus exactly **2 committed aux columns** per table (Lagrange kernel `l`, bridge running sum `σ`). Measured on the PR: **peak heap −70 % (221 GB → 66 GB), prove time +2.6 %**. That memory win is the point of resurrecting it (it composes with continuations: bigger epochs per box).

Main has moved ~148 commits since the merge-base: the entire constraint system was replaced (single-source constraints #764/#772), the acc column became the sole next-row OOD read (#823), the proof wire format became rkyv-in-place (#769), and the prover is now GPU-resident end-to-end (#748/#798/#799). **A git rebase/cherry-pick is not viable** (all 9 wiring files conflict against rewritten code). The strategy is:

1. **Port the three new modules wholesale** from the branch tip (`gkr.rs`, `sumcheck.rs`, `lagrange_kernel.rs`) — they are self-contained, conflict-free, and already contain all five soundness-fix commits.
2. **Re-implement the wiring by hand** against current main, as an **opt-in mode** (`LogUpMode::Gkr`), keeping standard LogUp the default and byte-identical.
3. Fix the known small defects during the port (dead proof fields, a verifier deserialization panic); **document but do not yet fix** the one open protocol-level soundness gap (multi-interaction leaf binding — see §6), which becomes the immediate follow-up.

## 1. Ground rules (non-negotiable, from project conventions)

- **Opt-in mode, default OFF.** With GKR disabled, the build must be **wire-identical to main**: same transcript, same proof bytes accepted. Gate: cross-version verification (old binary ↔ new proofs, both directions) — this is the king gate for constraint-system changes; proof determinism is NOT a goal (proofs are nondeterministic by design).
- **All new/changed constraints go through the single-source `ConstraintBuilder` emission path.** The bridge constraint must be ONE `emit`-style body serving all four interpretations (ProverEvalFolder / VerifierEvalFolder / CaptureBuilder→IR / MetaBuilder-derived meta). Never a hand-written parallel evaluator. Constraint bodies are declarative — no micro-opts without a measured bench win.
- **Verifier code is recursion-guest code.** `crypto/stark/src/verifier.rs`, `gkr.rs` (verify paths), `sumcheck.rs`, `lagrange_kernel.rs` all compile into the RV64 guest (`prover/src/recursion.rs` → `bench_vs/lambda/recursion`). No `HashMap`/`HashSet` on any verify-reachable path (audit the ported files — replace with sorted `Vec` + linear scan/binary search if found). Keep allocations in layer verification minimal.
- **Never trust proof-carried data for control flow or randomness.** Mode comes from AIR/verifier configuration, never from proof-field presence. All GKR randomness comes from the transcript (the branch already fixed this — preserve it).
- **Hygiene:** `cargo fmt` + `make lint` (workspace-level, not per-package clippy) before every push. No AI attribution anywhere. Bench runs happen on the bench server via PR comment (`/bench`, `/bench-abba`), never locally.

## 2. Source material

```bash
git fetch origin feat/gkr-logup                     # tip 78be304f, merge-base 7b42e51b
git show origin/feat/gkr-logup:crypto/stark/src/gkr.rs              > /tmp/gkr.rs
git show origin/feat/gkr-logup:crypto/stark/src/sumcheck.rs         > /tmp/sumcheck.rs
git show origin/feat/gkr-logup:crypto/stark/src/lagrange_kernel.rs  > /tmp/lagrange_kernel.rs
git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/lookup.rs        # wiring spec
git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/prover.rs
git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/verifier.rs
git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/proof/stark.rs
git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/traits.rs
git diff 7b42e51b..origin/feat/gkr-logup -- crypto/stark/src/tests/
```

Exclusions (do NOT port — see drift-analysis §1/§5): everything touching `fri/fri_functions.rs`, `constraints/evaluator.rs` (timers), `prover/src/tables/mod.rs`, the `commit_columns_bit_reversed`/`LdeTwiddles` hunks in prover.rs, the scaling-improvements docs spec. Ignore branches `feat/gkr-logup_opt` and `feat/logup_gkr_v2` entirely. Defer PR #489 (disk spill).

## 3. Target design on current main

### 3.1 Mode selection

Add `LogUpMode { Standard, Gkr }`:

- Carried by `AirWithBuses` (constructor parameter or a `with_logup_mode` builder), stored alongside `LogUpLayout`. In `Standard` mode every code path is exactly today's.
- VM assembly: thread a mode flag through `VmAirs` construction (`prover/src/lib.rs:478`) — one switch for all tables. For experiments, wire an env var (e.g. `LAMBDA_LOGUP_GKR=1`) at the `VmAirs` call sites; library API takes the enum explicitly.
- **Prover and verifier must agree out-of-band** (same `VmAirs` config). The verifier decides expectations from ITS mode: in `Gkr` mode `batch_gkr_proof` is required and `bus_public_inputs` must be absent; in `Standard` mode a proof carrying GKR fields is rejected (fail-closed both ways).

### 3.2 Aux layout in GKR mode

Per interacting table: aux = `[l (kernel), σ (bridge sum)]`, ext3 columns, aux width 2. Constants: `GKR_AUX_KERNEL_COL = 0`, `GKR_AUX_SIGMA_COL = 1`.

- `trace_layout()` → `(main_width, 2)`.
- `trace_ood_next_row_columns()` → `[main_width + GKR_AUX_SIGMA_COL]` (σ is the only next-row read; the kernel is current-row only). This slots directly into the #823 OOD-pruning machinery (`ood.rs::OodLayout`) — the pruned next-row block stays width 1.
- Boundary constraints (mode-aware in `AirWithBuses::boundary_constraints`, replacing the `acc[0]=0` pin): `l[0] = ∏ⱼ(1−rⱼ)` and `σ[0] = 0`.
- `composition_poly_degree_bound`: `logup_max_degree` becomes mode-aware — bridge constraint is degree 2 (vs 3 for standard batched terms), so the bound is `max(CS::max_degree(), 2)` in GKR mode. The verifier derives composition part count from the AIR (#699), so this stays consistent automatically — but note proof shape differs between modes by design.

### 3.3 Constraint emission (the part that MUST be re-designed, not transliterated)

The branch's `LookupBridgeSumConstraint` is a boxed `TransitionConstraint` — that world is gone. Write `emit_logup_gkr_constraints(builder, layout, num_base)` next to `emit_logup_constraints` (`lookup.rs:2256`), emitting exactly ONE ext constraint via `b.emit_ext(idx, expr)`:

```
σ_next − σ_curr − l_curr · batched_curr + Δ = 0        (degree 2, RowDomain::ALL)
batched_curr = Σⱼ γʲ·colⱼ(curr) + γᴷ·l_curr
```

where `colⱼ` ranges over the K distinct main columns referenced by any interaction (sorted order from `extract_column_indices`), and `Δ`, `γ`-powers, and the random point come from the extended rap-challenges vector (§3.4). Follow the invariants: base-operand-LEFT for base×ext products (`colⱼ · γʲ`, not `γʲ · colⱼ`), `const_base`/`const_signed` as the only constant path, exact-once emission per index. `MetaBuilder` then derives meta for free; `AirWithBuses::new` selects which emit body to run based on mode (both for meta derivation at construction and for `compute_transition{,_prover}` at runtime). The captured `ConstraintProgram` (lazy OnceLock) automatically reflects the mode — which keeps `constraint_program_tests`' folder==interpreter contract meaningful in both modes.

Access to challenges inside the body goes through the existing builder hooks (`challenge(idx)`, `alpha_pow`, `table_offset` — see `constraints/builder.rs:104-154`). Reuse `table_offset` for `Δ` (it plays exactly the acc-offset role `L/N` plays today, so `TransitionEvaluationContext.logup_table_offset` carries it unchanged) and map γ-powers onto `logup_alpha_powers` OR add a parallel `gkr_gamma_powers` slot in `TransitionEvaluationContext` — decide by whichever avoids overloading semantics the recursion guest also compiles; prefer a new explicit field.

### 3.4 Extended rap-challenges layout (GKR mode)

Adopt the branch's layout (branch lookup.rs diff 460-495): `[0]=z, [1]=α, [2]=γ, [3]=Δ, [4..4+K+1]=γ⁰..γᴷ, [4+K+1..]=r (instance random point, n_vars elements)`. Keep the branch's named constants (`LOGUP_CHALLENGE_Z/GAMMA`, `LOGUP_BRIDGE_OFFSET_IDX`, `LOGUP_GAMMA_POWERS_START`, `logup_random_point_start`). Note z/α/γ are shared across tables; Δ, the γ-power count (K varies per table), and r (length = log₂ trace_length, varies per table) are per-table.

### 3.5 Prover flow (into `multi_prove`, `prover.rs:2319`)

Insert between Phase B (z/α sampling, `prover.rs:2521-2530`) and Phase C pass 1 (aux build, 2543):

- **Phase B′ (GKR batch prove, shared transcript):** collect interacting tables; build leaf fractions + layer trees in parallel (`compute_logup_leaf_fractions`/`compute_logup_layers`, ported from the branch — they read `air.bus_interactions()` + host main-trace columns); `gkr_prove_batch(instances, transcript)`; `finalize_logup_gkr_result` per table (kernel-weighted `column_claims`, `l_mle_claim`). Free the layer trees before the LDE phase (this is where the memory win lives).
- **Phase B″ (binding + γ):** append every table's `column_claims` to the shared transcript (ascending table index, sorted column order), then sample the single shared `γ`.
- **Aux build stays in `build_auxiliary_trace`** (do NOT copy the branch's inline-in-prover.rs hack — it only existed because the old code structure forced it). Pass the full extended challenge vector (§3.4) into the existing Phase-C-pass-1 parallel call; in GKR mode the impl writes `l` (from `lagrange_kernel.rs`) and fills `σ` forward (`σ[0]=0`; per-row `batched` precomputed row-parallel, σ filled sequentially), and returns `bus_public_inputs = None`.
- Fork-time `append L` (`prover.rs:2954-2956`) is skipped in GKR mode (bpi is None; make the skip explicit and mode-checked, not accidental-on-None).
- Rounds 2-4 unchanged. `MultiProof` gains `batch_gkr_proof`; per-table `StarkProof` gains `column_claims` (drop the branch's dead `random_point` field and the stub `gkr_proof` — see branch-analysis §5). Single-table `prove` wrapper carries the batch proof (branch BUG-014).
- **Main-trace host access:** leaf-fraction building reads host main-trace columns — these exist regardless of GPU residency (#799 keeps LDE on device, not the raw trace). Verify this assumption at implementation time (`trace.columns_main()` availability in `multi_prove` before Phase C).

### 3.6 Verifier flow (into `multi_verify_views`, `verifier.rs:1126`)

Mirror insertion after shared z/α sampling (1216-1222), replacing the bpi-presence check (1232-1246) with the mode-aware rule from §3.1:

- Require `batch_gkr_proof`; compute per-table `n_vars = trace_length.trailing_zeros()` (validate `trace_length` is a power of two ≥ 1 first); `gkr_verify_batch(proof, n_layers_by_instance, transcript)` → shared random point + per-instance `(n̂, d̂)` claims.
- Per table: check `column_claims.len() == extract_column_indices(...).len()`; run `reconstruct_and_verify_gkr_claims` (ported as-is, gap documented — §6); derive `r = instance_eval_point(shared_point, n_vars)` **from the transcript-derived point only**.
- Append column_claims + sample γ exactly as the prover (Phase B″ mirror).
- Per-table: fork, absorb aux root, skip the L absorb in GKR mode, build extended rap challenges, `verify_rounds_2_to_4` unchanged — bridge checked as an ordinary OOD transition constraint (VerifierEvalFolder runs the same emit body), kernel bound by the `l[0]` boundary constraint + the γᴷ·l self-check inside the bridge (`Σl² = ∏(rₖ²+(1−rₖ)²)` folded into Δ/target — see branch-analysis §1.5).
- **Global bus balance replacement:** `Σ_instances root_nᵢ·root_dᵢ⁻¹ == expected_bus_balance` (reject on any zero root denominator). Must support the nonzero `expected_bus_balance` target exactly like today's check at `verifier.rs:1310-1330` (continuations/COMMIT-bus depend on it).
- Keep #815-style shape guards; the `ood_blocks_well_formed` check derives from the AIR so aux-width-2 flows through.

### 3.7 Proof format (`proof/stark.rs` + the rkyv view layer)

New types (from branch, minus dead fields): `BatchGkrProof { root_claims: Vec<(E,E)>, layer_proofs: Vec<BatchGkrLayerProof> }`, `BatchGkrLayerProof { sumcheck_proof, child_claims_by_instance: Vec<[E;4]> }`, `SumcheckProof`/`RoundPoly` from sumcheck.rs. All must derive **rkyv Archive/Serialize/Deserialize** (wire format, #769) AND get zero-copy accessors on the `StarkProofView`/multi-proof view layer, plus serde for the examples CLI. `MultiProof.batch_gkr_proof: Option<...>`, `StarkProof.gkr_column_claims: Option<Vec<(u32, E)>>` (or similar; rkyv-friendly index type). In-place validation rules for the new archived types must enforce the §5 length invariants at access time (the view layer is how the verifier reads untrusted bytes).

### 3.8 GPU path in GKR mode: v1 = CPU-only, explicitly gated

The GPU stack encodes the standard-LogUp aux shape and constraint set:
- `logup_gpu.rs::try_build_aux_resident_gpu` + term-column builder: term/acc layout → **must not fire** in GKR mode.
- Fused GPU composition (`constraint_ir/gpu_interp.rs::try_eval_composition_gpu`) consumes the captured program + `logup_alpha_powers`/`logup_table_offset` plumbing; the bridge constraint changes that challenge surface.

v1 decision: in GKR mode, force `device_only = false` and make every GPU `try_*` return None (cleanest: add `logup_mode == Standard` to `device_only_gate` and to the aux-GPU/composition-GPU entry conditions, each with a comment naming the assumption it protects). Main-trace GPU LDE/Merkle MAY be left on where it is shape-agnostic, but only if the `device_only` interactions are verified — when in doubt, CPU everything under GKR mode. The original −70 % heap / +2.6 % time bench was CPU-vs-CPU, so the A/B story survives. GPU support for GKR (aux build of l/σ on device, bridge in the fused kernel, GKR sumcheck kernels) is follow-up work (§8).

## 4. Execution phases (each ends compiling + green on its gate)

**Phase 0 — Branch + module port.** New branch off main (e.g. `feat/logup-gkr-v3`). Add `gkr.rs`, `sumcheck.rs`, `lagrange_kernel.rs` from the branch tip; register in `lib.rs`; fix compile drift (field-trait APIs, transcript API, rand/test-helper churn since April). Port their in-module tests. Audit all three files for HashMap/HashSet on verify paths (§1). Gate: `cargo test -p lambda-vm-crypto-stark gkr sumcheck lagrange` green (adjust package/filter names to the workspace layout).

**Phase 1 — Lookup-layer adapter (`lookup.rs`).** `LogUpMode`; `extract_column_indices`, `compute_logup_leaf_fractions`, `compute_logup_layers`, `finalize_logup_gkr_result`, `compute_bridge_params`, `extend_rap_challenges_with_bridge`, `reconstruct_and_verify_gkr_claims` (ported from branch diffs, adapted to current `BusInteraction`/`Multiplicity`/`Packing` — reuse `logup_gpu.rs::eval_fingerprint`-style per-row helpers where they fit); `emit_logup_gkr_constraints` per §3.3; mode-aware `AirWithBuses` (aux layout, meta derivation, boundary constraints, `trace_ood_next_row_columns`, degree bound, `build_auxiliary_trace`). Gate: unit tests — leaf fractions vs a direct per-row `m/fp` sum; `Σ leaf fractions == standard-mode table_contribution` on a small AIR (the two modes MUST agree on the total, this is the cross-mode oracle); meta/emit parity (MetaBuilder-derived meta matches expectations; `EmitTracker`-style exact-once holds).

**Phase 2 — Proof format + views.** §3.7 types with rkyv + serde + view accessors + length-validating access. Gate: roundtrip serialization tests (rkyv archive → view → values; serde for CLI), plus a malformed-bytes rejection test through the view layer.

**Phase 3 — Prover wiring.** §3.5 into `multi_prove` + single-table `prove` + `Round1Metadata` extension + GPU gating per §3.8. Gate: `Standard` mode untouched (full existing prover test suite green); `Gkr` mode single-table and multi-table prove runs produce proofs (verified in Phase 4).

**Phase 4 — Verifier wiring.** §3.6 into `multi_verify_views` + rounds replay + balance check. Add the branch's DoS guards AND the residual fix: bound `RoundPoly.evals` length (exactly 4 for degree-3 gates; reject otherwise — never assert), guard `1u64 << n_unused`, validate `trace_length`. Gate: GKR-mode prove→verify roundtrip green on unit AIRs + fibonacci examples; tampering any transcript-bound value (root_claims, column_claims, child_claims, σ OOD, aux root) → reject.

**Phase 5 — Test port + new tests.** Re-author the branch's `bus_tests` additions against the current harness (list in branch-analysis §8): the 5 GKR soundness tests, the single-table completeness test; KEEP the standard-mode tests they replaced (both modes coexist now — do not delete `test_tampered_table_contribution` etc.). Add: (a) a mode-mismatch test (Standard verifier rejects GKR proof and vice versa), (b) the **fabricated-leaves forgery test** for the §6 gap, `#[ignore]`d with a comment naming the gap (it documents the known hole and becomes the acceptance test for the follow-up fix). Gate: full `cargo test` workspace green; `make lint` clean.

**Phase 6 — System validation.**
1. **Standard-mode wire-identity (the king gate):** cross-version verification against an origin/main binary, both directions, examples + full VM programs (6/6). Any mismatch with GKR disabled = a bug in the mode plumbing.
2. **GKR-mode e2e:** full VM prove+verify on the 6 test programs + `executor/tests/ethrex_{5,20}_transfers.bin`, GKR on, CPU path.
3. **Memory + time:** the PR-comment bench on the bench server (`/bench` for the cheap tier; `/bench-abba` for the paired run) — GKR mode on vs main. Expect the heap win to have shifted since April (table count ~doubled; aux-heavy tables like CPU K=40: 21→2 aux cols still apply). Report peak-heap and prove-time deltas in the PR description. Do not run benches locally.
4. **Recursion-guest impact (measure, don't fix):** build the recursion guest with a GKR proof as input and record guest-cycle delta vs standard (the GKR verifier adds sumcheck transcript work in-circuit). This number decides whether recursion needs its own follow-up.

**Phase 7 — PR.** Draft PR referencing #485, with: motivation (memory numbers old + new), mode semantics, the documented §6 gap + follow-up plan, exclusions (drift-analysis §5). No AI attribution.

## 5. Defect fixes folded into the port (small, mandatory)

| Defect (branch-analysis ref) | Fix |
|---|---|
| Dead `LogUpGkrProof.random_point` + stub `gkr_proof` fields (§5) | Drop; keep only `column_claims` per table + the batch proof |
| `RoundPoly` deserialization → `sum_at_binary`/`evaluate` panic on <2 evals (§7.3) | Enforce eval-count == 4 at view/deserialization boundary; make the sumcheck helpers fallible or pre-checked |
| `1u64 << n_unused` debug overflow (§7.3) | Validate instance size spread / trace_length upstream; checked shift |
| Fork-time L absorb silently dead in GKR mode (§4 note) | Explicit mode-checked skip on both sides |

## 6. KNOWN OPEN SOUNDNESS GAP — documented, follow-up, not this PR

`reconstruct_and_verify_gkr_claims` is **fail-open for multi-interaction tables** (every production table): nothing binds the batch-GKR leaf claims `(n̂, d̂)` to the committed columns — the bridge binds only `column_claims ↔ trace`, and the leaf fraction is a nonlinear (cross-multiplied) combination of columns, so MLE evaluation doesn't factor through `column_claims`. A malicious prover can run an honest GKR over fabricated leaves (arbitrary root contribution → fake bus balance) while all present checks pass. Full analysis: branch-analysis §7 item 2.

Why not fix it in this PR: it is a protocol change to the GKR **input layer**, and mixing it into the port destroys the only available correctness reference (the branch's own behavior). Port faithfully first, keep the experiment measurable, land the fix as the immediate next PR before any production consideration.

Fix directions to evaluate in the follow-up (in rough preference order):
1. **Input-layer sumcheck**: one extra sumcheck reducing each instance's leaf claims `(n̂, d̂)` at point r to claims about the K column MLEs at r (the leaf is a low-degree polynomial in the columns per row — this is the standard LogUp-GKR construction; see Winterfell's and Stwo's logup-gkr input layers for reference shapes).
2. **Per-interaction instances**: leaves = `(±m_k, fp_k)` per interaction (numerator/denominator LINEAR in columns) → K GKR instances per table; MLE then commutes with the linear map, so `column_claims` directly verify the leaf claims. More instances, but the batch already handles mixed sizes.
The `#[ignore]`d forgery test from Phase 5 is the acceptance criterion.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| FS-order mistake in the B′/B″ insertion | The order is specified twice (branch-analysis §4 = branch behavior; §3.5/3.6 here = target). Prover and verifier are written from the SAME spec; every tamper test doubles as an FS check |
| Breaking standard mode subtly | Mode plumbed as data, no behavioral change when `Standard`; Phase 6.1 cross-version gate is decisive |
| GPU path fires on a GKR-shaped table | Explicit mode condition in `device_only_gate` + aux/composition `try_*` entries; add a debug assert that resident-aux never coexists with `Gkr` mode |
| Guest bloat / cycle blowup in recursion | Phase 6.4 measures; no HashMap audit in Phase 0; follow-up decides |
| rkyv view layer holes (panics on malformed archives) | Phase 2 malformed-bytes tests; length checks at accessor level (§5) |
| Memory win regressed by 26-table batch (more instances, deeper trees) | Layer trees are transient; free before LDE (§3.5). Phase 6.3 measures the real number |
| `trailing_zeros` on non-power-of-two/zero trace length | Validated upfront in verifier (§3.6) |

## 8. Follow-up queue (explicitly out of scope here)

1. **Leaf-binding soundness fix** (§6) — required before any production use.
2. **GPU support for GKR mode**: device build of `l`/`σ`, bridge constraint through the fused composition kernel (it's already just an emitted constraint → captured IR; the challenge plumbing is the work), GKR layer/sumcheck kernels.
3. **PR #489 disk-spill** re-evaluation on top (orthogonal memory scaling).
4. **Rayon FRI fold** (`3aa03e6c`) as a separate tiny perf PR — genuinely absent on main.
5. Recursion-guest optimization of the GKR verifier if Phase 6.4 numbers demand it (batched-FRI work #768 is the sibling effort).
