# Phase 0 handoff — constraint artifact track

Written 2026-07-30 by the phase0 agent. Branch `feat/phase0-constraint-ir`,
worktree `.../scratchpad/wt-phase0`, off `origin/main e0add1d5`. **Never pushed.**

Five commits, all green (`make lint` 0, stark 216, prover 530 lib tests):

```
b36f15fa  ConstraintArtifact + rkyv codec + the scoped verify-path unban
2a6f9036  all 28 production AIRs (3 continuation tables were in NO enumeration)
d2fb95c9  constraint-lowering design + the op census instrument
ef7587fd  design revised against the machine's real cost model
058ba5ef  per-epoch multiplier + the workload-shaped self-correction
1414d726  real continuation-epoch chunk counts, first-hand
69b3b348  uniform promotion reordered — epoch_label is the critical path
```

## State: what is done

**Phase 0 proper is complete.** Constraints serialize at build time
(`ConstraintArtifact` = flat program + zerofier metadata + AIR shape + degree
multiplier), round-trip bit-exactly against both folders on all 28 AIRs, and the
verify-path prohibition is scoped to CAPTURE with a guest-safe
`precaptured_constraint_program()` alongside. Nothing is wired into the
production verify path, as instructed.

**The lowering design is written and measured**
(`others/lfm-constraint-lowering-design.md`). Continuation epoch leg: 63,393
instructions at the minimum shape, 64,035 at a 2^20 epoch, 63–65K across any
plausible epoch size.

## State: what is NEXT, and it is not started

**The uniform promotion.** Fully specified in
`others/lfm-page-base-uniform-proposal.md`; **no code written**. Order, set by
the team lead and derived from the epoch composition:

1. **`epoch_label`** — the two L2G tables. FIRST, because an epoch proof's only
   parameterized AIR is `L2G_MEMORY`, so unpromoted the registry needs one
   program per epoch index and the ladder grows linearly with epoch count.
2. `page_base` — GLOBAL_MEMORY, when the global-proof leg comes into scope.
3. PAGE last (monolithic-only; gets the fix free once the mechanism exists).

### Read these three sections before writing anything

- **§4.3 — the `epoch_label` threat model.** The invariant is that the uniform is
  derived positionally from the verifier's own `enumerate()`, never from the
  bundle. Failure mode is epoch **replay or reorder**, not a wrong address.
- **§5.1 — the design refinement.** Uniforms resolve into
  `ConstraintProgram.base_uniforms` / `DeviceProgram.base_uniforms` alongside the
  constants, rather than being threaded as a new parameter through every
  evaluation entry point. Avoids churn across both walkers, the CUDA host side
  and every caller. **Carries a hazard**: `ConstraintProgram` becomes a hybrid of
  program identity and per-instance values; nothing must ever hash it including
  the uniforms. Latent today (only the artifact is hashed, and it stores the
  count only). **This is a design decision awaiting the lead's agreement, not a
  settled implementation detail.**
- **§4.3's three acceptance criteria**, of which the second is the real one:
  `test_split_verify_rejects_reordered_epochs` and
  `..._dropped_last_epoch` (`continuation.rs:1711`, `:1693`) must pass
  **unchanged**. They pop and swap epochs in a genuinely proved bundle. A
  promotion that required editing them broke something.

### Falsifications to run (not optional)

- Break the CPU walker and the CUDA walker **independently** and confirm the
  differential suites catch each. A suite never shown to catch a divergence is
  not yet a safety net — this repo's suites have now been shown to catch two
  distinct classes (structural wire change, and an evaluation-only change on the
  path with no structural check), so the bar is set.
- Delete `parameterized_airs_vary_per_parameter_value` only after showing it
  fails **for the right reason** (artifacts equal across labels), not merely that
  it fails.

## Instruments left behind (use them; do not re-derive)

All in `prover/src/tests/constraint_artifact_tests.rs`:

| test | answers |
|---|---|
| `constraint_op_census` | per-AIR instruction counts. **Read its "WHAT THIS INSTRUMENT CANNOT SEE" note first** — it cannot see how sub-proofs are assembled, and a census-only inference from it was the one thing this track got wrong. |
| `epoch_chunk_multiplier` | monolithic per-proof totals via real traces |
| `continuation_epoch_constraint_leg` | epoch composition, asserts the measured 24/25 sub-proof count |
| `continuation_epoch_chunk_counts_measured` | a real epoch's chunk counts, first-hand, no proving needed |
| `parameterized_airs_vary_per_parameter_value` | characterizes the four parameterized AIRs; becomes the promotion's falsifier |

Plus `prover/src/bin/compute_constraint_artifacts.rs` (generator; emits ONE
REPRESENTATIVE per parameterized table, not the full set — see its header) and
`crypto/stark/src/constraint_ir/artifact_tests.rs` (17 unit tests incl. the
rejection paths and nonzero `end_exemptions`, which no production AIR exercises).

## Things a successor would otherwise rediscover

- **`test_utils::production_airs()` is the single 28-AIR list**, and every suite
  asserts `NUM_PRODUCTION_AIRS`. That assert exists because three hand-copied
  lists all shared the same blind spot. Add tables there, once.
- **The IR's `dim` tags are prover-side.** The machine runs the verifier, where
  the frame is all-extension: 42,137 declared base, 2,916 actually base. Do not
  size anything from the declared dims.
- **Production zerofiers are uniform** (every AIR emits `RowDomain::ALL`), worth
  ≈50,900 instructions and the GPU path's precondition holding in fact.
- **Hash-consing makes peepholes unsound** without a single-consumer guard. This
  is documented on `ConstraintArtifact` itself, not just in the design doc.
- **Open, not mine to decide**: whether to check in generated artifacts (ruled
  no — generate at build time, pin by digest); and the `check_attestation`
  production gap, which is a real finding but not this track's.
