# Merge plan: bring `blake3-real-hash` up to date with `main`

**Goal:** merge current `origin/main` into the campaign branch so it is testable/explorable
against latest, keeping main's constraint-IR redesign authoritative and adapting the branch's
build-time constraint-artifact feature to it. PR #930 ends up up to date.

Grounded in two read-only investigations (this dir): `main-ir-spec.md`, `artifact-feature-map.md`.

## The core problem, settled

Main redesigned the device-IR: `DeviceProgram::lower()` now runs a liveness slot allocator and
encodes operands as `kind<<29 | payload` (OPK-tagged slots/uniforms), drops dead nodes + uniform
leaves, replaces per-node `dim` with a `res` slot word, and adds `num_base_slots`/`num_ext_slots`.
The OLD `lower()` was a 1:1 image of `ConstraintProgram` with **node-index** operands and per-node
`dim`. The branch's artifact serialized that OLD (node-index) form and its consumers
(`validate_self`, `program()`, the census, and the whole `prover/src/lfm/` recursion machine)
assume node-index operands.

New `lower()` is **lossy and one-way** → `DeviceProgram → ConstraintProgram` is impossible, and
`eval_program`/`eval_program_verifier` need the node-index `ConstraintProgram` form. So the artifact
**must** keep a node-index form. `ir.rs` (`ConstraintProgram`/`Op`/`Dim`) is byte-identical on both
branches and carries no serde/rkyv derives.

## The chosen approach — A (decoupled)

The artifact owns a POD node type `ArtifactNode { op, a, b, dim }` (rkyv, `#[repr(C)]`, node-index
operands) — i.e. exactly the OLD `DeviceNode` — decoupled from main's now-slot-based `DeviceNode`.
It serializes that; `program()` lifts it to a `ConstraintProgram` (unchanged); `device_program()`
re-derives the flat blob through main's production `DeviceProgram::lower(&self.program())`.
Soundness preserved: the device blob goes through the same `lower()` the prover/GPU use, and
`program()` still lifts to the `ConstraintProgram` the compiled folders are pinned against.

## Steps

**Setup (keep the clean branch pristine until validated):**
1. Tag the clean tip: `git tag blake3-campaign-preMerge ed1b7785`.
2. Dedicated worktree on a new branch: `git worktree add ../lambda_vm-blake3-merge -b blake3-real-hash-mainmerge blake3-real-hash`.

**Merge + mechanical conflicts (known from the trial merge):**
3. `git merge --no-commit --no-ff origin/main`.
4. Resolve 5 conflicts: `lookup.rs` (main's `Arc` + our `precaptured_program`, both), `continuation.rs`
   (`#[derive(Clone,Copy)] pub(crate)`), 3 test files (keep our generic `production_airs()` iteration).
5. HINT coverage gap: add `HINT` to `production_airs()`, `NUM_PRODUCTION_AIRS` 28→29.
6. Shared IR files (`device.rs`, `ir.rs`, `interp.rs`, `builder.rs`, `gpu_interp.rs`): main's versions win.

**Artifact reconciliation (approach A — the real work, artifact.rs):**
7. Define `ArtifactNode { op:u32, a:u32, b:u32, dim:u32 }` (rkyv derives, `#[repr(C)]`) + local
   `DIM_BASE`/`DIM_EXT` consts in `artifact.rs`.
8. `ConstraintArtifact.nodes: Vec<ArtifactNode>` (drop the DeviceNode dependence + the broken
   slot-size fields from the trial merge).
9. `capture()`: build `ArtifactNode`s via the OLD 1:1 map from `ConstraintProgram` (op tag, node-index
   a/b, dim), take `roots`/`num_base` from `prog` — NOT from `DeviceProgram::lower`.
10. `device_program()`: `DeviceProgram::lower(&self.program())`.
11. `program()` / `validate_self()`: unchanged (they already assume node-index — now correct).
12. Tests: `constraint_artifact_tests.rs` (census `DIM_BASE` import → artifact's) and
    `lfm/constraint_tests.rs:658` (`DeviceNode{...,dim}` literal → `ArtifactNode`).
13. `prover/src/lfm/*` — NO change (they read `artifact.program()`).

**Validation (round-trip FIRST — the direct signal that broke):**
14. `cargo check` → `constraint_artifact` suite (MUST pass) → `lfm::` (expect 306/19) →
    full lib suite categorized (confirm only pre-existing fixture/env failures, nothing in the edited
    modules) → chip gate `artifact_pin.py --check` (BLAKE3 chip unchanged by the merge).
15. Baseline: run the full lib suite on `blake3-campaign-preMerge` too, to diff pre-existing vs new.

**Finalize:**
16. Adversarial review of the artifact reconciliation (soundness-adjacent).
17. Once green + reviewed: fast-forward `blake3-real-hash` to the merged branch, push → PR #930 up to date.
    Keep `blake3-campaign-preMerge` tag as the recoverable pristine point.
