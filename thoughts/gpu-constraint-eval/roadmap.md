# GPU constraint evaluation — implementation status & execution plan

**Handoff doc.** Self-contained enough to continue without the originating discussion.
Describes the code as currently built, the decisions already made, and a detailed
checkbox plan to take it to a working, GPU-validated constraint evaluator.

---

## Goal & motivation

Evaluate STARK **transition constraints on the GPU**, end-to-end, producing the
composition-polynomial evaluations **on-device**. The point is **data residency**, not
constraint-eval speed (constraints are not the prover bottleneck): once LDE/Merkle/FRI run
on the GPU, evaluating constraints on the CPU forces a D2H round-trip of the (large) LDE
trace, which dominates. Keeping eval on-device removes that transfer.

## Architecture (decided)

Capture each table's constraints **once** into a flat, single-field **Goldilocks IR**
(typed `Dim1`=base `u64` / `Dim3`=degree-3 extension `[u64;3]` op-DAG), then **interpret**
that IR on CPU and GPU. One universal kernel; the per-table difference is data. Modeled on
OpenVM's `cuda-backend` (cloned at `others/openvm-stark-backend`, the closest reference —
same FRI-STARK / LDE-quotient protocol; better-matched than SP1).

### Decisions already made (don't relitigate without reason)
- **Capture front-end = Plan A (symbolic field)**: a recording field type whose ops build
  IR nodes; running a constraint's *existing* generic `evaluate::<SymField,SymExt>` emits
  the IR with **zero edits to constraint bodies**. (Plan B = rewrite each body to a builder;
  kept as fallback in `plan-builder-rewrite.md`. Stay on A unless LogUp capture gets messy.)
- **Backend = interpreter, not codegen** for v1. Codegen stays available later from the same IR.
- **GPU value array = global memory, no register allocation** to start (simplest, works for
  all program sizes). Add register allocation only if profiling needs it (Phase 6).
- **Keep the existing boxed CPU path** as the default + differential oracle behind a toggle.
- **Device field arithmetic already exists** — reuse `crypto/math-cuda/kernels/ext3.cuh`
  (`ext3::{add,sub,mul,mul_base}`, where `mul_base` = base×ext) and `kernels/goldilocks.cuh`.
  Do **not** build new field math.

---

## Phase 0 — CPU spike  ✅ DONE (PR #737, branch `spike/constraint-ir-symfield`)

Implemented and validated (builds, fmt/clippy clean, diff test green). Covers **base-field
algebraic constraints only**, single step (offset 0, row 0), main columns only — no aux,
no next-row, no LogUp, no uniforms, not wired into the prover, no GPU.

Files (all new under `crypto/stark/src/symbolic/`):
- `ir.rs` — `enum Dim { D1, D3 }`; `enum Op { Const1(u64), Const3([u64;3]), Var { main: bool, offset: u8, row: u8, col: u16 }, Add(u32,u32), Sub(u32,u32), Mul(u32,u32), Neg(u32), Embed(u32) }`; `struct ConstraintProgram { nodes: Vec<Op>, dims: Vec<Dim>, roots: Vec<u32> }`. Typing: `(D1,D1)->D1`, any `D3` operand -> `D3` (auto-embed); `Embed: D1->D3`.
- `sym_field.rs` — `SymField` (records `D1`), `SymExt` (records `D3`), `SymId { id: u32, dim: Dim }` (= `BaseType` for both). Thread-local `Arena { nodes, dims, cse: HashMap<(Op,Dim),u32> }` with hash-consing; `with_arena(f) -> (nodes, dims, R)`, `record`/`record_leaf`, `leaf_base`/`leaf_ext`. Node id 0 reserved = `Const1(0)` so `SymId::default()` is base-zero. `impl IsField for SymField`/`SymExt` (ops record nodes; `inv`/`div` = `unimplemented!()`; `eq` = `false`; `from_u64` folds the real Goldilocks reduction). `impl IsSubFieldOf<SymExt> for SymField` (mixed ops -> `D3`, `embed` -> `Embed`). `impl ByteConversion for SymId` = `unimplemented!()` stubs.
- `capture.rs` — `capture_constraint<T: TransitionConstraint<GoldilocksField, GoldilocksExtension>>(c: &T, num_main_cols: usize) -> ConstraintProgram`: builds a symbolic `TableView<SymField,SymExt>` (1 step, 1 row, `num_main_cols` `Var{main:true}` leaves; aux empty), runs `c.evaluate::<SymField,SymExt>(step)`, snapshots the arena, root = result id.
- `interp.rs` — `eval_program_base(prog, main_row: &[FieldElement<GoldilocksField>]) -> FieldElement<GoldilocksField>`: forward pass over nodes into a `Value { D1 | D3 }` array, reusing real `FieldElement` arithmetic; resolves `Var{col}` from `main_row`.
- `mod.rs` — wires submodules; re-exports `capture_constraint`, `eval_program_base`.

Also: `crypto/stark/src/lib.rs` (`pub mod symbolic;`); test `prover/src/tests/symbolic_ir_tests.rs` (+ registered in `prover/src/tests/mod.rs`) — captures `IsBit` (cond/uncond), `AddConstraint` (both carries), `ProductZero`, asserts the interpreter == real `evaluate` bit-for-bit over 1000 random rows (deterministic SplitMix64).

**Key fact this proved:** `SymField` only needs `IsField` + `IsSubFieldOf<SymExt>` (capture
never instantiates `AIR<Field=SymField>`); `IsFFTField`/`IsPrimeField`/`inv`/`div`/real
`ByteConversion` are all unreachable. Per-constraint IR is a handful of real nodes (the test
pads with 64 unused column leaves — a `capture_constraint` artifact; see Phase 1).

Run: `cargo test -p lambda-vm-prover symbolic_ir -- --nocapture`

---

## Phase 1 — Full CPU capture coverage (all constraints, prover + verifier)

Goal: capture **every** constraint of a real table into one `ConstraintProgram`, validated
on CPU. The GPU runs this same IR, so completeness/correctness must be nailed here first.

- [ ] **Extend the IR** (`ir.rs`): add leaf `Op` variants for the per-proof/per-row uniforms — `Periodic { idx }` (D1), `RapChallenge { idx }` (D3), `AlphaPow { idx }` (D3), `TableOffset` (D3), `Shift { which: u8 }` (D1). `Op::Var` already carries `offset`/`row`/`main` for next-row + aux reads.
- [ ] **Object-safe capture on the evaluator trait** (`crypto/stark/src/constraints/transition.rs`): add `fn capture(&self, ctx: &SymCaptureCtx, base: &mut Vec<SymId>, ext: &mut Vec<SymId>)` to `TransitionConstraintEvaluator` (object-safe: no generics in the signature; concrete sym types). Adapter (`TransitionConstraintAdapter`) override runs `self.0.evaluate::<SymField,SymExt>` → push a `D1` root for base constraints (`idx < num_base`), else `D3`.
- [ ] **Symbolic capture context** (`crypto/stark/src/symbolic/capture.rs`): build a `TransitionEvaluationContext::Prover` over sym types — a 2-step `Frame<SymField,SymExt>` of `Var` leaves at offsets `{0,1}` (main + aux), and uniform slices (`rap_challenges`, `logup_alpha_powers`, `logup_table_offset`, `packing_shifts`, `periodic_values`) filled with the new `Leaf` nodes.
- [ ] **Capture the LogUp constraints** (`crypto/stark/src/lookup.rs`): override `capture` for `LookupBatchedTermConstraint` (~line 1741) and `LookupAccumulatedConstraint` (~1868) to run their already-field-generic inner fns (`compute_fingerprint_from_step`, `compute_multiplicity_from_step`, the `evaluate_*_constraint` fns) under the sym context. The accumulated constraint reads aux at offset 0 **and** 1 → the 2-step symbolic frame handles it. `is_sender` is a compile-time bool → resolves at capture. (Fallback if this fights: hand-emit the 2 fixed LogUp programs; they're table-independent.)
- [ ] **Per-table program** (`crypto/stark/src/traits.rs`): `fn constraint_program(&self) -> ConstraintProgram` default that iterates `self.transition_constraints()`, captures all (inside one `with_arena`), writes `roots[constraint_idx()]`, sets `num_base = num_base_transition_constraints()`.
- [ ] **Full interpreter** (`interp.rs`): generalize to `eval_program(prog, inputs) -> (base: Vec<FE<F>>, ext: Vec<FE<E>>)` matching the `compute_transition_prover` contract — resolve all leaf kinds + offsets + aux; Dim1/Dim3 with auto-embed; add a verifier entry (all-D3 frame at the OOD point). Fix the `capture_constraint` leaf padding (record a `Var` leaf only when a column is actually read, or DCE unused leaves).
- [ ] **Acceptance test:** for the CPU table + ≥1 LogUp-heavy table, capture the full program, interpret per-row over a real LDE, and `assert_eq!` against `air.compute_transition_prover(...)` bit-for-bit; same for the verifier vs `air.compute_transition(...)` at the OOD point.

---

## Phase 2 — Wire interpreter into prover/verifier (CPU), behind a toggle

- [ ] Add a `symbolic-interp` Cargo feature (or runtime env toggle) in `crypto/stark/Cargo.toml`.
- [ ] Cache the `ConstraintProgram` once in `ConstraintEvaluator::new` (`crypto/stark/src/constraints/evaluator.rs`).
- [ ] In `evaluate_transitions` (same file), behind the toggle, replace the `air.compute_transition_prover(&ctx, base_buf, transition_buf)` call (~line 100) with the IR interpreter; keep the boxed path as default + oracle. Leave the `Σ βᵢ·Cᵢ·Zᵢ⁻¹ + boundary` accumulation untouched.
- [ ] Verifier: same swap at `crypto/stark/src/verifier.rs` (`air.compute_transition`).
- [ ] **Acceptance:** full prove→verify suite passes with the toggle ON, across all tables — `cargo test --release -p lambda-vm-prover` (incl. `test_prove_elfs_*`). This is the **CPU end-to-end** checkpoint; the IR is now proven complete and correct independent of GPU.

---

## Phase 3 — Device field primitives  ✅ ALREADY EXIST

Reuse `crypto/math-cuda/kernels/ext3.cuh` (`ext3::Fe3`, `ext3::{add,sub,mul,mul_base}`) and
`kernels/goldilocks.cuh`. Already used by the GPU FRI/inverse/barycentric/deep kernels.
Only remaining: confirm a `neg` (else `ext3::sub(zero, x)`) and include the header — do it
as part of Phase 4.

---

## Phase 4 — GPU interpreter kernel

Start **stripped** (mirror OpenVM's `GLOBAL=true` kernel: global-memory value array, no
register allocation, no bit-packed codec). Reference: `others/openvm-stark-backend/crates/cuda-backend/cuda/src/quotient.cu` (`cukernel_quotient`) and `cuda/include/codec.cuh`.

- [ ] **Device IR layout** (`crypto/stark/src/symbolic/` + `crypto/math-cuda`): serialize `ConstraintProgram` to a `#[repr(C)]` flat node array (`{ op_tag: u32, a: u32, b: u32, dim: u32 }`) + a constants table + `roots` + `num_base`. Plus per-proof uniform device buffers (rap challenges, alpha powers, table_offset, periodic columns, shift consts).
- [ ] **Kernel** (`crypto/math-cuda/kernels/constraint_interp.cu` + Rust wrapper in `crypto/math-cuda/src/`): one thread per LDE row (tiled). Forward pass over the node array into a **per-thread value array in global memory** (one slot per node, strided per thread). Resolve `Var{main/aux, offset, col}` from the device-resident LDE columns (`GpuLdeBase`/`GpuLdeExt3` keep-handles from `trace.gpu_main()`/`gpu_aux()`). Dim1 ops via `goldilocks.cuh`, Dim3 via `ext3.cuh` (`mul_base` for D1×D3). **Fused accumulation:** Horner `acc = acc*alpha + Cᵢ` over the `roots` flagged as constraints, then `acc *= inv_zeroifier[row]` → write the composition-poly evaluation. Output stays on device.
- [ ] **Host dispatch** (`crypto/stark/src/symbolic/gpu_interp.rs`): `try_eval_program_gpu<F,E>(...) -> Option<...>` gated on `TypeId::of::<F>() == GoldilocksField && TypeId::of::<E>() == Degree3GoldilocksExtensionField` + a size threshold (mirror `crypto/stark/src/gpu_lde.rs:119-152`). Upload program + uniforms once; launch; leave output device-resident. Fall back to the CPU interpreter / boxed path otherwise.
- [ ] **Pipeline integration:** add a whole-table GPU entry (e.g. `AIR::compute_transitions_batched(lde) -> Option<DeviceBuf>` tried by `evaluate_transitions` before the per-row loop) so the composition-poly evals are produced on-device and feed the existing GPU Merkle commit with **no D2H of the `Cᵢ` matrix**. Reconcile zerofier/boundary accounting with the CPU semantics.
- [ ] **Acceptance:** compiles under `cargo build -p lambda-vm-prover --features cuda`.

---

## Phase 5 — "Working on GPU" (the deliverable) — runs on the CUDA machine

- [ ] **GPU↔CPU parity test** (extend `prover/tests/cuda_path_integration.rs` / `cuda_fallback_tests.rs`): composition-poly evals on GPU == CPU interpreter == boxed path, per table, on real traces.
- [ ] **End-to-end GPU prove→verify** on a real ELF with `--features cuda`. A passing verify is the goal.
- [ ] **Benchmark** (bench server): prove time with GPU constraints vs CPU constraints — confirm the data-residency win (no LDE D2H for constraint eval).

---

## Phase 6 — Optimizations (only if a profile demands)

- [ ] **Register allocation** — port OpenVM's transpiler liveness + linear-scan (`others/openvm-stark-backend/crates/cuda-backend/src/transpiler/mod.rs`) to shrink the per-thread value array (local `FpExt[N]` for small programs, smaller global buffer for large ones like Dvrm/Shift/ecsm).
- [ ] **DCE / const-fold peephole** — drop unused column leaves; fold `×Const(0)`/`+Const(0)` left by the `eq=false` zero-skip.
- [ ] **Bit-packed codec** — only if H2D bandwidth shows up (unlikely; the rule stream is tiny and uploaded once).
- [ ] **Selective codegen** — given few-but-large tables, codegen the 1–3 hottest tables (nvcc does register allocation, no per-op dispatch) if interpreter overhead is material. Hybrid: interpreter baseline + codegen the hot ones.

---

## Gotchas / invariants

- **Single field:** Goldilocks base + degree-3 extension only. The IR's `Dim1`/`Dim3` and
  the `ext3.cuh` primitives cover everything.
- **Object safety:** generic methods can't live on `Box<dyn TransitionConstraintEvaluator>`.
  Capture runs once at setup (where concrete types exist), via the non-generic `capture`
  method. The per-row hot path only interprets the (data) IR.
- **`eq = false`:** defeats the runtime "skip zero term" optimization during capture, so the
  IR always emits the multiply — value-identical (×0 / +0 is a no-op), just unoptimized.
  A DCE/const-fold peephole (Phase 6) recovers it.
- **Don't D2H the `Cᵢ` matrix:** fuse the accumulation in the GPU kernel so only the
  (small) composition-poly evaluation crosses on-device into Merkle.
- **LDE columns are already device-resident** (`GpuLdeBase`/`GpuLdeExt3`); read them in place.

## Reference material (in-repo)

- `others/openvm-stark-backend/crates/cuda-backend/` — `src/transpiler/{mod.rs,codec.rs}`, `src/quotient/`, `cuda/src/quotient.cu`, `cuda/include/codec.cuh`. The closest working reference (BabyBear; for Goldilocks the only deltas are 64-bit constants needing a side table, degree-3 ext, and they run all-FpExt with no base/ext split).
- `crypto/stark/src/gpu_lde.rs` — the TypeId+transmute generic→concrete-Goldilocks GPU seam to mirror.
- `thoughts/gpu-constraint-eval/plan-symbolic-field.md` — the full Plan A design (this is what Phase 0 implemented; Phases 1+ detail its remaining sections).
- `thoughts/gpu-constraint-eval/plan-builder-rewrite.md` — Plan B (fallback capture front-end).
- `thoughts/gpu-constraint-eval/README.md` — motivation + the SP1/OpenVM/zisk survey.
