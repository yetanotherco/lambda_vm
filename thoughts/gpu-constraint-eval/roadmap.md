# GPU constraint evaluation — implementation status & execution plan

**Handoff doc.** Self-contained enough to continue without the originating discussion.
Describes the code as currently built, the decisions already made, and a detailed
checkbox plan to take it to a working, GPU-validated constraint evaluator.

> **Chosen capture front-end: Plan B (explicit `IrBuilder` + per-constraint `capture()`).**
> Two spikes were built to compare: Plan A (symbolic field) = PR #737 / branch
> `spike/constraint-ir-symfield`; Plan B (builder) = PR #739 / branch
> `spike/constraint-ir-builder`. Both pass the same bit-for-bit diff test and reuse the
> same IR + interpreter. **Plan B is the production direction** (cleaner end-state — no fake
> `IsField`, no thread-local arena, explicit/auditable). Plan A remains as PR #737 for
> reference / comparison.

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
- **Capture front-end = Plan B (explicit builder).** Each constraint implements an
  object-safe `Capture { fn capture(&self, &mut IrBuilder) }`, translating its `evaluate`
  body into builder calls (`main`/`aux`/`add`/`sub`/`mul`/`neg`/`const_*`/`emit`). No fake
  field, no arena, explicit and auditable. (Plan A — a recording "symbolic field" that
  captures with zero body edits — was spiked first to validate the IR/interpreter cheaply;
  kept as PR #737 + `plan-symbolic-field.md` for reference. We chose B for the cleaner
  production end-state.)
- **Backend = interpreter, not codegen** for v1. Codegen stays available later from the same IR.
- **GPU value array = global memory, no register allocation** to start (simplest, works for
  all program sizes). Add register allocation only if profiling needs it (Phase 6).
- **Keep the existing boxed CPU path** as the default + differential oracle behind a toggle
  (the `capture()` methods are added alongside `evaluate`, which stays).
- **Device field arithmetic already exists** — reuse `crypto/math-cuda/kernels/ext3.cuh`
  (`ext3::{add,sub,mul,mul_base}`, where `mul_base` = base×ext) and `kernels/goldilocks.cuh`.
  Do **not** build new field math.

---

## Phase 0 — CPU spikes  ✅ DONE (two draft PRs; Plan B is the production base)

Both spikes build, are fmt/clippy clean, and pass a bit-for-bit diff test (capture →
interpret == real `evaluate`, 1000 random rows) for `IsBit`/`Add`/`ProductZero`. They cover
**base-field algebraic constraints only**, single step (offset 0, row 0), main columns only
— no aux, no next-row, no LogUp, no uniforms, not wired into the prover, no GPU.

**Shared (identical in both):** the IR and the CPU interpreter.
- `ir.rs` — `enum Dim { D1, D3 }`; `enum Op { Const1(u64), Const3([u64;3]), Var { main: bool, offset: u8, row: u8, col: u16 }, Add(u32,u32), Sub(u32,u32), Mul(u32,u32), Neg(u32), Embed(u32) }`; `struct ConstraintProgram { nodes: Vec<Op>, dims: Vec<Dim>, roots: Vec<u32> }`. Typing: `(D1,D1)->D1`, any `D3` operand -> `D3` (auto-embed); `Embed: D1->D3`.
- `interp.rs` — `eval_program_base(prog, main_row) -> FieldElement<GoldilocksField>`: forward pass over nodes into a `Value { D1 | D3 }` array, reusing real `FieldElement` arithmetic; resolves `Var{col}` from the row.

**Plan B — the production base (PR #739, branch `spike/constraint-ir-builder`).** Module
`crypto/stark/src/constraint_ir/`:
- `ir.rs`, `interp.rs` — the shared IR + interpreter above (reused verbatim).
- `builder.rs` — `IrBuilder` (hash-conses nodes on `(Op, Dim)`, dedups base constants by value, dim-join `(D1,D1)->D1` else `D3`, reserves id 0 = `Const1(0)`) + `Expr { id, dim }`. Methods: `main(offset,col)`/`aux(offset,col)`, `const_base`/`const_signed`/`one`, `add`/`sub`/`mul`/`neg`, `emit(constraint_idx, e)`, `finish() -> ConstraintProgram`.
- `mod.rs` — object-safe `pub trait Capture { fn capture(&self, &mut IrBuilder); }`.
- Constraint impls (added **alongside** the unchanged `evaluate`, non-destructive): `IsBitConstraint`, `AddConstraint` (incl. `AddOperand`/`AddLinearTerm` lo/hi-limb mapping with i64 coeffs + the `inv_2_32` constant), `ProductZeroConstraint`.
- `prover/src/tests/constraint_ir_tests.rs` — the diff test. Node counts: product_zero **4**, is_bit_uncond **5**, is_bit_cond **7**, add_carry_0 **14**, add_carry_1 **21** (minimal — the builder only emits leaves for columns actually read).
- Run: `cargo test -p lambda-vm-prover constraint_ir_tests -- --nocapture`

**Plan A — reference only (PR #737, branch `spike/constraint-ir-symfield`).** Module
`crypto/stark/src/symbolic/` (`sym_field.rs` recording field + capture). Retired the
"can a symbolic type satisfy `IsField`?" question (yes — needs only `IsField` +
`IsSubFieldOf`; capture never builds `AIR<Field=SymField>`). Not the production path.

---

## Phase 1 — Full Plan-B capture coverage (all constraints, prover + verifier)

Goal: implement `Capture` for **every** constraint of a real table (all ~33 algebraic + the
2 LogUp), for both prover and verifier, validated on CPU. The GPU runs this same IR, so
completeness/correctness must be nailed here first.

- [ ] **Extend the IR** (`constraint_ir/ir.rs`): add leaf `Op` variants for the per-proof/per-row uniforms — `Periodic { idx }` (D1), `RapChallenge { idx }` (D3), `AlphaPow { idx }` (D3), `TableOffset` (D3), `Shift { which: u8 }` (D1). `Op::Var` already carries `offset`/`row`/`main` for next-row + aux reads.
- [ ] **Extend `IrBuilder`** (`constraint_ir/builder.rs`): add leaf constructors for the uniforms (`challenge`, `alpha_power`, `periodic`, `table_offset`, `shift`) and `const_ext([u64;3])`; ensure `aux(offset, col)` supports `offset=1` (next row). Make `emit` index `roots` by `constraint_idx` (the spike stores in emit order — switch to indexed for the full per-table program).
- [ ] **`Capture` for the remaining ~30 algebraic constraints** — mechanical translation of each `evaluate`/`compute` body to builder calls. Files: `prover/src/constraints/cpu.rs` (Arg2Exclusive, MemFlagsBit, RegNotReadIsZero, Arg2, RvdEqRes, BranchRvd, BranchCond, NextPcAdd), `prover/src/tables/{branch,commit,cpu32,dvrm,eq,ec_scalar,ecdas,ecsm,keccak,load,lt,memw,memw_aligned,memw_register,mul,shift,store}.rs`. The multi-kind ones (Dvrm 11 / Cpu32 8 / Shift 7 / Lt·Load·Mul 6) are the bulk — their `compute()` loops are statically bounded, so they unroll into builder calls at capture time.
- [ ] **`Capture` for the 2 LogUp constraints (the crux)** — `LookupBatchedTermConstraint` and `LookupAccumulatedConstraint` (`crypto/stark/src/lookup.rs`). Translate their bodies to builder calls: fingerprint = `challenge − Σ alpha_power·col` (mirror `compute_fingerprint_from_step`/`Packing::accumulate_fingerprint_with`), multiplicity (mirror `Multiplicity::evaluate_with`), `is_sender` as a compile-time `add` vs `neg`, the `c·fp_a·fp_b − …` / accumulated formulas. The accumulated one reads aux at offset 0 **and** 1 → use `aux(1, col)`. This is more work than Plan A's auto-capture (Plan A's inner fns were already field-generic) but is explicit and lives in one place.
- [ ] **Per-table program** (`crypto/stark/src/traits.rs`): `fn constraint_program(&self) -> ConstraintProgram` — iterate `self.transition_constraints()`, call `capture` on each into one `IrBuilder`, `roots[constraint_idx()]`, `num_base = num_base_transition_constraints()`. (Requires the object-safe `Capture` to be reachable from the boxed `TransitionConstraintEvaluator` — add `capture` to that trait, which is object-safe and matches the production design.)
- [ ] **Full interpreter** (`constraint_ir/interp.rs`): generalize to `eval_program(prog, inputs) -> (base: Vec<FE<F>>, ext: Vec<FE<E>>)` matching the `compute_transition_prover` contract — resolve all leaf kinds + offsets + aux; Dim1/Dim3 with auto-embed; add a verifier entry (all-D3 frame at the OOD point).
- [ ] **Acceptance test:** for the CPU table + ≥1 LogUp-heavy table, capture the full program, interpret per-row over a real LDE, and `assert_eq!` against `air.compute_transition_prover(...)` bit-for-bit; same for the verifier vs `air.compute_transition(...)` at the OOD point.

---

## Phase 2 — Wire interpreter into prover/verifier (CPU), behind a toggle

- [ ] Add a `constraint-ir` Cargo feature (or runtime env toggle) in `crypto/stark/Cargo.toml`.
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

- [ ] **Device IR layout** (`crypto/stark/src/constraint_ir/` + `crypto/math-cuda`): serialize `ConstraintProgram` to a `#[repr(C)]` flat node array (`{ op_tag: u32, a: u32, b: u32, dim: u32 }`) + a constants table + `roots` + `num_base`. Plus per-proof uniform device buffers (rap challenges, alpha powers, table_offset, periodic columns, shift consts).
- [ ] **Kernel** (`crypto/math-cuda/kernels/constraint_interp.cu` + Rust wrapper in `crypto/math-cuda/src/`): one thread per LDE row (tiled). Forward pass over the node array into a **per-thread value array in global memory** (one slot per node, strided per thread). Resolve `Var{main/aux, offset, col}` from the device-resident LDE columns (`GpuLdeBase`/`GpuLdeExt3` keep-handles from `trace.gpu_main()`/`gpu_aux()`). Dim1 ops via `goldilocks.cuh`, Dim3 via `ext3.cuh` (`mul_base` for D1×D3). **Fused accumulation:** Horner `acc = acc*alpha + Cᵢ` over the constraint roots, then `acc *= inv_zeroifier[row]` → write the composition-poly evaluation. Output stays on device.
- [ ] **Host dispatch** (`crypto/stark/src/constraint_ir/gpu_interp.rs`): `try_eval_program_gpu<F,E>(...) -> Option<...>` gated on `TypeId::of::<F>() == GoldilocksField && TypeId::of::<E>() == Degree3GoldilocksExtensionField` + a size threshold (mirror `crypto/stark/src/gpu_lde.rs:119-152`). Upload program + uniforms once; launch; leave output device-resident. Fall back to the CPU interpreter / boxed path otherwise.
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
- [ ] **DCE / const-fold peephole** — fold `×Const(0)`/`+Const(0)`; drop dead nodes.
- [ ] **Bit-packed codec** — only if H2D bandwidth shows up (unlikely; the rule stream is tiny and uploaded once).
- [ ] **Selective codegen** — given few-but-large tables, codegen the 1–3 hottest tables (nvcc does register allocation, no per-op dispatch) if interpreter overhead is material. Hybrid: interpreter baseline + codegen the hot ones.

---

## Gotchas / invariants

- **Single field:** Goldilocks base + degree-3 extension only. The IR's `Dim1`/`Dim3` and
  the `ext3.cuh` primitives cover everything.
- **Object safety:** generic methods can't live on `Box<dyn TransitionConstraintEvaluator>`.
  Plan B's `Capture` trait is **non-generic** (concrete `IrBuilder`), so it's object-safe;
  capture runs once at setup, the per-row hot path only interprets the (data) IR.
- **Don't D2H the `Cᵢ` matrix:** fuse the accumulation in the GPU kernel so only the
  (small) composition-poly evaluation crosses on-device into Merkle.
- **LDE columns are already device-resident** (`GpuLdeBase`/`GpuLdeExt3`); read them in place.
- *(Plan A only, not B:)* the symbolic-field path needed `eq → false` to defeat the runtime
  zero-skip during capture. Plan B has no such hack — it emits exactly what `capture` writes.

## Reference material (in-repo)

- `others/openvm-stark-backend/crates/cuda-backend/` — `src/transpiler/{mod.rs,codec.rs}`, `src/quotient/`, `cuda/src/quotient.cu`, `cuda/include/codec.cuh`. The closest working reference (BabyBear; for Goldilocks the only deltas are 64-bit constants needing a side table, degree-3 ext, and they run all-FpExt with no base/ext split).
- `crypto/stark/src/gpu_lde.rs` — the TypeId+transmute generic→concrete-Goldilocks GPU seam to mirror.
- `thoughts/gpu-constraint-eval/plan-builder-rewrite.md` — the full Plan B design (the chosen approach; Phases 1+ detail its remaining sections).
- `thoughts/gpu-constraint-eval/plan-symbolic-field.md` — Plan A (the reference/comparison spike, PR #737).
- `thoughts/gpu-constraint-eval/README.md` — motivation + the SP1/OpenVM/zisk survey.
- PRs: **#739** (Plan B, production base) · **#737** (Plan A, reference).
