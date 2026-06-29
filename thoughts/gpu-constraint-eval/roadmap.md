# Roadmap: from the CPU spike to working GPU constraint evaluation

Goal: evaluate STARK transition constraints **on the GPU**, end-to-end, producing the
composition-polynomial evaluations on-device (so the LDE never round-trips to host) — and
have it verified by a real prove→verify on GPU hardware.

Approach: **Plan A (symbolic field)** capture → single-field Goldilocks IR → interpreter,
on CPU and GPU. Backend modeled on OpenVM's `cuda-backend` (`others/openvm-stark-backend`).

The IR + interpreter is the contract: **validate it fully on CPU first**, then run the
*same* IR on GPU. Every phase ends in a green test.

Legend: 🟢 done · 🔜 critical path to "GPU working" · ⏭ optimization (after working).

---

## Phase 0 — CPU spike 🟢 DONE (PR #737)

`SymField`/`SymExt` + IR + CPU interpreter + diff test for base-field algebraic
constraints. Proves capture is feasible and the IR matches `evaluate` bit-for-bit.

---

## Phase 1 — Full CPU capture coverage 🔜

Make the IR cover **every** constraint of a real table (all 33 algebraic + the 2 LogUp),
for both prover and verifier. The GPU runs this same IR, so it must be complete here.

1.1 **Object-safe capture entry.** Add `fn capture(&self, ctx: &SymCaptureCtx, base: &mut Vec<SymId>, ext: &mut Vec<SymId>)` to `TransitionConstraintEvaluator` (`crypto/stark/src/constraints/transition.rs`). Default impl runs the verifier-shaped body symbolically; `TransitionConstraintAdapter` override runs `self.0.evaluate::<SymField,SymExt>`. Lets us capture by iterating the AIR's existing `Vec<Box<dyn …>>`.

1.2 **LogUp capture (the crux).** Build a symbolic `TransitionEvaluationContext::Prover` whose `rap_challenges`/`logup_alpha_powers`/`logup_table_offset`/`packing_shifts`/`periodic_values` are `Leaf` nodes, and a 2-step symbolic `Frame` (for the accumulated constraint's next-row `aux(1,·)` reads). Override `capture` on `LookupBatchedTermConstraint`/`LookupAccumulatedConstraint` (`crypto/stark/src/lookup.rs`) to run their already-generic inner fns under `SymField`/`SymExt`. Extend the IR `Op::Var` to carry `offset∈{0,1}` and `aux` flag; add `Leaf` variants for the uniforms.

1.3 **Per-table program.** `AIR::constraint_program(&self) -> ConstraintProgram` default method that captures all constraints, with `roots[constraint_idx]` and `num_base` aligned to `num_base_transition_constraints()`.

1.4 **Full CPU interpreter.** Extend `interp.rs` to: resolve all leaf kinds; handle Dim1/Dim3 with auto-embed; output the per-constraint `Cᵢ` (base buffer + ext buffer), matching the `compute_transition_prover` contract. Add a verifier entry (all-D3 frame at the OOD point).

**Milestone / test:** capture the full program for the CPU table (and 1–2 others incl. a LogUp-heavy one), interpret per-row over a real LDE, and `assert_eq!` against `air.compute_transition_prover(...)` bit-for-bit; same for the verifier at the OOD point.
**Risk:** LogUp context construction. *Mitigation:* the inner fns are already field-generic (verified); the spike proves the recording machinery. If it fights, fall back to hand-emitting the 2 LogUp programs (they're fixed, table-independent).

---

## Phase 2 — Wire interpreter into prover/verifier (CPU), behind a toggle 🔜

2.1 In `ConstraintEvaluator::evaluate_transitions` (`crypto/stark/src/constraints/evaluator.rs`), replace the `air.compute_transition_prover(...)` call with the IR interpreter, behind a feature/runtime toggle; keep the boxed path as default + oracle. Build/cache the `ConstraintProgram` once (in `ConstraintEvaluator::new`).

2.2 Verifier: same swap at `crypto/stark/src/verifier.rs` (`compute_transition`).

**Milestone / test:** full prove→verify suite passes with the interpreter as the live path, across **all** tables (CPU, MEMW, LOAD, DECODE, MUL, BRANCH, REGISTER, PAGE, BITWISE, LT, HALT, EC*, keccak, …). This is the **CPU end-to-end** checkpoint — the IR is now proven complete and correct, independent of any GPU work.

---

## Phase 3 — Device field primitives 🟢 ALREADY EXIST

Verified: `crypto/math-cuda/kernels/ext3.cuh` provides `ext3::Fe3` + `ext3::{add, sub, mul, mul_base}` (`mul_base` = base×ext, the subfield mul), and `kernels/goldilocks.cuh` the base ops — already used by the GPU FRI / inverse / barycentric / deep kernels. The constraint interpreter **reuses these directly**; there is no new field math to build. (The `gpu_lde.rs` "CPU fallback for extension columns" is about LDE column *dispatch*, not the arithmetic — ext3 LDE itself runs on GPU.)

Remaining: trivial — confirm a `neg` exists (else `ext3::sub(zero, x)`) and include the header. Treat as part of Phase 4.

---

## Phase 4 — GPU interpreter kernel 🔜

Start **stripped** (OpenVM's `GLOBAL=true` shape, no register allocation, no bit-packing).

4.1 **Device IR layout.** Serialize `ConstraintProgram` to a `#[repr(C)]` flat node array (`{op_tag, a, b, dim}`) + constants table + roots + `num_base`. Plus per-proof uniform buffers (rap challenges, alpha powers, table_offset, periodic columns, shift consts).

4.2 **Kernel** (`crypto/math-cuda/.../constraint_interp.cu` + Rust wrapper): one thread per LDE row (tiled), forward pass over nodes into a **per-thread value array in global memory** (one slot/node — works for all program sizes), resolving `Main/Aux{offset,col}` from the device-resident LDE columns (`GpuLdeBase`/`GpuLdeExt3` keep-handles), Dim1/Dim3 ops via Phase-3 primitives. **Fused accumulation:** Horner `Σ αⁱ·Cᵢ` + `× inv_zeroifier` → write the composition-poly evaluation per row. Output stays on device.

4.3 **Host dispatch** (`crypto/stark/src/symbolic/gpu_interp.rs`): TypeId-gate on `GoldilocksField`/`Degree3…` + size threshold (mirror `gpu_lde.rs`), upload program + uniforms once, launch, leave output device-resident; fall back to the CPU interpreter otherwise.

4.4 **Pipeline integration.** Feed the on-device composition-poly evals straight into the existing GPU Merkle commit (no D2H). Likely needs a whole-table GPU entry (`air.compute_transitions_batched(lde) -> Option<…>`) that `evaluate_transitions` tries before the per-row CPU loop. Keep `Σ βᵢ·Cᵢ·Zᵢ⁻¹`/boundary accounting consistent with the CPU path.

**Milestone:** kernel compiles under the `cuda` feature; host dispatch wired.

---

## Phase 5 — "Working on GPU" ✅ (the target) 🔜

Runs on **GPU hardware** (not in this dev sandbox — I'll hand you the commands; per project convention these run on the GPU/bench box).

5.1 **GPU↔CPU parity test** (extend `prover/tests/cuda_path_integration.rs` / `cuda_fallback_tests.rs`): composition-poly evals on GPU == CPU interpreter == boxed path, per table, on real traces.

5.2 **End-to-end GPU prove→verify** on a real ELF with the GPU constraint path enabled (`--features cuda`). A passing verify = **the deliverable** ("test on GPU and it works").

5.3 **Benchmark** (bench server): prove time with GPU constraints vs CPU constraints — confirm the data-residency win (no LDE D2H for constraint eval).

---

## Phase 6 — Optimizations ⏭ (only if a profile demands)

6.1 **Register allocation** — port OpenVM's transpiler liveness/linear-scan (`crates/cuda-backend/src/transpiler/mod.rs`) to shrink the per-thread value array → small programs in registers/local, big ones (Dvrm/Shift/ecsm) in a smaller global buffer. First optimization you'll likely need.
6.2 **DCE / const-fold peephole** — drop unused column leaves; fold `×Const(0)` left by the `eq=false` zero-skip.
6.3 **Bit-packed codec** — only if H2D bandwidth shows up (it won't early).
6.4 **Selective codegen** — given few-but-large tables, codegen the 1–3 hottest tables (nvcc does register allocation, no dispatch) if interpreter overhead is material. Hybrid: interpreter baseline + codegen the hot ones.

---

## Critical path & rough effort

```
Phase 1 (full capture + LogUp)      ~4–6 d   ┐
Phase 2 (wire CPU + validate)       ~3–4 d   ├─ CPU end-to-end correct
Phase 3 (device field ops)          ~0  (already exist — reuse ext3.cuh)
Phase 4 (GPU kernel + dispatch)     ~5–8 d   ┐  reuses existing primitives
Phase 5 (parity + e2e on GPU)       ~2–4 d   ┘  ← "working on GPU"
Phase 6 (optimize)                  as needed
```

~2.5–4 weeks to a working, validated GPU constraint evaluator (Phase 6 optional, perf-driven). Phase 3 dropping out is the main saving — the device field arithmetic is already there.

## What I can do here vs on your hardware

- **Phases 1–2** are CPU — I can implement and validate them in this repo directly.
- **Phase 4** (CUDA kernel) I can write, and compile-check under the `cuda` feature, but I **cannot run** GPU kernels in this sandbox. (Phase 3 needs nothing — the primitives exist.)
- **Phase 5** (parity + e2e + bench) runs on your **GPU box / bench server** — I'll provide exact commands; you run and report, I iterate.

## Decision points to confirm before/while building

- Keep **Plan A** (symbolic field) for full capture (spike validated it), or switch to Plan B's explicit `capture()` for the cleaner end-state? (Recommendation: stay Plan A through Phase 2; revisit only if LogUp capture is unexpectedly messy.)
- GPU value array: start **global-memory** (simplest, all tables) — defer register allocation to Phase 6 unless Phase 4 node-count data says otherwise.
- CPU path after Phase 2: keep the boxed path as the default oracle (toggle), or commit the interpreter as the CPU default too?
