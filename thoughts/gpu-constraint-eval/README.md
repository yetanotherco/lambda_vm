# GPU-ready constraint evaluation — design

Design docs for moving STARK transition-constraint evaluation onto the GPU.

## Why (the real motivation)

Not to make constraint evaluation faster — it isn't the CPU bottleneck. The goal is
**data residency**: keep the whole prove pipeline on-device so we never round-trip the
LDE trace across the PCIe bus.

```
LDE (GPU)  →  constraint eval / composition poly  →  Merkle commit (GPU)  →  FRI (GPU)
```

The LDE trace (main + aux columns × blowup factor) is the largest array in the pipeline.
If constraint eval stays on the CPU, every proof must D2H-copy the entire LDE and push
results back — that transfer dominates. `gpu_lde.rs` already keeps columns resident
(`GpuLdeBase`/`GpuLdeExt3` keep-handles); on-GPU constraint eval consumes them in place.

Consequence: the GPU kernel must also do the accumulation (`Σ αⁱ·Cᵢ·Zᵢ⁻¹`, Horner form
+ ÷Z) so it emits **composition-polynomial evaluations on-device**, not a raw `Cᵢ`
matrix (which would itself be a large D2H copy).

## The blocker

The prover is Rust; you can't run arbitrary Rust on a GPU. Constraints today are
`Vec<Box<dyn TransitionConstraintEvaluator>>` evaluated via a generic
`evaluate<FF,EE>` — two layers of dynamic dispatch, scalar, CPU-only. The logic has to
be re-expressed in a GPU-executable form.

## The decided architecture

**Capture each table's constraints once into a flat, single-field Goldilocks IR (a
typed `Dim1`/`Dim3` op-DAG), then interpret that IR** — on CPU (verifier/optional
prover) and on GPU (one universal Goldilocks kernel). Single source of truth → CPU and
GPU can't diverge. Not codegen, not a DSL, not hand-written per-table kernels.

- Field: Goldilocks base (`Dim1`, `u64`) + degree-3 extension (`Dim3`, `[u64;3]`).
- IR ops: `Add/Sub/Mul/Neg` + leaves (`Main/Aux{offset,col}`, `Const`, `Periodic`,
  `RapChallenge`, `AlphaPow`, `TableOffset`, `Shift`).
- Boundary: the zerofier/coefficient machinery stays in
  `ConstraintEvaluator::evaluate_transitions`; the IR replaces only the per-row,
  per-constraint step that produces each `Cᵢ` (on GPU, fused with the accumulation).

This is the same family SP1, OpenVM, and zisk converged on. zisk is the closest match
(Goldilocks, FRI-STARK LDE quotient).

## The two plans (the only open decision)

Both produce the **same IR** and feed the **same interpreter + GPU kernel + validation**.
They differ *only* in how the IR is captured.

| | [Plan A — symbolic field](./plan-symbolic-field.md) | [Plan B — builder rewrite](./plan-builder-rewrite.md) |
|---|---|---|
| Constraint edits | ~0 (record existing `evaluate` via a `SymField`) | ~600–800 LOC across 33 structs rewritten to `capture()` |
| Feasibility | HIGH — `SymField` needs only `IsField`+`IsSubFieldOf` (capture never builds an `AIR<Field=SymField>`); unreachable methods stubbed | No doubt; just labor |
| Risk shape | Concentrated in `SymField` — spike-able in 1–2 days | Spread across 33 transcriptions (Dvrm 11 / Cpu32 8 / Shift 7 kinds) |
| CPU path | can stay unchanged (IR GPU-only) | forced onto the interpreter (old `evaluate` deleted) |
| End state | recording field + arena + stubs retained | cleanest; generic `evaluate`+adapter deleted; ecosystem-idiomatic |
| Effort (CPU validated) | ~10–14 d | ~12–18 d |
| Effort (GPU) | ~6–10 d | ~5–7 d (identical, shared) |

Both lose the per-row LogUp zero-skip (value-identical; recover via a static
const-fold peephole). Neither AVX nor monomorphization differentiates them (AVX lives
in the shared interpreter; monomorphization is a third thing neither plan does).

## Reference implementation

`others/openvm-stark-backend` (cloned `openvm-org/stark-backend@v1.4.0`) is a working
implementation of this exact approach for a FRI-STARK LDE quotient. Key files:

- `crates/cuda-backend/src/transpiler/mod.rs` — lowers the symbolic DAG to three-address
  code + liveness/linear-scan register allocation (the **IR processing** — the most
  portable, field-agnostic piece; ~200 lines; the reg-alloc is optional for v1).
- `crates/cuda-backend/src/transpiler/codec.rs` + `cuda/include/codec.cuh` — encode rules
  to a 128-bit packed word.
- `crates/cuda-backend/cuda/src/quotient.cu` — the interpreter kernel: per-row loop over
  rules, fused Horner accumulation + ÷Z, per-thread intermediate buffer (local for small
  programs, global spill for large — solves GPU scratch pressure).

BabyBear→Goldilocks deltas to be aware of: the codec packs constants in 32 bits (fits
BabyBear's 31-bit modulus, **not** Goldilocks' 64 bits) → needs a side constant table;
extension is BabyBear's vs our degree-3; OpenVM evaluates everything in `FpExt` (no
base/ext split). It's a blueprint to port, not a crate to depend on (it's tied to
OpenVM's symbolic-DAG type, `PrimeField32` bound, and trace/bus conventions).

SP1's `sp1-gpu` is the same pattern via an SSA register-machine bytecode (~60+ opcodes,
operand types in the opcode); OpenVM puts operand types in the source tag (~6 ops) —
the latter is the better template for single-field Goldilocks (even fewer source types).

## Recommendation / next step

Spike **Plan A** first (1–2 days): implement `SymField`, capture the CPU table, diff the
interpreted IR bit-for-bit against the current evaluator, and dump per-table node counts.
The IR/interpreter/GPU kernel are shared, so switching to Plan B later costs almost
nothing. If `SymField` fights the trait tower, fall back to the Plan B rewrite.
