# LogUp aux-trace build on GPU — exp-7 checkpoint status

## What landed

End-to-end GPU pipeline for `compute_logup_batched_term_column`:

- `crypto/math-cuda/kernels/logup.cu`
  - `logup_pair_fingerprint` / `logup_single_fingerprint` — evaluates a
    `BusInteraction`'s fingerprint row-by-row from a bytecode descriptor
    supporting every `Packing` variant (Direct, Word2L, Word4L, DWordWL,
    DWordHHW, DWordWHH, DWordHL, DWordBL, QuadHL, QuadWL) plus `OP_LINEAR`
    for arbitrary linear combinations.
  - `logup_pair_term_assembly` / `logup_single_term_assembly` —
    evaluates `Multiplicity` (One/Column/Sum/Negated/Diff/Sum3/Linear)
    and combines with the inverted fingerprints into the term column.
- `crypto/math-cuda/src/logup.rs` — host-side wrappers + a
  `DeviceMainCols` handle so `build_auxiliary_trace` uploads the
  main-segment columns once per table instead of once per pair.
- `crypto/stark/src/logup_gpu.rs` — serializer from the native
  `BusValue` / `Multiplicity` / `LinearTerm` enums into the shared
  `FingerprintOp` / `LinearTerm` / `MultiplicityDesc` wire format, plus
  dispatch that turns an entire table's interaction list into committed
  + virtual term columns in one H2D of main_cols.

Coefficient handling: all `i64` / `u64` constants are canonicalized into
`[0, p)` on the Rust side, so the kernel never branches on sign.

## Parity

121 stark prove+verify tests pass with `LAMBDA_VM_GPU_LOGUP_THRESHOLD=0`
(forces the GPU path for every table). Verifier is untouched.

## Perf on fib_iterative_1M (46-core CPU + RTX 5090, 15-trial mean)

| Path                                    | avg    | aux-build wall |
|-----------------------------------------|--------|----------------|
| exp-7 CPU (threshold=MAX, default)      | 11.17s | —              |
| exp-7 GPU table-batched (threshold=0)   | 11.81s | 2.66s          |
| exp-7 GPU per-pair (earlier iteration)  | 16.06s | 5.09s          |

The per-pair version regressed badly because each pair re-uploaded the
~240 MB main trace. The table-batched version eliminates that redundant
H2D (upload once per table, dispatch all pairs against the shared
device buffer), which recovers 4s. It's still ~640 ms behind the
rayon-parallel CPU path — the 46-core CPU reads main_cols from RAM for
free, while the GPU must pay PCIe for it.

## Why it isn't a win yet

- **Nested parallelism → stream contention.** The prover already runs
  `build_auxiliary_trace` in parallel across ~12 tables. Each GPU-path
  table runs its pair kernels serially on one stream, so we have ~12
  concurrent streams competing for the device. That contention eats
  most of the per-table speedup.
- **H2D-dominated for large tables.** For the MEMW_R × 3.1M-row tables
  each H2D is ~750 MB — a sizeable fraction of the 70-100 ms budget
  per table, before any kernel fires.
- **CPU baseline is genuinely fast.** 46 rayon threads chewing through
  fingerprints + batch inverse + term assembly is hard to beat when
  the data is already in RAM.

## Default posture

Gated off by default via `LAMBDA_VM_GPU_LOGUP_THRESHOLD` (default
`usize::MAX`). Set the env var to `0` (or a `trace_len` threshold) to
force-enable for experiments. CPU-only build and `--features cuda`
without the env var both keep the old rayon path — zero regression.

## Where to go next

Plausible paths to turn this into a win:

1. **Cross-table batching.** Upload main_cols for all tables at once
   (or in a few fat batches) and let one stream chew through pairs
   without concurrent-stream contention. Requires restructuring the
   prover's table-parallel loop.
2. **Fused multi-pair kernel.** One kernel launch per table that walks
   all pairs using a batched bytecode layout, so per-pair CPU
   orchestration disappears.
3. **Keep the trace resident on device.** If the main LDE already
   lives on the GPU (as in the experimental-lde-resident checkpoint),
   the H2D vanishes and this path starts winning. That's a bigger
   architectural move, not a logup-local tweak.
