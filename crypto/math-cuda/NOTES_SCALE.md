# Scale + profile snapshot — exp-8

Profiling + scale benchmark run on top of `cuda/exp-7-logup-gpu`
(LogUp GPU path opt-in, threshold set to 1M rows). All numbers below
are **mean over 5 trials** (fib_4M is 3 trials) with
`LAMBDA_VM_GPU_LOGUP_THRESHOLD=1048576` exported. Bench binary built
via `cargo test -p lambda-vm-prover --release --features cuda,instruments`.

## Scale

| trace size | fib_iterative | GPU mean | Wall ratio | Per-row cost vs 1M |
|---|---|---|---|---|
| 1M rows | fib_iterative_1M | 12.52 s | 1.00× | 1.00 |
| 2M rows | fib_iterative_2M | 20.33 s | 1.62× | 0.81 |
| 4M rows | fib_iterative_4M | 32.30 s | 2.58× | 0.65 |

Doubling the trace size does **not** double the wall time — fixed
costs (GPU warm-up, kernel-launch overhead, transcript work) amortize.
Going from 1M to 4M is only 2.58× wall for 4× data, i.e. 35% cheaper
per row at the larger size. This is the **GPU-favored regime**: every
optimization that pays for per-table overhead compounds as tables get
bigger, and future work (exp-9 through exp-11) should be benched at
fib_4M in addition to fib_1M.

## Wall-time breakdown (fib_1M, representative trial @ 12.11 s)

```
Trace build                 2.39 s   19.7%    CPU (user-supervised area)
Round 1                     4.79 s   39.6%
  Main trace commits        1.50 s           GPU LDE + Merkle
    expand_columns_to_lde   1.45 s (agg)
    commit (Merkle)         0.54 s (agg)
  Aux trace build           2.15 s           LogUp, GPU when >1M rows
  Aux trace commit          1.15 s           GPU LDE + Merkle
Rounds 2–4                  4.52 s   37.4%    mixed GPU/CPU
Other                       0.19 s
```

## Where GPU vs CPU is at (fib_1M Rounds 2–4 aggregates)

Aggregates are summed across rayon threads; wall is a fraction of each.

```
R2  evaluate                     5.24 s agg   quotient eval, GPU (partial)
R4  queries & openings           3.88 s agg   CPU — ← remaining bottleneck
R2  decompose_and_extend_d2      2.88 s agg   GPU LDE on device handles
R3  OOD evaluation               1.77 s agg   GPU barycentric
R2  commit_composition_poly      1.74 s agg   GPU (R2 commit fuse)
R4  deep_composition_poly_evals  1.39 s agg   GPU R4 deep
R4  fri::commit_phase            1.11 s agg   GPU (device-resident)
R4  interpolate+evaluate_fft     0.51 s agg   small
```

## What's actionable

Ranked by expected wall-time yield on fib_1M:

1. **Aux trace build (2.15 s wall).** Today's LogUp path is neutral
   — 12 tables each run build_auxiliary_trace in rayon, each firing
   its own GPU stream. Fix: serialize GPU dispatch so streams don't
   contend on H2D / compute. Expected: 500–1000 ms.
   Checkpointed as `cuda/exp-9-logup-cross-table`.

2. **R4 queries & openings (~300 ms wall).** CPU today. pil2-proofman
   has this on GPU via `getTreeTracePols` + `genMerkleProof`; kernels
   are simple. Requires keeping the main-trace Merkle tree
   device-resident past R1. Expected: 200–300 ms.
   Checkpointed as `cuda/exp-10-fri-queries-on-gpu`.

3. **Device-resident main trace (1–2 s wall, architectural).**
   Eliminate the per-phase H2D of the main trace by building it
   straight into GPU memory (or uploading once post-build). Touches
   trace build (previously off-limits; now green-lit). Biggest single
   move. Checkpointed as `cuda/exp-11-device-trace`.

Profiling note: `nsys profile -t cuda,nvtx` on this box adds ≥10×
overhead on this workload (12-trial bench ran >12 min before we
killed it). Stick to `--features instruments` for wall-time
measurements; use `nsys` only on a single-trial run with `--sample=none
--cpuctxsw=none` and accept the slowdown.
