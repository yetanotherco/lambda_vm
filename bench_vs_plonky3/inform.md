# Lambda STARK vs Plonky3 — parallelism scaling report

> **Nota (2026-05-13):** los números de este reporte fueron tomados con el
> fork `yetanotherco/Plonky3#feat/goldilocks_deg3` (extensión binomial
> `x³-2`). Desde 2026-05-13 `bench_vs_plonky3` apunta a Plonky3 upstream
> con `CubicTrinomialExtensionField` (`x³-x-1`). Misma soundness y grado;
> diferencia esperada de un par de adds por mul Fp3.

## TL;DR

- Plonky3 parallelism scales up to ~16× on a 48-core EPYC; **Lambda tops out at ~8×**.
- **Single-thread gap is a flat ~2× (P3 faster).** Under parallelism it widens to **~4× in the worst case** (big proof, many threads).
- Lambda's bottleneck is an **O(N) serial section** (~10% of single-thread time) that grows linearly with problem size. Attacking it is the highest-ROI next step.
- SMT (past 48 threads) does not help at small/medium sizes and only marginally (~1-2%) at the largest size.

## Setup

- **Server**: `vm-benchmarks-1`, AMD EPYC 9454P — 48 physical cores, 96 logical (SMT=2), Zen 4, 12-channel DDR5.
- **Arch**: `x86_64`.
- **SIMD**: OFF on both provers — `RUSTFLAGS="-C target-feature=-avx2,-avx512f"`. Scalar Goldilocks on both sides. Residual SSE2 in p3-keccak (~7% of prove).
- **AIR**: shared Fibonacci implementation (cell-by-cell equivalent between Lambda and P3). 32 columns (`num-sequences=16`), blowup=2, 219 FRI queries, grinding=0, degree-3 extension field on both sides (p3-goldilocks-patched fork).
- **Lambda entry point**: bench calls `Prover::prove(...)`, which is a thin wrapper over `multi_prove` with a single-element AIR slice (`crypto/stark/src/prover.rs:1878-1894`). Same code path as multi-AIR — no legacy flow.
- **Measurement**: single-shot end-to-end prove (no verify). 3 runs per `(size, threads, prover)`, median reported.
- **Branch**: `bench_vs_p3`.

---

## Sweep 1 — size × threads (log-rows 17 / 19 / 21, T = 1..48 physical)

**Goal:** how does the Lambda vs P3 gap behave when both problem size and thread count change.

| Threads | L@17 | P3@17 | L/P3@17 | L@19 | P3@19 | L/P3@19 | L@21   | P3@21  | L/P3@21 |
|:------:|----:|-----:|:------:|----:|-----:|:------:|------:|------:|:------:|
| 1      | 1.878 | 0.918 | 2.05× | 7.189 | 3.687 | 1.95× | 29.591 | 14.876 | 1.99× |
| 2      | 1.111 | 0.545 | 2.04× | 4.040 | 2.017 | 2.00× | 16.591 |  7.842 | 2.12× |
| 4      | 0.750 | 0.330 | 2.27× | 2.756 | 1.212 | 2.27× | 10.995 |  4.355 | 2.53× |
| 8      | 0.420 | 0.191 | 2.20× | 1.545 | 0.721 | 2.14× |  6.329 |  2.475 | 2.56× |
| 16     | 0.337 | 0.127 | 2.65× | 1.140 | 0.457 | 2.50× |  4.561 |  1.490 | 3.06× |
| 32     | 0.316 | 0.109 | 2.90× | 0.967 | 0.376 | 2.57× |  3.793 |  0.943 | **4.02×** |
| 48     | 0.314 | 0.113 | 2.78× | 0.929 | 0.328 | 2.83× |  3.640 |  0.924 | 3.94× |

### Max speedup vs T=1 (how much parallelism each prover extracts)

| Size         | Lambda | P3     |
|:------------:|:------:|:------:|
| log=17       | 5.98×  | 8.12×  |
| log=19       | 7.74×  | 11.24× |
| log=21       | **8.13×** | **16.10×** |

### Key findings

1. **The gap scales poorly with both size AND threads.** At T=1 the ratio is ~2× regardless of size. At T=32-48 with log-rows=21 it grows to **~4×**.
2. **P3's max speedup grows with size** (8× → 11× → 16×). **Lambda's stays flat near 8×.** Bigger problem does not give Lambda more parallelism to exploit.
3. **Lambda's serial fraction grows O(N)** (Amdahl fit): ~0.28s @ log=17, ~0.78s @ log=19, ~3.0s @ log=21. This is not a fixed overhead — it is work running serial that scales with N.

---

## Sweep 2 — dense threads at max size (log-rows=21, 14 points from 1 to 96)

**Goal:** at the largest problem size (2.1M rows), sweep thread count in fine granularity across all 48 physical cores and into SMT territory. Resolves the curve's knee and shows exactly where each prover peaks.

| Threads | Lambda (s) | P3 (s) | L/P3  | Speedup Lambda | Speedup P3 | Eff. L | Eff. P3 |
|:------:|----------:|-------:|------:|---------------:|-----------:|-------:|--------:|
| 1      | 29.688    | 14.881 | 2.00× | 1.00×          | 1.00×      | 100%   | 100%    |
| 2      | 16.596    |  7.842 | 2.12× | 1.79×          | 1.90×      | 89%    | 95%     |
| 4      | 11.025    |  4.358 | 2.53× | 2.69×          | 3.42×      | 67%    | 85%     |
| 6      |  7.910    |  3.142 | 2.52× | 3.75×          | 4.74×      | 63%    | 79%     |
| 8      |  6.213    |  2.485 | 2.50× | 4.78×          | 5.99×      | 60%    | 75%     |
| 12     |  5.300    |  1.811 | 2.93× | 5.60×          | 8.22×      | 47%    | 69%     |
| 16     |  4.539    |  1.490 | 3.05× | 6.54×          | 9.99×      | 41%    | 62%     |
| 20     |  4.347    |  1.299 | 3.35× | 6.83×          | 11.46×     | 34%    | 57%     |
| 24     |  4.175    |  1.187 | 3.52× | 7.11×          | 12.54×     | 30%    | 52%     |
| 32     |  3.832    |  1.053 | 3.64× | 7.75×          | 14.13×     | 24%    | 44%     |
| **40** | **3.737** | **0.909** | **4.11×** | 7.94× | **16.37×** | 20% | 41% |
| 48     |  3.686    |  0.983 | 3.75× | 8.05×          | 15.14×     | 17%    | 32%     |
| 64 SMT |  3.656    |  1.028 | 3.56× | 8.12×          | 14.48×     | 13%    | 23%     |
| 96 SMT |  3.600    |  0.977 | 3.69× | **8.25×**      | 15.23×     | 9%     | 16%     |

### Four distinct scaling regimes

The curve is not a single smooth shape. It breaks into four zones, each revealing what bottlenecks at that operating point.

- **Zone A — T=1→8: near-linear on both.** P3 efficiency ≥75%, Lambda ≥60%. Dominant parallel work (constraint eval, core trace commit) uses the first cores well. The gap appears early because P3 uses batched row-wise FFT; Lambda is per-column.
- **Zone B — T=8→16: Lambda loses momentum.** Its per-column parallelism (32-column `into_par_iter` in the commit) is running out of granularity. P3 keeps decaying smoothly — it parallelizes finer (row/tile-based).
- **Zone C — T=16→40: Lambda flattens; P3 keeps gaining, peaks at T=40.** Lambda: 4.54s → 3.74s (only −18% with 2.5× more threads). P3: 1.49s → 0.91s (−39%). **The L/P3 gap peaks here at 4.11×** — P3 at its best, Lambda already capped.
- **Zone D — T=48→96 (SMT): diverges.** Lambda gains a bit more (3.69 → 3.60, +1.5%) because its serial-ish section has memory stalls hyperthreads can hide. P3 stalls/regresses — contention wins.

### Peak points

| Prover | Peak T   | Peak time | Peak speedup | Comment |
|:------:|:--------:|:---------:|:------------:|---------|
| Lambda | T=96 SMT | 3.600s    | 8.25×        | But 95% of that is already reached at T=16. |
| P3     | T=40     | 0.909s    | **16.37×**   | Clear peak, slight regression past T=40. |

### Amdahl fit — where Lambda is stuck

Fitting `S(T) = T / (1 + s·(T-1))`:

| Prover | s @ T=32 | s @ T=48 | s @ T=96 | Theoretical max (1/s) |
|:------:|:--------:|:--------:|:--------:|:---------------------:|
| Lambda | 10.1%    | 10.6%    | 11.2%    | **~9-10×**            |
| P3     | 4.1%     | —        | —        | ~27×                  |

- **Lambda is at ~90% of its Amdahl ceiling already.** Observed 8.25× vs theoretical 9-10×. Adding cores will not help — the fix has to be in the serial section.
- **P3 is bandwidth/contention-limited, not Amdahl-limited** (16.4× observed vs 27× theoretical).

---

## Bottom line

- For a single proof on a big server, **Lambda hits a ~8× parallelism ceiling**; P3 reaches 16×. That is the main gap.
- **Sweet spot for Lambda is T=16-24.** Giving it more threads barely helps.
- **Throughput mode — run two Lambda proofs in parallel at 24 threads each**: each proof ~4.2s, two proofs in parallel → effective 2.1s/proof, beating the 3.6s of a single 48-thread proof. Worth validating for batch workloads (memory bandwidth could bite, but 12-channel DDR5 should handle it).
- **For P3, always run at T=40**, not T=48. Small but consistent ~7% win.
- **Top next step to close the gap**: profile Lambda at T=48 / log-rows=21 with `perf record`, identify the ~10% O(N) serial section. Main suspects: trace-commit Merkle build, FRI fold serial chain, deep composition. That is where the 4× gap is hiding.

---

## Final report

**Date:** 2026-04-22

### TL;DR

- Plonky3 parallelism scales up to ~16× on a 48-core EPYC; **Lambda tops out at ~8×**.
- **Single-thread gap is a flat ~2× (P3 faster).** Under parallelism it widens to **~4× in the worst case** (big proof, many threads).
- Lambda's bottleneck is a **serial-like section whose absolute cost grows near-linearly with N** (~10% of single-thread time per Amdahl fit). Closing it is likely the highest-ROI direction — to be confirmed by profiling.
- SMT (past 48 threads) does not help at small/medium sizes and only marginally (~1-2%) at the largest size.

### Setup

- **Server**: `vm-benchmarks-1`, AMD EPYC 9454P — 48 physical cores, 96 logical (SMT=2), Zen 4, 12-channel DDR5.
- **Arch**: `x86_64`.
- **SIMD**: OFF on both provers — `RUSTFLAGS="-C target-feature=-avx2,-avx512f"`. Scalar Goldilocks on both sides. A residual SSE2 path remains in `p3-keccak` (estimated ~7% of prove time based on a prior M1 measurement; not re-measured on this server).
- **AIR**: shared Fibonacci implementation (cell-by-cell equivalent between Lambda and P3). 32 columns (`num-sequences=16`), blowup=2, 219 FRI queries, grinding=0, degree-3 extension field on both sides (p3-goldilocks-patched fork).
- **Lambda entry point**: bench calls `Prover::prove(...)`, which is a thin wrapper over `multi_prove` with a single-element AIR slice (`crypto/stark/src/prover.rs:1878-1894`). Same code path as multi-AIR — no legacy flow.
- **Measurement**: single-shot end-to-end prove (no verify). 3 runs per `(size, threads, prover)`, median reported.
- **Branch**: `bench_vs_p3`.

---

### Sweep 1 — size × threads (log-rows 17 / 19 / 21, T = 1..48 physical + SMT)

**Goal:** how does the Lambda vs P3 gap behave when both problem size and thread count change.

| Threads | L@17 | P3@17 | L/P3@17 | L@19 | P3@19 | L/P3@19 | L@21   | P3@21  | L/P3@21 |
|:------:|----:|-----:|:------:|----:|-----:|:------:|------:|------:|:------:|
| 1      | 1.878 | 0.918 | 2.05× | 7.189 | 3.687 | 1.95× | 29.591 | 14.876 | 1.99× |
| 2      | 1.111 | 0.545 | 2.04× | 4.040 | 2.017 | 2.00× | 16.591 |  7.842 | 2.12× |
| 4      | 0.750 | 0.330 | 2.27× | 2.756 | 1.212 | 2.27× | 10.995 |  4.355 | 2.53× |
| 8      | 0.420 | 0.191 | 2.20× | 1.545 | 0.721 | 2.14× |  6.329 |  2.475 | 2.56× |
| 16     | 0.337 | 0.127 | 2.65× | 1.140 | 0.457 | 2.50× |  4.561 |  1.490 | 3.06× |
| 32     | 0.316 | 0.109 | 2.90× | 0.967 | 0.376 | 2.57× |  3.793 |  0.943 | **4.02×** |
| 48     | 0.314 | 0.113 | 2.78× | 0.929 | 0.328 | 2.83× |  3.640 |  0.924 | 3.94× |
| 64 SMT | —     | —     | —     | 0.949 | 0.332 | 2.86× |  3.656 |  1.028 | 3.56× |
| 96 SMT | —     | —     | —     | 0.967 | 0.346 | 2.80× |  3.600 |  0.977 | 3.69× |

*log=17 was not measured at T=64/96 — the proof is short enough (≤0.32s from T=32 onward) that SMT noise would dominate the signal.*

### Max speedup vs T=1 (how much parallelism each prover extracts)

| Size         | Lambda | P3     |
|:------------:|:------:|:------:|
| log=17       | 5.98×  | 8.12×  |
| log=19       | 7.74×  | 11.24× |
| log=21       | **8.25×** | **16.37×** |

### Key findings

1. **The gap scales poorly with both size AND threads.** At T=1 the ratio is ~2× regardless of size. At T=32-48 with log-rows=21 it grows to **~4×**.
2. **P3's max speedup grows with size** (8× → 11× → 16×). **Lambda's stays near 8×.** Bigger problem does not give Lambda more parallelism to exploit.
3. **Lambda's serial-like section grows near-linearly with N** (Amdahl fit): ~0.28s @ log=17, ~0.78s @ log=19, ~3.0s @ log=21. Not a fixed overhead — it scales with problem size.
4. **SMT at large size (log=21) adds ~1-2%** for Lambda, nothing for P3. Not useful for single-proof latency, possibly useful for throughput (unmeasured).

---

### Sweep 2 — dense threads at max size (log-rows=21, 14 points from 1 to 96)

**Goal:** at the largest problem size (2.1M rows), sweep thread count in fine granularity across all 48 physical cores and into SMT territory. Resolves the curve's knee and shows exactly where each prover peaks.

| Threads | Lambda (s) | P3 (s) | L/P3  | Speedup Lambda | Speedup P3 | Eff. L | Eff. P3 |
|:------:|----------:|-------:|------:|---------------:|-----------:|-------:|--------:|
| 1      | 29.688    | 14.881 | 2.00× | 1.00×          | 1.00×      | 100%   | 100%    |
| 2      | 16.596    |  7.842 | 2.12× | 1.79×          | 1.90×      | 89%    | 95%     |
| 4      | 11.025    |  4.358 | 2.53× | 2.69×          | 3.42×      | 67%    | 85%     |
| 6      |  7.910    |  3.142 | 2.52× | 3.75×          | 4.74×      | 63%    | 79%     |
| 8      |  6.213    |  2.485 | 2.50× | 4.78×          | 5.99×      | 60%    | 75%     |
| 12     |  5.300    |  1.811 | 2.93× | 5.60×          | 8.22×      | 47%    | 69%     |
| 16     |  4.539    |  1.490 | 3.05× | 6.54×          | 9.99×      | 41%    | 62%     |
| 20     |  4.347    |  1.299 | 3.35× | 6.83×          | 11.46×     | 34%    | 57%     |
| 24     |  4.175    |  1.187 | 3.52× | 7.11×          | 12.54×     | 30%    | 52%     |
| 32     |  3.832    |  1.053 | 3.64× | 7.75×          | 14.13×     | 24%    | 44%     |
| **40** | **3.737** | **0.909** | **4.11×** | 7.94× | **16.37×** | 20% | 41% |
| 48     |  3.686    |  0.983 | 3.75× | 8.05×          | 15.14×     | 17%    | 32%     |
| 64 SMT |  3.656    |  1.028 | 3.56× | 8.12×          | 14.48×     | 13%    | 23%     |
| 96 SMT |  3.600    |  0.977 | 3.69× | **8.25×**      | 15.23×     | 9%     | 16%     |

### Four distinct scaling regimes

The curve is not a single smooth shape. It breaks into four zones, each revealing what bottlenecks at that operating point.

- **Zone A — T=1→8: near-linear on both.** P3 efficiency ≥75%, Lambda ≥60%. Dominant parallel work (constraint eval, core trace commit) uses the first cores well. The gap appears early because P3 uses batched row-wise FFT; Lambda is per-column.
- **Zone B — T=8→16: Lambda loses momentum.** Its per-column parallelism (32-column `into_par_iter` in the commit) is running out of granularity. P3 keeps decaying smoothly — it parallelizes finer (row/tile-based).
- **Zone C — T=16→40: Lambda flattens; P3 keeps gaining, peaks at T=40.** Lambda: 4.54s → 3.74s (only −18% with 2.5× more threads). P3: 1.49s → 0.91s (−39%). **The L/P3 gap peaks here at 4.11×** — P3 at its best, Lambda already capped.
- **Zone D — T=48→96 (SMT): diverges.** Lambda gains a bit more (3.69 → 3.60, +1.5%) because its serial-ish section has memory stalls hyperthreads can hide. P3 stalls/regresses — contention wins.

### Peak points

| Prover | Peak T   | Peak time | Peak speedup | Comment |
|:------:|:--------:|:---------:|:------------:|---------|
| Lambda | T=96 SMT | 3.600s    | 8.25×        | Past T=16, going from 4.54s to 3.60s costs 6× more cores for only ~21% extra time saved. |
| P3     | T=40     | 0.909s    | **16.37×**   | Clear peak, slight regression past T=40. |

### Amdahl fit — where Lambda is stuck

Fitting `S(T) = T / (1 + s·(T-1))`:

| Prover | s @ T=32 | s @ T=48 | s @ T=96 | Theoretical max (1/s) |
|:------:|:--------:|:--------:|:--------:|:---------------------:|
| Lambda | 10.1%    | 10.6%    | 11.2%    | **~9-10×**            |
| P3     | 4.1%     | —        | —        | ~27×                  |

- **Lambda is at ~90% of its Amdahl ceiling already.** Observed 8.25× vs theoretical 9-10×. Adding cores will not help — the fix has to be in the serial section.
- **P3 is bandwidth/contention-limited, not Amdahl-limited** (16.4× observed vs 27× theoretical).

---

### Bottom line

- For a single proof on a big server, **Lambda hits a ~8× parallelism ceiling**; P3 reaches 16×. That is the main gap.
- **Sweet spot for Lambda is T=16-24.** Giving it more threads barely helps.
- **Throughput mode — run two Lambda proofs in parallel at 24 threads each**: each proof ~4.2s; two proofs in parallel → effective 2.1s/proof, beating the 3.6s of a single 48-thread proof. Worth validating for batch workloads (memory bandwidth could bite, but 12-channel DDR5 should handle it).
- **For P3, always run at T=40**, not T=48. Small but consistent ~7% win.
- **Top next step to close the gap**: profile Lambda at T=48 / log-rows=21 with `perf record` and identify the ~10% serial-like section. Main suspects (to be verified): trace-commit Merkle build, FRI fold serial chain, deep composition polynomial.

---

**Raw data**: `bench_vs_plonky3/reports/scaling_scalar_multisize/` (Sweep 1), `bench_vs_plonky3/reports/scaling_scalar/` (Sweep 1 SMT rows at log=19), `bench_vs_plonky3/reports/scaling_log21_dense/` (Sweep 2). Each `results.tsv` + `metrics.txt` + `raw/*.stdout` per (threads, size) combination. Branch `bench_vs_p3`, commit recorded in `metrics.txt`.
