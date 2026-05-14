# bench_vs_plonky3 — Analysis Log

> **Nota (2026-05-13):** los números de este log fueron tomados con el fork
> `yetanotherco/Plonky3#feat/goldilocks_deg3` (extensión binomial `x³-2`,
> matching Lambda). Desde 2026-05-13 `bench_vs_plonky3` apunta a Plonky3
> upstream con `CubicTrinomialExtensionField` (`x³-x-1`, trinomial). Misma
> soundness y grado; mul Fp3 trinomial agrega un par de adds por
> multiplicación.

# Shareable report — Lambda STARK vs Plonky3 scaling study

## TL;DR

- On the shared Fibonacci AIR, **Plonky3 parallelism scales up to ~16× on a 48-core EPYC**, while Lambda tops out at **~8×**.
- **Single-thread gap is a flat ~2× (P3 faster).** Under parallelism the gap widens to **~4×** in the worst case (large proof, many threads).
- Lambda's bottleneck is an **O(N) serial section** (~10% of single-thread time, and it grows linearly with problem size). Attacking it is the highest-ROI next step.
- SMT (hyperthreading past 48) does not help at all at small/medium sizes, and only marginally (1-2%) at the largest size.

## Setup (both sweeps)

- Server `vm-benchmarks-1`, AMD EPYC 9454P (48 physical cores, Zen 4).
- SIMD disabled on both sides: `RUSTFLAGS="-C target-feature=-avx2,-avx512f"` (scalar Goldilocks; residual SSE2 in p3-keccak worth ~7%).
- Shared Fibonacci AIR, 32 columns, blowup=2, 219 FRI queries, degree-3 extension on both sides (p3-goldilocks-patched fork).
- Lambda calls `Prover::prove()` which is a thin wrapper over `multi_prove` — same code path as multi-AIR, no legacy flow.
- 3 runs per point, median reported.

## Sweep 1 — size × threads (log-rows 17 / 19 / 21, T = 1..48 physical)

Goal: see how the Lambda vs P3 gap behaves with both problem size and parallelism.

| Threads | L@17 / P3@17 / ratio | L@19 / P3@19 / ratio | L@21 / P3@21 / ratio |
|:------:|:--:|:--:|:--:|
| 1  | 1.88 / 0.92 / **2.05×** | 7.19 / 3.69 / **1.95×** | 29.59 / 14.88 / **1.99×** |
| 8  | 0.42 / 0.19 / 2.20× | 1.55 / 0.72 / 2.14× | 6.33 / 2.48 / 2.56× |
| 16 | 0.34 / 0.13 / 2.65× | 1.14 / 0.46 / 2.50× | 4.56 / 1.49 / 3.06× |
| 32 | 0.32 / 0.11 / 2.90× | 0.97 / 0.38 / 2.57× | 3.79 / 0.94 / 4.02× |
| 48 | 0.31 / 0.11 / 2.78× | 0.93 / 0.33 / **2.83×** | 3.64 / 0.92 / **3.94×** |

| Max speedup (T=48) | log=17 | log=19 | log=21 |
|:-:|:-:|:-:|:-:|
| Lambda | 5.98× | 7.74× | **8.13×** |
| P3     | 8.12× | 11.24× | **16.10×** |

**Key findings:**

1. **The gap scales poorly with parallelism AND size.** At T=1 the ratio is ~2× regardless of size. At T=32-48 with log-rows=21 it grows to ~4×.
2. **P3's max speedup grows with size** (8× → 11× → 16×). **Lambda's stays flat near 8×.** Bigger problem does not give Lambda more parallelism to exploit.
3. **Lambda's serial fraction grows O(N)** (Amdahl fit): ~0.28s @ log=17, ~0.78s @ log=19, ~3.0s @ log=21. This is not a fixed overhead — it is work running serial that scales with the problem.

## Sweep 2 — dense threads at max size (log-rows=21, 14 points from 1 to 96)

**Goal:** at the largest problem size (2.1M rows), sweep thread count in fine granularity across all 48 physical cores and into SMT territory (64, 96). This resolves the curve's knee and tells us exactly where each prover peaks, where it degrades, and how big the gap gets at each operating point.

### Full results

| Threads | Lambda (s) | P3 (s) | L/P3  | Speedup Lambda | Speedup P3 | Eff. L | Eff. P3 |
|:------:|---------:|-------:|------:|---------------:|-----------:|-------:|--------:|
| 1      | 29.688   | 14.881 | 2.00× | 1.00×          | 1.00×      | 100%   | 100%    |
| 2      | 16.596   |  7.842 | 2.12× | 1.79×          | 1.90×      | 89%    | 95%     |
| 4      | 11.025   |  4.358 | 2.53× | 2.69×          | 3.42×      | 67%    | 85%     |
| 6      |  7.910   |  3.142 | 2.52× | 3.75×          | 4.74×      | 63%    | 79%     |
| 8      |  6.213   |  2.485 | 2.50× | 4.78×          | 5.99×      | 60%    | 75%     |
| 12     |  5.300   |  1.811 | 2.93× | 5.60×          | 8.22×      | 47%    | 69%     |
| 16     |  4.539   |  1.490 | 3.05× | 6.54×          | 9.99×      | 41%    | 62%     |
| 20     |  4.347   |  1.299 | 3.35× | 6.83×          | 11.46×     | 34%    | 57%     |
| 24     |  4.175   |  1.187 | 3.52× | 7.11×          | 12.54×     | 30%    | 52%     |
| 32     |  3.832   |  1.053 | 3.64× | 7.75×          | 14.13×     | 24%    | 44%     |
| **40** | **3.737** | **0.909** | **4.11×** | 7.94× | **16.37×** | 20% | 41% |
| 48     |  3.686   |  0.983 | 3.75× | 8.05×          | 15.14×     | 17%    | 32%     |
| 64 SMT |  3.656   |  1.028 | 3.56× | 8.12×          | 14.48×     | 13%    | 23%     |
| 96 SMT |  3.600   |  0.977 | 3.69× | **8.25×**      | 15.23×     | 9%     | 16%     |

### Four distinct regimes

The curve is not a single smooth shape. It breaks into four zones, each telling us what is bottlenecking at that operating point:

- **Zone A (T=1→8): near-linear on both.** P3 efficiency stays ≥75%, Lambda ≥60%. Dominant parallel work (constraint eval, trace commit core) uses the first cores well.
- **Zone B (T=8→16): Lambda starts losing momentum.** Its per-column parallelism (32-column `into_par_iter` in the commit) is running out of granularity. P3 keeps decaying smoothly — it parallelizes finer (row/tile-based).
- **Zone C (T=16→40): Lambda flattens; P3 keeps gaining, peaks at T=40.** Lambda goes 4.54s → 3.74s (only −18% with 2.5× more threads). P3 goes 1.49s → 0.91s (−39%). **The L/P3 gap peaks here at 4.11×** — P3 is at its best, Lambda already capped.
- **Zone D (T=48→96, SMT): diverges.** Lambda gains a bit more (3.69 → 3.60, +1.5%) because its serial-ish section has memory stalls hyperthreads can hide. P3 stalls/regresses (0.98 → 1.03 → 0.98) — contention wins.

### Peak points

| Prover | Peak T | Peak time | Peak speedup | What it means |
|:------:|:------:|:---------:|:------------:|---------------|
| Lambda | T=96 SMT | 3.600s    | 8.25×       | But 95% of that is reached by T=16; going from 16 to 96 shaves only 20%. |
| P3     | T=40     | 0.909s    | 16.37×      | Clear peak, regresses slightly past 40. |

### Amdahl fit — where Lambda is stuck

Solving `S(T) = T / (1 + s·(T-1))` at multiple points:

| Prover | s @ T=32 | s @ T=48 | s @ T=96 | Theoretical max speedup (1/s) |
|:------:|:--------:|:--------:|:--------:|:-----------------------------:|
| Lambda | 10.1%    | 10.6%    | 11.2%    | ~9-10×                        |
| P3     | 4.1%     | —        | —        | ~27×                          |

- **Lambda is at ~90% of its Amdahl ceiling already.** 8.25× observed vs 9-10× theoretical. Adding cores will not help — the fix has to be in the serial section.
- **P3 is bandwidth/contention-limited, not Amdahl-limited.** 16.4× observed vs 27× theoretical — still has ~40% of parallel headroom that the hardware is eating.

### Conclusions

1. **Lambda parallelism saturates around T=16-24** (5.60× → 7.11×). Past that the marginal return is <5% per doubling.
2. **P3 scales efficiently through T=40** (16.37×). Past that, SMT contention eats the gains.
3. **The ratio widens to ~4× exactly when you would want the proof to be fastest** (big proof on a big server, many threads).
4. **SMT helps Lambda marginally at this size (+1-2%)** — different from smaller sizes where it hurt. The hyperthreads can hide memory stalls in the serial chunk.
5. **The highest-ROI optimization target is Lambda's ~10% serial section.** If halved, Lambda max speedup would jump from 8× toward ~14-15× — closing most of the gap without needing per-phase rework.

### Operational recommendations (at log-rows=21)

- **Cost/time sweet spot for a single Lambda proof: T=16-24.** Going from 48 down to 24 threads costs ~15% more wall-time (3.69s → 4.18s) but frees 24 cores for another job.
- **Throughput mode — run two Lambda proofs in parallel at 24 threads each**: each takes ~4.18s; two proofs complete in the same 4.18s wall-time, effective throughput 2.09s/proof. Beats the 3.69s of a single 48-thread proof if the memory bandwidth allows (EPYC has 12-channel DDR5, should). Worth a real side-by-side test.
- **For P3, always run at T=40**, not T=48. Small but consistent (~7%) win.

## Bottom line

- For a single proof on a big server, **Lambda hits a ~8× parallelism ceiling**; P3 reaches 16×. That is the main gap.
- **Sweet spot for Lambda is T=16-24**. Giving it more threads barely helps.
- **If the workload is throughput-oriented, running two Lambda proofs in parallel (24 threads each) may beat one proof at 48 threads** (~4.2s/proof × 2 proofs simultaneously = effective 2.1s/proof, vs 3.6s/proof solo). Worth validating.
- **Top next step to close the gap**: profile Lambda at T=48, log-rows=21 with `perf record`, identify the ~10% O(N) serial section (main suspects: trace-commit Merkle build, FRI fold serial chain, deep composition). That is where the 4× gap is hiding.

Raw numbers, per-run traces, and full analysis below.

---

## 2026-04-21 — Parallelism scaling sweep (SIMD off)

### Setup

- **Server**: `vm-benchmarks-1`, AMD EPYC 9454P (48 physical cores, 96 logical with SMT=2, Zen 4).
- **Arch**: `x86_64`.
- **SIMD**: OFF on both provers (`RUSTFLAGS="-C target-feature=-avx2,-avx512f"`, residual SSE2 in p3-keccak — ~7% of total prove time per prior M1 analysis).
- **AIR**: Shared Fibonacci. `log-rows=19` (524288 rows), `num-sequences=16` (32 columns), blowup=2, 219 FRI queries, grinding=0, extension degree 3 on both sides (p3-goldilocks-patched fork).
- **Runs**: 3 runs per `(T, prover)`, median reported.
- **Branch**: `bench_vs_p3`.
- **Lambda entry point**: bench calls `Prover::prove(...)` which is a thin wrapper over `multi_prove` with a single-element AIR slice (`crypto/stark/src/prover.rs:1878-1894`). There is no separate "legacy" / "unoptimized" path — both share the same implementation.
- **Raw reports**: `bench_vs_plonky3/reports/scaling_scalar/t{1,2,4,8,16,32,48,64,96}/`.

### Results

| Threads | Lambda (s) | P3 (s) | L/P3  | Speedup Lambda | Speedup P3 | Eff. Lambda | Eff. P3 |
|:------:|-----------:|-------:|------:|---------------:|-----------:|:-----------:|:-------:|
| 1      | 7.149      | 3.688  | 1.94× | 1.00×          | 1.00×      | 100%        | 100%    |
| 2      | 4.041      | 2.031  | 1.99× | 1.77×          | 1.82×      | 88.4%       | 90.8%   |
| 4      | 2.424      | 1.201  | 2.02× | 2.95×          | 3.07×      | 73.7%       | 76.8%   |
| 8      | 1.545      | 0.714  | 2.16× | 4.63×          | 5.17×      | 57.8%       | 64.6%   |
| 16     | 1.157      | 0.398  | 2.91× | 6.18×          | 9.27×      | 38.6%       | 57.9%   |
| 32     | 0.959      | 0.318  | **3.02×** | **7.46×**  | **11.60×** | 23.3%       | 36.2%   |
| 48     | 0.944      | 0.323  | 2.92× | 7.57×          | 11.42×     | 15.8%       | 23.8%   |
| 64 SMT | 0.949      | 0.332  | 2.86× | 7.53×          | 11.11×     | 11.8%       | 17.4%   |
| 96 SMT | 0.967      | 0.346  | 2.80× | 7.39×          | 10.66×     | 7.7%        | 11.1%   |

Individual run times for each 3-run batch are in each subfolder's `results.tsv` / `*.log`.

### Observations

1. **P3 scales better**. Peak speedup: P3 ~11.6× (T=32), Lambda ~7.5× (T=32-48). P3 extracts ~55% more parallelism.

2. **The Lambda/P3 gap grows with threads**. L/P3 ratio goes from **1.94× @ T=1** → **3.02× @ T=32**. It doubles. Not a constant offset: parallelism exposes serial (or coarse-grained) bottlenecks in Lambda that P3 does not have.

3. **Saturation at ≈32 cores** on both sides. From T=16 to T=32 Lambda improves only 1.21× (ideal would be 2×); from T=32 to T=48 it barely moves (0.959 → 0.944, 1.02×). P3 scales a bit better up to T=32 (1.25× from 16 to 32) but also flattens there.

4. **SMT does not help**. T=64 and T=96 are worse than T=48 on both provers. Cache/memory contention and serial bottlenecks dominate.

5. **Estimated serial fraction (Amdahl)**:
   - Lambda: max speedup 7.57× ⇒ serial fraction ≈ **13%**.
   - P3:     max speedup 11.60× ⇒ serial fraction ≈ **8%**.

6. **Single-thread: 1.94×**. Consistent with prior M1 analysis (Lambda ~3.0× slower due to full-LDE constraint eval + per-column FFT + deep composition overhead). That T=1 here is below the 3× reported on M1 suggests the EPYC hides some of the single-thread gap, which then surfaces under parallelism.

### Hypotheses for the scaling gap

- **Constraint eval** (30% of single-thread time in Lambda vs 14% in P3): if Lambda's `into_par_iter` parallelizes only across columns, 32 columns is the natural ceiling — matches the observed saturation at T=32.
- **Per-column FFT vs batched FFT** in P3 (`Radix2DitParallel`): P3 exposes finer-grained parallelism.
- **Deep composition + FRI fold** in Lambda: `par_iter` prototypes were stashed on another branch; the ~3% improvement measured on M1 suggests this is NOT the main driver.
- **Keccak** (17% of the prove): SSE2 residual identical on both sides, does not explain the scaling delta.

### Next steps

1. **Profile Lambda at T=48** with `cargo flamegraph` or `perf record -F 999 --call-graph dwarf -- ./prove_bench --prover lambda --log-rows 19 --num-sequences 16` to locate the ~13% serial section.
2. Repeat the sweep at **`log-rows=17`** and **`log-rows=21`** to see if the bottleneck shifts with problem size (memory vs compute).
3. Compare L/P3 ratio @ T=48 against prior M1 results (`1.07s / 0.35s = 3.05×` from context) — consistent with the 2.92× observed here.

### Raw files

- `bench_vs_plonky3/reports/scaling_scalar/t{1,2,4,8,16,32,48,64,96}/results.tsv`
- `bench_vs_plonky3/reports/scaling_scalar/t{1,2,4,8,16,32,48,64,96}/metrics.txt`
- `bench_vs_plonky3/reports/scaling_scalar/t{1,2,4,8,16,32,48,64,96}/raw/{lambda,p3}_log19_run{1,2,3}.stdout`
- `bench_vs_plonky3/reports/scaling_scalar/t*.log` (full stdout per sweep)

---

## 2026-04-21 — Multi-size parallelism sweep (SIMD off)

Same setup (EPYC 9454P, scalar, `--num-sequences 16`, 3 runs/median), sweeping three problem sizes and 7 physical-thread points (1, 2, 4, 8, 16, 32, 48). SMT skipped (already known to be non-additive).

**Raw reports**: `bench_vs_plonky3/reports/scaling_scalar_multisize/t{1,2,4,8,16,32,48}/` — each `results.tsv` contains 3 rows (log-rows 17/19/21).

### Median times (s)

| Threads | L @17 | P3 @17 | L/P3 @17 | L @19 | P3 @19 | L/P3 @19 | L @21  | P3 @21 | L/P3 @21 |
|:------:|------:|-------:|---------:|------:|-------:|---------:|-------:|-------:|---------:|
| 1      | 1.878 | 0.918  | 2.05×    | 7.189 | 3.687  | 1.95×    | 29.591 | 14.876 | 1.99×    |
| 2      | 1.111 | 0.545  | 2.04×    | 4.040 | 2.017  | 2.00×    | 16.591 |  7.842 | 2.12×    |
| 4      | 0.750 | 0.330  | 2.27×    | 2.756 | 1.212  | 2.27×    | 10.995 |  4.355 | 2.53×    |
| 8      | 0.420 | 0.191  | 2.20×    | 1.545 | 0.721  | 2.14×    |  6.329 |  2.475 | 2.56×    |
| 16     | 0.337 | 0.127  | 2.65×    | 1.140 | 0.457  | 2.50×    |  4.561 |  1.490 | 3.06×    |
| 32     | 0.316 | 0.109  | 2.90×    | 0.967 | 0.376  | 2.57×    |  3.793 |  0.943 | **4.02×**|
| 48     | 0.314 | 0.113  | 2.78×    | 0.929 | 0.328  | 2.83×    |  3.640 |  0.924 | 3.94×    |

### Speedup vs T=1

| Threads | S(L)@17 | S(P3)@17 | S(L)@19 | S(P3)@19 | S(L)@21 | S(P3)@21 |
|:------:|--------:|---------:|--------:|---------:|--------:|---------:|
| 1      | 1.00×   | 1.00×    | 1.00×   | 1.00×    | 1.00×   | 1.00×    |
| 2      | 1.69×   | 1.68×    | 1.78×   | 1.83×    | 1.78×   | 1.90×    |
| 4      | 2.50×   | 2.78×    | 2.61×   | 3.04×    | 2.69×   | 3.42×    |
| 8      | 4.47×   | 4.81×    | 4.65×   | 5.11×    | 4.68×   | 6.01×    |
| 16     | 5.57×   | 7.23×    | 6.31×   | 8.07×    | 6.49×   | 9.98×    |
| 32     | 5.94×   | 8.42×    | 7.43×   | 9.81×    | 7.80×   | 15.78×   |
| 48     | **5.98×** | **8.12×** | **7.74×** | **11.24×** | **8.13×** | **16.10×** |

### Observations

1. **As problem size grows, P3 scales dramatically better; Lambda barely improves.**
   - P3 max speedup: 8.1× (log=17) → 11.2× (log=19) → **16.1× (log=21)**.
   - Lambda max speedup: 6.0× → 7.7× → **8.1×**. Hardly moves when the problem quadruples.
   - Takeaway: P3 has parallelism that grows with available work; Lambda hits a ceiling near 8×.

2. **The L/P3 ratio worsens with both size and thread count — worst case is big + parallel.**
   - T=1: ~2× flat across all sizes (fixed single-thread offset).
   - T=48: 2.78× (log=17), 2.83× (log=19), **3.94× (log=21)**.
   - T=32 @ log=21 is the worst point: **4.02×**. Exactly when it matters most (large proof on a big server), Lambda is 4× slower.

3. **Estimated Amdahl serial fraction (from max speedup at T=48)**:

   | log-rows | Lambda serial % | P3 serial % | Lambda abs serial (s) | P3 abs serial (s) |
   |:------:|:--------------:|:-----------:|:---------------------:|:-----------------:|
   | 17     | ~15%           | ~10%        | ~0.28                 | ~0.09             |
   | 19     | ~11%           | ~7%         | ~0.78                 | ~0.25             |
   | 21     | ~10%           | ~4%         | ~3.02                 | ~0.61             |

   **Lambda's serial portion grows almost linearly with N** (0.28 → 0.78 → 3.0s, a factor of ~10× when going from 131k to 2M rows = 16×). So it is **NOT a fixed overhead**: there is O(N) work running serial (or at too-coarse granularity to use 32+ cores).

   P3's serial portion also grows with N but more slowly: 0.09 → 0.25 → 0.61, and relative to the total it drops from 10% to 4%.

   **The serial Lambda/P3 ratio stays ~3× across all sizes.** The gap comes primarily from this section Lambda does not parallelize (or parallelizes poorly).

4. **Lambda saturates earlier on small problems, later on large ones.**
   - log=17: Lambda flattens at T=16 (5.57 → 5.98 from 16 to 48, only 7%).
   - log=19: flattens at T=32 (7.43 → 7.74).
   - log=21: still climbs from T=32 to T=48 (7.80 → 8.13, 4%). Could gain a little more with extra cores, but <10%.
   - P3 saturates at T=32 for log=17 and log=19; at log=21 it still scales well at T=48 (15.78× → 16.10×).

5. **Reproducibility at the overlap point (log=19):** today's numbers match yesterday's within the ±2-5% expected run-to-run variance (Lambda T=4 is the noisiest point: 2.756 today vs 2.424 yesterday, because one outlier run of 2.444 dominates the 3-run median).

### Hypotheses, refined

With evidence from three sizes, Lambda's "serial O(N)" is the main suspect. Candidates:

- **Trace commit (FFT + Merkle)**: Lambda does per-column FFT, which parallelizes only up to 32 threads (# columns). P3 uses batched FFT (`Radix2DitParallel`) that parallelizes per-row / per-tile. This is suspect #1 for the O(N) serial.
- **Deep composition polynomial construction**: sequential in Lambda per prior analysis, and cost scales with N.
- **FRI commit phase**: each round builds a smaller Merkle, but rounds are a serial chain. Cost ~O(N log N), mostly serial in Lambda.
- **Constraint eval** is probably NOT the driver here: if it were, the Lambda ceiling would be exactly 32 threads (=# columns) with proportional speedup, but Lambda tops out at 8× — far below 32. The bottleneck is earlier in the pipeline.

### Next steps (updated)

1. **Profile Lambda at T=48, log-rows=21** (that's where the gap opens the most):
   ```bash
   perf record -F 499 --call-graph dwarf -g \
     -- ./target/release/prove_bench --prover lambda --log-rows 21 --num-sequences 16
   perf report --stdio | head -80
   ```
   Look for functions with high `% self` that are NOT under `rayon::*` — those are the serial ones.

2. **Check trace commit granularity**: in `crypto/stark/src/prover.rs`, is the Merkle tree built with `par_iter` per column (bad, 32-thread ceiling) or chunked (better)?

3. **Instrument per-phase timings**: add phase timers (trace commit / constraint eval / deep comp / FRI) in `prover::prove()` behind a `bench-timing` feature flag. Compare per-phase L/P3 ratio against the total.

4. **Try log-rows=22-23**: if Lambda still tops at ~8× for larger problems, that confirms the ceiling. If it improves, the serial section is further diluting.

### Raw files (multi-size)

- `bench_vs_plonky3/reports/scaling_scalar_multisize/t{1,2,4,8,16,32,48}/results.tsv` (3 rows each: log-rows 17/19/21)
- `bench_vs_plonky3/reports/scaling_scalar_multisize/t{1,2,4,8,16,32,48}/metrics.txt`
- `bench_vs_plonky3/reports/scaling_scalar_multisize/t{1,2,4,8,16,32,48}/raw/{lambda,p3}_log{17,19,21}_run{1,2,3}.stdout`
- `bench_vs_plonky3/reports/scaling_scalar_multisize/t*.log`

---

## 2026-04-22 — Clarification: bench uses `prove` (= `multi_prove` for 1 AIR)

**Question from colleague:** "Does the P3 comparison use `multi_prove`? Because the normal `prove` flow is not very optimized."

**Answer (verified in code, read-only):**

- The bench calls `Prover::<F, E, _>::prove(...)` in `bench_vs_plonky3/src/bin/prove_bench.rs:144`.
- **`prove` is a thin wrapper over `multi_prove`**. Defined as a default method on the `IsStarkProver` trait at `crypto/stark/src/prover.rs:1878-1894`:
  ```rust
  // "This is equivalent to calling `multi_prove` with a single-element slice."
  let air_trace_pairs = vec![(air, trace, pub_inputs)];
  Self::multi_prove(air_trace_pairs, transcript)
      .map(|mut multi_proof| multi_proof.proofs.remove(0))
  ```
- **No separate "legacy" / "unoptimized" code path exists.** `prove` and `multi_prove` share the same implementation (lines 1477-1876 in the same file).
- With 1 AIR, all inner loops in `multi_prove` iterate once — no multi-table overhead for this case.
- There is no single-table wrapper or `if num_airs == 1` fast path; the design intentionally unifies both paths.

**Conclusion:** the benchmark uses the optimized path. The colleague's concern does not apply in the current codebase — they may have been thinking of code that has since been refactored or of a different implementation.

**Action:** none on the bench. Just reply to the colleague.

---

## 2026-04-22 — Dense sweep at log-rows=21 (14 points, physical + SMT)

**Motivation:** the previous sweep only had power-of-2 thread counts at log=21. This dense sweep adds 6, 12, 20, 24, 40 to resolve the curve's knee, plus 64/96 to close the SMT case at large problem size.

**Raw reports:** `bench_vs_plonky3/reports/scaling_log21_dense/t{1,2,4,6,8,12,16,20,24,32,40,48,64,96}/`.

### Results (log-rows=21, 2,097,152 rows, scalar, 3 runs / median)

| Threads | Lambda (s) | P3 (s) | L/P3   | Speedup Lambda | Speedup P3 | Eff. L | Eff. P3 |
|:------:|----------:|-------:|-------:|---------------:|-----------:|-------:|--------:|
| 1      | 29.688    | 14.881 | 2.00×  | 1.00×          | 1.00×      | 100%   | 100%    |
| 2      | 16.596    |  7.842 | 2.12×  | 1.79×          | 1.90×      | 89%    | 95%     |
| 4      | 11.025    |  4.358 | 2.53×  | 2.69×          | 3.42×      | 67%    | 85%     |
| 6      |  7.910    |  3.142 | 2.52×  | 3.75×          | 4.74×      | 63%    | 79%     |
| 8      |  6.213    |  2.485 | 2.50×  | 4.78×          | 5.99×      | 60%    | 75%     |
| 12     |  5.300    |  1.811 | 2.93×  | 5.60×          | 8.22×      | 47%    | 69%     |
| 16     |  4.539    |  1.490 | 3.05×  | 6.54×          | 9.99×      | 41%    | 62%     |
| 20     |  4.347    |  1.299 | 3.35×  | 6.83×          | 11.46×     | 34%    | 57%     |
| 24     |  4.175    |  1.187 | 3.52×  | 7.11×          | 12.54×     | 30%    | 52%     |
| 32     |  3.832    |  1.053 | 3.64×  | 7.75×          | 14.13×     | 24%    | 44%     |
| 40     |  3.737    |  0.909 | **4.11×** | 7.94×       | **16.37×** | 20%    | 41%     |
| 48     |  3.686    |  0.983 | 3.75×  | 8.05×          | 15.14×     | 17%    | 32%     |
| 64 SMT |  3.656    |  1.028 | 3.56×  | 8.12×          | 14.48×     | 13%    | 23%     |
| 96 SMT |  3.600    |  0.977 | 3.69×  | **8.25×**      | 15.23×     | 9%     | 16%     |

### Four distinct scaling regimes

The dense sweep shows the curve is not a single smooth shape; there are **four zones** with different dynamics. Each tells us something about which bottleneck dominates.

**Zone A — T=1→8: near-linear scaling on both sides**

- Lambda efficiency: 89% → 67% → 60%. Parallelizes, but with non-trivial overhead.
- P3 efficiency: 95% → 85% → 75%. Near ideal.

Interpretation: the dominant parallel work (constraint eval, core trace commit) uses the first cores efficiently. The difference shows up early: P3 has better granularity in the commit because it uses batched row-wise FFT; Lambda is per-column.

**Zone B — T=8→16: Lambda starts losing momentum**

- Lambda: 60% → 47% → 41%. Sharp drop.
- P3: 75% → 69% → 62%. Smooth decay.

Interpretation: Lambda is approaching the ceiling of its coarse-grained parallelism. The `into_par_iter` over 32 columns of the commit is already split between 16 workers; each worker handles 2 columns. Going past that would require splitting a single column across workers.

**Zone C — T=16→40: Lambda flattens; P3 keeps gaining**

- Lambda: 4.539s → 3.737s. Only -18% with 2.5× more threads. Efficiency down to 20%.
- P3: 1.490s → 0.909s. -39% with 2.5× more threads. Efficiency down to 41%. **P3 hits its absolute peak at T=40: 0.909s, 16.37× speedup.**

Interpretation: this is where the gap opens. P3 keeps parallelizing at fine grain (LDE tiles, FRI folds with rayon) and exploits 40 cores. Lambda has O(N) work running serial or poorly parallelized — **the ~10% Amdahl serial section is now dominant**.

**The L/P3 gap peaks here (4.11× at T=40)** precisely because P3 is at its best while Lambda has already capped.

**Zone D — T=48→96 (SMT): each prover reacts differently**

- Lambda: 3.686 → 3.656 → 3.600. Marginal improvement (1.5%), no regression.
- P3: 0.983 → 1.028 → 0.977. Stalls with a slight regression; the peak is behind.

Interpretation: in SMT, hyperthread pairs share the same physical core. Lambda gains a bit because its serial-ish section has memory stalls that hyperthreads can hide. P3 is tight enough that hyperthreads only add contention.

### Amdahl serial fraction (fitted at multiple points)

Fitting `S(T) = T / (1 + s·(T-1))`:

| Prover | s @ T=32 | s @ T=40 | s @ T=48 | s @ T=96 | Smax = 1/s |
|:------:|:--------:|:--------:|:--------:|:--------:|:----------:|
| Lambda | 10.1%    | —        | 10.6%    | 11.2%    | ~9-10×     |
| P3     | 4.1%     | **3.7%** | —        | —        | ~27×       |

- **Lambda is at ~90% of the Amdahl ceiling implied by its serial fraction.** Observed max 8.25×, theoretical max 9-10×. Adding cores will not change the picture — we need to **attack the serial section**.
- **P3 still has room** (16.4× observed vs 27× theoretical). The gap is eaten by memory bandwidth and contention, not Amdahl.

### Practical implications

1. **For Lambda, scaling returns drop off steeply past T=16.** If you have a 48-core farm and want to use it, a single Lambda prove leaves ~30 cores idle for most of the run.

2. **Running two proofs in parallel on the same server may beat giving 48 threads to one proof.** At 24 + 24 threads, each proof takes ~4.18s (Lambda @ T=24). Two proofs finishing in 4.18s = **throughput equivalent to one proof in 2.09s**, much better than the 3.64s a single proof delivers with 48 threads. Worth validating with a real side-by-side run; memory bandwidth might bite, but the EPYC's 12-channel DDR5 should handle it.

3. **Cost/time sweet spot for Lambda**: around **T=16-24**. Going from 29.7s at T=1 to ~4.5s at T=16 (6.5× speedup); going from 24 to 48 threads only shaves 0.5s more.

4. **For P3, the clear sweet spot is T=32-40**. Peak at T=40 (0.909s). More threads degrade or flatten.

### What's the ~10% Lambda serial section?

Candidates, ordered by likelihood given the evidence:

1. **Trace-commit Merkle tree** (high): if built without parallelizing across tree levels, it is O(N) serial. P3 uses `MerkleTreeMmcs` with parallel build.
2. **FRI fold serial chain**: rounds are inherently serial against each other (round k depends on round k-1), but each round internally should parallelize. If Lambda does not parallelize well within a round, the per-round cost adds serially.
3. **Deep composition polynomial**: summing the combinations is O(N); if not chunked properly it runs serial.
4. **Setup / grinding / transcript**: small fixed costs. Do not explain 10% at log=21 but do explain the 15% at log=17 (where absolute overhead inflates the serial %).

Next step is to profile Lambda at T=48 / log=21 with `perf record` and find high `% self` functions outside `rayon::`. That profile would close the analysis.

### Raw files (dense log=21 sweep)

- `bench_vs_plonky3/reports/scaling_log21_dense/t{1,2,4,6,8,12,16,20,24,32,40,48,64,96}/results.tsv`
- `bench_vs_plonky3/reports/scaling_log21_dense/t{...}/metrics.txt`
- `bench_vs_plonky3/reports/scaling_log21_dense/t{...}/raw/{lambda,p3}_log21_run{1,2,3}.stdout`
- `bench_vs_plonky3/reports/scaling_log21_dense/t*.log`

---

## Appendix — Glossary

### Amdahl's Law and "Amdahl fit"

**Amdahl's Law** predicts how much speedup a parallel program can achieve given the fraction of work that must run serially. If `s` is the serial fraction (between 0 and 1) and `T` is the number of threads:

```
S(T) = T / (1 + s · (T - 1))
```

where `S(T)` is the speedup measured vs the single-threaded version.

**Key consequence:** as `T → ∞`, the speedup approaches `1/s`. If 10% of the program is serial, the speedup ceiling is ~10× regardless of how many cores are thrown at it.

**Example ceilings:**

| Serial (s) | Theoretical max speedup (1/s) | Speedup @ T=48 |
|:---------:|:-----------------------------:|:---------------:|
| 1%        | 100×                          | 31.2×           |
| 5%        | 20×                           | 15.2×           |
| 10%       | 10×                           | 8.8×            |
| 25%       | 4×                            | 3.8×            |
| 50%       | 2×                            | 1.96×           |

**"Amdahl fit"** = using measured speedups at several thread counts to solve for `s`. Example: at `T=32`, Lambda's measured speedup is `7.75×`. Plugging in:

```
7.75 = 32 / (1 + s · 31)
1 + 31s = 32 / 7.75 = 4.13
31s = 3.13
s ≈ 0.101 = 10.1%
```

We did this fit at T=32, T=48, and T=96 for Lambda, getting s between 10.1% and 11.2% — a consistent result across thread counts means the Amdahl model explains the observed behavior well. The inverse `1/s ≈ 9-10×` is therefore the theoretical max speedup for Lambda on this workload.

**How we used it in the analysis:**

- **Lambda**: s ≈ 10.6% → ceiling ~9-10×. Observed 8.25× at T=96 = 90% of the ceiling already. Adding cores will not help; the optimization target is the serial section.
- **P3**: s ≈ 4.1% → ceiling ~27×. Observed 16.4× = only 60% of the ceiling. P3 is **not Amdahl-limited** — it has headroom, but another factor (most likely memory bandwidth / cache contention) is limiting scaling.

Different diagnoses point to different fixes — for Lambda, rewrite the serial chunk; for P3, improve memory access patterns or reduce bandwidth pressure.

**Limitations of the model.** Amdahl assumes the parallel portion scales linearly with `T`, which is an approximation:
- It does not account for memory bandwidth or cache contention.
- It does not model synchronization overhead between threads.
- It assumes `s` is constant, but empirically `s` can grow with problem size (as we saw: Lambda's absolute serial time grows near-linearly with N).

For these reasons we report "Amdahl fit" as an estimate of the ceiling and a pointer to where optimization effort should go, not as an exact description of the runtime.
