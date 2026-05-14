# `bench_vs_plonky3` — Methodology and Improvement Analysis

This document explains **what** `bench_vs_plonky3` measures, **how** it measures
it, and **what could be done differently**. It complements `README.md` (usage),
`ANALYSIS_LOG.md` (raw experiments), and `inform.md` (final scaling report).

> Last updated: 2026-05-04 — branch `bench_vs_p3`.

---

## TL;DR

- The harness times **prove-only wall-clock** (`Instant::now()` around `prove`)
  on a single shared multi-sequence Fibonacci-pair AIR. Trace generation and AIR
  construction are **outside** the timer. Verification is timed separately and
  does not affect the L/P3 ratio.
- 10 runs per `(size, prover)` by default, **median + CV** reported. Raw
  per-run metrics are saved in `raw_metrics.tsv`.
- Per-phase breakdown is **opt-in** via `run.sh --breakdown` or the older
  ignored test (`instruments_breakdown`). The nightly does **not** enable it.
- **Now measured**: proof size, verifier time, process peak RSS, rows/sec,
  cells/sec. **Still not measured**: perf counters and phase breakdown in CI.
- The chosen AIR (Fibonacci-pair) is **single-table, base-only, no LogUp,
  no RAP challenges** — it exercises FFT + Merkle commit + FRI, but **not**
  the lookup-argument bookkeeping that dominates Lambda VM in production.
  The L/P3 ratio measured here may not reflect the realistic gap.

---

## 1. What is measured

### 1.1 The AIR

Both sides prove a **multi-sequence Fibonacci-pair AIR**, 2 Fibonacci steps
packed per row:

```text
local.left  = x_{2i}
local.right = x_{2i+1}
next.left   = x_{2i+2} = local.left + local.right
next.right  = x_{2i+3} = local.right + next.left
```

For `num_sequences = N`:

| Property                  | Value          |
|---------------------------|----------------|
| Main columns              | `2N`           |
| Aux columns               | `0`            |
| Transition constraints    | `2N` (degree 1, `end_exemptions = 1`) |
| Boundary constraints      | `2N` (pin `(a,b)` at row 0) |
| Trace rows                | `1 << log_rows` (default `19` ⇒ 524 288) |
| Lookups / RAP challenges  | none           |

**Why "pair" and not the simple Fibonacci**: Lambda's natural Fibonacci had a
3-row transition (`row+2`), and Plonky3's `Air` API uses a 2-row local/next
window. Packing two steps per row maps Lambda's transition into P3's window
**without changing the committed cell count** (`bench_vs_plonky3/src/lambda_fibonacci_pair.rs:1-17`,
`bench_vs_plonky3/src/plonky3_fibonacci.rs:11-18`).

**Cell-by-cell equivalence**: the test `lambda_pair_trace_matches_plonky3_trace`
at `bench_vs_plonky3/src/lib.rs:343` checks every `(row, col)` of both traces
for `num_sequences=3, rows=16`. This is the only correctness coupling — it
runs on a tiny fixed input and is **not** parameterised, so it does not assert
the equality at the benchmark sizes.

### 1.2 Shared configuration

Both sides use the same proof params (`bench_vs_plonky3/src/lib.rs:23` and
`bench_vs_plonky3/src/plonky3_config.rs:51`):

| Parameter     | Value | Rationale                                     |
|---------------|-------|-----------------------------------------------|
| Blowup factor | 2     | Matches Lambda production `GoldilocksCubicProofOptions::with_blowup(2)` |
| FRI queries   | 219   | Matches production                            |
| Grinding      | 0     | Excluded from timing on both sides            |
| Coset offset  | 3     | Matches Lambda                                |
| Extension     | degree 3 | Lambda: `x³ - 2` (binomial); Plonky3 upstream: `x³ - x - 1` (trinomial) |

The `p3-*` crates point to upstream Plonky3 (`Plonky3/Plonky3.git`) with
no rev pin — `Cargo.lock` captures the HEAD of the default branch at fetch
time. Upstream gained cubic-extension support for Goldilocks in PR #1497
(Apr 2026, commit `251b13d4`) via the `CubicTrinomialExtendable` trait,
with irreducible `x³ - x - 1`. Lambda's
`Degree3GoldilocksExtensionField` uses `x³ - 2`. Both extensions are
`GF(p³)` with `p = 2⁶⁴ - 2³² + 1`, so the security level is identical —
the difference is in ext-field multiplication: the binomial reduction
(`x³ ≡ 2`) is cheaper than the trinomial one (`x³ ≡ x + 1`) by a few
operations per multiplication. The grueso of prove time (FFT, Merkle
commit of the main trace) runs on the base field Goldilocks and is
unaffected; the extension cost shows up in FRI folds, DEEP, and challenges
(typically <15% of total prove time).

**Fairness lost / not addressed**: the Fiat-Shamir transcript hashes differ
(Lambda: `DefaultTranscript`; P3: `SerializingChallenger64<HashChallenger<u8, ByteHash, 32>>`).
Only the PCS / MMCS hashing matches (Keccak-based on both sides). This is
fine for compute-time comparison but means proofs are not cross-verifiable.

### 1.3 Measurement window

Two execution paths, both prove-only:

1. **`bench_vs_plonky3/src/bin/prove_bench.rs`** — used by `run.sh` and the
   nightly. The prove ratio still comes from `Instant::now()` around a single
   call:
   - `run_lambda` (`prove_bench.rs:128`): `Prover::prove(...)` only.
   - `run_p3` (`prove_bench.rs:154`): `p3_uni_stark::prove(...)` only.

   **Excluded** from the prove timer on both sides: `compute_trace`,
   `create_public_inputs` / `public_values`, AIR / config construction,
   `verify`, proof serialisation. After proving, the harness serialises the
   proof with `serde_cbor` to report `proof_size_bytes`, then times `verify`
   separately and reports process peak RSS via `getrusage`.

2. **`bench_vs_plonky3/benches/stark_comparison.rs`** — a Criterion benchmark
   that exists in the repo. It uses `Throughput::Elements` reporting and is
   not exercised by `run.sh`. The `prove_bench` bin is the source of truth for
   the nightly TSV.

### 1.4 Phase breakdown — `run.sh --breakdown`

For optimization work, prefer:

```bash
./bench_vs_plonky3/run.sh --log-rows 19 --num-sequences 16 --runs 1 \
  --breakdown --report-dir /tmp/p3_breakdown --no-color
```

This rebuilds `prove_bench` with `--features instruments`, passes
`--breakdown` to the bin, and writes `breakdown.tsv` with one row per
Lambda phase/sub-op and one row per Plonky3 tracing span.

The older test path still exists:

Defined at `bench_vs_plonky3/src/lib.rs:80`. Hardcoded shape: `num_sequences=16`,
`rows = 1 << 19`. Activated by:

```bash
cargo test -p bench-vs-plonky3 --features instruments --release -- \
  instruments_breakdown --ignored --nocapture
```

The feature `instruments` ⇒ `stark/instruments` (`bench_vs_plonky3/Cargo.toml:53`,
`crypto/stark/Cargo.toml:39`). Without it, the timers in
`crypto/stark/src/instruments.rs` are no-ops and the breakdown prints zeros.

**Lambda side — explicit timers** (`crypto/stark/src/instruments.rs`):

- `prepass`, `main_commits`, `aux_build`, `aux_commit`, `rounds_2_4`
- `round1_sub` ⇒ `main_lde`, `main_merkle`
- `table_timings: Vec<(name, rows, dur, TableSubOps)>` per table:
  `constraints`, `comp_decompose`, `comp_commit`, `ood`, `deep_comp`,
  `deep_extend`, `fri_commit`, `queries`

**Plonky3 side — `tracing` spans**: a custom `tracing_subscriber::Layer`
(`P3TimingLayer`, `lib.rs:221`) wraps `on_new_span` / `on_enter` / `on_exit`,
accumulates per-span elapsed time at filter `LevelFilter::DEBUG`, and prints
spans sorted descending. Spans nest (`prove ⊃ compute_quotient_values`), so
`Σ spans > total` is expected.

> Caveat: the rayon re-entry guard (`lib.rs:248-258`) only starts timing on the
> first enter after each exit. Across-thread re-entries are not double-counted
> but can underreport when the span is concurrently active on multiple
> threads. Treat the P3 numbers as ordering hints rather than precise totals.

---

## 2. How it is measured

### 2.1 `run.sh` orchestration

Flags (`run.sh:40-89`): `--log-rows K…`, `--num-sequences N` (default 16),
`--runs N` (default 10), `--lambda-only`, `--p3-only`, `--report-dir DIR`,
`--scalar`, `--no-color`.

For each `(size, prover)`:
1. Builds `prove_bench` once with `cargo build --release` (`run.sh:160`).
2. Runs the binary `RUNS` times, capturing the tab-separated `METRICS` line.
3. Sorts the prove times and computes **median + CV** via `sort -g | awk`.
4. Prints summary: `log-rows | rows | Lambda prove median | Lambda CV | P3 prove median | P3 CV | L/P3 ratio`.

`--scalar` (`run.sh:117`): on `x86_64`/`amd64` exports
`RUSTFLAGS="-C target-feature=-avx2,-avx512f"`. Other arches: warning, no-op.
Disables AVX2/AVX-512 so Goldilocks (and most of Keccak) run scalar on both
sides; **residual SSE2 in `p3-keccak` (~7%) is intentionally not disabled**.

`--report-dir DIR` writes:

| File              | Content                                            |
|-------------------|----------------------------------------------------|
| `results.tsv`     | Summary TSV: prove medians/CVs, verify medians, proof-size medians, RSS medians, ratio, runs |
| `raw_metrics.tsv` | Per-invocation TSV: workload, prover, rows, cols, prove_s, verify_s, proof_size_bytes, peak_rss_kb, rows/sec, cells/sec |
| `metrics.txt`     | timestamp, git_sha, git_tree, arch, num_sequences, blowup, queries, scalar, slash-joined series |
| `raw/*.stdout`    | per-invocation bin stdouts                         |

### 2.2 Nightly CI

`.github/workflows/bench-vs-p3-nightly.yml`:

- Cron: `30 7 * * *` UTC (04:30 BA), plus `workflow_dispatch`.
- Runner: self-hosted `[bench]`. `timeout-minutes: 60`.
- Single command: `run.sh --log-rows 19 --num-sequences 16 --runs 10 --scalar --report-dir bench_vs_p3_artifacts --no-color`.
- Artifact `bench-vs-p3-nightly-{run_number}-{sha}` retained 90 days.
- `--breakdown` is **not** part of the nightly path because it changes the
  timing profile and is meant for diagnosis, not historical totals.

### 2.3 Metrics that are NOT reported

Confirmed absent from the bin / `run.sh` / nightly:

- **CPU / perf counters** — no `perf stat`, no PMU readings.
- **Per-phase breakdown in CI** — `--breakdown` is intentionally manual.

---

## 3. How peers measure (comparison)

| Dimension                | Lambda (this bench) | Plonky3 examples         | SP1                         | RISC0                       | OpenVM                      |
|--------------------------|---------------------|--------------------------|------------------------------|-----------------------------|------------------------------|
| Framework                | manual `Instant`; Criterion file present but unused by `run.sh` | Criterion (crate benches) + custom `main` (examples) | Custom `Instant::now()`      | Custom `hotbench`           | Divan (execute) + custom binary (prove) |
| AIR / program            | Fibonacci-pair (synthetic) | KeccakAir / Poseidon2 / FibAir (synthetic) | Real RISC-V ELF (fibonacci, tendermint, ssz-withdrawals) | Real RISC-V ELF (Fibonacci) | Real RISC-V ELF (revm, keccak, pairing) |
| Verify time              | No (bin) / yes (file Criterion bench) | Yes (within prove loop) | Yes (separate timer)         | Per-phase structural        | Yes (spanned)                |
| Proof size               | No                  | **Yes** (`report_proof_size`, postcard) | No                           | No                          | Not in prove benches         |
| Peak memory              | No                  | No (jemalloc set, not queried) | No                           | No                          | No                           |
| Throughput unit          | seconds only        | Criterion `Throughput`   | cycles/sec (kHz)             | rows/sec (Hz)               | guest cycles per fn (`--profiling`) |
| Phase breakdown          | Opt-in (`--features instruments`) | **Always-on** tracing spans (`tracing_forest`) | Per-phase `Instant`         | Separate bench per phase    | Always-on `tracing` spans    |
| Statistics               | 10 runs, median + CV | Criterion CV / percentiles | Single run                  | min/avg/max + S3 baseline delta | Divan: full distribution     |
| Regression detection     | Manual diff of artifacts | None automatic         | PR comment table to Slack/GH | **Auto via `github-action-benchmark` + S3 baseline** | Manual                       |
| CI cadence               | Nightly @ 07:30 UTC | Not found upstream       | Per-PR (64-cpu + CUDA)       | Per-PR (RTX 4090 + M2 Pro)  | Manual fixture upload        |

**Standout patterns from peers**:

- **Plonky3** treats per-phase breakdown as a first-class output. `tracing`
  spans are always active when a subscriber is registered, so every example
  prints a tree of phase timings without recompilation.
- **RISC0** benchmarks against a stored S3 baseline and posts the delta as a
  PR comment; a regression is visible at PR time, not weeks later.
- **SP1** reports throughput as **cycles per second** so numbers are
  comparable across programs of different lengths.
- **OpenVM** uses Divan for execution-only benches (gives full distribution)
  and a custom binary that emits structured output for ingestion.

---

## 4. What could we do differently — analysis

Ordered by **ROI** (insight unlocked / engineering effort).

### 4.1 Critical gaps — quick wins

| # | Change | Effort | Why it matters |
|---|--------|--------|----------------|
| A | **Measure proof size** (`serde_cbor`-serialise the proof, write `proof_size_bytes` into the TSV) | Done | The FRI optimisation plan (`PLAN-fri-optimizations.md`) trades commit work for smaller proofs. Without proof-size numbers, there is **no business case** for that work. P3 already publishes ~2-3× smaller proofs with `folding_factor=4` + `last_layer_degree_bound=7`. |
| B | **Measure verifier time** (call `verify`, time it, write `verify_s` into the TSV) | Done | Light-client / on-chain / recursion-cost depend on this. |
| C | **Measure peak RSS** (`getrusage(RUSAGE_SELF).ru_maxrss`) | Done | Hypothesis: Lambda uses 2-3× more memory. If true, it explains the scaling ceiling at log-rows ≥ 21 (memory-bandwidth bound). |
| D | **Bump runs to 10-20 + report CV** (instead of 3-run median) | Done | Three runs cannot reliably detect changes <8% (CV unbounded). With 10+ samples + CV we can declare regressions/wins at smaller margins. |
| E | **CPU pinning + governor=performance** (`taskset -c 0`, `cpupower frequency-set -g performance`) on the bench runner | S | Cuts variance 2-5× on shared/contended servers. Pre-requisite for D to be meaningful. |

### 4.2 Methodology upgrades

| # | Change | Effort | Why |
|---|--------|--------|-----|
| F | **Hyperfine wrapping `prove_bench`** instead of bash-loop median | S | Built-in warmup, outlier detection, JSON export, run-to-run cache flushing. Drop-in replacement of `run_prover()`. |
| G | **Migrate Lambda's `instruments` to `tracing` spans** | M | Aligns with P3's hierarchy. Output becomes apple-to-apple comparable. Removes the `--features instruments` recompile step (use `RUST_LOG=debug` instead). Enables the breakdown to run **always**, matching Plonky3 / OpenVM. |
| H | **`perf stat` per run** (cycles, instructions, IPC, LLC-load-misses, branch-misses) | M | Distinguishes bandwidth-bound vs compute-bound vs branch-mispredict. P3's FFT runs at IPC ~2.5; if Lambda measures ~1.2 the fix is batched/streaming FFT, not micro-ops. |
| I | **`flamegraph` on T=48, log-rows=21** | M | Direct attribution of the ~10% serial section that the Amdahl fit identified (`ANALYSIS_LOG.md`). |
| J | **Regression alerts**: stash the last green TSV in a known location, fail the CI when a new run drifts >X% beyond its CI | M | Today the artifact is saved 90 days but no one looks until something is suspect. RISC0's `github-action-benchmark` pattern is a template. |
| K | **Differential test of the AIR**: P3 verifies a Lambda proof and vice versa for a small input | L | Currently only the trace is asserted equal cell-by-cell. A cross-verify would catch divergence in commitments, FRI parameters, or transcript without anyone noticing. (Requires aligning Fiat-Shamir transcripts — large task.) |

### 4.3 Coverage gaps — the AIR itself

The Fibonacci-pair AIR exercises **FFT + Merkle commit + FRI** and **nothing
else**. Lambda VM in production spends ~30-50% of prove time on
lookup-argument bookkeeping and aux materialisation, none of which is
measured here. The L/P3 ratio in the nightly therefore reflects a
best-case-for-P3 / best-case-for-the-FFT-pipeline; it does **not** predict
how Lambda will compare against P3 on a real VM workload.

| # | Change | Effort | What new info |
|---|--------|--------|----------------|
| L | **Add a 1-lookup-column variant** (range-check on a single byte column) | S | Exercises aux build + RAP challenges; closes 80% of the realism gap for 20% of the effort. |
| M | **Mini multi-table variant**: 2 tables + 1 LogUp interaction between them | M | Isolates multi-table coordination cost (per-table Merkle, cross-table commitments). |
| N | **Reuse the Keccak-precompile AIR** (already in the repo, spec-compliant, multi-row, lookup-heavy, degree 3) | M | Highest realism; ports trivially because the AIR is already there. |
| O | **Add a Poseidon2 AIR variant** | L | Useful because P3 publishes Poseidon2 numbers — gives a triangulation point against the public benchmark. |

### 4.4 Sweep coverage

| # | Change | Effort | What new info |
|---|--------|--------|----------------|
| P | **Sweep `log-rows ∈ {14, 16, 18, 20, 22}`** | S | Today: only 17/19/21. Filling 14/16/18/20/22 maps the curve where the constant-overhead vs O(N) regimes meet (small) and where memory bandwidth bites (large). |
| Q | **Sweep `num-sequences ∈ {4, 8, 16, 32, 64}`** (8-128 columns) | S | Tests the columns-axis ceiling. Lambda's `into_par_iter` over columns saturates at `T = num_columns`; this sweep reveals exactly where. |
| R | **Sweep `blowup ∈ {2, 4, 8}`** | S | Isolates FFT cost. Blowup=8 is ≈4× more FFT work; if the L/P3 ratio does **not** scale predictably, there is non-FFT overhead being amortised. |
| S | **Sweep `queries ∈ {50, 100, 219, 400}`** | S | Maps the FRI query cost separately from trace generation. Useful for budget tradeoffs (acceptable security vs proof size). |
| T | **Add P3 degree-2 baseline** alongside the current degree-3 trinomial | M | Quantifies how much of the "P3 advantage" is attributable to the cheaper degree-2 extension. P3-as-shipped already supports both; the current bench uses degree-3 to match Lambda's grado. If the delta is large, the L/P3 ratio understates what users would see with default P3 settings. |

---

## 5. Recommended order of attack

If picking only a handful, this is the order with the best
information-per-engineering-hour:

1. **A + B + C** (proof size + verifier time + peak RSS). Single PR, ≤1 day,
   immediately closes the three biggest reporting gaps.
2. **D + E** (10-20 runs + CV + pinning + governor). One day. Without it,
   nothing else can be trusted at <8%.
3. **L** (add a 1-lookup variant). One PR. Restores realism that the current
   single-AIR bench cannot deliver.
4. **G** (migrate `instruments` to `tracing`). One week. Unlocks always-on
   breakdown and direct comparison with P3's tree.
5. **N** (Keccak-precompile AIR variant). One-two weeks. Highest-realism bench
   we can ship without inventing a new AIR.

The remaining items (`H`, `I`, `J`, `O–T`) are excellent follow-ups once 1-5
are in place. The current methodology is sufficient to detect 8-10%
regressions on a single AIR; with 1-5 it would detect 3% regressions on a
representative AIR with full breakdown — the level of rigour the optimisation
work in `PLAN-fri-optimizations.md` and `optimizations_3mi.md` requires.

---

## 6. References

- `bench_vs_plonky3/README.md` — usage.
- `bench_vs_plonky3/ANALYSIS_LOG.md` — raw experiments.
- `bench_vs_plonky3/inform.md` — final scaling report (parallelism).
- `PLAN-fri-optimizations.md` — FRI early-stop + folding factor plan.
- `optimizations_3mi.md` — chip-level optimisation entries.
- Plonky3 examples: `keccak-air/examples/`, `poseidon2-air/examples/`,
  `batch-stark/benches/prove_batch.rs` (uses `tracing_forest`).
- SP1 perf: `crates/perf/src/perf.rs`, `crates/eval/src/lib.rs`.
- RISC0 hotbench: `tools/hotbench/src/lib.rs`.
- OpenVM benches: `benchmarks/execute/benches/execute.rs`,
  `benchmarks/prove/src/bin/`.
