# Lambda STARK vs Plonky3 Benchmark

Compares **single-shot end-to-end proving time** for an identical multi-sequence
Fibonacci AIR. Complements `bench_vs/` (which compares Lambda VM vs SP1 on a
full guest program) by isolating the STARK prover — no VM execution, no trace
builder, just one AIR and two provers.

## What is measured

Both provers prove the same AIR:

- **Columns** = `2 × num_sequences` (default 16 sequences → 32 columns).
- **Rows** = `2 ^ log_rows` (default `19` → 524 288 rows).
- **Blowup** = 2 (matches Lambda production `GoldilocksCubicProofOptions::with_blowup(2)`).
- **FRI queries** = 219, grinding = 0.

The timing window on both sides is **`Instant::now()` around `prove`, no
verification, no proof serialization**:

| Phase                                | Lambda STARK | Plonky3 |
|--------------------------------------|:------------:|:-------:|
| Build AIR + trace                    | ❌ (outside) | ❌ (outside) |
| Build public inputs                  | ❌ (outside) | ❌ (outside) |
| Prove (Round 1 → Round 4)            | ✅          | ✅ (`p3_uni_stark::prove`) |
| Proof serialize / disk write         | ❌          | ❌ |
| Verify                               | ❌          | ❌ |

Lambda's trace, public inputs, and AIR are constructed via
`lambda_fibonacci_pair::{compute_trace, create_public_inputs, FibonacciPairMultiColAIR}`.
Plonky3's counterpart uses `plonky3_fibonacci::{P3FibonacciAir, generate_fibonacci_trace, public_values}`
with `plonky3_config::matched_params_config`. Both AIRs are **cell-by-cell
equivalent** — this is asserted by the `lambda_pair_trace_matches_plonky3_trace`
test.

## Usage

```bash
# Default: log-rows=19, num-sequences=16, runs=3, cubic extension, no scalar
./bench_vs_plonky3/run.sh

# Size sweep
./bench_vs_plonky3/run.sh --log-rows 17 18 19 20

# Single prover
./bench_vs_plonky3/run.sh --lambda-only
./bench_vs_plonky3/run.sh --p3-only

# Scalar mode on both sides (x86_64 only — disables AVX2/AVX-512)
./bench_vs_plonky3/run.sh --scalar

# Write machine-readable artifacts
./bench_vs_plonky3/run.sh --report-dir /tmp/p3_report --no-color
```

### Flags

| Flag | Default | Effect |
|---|---|---|
| `--log-rows K [K ...]` | `19` | One or more power-of-2 row counts. |
| `--num-sequences N` | `16` | Number of Fibonacci sequences (columns = `2 × N`). |
| `--runs N` | `3` | Runs per `(size, prover)`; median is reported. |
| `--lambda-only` / `--p3-only` | both | Restrict to a single prover. |
| `--report-dir DIR` | — | Write TSV + metrics + raw stdouts. |
| `--scalar` | off | Pin `RUSTFLAGS="-C target-feature=-avx2,-avx512f"` so Goldilocks (and most of Keccak) run scalar on both sides. x86_64 only; on other archs the flag is ignored with a warning. Residual SSE2 on `p3-keccak` remains (~7% of total prove time). |
| `--no-color` | off | Disable ANSI colors. |
| `-h` / `--help` | — | Print usage. |

## Output

Stdout (without `--report-dir`):

```
=== STARK prove benchmark: Lambda vs Plonky3 ===
  log-rows:       19
  num-sequences:  16  (columns = 32)
  runs/size:      3  (median reported)
  p3 extension:   degree 3 trinomial x^3-x-1 (upstream Plonky3)
  scalar mode:    on  (arch=x86_64, RUSTFLAGS="-C target-feature=-avx2,-avx512f")

[build] prove_bench
--- log-rows=19  (rows = 524288) ---
  [lambda] median 2.444s from 3 runs: 2.444,2.279,2.830
  [p3]     median 0.988s from 3 runs: 0.981,0.993,0.988

=== Summary ===
  log-rows   rows              Lambda (s)          P3 (s)        L/P3
  --------   ----              ----------          ------        ----
  19         524288                2.444s          0.988s      2.474x  (P3 faster)

Timing window: single-shot end-to-end prove.
Ratio = Lambda / P3. ratio > 1 → P3 faster (Lambda took ratio× longer); ratio < 1 → Lambda faster.
```

With `--report-dir DIR` the script writes:

- `results.tsv` — tab-separated raw data (`log_rows, rows, lambda_median_s,
  p3_median_s, ratio_lambda_over_p3, runs`).
- `metrics.txt` — key=value pairs with the config used (arch, scalar flag,
  extension degree, blowup, queries, runs, rustflags) and the per-series
  values slash-joined (so post-processing scripts can split easily).
- `raw/` — per-invocation stdouts (`{prover}_log{K}_run{i}.stdout`).

No markdown file is generated — the TSV is the single source of truth for
downstream tooling.

## Nightly

A GitHub Actions workflow (`.github/workflows/bench-vs-p3-nightly.yml`) runs
daily at 07:30 UTC (04:30 Buenos Aires, after the SP1 nightly completes) on
the self-hosted `bench` runner. It executes:

```bash
bash ./bench_vs_plonky3/run.sh \
  --log-rows 19 \
  --num-sequences 16 \
  --runs 3 \
  --scalar \
  --report-dir bench_vs_p3_artifacts \
  --no-color
```

The `bench_vs_p3_artifacts/` directory is uploaded as an artifact named
`bench-vs-p3-nightly-<run_number>-<sha>` with 90-day retention.

## Breakdown (per-phase timing) for manual analysis

The nightly only reports wall-clock totals. When you need to see *where* the
time goes (constraint eval vs FFT vs FRI vs Merkle vs queries on the Lambda
side, and the per-span breakdown on the Plonky3 side), run the
`instruments_breakdown` test:

```bash
# x86_64 (server), Goldilocks scalar:
RUSTFLAGS="-C target-feature=-avx2,-avx512f" \
cargo test -p bench-vs-plonky3 --features instruments --release -- \
  instruments_breakdown --nocapture
```

- `--features instruments` activates `stark/instruments` — without it, the
  per-phase timers are no-ops and the Lambda breakdown prints zeros.
- `--release` is mandatory (debug numbers are meaningless).
- `--nocapture` is required to see the output (`cargo test` swallows stdout
  otherwise).
- The test hardcodes `num_sequences = 16`, `rows = 1 << 19` (524 288), same
  shape as the nightly, so the breakdown maps onto the nightly numbers.
- Output is split in two sections:
  - **Lambda**: explicit per-phase totals (Pre-pass / R1 Main commits / R1 Aux
    build+commit / Rounds 2-4) plus sub-ops (Main LDE, Main Merkle, constraint
    eval, decompose+extend, composition Merkle, OOD, deep comp, deep extend,
    FRI commit, queries+open).
  - **Plonky3**: every `tracing` span emitted at DEBUG during
    `p3_uni_stark::prove`, sorted by wall-clock descending, filtered ≥ 0.1 ms.
    Spans nest (e.g. `prove ⊃ compute_quotient_values`), so Σspans > total is
    expected and not a bug. `(unaccounted)` can be negative from nesting.

The nightly does **not** activate this path — it would add ~1 % overhead and
pollute the historical wall-clock numbers.

## Notes on fairness

- **Extension field**: Lambda uses `Degree3GoldilocksExtensionField` with
  irreducible `x^3 - 2` (binomial); Plonky3 upstream uses
  `CubicTrinomialExtensionField` with `x^3 - x - 1` (trinomial). Same
  degree, same conjectured soundness; Fp3 multiplication differs by a
  few extra additions on the trinomial side.
- **Parallelism**: both provers are multi-threaded by default. Lambda pulls
  rayon via `stark/parallel`; Plonky3 pulls rayon via
  `p3-uni-stark` / `p3-dft` (hardcoded `features = ["parallel"]`, always on).
- **SIMD**: without `--scalar`, each side uses whatever target-features the
  compiler decides from the host CPU. `--scalar` (x86_64 only) disables AVX2
  and AVX-512 so Goldilocks arithmetic is scalar on both sides. `p3-keccak`'s
  SSE2 path on x86 is not disabled.
- **Queries / grinding**: same `blowup=2`, `queries=219`, `grinding=0` on both
  sides. Security models differ (Lambda: Johnson-bound, ~108 bits proven;
  P3: conjectured, 219 queries × 1 bit = 219 bits, capped at 192 by the
  cubic extension field) — the compute work is equivalent, the claimed
  soundness is not.
