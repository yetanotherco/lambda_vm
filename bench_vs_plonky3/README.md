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
# Default: log-rows=19, num-sequences=16, runs=10, cubic extension, no scalar
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
| `--runs N` | `10` | Runs per `(size, prover)`; median + CV are reported. |
| `--lambda-only` / `--p3-only` | both | Restrict to a single prover. |
| `--report-dir DIR` | — | Write TSV + metrics + raw stdouts + raw audits. |
| `--scalar` | off | Pin `RUSTFLAGS="-C target-feature=-avx2,-avx512f"` so Goldilocks field arithmetic runs scalar on both sides. x86_64 only; on other archs the flag is ignored with a warning. The MMCS is already scalar regardless of this flag (see [P3 config: scalar MMCS](#p3-config-scalar-mmcs)). |
| `--no-color` | off | Disable ANSI colors. |
| `-h` / `--help` | — | Print usage. |

## Output

Stdout (without `--report-dir`):

```
=== STARK prove benchmark: Lambda vs Plonky3 ===
  log-rows:       19
  num-sequences:  16  (columns = 32)
  runs/size:      10  (median + CV reported)
  p3 extension:   upstream CubicTrinomialExtensionField (x^3 - x - 1)
  p3 mmcs:        scalar Keccak256 (val_packing_width=1, hash_lanes=1)
  proof params:   blowup=2, queries=219, grinding=0
  scalar mode:    on  (arch=x86_64, RUSTFLAGS="-C target-feature=-avx2,-avx512f")

[build] prove_bench
--- log-rows=19  (rows = 524288) ---
  [lambda] prove median 0.574s (CV 3.07%), verify 0.024s, proof 4116000 B, rss 805000 KB
  [p3]     prove median 0.324s (CV 2.85%), verify 0.019s, proof 1987000 B, rss 627000 KB

=== Summary ===
  log-rows   rows              Lambda (s)      L CV%          P3 (s)     P3 CV%        L/P3
  --------   ----              ----------      -----          ------     ------        ----
  19         524288              0.574s         3.07%        0.324s       2.85%      1.770x  (P3 faster)

Timing window: prove only for the ratio. Verify, proof size, RSS and throughput are reported separately.
```

With `--report-dir DIR` the script writes:

- `results.tsv` — tab-separated, one row per `log_rows` size with 14 columns:
  `log_rows, rows, lambda_prove_median_s, lambda_prove_cv_pct,
  lambda_verify_median_s, lambda_proof_size_bytes_median,
  lambda_peak_rss_kb_median, p3_prove_median_s, p3_prove_cv_pct,
  p3_verify_median_s, p3_proof_size_bytes_median, p3_peak_rss_kb_median,
  ratio_lambda_over_p3, runs`.
- `raw_metrics.tsv` — one row per `(prover, log_rows, run)` with all
  `METRICS` fields parsed out.
- `raw_audits.tsv` — one row per `(prover, log_rows, run)` with the AUDIT
  line emitted by `prove_bench` before each prove call. Lets you confirm in
  retrospect that `val_packing_width=1`, `hash_lanes=1`,
  `base_transition_constraints=2×num_sequences`, etc. Don't trust a number
  without skimming this file.
- `metrics.txt` — key=value pairs with the config used (arch, scalar flag,
  extension, mmcs choice, blowup, queries, runs, rustflags) and the
  per-series values slash-joined (so post-processing scripts can split easily).
- `raw/` — per-invocation stdouts (`{prover}_log{K}_run{i}.stdout`).

No markdown file is generated — the TSV is the single source of truth for
downstream tooling.

## Nightly

The Lambda-vs-Plonky3 bench is part of the shared
`.github/workflows/bench-vs-nightly.yml` workflow, which runs daily at
06:00 UTC (03:00 Buenos Aires) on the self-hosted `bench` runner. The P3
step executes after the Lambda-vs-SP1 and ethrex empty-block steps:

```bash
bash ./bench_vs_plonky3/run.sh \
  --log-rows 21 \
  --num-sequences 16 \
  --runs 10 \
  --scalar \
  --report-dir bench_vs_artifacts/p3 \
  --no-color
```

A `cargo update -p p3-*` runs before this step so the bench tracks the
latest upstream Plonky3 `main`. The full `bench_vs_artifacts/` directory
(SP1 + ethrex + P3 outputs) is uploaded as one artifact named
`bench-vs-nightly-<run_number>-<sha>` with 90-day retention. A "Lambda
VM vs Plonky3" section is appended to the same Slack post that publishes
the SP1 and ethrex results.

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

## P3 config: scalar MMCS

`plonky3_config.rs` sets up the P3 stark config with a deliberately
**non-production** MMCS:

```rust
type ByteHash = Keccak256Hash;                               // tiny_keccak scalar
type FieldHash = SerializingHasher<ByteHash>;
type MyCompress = CompressionFunctionFromHasher<ByteHash, 2, 32>;
pub type ValMmcs = MerkleTreeMmcs<Val, u8, FieldHash, MyCompress, 2, 32>;
```

The Plonky3 default for Goldilocks MMCS uses `PaddingFreeSponge<KeccakF, 25,
17, 4>` with leaves `[Val; VECTOR_LEN]` and digests `[u64; VECTOR_LEN]`,
where `VECTOR_LEN` is set at compile-time per arch: NEON=2, AVX-512=8,
AVX2=4, SSE2=2, fallback=1. That gives Plonky3 a free `N×` Keccak speedup
on every Merkle node — which Lambda's `sha3::Keccak256` cannot exploit
because the Lambda MMCS hashes a single input at a time.

The scalar config here makes both sides hash one input per Keccak call.
Both still use the **same Keccak-f[1600] permutation** (capacity 512, rate
1088, 256-bit output, Keccak-original 0x01 padding); the only thing
removed is data-parallel lanes on the P3 side. Consequence: the ratio
published by this bench is **apples-to-apples scalar**, not "Plonky3 as
shipped in production." If you want the production-realistic P3 number,
swap the MMCS back to the vector-lane variant from upstream's examples.

On aarch64 with `feature="asm"` enabled in `crypto/crypto`, Lambda's
`sha3::Keccak256` uses ARMv8 SHA3 intrinsics, which speeds up *one* Keccak
call (no data parallelism). `tiny_keccak`'s `Keccak256Hash` on P3 is pure
Rust and gets no such acceleration. On x86_64 server, neither side has
that path, so the comparison is cleanest there.

## Notes on fairness

- **Extension field**: Plonky3 runs upstream `CubicTrinomialExtensionField`
  over Goldilocks (`x^3 - x - 1`); Lambda runs `Degree3GoldilocksExtensionField`
  (`x^3 - 2`). Both are degree-3 irreducible extensions of `GF(p)` with the
  same field size and the same soundness. Cell-by-cell trace equivalence is
  asserted by `lambda_pair_trace_matches_plonky3_trace`.
- **Parallelism**: both provers are multi-threaded by default. Lambda pulls
  rayon via `stark/parallel`; Plonky3 pulls rayon via `p3-uni-stark` /
  `p3-dft` (hardcoded `features = ["parallel"]`, always on).
- **SIMD**: the MMCS Keccak is scalar on both sides (see above). For
  Goldilocks field arithmetic, without `--scalar` each side uses whatever
  target-features the compiler decides from the host CPU. `--scalar`
  (x86_64 only) disables AVX2 / AVX-512.
- **AIR base-field path**: the Lambda AIR overrides
  `num_base_transition_constraints` and implements `evaluate_prover` so its
  Fibonacci transition constraints are evaluated in the base field (F×E,
  ≈3 muls/term) instead of the default extension path (E×E, ≈9 muls/term).
  This matches what the production Lambda STARK does for all
  domain-constraint AIRs.
- **Queries / grinding**: same `blowup=2`, `queries=219`, `grinding=0` on both
  sides. Security models differ (Lambda: Johnson-bound, ~108 bits proven;
  P3: conjectured, 219 queries × 1 bit = 219 bits, capped at 192 by the
  cubic extension field) — the compute work is equivalent, the claimed
  soundness is not.
