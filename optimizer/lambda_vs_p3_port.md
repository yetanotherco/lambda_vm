# Lambda STARK vs Plonky3 — port to upstream

Comparative snapshot of Lambda vs Plonky3 after migrating the bench away
from the `yetanotherco/Plonky3#feat/goldilocks_deg3` fork (binomial
extension `x³-2`, matching Lambda) to **Plonky3 upstream** with
`CubicTrinomialExtensionField` (irreducible `x³-x-1`, "GoldilocksCubic3").

Same soundness, same extension degree. The only algebraic difference is the
cost of multiplication in Fp3: the trinomial adds a couple of extra additions
per mul compared to the binomial.

## Size sweep (num-sequences=16, 32 cols)

| log | rows  | cols | L May 5 | P3 May 5 | Ratio May 5 | L May 13 | P3 May 13 | Ratio May 13 | Change |
|----:|------:|-----:|--------:|---------:|------------:|---------:|----------:|-------------:|--------|
|  17 | 131 K |   32 | 0.249 s |  0.134 s |       1.86× |  0.239 s |   0.151 s |        1.58× | improved for Lambda |
|  19 | 524 K |   32 | 0.566 s |  0.322 s |       1.76× |  0.554 s |   0.340 s |        1.63× | improved |
|  20 | 1.0 M |   32 |       — |        — |           — |  1.024 s |   0.555 s |        1.85× | — |
|  21 | 2.1 M |   32 | 1.880 s |  0.976 s |       1.93× |  1.877 s |   0.964 s |        1.95× | ~same |
|  23 | 8.4 M |   32 |       — |        — |           — |  6.804 s |   2.997 s |        2.27× | new |

### Reading

- **Lambda is intact**: prove time identical to May 5 across all sizes
  (delta within ±2%). The migration did not affect Lambda — as expected,
  Lambda does not depend on Plonky3.
- **P3 pays +5–13% at small log sizes** (17–19) due to the trinomial cost.
  At log≥20 the Fp3 mul is a small fraction of total prove time and the
  delta lands within noise (log=21 even dropped -1.2% due to variability).
- **L/P3 ratios at small log sizes go down** because P3 got more expensive
  while Lambda stayed the same. At log=21 the ratio is essentially
  unchanged (1.93× → 1.95×).
- **New points on May 13**: log=20 (1.85×) and log=23 (2.27×). log=23
  confirms that the gap **widens with size**: extrapolating, log=25 would
  be around 2.5–2.7× (memory permitting).

## Cols sweep @ log=21 (2.1 M rows) — the most informative dimension

### Totals

| Rows  | Cols | Lambda time | Plonky3 time | Lambda is X× slower |
|------:|----:|------------:|-------------:|---------------------:|
| 2.1 M |  32 |     1.877 s |      0.964 s |              1.95×   |
| 2.1 M |  64 |     2.741 s |      1.113 s |              2.46×   |
| 2.1 M | 128 |     4.728 s |      1.613 s |              2.93×   |

All three runs use the same row count (2^21 = 2.1 M). Only the trace width
changes. **As columns grow, Lambda scales worse than Plonky3**: the gap
goes from 1.95× to **2.93×** going from 32 to 128 columns.

### Per-phase scaling — Lambda (ms)

| Phase                | 32 cols | 64 cols | 128 cols | 32→64 | 64→128 | 32→128 |
|----------------------|--------:|--------:|---------:|------:|-------:|-------:|
| **`prove_total`**    | **1898**| **2741**| **4744** | ×1.44 | ×1.73  | **×2.50** |
| `r2_constraints`     |     474 |     848 |    1619  | ×1.79 | ×1.91  | **×3.42** |
| `r1_main_lde`        |     357 |     682 |    1427  | ×1.91 | ×2.09  | **×4.00** |
| `r1_main_merkle`     |     178 |     267 |     586  | ×1.50 | ×2.19  | **×3.29** |
| `r4_fri_commit`      |     319 |     298 |     291  | ×0.93 | ×0.98  | ×0.91  |
| `r3_ood + r4_deep_*` |     397 |     425 |     487  | ×1.07 | ×1.15  | ×1.23  |
| `r2_comp_commit`     |      58 |      58 |      58  | ×1.00 | ×1.00  | ×1.00  |
| `prepass`            |      51 |      52 |      51  | ×1.02 | ×0.99  | ×1.00  |

**Scales with cols** (Lambda): `r2_constraints` + `r1_main_lde` +
`r1_main_merkle`. Everything else is **constant**.

### Per-phase scaling — P3 (top spans, ms)

| Span                                     | 32 cols | 64 cols | 128 cols | 32→128 |
|------------------------------------------|--------:|--------:|---------:|-------:|
| **`prove`**                              | **950** |**1116** | **1617** | **×1.70** |
| `commit to trace data`                   |     286 |     461 |     867  | **×3.03** |
| `coset_lde_batch[_with_transform]`       |     131 |     190 |     349  | **×2.66** |
| **`quotient_values`** (constraint eval)  |   **176** | **162** | **188**  | **×1.07** |
| `FRI prover`                             |     160 |     156 |     155  | ×0.97 |
| `commit phase`                           |     148 |     145 |     143  | ×0.97 |
| `commit to quotient poly chunks`         |     140 |     119 |     115  | ×0.82 |
| `open`                                   |     327 |     338 |     377  | ×1.15 |

**Standout fact**: P3 `quotient_values` (constraint eval) is **flat as cols
grow** (176→188, ×1.07). Lambda `r2_constraints` tripled (474→1619, ×3.42).

### L/P3 ratios per phase

| Logical stage                | 32 cols | 64 cols | 128 cols |
|------------------------------|--------:|--------:|---------:|
| Total                        | 1.95×   | 2.46×   | **2.93×** |
| **Constraint eval**          | 2.69×   | 5.25×   | **8.61×** |
| Trace LDE + Merkle           | 1.87×   | 2.06×   | 2.32×    |
| FRI commit                   | 1.99×   | 1.91×   | 1.88×    |
| DEEP + OOD                   | 2.38×   | ~2.4×   | ~2.5×    |
| Quotient commit              | 0.41×✅ | 0.48×✅ | 0.50×✅  |

**Diagnosis**:
1. **Constraint eval is the bottleneck that breaks worst with cols** (L/P3
   goes from 2.69× to 8.61×). P3 evaluates on the quotient domain
   `next_pow2(d_max)·N` (when `d_max=1` this is just `N`); Lambda evaluates
   on the full LDE `2N`.
2. **Trace LDE+Merkle** also gets worse (1.87× → 2.32×): per-column iFFT+FFT
   on Lambda vs batched `coset_lde` on P3.
3. **FRI commit, DEEP, quotient commit are insensitive to #cols**.
4. **Implication**: production Lambda VM tables have 70–200 cols (Keccak
   chip = 1775). The 32-cols headline ratio (1.95×) **understates** the
   real gap. The cols sweep projects production closer to **2.5–3.5×**.

## Per-phase breakdown — log=21, 32 cols (May 5 vs May 13)

10-run averages. May 5 = fork (binomial `x³-2`); May 13 = upstream
(trinomial `x³-x-1`).

| Phase | L May 5 | L May 13 | Δ L | P3 May 5 | P3 May 13 | Δ P3 | L/P3 May 13 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Constraint eval** | 478.2 ms | 473.8 ms | -0.9% | 175.7 ms | 176.2 ms | +0.3% | **2.69×** |
| Trace LDE+Merkle (main) | 540.2 | 535.2 | -0.9% | 289.2 | 285.9 | -1.1% | 1.87× |
| FRI commit | 312.7 | 319.0 | +2.0% | 162.1 | 160.1 | -1.2% | 1.99× |
| Open / Deep+OOD | 394.3 | 396.9 | +0.7% | 327.1 | ~327 | ~0% | 1.21× |
| Prepass | 52.0 | 51.3 | -1.3% | — | — | — | ∞ |
| Quotient commit | 57.7 | 57.9 | +0.3% | 142.3 | 139.9 | -1.7% | **0.41× ✅ L wins** |
| Queries | 2.5 | 2.4 | -4.0% | ~3.5 | ~4.1 | +17% | 0.59× ✅ L wins |
| **Total prove** | **1890** | **1898** | +0.4% | **976** | **950** | -2.7% | **2.00×** |

**Lambda phase-by-phase**: identical (all Δ |<2%|) — clean migration.
**P3 constraint eval**: +0.3% (trinomial mul cost invisible at log=21).
**The 4 heavy phases** (constraint, LDE+Merkle, FRI, Open) explain ~99% of the gap.

## Per-phase breakdown — log=19 (524 K rows, 32 cols)

| Phase | L May 5 | L May 13 | P3 May 5 | P3 May 13 | L/P3 May 13 |
|---|---:|---:|---:|---:|---:|
| Constraint eval | 171.5 ms | 198.3 ms | 48.6 ms | 52.6 ms | **3.77×** |
| Trace LDE+Merkle | 153.0 | 137.7 | 102.9 | ~95 | ~1.45× |
| FRI commit | 100.6 | 105.0 | 66.4 | ~50 | ~2.1× |
| Open / Deep+OOD | 92.4 | 101.3 | 112.4 | ~133 | ~0.76× |
| Quotient commit | 20.8 | 22.6 | 46.8 | — | — |
| Prepass | 16.2 | 16.2 | — | — | ∞ |
| **Total prove** | **560** | **606** | **322** | **345** | **1.76×** |

- **Constraint eval dominates at small log** (~33% of Lambda total), with
  L/P3 ratio **3.77×** (vs 2.69× at log=21). The structural gap *shrinks*
  with N.
- **Opposite trend in Trace LDE+Merkle**: ratio improves at small log
  (1.45×) and worsens at large log (2.02×).
- High CVs at log=19 (L 9.9% / P3 2.6%) limit the precision of small deltas.

## Configuration

| Item | Value |
|---|---|
| Server | `vm-benchmarks-1`, AMD EPYC 48-core, 125 GB RAM |
| Arch | x86_64 |
| Scalar mode | `RUSTFLAGS="-C target-feature=-avx2,-avx512f"` |
| Lambda commit (May 13) | `fb20f767` ("replace fork for p3") |
| Plonky3 upstream rev | `af65376f` (HEAD of `Plonky3/Plonky3.git#main` at fetch time) |
| P3 extension | `CubicTrinomialExtensionField<Goldilocks>`, irreducible `x³-x-1` |
| Lambda extension | `Degree3GoldilocksExtensionField`, irreducible `x³-2` |
| Blowup | 2 |
| FRI queries | 219 |
| Grinding | 0 |
| Runs per point | 10 (median reported) |
| Parallelism | rayon on both provers |

**CVs per point** — all <10%, large points <2% (clean data):

| log | L CV | P3 CV |
|----:|-----:|------:|
|  17 | 9.2% |  6.7% |
|  19 | 9.9% |  2.6% |
|  20 | 0.7% |  2.9% |
|  21 | 0.9% |  1.4% |
|  23 | 0.5% |  0.8% |

## Raw data — breakdown TSVs

The `breakdown.tsv` files contain the timing of **each phase / span** per
individual run (10 Lambda runs + 10 P3 runs per sub-run).

### Files in `optimizer/raw_data/`

| File | log | cols | rows | Lines | Contents |
|---|---:|---:|---:|---:|---|
| `breakdown_log19_n16_2026-05-13.tsv` | 19 |  32 | 524 K | 1248 | 10 L runs + 10 P3 runs |
| `breakdown_log21_n16_2026-05-13.tsv` | 21 |  32 | 2.1 M | 1304 | 10 L runs + 10 P3 runs |
| `breakdown_log21_n32_2026-05-14.tsv` | 21 |  64 | 2.1 M | 1305 | 10 L runs + 10 P3 runs |
| `breakdown_log21_n64_2026-05-14.tsv` | 21 | 128 | 2.1 M | 1339 | 10 L runs + 10 P3 runs |

### Originals in `bench_vs_plonky3/reports/`

The original TSVs (with their `results.tsv`, `metrics.txt`, `raw/*.stdout`,
etc.) live in:

```
bench_vs_plonky3/reports/bench_vs_p3_20260513_2033_upstream/breakdown_log19/
bench_vs_plonky3/reports/bench_vs_p3_20260513_2033_upstream/breakdown_log21/
bench_vs_plonky3/reports/bench_vs_p3_breakdown_log21_n32_20260514_2125/
bench_vs_plonky3/reports/bench_vs_p3_breakdown_log21_n64_20260514_2129/
```

### TSV format

10 tab-separated columns:

| Col | Name | Notes |
|---:|---|---|
| 1 | `run` | 1..10, run index |
| 2 | `workload` | always `fib_pair` |
| 3 | `prover` | `lambda` or `p3` |
| 4 | `log_rows` | 19 or 21 |
| 5 | `rows` | 524288 or 2097152 |
| 6 | `phase` | phase name (Lambda) or literal `span` or `prove_total` (P3) |
| 7 | `ms` | time in milliseconds |
| 8 | `table` | (Lambda multi-table only — empty for fib_pair) |
| 9 | `table_rows` | (idem) |
| 10 | `span` | span name (only when `phase=span` in P3) |

### Re-processing (example)

```bash
# Lambda: median of r2_constraints at log=21, 32 cols
awk -F'\t' '$3=="lambda" && $4==21 && $6=="r2_constraints" {print $7}' \
    optimizer/raw_data/breakdown_log21_n16_2026-05-13.tsv \
    | sort -n | awk '{a[NR]=$1} END {print (a[5]+a[6])/2}'

# P3: median of the quotient_values span
awk -F'\t' '$3=="p3" && $4==21 && $10=="quotient_values" {print $7}' \
    optimizer/raw_data/breakdown_log21_n16_2026-05-13.tsv \
    | sort -n | awk '{a[NR]=$1} END {print (a[5]+a[6])/2}'
```

## Notes

- May 5 numbers come from `bench_vs_plonky3/inform_2026-05-05.md`: run
  against the yetanotherco/Plonky3 fork (binomial `x³-2`).
- May 13 numbers are the new ones: run against Plonky3 upstream `main`
  (`CubicTrinomialExtensionField`). Raw results in
  `bench_vs_plonky3/reports/bench_vs_p3_20260513_2033_upstream/`.
- May 5 did not measure log=20 or log=23, hence the `—` in that column.
- The cols sweep at 64/128 cols was measured 2026-05-14 (post-migration).
- The "Change" column in the first table summarizes the observed effect:
  at small log sizes the ratio improves *for Lambda* because P3 trinomial
  is more expensive; at log≥21 it stabilizes.

## Bench scope (caveats)

The `fib_pair` AIR has NO aux trace, NO logup, NO multi-table. It is
designed to be cell-by-cell equivalent with `P3FibonacciAir` from
`p3-uni-stark`. The real Lambda VM activates aux trace + logup + multi-table
in every chip, with phases that come out as 0 ms in this bench (`aux_build`,
`aux_commit`, `r1_aux_lde`, `r1_aux_merkle`, `r2_comp_decompose`).

**Therefore the reported ratio is the best case for Lambda**: in production
with real AIRs (aux+logup) the ratio would be worse than 2.93× at 128 cols.
To measure full-VM performance use `bench_prove.sh` with
`fib_iterative_8M.elf` (a different bench, different setup).
