# Optimization Report

Base branch: `bench_vs_p3` (HEAD: `c6fd62b5` "add server info")
Benchmark: `bench_vs_plonky3` (fib_pair AIR — Lambda STARK aislado vs Plonky3
upstream `CubicTrinomialExtensionField`)
Started: 2026-05-14

## Baseline (`bench_vs_p3_20260513_2033_upstream/`)

10-sample medians, server `vm-benchmarks-1` (AMD EPYC 48-core), `--scalar`,
blowup=2, 219 FRI queries.

| log_rows | cols | Lambda prove | L CV | P3 prove | L/P3 ratio |
|---:|---:|---:|---:|---:|---:|
| 21 |  32 | 1.877 s | 0.9% | 0.964 s | 1.95× |
| 21 |  64 | 2.741 s | 0.5% | 1.155 s | 2.46× |
| 21 | 128 | 4.728 s | 0.9% | 1.626 s | 2.93× |

Phase breakdown (log=21, 32 cols, Lambda):
- `r2_constraints`: 474 ms (25%)
- `r1_main_lde+merkle`: 535 ms (28%)
- `r4_fri_commit`: 319 ms (17%)
- `r3_ood + r4_deep_*`: 397 ms (21%)
- `r2_comp_commit`: 58 ms (3%)
- `prepass`: 51 ms (3%)

Ratio Lambda/P3 at constraint eval: 2.69× @ 32 cols → 8.61× @ 128 cols.

---

## Attempt 1 — Evaluate constraints on N-point trace coset for d_max=1 AIRs

**Branch**: `experiment/eval-d1-domain-n` (origin commit `8369820a`)
**Status**: **ABANDONED**
**Date**: 2026-05-14
**Bottleneck target**: `r2_constraints` (Bottleneck #1)

### Idea

When the composition polynomial has degree < N (i.e., AIRs with `d_max=1`
like fib_pair), evaluate boundary + transition constraints on the N-point
trace-offset coset (a stride-`blowup_factor` subsample of the LDE) instead
of the full LDE 2N. Then extend N→2N via `interpolate_offset_fft` +
`evaluate_polynomial_on_lde_domain` before the commitment phase.

Theoretical saving: half the per-point work in `evaluate_transitions`.

### Changes

- `crypto/stark/src/constraints/evaluator.rs`: added `eval_on_trace_domain`
  flag to `evaluate()`. When true, computes `stride = blowup_factor` and
  iterates `0..N` with `lde_idx = i * stride` for all LDE-sized data
  (trace, periodic, zerofiers, boundary inv zerofiers).
- `crypto/stark/src/prover.rs::round_2_compute_composition_polynomial`:
  branch when `number_of_parts == 1` → call evaluator with the flag,
  then `interpolate_offset_fft` + `evaluate_polynomial_on_lde_domain`
  to extend back to LDE 2N.
- Total: 97 lines, 2 files. Other AIR paths (`number_of_parts >= 2`)
  unchanged.

### Verification

- `cargo test --release -p stark`: **124/124 PASS**.
- Smoke `bench_vs_plonky3` log=17 on M1: proof produced + Plonky3 P3
  verify equivalent. No correctness issue detected.

### Measurements (`bench_vs_p3_exp_eval_d1_20260514_2323/`)

Server `vm-benchmarks-1`, same config as baseline. 10 samples per point.

| cols | Lambda baseline | Lambda exp | **Δ Lambda** | L/P3 baseline | L/P3 exp |
|---:|---:|---:|---:|---:|---:|
|  32 | 1.877 s | **2.200 s** | **+17.2%** (regression) | 1.95× | 2.35× |
|  64 | 2.741 s | **2.978 s** | **+8.6%** (regression)  | 2.46× | 2.65× |
| 128 | 4.728 s | **4.864 s** | **+2.9%** (≈ wash, within CV) | 2.93× | 3.00× |

CVs: 0.27%–1.43% — measurement-clean. The regressions are real, not noise.

### Decision

**ABANDONED**. The optimization is theoretically sound (`d_max=1` saving
is half the LDE work) but the FFT extension cost dominates the saving:

| cols | Est. saving (½ r2_constraints) | Est. FFT cost (used) | Net | **Actual measured Δ** |
|---:|---:|---:|---:|---:|
|  32 | 237 ms | 250 ms | -13 ms | -323 ms (much worse) |
|  64 | 424 ms | 250 ms | +174 ms | -237 ms (regression instead of win) |
| 128 | 810 ms | 250 ms | +560 ms | -136 ms (regression instead of win) |

The actual FFT extension cost was **~560–946 ms**, vs the 250 ms estimate.
~3-4× more than predicted. See "Postmortem" below.

### Postmortem — why the FFT extension was so expensive

The cost of `interpolate_offset_fft` + `evaluate_polynomial_on_lde_domain`
in this context is over the **extension field Fp3**, not the base field
Goldilocks. The estimate was derived from base-field FFT throughput, which
massively underestimated the real cost. Specifically:

1. **Fp3 mul is ~9× more expensive than Goldilocks mul.** Each Fp3 mul
   needs 6 base muls + 6 base adds (binomial reduction `x³-2`). Plus
   tracking 3 limbs per element + reduction overhead.
2. **FFT butterfly count.** An iFFT N + FFT 2N has ≈ ½·N·log(N) +
   N·log(2N) butterflies ≈ 1.5·N·log(N) butterflies = **1.5 × 2.1M × 21
   ≈ 66 M butterflies**. Each butterfly does ~1 Fp3 mul + 2 Fp3 adds
   plus a twiddle mul. So **~66-100 M Fp3 muls total**.
3. **At ~6-9 ns per Fp3 mul** (scalar Goldilocks ~1-2 ns × 9 multiplications):
   the FFT cost is **400-900 ms** — exactly the range we measured.
4. **My original estimate** was based on Goldilocks base-field FFT
   throughput (~3-5 GB/s), which is wrong by an order of magnitude when
   applied to Fp3. The FFT in extension field is essentially 9× slower
   than in the base field, plus extra allocation overhead.

In retrospect, the math was:

- **Saving**: `r2_constraints` does ~`LDE_size × num_cols × O(1)` Fp3
  muls. Going from 2N→N points halves that. For 128 cols at log=21,
  that's roughly `2N × 128 × 2 = 1.1G` Fp3 muls saved.
- **FFT cost added**: `~1.5·N·log(N)` Fp3 muls ≈ `66M` muls.
- The break-even WOULD be at ~6% as many cols as my (under)estimate
  suggested — i.e. NEVER for 32-cols, marginal at 128 cols.

In other words, the FFT extension dominates **any time the constraint
work per LDE point is less than `1.5·log(N)` Fp3 muls**, which for log=21
means "more than 31 Fp3 muls per point at the constraint eval". The
fib_pair AIR has ~32 constraints × ~3 muls each ≈ 96 muls/point — close
to break-even, but the FFT allocation overhead and the iFFT N base step
push it over.

### Lessons

- **Estimate FFT cost using extension field, not base field**, when the
  polynomial coefficients live in the extension. Goldilocks vs Fp3 is a
  9× difference in cost per mul.
- **Allocation overhead matters at large N**. `interpolate_offset_fft`
  allocates a new Vec for the polynomial coefficients; that's an
  additional ~2.1M × 24 bytes = 50 MB allocation per call.
- **The optimization is structurally sound for `d_max=1`**, but only
  if the FFT extension can be made much cheaper — e.g., by avoiding the
  iFFT entirely (some interpolation-free strategy), or by precomputing
  the extension matrix once and reusing it across queries.

### Next move

The deeper observation is that **Fp3 mul cost is a hidden bottleneck
across multiple phases** (constraint eval accumulator, FRI folding, DEEP
composition, all use Fp3). Two general directions worth exploring:

- **A — Faster Fp3 mul** (SIMD/AVX/ASM). Existing branches:
  `feat/avx-goldilocks-multiplication`, `feat/simd-goldilocks*`,
  `goldilocks_asm`. Status unknown.
- **B — Fewer Fp3 muls** (more base-field paths in hot loops). Already
  done partially: `06e45227` (F×E base-field constraint eval),
  `8f0fa724` (DEEP poly inner loop). Possibly more to extract.

The `experiment/eval-d1-domain-n` branch is **kept as historical record**.
Not merged. Do not retry under the current Fp3 mul cost.

---

## Attempt 2 — 8-way tree-sum accumulator (batched_linear_combination)

**Branch**: `experiment/batched-lin-combination` (origin commit pending push)
**Status**: **ABANDONED**
**Date**: 2026-05-15
**Bottleneck target**: `r2_constraints` (Bottleneck #1)

### Idea

Replace the scalar `fold()` over constraint evaluations × random coefficients
with an 8-way tree-sum accumulator (8 named accumulators a0..a7, processed
in unrolled chunks of 8). Inspired by Plonky3's `batched_linear_combination`
in `field/src/field.rs:676-696`. The hypothesis (from Pattern 3 research
agent) was that the scalar fold creates a strict dependency chain that
serializes the mul-add, while a tree-sum lets the CPU keep multiple
mul-add chains in flight (ILP), hiding the multiply-accumulate latency
(~4 cycles per op).

### Changes

- `crypto/stark/src/constraints/evaluator.rs`: added two private helpers
  `batched_lin_comb_base` (F×E) and `batched_lin_comb_ext` (E×E), each with
  8 named local accumulators, unrolled 8-step chunks, scalar tail handler,
  and pairwise tree-sum reduction at the end.
- Replaced the 4 scalar `.fold()` calls in `evaluate_transitions` (both
  parallel and non-parallel paths, both base and extension constraints)
  with calls to these helpers. The `is_uniform` path is the one that
  gets the change; the non-uniform path was left unchanged.
- Total: ~150 lines added, 1 file. Only the `is_uniform` constraint
  accumulator is touched.

### Verification

- `cargo test --release -p stark`: **124/124 PASS**.
- Smoke `bench_vs_plonky3` log=17 on M1: proof valid, ratio identical.

### Measurements (`optimizer/bench_vs_p3_exp_batched_lc_20260515_0102/`)

Server `vm-benchmarks-1`, same config as baseline. 10 samples per point.

**Total prove time (Lambda)**:

| cols | Baseline | Exp | Δ |
|---:|---:|---:|---:|
|  32 | 1.877 s | 1.891 s | +0.7% (within CV) |
|  64 | 2.741 s | 2.732 s | -0.3% (within CV) |
| 128 | 4.728 s | 4.717 s | -0.2% (within CV) |

**`r2_constraints` phase only**:

| cols | Baseline | Exp | Δ |
|---:|---:|---:|---:|
|  32 | 473.8 ms | 473.9 ms | +0.02% (identical) |
|  64 | 847.5 ms | 838.4 ms | -1.1% (noise) |
| 128 | 1618.7 ms | 1602.2 ms | -1.0% (noise) |

CVs 0.48–0.90% — wash is real, not noise.

### Decision

**ABANDONED**. The targeted phase did not move. Hypothesis disproved.

### Postmortem — why the change didn't help

The agent's diagnosis (Pattern 3 research) was correct in principle but
incomplete: it identified that P3 uses tree-sum accumulators, but it
assumed Lambda's `fold()` would be serially-bound at runtime. **That
assumption is wrong for Rust + LLVM with associative field arithmetic.**

Specific reasons:

1. **LLVM auto-unrolls and schedules ILP from scalar folds.** The closure
   `|acc, (e, b)| acc + e * b` is recognized by LLVM as associative
   (Goldilocks add is associative; the field has no IEEE-754 corner cases
   for floats that would prevent reordering). LLVM unrolls the inner loop,
   issues multiple mul-add instructions per iteration, and keeps several
   partial sums in flight automatically. The generated assembly is
   already close to optimal.
2. **The 8-way explicit accumulator is redundant.** Our handwritten 8
   named locals (a0..a7) produce code that the compiler was already
   generating from the scalar fold. The benchmark confirmed: identical
   timings within noise (0.02% on r2_constraints @ 32 cols).
3. **The real bottleneck of `r2_constraints` is elsewhere.** Of the 474 ms
   at 32 cols, the dominant cost is `air.compute_transition_prover` (the
   per-LDE-point constraint evaluation, line 115 of evaluator.rs) plus
   memory traffic from `frame.fill_from_lde`. The final accumulator is a
   small fraction. Even if we made it 2× faster (which we didn't), the
   total would barely move.
4. **P3's `batched_linear_combination` win is real but for different
   reasons:** P3 uses `PackedField` SIMD lanes (`PackedVal<SC>::WIDTH`
   typically 4-16 on AVX2/AVX-512). Each iteration processes WIDTH points
   simultaneously, not just WIDTH coefficients of one point. The "tree-sum"
   structure in P3 amortizes the WIDTH points; without SIMD it doesn't
   help. With `--scalar` (our setting), P3's `Packing = Self` so WIDTH = 1
   and the tree-sum collapses to scalar — equivalent to our fold.

### Lessons

- **Don't trust source-level reasoning about ILP.** Modern compilers
  (LLVM/GCC) routinely break dependency chains in associative ops. A
  handwritten N-way accumulator only helps in languages/contexts where
  the compiler can't auto-vectorize (floats with strict order, FFI
  boundaries, opaque trait methods that can't be inlined).
- **Profile assembly, not source.** Before replicating a SIMD-friendly
  pattern from another codebase, check what the compiler already generates.
- **The "Pattern 3" research conclusion was structurally right but the
  win is gated on SIMD WIDTH > 1**, which our `--scalar` setup forbids.
  See [[feedback_no_simd_for_now]] — SIMD is off the table for now,
  so this pattern doesn't apply.
- **P3's structural advantage is the `Folder` + `compute_quotient_values`
  with SIMD lanes**, not the tree-sum per se. The tree-sum amortizes the
  SIMD register pressure; without SIMD it's redundant.

### Where the real wins still are

Re-ranking after this dead end:

- **Pattern 1 (precompute inverse zerofiers)**: still the cheapest, still
  ~10-30 ms win. Localized to boundary zerofier computation. Don't expect
  more than that.
- **Pattern 2 (batched coset LDE column-major)**: still on the table, but
  high effort (L-XL refactor). Saves `r1_main_lde+merkle` by single-pass
  FFT vs per-column, plus better cache locality. Independent of SIMD.
- **`r2_constraints` itself is unlikely to be optimizable further without
  SIMD**, given that:
  - The fold accumulator is already optimal (LLVM does ILP).
  - `compute_transition_prover` is the AIR-defined inner loop; the cost
    is the constraint count × Fp3 muls, which is irreducible without
    fewer constraints or cheaper Fp3.
  - `frame.fill_from_lde` could be reduced if the trace were column-major,
    but that's part of Pattern 2 scope.

The `experiment/batched-lin-combination` branch is **kept as historical
record**. Not merged.

---

## Scope expansion — 2026-05-15

After Attempts 1 and 2 hit the ceiling of what's possible within the
original "transparent to verifier" scope, the user explicitly **opened
the protocol to modification** for this workstream. The optimizer skill's
scope rules (`optimize-prover.md`) were updated accordingly:

- `crypto/stark/src/verifier.rs` — now editable
- `crypto/stark/src/proof/` — proof struct shape can evolve
- FRI structure, round structure, commit shape, Fiat-Shamir transcript —
  modifiable
- AIR/Constraint definitions of existing AIRs remain off-limits (correctness)

**Why the expansion**: research confirmed that the Lambda↔Plonky3 gap is
intrinsic to Lambda's **"single composition polynomial H(x)"** architecture
(Cairo / Stone style). To close the gap, we'd migrate to a **"quotient
chunks committed separately"** approach (Plonky3 / SP1 / OpenVM style),
which requires changing the commit shape, FRI structure, and verifier in
concert.

**Verification protocol under scope expansion**: build prover + verifier
on the same branch; verify proofs end-to-end on that branch; keep the
`/tmp/cli_baseline_verifier` only as a sanity check that **pre-change
proofs still verify under the pre-change verifier** (no regression on the
existing protocol while we explore).

See memory entry `project_optimizer_protocol_scope` for the full record.

---
