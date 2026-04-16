# Lambda STARK vs Plonky3 — Analysis Log

## Session: 2026-04-14 to 2026-04-16

---

## 0. Final Server Baseline (2026-04-16)

**Config:** blowup=2, 219 queries, grinding=0, ext degree 3 both, scalar (no AVX2), parallel (rayon both), identical AIR (32 cols × 2^18).

**Command:** `RUSTFLAGS="-C target-feature=-avx2,-avx512f" cargo bench -p bench-vs-plonky3`

### Prove

| Prover | Time | Throughput |
|--------|------|------------|
| Lambda | **1.213 s** | 6.92 Melem/s |
| Plonky3 | **479 ms** | 17.50 Melem/s |
| **Ratio** | **2.53×** | |

### Verify

| Prover | Time |
|--------|------|
| Lambda | **23.3 ms** |
| Plonky3 | **20.4 ms** |
| **Ratio** | **1.14×** |

### Gap attribution (734ms = 1213 - 479)

Extension field is MATCHED (both degree 3). The 2.53× is pure algorithm/implementation:

| Cause | Est. savings | % of gap | Effort |
|-------|-------------|----------|--------|
| **Quotient domain eval** (2^18 vs 2^19 LDE) | ~220ms | 30% | Low |
| **Batched FFT** (coset_lde_batch vs per-column) | ~150ms | 20% | Medium |
| **Alpha decomposition + monomorphization** | ~100ms | 14% | Medium-High |
| **FRI folding parallel** | ~73ms | 10% | Very low |
| **Boundary selectors** (vs zerofier precompute) | ~45ms | 6% | Low |
| **Memory allocation patterns** | ~37ms | 5% | Low |
| **SSE2 Keccak residual** (~7% hash advantage) | ~50ms | 7% | N/A (can't fix) |
| Other (compilation, unrolling, tuning) | ~59ms | 8% | - |

### Predicted instruments breakdown (blowup=2, 219q)

| Phase | Predicted time | % |
|-------|---------------|---|
| FRI queries (R4) | 180ms | 15% ← NEW bottleneck (2.19× queries) |
| R2 constraint eval | 168ms | 14% |
| R4 deep comp poly | 131ms | 11% |
| R1 Main Merkle | 105ms | 9% |
| R4 FRI commit | 76ms | 6% |
| R1 reconstruct LDE | 71ms | 6% |
| R3 OOD eval | 71ms | 6% |
| R1 Main LDE | 65ms | 5% |
| R4 deep extend | 52ms | 4% |
| R2 comp Merkle | 13ms | 1% |
| Pre-pass | 11ms | 1% |

### Optimization roadmap (ranked by impact/effort)

| # | Optimization | Savings | Effort | Result |
|---|-------------|---------|--------|--------|
| 1 | Quotient domain (stride=blowup in evaluator) | ~80ms | 1h | 1.13s |
| 2 | Parallel FRI fold (par_iter) | ~40ms | 30min | 1.09s |
| 3 | Boundary selectors (replace zerofier precompute) | ~45ms | 2h | 1.05s |
| 4 | LogUp alpha precompute | ~10ms | 30min | 1.04s |
| 5 | Monomorphize constraints (enum dispatch) | ~35ms | 4h | 1.00s |
| 6 | Batched FFT (coset_lde_batch pattern) | ~150ms | 8h | 0.85s |
| 7 | Row-major trace storage | ~20ms | 8h | 0.83s |

**With items 1-5 (~210ms, ~8h work):** Lambda ~1.0s vs Plonky3 0.48s = **2.08×**
**With items 1-7 (~380ms, ~24h work):** Lambda ~0.83s vs Plonky3 0.48s = **1.73×**
**Remaining gap** after all: ~350ms from SSE2 Keccak + deep comp + Plonky3 micro-optimizations

### M1 instruments breakdown (with PR #492, blowup=2, ext3 both)

**Command:** `RUSTFLAGS="-C target-feature=-sha3" cargo test -p bench-vs-plonky3 --features instruments --release -- instruments_breakdown --nocapture`

| Fase | Lambda (1.068s) | % | Plonky3 (352ms) | % | Ratio |
|------|-----------------|---|-----------------|---|-------|
| Trace commit (LDE+Merkle) | 317ms (LDE 127 + Merkle 165) | 30% | 138ms (commit to trace data) | 39% | 2.3× |
| **Constraint eval** | **325ms** | **30%** | **50ms** (quotient_values) | **14%** | **6.5×** |
| Quotient commit | 53ms | 5% | 49ms | 14% | 1.1× |
| OOD eval | 62ms | 6% | ~10ms (Lagrange interp) | 3% | 6.2× |
| Deep comp poly | 173ms | 16% | (inside "open") | | |
| Deep extend | 36ms | 3% | | | |
| FRI commit (folding+Merkle) | 83ms | 8% | 47ms (commit phase) | 13% | 1.8× |
| FRI queries | 1ms | 0% | 2ms (query phase) | 1% | — |
| Open total | 293ms | 27% | 110ms | 31% | 2.7× |
| Pre-pass | 7ms | 1% | — | | |

---

## Fairness Audit

### AIR equivalence: VERIFIED

Both AIRs prove the same mathematical statement:
- 32 cols × 2^18 rows, 2-row window
- Constraint 1: `next_left = local_left + local_right`
- Constraint 2: `next_right = local_right + next_left`
- Boundary: row 0 pins `(a_s, b_s) = (s+1, s+2)` per sequence
- Test `lambda_pair_trace_matches_plonky3_trace` verifies ALL cells (not subset)
- Mathematical trace for seq (1,2): (1,2)→(3,5)→(8,13)→(21,34) — identical both sides

### Parameters: ALL MATCHED (except noted)

| Parameter | Lambda | Plonky3 | Status |
|-----------|--------|---------|--------|
| Base field | Goldilocks | Goldilocks | ✅ |
| Extension | degree 3 (`x³−2`) | degree 3 (`x³−2`, vendored) | ✅ |
| Blowup | 2 | 2 (log_blowup=1) | ✅ |
| FRI queries | 219 | 219 | ✅ |
| Grinding | 0 | 0 | ✅ |
| Hash | Keccak-256 | Keccak-256 | ✅ |
| Rayon | ON | ON (p3-uni-stark/parallel + p3-dft/parallel) | ✅ |
| SIMD Goldilocks | OFF | OFF (NEON patched to `Self`) | ✅ |
| SIMD Keccak (x86) | scalar (sha3 crate) | SSE2 2-wide | ⚠️ residual |
| SIMD Keccak (M1 with -sha3) | scalar | scalar (fallback) | ✅ |

### Platform fairness guide

| Platform | Command | Keccak P3 | Goldilocks P3 | Fairness |
|----------|---------|-----------|---------------|----------|
| **M1 + `-sha3`** | `RUSTFLAGS="-C target-feature=-sha3" cargo bench ...` | Scalar | Scalar | **100% fair** |
| M1 no flags | `cargo bench ...` | NEON SHA3 HW | Scalar | P3 has Keccak HW |
| **x86 + `-avx2,-avx512f`** | `RUSTFLAGS="-C target-feature=-avx2,-avx512f" cargo bench ...` | SSE2 2-wide | Scalar | ~93% fair |
| x86 no flags | `cargo bench ...` | AVX2 4-wide | AVX2 4-wide | P3 has full SIMD |

**For fairest comparison: M1 with `-sha3`** — only platform where everything is scalar both sides.

### Security model asymmetry (doesn't affect compute, affects interpretation)

- **Lambda (Johnson Bound, proven):** 219 queries × 0.49 bits/query = **~108 bits** proven security
- **Plonky3 (ethSTARK conjecture):** 219 queries × 1.0 bit/query = **~219 bits** conjectured (cap 192 by field)
- Same 219 queries = same computational work. Different security interpretation.
- For "matched security" at 108 conjectured bits, P3 would need only ~108 queries (half the FRI work)

### What's NOT unfairness (architectural differences = what we measure)

These are implementation choices, not benchmark bias:
- Quotient domain eval (P3) vs full LDE eval (Lambda) → 6.5× constraint eval
- Monomorphization (P3) vs vtable dispatch (Lambda) → ~1.2× overhead
- Batched FFT (P3) vs per-column (Lambda) → ~2× trace commit
- Row-major (P3) vs column-major (Lambda) → cache efficiency
- Boundary selectors (P3) vs zerofier precompute (Lambda) → ~2× boundary cost

### What IS potential unfairness

1. SSE2 Keccak on x86 — P3 gets 2-wide Keccak, Lambda doesn't. ~7% of total. Unavoidable on x86.
2. Lambda samples NO extra LogUp/bus challenges for this AIR (verified: `has_aux_trace() = false` skips sampling).
3. Lambda wraps in `multi_prove` with vec of 1 — transcript clone overhead is negligible.

**Conclusion: The benchmark is fair for comparing prover implementation efficiency.**

---

## 1. Benchmark Setup

### AIR (identical both sides)
- 16 Fibonacci sequences, 2 cols/sequence = **32 columns**
- **2^18 rows** (each row packs 2 Fibonacci steps → 2^19 effective steps)
- 2-row window: `next.left = local.left + local.right`, `next.right = local.right + next.left`
- 32 boundary constraints pinning initial values via public inputs
- Test `lambda_pair_trace_matches_plonky3_trace` verifies cell-by-cell equivalence

### Matched parameters
- Base field: Goldilocks (p = 2^64 − 2^32 + 1)
- Blowup: 4
- FRI queries: 100
- Grinding: 0
- Hash: Keccak-256 (scalar on both sides when `-C target-feature=-sha3`)

### Unmatched (architectural)
- **Extension field:** Lambda degree 3 (`x^3 - 2`, 192-bit), Plonky3 degree 2 (`x^2 - 7`, 128-bit)
  - Plonky3 0.5.2 has Goldilocks extensions for degree 2 and 5, but NOT degree 3
  - Lambda ext-mul: 9 base muls + 3 reduce128
  - Plonky3 ext-mul: 4 base muls + 2 adds
- **Prover architecture:** Lambda multi_prove (even for 1 AIR), Plonky3 uni-stark

### Patches applied
1. `bench_vs_plonky3/vendor-p3-goldilocks/` — `Packing = Self` on aarch64 (disables NEON)
2. `p3-uni-stark` and `p3-dft` features `["parallel"]` enabled
3. `stark` feature `parallel` enabled by default in bench

### Files
- `bench_vs_plonky3/src/lambda_fibonacci_pair.rs` — Lambda AIR matching P3 shape
- `bench_vs_plonky3/src/plonky3_fibonacci.rs` — Plonky3 AIR
- `bench_vs_plonky3/src/plonky3_config.rs` — P3 config (matched FRI params)
- `bench_vs_plonky3/benches/stark_comparison.rs` — Criterion benchmark
- `bench_vs_plonky3/vendor-p3-goldilocks/` — Patched p3-goldilocks (no NEON)
- Root `Cargo.toml` — `[patch.crates-io]` for vendor p3-goldilocks

---

## 2. Measurements

### Config A: Both rayon, no SIMD, no SHA3 HW (M1 Max)

Command: `RUSTFLAGS="-C target-feature=-sha3" cargo bench -p bench-vs-plonky3`

| | Lambda | Plonky3 | Ratio |
|--|--------|---------|-------|
| **Prove** | **2.09s** [1.99, 2.20] | **0.86s** [0.84, 0.87] | **P3 2.43× faster** |
| **Verify** | **6.58ms** | **6.76ms** | **Lambda 1.03× faster** |

### Config B: Lambda rayon ON, Plonky3 rayon OFF, NEON ON (M1 — earlier run)

Command: `RUSTFLAGS="-C target-feature=-sha3" cargo bench -p bench-vs-plonky3` (before adding p3 parallel features)

| | Lambda | Plonky3 | Ratio |
|--|--------|---------|-------|
| **Prove** | **3.46s** | **2.92s** | **P3 1.18× faster** |

### Config C: Lambda rayon ON, Plonky3 rayon OFF, NEON ON, SHA3 HW ON (M1 — first run)

Command: `cargo bench -p bench-vs-plonky3` (no RUSTFLAGS)

| | Lambda | Plonky3 | Ratio |
|--|--------|---------|-------|
| **Prove** | **3.21s** | **1.67s** | **P3 1.92× faster** |

### Server instruments breakdown (Lambda only, 16 cols × 2^18 pair AIR)

Total: **1.246s**

| Phase | Time | % |
|-------|------|---|
| R2 constraint eval | 336ms | 27% |
| R1 Main Merkle | 211ms | 17% |
| R1 reconstruct (re-LDE) | 143ms | 11% |
| R4 deep comp poly | 131ms | 11% |
| R1 Main LDE | 130ms | 10% |
| R4 FRI commit | 80ms | 6% |
| R3 OOD eval | 71ms | 6% |
| R2 comp Merkle | 54ms | 4% |
| R4 deep extend | 43ms | 3% |
| Pre-pass | 11ms | 1% |

---

## 3. Root Cause Analysis

### Why Plonky3 is ~2.4× faster (Config A)

#### 3a. Constraint eval domain: 4× overhead (biggest factor)
- Lambda evaluates constraints on full LDE domain: `N × blowup = 2^20 points` (`evaluator.rs:274`)
- Plonky3 evaluates on quotient domain: `N = 2^18 points`, then extends via iFFT + FFT
- Lambda does 4× more constraint evaluations (each involving ext-field ops, frame fill, zerofier division)
- **Estimated contribution: 1.5-2× of the gap**

#### 3b. Extension field degree 3 vs 2
- Lambda: 9 base muls per ext-mul (`extensions_goldilocks.rs:293-309`)
- Plonky3: 4 base muls per ext-mul (`binomial_extension.rs:747-762`)
- Affects: composition poly, FRI folding, DEEP openings, OOD
- **Estimated contribution: 1.3-1.5× of the gap**

#### 3c. Virtual dispatch vs monomorphization
- Lambda: `Vec<Box<dyn TransitionConstraint>>` → vtable call per constraint per point (`traits.rs:248-250`)
- Plonky3: `air.eval(&mut folder)` → monomorphized, all constraints inlined
- For 32 constraints × 2^20 points = 32M vtable dispatches in Lambda
- **Estimated contribution: 1.1-1.2× of the gap**

#### 3d. Data layout: column-major vs row-major
- Lambda: column-major (cache miss per column access in constraint loop)
- Plonky3: row-major (contiguous data per row)
- **Estimated contribution: 1.05-1.1× of the gap**

#### 3e. FRI folding sequential vs parallel
- Lambda: sequential loop in `fold_evaluations_in_place` (`fri_functions.rs:21`)
- Plonky3: `par_rows()` parallelized
- **Estimated contribution: 1.03-1.05× of the gap**

#### Combined: 1.5 × 1.4 × 1.15 × 1.07 × 1.04 ≈ **2.7× (close to measured 2.43×)**

### Why verify is roughly equal
- Verify doesn't do LDE, Merkle, or constraint eval
- Only ~100 point openings + FRI check
- Extension field penalty minimal at small N
- Lambda's implementation is competitive on this path

---

## 4. SIMD Analysis (from profiling session)

### NEON (aarch64/M1)
- `target_feature="neon"` and `target_feature="sha3"` are **default on aarch64-apple-darwin**
- Plonky3 uses `PackedGoldilocksNeon` (WIDTH=2) unconditionally on aarch64 via `#[cfg(target_arch = "aarch64")]`
- Plonky3 Keccak uses NEON SHA3 instructions (`veor3q_u64`, `vbcaxq_u64`, etc.)
- Lambda has NO SIMD in the prover
- **Goldilocks NEON base-field mul is 0.92× SLOWER** than scalar (no native 64×64→128 on NEON)
- **Fp3 NEON mul is 1.40× faster** (parallelism helps with 3 components)
- **FFT with SIMD was 0.88× (slower)** due to pack/unpack overhead

### Disabling SIMD
- NEON packing: patched via `vendor-p3-goldilocks` (`type Packing = Self` on aarch64)
- SHA3 hardware Keccak: `-C target-feature=-sha3` (RUSTFLAGS)
- Cannot disable NEON via RUSTFLAGS alone (intrinsics used without `#[target_feature]` annotation)

### x86_64 (server)
- Without `-C target-cpu=native`: only SSE2 (no AVX2) → Plonky3 scalar too
- With AVX2: `PackedGoldilocksAVX2` (WIDTH=4) — has native `mulq` so SIMD IS beneficial
- For fair scalar comparison on x86: `RUSTFLAGS="-C target-feature=-avx2,-avx512f"`

---

## 5. Plonky3 Parallelism

- `p3-maybe-rayon` feature `parallel` is NOT enabled by default
- Without it, all `par_iter()` calls fall back to `core::iter` (sequential)
- `Radix2DitParallel` is "parallel" in name only without the feature
- Must explicitly enable: `p3-uni-stark = { version = "0.5.2", features = ["parallel"] }` + `p3-dft = ...`
- Verified via `cargo tree -e features | grep p3-maybe-rayon`

---

## 6. Lambda Profiling Results (server, profile_prover, 2^20 × 16 cols)

### Single-threaded (38.7s)
| Component | % | Category |
|-----------|---|----------|
| Constraint evaluation | 32.1% | Compute |
| Keccak hashing | 15.1% | Hashing |
| Deep composition poly | 14.0% | Compute |
| Merkle tree build | 12.0% | Hashing |
| Field multiplication | 11.1% | Compute |
| FFT | 10.5% | FFT |
| Other | 5.2% | |

### Parallel (12 threads, 19.2s — 2.02× speedup)
| Metric | Value |
|--------|-------|
| Parallel efficiency | 16.8% of ideal 12× |
| CPU utilization | 30.6% |
| Main thread work | 13.3s |
| Worker thread work | ~5s each |
| New #1 bottleneck | Keccak (16.7%) |

### Key profiling findings
- 100% CPU-bound (no memory/IO stalls)
- SIMD PackedGoldilocks types exist but are NOT used by prover
- Iterator overhead (Map::fold + FnMut): 7.6%
- Memory allocation overhead: 8.9% (page faults + malloc + cfree)
- Amdahl's Law: ~34% serial portion limits parallel speedup

---

## 7. Optimizations Implemented (then stashed)

### Item 2: Parallel FRI folding
- File: `crypto/stark/src/fri/fri_functions.rs`
- Change: `(0..half).into_par_iter().map().collect()` with `#[cfg(feature = "parallel")]`
- Also: `crypto/stark/src/fri/mod.rs` — added `Send + Sync` bounds
- Tests: 450/450 passed (121 stark + 326 VM + 3 bench)

### Item 3: Quotient domain constraint evaluation
- File: `crypto/stark/src/constraints/evaluator.rs` — added `lde_stride: usize` parameter
- File: `crypto/stark/src/prover.rs` — when `number_of_parts == 1`, uses `lde_stride = blowup_factor`
  then extends N evaluations to LDE via `interpolate_offset_fft + evaluate_polynomial_on_lde_domain`
- Tests: 450/450 passed
- Impact on M1: 2.09s → 2.02s (~3%, within Criterion noise)
- Impact limited because iFFT+FFT extension cost offsets constraint eval savings

### Why stashed
User wants clean baseline first (fair comparison), then optimize. These changes are ready to re-apply.

---

## 8. Optimization Priority (from profiling data)

### With parallel enabled (real-world scenario)

| # | Optimization | Impact (parallel) | Effort | Status |
|---|-------------|-------------------|--------|--------|
| 1 | PR 492 (LDE cache) | 5-8% (reduces serial) | Done (PR open) | Waiting merge |
| 2 | BLAKE3 hash | ~12% (Keccak is parallel bottleneck) | Low | Not started |
| 3 | Quotient domain eval | 3-5% (constraint eval parallelized already) | Medium | Implemented, stashed |
| 4 | Reduce allocations | 5-8% | Medium | Not started |
| 5 | Parallel FRI fold | ~3% | Low | Implemented, stashed |
| 6 | Monomorphize constraints | 3-5% | High | Not started |

### Plonky3 degree-3 extension (Option C)
- Would eliminate the last asymmetric variable in the comparison
- Requires implementing `BinomiallyExtendable<3>` for Goldilocks in vendored crate
- Need Sage computation for: `DTH_ROOT = 2^((p-1)/3)`, `EXT_GENERATOR`
- Expected: gap drops from 2.43× to ~1.5-1.7× (confirms extension degree accounts for ~40% of gap)

---

## 9. How to Run

### M1 / aarch64 (scalar comparison)
```bash
RUSTFLAGS="-C target-feature=-sha3" cargo bench -p bench-vs-plonky3
```

### x86_64 server (scalar comparison, no AVX2)
```bash
cargo bench -p bench-vs-plonky3
# or explicitly: RUSTFLAGS="-C target-feature=-avx2,-avx512f" cargo bench ...
```

### With instruments (Lambda phase breakdown)
```bash
# Add "instruments" to stark features in bench_vs_plonky3/Cargo.toml first
cargo bench -p bench-vs-plonky3 --features stark/instruments
```

### Verify correctness
```bash
cargo test -p bench-vs-plonky3  # 3 tests
cargo test -p stark --lib       # 121 tests
cargo test -p lambda-vm-prover  # 326 tests
```

---

## 10. Key Files Reference

| File | Purpose |
|------|---------|
| `bench_vs_plonky3/src/lambda_fibonacci_pair.rs` | Lambda AIR (32 cols, 2-row window) |
| `bench_vs_plonky3/src/plonky3_fibonacci.rs` | Plonky3 AIR (matching) |
| `bench_vs_plonky3/src/plonky3_config.rs` | P3 config (FRI params matched) |
| `bench_vs_plonky3/benches/stark_comparison.rs` | Criterion benchmark |
| `bench_vs_plonky3/vendor-p3-goldilocks/` | Patched p3-goldilocks (no NEON) |
| `crypto/stark/src/constraints/evaluator.rs` | Constraint eval loop (bottleneck) |
| `crypto/stark/src/prover.rs` | Prover pipeline (Round 1-4) |
| `crypto/stark/src/fri/fri_functions.rs` | FRI folding |
| `crypto/stark/src/domain.rs` | LDE domain definition |
| `crypto/math/src/fft/polynomial.rs` | FFT / coset_lde_full_expand |
