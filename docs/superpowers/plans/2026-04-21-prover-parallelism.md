# Prover Parallelism Improvements

**Goal:** Close the ~2x performance gap with Plonky3 by parallelizing the hottest sequential code paths in the STARK prover.

**Architecture:** Five independent parallelism improvements. No protocol changes, no new data structures. Each produces identical outputs but runs faster. All use Rayon gated behind `#[cfg(feature = "parallel")]`.

---

## Task 1: Parallel FRI fold

**File:** `crypto/stark/src/fri/fri_functions.rs`

`fold_evaluations_in_place` is a plain `for j in 0..half` loop. At layer 0 with LDE 2^21, ~1M sequential iterations. Each output depends only on its own pair -- no cross-iteration dependency. Use `into_par_iter` with a temp buffer (aliasing prevents in-place parallel write).

## Task 2: Parallel FRI leaf construction

**File:** `crypto/stark/src/fri/mod.rs`

Leaf array `evals.chunks_exact(2).map(...).collect()` is sequential. Use `par_chunks_exact`.

## Task 3: Parallel LogUp fingerprint computation

**File:** `crypto/stark/src/lookup.rs`

Two fingerprint loops in `compute_logup_batched_term_column` (lines 1619-1650): sequential over trace_len rows. Each row reads shared immutable data. Use `into_par_iter`. Also parallelize the final term computation loop and `compute_logup_term_column`.

## Task 4: Parallel table_contribution sum

**File:** `crypto/stark/src/lookup.rs`

`build_accumulated_column_from_terms` sums term columns across all rows sequentially. Use `into_par_iter` with `reduce`. Note: the accumulated column running-sum loop CANNOT be parallelized.

## Task 5: Chunked parallel batch inverse

**Files:** `crypto/math/src/field/element.rs`, `crypto/stark/src/lookup.rs`

Montgomery batch inverse is sequential. Split into K=num_threads chunks, run one independent batch inverse per chunk via `par_chunks_mut`. Cost: K-1 extra inversions, but O(N/K) per thread. Threshold at 1024 elements.

## Task 6: Benchmark and validate

Run `cargo bench --bench profile_vm_prover --features "parallel,instruments"`. Compare against baseline. Push.

---

## Expected Impact

| Optimization | Sequential cost | Parallel | Speedup |
|---|---|---|---|
| FRI fold | O(N) ext-field per layer | O(N/P) | ~P |
| FRI leaves | O(N) clones per layer | O(N/P) | ~P |
| Fingerprint loops | O(N) F*E per pair | O(N/P) | ~P |
| Batch inverse | O(N) prefix-suffix | O(N/P + K inv) | ~P large N |
| table_contribution | O(N*cols) | O(N*cols/P) | ~P |

These target ~20-30% of prover time. Combined with MMCS/shared-FRI (~30-40%), closes most of the 2x gap.
