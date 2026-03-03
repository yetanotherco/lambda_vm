# Parallel Auxiliary Trace Building — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Parallelize the three sequential bottlenecks in auxiliary trace construction — fingerprint computation, batch inversion, and accumulated column prefix sum — following Plonky3's chunked parallelism patterns.

**Architecture:** Add `parallel_batch_inverse` to `FieldElement`, restructure `compute_logup_term_column` to use `par_chunks_mut` for fingerprint computation, and replace `build_accumulated_column_from_terms` with a 3-phase parallel prefix sum.

**Tech Stack:** Rust, Rayon (`par_chunks_mut`, `par_iter`), existing `math` and `stark` crates.

---

### Task 1: Add `parallel_batch_inverse` to FieldElement

**Files:**
- Modify: `crypto/math/src/field/element.rs:48-72`
- Test: `crypto/math/src/tests/fft_friendly_extensions_goldilocks_tests.rs`

**Step 1: Add the `parallel_batch_inverse` method**

Add this method to the existing `#[cfg(feature = "alloc")] impl<F: IsField> FieldElement<F>` block at `element.rs:48`, right after `inplace_batch_inverse` (line 72):

```rust
/// Chunk size for parallel batch inversion. Each chunk runs independent
/// Montgomery's trick. Trade: N/CHUNK_SIZE inversions instead of 1, but
/// full Rayon parallelism. For Goldilocks: 1 inversion ≈ 72 mults, so
/// overhead is negligible.
#[cfg(feature = "parallel")]
const BATCH_INV_CHUNK_SIZE: usize = 1024;

/// Parallel variant of `inplace_batch_inverse`. Splits the slice into chunks
/// of `BATCH_INV_CHUNK_SIZE` and runs independent Montgomery's trick per chunk.
/// Falls back to sequential for small slices.
#[cfg(feature = "parallel")]
pub fn parallel_batch_inverse(numbers: &mut [Self]) -> Result<(), FieldError>
where
    Self: Send,
{
    use rayon::prelude::*;
    if numbers.len() <= BATCH_INV_CHUNK_SIZE {
        return Self::inplace_batch_inverse(numbers);
    }
    numbers
        .par_chunks_mut(BATCH_INV_CHUNK_SIZE)
        .try_for_each(|chunk| Self::inplace_batch_inverse(chunk))
}
```

**Step 2: Write tests for parallel batch inverse**

Add these tests to `crypto/math/src/tests/fft_friendly_extensions_goldilocks_tests.rs`, after the existing `test_fp3_batch_inverse_large` test (line 425):

```rust
/// Test parallel batch inverse for Fp3 (cubic extension, same type used in STARK).
/// Uses > 1024 elements to exercise multi-chunk path.
#[test]
#[cfg(feature = "parallel")]
fn test_fp3_parallel_batch_inverse() {
    let elements: Vec<Fp3E> = (1u64..=2048)
        .map(|i| Fp3E::new([FpE::from(i), FpE::from(i + 100), FpE::from(i + 200)]))
        .collect();

    let original = elements.clone();
    let mut to_invert = elements;

    Fp3E::parallel_batch_inverse(&mut to_invert).unwrap();

    for (inv, orig) in to_invert.iter().zip(original.iter()) {
        assert_eq!(*inv * *orig, Fp3E::one());
    }
}

/// Test parallel batch inverse falls back to sequential for small slices.
#[test]
#[cfg(feature = "parallel")]
fn test_fp3_parallel_batch_inverse_small() {
    let elements: Vec<Fp3E> = (1u64..=10)
        .map(|i| Fp3E::new([FpE::from(i), FpE::from(i + 100), FpE::from(i + 200)]))
        .collect();

    let original = elements.clone();
    let mut to_invert = elements;

    Fp3E::parallel_batch_inverse(&mut to_invert).unwrap();

    for (inv, orig) in to_invert.iter().zip(original.iter()) {
        assert_eq!(*inv * *orig, Fp3E::one());
    }
}

/// Verify parallel and sequential batch inverse produce identical results.
#[test]
#[cfg(feature = "parallel")]
fn test_fp3_parallel_batch_inverse_matches_sequential() {
    let elements: Vec<Fp3E> = (1u64..=3000)
        .map(|i| Fp3E::new([FpE::from(i), FpE::from(i + 100), FpE::from(i + 200)]))
        .collect();

    let mut sequential = elements.clone();
    let mut parallel = elements;

    Fp3E::inplace_batch_inverse(&mut sequential).unwrap();
    Fp3E::parallel_batch_inverse(&mut parallel).unwrap();

    assert_eq!(sequential, parallel);
}
```

**Step 3: Run tests**

Run: `cargo test -p math --features parallel -- batch_inverse`
Expected: All batch inverse tests pass (existing + 3 new).

**Step 4: Commit**

```bash
git add crypto/math/src/field/element.rs crypto/math/src/tests/fft_friendly_extensions_goldilocks_tests.rs
git commit -m "Add parallel_batch_inverse using chunked Montgomery's trick"
```

---

### Task 2: Parallelize fingerprint computation in `compute_logup_term_column`

**Files:**
- Modify: `crypto/stark/src/lookup.rs:1183-1320`

**Step 1: Add rayon imports for par_chunks_mut**

At line 22 of `lookup.rs`, extend the existing rayon import:

```rust
#[cfg(feature = "parallel")]
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};
```

Replace the existing:
```rust
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
```

**Step 2: Restructure fingerprint loop to use `par_chunks_mut`**

Replace the sequential fingerprint loop (lines 1270-1309) in `compute_logup_term_column` with parallel computation. The replacement code:

```rust
    // Fingerprint computation — each row is independent (reads immutable main_segment_cols).
    // Use par_chunks_mut for row-level parallelism within each interaction.
    let bus_id_f = FieldElement::<F>::from(table_interaction.bus_id);
    let mut fingerprints: Vec<FieldElement<E>> = vec![FieldElement::zero(); trace_len];

    #[cfg(feature = "parallel")]
    {
        const FINGERPRINT_CHUNK_SIZE: usize = 1024;
        fingerprints
            .par_chunks_mut(FINGERPRINT_CHUNK_SIZE)
            .enumerate()
            .for_each(|(chunk_idx, chunk)| {
                let start = chunk_idx * FINGERPRINT_CHUNK_SIZE;
                for (local_i, fp) in chunk.iter_mut().enumerate() {
                    let row = start + local_i;
                    let mut linear_combination = &bus_id_f * &alpha_powers[0];
                    let mut alpha_offset = 1;
                    for bv in &table_interaction.values {
                        let consumed = bv.accumulate_fingerprint(
                            main_segment_cols,
                            row,
                            &alpha_powers,
                            alpha_offset,
                            &mut linear_combination,
                        );
                        alpha_offset += consumed;
                    }
                    *fp = z - &linear_combination;
                }
            });
    }

    #[cfg(not(feature = "parallel"))]
    {
        for row in 0..trace_len {
            let mut linear_combination = &bus_id_f * &alpha_powers[0];
            let mut alpha_offset = 1;
            for bv in &table_interaction.values {
                let consumed = bv.accumulate_fingerprint(
                    main_segment_cols,
                    row,
                    &alpha_powers,
                    alpha_offset,
                    &mut linear_combination,
                );
                alpha_offset += consumed;
            }
            fingerprints[row] = z - &linear_combination;
        }
    }
```

Note: The `#[cfg(feature = "debug-checks")]` block inside the fingerprint loop needs to stay in the non-parallel path only, or be moved to a separate pass. Easiest: keep it only in the `#[cfg(not(feature = "parallel"))]` path. The `debug-checks` and `parallel` features are never used together in practice (debug-checks is for development only).

**Step 3: Replace sequential batch inversion with parallel**

Replace line 1311:
```rust
    FieldElement::inplace_batch_inverse(&mut fingerprints)
```
With:
```rust
    #[cfg(feature = "parallel")]
    FieldElement::parallel_batch_inverse(&mut fingerprints)
        .expect("fingerprint is zero - probability of sampling zero is negligible");
    #[cfg(not(feature = "parallel"))]
    FieldElement::inplace_batch_inverse(&mut fingerprints)
        .expect("fingerprint is zero - probability of sampling zero is negligible");
```

**Step 4: Parallelize term computation**

Replace the sequential term computation (lines 1314-1319):
```rust
    multiplicities
        .iter()
        .zip(fingerprints.iter())
        .map(|(multiplicity, fingerprint_inv)| multiplicity * &sign * fingerprint_inv)
        .collect()
```
With:
```rust
    #[cfg(feature = "parallel")]
    {
        multiplicities
            .par_iter()
            .zip(fingerprints.par_iter())
            .map(|(multiplicity, fingerprint_inv)| multiplicity * &sign * fingerprint_inv)
            .collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        multiplicities
            .iter()
            .zip(fingerprints.iter())
            .map(|(multiplicity, fingerprint_inv)| multiplicity * &sign * fingerprint_inv)
            .collect()
    }
```

For this to work, `multiplicities` needs to be a `Vec` or `&[FieldElement<F>]` that implements `IntoParallelRefIterator`. The current code has `multiplicities: &[FieldElement<F>]` which works with `par_iter()` via Rayon's slice implementation.

**Step 5: Run tests**

Run: `cargo test -p stark --features parallel`
Expected: All STARK tests pass.

Run: `cargo test -p lambda-vm-prover --features parallel -- prove_elfs`
Expected: All VM prover tests pass.

**Step 6: Commit**

```bash
git add crypto/stark/src/lookup.rs
git commit -m "Parallelize fingerprint computation and batch inversion in LogUp term column"
```

---

### Task 3: Parallel prefix sum for accumulated column

**Files:**
- Modify: `crypto/stark/src/lookup.rs:1322-1350` (function `build_accumulated_column_from_terms`)

**Step 1: Rewrite `build_accumulated_column_from_terms` with parallel prefix sum**

Replace the function at lines 1328-1350 with a version that returns the accumulated column as a `Vec` instead of writing directly to the trace. This enables the parallel prefix sum to operate on a contiguous buffer.

First, change the function signature and implementation:

```rust
/// Builds the accumulated column from pre-computed term columns using
/// parallel prefix sum (3-phase algorithm from Plonky3).
///
/// Phase A: Local prefix sums per chunk (parallel)
/// Phase B: Sequential combine of chunk totals into global offsets
/// Phase C: Fold global offsets into local prefix sums (parallel)
///
/// Returns the final accumulated value (for BusPublicInputs).
fn build_accumulated_column_from_terms<F, E>(
    acc_column_idx: usize,
    term_columns: &[Vec<FieldElement<E>>],
    trace: &mut TraceTable<F, E>,
) where
    F: IsFFTField + IsSubFieldOf<E> + Send + Sync,
    E: IsField + Send + Sync,
{
    if term_columns.is_empty() {
        return;
    }
    let trace_len = term_columns[0].len();

    // Compute row sums: for each row, sum all term columns
    let mut row_sums: Vec<FieldElement<E>> = Vec::with_capacity(trace_len);
    for row in 0..trace_len {
        let mut s = FieldElement::<E>::zero();
        for col in term_columns {
            s = s + &col[row];
        }
        row_sums.push(s);
    }

    // Prefix sum (acc[i] = Σ_{j=0..=i} row_sums[j])
    #[cfg(feature = "parallel")]
    let acc = parallel_prefix_sum(&row_sums);

    #[cfg(not(feature = "parallel"))]
    let acc = {
        let mut acc = Vec::with_capacity(trace_len);
        let mut running = FieldElement::<E>::zero();
        for s in &row_sums {
            running = &running + s;
            acc.push(running.clone());
        }
        acc
    };

    // Write to trace
    for (row, value) in acc.iter().enumerate() {
        trace.set_aux(row, acc_column_idx, value.clone());
    }
}

/// 3-phase parallel prefix sum.
///
/// Given `data[0..N]`, computes `result[i] = Σ_{j=0..=i} data[j]`.
///
/// Phase A: Each chunk computes its local inclusive prefix sum.
/// Phase B: Sequential scan of chunk totals into global offsets.
/// Phase C: Each chunk (except first) adds its global offset.
#[cfg(feature = "parallel")]
fn parallel_prefix_sum<E>(data: &[FieldElement<E>]) -> Vec<FieldElement<E>>
where
    E: IsField + Send + Sync,
{
    use rayon::prelude::*;

    const PREFIX_SUM_CHUNK_SIZE: usize = 1024;

    let n = data.len();
    if n <= PREFIX_SUM_CHUNK_SIZE {
        // Sequential fallback for small inputs
        let mut result = Vec::with_capacity(n);
        let mut running = FieldElement::<E>::zero();
        for d in data {
            running = &running + d;
            result.push(running.clone());
        }
        return result;
    }

    // Phase A: Local inclusive prefix sums per chunk
    let num_chunks = n.div_ceil(PREFIX_SUM_CHUNK_SIZE);
    let mut result: Vec<FieldElement<E>> = vec![FieldElement::zero(); n];

    // Collect chunk totals
    let chunk_totals: Vec<FieldElement<E>> = result
        .par_chunks_mut(PREFIX_SUM_CHUNK_SIZE)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let start = chunk_idx * PREFIX_SUM_CHUNK_SIZE;
            let mut running = FieldElement::<E>::zero();
            for (local_i, slot) in chunk.iter_mut().enumerate() {
                running = &running + &data[start + local_i];
                *slot = running.clone();
            }
            running
        })
        .collect();

    // Phase B: Sequential combine — compute global offsets from chunk totals
    let mut offsets: Vec<FieldElement<E>> = Vec::with_capacity(num_chunks);
    offsets.push(FieldElement::zero()); // first chunk has no offset
    let mut cumulative = FieldElement::<E>::zero();
    for total in chunk_totals.iter().take(num_chunks - 1) {
        cumulative = &cumulative + total;
        offsets.push(cumulative.clone());
    }

    // Phase C: Fold offsets into local prefix sums (skip first chunk — offset is 0)
    result
        .par_chunks_mut(PREFIX_SUM_CHUNK_SIZE)
        .enumerate()
        .skip(1)
        .for_each(|(chunk_idx, chunk)| {
            let offset = &offsets[chunk_idx];
            for slot in chunk.iter_mut() {
                *slot = &*slot + offset;
            }
        });

    result
}
```

**Step 2: Run tests**

Run: `cargo test -p stark --features parallel`
Expected: All STARK tests pass.

Run: `cargo test -p lambda-vm-prover --features parallel -- prove_elfs`
Expected: All VM prover tests pass.

**Step 3: Commit**

```bash
git add crypto/stark/src/lookup.rs
git commit -m "Replace sequential accumulated column with 3-phase parallel prefix sum"
```

---

### Task 4: Full integration verification and benchmark

**Files:**
- No new files — verification only.

**Step 1: Run full test suite**

Run: `cargo test -p math --features parallel`
Expected: All math tests pass.

Run: `cargo test -p stark --features parallel`
Expected: All STARK tests pass.

Run: `cargo test -p lambda-vm-prover --features parallel -- prove_elfs`
Expected: All VM prover tests pass (43 ELF tests).

**Step 2: Run benchmarks (before/after comparison)**

Run: `cargo bench -p lambda-vm-prover -- vm_prover`
Expected: Report benchmark numbers. The improvement should be visible on tables with large trace lengths (CPU, MEMW, DVRM — all 2^16+ rows).

**Step 3: Final commit if any cleanup needed**

Only if tests revealed issues that needed fixing.
