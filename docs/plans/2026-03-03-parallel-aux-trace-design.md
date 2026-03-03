# Parallel Auxiliary Trace Building — Design

**Goal:** Parallelize the three sequential bottlenecks in auxiliary trace construction: batch inversion, fingerprint computation, and accumulated column prefix sum.

**Architecture:** Follow Plonky3's chunked parallelism patterns — `par_chunks` for row-level parallelism within each interaction, chunked batch inversion with independent Montgomery's trick per chunk, and 3-phase parallel prefix sum for the accumulated column.

## Current State

The auxiliary trace build flow in `compute_logup_term_column` (lookup.rs:1182-1320) processes each interaction with three sequential phases:

1. **Fingerprint computation** — `for row in 0..trace_len` loop computing `z - Σ(element_i × α^i)` per row. Each row is independent (reads immutable `main_segment_cols`), but the loop is sequential. Cost: O(N × M) where M = bus elements per interaction.

2. **Batch inversion** — `inplace_batch_inverse` (element.rs:54-72) uses Montgomery's trick: forward prefix product → single inversion → backward extraction. Entirely sequential with loop-carried dependency on prefix products. Cost: 1 inversion + 3N mults.

3. **Accumulated column** — `build_accumulated_column_from_terms` (lookup.rs:1328-1350) computes running sum `acc[i] = acc[i-1] + Σ terms[i]`. Inherently sequential prefix sum. Cost: O(N × K) additions where K = num interactions.

The only parallelism currently exploited is at the interaction level (1-3 interactions per table via `par_iter` in `build_auxiliary_trace`), which underutilizes cores on high-count machines.

## Plonky3 Reference

Plonky3's `batch_multiplicative_inverse` (field/src/batch_inverse.rs):
- `CHUNK_SIZE = 1024`, `WIDTH = 4` for ILP
- `par_chunks(CHUNK_SIZE)` — each chunk gets independent Montgomery's trick
- Trade: N/1024 inversions instead of 1, but full Rayon parallelism
- For Goldilocks: 1 inversion ≈ 72 mults, so overhead for N=2^20 is ~73k extra mults (negligible vs 3M total)

Plonky3's `generate_permutation` (lookup/src/logup.rs):
- Phase 1: `par_chunks_mut` for denominator/multiplicity computation across rows
- Phase 2: `batch_multiplicative_inverse` (parallel, as above)
- Phase 3: Parallel prefix sum — local inclusive sums per chunk → sequential combine → fold offsets back

## Design

### 1. Parallel Batch Inversion

Add `parallel_batch_inverse` to `FieldElement` in `crypto/math/src/field/element.rs`, gated on `#[cfg(feature = "parallel")]`:

```rust
const BATCH_INV_CHUNK_SIZE: usize = 1024;

pub fn parallel_batch_inverse(numbers: &mut [Self]) -> Result<(), FieldError> {
    if numbers.len() <= BATCH_INV_CHUNK_SIZE {
        return Self::inplace_batch_inverse(numbers);
    }
    numbers.par_chunks_mut(BATCH_INV_CHUNK_SIZE)
        .try_for_each(|chunk| Self::inplace_batch_inverse(chunk))
}
```

Each chunk runs independent Montgomery's trick. No data dependency between chunks. Results are identical — each element gets its correct inverse regardless of batching boundary.

The existing `inplace_batch_inverse` stays unchanged for the non-parallel path and as the per-chunk worker.

### 2. Parallel Fingerprint Computation

Restructure `compute_logup_term_column` (lookup.rs:1270-1319) to use `par_chunks`:

```rust
// Pre-allocate fingerprints
let mut fingerprints: Vec<FieldElement<E>> = vec![FieldElement::zero(); trace_len];

// Parallel fingerprint computation — each row is independent
#[cfg(feature = "parallel")]
fingerprints.par_chunks_mut(CHUNK_SIZE).enumerate().for_each(|(chunk_idx, chunk)| {
    let start = chunk_idx * CHUNK_SIZE;
    for (local_i, fp) in chunk.iter_mut().enumerate() {
        let row = start + local_i;
        // accumulate fingerprint from main_segment_cols (immutable, shared)
        *fp = z - compute_fingerprint_for_row(row, ...);
    }
});

// Parallel batch inversion
FieldElement::parallel_batch_inverse(&mut fingerprints)?;

// Parallel term computation
let terms: Vec<FieldElement<E>> = multiplicities.par_iter()
    .zip(fingerprints.par_iter())
    .map(|(m, fp_inv)| m * &sign * fp_inv)
    .collect();
```

The fingerprint loop body reads `main_segment_cols[col][row]` — column-major access, each row independent. Safe to parallelize with shared immutable borrows.

### 3. Parallel Prefix Sum for Accumulated Column

Replace `build_accumulated_column_from_terms` with a 3-phase parallel prefix sum:

**Phase A — Local prefix sums per chunk:**
Each Rayon chunk computes inclusive prefix sum of `Σ terms[row]` within its range. Store results + chunk total.

**Phase B — Sequential combine:**
Accumulate chunk totals into global offsets: `offset[0] = 0, offset[k] = offset[k-1] + total[k-1]`.

**Phase C — Fold offsets:**
Each chunk adds its global offset to all its prefix sums.

With our circular constraint (`acc[N-1] = 0`), the prefix sum builds `acc[i] = Σ_{j=0..i} row_sum[j]`, then we compute `table_contribution = acc[N-1]` and shift: `acc[i] -= table_contribution` so `acc[N-1] = 0`. This shift is embarrassingly parallel.

The `table_contribution` value is returned in `BusPublicInputs` for the verifier.

## Files Modified

| File | Change |
|------|--------|
| `crypto/math/src/field/element.rs` | Add `parallel_batch_inverse` method |
| `crypto/math/Cargo.toml` | Add optional `rayon` dep for parallel feature |
| `crypto/stark/src/lookup.rs` | Parallelize `compute_logup_term_column` internals + `build_accumulated_column_from_terms` |

## Verification

1. `cargo test -p math` — batch inversion correctness
2. `cargo test -p stark --features parallel` — all STARK tests
3. `cargo test -p lambda-vm-prover --features parallel -- prove_elfs` — full VM tests
4. `cargo bench -p lambda-vm-prover -- vm_prover` — before/after comparison

## Risks

- **CHUNK_SIZE tuning**: 1024 is Plonky3's default. May need benchmarking for Goldilocks cubic extension field elements (larger than BabyBear). Could profile with 512, 1024, 2048.
- **Small tables**: Tables with <1024 rows (HALT=2, DECODE=6) fall through to sequential path. No regression.
- **Memory**: Parallel prefix sum needs one extra `Vec<FieldElement<E>>` of size `num_chunks` for offsets. Negligible.
