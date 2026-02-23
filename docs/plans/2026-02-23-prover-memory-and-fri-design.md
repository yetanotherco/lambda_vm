# Prover Memory Reduction and FRI Pipeline Optimization

**Date**: 2026-02-23
**Branch**: `perf/p2.2-p3.1-fused-lde-sequential-proving`
**Baseline**: prove 25.7s (vm_32k local), ~55s (remote larger trace)

## Problem

After the eval-form STARK migration and sequential per-table proving, the prover still has:
1. **Redundant cloning** of LDE columns and Merkle trees in `reconstruct_round1`
2. **Unnecessary coefficient-form round-trip** in Round 4 (deep composition -> FRI)
3. **FRI layers store full evaluation vectors** even though only point lookups are needed at query time
4. **Domain stores all coset points** in a Vec (16MB for large tables)

## Design

### Item 1: LDETraceTable borrows pool buffers instead of cloning

**Current**: `reconstruct_round1` calls `LDETraceTable::from_columns_borrowed` which clones every column from the pool.
For the Bitwise table (22 cols x 2^21 rows x 8 bytes), this is ~352MB of allocations per proof.

**Proposed**: Change `LDETraceTable` to hold references to the pool buffers instead of owned data.

```rust
pub struct LDETraceTable<'a, F, E> {
    main_columns: &'a [Vec<FieldElement<F>>],
    aux_columns: &'a [Vec<FieldElement<E>>],
    num_main: usize,  // actual columns used (pool may be larger)
    num_aux: usize,
    lde_step_size: usize,
    blowup_factor: usize,
}
```

Add a lifetime parameter. `from_columns_borrowed` becomes zero-copy. The `from_columns` (owned) variant remains for non-pool paths.

`Round1` also gains a lifetime since it contains `LDETraceTable`. This lifetime is scoped to `prove_rounds_2_to_4`, which borrows the pools.

**Files**: `trace.rs` (LDETraceTable), `prover.rs` (Round1, reconstruct_round1, prove_rounds_2_to_4)

### Item 2: Share Merkle trees via Rc instead of cloning

**Current**: `reconstruct_round1` clones `metadata.main_merkle_tree` and `metadata.aux_merkle_tree`. Each Merkle tree stores `2N-1` nodes of 32 bytes. For Bitwise (2^21 leaves): 64MB per clone. Across 12 tables: ~768MB total.

**Proposed**: Store Merkle trees in `Rc<MerkleTree<B>>` in both `Round1Metadata` and `Round1CommitmentData`.

```rust
pub struct Round1Metadata<Field, FieldExtension> {
    main_merkle_tree: Rc<BatchedMerkleTree<Field>>,
    // ...
}
pub struct Round1CommitmentData<F> {
    pub(crate) lde_trace_merkle_tree: Rc<BatchedMerkleTree<F>>,
    // ...
}
```

`reconstruct_round1` does `Rc::clone()` (pointer copy) instead of deep clone. Phase A wraps the tree in `Rc` at creation time. `open_trace_polys_*` methods take `&MerkleTree` (deref through Rc transparently).

**Files**: `prover.rs` (Round1Metadata, Round1CommitmentData, reconstruct_round1, Phase A, open_* methods)

### Item 3: Eval-form FRI commit (eliminate coefficient round-trip)

**Key insight**: Starting from the LDE of the DEEP polynomial, folding as the verifier does gives you the LDE of the next layer. No coefficient-form intermediary is ever needed.

**Current flow** (Round 4):
1. `deep_evals`: N evaluations on trace-size coset (output of `compute_deep_composition_poly_evaluations`)
2. `interpolate_offset_fft`: unscale(N) + iFFT(N) -> Polynomial (N coefficients allocated)
3. `commit_phase` calls `evaluate_offset_fft`: scale(2N) + FFT(2N) -> 2N evaluations on LDE coset
4. `bit_reverse_permute` + `fold_evaluations_in_place` loop

Steps 2-3 form a needless round-trip through coefficient form. The deep polynomial has degree < N, and we already have its evaluations. We just need to extend them to the full LDE domain.

**Proposed flow**:
1. `deep_evals`: N evaluations on trace-size coset (same as before)
2. `coset_lde(deep_evals, blowup_factor)`: fused iFFT(N) + zero-pad + FFT(2N) -> 2N evaluations on LDE coset (no Polynomial allocation, coset scaling fused into FFT butterflies)
3. `bit_reverse_permute` + `fold_evaluations_in_place` loop (same as before, FRI folds evaluations directly — each fold produces the LDE of the next layer)

Create `commit_phase_from_evaluations` that accepts pre-computed, bit-reversed evaluations:

```rust
pub fn commit_phase_from_evaluations<F, E>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,  // already bit-reversed on LDE coset
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> (FieldElement<E>, Vec<FriLayer<E, ...>>)
```

In Round 4:
```rust
// Extend deep evals from trace coset (N) to LDE coset (2N), then fold directly
let mut lde_evals = Polynomial::coset_lde_full(&deep_evals, blowup_factor, &coset_offset);
in_place_bit_reverse_permute(&mut lde_evals);
let (fri_last_value, fri_layers) = fri::commit_phase_from_evaluations(
    number_layers, lde_evals, transcript, &coset_offset, domain_size
);
```

**Savings**: Eliminates the Polynomial allocation (N extension field elements), the separate `scale()` pass (N muls), and replaces `interpolate_offset_fft` + `evaluate_offset_fft` with a single fused `coset_lde`. The FFT operations (iFFT(N) + FFT(2N)) are the same, but fused coset scaling avoids an extra memory pass. The FRI folding loop is identical — it already works in evaluation form, producing each layer's LDE by folding paired evaluations.

**Files**: `fri/mod.rs` (new `commit_phase_from_evaluations`), `prover.rs` (round_4)

### Item 4: FRI layer memory reduction

**Current**: Each `FriLayer` stores `evaluation: Vec<FieldElement<E>>` (the full folded evaluation vector) plus the Merkle tree. These evaluations are only used in `query_phase` for point lookups at ~30 positions.

For 20 FRI layers with geometrically shrinking sizes (2N, N, N/2, ..., 2), total evaluation storage is ~4N extension field elements = ~4 * 2^21 * 24 bytes = ~192MB for Bitwise.

**Proposed**: Don't store evaluations in FriLayer. Instead, during the commit phase, the folded evals are used transiently (build Merkle tree, then the eval vector is folded to the next level). For query phase, reconstruct needed values from Merkle tree openings.

Actually, looking more carefully: `query_phase` accesses `layer.evaluation[index ^ 1]` for the symmetric element at each layer. The Merkle tree doesn't store raw evaluations. So we do need to keep evaluations for query phase.

**Revised approach**: Instead of storing all evaluations, store only the (index, evaluation) pairs needed for queries. But queries come *after* commit phase, so we don't know indices yet.

**Alternative**: Accept the current memory usage but make the eval storage more compact:
- After building Merkle tree for a layer, swap the eval Vec from the working buffer instead of cloning (the current code does `FriLayer::new(&evals, ...)` which likely clones).

Let me check:

```rust
pub fn new(evaluation: &[FieldElement<F>], merkle_tree: ...) -> Self {
    Self { evaluation: evaluation.to_vec(), ... }
```

Yes, it clones. We can change to accept owned Vec:
```rust
pub fn new_owned(evaluation: Vec<FieldElement<F>>, merkle_tree: ...) -> Self
```

And in the commit loop, clone only what we need for the next fold:

```rust
let layer_evals = evals[..current_domain_size].to_vec();  // store for queries
fold_evaluations_in_place(&mut evals, ...);  // fold in place for next layer
fri_layer_list.push(FriLayer::new_owned(layer_evals, merkle_tree, ...));
```

Wait, but we're already folding in place. The issue is that after folding, the first half of `evals` contains the folded values and we've lost the pre-fold values. We store them via `FriLayer::new(&evals, ...)` *before* the next fold. So the current pattern is:

```
fold → store (clone) → fold → store (clone) → ...
```

We can change to:
```
fold → split evals into (stored, working) → fold working → ...
```

But this requires allocating a new Vec for each layer anyway. The clone is unavoidable if we need both the stored version and the working version. The savings would come from not storing beyond what query phase needs.

Since the query positions are determined *after* all FRI layers are committed, we can't prune. **Downgrade this item to "investigate further"** — the main savings are in items 1-3.

### Item 5: On-the-fly domain coset points

**Current**: `Domain` stores `lde_roots_of_unity_coset: Vec<FieldElement<F>>` with all 2N coset points (16MB for 2^21). This is used in:
- Boundary constraint evaluation (evaluator.rs)
- DEEP composition poly (prover.rs line 1483-1493)
- Decompose-and-extend (prover.rs line 1061-1063)
- Zerofier evaluations (transition.rs)
- OOD sampling (transcript)

**Proposed**: Replace `lde_roots_of_unity_coset` with a generator + offset, and compute points on-the-fly or in small batches. Since the coset is `{g * omega^i | i=0..2N}` where `g` is the offset and `omega` is the 2N-th root of unity:

```rust
pub struct Domain<F: IsFFTField> {
    // Remove: lde_roots_of_unity_coset: Vec<FieldElement<F>>,
    // Add:
    lde_primitive_root: FieldElement<F>,
    lde_domain_size: usize,
    // Keep: coset_offset, blowup_factor, etc.
}

impl Domain<F> {
    /// Compute coset point at index i: coset_offset * lde_primitive_root^i
    fn coset_point(&self, i: usize) -> FieldElement<F> { ... }
}
```

Most uses iterate sequentially, so we can use a running product (multiply by omega each step) instead of pow(i). The boundary and DEEP eval loops already iterate 0..N, making this straightforward.

The tradeoff: one extra multiplication per coset point access vs 16MB memory savings. For sequential access patterns this is essentially free.

**Files**: `domain.rs`, `evaluator.rs`, `prover.rs`, `transition.rs`

## Execution Order

1. **Item 3** (eval-form FRI) — addresses the user's primary concern, moderate complexity
2. **Item 2** (Rc Merkle trees) — large memory savings, straightforward
3. **Item 1** (LDE borrow) — largest memory savings, requires lifetime propagation
4. **Item 5** (domain on-the-fly) — moderate savings, low risk
5. **Item 4** (FRI layer memory) — investigate, likely limited gain

## Verification

After each item:
- `cargo test --release -p stark` (150 tests)
- `cargo test --release -p lambda-vm-prover` (282 tests)
- `cargo bench --bench vm_prover_benchmark` (local)
- Re-run remote benchmark to confirm no regression
