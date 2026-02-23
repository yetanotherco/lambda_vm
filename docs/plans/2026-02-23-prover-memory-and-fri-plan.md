# Prover Memory Reduction and FRI Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Eliminate redundant cloning (LDE columns, Merkle trees) and remove the coefficient-form round-trip in FRI, reducing peak memory by ~1GB and saving FFT passes.

**Architecture:** Sequential per-table proving already recomputes LDE per table. This plan makes those recomputations zero-copy by borrowing pool buffers, shares Merkle trees via Rc, and feeds FRI directly with evaluation-form data via `coset_lde` + `commit_phase_from_evaluations`.

**Tech Stack:** Rust, Goldilocks field (64-bit), Keccak256 Merkle trees, Rayon parallelism, Bowers FFT

---

## Task 1: Eval-form FRI commit (eliminate coefficient round-trip)

**Files:**
- Modify: `crypto/stark/src/fri/mod.rs:20-93` (add `commit_phase_from_evaluations`)
- Modify: `crypto/stark/src/prover.rs:1369-1385` (Round 4 FRI call site)

**Step 1: Add `commit_phase_from_evaluations` to `fri/mod.rs`**

This function accepts pre-computed, bit-reversed evaluations on the LDE coset, skipping the initial FFT that `commit_phase` does. The folding loop, Merkle tree building, and transcript interaction are identical.

Add after the existing `commit_phase` function (after line 93):

```rust
/// FRI commit phase starting from pre-computed evaluations (no initial FFT).
///
/// `evals` must be bit-reversed evaluations on the LDE coset of size `domain_size`.
/// This is used when the caller already has evaluations (e.g., from `coset_lde`),
/// eliminating the coefficient-form round-trip through `evaluate_offset_fft`.
pub fn commit_phase_from_evaluations<F: IsFFTField + IsSubFieldOf<E>, E: IsField>(
    number_layers: usize,
    mut evals: Vec<FieldElement<E>>,
    transcript: &mut impl IsStarkTranscript<E, F>,
    coset_offset: &FieldElement<F>,
    domain_size: usize,
) -> (
    FieldElement<E>,
    Vec<FriLayer<E, FriLayerMerkleTreeBackend<E>>>,
)
where
    FieldElement<F>: AsBytes + Sync + Send,
    FieldElement<E>: AsBytes + Sync + Send,
{
    // Inverse twiddle factors for evaluation-form folding
    let mut inv_twiddles = compute_coset_twiddles_inv(coset_offset, domain_size);

    let mut fri_layer_list = Vec::with_capacity(number_layers);
    let mut current_coset_offset = coset_offset.clone();
    let mut current_domain_size = domain_size;

    for _ in 1..number_layers {
        // <<<< Receive challenge 𝜁ₖ₋₁
        let zeta = transcript.sample_field_element();
        current_coset_offset = current_coset_offset.square();
        current_domain_size /= 2;

        // Fold evaluations in-place (no FFT needed)
        fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

        // Build Merkle tree from consecutive pairs
        let leaves: Vec<[FieldElement<E>; 2]> = evals
            .chunks_exact(2)
            .map(|chunk| [chunk[0].clone(), chunk[1].clone()])
            .collect();
        let merkle_tree = FriLayerMerkleTree::build(&leaves)
            .expect("FRI commit: Merkle tree construction must succeed");
        let root = merkle_tree.root;
        fri_layer_list.push(FriLayer::new(
            &evals,
            merkle_tree,
            current_coset_offset.clone().to_extension(),
            current_domain_size,
        ));

        // >>>> Send commitment: [pₖ]
        transcript.append_bytes(&root);

        // Update twiddles for next level
        update_twiddles_in_place(&mut inv_twiddles);
    }

    // <<<< Receive challenge: 𝜁ₙ₋₁
    let zeta = transcript.sample_field_element();

    // Final fold
    fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);

    let last_value = evals
        .first()
        .unwrap_or(&FieldElement::zero())
        .clone();

    // >>>> Send value: pₙ
    transcript.append_field_element(&last_value);

    (last_value, fri_layer_list)
}
```

**Step 2: Update Round 4 in `prover.rs` to use eval-form FRI**

Replace lines 1369-1385 in `round_4_compute_and_run_fri_on_the_deep_composition_polynomial`:

```rust
// OLD (remove):
// let deep_composition_poly = Polynomial::interpolate_offset_fft::<Field>(
//     &deep_evals, &domain.coset_offset,
// ).expect("iFFT should succeed");
// let domain_size = domain.lde_roots_of_unity_coset.len();
// let (fri_last_value, fri_layers) = fri::commit_phase::<Field, FieldExtension>(
//     domain.root_order as usize, deep_composition_poly, transcript,
//     &coset_offset, domain_size,
// );

// NEW:
// Extend deep evals from trace coset (N) to LDE coset (2N) via fused coset LDE,
// then fold evaluations directly — no coefficient-form intermediary.
let domain_size = domain.lde_roots_of_unity_coset.len();
let mut lde_evals = Polynomial::coset_lde::<Field>(&deep_evals, domain.blowup_factor, &coset_offset)
    .expect("coset LDE should succeed: domain size is a power of 2");
in_place_bit_reverse_permute(&mut lde_evals);

let (fri_last_value, fri_layers) = fri::commit_phase_from_evaluations::<Field, FieldExtension>(
    domain.root_order as usize,
    lde_evals,
    transcript,
    &coset_offset,
    domain_size,
);
```

Add import at top of `prover.rs` if not already present:
```rust
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
```
(Already imported at line 7.)

**Step 3: Run tests**

Run: `cargo test --release -p stark`
Expected: 150 passed, 0 failed

Run: `cargo test --release -p lambda-vm-prover`
Expected: 282 passed, 0 failed

**Step 4: Commit**

```bash
git add crypto/stark/src/fri/mod.rs crypto/stark/src/prover.rs
git commit -m "Eval-form FRI: feed coset LDE evaluations directly, skip coefficient round-trip"
```

---

## Task 2: Share Merkle trees via Rc

**Files:**
- Modify: `crypto/stark/src/prover.rs` (Round1CommitmentData, Round1Metadata, reconstruct_round1, Phase A, Phase C, commit_*_lightweight functions)

**Step 1: Add Rc import and update structs**

At top of `prover.rs`, add:
```rust
use std::rc::Rc;
```

Change `Round1CommitmentData`:
```rust
pub struct Round1CommitmentData<F> where F: IsField, FieldElement<F>: AsBytes {
    pub(crate) lde_trace_merkle_tree: Rc<BatchedMerkleTree<F>>,
    pub(crate) lde_trace_merkle_root: Commitment,
    pub(crate) precomputed_merkle_tree: Option<Rc<BatchedMerkleTree<F>>>,
    pub(crate) precomputed_merkle_root: Option<Commitment>,
    pub(crate) num_precomputed_cols: usize,
}
```

Change `Round1Metadata`:
```rust
pub struct Round1Metadata<Field, FieldExtension> where ... {
    main_merkle_tree: Rc<BatchedMerkleTree<Field>>,
    main_merkle_root: Commitment,
    precomputed_merkle_tree: Option<Rc<BatchedMerkleTree<Field>>>,
    precomputed_merkle_root: Option<Commitment>,
    num_precomputed_cols: usize,
    aux_merkle_tree: Option<Rc<BatchedMerkleTree<FieldExtension>>>,
    aux_merkle_root: Option<Commitment>,
    rap_challenges: Vec<FieldElement<FieldExtension>>,
    bus_public_inputs: Option<BusPublicInputs<FieldExtension>>,
}
```

**Step 2: Update Phase A to wrap trees in Rc**

In the `main_commits` tuple and all creation sites, wrap Merkle trees:

Where `commit_columns_bit_reversed` or `batch_commit_main` returns `(tree, root)`:
```rust
main_commits.push((Rc::new(tree), root, precomputed_tree.map(Rc::new), precomputed_root, num_precomputed));
```

Similarly in Phase C aux tree creation:
```rust
(Some(Rc::new(tree)), Some(root))
```

**Step 3: Update `reconstruct_round1` to Rc::clone**

Replace `.clone()` on Merkle trees with `Rc::clone()`:
```rust
let main = Round1CommitmentData::<Field> {
    lde_trace_merkle_tree: Rc::clone(&metadata.main_merkle_tree),
    lde_trace_merkle_root: metadata.main_merkle_root,
    precomputed_merkle_tree: metadata.precomputed_merkle_tree.as_ref().map(Rc::clone),
    precomputed_merkle_root: metadata.precomputed_merkle_root,
    num_precomputed_cols: metadata.num_precomputed_cols,
};
// ... aux similarly
```

**Step 4: Update `open_trace_polys_*` to accept &MerkleTree**

The `open_*` functions take `&BatchedMerkleTree<F>`. Since `Rc<T>` derefs to `&T`, call sites like `&round_1_result.main.lde_trace_merkle_tree` automatically deref. No signature changes needed.

**Step 5: Update `commit_main_trace_lightweight` and `commit_preprocessed_trace_lightweight`**

These functions create the Merkle tree. Their return types must change to include `Rc`:
```rust
// Return type changes from (BatchedMerkleTree<Field>, Commitment, ...) to:
(Rc<BatchedMerkleTree<Field>>, Commitment, ...)
```

Or simpler: keep them returning raw trees and wrap in `Rc` at the call site in Phase A.

**Step 6: Run tests**

Run: `cargo test --release -p stark`
Expected: 150 passed

Run: `cargo test --release -p lambda-vm-prover`
Expected: 282 passed

**Step 7: Commit**

```bash
git add crypto/stark/src/prover.rs
git commit -m "Share Merkle trees via Rc to eliminate deep clones in reconstruct_round1"
```

---

## Task 3: LDETraceTable borrows pool buffers

**Files:**
- Modify: `crypto/stark/src/trace.rs:263-370` (LDETraceTable struct and methods)
- Modify: `crypto/stark/src/prover.rs` (Round1, reconstruct_round1, prove_rounds_2_to_4)
- Modify: `crypto/stark/src/constraints/evaluator.rs` (uses LDETraceTable)
- Modify: `crypto/stark/src/frame.rs` (uses LDETraceTable)
- Modify: `crypto/stark/src/debug.rs` (uses LDETraceTable)

**Step 1: Create borrowed variant of LDETraceTable**

Add a new struct alongside the existing one (keep the owned variant for `from_columns` which is used in debug.rs and tests):

```rust
/// Borrowed variant of LDETraceTable — references pool buffers without cloning.
pub struct LDETraceTableRef<'a, F, E>
where
    E: IsField,
    F: IsSubFieldOf<E> + IsField,
{
    main_columns: &'a [Vec<FieldElement<F>>],
    aux_columns: &'a [Vec<FieldElement<E>>],
    num_main: usize,
    num_aux: usize,
    pub(crate) lde_step_size: usize,
    pub(crate) blowup_factor: usize,
}
```

Implement the same `get_main`, `get_aux`, `num_main_cols`, `num_aux_cols`, `num_rows`, `gather_*` methods.

**Step 2: Create a trait for shared LDE access**

To avoid duplicating all code that uses LDETraceTable, create a trait:

```rust
pub trait LDETraceAccess<F: IsField, E: IsField> {
    fn get_main(&self, row: usize, col: usize) -> &FieldElement<F>;
    fn get_aux(&self, row: usize, col: usize) -> &FieldElement<E>;
    fn num_main_cols(&self) -> usize;
    fn num_aux_cols(&self) -> usize;
    fn num_rows(&self) -> usize;
    fn lde_step_size(&self) -> usize;
    fn blowup_factor(&self) -> usize;
    fn gather_main_row(&self, row_idx: usize) -> Vec<FieldElement<F>>;
    fn gather_aux_row(&self, row_idx: usize) -> Vec<FieldElement<E>>;
    fn gather_main_row_range(&self, row_idx: usize, col_start: usize, col_end: usize) -> Vec<FieldElement<F>>;
}
```

Implement for both `LDETraceTable` (owned) and `LDETraceTableRef` (borrowed).

**Alternative (simpler):** Instead of a trait, make `LDETraceTable` generic over storage using an enum:

```rust
enum ColumnStorage<'a, T> {
    Owned(Vec<Vec<T>>),
    Borrowed(&'a [Vec<T>], usize),  // slice + actual count
}
```

This avoids trait propagation entirely. Choose based on implementation complexity.

**Step 3: Update `from_columns_borrowed` to create zero-copy reference**

```rust
pub fn from_pool_ref(
    main_columns: &'a [Vec<FieldElement<F>>],
    aux_columns: &'a [Vec<FieldElement<E>>],
    num_main: usize,
    num_aux: usize,
    trace_step_size: usize,
    blowup_factor: usize,
) -> LDETraceTableRef<'a, F, E> { ... }
```

**Step 4: Update `reconstruct_round1` and `Round1`**

`Round1.lde_trace` changes from `LDETraceTable` to `LDETraceTableRef<'a, ...>`. `Round1` gains a lifetime. `prove_rounds_2_to_4` borrows pools and creates `Round1<'_>`.

**Step 5: Run tests**

Run: `cargo test --release -p stark`
Expected: 150 passed

Run: `cargo test --release -p lambda-vm-prover`
Expected: 282 passed

**Step 6: Commit**

```bash
git add crypto/stark/src/trace.rs crypto/stark/src/prover.rs crypto/stark/src/constraints/evaluator.rs crypto/stark/src/frame.rs crypto/stark/src/debug.rs
git commit -m "Zero-copy LDETraceTable: borrow pool buffers instead of cloning in reconstruct_round1"
```

---

## Task 4: On-the-fly domain coset points

**Files:**
- Modify: `crypto/stark/src/domain.rs` (Domain struct)
- Modify: `crypto/stark/src/prover.rs` (DEEP composition, decompose_and_extend, OOD sampling)
- Modify: `crypto/stark/src/constraints/evaluator.rs` (boundary eval)
- Modify: `crypto/stark/src/constraints/transition.rs` (zerofier eval)
- Modify: `crypto/stark/src/verifier.rs` (uses domain coset points)

**Step 1: Add generator fields to Domain**

Keep `lde_roots_of_unity_coset` for now but add:
```rust
pub(crate) lde_primitive_root: FieldElement<F>,
pub(crate) lde_domain_size: usize,
```

Populate in Domain constructor:
```rust
let lde_primitive_root = F::get_primitive_root_of_unity(lde_root_order as u64).unwrap();
let lde_domain_size = trace_length * blowup_factor;
```

**Step 2: Add `coset_point_iter` helper**

```rust
impl<F: IsFFTField> Domain<F> {
    /// Iterator that yields coset points g*omega^i for i=0..lde_domain_size.
    /// Uses running product (one mul per step) instead of pow().
    pub fn coset_point_iter(&self) -> impl Iterator<Item = FieldElement<F>> + '_ {
        let mut current = self.coset_offset.clone();
        (0..self.lde_domain_size).map(move |_| {
            let point = current.clone();
            current = &current * &self.lde_primitive_root;
            point
        })
    }
}
```

**Step 3: Replace Vec access sites one by one**

For each site that accesses `domain.lde_roots_of_unity_coset[i]`:
- If sequential iteration: use `coset_point_iter()`
- If indexed access at `i * bf` stride: precompute the strided subset (N points instead of 2N)
- If random access (e.g., OOD sampling): keep Vec or compute on demand

Start with `prover.rs` DEEP composition (lines 1482-1493) which is the biggest user. The boundary evaluator and zerofier can follow.

**Step 4: Once all sites are migrated, remove `lde_roots_of_unity_coset` from Domain**

**Step 5: Run tests**

Run: `cargo test --release -p stark && cargo test --release -p lambda-vm-prover`

**Step 6: Commit**

```bash
git add crypto/stark/src/domain.rs crypto/stark/src/prover.rs crypto/stark/src/constraints/evaluator.rs crypto/stark/src/constraints/transition.rs crypto/stark/src/verifier.rs
git commit -m "Compute domain coset points on-the-fly instead of storing 2N-element Vec"
```

---

## Task 5: FRI layer eval ownership transfer

**Files:**
- Modify: `crypto/stark/src/fri/fri_commitment.rs:26-38` (FriLayer::new)
- Modify: `crypto/stark/src/fri/mod.rs:47-76` (commit loop)

**Step 1: Add `new_owned` to FriLayer**

```rust
pub fn new_owned(
    evaluation: Vec<FieldElement<F>>,
    merkle_tree: MerkleTree<B>,
    coset_offset: FieldElement<F>,
    domain_size: usize,
) -> Self {
    Self { evaluation, merkle_tree, coset_offset, domain_size }
}
```

**Step 2: Update commit loop to clone-then-fold**

In `commit_phase_from_evaluations`, change the pattern from:
```
fold → store(clone) → ...
```
to:
```
fold → clone-for-storage → store(owned) → ...
```

Since we fold in-place and then store, we need to save the pre-fold state:
```rust
fold_evaluations_in_place(&mut evals, &zeta, &inv_twiddles);
// evals now contains folded values — store them
let layer_evals = evals[..current_domain_size].to_vec();
// ... build merkle tree from layer_evals ...
fri_layer_list.push(FriLayer::new_owned(layer_evals, merkle_tree, ...));
```

Wait — this is the same allocation. The real savings: currently `FriLayer::new` does `evaluation.to_vec()` which clones the slice. If we pass the Vec directly, we avoid one clone. But we still need to preserve `evals` for the next fold.

The actual fix: after building the Merkle tree and before the next fold, the current code stores via `FriLayer::new(&evals, ...)` (clone). Then folds in place. We can instead: split evals into stored + working, but that still allocates.

This is a minor optimization. Save pre-fold evals only for the first (largest) layer where it matters most, or skip this task if gains are <1%.

**Step 3: Run tests and commit**

Run: `cargo test --release -p stark && cargo test --release -p lambda-vm-prover`

```bash
git add crypto/stark/src/fri/fri_commitment.rs crypto/stark/src/fri/mod.rs
git commit -m "FRI layer: transfer eval ownership to avoid redundant clone"
```

---

## Verification

After all tasks:
- `cargo test --release -p stark` (150 tests)
- `cargo test --release -p lambda-vm-prover` (282 tests)
- `cargo bench --bench vm_prover_benchmark` (local baseline: prove 25.7s)
- Remote benchmark (baseline: ~55s)
- Profile with `samply` to confirm FFT/allocation category shifts
