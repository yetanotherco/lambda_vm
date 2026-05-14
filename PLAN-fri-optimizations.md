# Plan: FRI Early Stopping + Higher Folding Factor

## Goal

Add two FRI optimizations to the current evaluation-based FRI on main:

1. **Early stopping** (`fri_last_layer_degree_bound`): Stop folding when the polynomial degree reaches a configurable bound, send remaining coefficients directly. Reduces the number of committed FRI layers.
2. **Higher folding factor** (`fri_folding_factor`): Group multiple binary folds into one committed layer. Each Merkle leaf contains `folding_factor` evaluations instead of 2. Reduces the number of Merkle trees and proof size.

## Current State (main)

- FRI is evaluation-based: `commit_phase_from_evaluations` takes pre-computed bit-reversed evaluations and folds in-place using `fold_evaluations_in_place`
- Each layer: sample zeta, fold evaluations (halve the vector), build Merkle tree from pairs `[eval[2j], eval[2j+1]]`, commit
- `fri_last_value` is a single `FieldElement<E>` (constant polynomial after all folds)
- `FriDecommitment` has `layers_evaluations_sym: Vec<FieldElement<F>>` (one symmetric element per layer)
- Query phase: `index ^= 1` for symmetric element, `index >>= 1` to descend
- Verifier: `zetas` collected one per Merkle root, plus one final challenge. Fold formula: `v = (v + sym) + inv_tw * zeta * (v - sym)`

## Changes Required

### 1. `ProofOptions` (`proof/options.rs`)

Add two fields:
```rust
pub fri_last_layer_degree_bound: usize,  // default 0 (fold to constant)
pub fri_folding_factor: usize,           // default 2 (one fold per layer)
```

Add `validate()` method:
- `fri_folding_factor` must be power of 2 >= 2
- `fri_last_layer_degree_bound` must be 0 or `(bound+1)` must be power of 2

Update `default_test_options()` to include defaults.
Update `GoldilocksCubicProofOptions` to use `fri_last_layer_degree_bound: 7, fri_folding_factor: 4`.

Add error variants `InvalidFoldingFactor(usize)` and `InvalidDegreeBound(usize)`.

### 2. `StarkProof` (`proof/stark.rs`)

Change `fri_last_value: FieldElement<E>` → `fri_last_value: Vec<FieldElement<E>>`.

This is because with early stopping, the last polynomial has degree > 0, so we send all its coefficients (not just a single constant).

### 3. `FriDecommitment` (`fri/fri_decommit.rs`)

Change `layers_evaluations_sym: Vec<FieldElement<F>>` → `layers_evaluations_sym: Vec<Vec<FieldElement<F>>>`.

With `folding_factor > 2`, each Merkle leaf has `folding_factor` evaluations. The query reveals one (the verifier reconstructs it), and the prover sends the other `folding_factor - 1` as siblings.

### 4. FRI Commit Phase (`fri/mod.rs`)

Update `commit_phase_from_evaluations` signature to accept `folding_factor` and `last_layer_degree_bound`.

Logic changes:
- `log_f = folding_factor.trailing_zeros()`
- `last_poly_log = if bound == 0 { 0 } else { (bound+1).trailing_zeros() }`
- `folds_to_perform = number_layers - last_poly_log`
- Initial fold: sample zeta, fold once (no commitment)
- For each committed layer: build Merkle from chunks of `folding_factor`, commit, then do `log_f` folds (one challenge each)
- After all committed layers: the remaining evaluations ARE the last polynomial. Convert to coefficients via iFFT, send all coefficients.
- Return `(Vec<FieldElement<E>>, Vec<FriLayer>)` instead of `(FieldElement<E>, Vec<FriLayer>)`

Key detail: Merkle leaves change from `[eval[2j], eval[2j+1]]` (pairs) to `eval[leaf_idx * ff .. (leaf_idx+1) * ff]` (chunks of `folding_factor`).

### 5. FRI Query Phase (`fri/mod.rs`)

Update `query_phase` to accept `folding_factor`:
- `leaf_index = index >> log_f`
- `known_pos = index % folding_factor`
- Collect all sibling evaluations EXCEPT `known_pos`
- Merkle proof at `leaf_index`
- `index = leaf_index` (instead of `index >>= 1`)

### 6. Verifier (`verifier.rs`)

#### Challenge replay (`step_1_replay_rounds_and_recover_challenges`)
Current: one challenge per Merkle root, plus one final.
New: initial challenge, then for each Merkle root: append root, sample `log_f` challenges. Then append last polynomial coefficients.

#### FRI verification (`verify_query_and_sym_openings`)
Current: single symmetric element per layer, `v = (v + sym) + inv_tw * zeta * (v - sym)`, `index >>= 1`.
New: `folding_factor` elements per leaf, `log_f` binary folds per committed layer. For each committed layer:
1. Verify Merkle opening at `leaf_index`
2. Reconstruct the full leaf from `known_pos` (verifier's computed value) + `folding_factor - 1` siblings
3. Perform `log_f` binary folds using the `log_f` challenges for this layer
4. Update `index = leaf_index`

After all layers: verify that the reconstructed value matches evaluation of the last polynomial (from `fri_last_value` coefficients) at the appropriate point.

#### `verify_fri_layer_openings`
Needs to handle leaves with `folding_factor` elements instead of 2.

### 7. Prover integration (`prover.rs`)

- Pass `folding_factor` and `last_layer_degree_bound` to `commit_phase_from_evaluations`
- Pass `folding_factor` to `query_phase`
- Update proof construction: `fri_last_value` is now a `Vec`
- Add validation call `proof_options.validate()` early in `multi_prove`

### 8. Tests

- Update all tests that construct `ProofOptions` to include new fields
- Update all tests that reference `fri_last_value` (it's now a Vec)
- Update all tests that reference `FriDecommitment.layers_evaluations_sym` (now Vec<Vec>)
- Add new tests:
  - `validate()` rejects invalid folding factor / degree bound
  - Prove/verify roundtrip with `folding_factor=4, last_layer_degree_bound=7`
  - Prove/verify roundtrip with `folding_factor=2, last_layer_degree_bound=0` (backwards compat)

## Execution Order

1. `ProofOptions` — add fields, validation, errors (standalone, nothing breaks)
2. `StarkProof` — change `fri_last_value` type (breaks compilation everywhere — fix cascading)
3. `FriDecommitment` — change `layers_evaluations_sym` type (more cascading fixes)
4. `fri/mod.rs` — update `commit_phase_from_evaluations` and `query_phase`
5. `verifier.rs` — update challenge replay and FRI verification
6. `prover.rs` — wire everything together
7. Tests — fix and add new ones
8. Lint + test

## Risk

The evaluation-based FRI on main folds evaluations in-place. With early stopping, after the last committed layer we have a shortened evaluation vector that represents a polynomial of degree <= `last_layer_degree_bound`. We need to convert this to coefficients (iFFT) to send them. This requires importing the iFFT machinery into the FRI module.

With `folding_factor > 2`, the twiddle update changes: instead of `update_twiddles_in_place` once per fold, we do it `log_f` times per committed layer.
