# Scaling Improvements: Unified Table Sizing + MMCS + Shared FRI

## Problem

Lambda VM's proving time scales with slope ~5.1s per million steps vs SP1's
~2.6s/M — roughly 2x worse. Three structural inefficiencies contribute:

1. **Twiddle waste from diverse table sizes**: Tables at 3 different max_rows
   (2^19, 2^20, 2^21) require 3 distinct twiddle sets, wasting 272 MB in
   redundant copies and preventing domain sharing.

2. **Per-table independent commitments**: Each table gets its own Merkle tree
   for main trace, aux trace, and composition polynomial — ~30 separate Keccak
   trees per proof. Each tree's construction and query opening is independent
   overhead.

3. **Per-table independent FRI**: Each table runs its own 19-layer FRI with its
   own Merkle trees. For 12+ tables, that's ~200 FRI-layer trees and 219
   queries repeated per table.

## Goals

- Reduce the per-step proving slope by ~1.5-2x
- Uniform table domain size to eliminate twiddle diversity
- Batch all table commitments into shared Merkle trees (MMCS)
- Share one FRI instance across all tables
- Reduce proof size (fewer roots, fewer query openings)

## Non-Goals

- Changing the hash function (Keccak stays)
- Changing the field (Goldilocks stays)
- Execution sharding or recursion
- GKR-based LogUp (stay with committed aux columns)

---

## Item 1: Unified Table Cap + Twiddle Dedup + Quick Wins

### Uniform max_rows = 2^20

Cap all tables at 2^20 rows. Currently:
- 2^19: CPU (74 cols), MEMW (49), MEMW_A (30), DVRM (34)
- 2^20: MUL (26), SHIFT (26), BITWISE (21)
- 2^21: LT (15), LOAD (18), BRANCH (14), MEMW_R (10)

Tables at 2^21 produce 2x more chunks (each 2^20) but each chunk's FFT is 2x
cheaper and all tables share one twiddle/domain. Tables at 2^19 are unchanged
(they already chunk below 2^20). The `MaxRowsConfig` formula
`max_rows = (127 × 2^19) / eff_width` stays, but we add
`.min(1 << 20)` to cap.

Net effect: every chunk has domain size 2^20, LDE size 2^21. One shared
`LdeTwiddles` of 32 MB replaces 384 MB of redundant copies.

### Twiddle deduplication

In `multi_prove`'s pre-pass, deduplicate by domain size: build one
`Arc<LdeTwiddles>` per distinct `(trace_order, lde_order)` pair and clone
the `Arc` for same-size tables. With uniform cap, this collapses to 1 shared
set.

### Parallelize FRI fold

`fold_evaluations_in_place` in `fri/fri_functions.rs` is sequential. Replace
with `par_chunks_mut` (matching Plonky3's approach). For a 2^20 domain, this
parallelizes 2^19 fold operations.

### Eliminate per-row Vec alloc in commit

`commit_columns_bit_reversed` allocates a `Vec<FieldElement>` per LDE row
(262K allocs per tree). Replace with `map_init` thread-local row buffer,
matching the existing `commit_composition_polynomial` pattern.

### Files changed

| File | Change |
|------|--------|
| `prover/src/tables/mod.rs` | Add `.min(1 << 20)` cap to max_rows |
| `crypto/stark/src/prover.rs` | Deduplicate twiddles, fix commit alloc |
| `crypto/stark/src/fri/fri_functions.rs` | Parallelize fold |

---

## Item 2: MMCS Batched Commitment

### Overview

Replace per-table independent Merkle trees with Plonky3-style batched
commitments. All tables' main LDE columns go into one Merkle tree. All aux
columns into another. One composition tree for all tables.

### How Plonky3 MMCS works

Multiple matrices of different heights share one tree via "jagged"
construction:

1. All tallest-height matrices have their rows concatenated and hashed together
   into one leaf per row index.
2. When building upward, shorter matrices are "injected" at the tree level
   matching their height — an extra compression step merges the injected
   row hashes with the existing internal nodes.
3. Opening at a global index: for a matrix shorter than the tallest, the
   index is right-shifted by `log2(max_height) - log2(matrix_height)`.

With uniform cap (Item 1), all table chunks have the same height (2^20).
This simplifies MMCS to just concatenating all columns from all tables into
one wide row per leaf, with no jagged injection needed.

### Adaptation for lambda_vm

**Phase A — one batched main commitment:**

Currently each table commits independently:
```
for each table:
    extract main columns → LDE → commit_columns_bit_reversed → root
    append root to transcript
```

Replace with:
```
for each table:
    extract main columns → LDE → collect into all_main_columns[]
commit_columns_bit_reversed(all_main_columns) → one root
append one root to transcript
```

The single Merkle tree has leaves:
`leaf[i] = Keccak256(table0_col0[br_i] || table0_col1[br_i] || ... || tableN_colM[br_i])`

This is exactly how `BatchedMerkleTreeBackend` already works — it hashes a
`Vec<FieldElement>` per row. The only change is that the Vec contains columns
from ALL tables, not just one.

**Phase C — one batched aux commitment:**

Same approach: all aux columns from all tables into one tree.

**Round 2 — one batched composition commitment:**

All tables' composition polynomial parts committed together.

**Proof format change:**

```rust
// Before: MultiProof { proofs: Vec<StarkProof> }
// After:
pub struct BatchedProof<F, E, PI> {
    // Shared commitments
    pub main_merkle_root: Commitment,
    pub aux_merkle_root: Option<Commitment>,
    pub composition_merkle_root: Commitment,

    // Per-table data (OOD evaluations, public inputs)
    pub table_data: Vec<TableProofData<F, E, PI>>,

    // Shared FRI (Item 3)
    pub fri_layers_merkle_roots: Vec<Commitment>,
    pub fri_last_value: FieldElement<E>,

    // Shared queries
    pub query_openings: Vec<BatchedQueryOpening<F, E>>,
    pub nonce: Option<u64>,
}

pub struct TableProofData<F, E, PI> {
    pub trace_length: usize,
    pub trace_ood_evaluations: Table<E>,
    pub composition_poly_parts_ood_evaluation: Vec<FieldElement<E>>,
    pub bus_public_inputs: Option<BusPublicInputs<E>>,
    pub public_inputs: PI,
}
```

**Query opening format:**

Each query opens ONE row from the shared main tree, ONE from the shared aux
tree, and ONE from the shared composition tree — instead of opening from
each table's separate trees. The verifier extracts per-table columns from
the opened row by column offset.

### Column offset tracking

Each table's columns start at a known offset in the batched commitment.
The prover and verifier agree on the column layout:

```rust
struct BatchedLayout {
    // main_col_offset[table_idx] = starting column in the batched main tree
    main_col_offsets: Vec<usize>,
    // aux_col_offset[table_idx] = starting column in the batched aux tree
    aux_col_offsets: Vec<usize>,
    // comp_col_offset[table_idx] = starting column in the batched comp tree
    comp_col_offsets: Vec<usize>,
}
```

### Files changed

| File | Change |
|------|--------|
| `crypto/stark/src/prover.rs` | Batched commit in Phase A/C, batched comp in Round 2, batched openings in Round 4 |
| `crypto/stark/src/verifier.rs` | Verify against batched trees, extract per-table columns from opened rows |
| `crypto/stark/src/proof/stark.rs` | New `BatchedProof` struct |
| `crypto/stark/src/config.rs` | No change (same `BatchedMerkleTreeBackend`) |

---

## Item 3: Shared FRI

### Overview

Replace per-table independent FRI with one shared FRI instance. All tables'
deep composition polynomials are randomly batched into one polynomial, which
is then FRI-committed and queried once.

### How Plonky3 does it

In `TwoAdicFriPcs::open()`:

1. For each table, compute the DEEP quotient polynomial at each query point
2. Batch all tables' quotients into one evaluation vector using random `alpha`
   powers
3. Run ONE `commit_phase_from_evaluations` on the batched vector
4. Query ONE set of indices; open from the shared MMCS trees

With uniform table sizing (Item 1), all tables have the same domain. The
batching is a simple weighted sum:

```
batched[i] = Σ_tables alpha^t * deep_quotient_table_t[i]
```

### Adaptation for lambda_vm

**Round 4 changes:**

Currently:
```
for each table:
    compute deep composition poly evaluations
    iFFT → FFT to extend to LDE
    commit_phase_from_evaluations (19 FRI layers)
    query_phase (219 queries)
```

Replace with:
```
for each table:
    compute deep composition poly evaluations → deep_evals[table]

// Batch all tables with alpha powers
alpha = transcript.sample()
batched_evals = Σ alpha^t * deep_evals[t]

// One FRI
commit_phase_from_evaluations(batched_evals)
// One query set
for each iota in iotas:
    open from shared main tree, shared aux tree, shared comp tree
    open from FRI layer trees
```

**Proof size reduction:**

| Component | Before (per table) | After (shared) |
|-----------|-------------------|----------------|
| FRI layer roots | 19 × N_tables | 19 |
| FRI decommitments | 219 × N_tables | 219 |
| Trace openings | 219 × 2 proofs × N_tables | 219 × 2 proofs (one wide row) |
| Composition openings | 219 × 1 proof × N_tables | 219 × 1 proof |

### Verifier changes

The verifier reconstructs the batched deep quotient from the opened values
and verifies against the shared FRI. It also extracts per-table OOD
evaluations to verify constraint satisfaction independently per table.

### Files changed

| File | Change |
|------|--------|
| `crypto/stark/src/prover.rs` | Batched deep quotient, shared FRI commit/query |
| `crypto/stark/src/verifier.rs` | Verify shared FRI, reconstruct per-table quotients |
| `crypto/stark/src/proof/stark.rs` | Shared FRI fields in `BatchedProof` |
| `crypto/stark/src/fri/mod.rs` | No structural change (same `commit_phase_from_evaluations`) |

---

## Expected Impact

### Item 1 (quick wins)

| Metric | Before | After |
|--------|--------|-------|
| Twiddle memory | 384 MB | 32 MB |
| FRI fold parallelism | sequential | par_chunks_mut |
| Commit allocs | 262K heap allocs/tree | 0 (thread-local buf) |
| Table chunk sizes | 3 different | 1 uniform |

### Item 2 (MMCS)

| Metric | Before | After |
|--------|--------|-------|
| Main Merkle trees | ~12 | 1 |
| Aux Merkle trees | ~10 | 1 |
| Composition trees | ~10 | 1 |
| Total commit-phase trees | ~32 | 3 |
| Keccak hash calls (commit) | ~32 × 2N | ~3 × 2N (wider rows) |

### Item 3 (shared FRI)

| Metric | Before | After |
|--------|--------|-------|
| FRI instances | ~12 | 1 |
| FRI layer trees | ~12 × 19 = 228 | 19 |
| Query openings | 219 × 12 = 2,628 | 219 |
| Proof size (FRI portion) | ~12x | 1x |

### Combined slope estimate

With all three items: the per-step cost drops by reducing the multiplier on
Merkle hashing (~12x → 1x for FRI, ~32x → 3x for commits) and FRI work.
Conservative estimate: slope drops from ~5.1s/M to ~3-3.5s/M, approaching
SP1's 2.6s/M.

---

## Implementation Order

1. **Item 1** first — prerequisite for Item 2 (uniform sizing simplifies MMCS)
2. **Item 2** next — prerequisite for Item 3 (shared trees enable shared FRI)
3. **Item 3** last — builds on Item 2's shared commitment infrastructure

Each item is independently benchmarkable. Item 1 alone provides measurable
improvement. Items 2+3 together provide the largest structural gain.

## Testing Strategy

- Each item must pass all existing stark crate tests (121 tests)
- Items 2+3 change the proof format → verifier tests must be updated
- Benchmark after each item at 1M, 4M, 8M steps to track slope improvement
- Compare proof sizes before/after Items 2+3
