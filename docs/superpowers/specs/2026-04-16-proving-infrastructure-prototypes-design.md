# Proving Infrastructure Prototypes

## Goal

Reduce proving time by optimizing the proving infrastructure — commitment batching, FRI unification, and Merkle tree construction — without changing the AIR design (table columns, constraints, or bus interactions).

## Context

Lambda VM's prover generates one Merkle tree and one FRI proof per table. With 18+ sub-proofs for a typical program, the per-table overhead dominates: ~54 Merkle trees, ~3,942 FRI query sets (219 queries × 18 tables), and forked transcripts per table.

Plonky3 (used by SP1, OpenVM) batches all tables into a single commitment and runs one FRI proof. This reduces Merkle trees to ~3 and FRI queries to ~100-219 (constant, not multiplied by table count).

The changes live on a branch off main. Each commit replaces the current approach — no feature flags, no duplication. The baseline is main; benchmarks measure before/after on the branch.

## Architecture

Four sequential commits, each leaving the prover functional (tests pass). A fifth item (blowup tuning) is configuration-only.

```
main (baseline)
  └─ commit 1: skip empty tables
       └─ commit 2: Merkle leaf hashing optimization
            └─ commit 3: batched commitment + uniform max_rows
                 └─ commit 4: single FRI proof
```

## Commit 1: Skip Empty Tables

### Problem

Tables with zero operations (MUL, DVRM, SHIFT, LOAD, MEMW, MEMW_A for Fibonacci) generate 4-row dummy traces with full commitment + FRI proof overhead.

### Change

In `trace_builder.rs`, `chunk_and_generate` returns empty `Vec` when `ops.is_empty()` for optional tables.

Optional tables (can have 0 chunks): MUL, DVRM, SHIFT, LOAD, MEMW, MEMW_A, LT.

Required tables (always >= 1 chunk): CPU, MEMW_R, BRANCH, BITWISE, DECODE, REGISTER, HALT, COMMIT, PAGE.

In `lib.rs`:
- `TableCounts::validate()` accepts 0 for optional tables.
- `VmAirs::new()` conditionally creates AIRs only for tables with count > 0.
- `VmAirs::air_trace_pairs()` and `air_refs()` skip tables with count 0.
- `verify_with_options()` adjusts expected_proof_count to exclude zero-count tables.

### Verifier impact

The verifier reconstructs AIRs from `TableCounts`. Zero-count tables produce no AIR, no trace pair, no sub-proof. The `expected_proof_count` formula changes from `table_counts.total() + 5 + page_configs.len()` to only counting non-zero entries plus fixed tables.

## Commit 2: Merkle Leaf Hashing Optimization

### Problem

`commit_columns_bit_reversed` allocates a `Vec<FieldElement>` per leaf to collect column values before hashing. For CPU LDE (2^20 rows × 74 cols), that's 2^20 Vec allocations of 592 bytes each. The Vecs are immediately dropped after hashing.

### Change

Replace the per-leaf Vec collection with incremental hashing directly from columns:

```rust
fn commit_columns_bit_reversed<E>(
    columns: &[Vec<FieldElement<E>>],
) -> Option<(BatchedMerkleTree<E>, Commitment)>
where
    FieldElement<E>: AsBytes + Sync + Send,
    E: IsField,
{
    // ... validation ...

    let hashed_leaves: Vec<Commitment> = iter
        .map(|row_idx| {
            let br_idx = reverse_index(row_idx, num_rows as u64);
            let mut hasher = Keccak256::new();
            for col_idx in 0..num_cols {
                hasher.update(columns[col_idx][br_idx].as_bytes());
            }
            hasher.finalize().into()
        })
        .collect();

    BatchedMerkleTree::<E>::build_from_hashed_leaves(hashed_leaves)
        .map(|tree| { let root = tree.root; (tree, root) })
}
```

This eliminates 2^20 Vec allocations per tree construction. The Keccak hasher absorbs bytes incrementally with the same result.

The `hash_data` method on `FieldElementVectorBackend` remains for other use cases, but the hot-path commitment code bypasses it.

### Verifier impact

None. The Merkle tree structure and hash values are identical — same bytes hashed in the same order.

## Commit 3: Batched Commitment + Uniform Max Rows

### Problem

Each table has its own Merkle tree. With 18+ tables, that's 18+ trees for main trace alone, plus 18+ for aux, plus 18+ for composition = ~54 trees. Each tree requires O(N) hashing.

Different tables have different max_rows (2^19 to 2^21), preventing simple batching.

### Change

#### Uniform max_rows

All tables use `max_rows = 2^20`. The `MaxRowsConfig` and `max_rows` module set every table to `1 << 20`. Tables that previously allowed 2^21 (LT, LOAD, BRANCH, MEMW_R) will generate more chunks when they exceed 2^20 rows, but all chunks have the same LDE height (2^21 with blowup=2).

This means for Fibonacci 1M, MEMW_R (~3M rows) generates 3 chunks of 2^20 instead of 2 chunks of 2^21. Total sub-AIR count increases slightly, but all are the same height.

#### Batched commitment

All main trace columns from all tables (chunks) are committed in ONE Merkle tree. Each leaf concatenates the columns of all tables at that row index:

```
leaf[i] = Hash(table0_col0[i] || table0_col1[i] || ... || tableN_colM[i])
```

A `CommitmentLayout` struct tracks which columns belong to which table:

```rust
struct CommitmentLayout {
    /// For each table/chunk: (air_index, start_column, num_columns)
    entries: Vec<(usize, usize, usize)>,
    /// Total columns across all tables
    total_columns: usize,
}
```

The commitment function takes ALL columns from ALL tables at once:

```rust
fn commit_all_tables(
    all_columns: &[Vec<FieldElement<F>>],  // total_columns × lde_size
    layout: &CommitmentLayout,
) -> (BatchedMerkleTree<F>, Commitment)
```

Same approach for aux traces (one tree for all aux) and composition polynomials (one tree for all quotients).

#### Unified transcript

Remove transcript forking. The single transcript observes:
1. Main commitment root (one)
2. Aux commitment root (one)
3. All bus_public_inputs (sequentially, by table index)
4. Composition commitment root (one)

Table ordering in the transcript is deterministic (by table index in `air_trace_pairs` order).

#### Multi_prove restructure

The current `multi_prove` flow:

```
Phase A: for each table → LDE → commit → cache LDE
Phase B: sample LogUp challenges
Phase C: for each table → build aux → LDE → commit → cache aux LDE
Phase D: for each table → reconstruct → constraints → composition → DEEP → FRI
```

New flow:

```
Phase A: for each table → LDE → cache LDE columns
         commit ALL main columns together → 1 tree, 1 root
Phase B: sample LogUp challenges
Phase C: for each table → build aux → cache aux LDE columns
         commit ALL aux columns together → 1 tree, 1 root
Phase D: for each table → evaluate constraints → composition poly → cache comp columns
         commit ALL composition columns together → 1 tree, 1 root
         (Single FRI in commit 4; per-table FRI temporarily in this commit)
```

The key change: LDE computation remains per-table (parallel, using pool sets). But commitment is a single call after all tables' LDE columns are collected.

#### Opening structure

When FRI queries need to open a specific table's columns, they open the leaf of the combined tree and extract the relevant column range using `CommitmentLayout`. The Merkle path is shared — opening column 5 of CPU and column 3 of MEMW at the same row uses one path, not two.

The `PolynomialOpenings` and `DeepPolynomialOpenings` structs change to reference columns by (table_index, local_column_index) instead of storing per-table Merkle trees.

### Verifier impact

The verifier reconstructs `CommitmentLayout` from the AIR list (deterministic). It receives 3 Merkle roots (main, aux, composition) instead of 3×N. Opening verification uses the same layout to check that opened values at queried positions match the committed tree.

`MultiProof` structure changes:
- Before: `proofs: Vec<StarkProof>` (one per table, each with own Merkle trees)
- After: shared commitments + per-table openings referencing the shared trees

## Commit 4: Single FRI Proof

### Problem

Even with batched commitments (commit 3), FRI is still per-table: each table has its own DEEP composition polynomial and FRI folding. With 18+ tables, that's 18+ FRI proofs, each with 219 queries.

### Change

#### Unified DEEP composition

After evaluating constraints and computing composition polynomials per-table (these are AIR-specific and must remain per-table), combine ALL polynomials into a single DEEP composition:

```
1. Sample OOD point z (ONE point, shared)
2. For each table i:
     Evaluate all columns at z via barycentric interpolation on cached LDE
     Evaluate composition parts at z
3. Sample batching challenge γ
4. Compute unified DEEP polynomial:
     DEEP(x) = Σ_i γ^i · DEEP_i(x)
   where DEEP_i(x) = Σ_j α^j · (f_{i,j}(x) - f_{i,j}(z)) / (x - z)
5. Run ONE FRI on DEEP(x)
```

All tables share the same `z` point and the same FRI folding sequence.

#### FRI query savings

Before: 219 queries × N tables × 3 trees per table
After: 219 queries × 3 trees (main, aux, composition)

Each query opens one leaf per tree. That leaf contains columns from ALL tables. The Merkle path is depth ~21. Total hash verifications: 219 × 3 × 21 = ~13,800 (vs ~236,000 before).

#### Transcript flow

```
1. Observe main commitment root
2. Observe public values per table
3. Sample LogUp challenges (α, β)
4. Observe aux commitment root
5. Observe bus_public_inputs per table
6. Sample constraint batching challenge λ
7. For each table: evaluate constraints, compute quotient
8. Observe composition commitment root
9. Sample OOD point z
10. For each table: compute openings at z, append to transcript
11. Sample DEEP batching challenge γ
12. Compute unified DEEP polynomial
13. FRI commit phase (fold, commit FRI layers)
14. Sample FRI query indices
15. FRI query phase (open all trees at queried indices)
```

#### Proof structure

```rust
struct UnifiedProof<F, E> {
    /// Per-table data: public inputs, OOD evaluations, bus_public_inputs
    per_table: Vec<TableProofData<F, E>>,
    /// Shared commitments (main, aux, composition)
    commitments: SharedCommitments,
    /// Layout describing which columns belong to which table
    layout: CommitmentLayout,
    /// Single FRI decommitment covering all tables
    fri_proof: FriProof<F>,
    /// Grinding nonce
    grinding_seed: [u8; 32],
}

struct TableProofData<F, E> {
    /// Evaluations of this table's columns at the OOD point z
    trace_ood_evaluations: PolynomialOpenings<E>,
    /// Evaluations of this table's composition parts at z
    composition_ood_evaluations: Vec<FieldElement<E>>,
    /// Bus public inputs (table_contribution)
    bus_public_inputs: Option<BusPublicInputs<E>>,
}
```

### Verifier impact

The verifier:
1. Reconstructs `CommitmentLayout` from AIRs (deterministic)
2. Receives 3 shared commitment roots
3. Re-derives all challenges from the single transcript
4. For each table: verifies constraint evaluations at z match the claimed composition values
5. Verifies the single FRI proof against the combined DEEP polynomial
6. Checks FRI query openings against the shared Merkle trees

## Blowup Factor Tuning

No code changes needed. The benchmark compares `GoldilocksCubicProofOptions::with_blowup(2)` vs `with_blowup(4)`. With single FRI (commit 4), the trade-off shifts:

- Blowup=2: LDE = 2×N, 219 FRI queries, depth ~21
- Blowup=4: LDE = 4×N (2× more FFT + hashing), 110 FRI queries, depth ~22

The benchmark reports proving time and proof size for each. The optimal blowup depends on the ratio of LDE cost (linear in blowup) vs FRI query cost (linear in queries × depth).

## Testing

Each commit must pass the existing prove_and_verify tests. The test programs exercise different table combinations:
- `test_add_8`: CPU + MEMW_R + BRANCH (simple, most tables empty)
- `all_instructions_64`: all tables active
- Programs with LOAD/STORE: MEMW_A active

After commit 3 (uniform max_rows), programs that previously fit in 1 chunk of 2^21 may need 2 chunks of 2^20. The tests must still pass.

After commit 4 (single FRI), the proof format changes completely. All test assertions on proof structure need updating.

## What This Does NOT Change

- Table column layouts (CPU stays 74 cols, MEMW_R stays 10 cols, etc.)
- Constraint definitions (no AIR changes)
- Bus interactions (same LogUp with same batching)
- Trace generation (same phases 0-5 in trace_builder)
- Execution (same executor, same ELFs)
- Field (Goldilocks) or extension (cubic)
- Hash function (Keccak256)
