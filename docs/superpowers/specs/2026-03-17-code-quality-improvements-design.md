# Code Quality Improvements

**Date:** 2026-03-17
**Branch:** `chore/code-quality-improvements` from `origin/main`
**Approach:** Extract shared helpers, add macros, consolidate constants. No behavioral changes.

## Baseline

Production LOC: 34,866 (excluding tests, benchmarks, examples)

```bash
find ./prover/src ./crypto/stark/src ./crypto/math/src ./crypto/crypto/src ./executor/src ./bin/cli/src -name '*.rs' -not -name '*test*' -not -path '*/tests/*' -not -path '*/examples/*' | xargs wc -l
```

## Group 1: `impl_simple_evaluate!()` macro (~310 lines)

**Problem:** 23 constraint structs copy-paste the same ~17-line `evaluate()` method with a Prover/Verifier match. Every instance calls the banned `to_extension()` pattern.

**Why not a trait default:** The existing `compute` methods are generic inherent methods `fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>`. The Prover arm calls `compute::<GoldilocksField, GoldilocksExtension>` (returns base field), while the Verifier arm calls `compute::<GoldilocksExtension, GoldilocksExtension>` (returns extension field). A trait default method cannot call a generic inherent method with two different type instantiations.

**Solution:** A `macro_rules!` macro that emits the boilerplate `evaluate()` body. Each struct uses `impl_simple_evaluate!();` inside its trait impl block.

### 1.1 Define the macro

**File:** `prover/src/constraints/mod.rs` (or a new `prover/src/constraints/macros.rs`)

```rust
/// Emits the standard `evaluate()` body for simple constraints whose
/// `compute()` method is generic over `<F, E>` and returns `FieldElement<F>`.
///
/// Usage: place `impl_simple_evaluate!();` inside the
/// `impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MyConstraint`
/// block, alongside `degree()`, `constraint_idx()`, and `end_exemptions()`.
macro_rules! impl_simple_evaluate {
    () => {
        fn evaluate(
            &self,
            evaluation_context: &stark::traits::TransitionEvaluationContext<
                crate::test_utils::F,
                crate::test_utils::E,
            >,
            transition_evaluations: &mut [math::field::element::FieldElement<
                crate::test_utils::E,
            >],
        ) {
            match evaluation_context {
                stark::traits::TransitionEvaluationContext::Prover { frame, .. } => {
                    let v = self.compute(frame.get_evaluation_step(0));
                    transition_evaluations[self.constraint_idx] = v.to_extension();
                }
                stark::traits::TransitionEvaluationContext::Verifier { frame, .. } => {
                    let v = self.compute(frame.get_evaluation_step(0));
                    transition_evaluations[self.constraint_idx] = v;
                }
            }
        }
    };
}
pub(crate) use impl_simple_evaluate;
```

Note: `to_extension()` remains in the macro body (single source location). Eliminating it entirely requires changing the trait's `evaluate` signature or the constraint evaluation framework — out of scope for this PR.

### 1.2 Update 23 constraint impls

For each struct, replace the ~17-line `evaluate()` with `impl_simple_evaluate!();`:

```rust
// BEFORE (17 lines):
fn evaluate(
    &self,
    evaluation_context: &TransitionEvaluationContext<GoldilocksField, GoldilocksExtension>,
    transition_evaluations: &mut [FieldElement<GoldilocksExtension>],
) {
    match evaluation_context {
        TransitionEvaluationContext::Prover { frame, .. } => {
            let v = self.compute(frame.get_evaluation_step(0));
            transition_evaluations[self.constraint_idx] = v.to_extension();
        }
        TransitionEvaluationContext::Verifier { frame, .. } => {
            let v = self.compute(frame.get_evaluation_step(0));
            transition_evaluations[self.constraint_idx] = v;
        }
    }
}

// AFTER (1 line):
impl_simple_evaluate!();
```

**Files affected (23 structs):**
- `prover/src/constraints/cpu.rs` — 13 structs
- `prover/src/constraints/templates.rs` — 2 structs
- `prover/src/tables/branch.rs` — 1 struct
- `prover/src/tables/commit.rs` — 1 struct
- `prover/src/tables/dvrm.rs` — 1 struct
- `prover/src/tables/load.rs` — 1 struct
- `prover/src/tables/lt.rs` — 1 struct
- `prover/src/tables/memw.rs` — 1 struct
- `prover/src/tables/mul.rs` — 1 struct
- `prover/src/tables/shift.rs` — 1 struct

**NOT changed:** `LookupBatchedTermConstraint` and `LookupAccumulatedConstraint` in `crypto/stark/src/lookup.rs` — they have custom `evaluate` bodies using `rap_challenges` and `logup_alpha_powers`.

**Savings:** 23 × ~16 lines removed, +10 lines for macro definition = **~358 lines saved → ~310 net**.

## Group 2: `commit_columns_to_lde` shared helper (~135 lines)

**Problem:** 4 preprocessed tables duplicate the identical LDE→commit pipeline (interpolate → LDE → bit-reverse → columns2rows → Merkle build).

**Solution:** Extract to `prover/src/tables/mod.rs`.

### 2.1 Add helper function

**File:** `prover/src/tables/mod.rs`

```rust
/// Commits precomputed columns to a Merkle root over their LDE.
///
/// Standard preprocessed-table commitment pipeline:
///   interpolate_fft → coset-LDE → bit-reverse → columns2rows → BatchedMerkleTree
pub(crate) fn commit_columns_to_lde(
    columns: Vec<Vec<FE>>,
    num_rows: usize,
    options: &ProofOptions,
) -> Commitment {
    // ... (single implementation with #[cfg(feature = "parallel")] guard)
}
```

### 2.2 Simplify 4 table files

Each table's commitment function reduces to: generate columns → call `super::commit_columns_to_lde(columns, num_rows, options)`.

| File | Function | Lines before → after |
|---|---|---|
| `bitwise.rs` | `preprocessed_commitment` | ~94 → ~30 |
| `decode.rs` | `commitment_from_elf` | ~55 → ~20 |
| `page.rs` | `precomputed_commitment_cached` (inner) | ~55 → ~20 |
| `register.rs` | `preprocessed_commitment` | ~40 → ~16 |

## Group 3: Trace fill helpers (~50 lines)

**Problem:** 13 DWordHL blocks (4 lines each) and 10 DWordWL blocks (2 lines each) repeated across 6 table files.

**Solution:** Add two inline helpers to `prover/src/tables/types.rs`.

```rust
#[inline(always)]
pub(crate) fn fill_dword_hl(data: &mut [FE], base: usize, cols: [usize; 4], val: u64) {
    data[base + cols[0]] = FE::from(val & 0xFFFF);
    data[base + cols[1]] = FE::from((val >> 16) & 0xFFFF);
    data[base + cols[2]] = FE::from((val >> 32) & 0xFFFF);
    data[base + cols[3]] = FE::from((val >> 48) & 0xFFFF);
}

#[inline(always)]
pub(crate) fn fill_dword_wl(data: &mut [FE], base: usize, cols: [usize; 2], val: u64) {
    data[base + cols[0]] = FE::from(val & 0xFFFF_FFFF);
    data[base + cols[1]] = FE::from(val >> 32);
}
```

**Files affected:** `dvrm.rs` (7 sites), `mul.rs` (4 sites), `commit.rs` (4 sites), `lt.rs` (1 site), `memw.rs` (3 sites).

## Group 4: Trace builder deduplication (~139 lines)

### 4.1 `collect_lt_from_memw` → delegate (~62 lines)

**File:** `prover/src/tables/trace_builder.rs`

Replace the 64-line body of `collect_lt_from_memw` with:
```rust
fn collect_lt_from_memw(memw_ops: &[MemwOperation]) -> Vec<LtOperation> {
    memw_ops.iter().flat_map(|op| op.collect_lt_lookups()).collect()
}
```

The identical logic already exists as `MemwOperation::collect_lt_lookups` in `memw.rs`.

### 4.2 `push_is_half_u64` helper (~77 lines)

**File:** `prover/src/tables/trace_builder.rs`

Add helper, replace 11 identical 8-line `for shift in [0, 16, 32, 48]` loops:

```rust
#[inline]
fn push_is_half_u64(ops: &mut Vec<BitwiseOperation>, value: u64) {
    for shift in [0u32, 16, 32, 48] {
        let half = ((value >> shift) & 0xFFFF) as u16;
        ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (half & 0xFF) as u8,
            (half >> 8) as u8,
        ));
    }
}
```

## Group 5: Bus interaction helpers (~198 lines)

### 5.1 IS_HALF column helper (~136 lines)

Add to each table file (or a shared location):

```rust
fn push_is_half_columns(
    interactions: &mut Vec<BusInteraction>,
    cols: &[usize],
    multiplicity: Multiplicity,
) {
    for &col in cols {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            multiplicity.clone(),
            vec![BusValue::Packed { start_column: col, packing: Packing::Direct }],
        ));
    }
}
```

**Files affected:** `dvrm.rs` (5 groups → 5 calls, ~70 lines saved), `mul.rs` (2 groups, ~20 lines), `lt.rs` (4 items, ~46 lines).

**Prerequisite:** `Multiplicity` must implement `Clone`. Verify and add `#[derive(Clone)]` if needed.

### 5.2 AND/OR/XOR byte loop (~62 lines)

**File:** `prover/src/tables/cpu.rs`

Replace 3 near-identical 22-line loops with a data-driven loop:

```rust
for &(bus_id, mult_col) in &[
    (BusId::AndByte, cols::AND),
    (BusId::OrByte,  cols::OR),
    (BusId::XorByte, cols::XOR),
] {
    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            bus_id, Multiplicity::Column(mult_col),
            vec![
                BusValue::Packed { start_column: cols::ARG1[i], packing: Packing::Direct },
                BusValue::Packed { start_column: cols::ARG2[i], packing: Packing::Direct },
                BusValue::Packed { start_column: cols::RES[i],  packing: Packing::Direct },
            ],
        ));
    }
}
```

## Group 6: Constant consolidation (~25 lines + 17 clarified)

### 6.1 Consolidate shift constants

**File:** `prover/src/tables/types.rs` — add missing constants:
```rust
pub const SHIFT_8: u64 = 1 << 8;
pub const SHIFT_24: u64 = 1 << 24;
```

(`SHIFT_16` and `SHIFT_32` already exist here.)

**Remove duplicates from:**
- `prover/src/tables/branch.rs:99` — local `const SHIFT_8`
- `crypto/stark/src/lookup.rs` — `SHIFT_8`, `SHIFT_16`, `SHIFT_32` (already removed in simplification branch, verify on main)

### 6.2 Unify inverse constants

**File:** `prover/src/tables/types.rs`

`INV_2_32` (in types.rs) = `INV_SHIFT_32` (in templates.rs) = `18446744065119617026`. Pick one name (`INV_SHIFT_32`), define it in `types.rs`, import from `templates.rs`.

### 6.3 Consolidate `SIGN_FILL`

Move from local `const` in `dvrm.rs:151` and `mul.rs:131` to `types.rs`:
```rust
pub const SIGN_FILL: u64 = 0xFFFF;
```

### 6.4 Replace magic `coefficient:` literals

In `cpu.rs` and `bitwise.rs`, replace:
- `coefficient: 65536` → `coefficient: SHIFT_16 as i64`
- `coefficient: 256` → `coefficient: SHIFT_8 as i64`
- `coefficient: 16777216` → `coefficient: SHIFT_24 as i64`

~17 sites across the two files.

## Verification

After each group:
1. `cargo check` — no compilation errors
2. `cargo test -p math -p stark -p crypto -p lambda-vm-prover --lib` — all tests pass
3. `cargo clippy --all-targets` — no new warnings

## Estimated Impact

| Group | Description | Lines saved |
|---|---|---|
| 1 | impl_simple_evaluate!() macro | ~310 |
| 2 | commit_columns_to_lde | ~135 |
| 3 | fill_dword_hl/wl | ~50 |
| 4 | trace_builder dedup | ~139 |
| 5 | bus interaction helpers | ~198 |
| 6 | constant consolidation | ~25 + readability |
| **Total** | | **~857 lines** |

Production LOC target: ~34,010 (from 34,866)

## Implementation Order

Groups are independent and can be parallelized. Suggested order for sequential execution:

1. **Group 6** (constants) — smallest, unblocks Group 5
2. **Group 3** (fill helpers) — small, self-contained
3. **Group 4** (trace_builder) — small, self-contained
4. **Group 5** (bus interaction helpers) — depends on Group 6 for constant names
5. **Group 2** (commit helper) — medium, self-contained
6. **Group 1** (trait default) — largest, most impactful, touches many files
