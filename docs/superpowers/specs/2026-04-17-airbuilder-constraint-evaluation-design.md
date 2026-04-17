# AirBuilder Constraint Evaluation

## Goal

Replace the current virtual-dispatch constraint evaluation (`Box<dyn TransitionConstraint>`) with a Plonky3-style `AirBuilder` pattern: monomorphized, fused alpha combination, no intermediate frame copy. Expected ~10-20% speedup in constraint evaluation (5-10% overall proving time).

## Context

### Current Architecture

Each table's AIR stores constraints as `Vec<Box<dyn TransitionConstraint>>`. For each LDE domain point:

1. **Frame fill** (~5-10% of constraint time): Copy ~140 column values from LDE into a pre-allocated `Frame` struct via `fill_from_lde`.
2. **Virtual dispatch** (<1%): Iterate `transition_constraints()`, call `c.evaluate(ctx, buf)` on each `Box<dyn>`.
3. **Buffer write**: Each constraint writes its result to `transition_buf[constraint_idx]`.
4. **Second pass** (~5%): Accumulate `Σ(βⁱ * buf[i])` over all constraints.
5. **Zerofier multiply**: Apply precomputed `1/Z(x)`.

The `TransitionEvaluationContext` enum has two variants (`Prover`/`Verifier`), and every constraint must `match` on it — duplicating logic for base vs extension field types.

### What Plonky3 Does

Plonky3's `Air::eval_all<AB: AirBuilder>(&self, builder: &mut AB)` method:

- **Generic over `AirBuilder`**: The builder is monomorphized at compile time — no vtable.
- **Direct column access**: `builder.main().row_slice(0)` returns a slice into the LDE, no frame copy.
- **Fused accumulation**: `builder.assert_zero(expr)` immediately multiplies by `α^i` and adds to a running sum. No intermediate buffer.
- **Single code path**: The `AirBuilder` trait handles both prover (base field) and verifier (extension field) via associated types.

## Architecture

### New Trait: `AirBuilder`

```rust
/// Builder that accumulates constraint evaluations with alpha combination.
/// Generic over field type — same code path for prover and verifier.
pub trait AirBuilder {
    type F: IsField;
    
    /// Read a main trace column value at the given row offset (0 = current, 1 = next).
    fn main(&self, offset: usize, col: usize) -> FieldElement<Self::F>;
    
    /// Read an aux trace column value.
    fn aux(&self, offset: usize, col: usize) -> FieldElement<Self::F>;
    
    /// Assert that expr == 0. Internally: accumulator += α^i * expr.
    fn assert_zero(&mut self, expr: FieldElement<Self::F>);
    
    /// Access to RAP challenges (LogUp alpha, etc).
    fn challenge(&self, idx: usize) -> &FieldElement<Self::F>;
    
    /// Access to LogUp alpha powers.
    fn logup_alpha_power(&self, idx: usize) -> &FieldElement<Self::F>;
    
    /// Access to LogUp table offset.
    fn logup_table_offset(&self) -> &FieldElement<Self::F>;
}
```

### New AIR Method: `eval_constraints`

Each table's AIR implements a single `eval_constraints` method instead of returning `Vec<Box<dyn>>`:

```rust
pub trait AIR {
    // ... existing methods ...
    
    /// Evaluate all transition constraints using the builder pattern.
    /// Replaces transition_constraints() + compute_transition_into().
    fn eval_constraints<AB: AirBuilder<F = Self::FieldExtension>>(&self, builder: &mut AB);
}
```

Example for a simple constraint (IS_BIT on column X):

```rust
// Before: separate struct + trait impl + match on Prover/Verifier
// After: inline in eval_constraints()
fn eval_constraints<AB: AirBuilder>(&self, builder: &mut AB) {
    let x = builder.main(0, cols::PC_DOUBLE_READ);
    builder.assert_zero(x * (FieldElement::one() - x));
}
```

### Builder Implementations

**ProverBuilder**: Used during Round 2 constraint evaluation.

```rust
struct ProverBuilder<'a, F, E> {
    /// Direct reference to LDE columns (no frame copy)
    lde_trace: &'a LDETraceTable<F, E>,
    /// Current row index in LDE domain
    row: usize,
    /// LDE offsets for multi-row access
    offsets: &'a [usize],
    /// Running accumulator: Σ(α^i * constraint_i)
    accumulator: FieldElement<E>,
    /// Current alpha power (incremented after each assert_zero)
    alpha_power: FieldElement<E>,
    /// Alpha challenge for composition
    alpha: FieldElement<E>,
    /// RAP challenges
    rap_challenges: &'a [FieldElement<E>],
    logup_alpha_powers: &'a [FieldElement<E>],
    logup_table_offset: &'a FieldElement<E>,
}
```

Key: `main(offset, col)` reads directly from `lde_trace.get_main(row + offset * step_size, col)` — no frame allocation, no copy.

**VerifierBuilder**: Used during OOD verification. Same API but reads from the verifier's frame (extension field only).

### Evaluator Changes

The hot loop in `evaluator.rs` simplifies from:

```
frame.fill_from_lde(...)          // REMOVED
air.compute_transition_into(...)  // REPLACED
sum = Σ(β^i * buf[i])            // FUSED INTO BUILDER
```

To:

```
let mut builder = ProverBuilder::new(lde_trace, row, ...);
air.eval_constraints(&mut builder);
let result = zerofier * builder.accumulator + boundary;
```

### Migration Strategy

The change is mechanical per table. Each table's AIR currently has:

1. A `create_all_*_constraints()` function returning `(Vec<IsBitConstraint>, Vec<AddConstraint>, Vec<Box<dyn...>>)`
2. The AIR stores these in fields and returns them from `transition_constraints()`

After migration, each table's AIR has an `eval_constraints()` method that contains the same logic inline. The individual constraint structs (`IsBitConstraint`, `AddConstraint`, `PcDoubleReadRs1Constraint`, etc.) are deleted.

### Zerofier Handling

Current: All constraints share the same zerofier (uniform fast path) or grouped zerofiers. This is already optimal.

With AirBuilder: The uniform case is unchanged — the builder accumulates `Σ(α^i * expr_i)` and the evaluator multiplies by `1/Z(x)` once. For non-uniform zerofiers (rare in this codebase), the builder would need a `assert_zero_with_zerofier(expr, group_id)` variant.

Since all transition constraints in lambda_vm currently have `period=1, offset=0, end_exemptions=0` (uniform), the simple `assert_zero` approach works for all existing constraints.

## Files Changed

| File | Change |
|------|--------|
| `crypto/stark/src/traits.rs` | Add `AirBuilder` trait, add `eval_constraints()` to `AIR` trait |
| `crypto/stark/src/constraints/evaluator.rs` | Replace hot loop with builder-based evaluation |
| `crypto/stark/src/frame.rs` | Keep for verifier; prover no longer uses Frame |
| `prover/src/constraints/cpu.rs` | Rewrite: delete constraint structs, implement `eval_constraints()` |
| `prover/src/constraints/templates.rs` | Delete `IsBitConstraint`, `AddConstraint` structs; replace with helper fns |
| `prover/src/constraints/*.rs` | Same pattern for all table constraint files |
| `prover/src/tables/*.rs` | Update AIR impls to provide `eval_constraints()` |

## Scope

### In Scope

- `AirBuilder` trait and `ProverBuilder`/`VerifierBuilder` implementations
- Migrate all table AIRs to `eval_constraints()` pattern
- Delete `TransitionConstraint` trait objects and individual constraint structs
- Remove `Frame::fill_from_lde` from prover hot path (keep for verifier)

### Out of Scope

- Column-major storage migration (separate effort)
- Symbolic expression compilation / CSE (future optimization on top of builder)
- Changes to bus interactions, LogUp, or boundary constraints
- Changes to zerofier computation or composition decomposition

## Risks

- **Large diff**: Every table's constraint code is rewritten. Must be done atomically or table-by-table with a compatibility shim.
- **Verifier regression**: Verifier uses the same constraints. The `VerifierBuilder` must produce identical results.
- **LogUp constraints**: The most complex constraints (fingerprint computation, accumulated column) must be migrated carefully.

## Verification

- All existing tests must pass (same constraint logic, different evaluation mechanism)
- Proof output must be bit-identical (same alpha combination, same zerofier division)
- Benchmark: `vm_prover_benchmark` before/after on `fib_iterative_8M`
