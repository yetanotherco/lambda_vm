# Compiled Constraint Evaluator

## Problem

The constraint evaluation hot loop in `evaluator.rs` dispatches 77 virtual
calls per LDE row via `Vec<Box<dyn TransitionConstraint>>`. For the CPU table
at 2^20 rows with blowup 4, this is 77 × 8M = 616M indirect calls per proof.
Each call goes through:
1. vtable lookup → function pointer load → branch
2. Prover/Verifier match arm
3. Base-field computation
4. `to_extension()` embed (66 of 77 constraints) — unnecessary 3-limb construction
5. Write to `transition_buf[idx]`
6. Later: dot product of 77 E×E multiplications with beta coefficients

Additional waste:
- 66 `to_extension()` embeds where F×E multiplication would suffice
- `AddConstraint` carry_1 recomputes carry_0 (4 duplicated passes)
- `Frame::fill_from_lde` copies 258 elements into an intermediate buffer
- `transition_buf` zeroed on every iteration despite unconditional overwrites
- Dynamic dispatch on `Multiplicity`, `AddOperand`, `BusValue` enums per row

## Goal

Replace the virtual dispatch loop with a **compiled evaluation function per
table** that:
- Reads columns directly from LDE arrays (no Frame intermediate)
- Computes all constraints in a single pass
- Uses F×E multiplication for base-field constraints (skip `to_extension()`)
- Shares carry_0 computation across carry_1 constraints
- Fuses the zerofier multiplication into the accumulation
- Accumulates directly into a single extension-field result

## Design

### New Trait Method: `evaluate_all_transitions`

Add an optional method to the `AIR` trait (with a default fallback to the
existing `compute_transition_into`):

```rust
trait AIR {
    /// Compiled constraint evaluation: computes the weighted sum
    /// Σ beta_i * constraint_i(row) directly, bypassing virtual dispatch.
    /// Returns the pre-zerofier composition value.
    ///
    /// Default: falls back to compute_transition_into + dot product.
    fn evaluate_all_transitions_compiled(
        &self,
        main_curr: &[FieldElement<F>],
        main_next: &[FieldElement<F>],
        aux_curr: &[FieldElement<E>],
        aux_next: &[FieldElement<E>],
        betas: &[FieldElement<E>],
        rap_challenges: &[FieldElement<E>],
        periodic_values: &[FieldElement<F>],
    ) -> Option<FieldElement<E>> {
        None  // default: use virtual dispatch
    }
}
```

When `Some(value)` is returned, the evaluator uses it directly (skip
`compute_transition_into` + dot product). When `None`, falls back to existing
path.

### Compiled Evaluator for CPU Table

Implement `evaluate_all_transitions_compiled` for the CPU AIR. The function
body is a hand-written (or macro-generated) Rust function that:

1. Reads specific columns by constant index from the input slices
2. Evaluates all 66 native constraints + 11 LogUp constraints
3. For base-field constraints: `acc += beta[i] * constraint_value` using F×E mul
4. For ext-field constraints (LogUp): `acc += beta[i] * constraint_value` using E×E mul
5. Returns the accumulated sum

### Direct LDE Access (Skip Frame)

Modify `evaluate_transitions` in `evaluator.rs` to pass raw column slices
to `evaluate_all_transitions_compiled` instead of building a `Frame`. The
offsets for current/next row in column-major LDE:

```rust
let curr_offset = i;
let next_offset = (i + step_size * blowup_factor) % lde_size;
// Read: columns[col][curr_offset], columns[col][next_offset]
```

This eliminates the 258-element `fill_from_lde` copy per row.

### Implementation Approach

Rather than hand-writing the compiled function (error-prone for 77
constraints), use a **macro or code-generation** approach:

Option A: Implement `evaluate_all_transitions_compiled` directly in
`prover/src/constraints/cpu.rs` by calling each constraint's `compute()`
method inline without virtual dispatch, accumulating with F×E multiplication.

Option B: Generate the function via a proc-macro that reads the constraint
definitions.

**Recommendation: Option A** — direct implementation in cpu.rs. The
constraints are fixed at compile time, and explicit code is easier to audit
and optimize.

## Files Changed

| File | Change |
|------|--------|
| `crypto/stark/src/traits.rs` | Add `evaluate_all_transitions_compiled` to AIR trait |
| `crypto/stark/src/constraints/evaluator.rs` | Use compiled path when available |
| `prover/src/constraints/cpu.rs` | Implement compiled evaluator for CPU table |

## Expected Impact

- Eliminate 77 vtable dispatches per LDE row (~616M indirect calls)
- Eliminate 66 `to_extension()` embeds per LDE row (replace with F×E mul)
- Eliminate 258-element Frame copy per LDE row
- Share carry_0 computation across carry_1 constraints (save ~4 passes)
- Estimated: ~10-15% total proving time reduction (constraint eval is ~19%)
