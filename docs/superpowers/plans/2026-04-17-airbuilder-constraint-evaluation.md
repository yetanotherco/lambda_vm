# AirBuilder Constraint Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace virtual-dispatch constraint evaluation with a Plonky3-style AirBuilder: monomorphized, fused alpha combination, no frame copy in prover hot loop.

**Architecture:** Add `AirBuilder` trait with `ProverBuilder` (reads directly from LDE, accumulates constraints with alpha) and `VerifierBuilder` (reads from OOD frame). Each table AIR implements `eval_constraints_with_builder()` instead of returning `Vec<Box<dyn TransitionConstraint>>`. Migration is incremental via a `uses_builder()` flag.

**Tech Stack:** Rust, Goldilocks field, existing STARK prover in `crypto/stark/`

---

## File Structure

| File | Role |
|------|------|
| `crypto/stark/src/air_builder.rs` (NEW) | `AirBuilder` trait, `ProverBuilder`, `VerifierBuilder` |
| `crypto/stark/src/traits.rs` | Add `eval_constraints_with_builder()` + `uses_builder()` to `AIR` trait |
| `crypto/stark/src/constraints/evaluator.rs` | Add builder-based hot loop path |
| `crypto/stark/src/verifier.rs` | Add builder-based OOD evaluation path |
| `prover/src/constraints/helpers.rs` (NEW) | `assert_is_bit()`, `assert_is_bit_cond()` helper fns |
| `prover/src/tables/*.rs` | Override `eval_constraints_with_builder()` per table |

---

### Task 1: AirBuilder Trait, ProverBuilder, and VerifierBuilder

**Files:**
- Create: `crypto/stark/src/air_builder.rs`
- Modify: `crypto/stark/src/lib.rs` (add `pub mod air_builder;`)

- [ ] **Step 1: Create `air_builder.rs` with trait and both builder implementations**

```rust
// crypto/stark/src/air_builder.rs
use crate::frame::Frame;
use crate::trace::LDETraceTable;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};

/// Plonky3-style builder for fused constraint evaluation + alpha combination.
///
/// Constraints call `assert_zero(expr)` which internally accumulates
/// alpha^i * expr into a running sum. No intermediate buffer, no vtable dispatch.
pub trait AirBuilder {
    type F: IsField;

    /// Read main trace column at (row_offset, col). offset=0 is current row.
    fn main(&self, offset: usize, col: usize) -> FieldElement<Self::F>;

    /// Read aux trace column at (row_offset, col).
    fn aux(&self, offset: usize, col: usize) -> FieldElement<Self::F>;

    /// Assert expr == 0. Internally: accumulator += alpha^constraint_idx * expr.
    fn assert_zero(&mut self, expr: FieldElement<Self::F>);

    /// RAP challenge by index.
    fn challenge(&self, idx: usize) -> &FieldElement<Self::F>;

    /// Pre-computed LogUp alpha powers.
    fn logup_alpha_power(&self, idx: usize) -> &FieldElement<Self::F>;

    /// LogUp table offset (L/N).
    fn logup_table_offset(&self) -> &FieldElement<Self::F>;
}

// ---------------------------------------------------------------------------
// ProverBuilder: reads directly from LDE columns, accumulates in extension field
// ---------------------------------------------------------------------------

pub struct ProverBuilder<'a, F: IsSubFieldOf<E> + IsFFTField, E: IsField> {
    lde_trace: &'a LDETraceTable<F, E>,
    row: usize,
    step_size: usize,
    num_rows: usize,
    accumulator: FieldElement<E>,
    alpha: FieldElement<E>,
    alpha_power: FieldElement<E>,
    rap_challenges: &'a [FieldElement<E>],
    logup_alpha_powers: &'a [FieldElement<E>],
    logup_table_offset_val: &'a FieldElement<E>,
}

impl<'a, F, E> ProverBuilder<'a, F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(
        lde_trace: &'a LDETraceTable<F, E>,
        row: usize,
        alpha: &FieldElement<E>,
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    ) -> Self {
        Self {
            lde_trace,
            row,
            step_size: lde_trace.lde_step_size,
            num_rows: lde_trace.num_rows(),
            accumulator: FieldElement::zero(),
            alpha: alpha.clone(),
            alpha_power: FieldElement::one(),
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset_val: logup_table_offset,
        }
    }

    /// Consume the builder and return the accumulated sum.
    pub fn finish(self) -> FieldElement<E> {
        self.accumulator
    }
}

impl<'a, F, E> AirBuilder for ProverBuilder<'a, F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    type F = E;

    #[inline]
    fn main(&self, offset: usize, col: usize) -> FieldElement<E> {
        let lde_row = (self.row + offset * self.step_size) % self.num_rows;
        self.lde_trace.get_main(lde_row, col).to_extension()
    }

    #[inline]
    fn aux(&self, offset: usize, col: usize) -> FieldElement<E> {
        let lde_row = (self.row + offset * self.step_size) % self.num_rows;
        self.lde_trace.get_aux(lde_row, col).clone()
    }

    #[inline]
    fn assert_zero(&mut self, expr: FieldElement<E>) {
        self.accumulator = &self.accumulator + &self.alpha_power * &expr;
        self.alpha_power = &self.alpha_power * &self.alpha;
    }

    fn challenge(&self, idx: usize) -> &FieldElement<E> {
        &self.rap_challenges[idx]
    }

    fn logup_alpha_power(&self, idx: usize) -> &FieldElement<E> {
        &self.logup_alpha_powers[idx]
    }

    fn logup_table_offset(&self) -> &FieldElement<E> {
        self.logup_table_offset_val
    }
}

// ---------------------------------------------------------------------------
// VerifierBuilder: reads from OOD frame (extension field only)
// ---------------------------------------------------------------------------

pub struct VerifierBuilder<'a, E: IsField> {
    frame: &'a Frame<E, E>,
    accumulator: FieldElement<E>,
    alpha: FieldElement<E>,
    alpha_power: FieldElement<E>,
    rap_challenges: &'a [FieldElement<E>],
    logup_alpha_powers: &'a [FieldElement<E>],
    logup_table_offset_val: &'a FieldElement<E>,
}

impl<'a, E: IsField> VerifierBuilder<'a, E> {
    pub fn new(
        frame: &'a Frame<E, E>,
        alpha: &FieldElement<E>,
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    ) -> Self {
        Self {
            frame,
            accumulator: FieldElement::zero(),
            alpha: alpha.clone(),
            alpha_power: FieldElement::one(),
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset_val: logup_table_offset,
        }
    }

    pub fn finish(self) -> FieldElement<E> {
        self.accumulator
    }
}

impl<'a, E: IsField> AirBuilder for VerifierBuilder<'a, E> {
    type F = E;

    #[inline]
    fn main(&self, offset: usize, col: usize) -> FieldElement<E> {
        self.frame
            .get_evaluation_step(offset)
            .get_main_evaluation_element(0, col)
            .clone()
    }

    #[inline]
    fn aux(&self, offset: usize, col: usize) -> FieldElement<E> {
        self.frame
            .get_evaluation_step(offset)
            .get_aux_evaluation_element(0, col)
            .clone()
    }

    #[inline]
    fn assert_zero(&mut self, expr: FieldElement<E>) {
        self.accumulator = &self.accumulator + &self.alpha_power * &expr;
        self.alpha_power = &self.alpha_power * &self.alpha;
    }

    fn challenge(&self, idx: usize) -> &FieldElement<E> {
        &self.rap_challenges[idx]
    }

    fn logup_alpha_power(&self, idx: usize) -> &FieldElement<E> {
        &self.logup_alpha_powers[idx]
    }

    fn logup_table_offset(&self) -> &FieldElement<E> {
        self.logup_table_offset_val
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

In `crypto/stark/src/lib.rs`, add:
```rust
pub mod air_builder;
```

- [ ] **Step 3: Build**

Run: `cargo build -p stark`
Expected: compiles with no errors.

- [ ] **Step 4: Commit**

```bash
git add crypto/stark/src/air_builder.rs crypto/stark/src/lib.rs
git commit -m "feat: add AirBuilder trait, ProverBuilder, and VerifierBuilder"
```

---

### Task 2: AIR Trait Extension

**Files:**
- Modify: `crypto/stark/src/traits.rs`

- [ ] **Step 1: Add default methods to AIR trait**

Add after `compute_transition_into`:

```rust
    /// Whether this AIR uses the AirBuilder pattern for constraint evaluation.
    /// Tables override to return true after migrating to eval_constraints_with_builder().
    fn uses_builder(&self) -> bool {
        false
    }

    /// Evaluate all transition constraints using the AirBuilder pattern.
    /// Override this (and uses_builder -> true) to use fused alpha combination.
    fn eval_constraints_with_builder<AB: crate::air_builder::AirBuilder>(&self, _builder: &mut AB) {
        unimplemented!("Override eval_constraints_with_builder and uses_builder for this AIR")
    }
```

- [ ] **Step 2: Build**

Run: `cargo build -p stark`

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/traits.rs
git commit -m "feat: add uses_builder and eval_constraints_with_builder to AIR trait"
```

---

### Task 3: Evaluator Builder Path

**Files:**
- Modify: `crypto/stark/src/constraints/evaluator.rs`

- [ ] **Step 1: Add composition_alpha parameter and builder branch**

The evaluator needs the raw alpha challenge (not just the pre-computed powers) so the builder can generate its own powers internally.

Add `composition_alpha: &FieldElement<FieldExtension>` parameter to `evaluate_transitions`. Thread it from `evaluate()` by recovering alpha from `transition_coefficients[1]` (since coefficients are `[1, alpha, alpha^2, ...]`).

In the `map_init` closure, add a branch at the top:

```rust
if air.uses_builder() {
    let mut builder = crate::air_builder::ProverBuilder::new(
        lde_trace, i, composition_alpha,
        rap_challenges, &logup_alpha_powers, logup_table_offset,
    );
    air.eval_constraints_with_builder(&mut builder);
    let acc = if is_uniform {
        zerofier_data.get_uniform(i) * &builder.finish()
    } else {
        panic!("AirBuilder requires uniform zerofiers")
    };
    acc + boundary
} else {
    // ... existing code unchanged ...
}
```

Do the same for the non-parallel `#[cfg(not(feature = "parallel"))]` path.

- [ ] **Step 2: Build and run tests**

Run: `cargo test -p stark`
Expected: all pass (no table uses builder yet).

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/constraints/evaluator.rs
git commit -m "feat: add AirBuilder path to evaluator hot loop"
```

---

### Task 4: Verifier Builder Path

**Files:**
- Modify: `crypto/stark/src/verifier.rs`

- [ ] **Step 1: Add builder branch in step_2_verify_constraints**

Before the existing `compute_transition` call, add:

```rust
let transition_c_i_evaluations_sum = if air.uses_builder() {
    let composition_alpha = if challenges.transition_coeffs.len() >= 2 {
        challenges.transition_coeffs[1].clone()
    } else {
        FieldElement::one()
    };
    let mut builder = crate::air_builder::VerifierBuilder::new(
        &ood_frame, &composition_alpha,
        &challenges.rap_challenges, &logup_alpha_powers, &logup_table_offset,
    );
    air.eval_constraints_with_builder(&mut builder);
    // All zerofiers uniform: z^n - 1 is the single denominator
    let z_n = challenges.z.pow(trace_length as u64);
    let zerofier_at_z = &z_n - FieldElement::one();
    match zerofier_at_z.inv() {
        Ok(inv) => builder.finish() * inv,
        Err(_) => return false,
    }
} else {
    // ... existing code for old path ...
};
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p stark`

- [ ] **Step 3: Commit**

```bash
git add crypto/stark/src/verifier.rs
git commit -m "feat: add AirBuilder path to verifier"
```

---

### Task 5: Constraint Helpers

**Files:**
- Create: `prover/src/constraints/helpers.rs`
- Modify: `prover/src/constraints/mod.rs`

- [ ] **Step 1: Create helpers.rs**

```rust
// prover/src/constraints/helpers.rs
//! Helper functions for common constraint patterns used with AirBuilder.

use math::field::element::FieldElement;
use stark::air_builder::AirBuilder;

/// IS_BIT: x*(1-x) == 0
#[inline]
pub fn assert_is_bit<AB: AirBuilder>(builder: &mut AB, x: FieldElement<AB::F>) {
    let one = FieldElement::<AB::F>::one();
    builder.assert_zero(x.clone() * (one - x));
}

/// Conditional IS_BIT: cond * x * (1-x) == 0
#[inline]
pub fn assert_is_bit_cond<AB: AirBuilder>(
    builder: &mut AB,
    cond: FieldElement<AB::F>,
    x: FieldElement<AB::F>,
) {
    let one = FieldElement::<AB::F>::one();
    builder.assert_zero(cond * x.clone() * (one - x));
}

/// Zero when flag is off: (1 - flag) * value == 0
#[inline]
pub fn assert_zero_when_off<AB: AirBuilder>(
    builder: &mut AB,
    flag: FieldElement<AB::F>,
    value: FieldElement<AB::F>,
) {
    builder.assert_zero((FieldElement::<AB::F>::one() - flag) * value);
}
```

- [ ] **Step 2: Add module to mod.rs**

```rust
pub mod helpers;
```

- [ ] **Step 3: Build**

Run: `cargo build -p lambda-vm-prover`

- [ ] **Step 4: Commit**

```bash
git add prover/src/constraints/helpers.rs prover/src/constraints/mod.rs
git commit -m "feat: add constraint helper functions for AirBuilder"
```

---

### Task 6: Migrate MEMW_R (3 constraints - validates the pattern)

**Files:**
- Modify: `prover/src/tables/memw_register.rs`

- [ ] **Step 1: Read existing constraints**

The 3 constraints in `constraints()` function:
1. IS_BIT(MU_READ)
2. IS_BIT(MU_WRITE)
3. MuSumIsBit: `(mu_read + mu_write) * (1 - mu_read - mu_write) == 0`

- [ ] **Step 2: Add eval_constraints_with_builder to MemwRegisterAir**

```rust
fn uses_builder(&self) -> bool {
    true
}

fn eval_constraints_with_builder<AB: stark::air_builder::AirBuilder>(&self, builder: &mut AB) {
    use crate::constraints::helpers::assert_is_bit;
    let mu_read = builder.main(0, cols::MU_READ);
    let mu_write = builder.main(0, cols::MU_WRITE);
    assert_is_bit(builder, mu_read.clone());
    assert_is_bit(builder, mu_write.clone());
    let mu_sum = &mu_read + &mu_write;
    builder.assert_zero(mu_sum.clone() * (FieldElement::one() - mu_sum));
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p lambda-vm-prover tests::constraints_tests`
Run: `cargo test -p lambda-vm-prover tests::cpu_tests`

- [ ] **Step 4: Commit**

```bash
git add prover/src/tables/memw_register.rs
git commit -m "feat: migrate MEMW_R to AirBuilder"
```

---

### Task 7: Migrate Remaining Tables (BRANCH, MEMW_A, MUL, COMMIT, LOAD, MEMW, SHIFT, DVRM)

Each table follows the same pattern as Task 6. For each:
1. Read the existing constraint structs and their expressions
2. Override `uses_builder() -> true` and `eval_constraints_with_builder()`
3. Translate each constraint's expression into `builder.assert_zero(expr)`
4. Run tests, commit

- [ ] **Step 1: Migrate tables in order of ascending complexity**

Order: BRANCH (4) -> MEMW_A (4) -> MUL (6) -> COMMIT (8) -> LOAD (9) -> MEMW (12) -> SHIFT (16) -> DVRM (17+)

One commit per table.

- [ ] **Step 2: Run full tests after each table**

Run: `cargo test -p lambda-vm-prover`

---

### Task 8: Migrate CPU (69 constraints)

**Files:**
- Modify: `prover/src/tables/cpu.rs`
- Modify: `prover/src/constraints/cpu.rs`

- [ ] **Step 1: Add eval_cpu_constraints function**

Create `pub fn eval_cpu_constraints<AB: stark::air_builder::AirBuilder>(builder: &mut AB)` in `prover/src/constraints/cpu.rs`. Translate all 69 constraints:

- 34 IS_BIT: `for &col in BIT_FLAG_COLUMNS { assert_is_bit(builder, builder.main(0, col)); }`
- 8 ADD/SUB/JALR: translate carry-embedding algebra inline
- 27 custom: translate each struct's `compute()` expression

- [ ] **Step 2: Override uses_builder and eval_constraints_with_builder in CpuAir**

```rust
fn uses_builder(&self) -> bool { true }
fn eval_constraints_with_builder<AB: stark::air_builder::AirBuilder>(&self, builder: &mut AB) {
    crate::constraints::cpu::eval_cpu_constraints(builder);
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p lambda-vm-prover tests::constraints_tests`
Run: `cargo test -p lambda-vm-prover tests::cpu_tests`

- [ ] **Step 4: Commit**

```bash
git add prover/src/tables/cpu.rs prover/src/constraints/cpu.rs
git commit -m "feat: migrate CPU to AirBuilder (69 constraints)"
```

---

### Task 9: Remove Old Constraint Infrastructure

After all tables migrated:

- [ ] **Step 1: Remove old path from evaluator and verifier**

Remove the `if air.uses_builder() {} else {}` branching — keep only the builder path. Remove `uses_builder()` default or change default to `true`.

- [ ] **Step 2: Delete constraint structs**

Delete `IsBitConstraint`, `AddConstraint`, and all per-table constraint structs (ShiftConstraint, DvrmConstraint, etc.). Delete `create_all_cpu_constraints()` and similar factory functions. Remove `transition_constraints()` storage from AIR struct fields.

- [ ] **Step 3: Remove Frame from prover hot loop**

Remove the `Frame` allocation from the `map_init` closure. Keep `Frame` for verifier use.

- [ ] **Step 4: Run full test suite**

Run: `cargo test -p lambda-vm-prover`
Run: `cargo test -p stark`

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor: remove old virtual-dispatch constraint infrastructure"
```

---

### Task 10: Benchmark

- [ ] **Step 1: Run prover benchmark**

Run: `cargo bench -p lambda-vm-prover -- vm_prover_benchmark`

Compare against baseline (main branch). Expected: ~5-10% proving time improvement.

- [ ] **Step 2: Document results in PR or design doc**

Record before/after numbers for constraint evaluation time and total proving time.
