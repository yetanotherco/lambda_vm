use crate::frame::Frame;
use crate::trace::LDETraceTable;
use math::field::element::FieldElement;
use math::field::traits::{IsFFTField, IsField, IsSubFieldOf};

/// Plonky3-style builder for fused constraint evaluation + alpha combination.
///
/// Constraints call `assert_zero(expr)` which internally accumulates
/// alpha^i * expr into a running sum. No intermediate buffer, no vtable dispatch.
///
/// Parameterized by field type `F` (not an associated type) so that `dyn AirBuilder<F>`
/// is object-safe. This allows the AIR trait to remain dyn-compatible while supporting
/// the builder pattern: `eval_constraints(&self, builder: &mut dyn AirBuilder<E>)`.
pub trait AirBuilder<F: IsField> {
    /// Read main trace column at (row_offset, col). offset=0 is current row.
    fn main(&self, offset: usize, col: usize) -> FieldElement<F>;

    /// Read aux trace column at (row_offset, col).
    fn aux(&self, offset: usize, col: usize) -> FieldElement<F>;

    /// Assert expr == 0. Internally: accumulator += alpha^constraint_idx * expr.
    fn assert_zero(&mut self, expr: FieldElement<F>);

    /// RAP challenge by index.
    fn challenge(&self, idx: usize) -> &FieldElement<F>;

    /// Pre-computed LogUp alpha powers.
    fn logup_alpha_power(&self, idx: usize) -> &FieldElement<F>;

    /// LogUp table offset (L/N).
    fn logup_table_offset(&self) -> &FieldElement<F>;
}

/// Base-field builder for main trace constraints.
///
/// Main trace constraints compute in base field F, and `assert_zero_base(expr_F)`
/// accumulates into the extension-field sum using F*E multiplication (3 base muls
/// instead of 6 for E*E). This gives ~2x speedup for pure main-trace constraints.
///
/// The `main_base(col)` method reads from a pre-fetched row cache (contiguous in memory)
/// instead of random column-major access, improving cache locality.
pub trait MainAirBuilder<F: IsSubFieldOf<E> + IsField, E: IsField> {
    /// Read main trace column value in base field (no extension conversion).
    /// Only supports offset=0 (current row).
    fn main_base(&self, col: usize) -> FieldElement<F>;

    /// Assert expr == 0 with base-field expression.
    /// Internally: accumulator_E += alpha_power_E * expr_F (F*E multiply, 3 base muls).
    fn assert_zero_base(&mut self, expr: FieldElement<F>);
}

pub struct ProverBuilder<'a, F: IsSubFieldOf<E> + IsFFTField, E: IsField> {
    lde_trace: &'a LDETraceTable<F, E>,
    row: usize,
    step_size: usize,
    num_rows: usize,
    accumulator: FieldElement<E>,
    /// Pre-computed composition alpha powers [1, α, α², ...].
    /// Indexed by constraint_idx to avoid E×E multiply per constraint.
    alpha_powers: &'a [FieldElement<E>],
    constraint_idx: usize,
    rap_challenges: &'a [FieldElement<E>],
    logup_alpha_powers: &'a [FieldElement<E>],
    logup_table_offset_val: &'a FieldElement<E>,
    /// Pre-fetched main trace row for base-field access (contiguous in memory).
    /// Populated by `new_with_cache` to avoid per-iteration allocation.
    main_row_cache: &'a [FieldElement<F>],
}

impl<'a, F, E> ProverBuilder<'a, F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    pub fn new(
        lde_trace: &'a LDETraceTable<F, E>,
        row: usize,
        alpha_powers: &'a [FieldElement<E>],
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
            alpha_powers,
            constraint_idx: 0,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset_val: logup_table_offset,
            main_row_cache: &[],
        }
    }

    /// Create a ProverBuilder with a pre-allocated row cache buffer.
    ///
    /// The `row_cache` buffer is filled with the current row's main trace values.
    /// This avoids allocating a new Vec per LDE domain point in the hot loop.
    /// The buffer must have length >= `lde_trace.num_main_cols()`.
    pub fn new_with_cache(
        lde_trace: &'a LDETraceTable<F, E>,
        row: usize,
        alpha_powers: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
        row_cache: &'a mut Vec<FieldElement<F>>,
    ) -> Self {
        // Fill the cache with the current row's main trace values (contiguous writes).
        let num_main_cols = lde_trace.num_main_cols();
        row_cache.clear();
        row_cache.reserve(num_main_cols);
        for col in 0..num_main_cols {
            row_cache.push(lde_trace.get_main(row, col).clone());
        }
        Self {
            lde_trace,
            row,
            step_size: lde_trace.lde_step_size,
            num_rows: lde_trace.num_rows(),
            accumulator: FieldElement::zero(),
            alpha_powers,
            constraint_idx: 0,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset_val: logup_table_offset,
            main_row_cache: row_cache.as_slice(),
        }
    }

    pub fn finish(self) -> FieldElement<E> {
        self.accumulator
    }
}

impl<'a, F, E> AirBuilder<E> for ProverBuilder<'a, F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    #[inline]
    fn main(&self, offset: usize, col: usize) -> FieldElement<E> {
        if offset == 0 && !self.main_row_cache.is_empty() {
            self.main_row_cache[col].clone().to_extension()
        } else {
            let lde_row = (self.row + offset * self.step_size) % self.num_rows;
            self.lde_trace.get_main(lde_row, col).clone().to_extension()
        }
    }

    #[inline]
    fn aux(&self, offset: usize, col: usize) -> FieldElement<E> {
        let lde_row = (self.row + offset * self.step_size) % self.num_rows;
        self.lde_trace.get_aux(lde_row, col).clone()
    }

    #[inline]
    fn assert_zero(&mut self, expr: FieldElement<E>) {
        // Use pre-computed alpha power — no E×E multiply per constraint
        self.accumulator = &self.accumulator + &self.alpha_powers[self.constraint_idx] * &expr;
        self.constraint_idx += 1;
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

impl<'a, F, E> MainAirBuilder<F, E> for ProverBuilder<'a, F, E>
where
    F: IsSubFieldOf<E> + IsFFTField + Send + Sync,
    E: IsField + Send + Sync,
{
    #[inline]
    fn main_base(&self, col: usize) -> FieldElement<F> {
        self.main_row_cache[col].clone()
    }

    #[inline]
    fn assert_zero_base(&mut self, expr: FieldElement<F>) {
        // F×E multiplication (3 base muls) with pre-computed alpha power (no E×E)
        self.accumulator = &self.accumulator + &expr * &self.alpha_powers[self.constraint_idx];
        self.constraint_idx += 1;
    }
}

pub struct VerifierBuilder<'a, E: IsField> {
    frame: &'a Frame<E, E>,
    accumulator: FieldElement<E>,
    alpha_powers: &'a [FieldElement<E>],
    constraint_idx: usize,
    rap_challenges: &'a [FieldElement<E>],
    logup_alpha_powers: &'a [FieldElement<E>],
    logup_table_offset_val: &'a FieldElement<E>,
}

impl<'a, E: IsField> VerifierBuilder<'a, E> {
    pub fn new(
        frame: &'a Frame<E, E>,
        alpha_powers: &'a [FieldElement<E>],
        rap_challenges: &'a [FieldElement<E>],
        logup_alpha_powers: &'a [FieldElement<E>],
        logup_table_offset: &'a FieldElement<E>,
    ) -> Self {
        Self {
            frame,
            accumulator: FieldElement::zero(),
            alpha_powers,
            constraint_idx: 0,
            rap_challenges,
            logup_alpha_powers,
            logup_table_offset_val: logup_table_offset,
        }
    }

    pub fn finish(self) -> FieldElement<E> {
        self.accumulator
    }
}

impl<'a, E: IsField> AirBuilder<E> for VerifierBuilder<'a, E> {
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
        self.accumulator = &self.accumulator + &self.alpha_powers[self.constraint_idx] * &expr;
        self.constraint_idx += 1;
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

/// For the verifier, F = E, so `main_base` returns extension field values from the frame.
impl<'a, E: IsField> MainAirBuilder<E, E> for VerifierBuilder<'a, E> {
    #[inline]
    fn main_base(&self, col: usize) -> FieldElement<E> {
        self.frame
            .get_evaluation_step(0)
            .get_main_evaluation_element(0, col)
            .clone()
    }

    #[inline]
    fn assert_zero_base(&mut self, expr: FieldElement<E>) {
        // Use pre-computed alpha power (same indexing as assert_zero)
        self.accumulator = &self.accumulator + &self.alpha_powers[self.constraint_idx] * &expr;
        self.constraint_idx += 1;
    }
}

/// Adapter that wraps `&mut dyn AirBuilder<E>` as `MainAirBuilder<E, E>`.
///
/// Used by `eval_constraints_with_builder` (verifier path) to call `main_builder_fn`
/// closures typed as `Fn(&mut dyn MainAirBuilder<F, E>)`. When F = E (verifier),
/// the adapter delegates `main_base` to `main(0, col)` and `assert_zero_base` to `assert_zero`.
pub struct AirBuilderAsMain<'a, E: IsField> {
    inner: &'a mut dyn AirBuilder<E>,
}

impl<'a, E: IsField> AirBuilderAsMain<'a, E> {
    pub fn new(inner: &'a mut dyn AirBuilder<E>) -> Self {
        Self { inner }
    }
}

impl<'a, E: IsField> MainAirBuilder<E, E> for AirBuilderAsMain<'a, E> {
    #[inline]
    fn main_base(&self, col: usize) -> FieldElement<E> {
        self.inner.main(0, col)
    }

    #[inline]
    fn assert_zero_base(&mut self, expr: FieldElement<E>) {
        self.inner.assert_zero(expr);
    }
}
