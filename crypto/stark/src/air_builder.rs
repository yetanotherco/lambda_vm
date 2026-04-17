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
        self.lde_trace.get_main(lde_row, col).clone().to_extension()
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
