//! Monomorphized constraint folding for the prover.
//!
//! `ProverConstraintBuilder` replaces the `Frame` + boxed-`TransitionConstraintEvaluator`
//! per-row machinery: a table's `eval` body hands the folder each constraint's
//! **residual** (via `fold` / `fold_ext`), and each residual is folded inline into a
//! single accumulator using the existing per-group zerofier evaluations and
//! transition coefficients — the same value the current `evaluate_transitions` loop
//! computes, so proofs stay byte-identical.
//!
//! A *residual* is the raw constraint polynomial value at the row: it vanishes on
//! the trace domain when the witness is valid, but generally not on the LDE — so we
//! don't "assert" it is zero, we weight it by the inverse-zerofier and the
//! coefficient and accumulate it into the composition (quotient).
//!
//! Constraints are emitted in `constraint_idx` order (base/domain first, then the
//! extension LogUp constraints), so `coeffs[idx]` and `zerofier.get(idx, row)` line
//! up exactly with today's indexing.

use crate::constraints::row_view::RowView;
use crate::trace::LDETraceTable;
use crate::traits::ZerofierEvaluations;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

/// Folds a table's constraints at one LDE row into the composition value.
pub struct ProverConstraintBuilder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    view: RowView<'a, F, E>,
    zerofier: &'a ZerofierEvaluations<F>,
    coeffs: &'a [FieldElement<E>],
    row: usize,
    acc: FieldElement<E>,
    idx: usize,
}

impl<'a, F, E> ProverConstraintBuilder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    /// Build a folder anchored at LDE row `row`.
    pub fn new(
        lde_trace: &'a LDETraceTable<F, E>,
        row: usize,
        zerofier: &'a ZerofierEvaluations<F>,
        coeffs: &'a [FieldElement<E>],
    ) -> Self {
        Self {
            view: RowView::new(lde_trace, row),
            zerofier,
            coeffs,
            row,
            acc: FieldElement::<E>::zero(),
            idx: 0,
        }
    }

    /// Current-row main cell.
    #[inline]
    pub fn main(&self, col: usize) -> &FieldElement<F> {
        self.view.get_main(0, 0, col)
    }

    /// Current-row aux cell.
    #[inline]
    pub fn aux(&self, col: usize) -> &FieldElement<E> {
        self.view.get_aux(0, 0, col)
    }

    /// Next-row aux cell (offset 1) — only the LogUp running-sum constraint needs this.
    #[inline]
    pub fn aux_next(&self, col: usize) -> &FieldElement<E> {
        self.view.get_aux(1, 0, col)
    }

    /// Fold a base-field (domain) constraint **residual** into the composition
    /// value: `acc += zerofier_inv(idx, row) · residual · coeff[idx]`, then advance
    /// the index. (`zerofier.get` returns the inverse-zerofier evaluation.)
    pub fn fold(&mut self, residual: FieldElement<F>) {
        let term = (self.zerofier.get(self.idx, self.row) * &residual) * &self.coeffs[self.idx];
        self.acc = &self.acc + &term;
        self.idx += 1;
    }

    /// Fold an extension-field (LogUp) residual. Same fold as [`fold`](Self::fold)
    /// but over the extension field.
    pub fn fold_ext(&mut self, residual: FieldElement<E>) {
        let term = (self.zerofier.get(self.idx, self.row) * &residual) * &self.coeffs[self.idx];
        self.acc = &self.acc + &term;
        self.idx += 1;
    }

    /// Consume the folder and return the accumulated transition value at this row.
    /// (Boundary contribution is added by the caller.)
    pub fn finish(self) -> FieldElement<E> {
        self.acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;

    fn fe(x: u64) -> FieldElement<F> {
        FieldElement::<F>::from(x)
    }

    /// The folder must accumulate exactly Σ zerofier(idx,row)·value·coeff[idx],
    /// matching `evaluate_transitions`, including the `lde_idx % group.len()` wrap
    /// and base/ext constraints sharing one running index.
    #[test]
    fn folds_zerofier_coeff_value_in_constraint_order() {
        // Minimal LDE just so the builder can be constructed; the fold values come
        // from the assert_* arguments, not from the row cells.
        let n = 4;
        let main_columns: Vec<Vec<FieldElement<F>>> = vec![(0..n).map(|r| fe(r as u64)).collect()];
        let aux_columns: Vec<Vec<FieldElement<F>>> = vec![(0..n).map(|r| fe(r as u64)).collect()];
        let lde = LDETraceTable::<F, F>::from_columns(main_columns, aux_columns, 1, 1);

        // Two zerofier groups: constraints 0,1 share group0 (len 4); constraint 2 is
        // group1 (len 2) to exercise the modulo wrap.
        let group0: Vec<FieldElement<F>> = vec![fe(2), fe(3), fe(5), fe(7)];
        let group1: Vec<FieldElement<F>> = vec![fe(11), fe(13)];
        let zerofier = ZerofierEvaluations::<F> {
            groups: vec![group0.clone(), group1.clone()],
            constraint_to_group: vec![0, 0, 1],
        };

        let coeffs = [fe(100), fe(200), fe(300)];

        let row = 3usize; // group1 wraps: 3 % 2 == 1 -> group1[1]
        let v0 = fe(9);
        let v1 = fe(8);
        let w0 = fe(6);

        let mut cb = ProverConstraintBuilder::<F, F>::new(&lde, row, &zerofier, &coeffs);
        cb.fold(v0.clone());
        cb.fold(v1.clone());
        cb.fold_ext(w0.clone());
        let got = cb.finish();

        let expected = &group0[3] * &v0 * &coeffs[0]
            + &group0[3] * &v1 * &coeffs[1]
            + &group1[1] * &w0 * &coeffs[2];

        assert_eq!(got, expected);
    }

    /// `main`/`aux` read the current row (offset 0); `aux_next` reads the next
    /// step (offset 1) — the only multi-row access the builder exposes.
    #[test]
    fn accessors_read_current_and_next_row() {
        // trace_step=1, blowup=2 -> lde_step_size=2.
        let n = 8;
        let main_columns: Vec<Vec<FieldElement<F>>> = vec![(0..n).map(|r| fe(r as u64)).collect()];
        let aux_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(100 + r as u64)).collect()];
        let lde = LDETraceTable::<F, F>::from_columns(main_columns, aux_columns, 1, 2);

        let zerofier = ZerofierEvaluations::<F> {
            groups: vec![vec![fe(1)]],
            constraint_to_group: vec![0],
        };
        let coeffs = [fe(1)];

        let cb = ProverConstraintBuilder::<F, F>::new(&lde, 1, &zerofier, &coeffs);

        // anchor row=1: main[0][1]=1, aux[0][1]=101, aux_next = aux[0][(1+2)%8]=103.
        assert_eq!(cb.main(0), &fe(1));
        assert_eq!(cb.aux(0), &fe(101));
        assert_eq!(cb.aux_next(0), &fe(103));
    }
}
