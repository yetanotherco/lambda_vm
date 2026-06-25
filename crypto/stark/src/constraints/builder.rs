//! Monomorphized constraint folding for prover and verifier.
//!
//! A table's `eval` body hands a [`ConstraintBuilder`] each constraint's **residual**
//! (the raw constraint value, 0 on the trace domain when the witness is valid) via
//! `fold` / `fold_ext`. The same `eval` body, monomorphized over the builder type,
//! drives both the prover (folding over the LDE) and the verifier (folding at the OOD
//! point `z`) — no `Frame`, no boxed-constraint dispatch.
//!
//! A *residual* is not "asserted" to be zero: it generally is not zero on the LDE.
//! The builder weights it by the inverse-zerofier and the coefficient and accumulates
//! it into the composition (quotient).

use crate::constraints::row_view::RowView;
use crate::frame::Frame;
use crate::lookup::PackingShifts;
use crate::trace::LDETraceTable;
use crate::traits::ZerofierEvaluations;
use math::field::{
    element::FieldElement,
    traits::{IsField, IsSubFieldOf},
};

/// What a table's `eval` body writes constraint residuals into.
///
/// `F` is the main field — the base field for the prover, the extension for the
/// verifier; `E` is the extension. Domain constraints fold an `F` residual; LogUp
/// constraints fold an `E` residual. The same `eval` body is monomorphized over each
/// concrete builder, so intermediate values are computed once and shared, and there
/// is no per-row allocation or dynamic dispatch.
pub trait ConstraintBuilder {
    /// Main field: base field on the prover, extension on the verifier.
    type F: IsSubFieldOf<Self::E>;
    /// Extension field.
    type E: IsField;

    /// Current-row main cell.
    fn main(&self, col: usize) -> &FieldElement<Self::F>;
    /// Current-row aux cell.
    fn aux(&self, col: usize) -> &FieldElement<Self::E>;
    /// Next-row aux cell — only the LogUp running-sum constraint needs this.
    fn aux_next(&self, col: usize) -> &FieldElement<Self::E>;

    /// Fold a base-field (domain) constraint residual.
    fn fold(&mut self, residual: FieldElement<Self::F>);
    /// Fold an extension-field (LogUp) constraint residual.
    fn fold_ext(&mut self, residual: FieldElement<Self::E>);
}

/// Row-independent context a table's `eval` reads alongside the builder: LogUp
/// challenges and the periodic-column values at the current row. Passed as a second
/// argument to the eval so the builder stays focused on row access + folding.
///
/// `F` is the main field (base on the prover, extension on the verifier), `E` the
/// extension — matching the builder's associated types.
#[derive(Clone, Copy)]
pub struct ConstraintContext<'a, F: IsField, E: IsField> {
    /// RAP/LogUp challenges (e.g. `z = rap_challenges[0]`).
    pub rap_challenges: &'a [FieldElement<E>],
    /// Precomputed powers of the LogUp `alpha` challenge.
    pub logup_alpha_powers: &'a [FieldElement<E>],
    /// `table_contribution / N`, added by the LogUp accumulated constraint.
    pub logup_table_offset: &'a FieldElement<E>,
    /// Precomputed packing-shift constants for fingerprint combining.
    pub packing_shifts: &'a PackingShifts<F>,
    /// Periodic-column values at the current row.
    pub periodic: &'a [FieldElement<F>],
}

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
    /// One running sum per zerofier group (`Σ residual·coeff` over the constraints in
    /// that group). The shared zerofier is applied once per group in `finish`, not
    /// once per term — fewer multiplications when constraints share a zerofier.
    group_sums: Vec<FieldElement<E>>,
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
            group_sums: vec![FieldElement::<E>::zero(); zerofier.groups.len()],
            idx: 0,
        }
    }

    /// Consume the folder: apply each group's zerofier to its running sum once and
    /// sum across groups — `Σ_g zerofier_inv(group g, row) · group_sum[g]`. Equals the
    /// per-term fold exactly (field distributivity), so proofs stay byte-identical.
    /// (Boundary contribution is added by the caller.)
    pub fn finish(self) -> FieldElement<E> {
        let mut acc = FieldElement::<E>::zero();
        for (g, sum) in self.group_sums.iter().enumerate() {
            let group = &self.zerofier.groups[g];
            let z = &group[self.row % group.len()];
            acc = &acc + &(z * sum);
        }
        acc
    }
}

impl<'a, F, E> ConstraintBuilder for ProverConstraintBuilder<'a, F, E>
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    type F = F;
    type E = E;

    #[inline]
    fn main(&self, col: usize) -> &FieldElement<F> {
        self.view.get_main(0, 0, col)
    }

    #[inline]
    fn aux(&self, col: usize) -> &FieldElement<E> {
        self.view.get_aux(0, 0, col)
    }

    #[inline]
    fn aux_next(&self, col: usize) -> &FieldElement<E> {
        self.view.get_aux(1, 0, col)
    }

    /// Accumulate `residual · coeff[idx]` into this constraint's zerofier-group running
    /// sum; the shared zerofier is applied once per group in `finish`.
    fn fold(&mut self, residual: FieldElement<F>) {
        let g = self.zerofier.constraint_to_group[self.idx];
        self.group_sums[g] = &self.group_sums[g] + &(&residual * &self.coeffs[self.idx]);
        self.idx += 1;
    }

    fn fold_ext(&mut self, residual: FieldElement<E>) {
        let g = self.zerofier.constraint_to_group[self.idx];
        self.group_sums[g] = &self.group_sums[g] + &(&residual * &self.coeffs[self.idx]);
        self.idx += 1;
    }
}

/// Folds a table's constraints at the out-of-domain point `z` (verifier side).
///
/// Same residual fold as [`ProverConstraintBuilder`], but the residuals are evaluated
/// at `z` (all in the extension field), the zerofier per constraint is the single
/// inverse-zerofier value `evaluate_zerofier(z)` (not an LDE vector), and the row
/// source is the OOD trace-evaluation frame (`into_frame`): step 0 holds the `z`
/// evaluations, step 1 the `z·g` (next-row) evaluations. Runs once per proof (cold),
/// so it keeps the simple per-term fold.
pub struct VerifierConstraintBuilder<'a, E: IsField> {
    frame: &'a Frame<E, E>,
    zerofiers_z: &'a [FieldElement<E>],
    coeffs: &'a [FieldElement<E>],
    acc: FieldElement<E>,
    idx: usize,
}

impl<'a, E: IsField> VerifierConstraintBuilder<'a, E> {
    /// `zerofiers_z[c]` is the inverse-zerofier at `z` for constraint `c` (the
    /// verifier's `evaluate_zerofier` result), indexed by `constraint_idx`.
    pub fn new(
        frame: &'a Frame<E, E>,
        zerofiers_z: &'a [FieldElement<E>],
        coeffs: &'a [FieldElement<E>],
    ) -> Self {
        Self {
            frame,
            zerofiers_z,
            coeffs,
            acc: FieldElement::<E>::zero(),
            idx: 0,
        }
    }

    /// Consume the folder and return the accumulated transition value at `z`.
    pub fn finish(self) -> FieldElement<E> {
        self.acc
    }
}

impl<'a, E: IsField> ConstraintBuilder for VerifierConstraintBuilder<'a, E> {
    type F = E;
    type E = E;

    #[inline]
    fn main(&self, col: usize) -> &FieldElement<E> {
        self.frame
            .get_evaluation_step(0)
            .get_main_evaluation_element(0, col)
    }

    #[inline]
    fn aux(&self, col: usize) -> &FieldElement<E> {
        self.frame
            .get_evaluation_step(0)
            .get_aux_evaluation_element(0, col)
    }

    #[inline]
    fn aux_next(&self, col: usize) -> &FieldElement<E> {
        self.frame
            .get_evaluation_step(1)
            .get_aux_evaluation_element(0, col)
    }

    /// `acc += zerofier_z[idx] · residual · coeff[idx]`, mirroring `verifier.rs`'s
    /// `beta * eval * denominator`.
    fn fold(&mut self, residual: FieldElement<E>) {
        let term = &self.zerofiers_z[self.idx] * &residual * &self.coeffs[self.idx];
        self.acc = &self.acc + &term;
        self.idx += 1;
    }

    fn fold_ext(&mut self, residual: FieldElement<E>) {
        let term = &self.zerofiers_z[self.idx] * &residual * &self.coeffs[self.idx];
        self.acc = &self.acc + &term;
        self.idx += 1;
    }
}

/// A table's constraint set, as an object-safe per-table interface.
///
/// Each VM table provides one implementation whose two methods both call a single
/// generic `eval<CB: ConstraintBuilder>` body — so the constraint logic is written
/// once and monomorphized for the prover and the verifier. The prover/verifier hold
/// this boxed and call it ONCE per table per row (the only remaining dynamic
/// dispatch); the monomorphized eval body then runs straight-line, sharing
/// intermediate values across the table's constraints.
pub trait TableConstraints<F, E>: Send + Sync
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    /// Fold every residual of this table into the prover folder (over the LDE).
    fn eval_prover(&self, cb: &mut ProverConstraintBuilder<F, E>, ctx: &ConstraintContext<F, E>);
    /// Fold every residual of this table into the verifier folder (at `z`).
    fn eval_verifier(&self, cb: &mut VerifierConstraintBuilder<E>, ctx: &ConstraintContext<E, E>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::TableView;
    use math::field::goldilocks::GoldilocksField;

    type F = GoldilocksField;

    fn fe(x: u64) -> FieldElement<F> {
        FieldElement::<F>::from(x)
    }

    /// A constraint `eval` body generic over the builder: fold `main[0]` as a domain
    /// residual, then `aux[0]` as a LogUp residual. Used to prove the same body runs
    /// over both the prover and verifier builders.
    fn fold_main0_then_aux0<CB: ConstraintBuilder>(cb: &mut CB) {
        let m = cb.main(0).clone();
        cb.fold(m);
        let a = cb.aux(0).clone();
        cb.fold_ext(a);
    }

    /// The folder must accumulate exactly Σ zerofier(idx,row)·value·coeff[idx], grouped
    /// by zerofier, including the `lde_idx % group.len()` wrap and base/ext constraints
    /// sharing one running index.
    #[test]
    fn folds_zerofier_coeff_value_in_constraint_order() {
        let n = 4;
        let main_columns: Vec<Vec<FieldElement<F>>> = vec![(0..n).map(|r| fe(r as u64)).collect()];
        let aux_columns: Vec<Vec<FieldElement<F>>> = vec![(0..n).map(|r| fe(r as u64)).collect()];
        let lde = LDETraceTable::<F, F>::from_columns(main_columns, aux_columns, 1, 1);

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

    /// `main`/`aux` read the current row (offset 0); `aux_next` reads the next step
    /// (offset 1) — the only multi-row access the builder exposes.
    #[test]
    fn accessors_read_current_and_next_row() {
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

    /// Verifier folder accumulates Σ zerofier_z[idx]·residual·coeff[idx], the same
    /// combination `verifier.rs` computes (`beta * eval * denominator`).
    #[test]
    fn verifier_folds_zerofier_coeff_residual() {
        let frame = Frame::<F, F>::preallocate(2, 1, 1, 1);
        let zerofiers_z = [fe(2), fe(3), fe(5)];
        let coeffs = [fe(100), fe(200), fe(300)];
        let r0 = fe(9);
        let r1 = fe(8);
        let r2 = fe(6);

        let mut cb = VerifierConstraintBuilder::<F>::new(&frame, &zerofiers_z, &coeffs);
        cb.fold(r0.clone());
        cb.fold(r1.clone());
        cb.fold_ext(r2.clone());
        let got = cb.finish();

        let expected = &zerofiers_z[0] * &r0 * &coeffs[0]
            + &zerofiers_z[1] * &r1 * &coeffs[1]
            + &zerofiers_z[2] * &r2 * &coeffs[2];
        assert_eq!(got, expected);
    }

    /// `main`/`aux` read step 0 (`z`); `aux_next` reads step 1 (`z·g`).
    #[test]
    fn verifier_accessors_read_z_and_next_row() {
        let step0 = TableView::new(vec![vec![fe(10)]], vec![vec![fe(20)]]);
        let step1 = TableView::new(vec![vec![fe(0)]], vec![vec![fe(21)]]);
        let frame = Frame::new(vec![step0, step1]);
        let zerofiers_z = [fe(1)];
        let coeffs = [fe(1)];

        let cb = VerifierConstraintBuilder::<F>::new(&frame, &zerofiers_z, &coeffs);
        assert_eq!(cb.main(0), &fe(10));
        assert_eq!(cb.aux(0), &fe(20));
        assert_eq!(cb.aux_next(0), &fe(21));
    }

    /// The SAME generic `eval` body runs over the prover builder (folds over the LDE).
    #[test]
    fn generic_eval_runs_over_prover_builder() {
        let n = 4;
        let main_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(r as u64 + 1)).collect()]; // main[0][row] = row+1
        let aux_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(50 + r as u64)).collect()]; // aux[0][row] = 50+row
        let lde = LDETraceTable::<F, F>::from_columns(main_columns, aux_columns, 1, 1);

        let group0 = vec![fe(2), fe(3), fe(5), fe(7)];
        let group1 = vec![fe(11), fe(13), fe(17), fe(19)];
        let zerofier = ZerofierEvaluations::<F> {
            groups: vec![group0.clone(), group1.clone()],
            constraint_to_group: vec![0, 1],
        };
        let coeffs = [fe(100), fe(200)];
        let row = 2usize;

        let mut cb = ProverConstraintBuilder::<F, F>::new(&lde, row, &zerofier, &coeffs);
        fold_main0_then_aux0(&mut cb);
        let got = cb.finish();

        // residual0 = main[0][2] = 3 (domain); residual1 = aux[0][2] = 52 (logup).
        let expected =
            &group0[2] * &fe(3) * &coeffs[0] + &group1[2] * &fe(52) * &coeffs[1];
        assert_eq!(got, expected);
    }

    /// The SAME generic `eval` body runs over the verifier builder (folds at `z`).
    #[test]
    fn generic_eval_runs_over_verifier_builder() {
        let step0 = TableView::new(vec![vec![fe(3)]], vec![vec![fe(52)]]);
        let step1 = TableView::new(vec![vec![fe(0)]], vec![vec![fe(0)]]);
        let frame = Frame::new(vec![step0, step1]);
        let zerofiers_z = [fe(5), fe(17)];
        let coeffs = [fe(100), fe(200)];

        let mut cb = VerifierConstraintBuilder::<F>::new(&frame, &zerofiers_z, &coeffs);
        fold_main0_then_aux0(&mut cb);
        let got = cb.finish();

        let expected =
            &zerofiers_z[0] * &fe(3) * &coeffs[0] + &zerofiers_z[1] * &fe(52) * &coeffs[1];
        assert_eq!(got, expected);
    }

    struct DummyTable;
    impl TableConstraints<F, F> for DummyTable {
        fn eval_prover(&self, cb: &mut ProverConstraintBuilder<F, F>, _ctx: &ConstraintContext<F, F>) {
            fold_main0_then_aux0(cb);
        }
        fn eval_verifier(&self, cb: &mut VerifierConstraintBuilder<F>, _ctx: &ConstraintContext<F, F>) {
            fold_main0_then_aux0(cb);
        }
    }

    /// The same table-constraints object drives both builders through an object-safe
    /// boxed call (the per-table dispatch), each folding correctly.
    #[test]
    fn table_constraints_dispatches_to_both_builders_via_box() {
        let table: Box<dyn TableConstraints<F, F>> = Box::new(DummyTable);
        let zero = fe(0);
        let shifts = PackingShifts::<F>::new();
        let ctx = ConstraintContext {
            rap_challenges: &[],
            logup_alpha_powers: &[],
            logup_table_offset: &zero,
            packing_shifts: &shifts,
            periodic: &[],
        };

        // Prover side: fold over a 1-col LDE at row 2.
        let n = 4;
        let main_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(r as u64 + 1)).collect()];
        let aux_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(50 + r as u64)).collect()];
        let lde = LDETraceTable::<F, F>::from_columns(main_columns, aux_columns, 1, 1);
        let zerofier = ZerofierEvaluations::<F> {
            groups: vec![
                vec![fe(2), fe(3), fe(5), fe(7)],
                vec![fe(11), fe(13), fe(17), fe(19)],
            ],
            constraint_to_group: vec![0, 1],
        };
        let coeffs = [fe(100), fe(200)];
        let mut pcb = ProverConstraintBuilder::<F, F>::new(&lde, 2, &zerofier, &coeffs);
        table.eval_prover(&mut pcb, &ctx);
        let prover_expected = &zerofier.groups[0][2] * &fe(3) * &coeffs[0]
            + &zerofier.groups[1][2] * &fe(52) * &coeffs[1];
        assert_eq!(pcb.finish(), prover_expected);

        // Verifier side: fold at z.
        let step0 = TableView::new(vec![vec![fe(3)]], vec![vec![fe(52)]]);
        let step1 = TableView::new(vec![vec![fe(0)]], vec![vec![fe(0)]]);
        let frame = Frame::new(vec![step0, step1]);
        let zerofiers_z = [fe(5), fe(17)];
        let mut vcb = VerifierConstraintBuilder::<F>::new(&frame, &zerofiers_z, &coeffs);
        table.eval_verifier(&mut vcb, &ctx);
        let verifier_expected =
            &zerofiers_z[0] * &fe(3) * &coeffs[0] + &zerofiers_z[1] * &fe(52) * &coeffs[1];
        assert_eq!(vcb.finish(), verifier_expected);
    }

    /// A generic eval that reads a LogUp challenge from the context: fold main[0]
    /// (domain) and aux[0]*z (logup), proving the context is threaded through.
    fn fold_with_challenge<CB: ConstraintBuilder>(
        cb: &mut CB,
        ctx: &ConstraintContext<CB::F, CB::E>,
    ) {
        let m = cb.main(0).clone();
        cb.fold(m);
        let weighted = cb.aux(0) * &ctx.rap_challenges[0];
        cb.fold_ext(weighted);
    }

    struct ChallengeTable;
    impl TableConstraints<F, F> for ChallengeTable {
        fn eval_prover(&self, cb: &mut ProverConstraintBuilder<F, F>, ctx: &ConstraintContext<F, F>) {
            fold_with_challenge(cb, ctx);
        }
        fn eval_verifier(&self, cb: &mut VerifierConstraintBuilder<F>, ctx: &ConstraintContext<F, F>) {
            fold_with_challenge(cb, ctx);
        }
    }

    /// The eval reads `ctx.rap_challenges[0]` and folds with it — verifies context
    /// threading end-to-end through TableConstraints (prover side).
    #[test]
    fn eval_reads_challenge_from_context() {
        let n = 4;
        let main_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(r as u64 + 1)).collect()];
        let aux_columns: Vec<Vec<FieldElement<F>>> =
            vec![(0..n).map(|r| fe(50 + r as u64)).collect()];
        let lde = LDETraceTable::<F, F>::from_columns(main_columns, aux_columns, 1, 1);
        let zerofier = ZerofierEvaluations::<F> {
            groups: vec![
                vec![fe(2), fe(3), fe(5), fe(7)],
                vec![fe(11), fe(13), fe(17), fe(19)],
            ],
            constraint_to_group: vec![0, 1],
        };
        let coeffs = [fe(100), fe(200)];

        let z = fe(9);
        let shifts = PackingShifts::<F>::new();
        let zero = fe(0);
        let rap = [z.clone()];
        let ctx = ConstraintContext {
            rap_challenges: &rap,
            logup_alpha_powers: &[],
            logup_table_offset: &zero,
            packing_shifts: &shifts,
            periodic: &[],
        };

        let table = ChallengeTable;
        let mut pcb = ProverConstraintBuilder::<F, F>::new(&lde, 2, &zerofier, &coeffs);
        table.eval_prover(&mut pcb, &ctx);

        let expected = &zerofier.groups[0][2] * &fe(3) * &coeffs[0]
            + &zerofier.groups[1][2] * &(&fe(52) * &z) * &coeffs[1];
        assert_eq!(pcb.finish(), expected);
    }
}
