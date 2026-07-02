//! Differential tests for the per-table [`ConstraintSet`] single-body
//! conversions (PR B, group B tables).
//!
//! Each converted table exposes a new `XxxConstraints: ConstraintSet` whose
//! `meta()`/`eval()` must reproduce, index-for-index, the OLD boxed builder
//! function (`eq_constraints`, `store_constraints`, …) that still lives in the
//! same file as the differential oracle. For every table we assert, on
//! [`TRIALS`] random rows (off-trace points, where a weakened or slipped
//! transcription diverges with overwhelming probability):
//!
//! 1. `ProverEvalFolder` output == old `evaluate_prover` (base field);
//! 2. `VerifierEvalFolder` output == old `evaluate_verifier` (extension);
//! 3. `CaptureBuilder` → flatten → `eval_program_base` == old `evaluate_prover`;
//!
//! plus meta parity vs the old boxed objects (count, `num_base`, and per-idx
//! degree / period / offset / exemptions_period / periodic_exemptions_offset /
//! end_exemptions), and tree-measured degree (`CaptureBuilder::finish`) ==
//! declared `meta.degree`.
//!
//! All group-B tables read the current row only (offset 0) and are entirely
//! base-field, so `eval_program_base` (single `main_row`, row 0) is the
//! interpreter oracle.

use math::field::element::FieldElement;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, VerifierEvalFolder,
    num_base_from_meta,
};
use stark::constraints::transition::TransitionConstraintEvaluator;
use stark::frame::Frame;
use stark::lookup::PackingShifts;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;
type Fp3 = FieldElement<Gl3>;

const TRIALS: usize = 1000;

/// Deterministic SplitMix64.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The OLD boxed constraint builder result, used as the differential oracle.
type OldVec = Vec<Box<dyn TransitionConstraintEvaluator<Gl, Gl3>>>;

/// Run the full three-way differential + meta-parity check for one table.
///
/// * `old` — the OLD boxed constraints built at `idx_start = 0`.
/// * `set` — the NEW [`ConstraintSet`].
/// * `num_cols` — the table's `cols::NUM_COLUMNS`.
fn check_table<CS: ConstraintSet<Gl, Gl3>>(label: &str, old: &OldVec, set: &CS, num_cols: usize) {
    let n = old.len();
    let meta = set.meta();

    // --- count / meta parity vs the old boxed objects ---
    assert_eq!(meta.len(), n, "[{label}] constraint count");
    assert_eq!(
        num_base_from_meta(&meta),
        n,
        "[{label}] all-base num_base (group-B tables are entirely base-field)"
    );
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Base, "[{label}] meta kind {i}");
        // The old boxed objects expose their idx and zerofier params directly.
        let o = &old[i];
        assert_eq!(o.constraint_idx(), i, "[{label}] old idx {i} out of order");
        assert_eq!(m.degree, o.degree(), "[{label}] degree {i}");
        assert_eq!(m.period, o.period(), "[{label}] period {i}");
        assert_eq!(m.offset, o.offset(), "[{label}] offset {i}");
        assert_eq!(
            m.exemptions_period,
            o.exemptions_period(),
            "[{label}] exemptions_period {i}"
        );
        assert_eq!(
            m.periodic_exemptions_offset,
            o.periodic_exemptions_offset(),
            "[{label}] periodic_exemptions_offset {i}"
        );
        assert_eq!(
            m.end_exemptions,
            o.end_exemptions(),
            "[{label}] end_exemptions {i}"
        );
    }

    // --- capture once; tree-measured degree == declared ---
    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    set.eval(&mut cb);
    let (prog, degrees) = cb.finish(n);
    assert_eq!(degrees.len(), n, "[{label}] one emit per constraint");
    for &(idx, measured) in &degrees {
        assert_eq!(
            measured, meta[idx].degree,
            "[{label}] constraint {idx}: tree degree {measured} != declared {}",
            meta[idx].degree
        );
    }

    let shifts = PackingShifts::<Gl>::new();
    let vshifts = PackingShifts::<Gl3>::new();
    let no_periodic: Vec<FE> = vec![];
    let no_periodic_e: Vec<Fp3> = vec![];
    let no_ch: Vec<Fp3> = vec![];
    let offset_e = Fp3::zero();

    let mut rng = SplitMix64(0x5EED_0000_0000_0000 ^ label.len() as u64);
    for trial in 0..TRIALS {
        let row: Vec<FE> = (0..num_cols).map(|_| FE::from(rng.next_u64())).collect();
        let row_e: Vec<Fp3> = row.iter().map(|x| x.to_extension()).collect();

        // --- OLD oracle: prover (base) + verifier (ext) evaluations ---
        let old_frame =
            Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.clone()], vec![vec![]])]);
        let old_ctx = TransitionEvaluationContext::new_prover(
            &old_frame,
            &no_periodic,
            &no_ch,
            &no_ch,
            &offset_e,
            &shifts,
        );
        let mut old_base = vec![FE::zero(); n];
        let mut old_ext_p = vec![Fp3::zero(); n];
        for c in old.iter() {
            c.evaluate_prover(&old_ctx, &mut old_base, &mut old_ext_p);
        }

        let old_frame_e =
            Frame::<Gl3, Gl3>::new(vec![TableView::new(vec![row_e.clone()], vec![vec![]])]);
        let old_vctx = TransitionEvaluationContext::<Gl, Gl3>::new_verifier(
            &old_frame_e,
            &no_periodic_e,
            &no_ch,
            &no_ch,
            &offset_e,
            &vshifts,
        );
        let mut old_ext_v = vec![Fp3::zero(); n];
        for c in old.iter() {
            c.evaluate_verifier(&old_vctx, &mut old_ext_v);
        }

        // --- 1. ProverEvalFolder == old evaluate_prover (base) ---
        let frame = Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.clone()], vec![vec![]])]);
        let ctx = TransitionEvaluationContext::new_prover(
            &frame,
            &no_periodic,
            &no_ch,
            &no_ch,
            &offset_e,
            &shifts,
        );
        let mut base_out = vec![FE::zero(); n];
        let mut ext_out = vec![Fp3::zero(); n];
        let mut folder = ProverEvalFolder::new(&ctx, &mut base_out, &mut ext_out);
        set.eval(&mut folder);
        folder.assert_all_emitted();
        for i in 0..n {
            assert_eq!(
                base_out[i], old_base[i],
                "[{label}] prover folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- 2. VerifierEvalFolder == old evaluate_verifier (ext) ---
        let frame_e =
            Frame::<Gl3, Gl3>::new(vec![TableView::new(vec![row_e.clone()], vec![vec![]])]);
        let vctx = TransitionEvaluationContext::<Gl, Gl3>::new_verifier(
            &frame_e,
            &no_periodic_e,
            &no_ch,
            &no_ch,
            &offset_e,
            &vshifts,
        );
        let mut vext_out = vec![Fp3::zero(); n];
        let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
        set.eval(&mut vfolder);
        vfolder.assert_all_emitted();
        for i in 0..n {
            assert_eq!(
                vext_out[i], old_ext_v[i],
                "[{label}] verifier folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- 3. capture → flatten → interpret == old evaluate_prover (base) ---
        for i in 0..n {
            assert_eq!(
                eval_program_base(&prog, i, &row),
                old_base[i],
                "[{label}] interpreter mismatch, constraint {i}, trial {trial}"
            );
        }
    }
}

// =============================================================================
// eq.rs
// =============================================================================

mod eq {
    use super::*;
    use crate::tables::eq::{EqConstraints, cols, eq_constraints};

    #[test]
    fn eq_constraint_set_matches_old() {
        let (old, _) = eq_constraints(0);
        check_table("eq", &old, &EqConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// store.rs
// =============================================================================

mod store {
    use super::*;
    use crate::tables::store::{StoreConstraints, cols, store_constraints};

    #[test]
    fn store_constraint_set_matches_old() {
        let (old, _) = store_constraints(0);
        check_table("store", &old, &StoreConstraints, cols::NUM_COLUMNS);
    }
}
