//! Differential tests for the single-source `ConstraintSet` table conversions
//! (group A: dvrm, shift, mul, lt, load, ecsm, ecdas, ec_scalar).
//!
//! Each table's new `XxxConstraints` implementing [`ConstraintSet`] is checked
//! against the OLD boxed constraint builder it transcribes (the old structs +
//! `*_constraints` builders stay in-branch as this oracle until the final
//! deletion phase), on [`TRIALS`] random rows — off-trace points, where a
//! weakened or slipped transcription diverges with overwhelming probability:
//!
//! 1. `ProverEvalFolder` output == old `evaluate_prover`;
//! 2. `VerifierEvalFolder` output == old `evaluate_verifier`;
//! 3. `CaptureBuilder` → flatten → interpret == old `evaluate_prover`;
//!
//! plus meta parity vs the old boxed objects (count, `num_base`, and per-idx
//! degree / period / offset / exemptions_period / periodic_exemptions_offset /
//! end_exemptions), and tree-measured degree (`CaptureBuilder::finish`) ==
//! declared `meta.degree`.

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

/// The full differential check for one table's `ConstraintSet` against the
/// OLD boxed constraint list (built via the old builder at `idx_start = 0`).
///
/// `num_cols` is the table's column count; frames are single-step (none of
/// these tables read next-row cells).
fn check_set_vs_old<CS>(
    label: &str,
    set: &CS,
    old: &[Box<dyn TransitionConstraintEvaluator<Gl, Gl3>>],
    num_cols: usize,
) where
    CS: ConstraintSet<Gl, Gl3>,
{
    let meta = set.meta();
    let n = meta.len();
    assert_eq!(old.len(), n, "[{label}] constraint count");

    // --- meta parity vs the old boxed objects ---
    let num_base = num_base_from_meta(&meta);
    assert_eq!(num_base, n, "[{label}] all-base num_base");
    // Build a lookup from the old objects by constraint_idx.
    let mut old_by_idx: Vec<Option<&Box<dyn TransitionConstraintEvaluator<Gl, Gl3>>>> =
        vec![None; n];
    for c in old.iter() {
        let i = c.constraint_idx();
        assert!(i < n, "[{label}] old constraint idx {i} out of range");
        assert!(old_by_idx[i].is_none(), "[{label}] duplicate old idx {i}");
        old_by_idx[i] = Some(c);
    }
    for (i, m) in meta.iter().enumerate() {
        let c = old_by_idx[i].expect("dense old idx");
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Base, "[{label}] meta kind {i}");
        assert_eq!(m.degree, c.degree(), "[{label}] degree {i}");
        assert_eq!(m.period, c.period(), "[{label}] period {i}");
        assert_eq!(m.offset, c.offset(), "[{label}] offset {i}");
        assert_eq!(
            m.exemptions_period,
            c.exemptions_period(),
            "[{label}] exemptions_period {i}"
        );
        assert_eq!(
            m.periodic_exemptions_offset,
            c.periodic_exemptions_offset(),
            "[{label}] periodic_exemptions_offset {i}"
        );
        assert_eq!(
            m.end_exemptions,
            c.end_exemptions(),
            "[{label}] end_exemptions {i}"
        );
    }

    // --- capture once; tree-measured degree <= declared (== old degree()) ---
    //
    // The declared `meta.degree` reproduces the OLD struct's `degree()` exactly
    // (asserted above at the meta-parity loop), which the engine uses as the
    // composition-poly degree bound. For most tables that bound is tight, so
    // tree-measured == declared. A few constraints (the ecsm/ecdas convolution
    // TAILS — `ConvCarry` at large `i`) legitimately have a lower EXACT degree
    // than their uniform declared bound: at limb `i` near the top, every
    // remaining product has a zeroed (constant) factor, so the surviving
    // expression is degree 1 while the struct declares 2 (resp. 3). The
    // soundness-relevant invariant is therefore `measured <= declared` (the
    // real degree must never EXCEED the bound the composition polynomial is
    // sized for) — an over-declaration is safe, an under-declaration is not.
    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    set.eval(&mut cb);
    let (prog, degrees) = cb.finish(num_base);
    assert_eq!(degrees.len(), n, "[{label}] one emit per constraint");
    for &(idx, measured) in &degrees {
        assert!(
            measured <= meta[idx].degree,
            "[{label}] constraint {idx}: tree degree {measured} EXCEEDS declared {}",
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

        // --- old prover-side reference: evaluate_prover into base_evals ---
        let frame = Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.clone()], vec![vec![]])]);
        let ctx = TransitionEvaluationContext::new_prover(
            &frame,
            &no_periodic,
            &no_ch,
            &no_ch,
            &offset_e,
            &shifts,
        );
        let mut old_base = vec![FE::zero(); n];
        let mut old_ext_scratch = vec![Fp3::zero(); n];
        for c in old.iter() {
            c.evaluate_prover(&ctx, &mut old_base, &mut old_ext_scratch);
        }

        // --- old verifier-side reference: evaluate_verifier into ext_evals ---
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
        let mut old_vext = vec![Fp3::zero(); n];
        for c in old.iter() {
            c.evaluate_verifier(&vctx, &mut old_vext);
        }

        // --- 1. ProverEvalFolder == old evaluate_prover ---
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

        // --- 2. VerifierEvalFolder == old evaluate_verifier ---
        let mut vext_out = vec![Fp3::zero(); n];
        let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
        set.eval(&mut vfolder);
        vfolder.assert_all_emitted();
        for i in 0..n {
            assert_eq!(
                vext_out[i], old_vext[i],
                "[{label}] verifier folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- 3. capture → flatten → interpret == old evaluate_prover ---
        for (i, expected) in old_base.iter().enumerate() {
            assert_eq!(
                &eval_program_base(&prog, i, &row),
                expected,
                "[{label}] interpreter mismatch, constraint {i}, trial {trial}"
            );
        }
    }
}

// =============================================================================
// lt.rs
// =============================================================================

mod lt {
    use super::*;
    use crate::tables::lt::{LtConstraints, cols, lt_constraints};
    use stark::constraints::transition::TransitionConstraint;

    #[test]
    fn lt_constraint_set_matches_old() {
        let (old, next) = lt_constraints(0);
        assert_eq!(next, old.len());
        let boxed: Vec<_> = old.into_iter().map(|c| c.boxed()).collect();
        check_set_vs_old("lt", &LtConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// dvrm.rs
// =============================================================================

mod dvrm {
    use super::*;
    use crate::tables::dvrm::{DvrmConstraints, cols, dvrm_constraints};
    use stark::constraints::transition::TransitionConstraint;

    #[test]
    fn dvrm_constraint_set_matches_old() {
        let (old, next) = dvrm_constraints(0);
        assert_eq!(next, old.len());
        let boxed: Vec<_> = old.into_iter().map(|c| c.boxed()).collect();
        check_set_vs_old("dvrm", &DvrmConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// shift.rs
// =============================================================================

mod shift {
    use super::*;
    use crate::tables::shift::{ShiftConstraints, cols, shift_constraints};
    use stark::constraints::transition::TransitionConstraint;

    #[test]
    fn shift_constraint_set_matches_old() {
        let (old, next) = shift_constraints(0);
        assert_eq!(next, old.len());
        let boxed: Vec<_> = old.into_iter().map(|c| c.boxed()).collect();
        check_set_vs_old("shift", &ShiftConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// mul.rs
// =============================================================================

mod mul {
    use super::*;
    use crate::tables::mul::{MulConstraints, cols, mul_constraints};
    use stark::constraints::transition::TransitionConstraint;

    #[test]
    fn mul_constraint_set_matches_old() {
        let (old, next) = mul_constraints(0);
        assert_eq!(next, old.len());
        let boxed: Vec<_> = old.into_iter().map(|c| c.boxed()).collect();
        check_set_vs_old("mul", &MulConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// load.rs
// =============================================================================

mod load {
    use super::*;
    use crate::tables::load::{LoadConstraints, cols, constraints as load_constraints};

    #[test]
    fn load_constraint_set_matches_old() {
        // `constraints()` already returns boxed evaluators (idx_start = 0).
        let boxed = load_constraints();
        check_set_vs_old("load", &LoadConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// ecsm.rs
// =============================================================================

mod ecsm {
    use super::*;
    use crate::tables::ecsm::{EcsmConstraints, cols, create_constraints};

    #[test]
    fn ecsm_constraint_set_matches_old() {
        let (boxed, next) = create_constraints(0);
        assert_eq!(next, boxed.len());
        check_set_vs_old("ecsm", &EcsmConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// ecdas.rs
// =============================================================================

mod ecdas {
    use super::*;
    use crate::tables::ecdas::{EcdasConstraints, cols, create_constraints};

    #[test]
    fn ecdas_constraint_set_matches_old() {
        let (boxed, next) = create_constraints(0);
        assert_eq!(next, boxed.len());
        check_set_vs_old("ecdas", &EcdasConstraints, &boxed, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// ec_scalar.rs
// =============================================================================

mod ec_scalar {
    use super::*;
    use crate::tables::ec_scalar::{EcScalarConstraints, cols, create_constraints};

    #[test]
    fn ec_scalar_constraint_set_matches_old() {
        let (boxed, next) = create_constraints(0);
        assert_eq!(next, boxed.len());
        check_set_vs_old("ec_scalar", &EcScalarConstraints, &boxed, cols::NUM_COLUMNS);
    }
}
