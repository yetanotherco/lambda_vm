//! Folder-vs-capture-interpret regression tests for the single-source
//! `ConstraintSet` table bodies (group A: dvrm, shift, mul, lt, load, ecsm,
//! ecdas, ec_scalar).
//!
//! Each table's single `eval` body is run three ways — the `ProverEvalFolder`
//! (base), the `VerifierEvalFolder` (extension), and the `CaptureBuilder` → flat
//! IR → `eval_program_base` interpreter — and asserted to agree on [`TRIALS`]
//! random off-trace rows. All three derive from the ONE body, so this pins that
//! capture/interpretation stays faithful to the compiled folder. We also assert
//! the meta invariants (dense, idx-ordered, all-base) and that each root's
//! tree-measured degree does not EXCEED its declared `meta.degree`.

use math::field::element::FieldElement;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, VerifierEvalFolder,
    num_base_from_meta,
};
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

/// Folder-vs-capture-interpret regression check for one table's
/// [`ConstraintSet`]. The single body is run three ways (prover folder, verifier
/// folder, captured-IR interpreter) and asserted to agree on random off-trace
/// rows — all three derive from the ONE body, so this pins that
/// capture/interpretation stays faithful to the compiled folder.
///
/// `num_cols` is the table's column count; frames are single-step (none of
/// these tables read next-row cells).
fn check_set<CS>(label: &str, set: &CS, num_cols: usize)
where
    CS: ConstraintSet<Gl, Gl3>,
{
    let meta = set.meta();
    let n = meta.len();

    // --- meta invariants: dense, idx-ordered, all-base (group-A tables). ---
    let num_base = num_base_from_meta(&meta);
    assert_eq!(num_base, n, "[{label}] all-base num_base");
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Base, "[{label}] meta kind {i}");
    }

    // --- capture once; tree-measured degree <= declared ---
    //
    // The declared `meta.degree` is what the engine uses as the composition-poly
    // degree bound. For most tables that bound is tight, so tree-measured ==
    // declared. A few constraints (the ecsm/ecdas convolution TAILS — `ConvCarry`
    // at large `i`) legitimately have a lower EXACT degree than their uniform
    // declared bound: at limb `i` near the top, every remaining product has a
    // zeroed (constant) factor, so the surviving expression is degree 1 while the
    // meta declares 2 (resp. 3). The soundness-relevant invariant is therefore
    // `measured <= declared` (the real degree must never EXCEED the bound the
    // composition polynomial is sized for) — over-declaration is safe,
    // under-declaration is not.
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

        // --- ProverEvalFolder (base) ---
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

        // --- VerifierEvalFolder (ext) ---
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

        // Prover folder (promoted) == verifier folder.
        for i in 0..n {
            assert_eq!(
                base_out[i].to_extension(),
                vext_out[i],
                "[{label}] prover-vs-verifier folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- capture → flatten → interpret == ProverEvalFolder (base) ---
        for (i, expected) in base_out.iter().enumerate() {
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
    use crate::tables::lt::{LtConstraints, cols};

    #[test]
    fn lt_constraint_set_folder_capture_agree() {
        check_set("lt", &LtConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// dvrm.rs
// =============================================================================

mod dvrm {
    use super::*;
    use crate::tables::dvrm::{DvrmConstraints, cols};

    #[test]
    fn dvrm_constraint_set_folder_capture_agree() {
        check_set("dvrm", &DvrmConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// shift.rs
// =============================================================================

mod shift {
    use super::*;
    use crate::tables::shift::{ShiftConstraints, cols};

    #[test]
    fn shift_constraint_set_folder_capture_agree() {
        check_set("shift", &ShiftConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// mul.rs
// =============================================================================

mod mul {
    use super::*;
    use crate::tables::mul::{MulConstraints, cols};

    #[test]
    fn mul_constraint_set_folder_capture_agree() {
        check_set("mul", &MulConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// load.rs
// =============================================================================

mod load {
    use super::*;
    use crate::tables::load::{LoadConstraints, cols};

    #[test]
    fn load_constraint_set_folder_capture_agree() {
        check_set("load", &LoadConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// ecsm.rs
// =============================================================================

mod ecsm {
    use super::*;
    use crate::tables::ecsm::{EcsmConstraints, cols};

    #[test]
    fn ecsm_constraint_set_folder_capture_agree() {
        check_set("ecsm", &EcsmConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// ecdas.rs
// =============================================================================

mod ecdas {
    use super::*;
    use crate::tables::ecdas::{EcdasConstraints, cols};

    #[test]
    fn ecdas_constraint_set_folder_capture_agree() {
        check_set("ecdas", &EcdasConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// ec_scalar.rs
// =============================================================================

mod ec_scalar {
    use super::*;
    use crate::tables::ec_scalar::{EcScalarConstraints, cols};

    #[test]
    fn ec_scalar_constraint_set_folder_capture_agree() {
        check_set("ec_scalar", &EcScalarConstraints, cols::NUM_COLUMNS);
    }
}
