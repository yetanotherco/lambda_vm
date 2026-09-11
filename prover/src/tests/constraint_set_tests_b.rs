//! Folder-vs-capture-interpret regression tests for the per-table
//! [`ConstraintSet`] single bodies (group B tables).
//!
//! Each table's single `eval` body is exercised three ways — the
//! `ProverEvalFolder` (base), the `VerifierEvalFolder` (extension), and the
//! `CaptureBuilder` → flat IR → `eval_program_base` interpreter — and we assert
//! they agree on [`TRIALS`] random off-trace rows. All three derive from the
//! ONE body, so this pins that capture/interpretation stays faithful to the
//! compiled folder (the GPU/interpreter path a divergence would silently break).
//! We also assert the meta invariants (dense, idx-ordered, all-base) and that
//! each root's tree-measured degree equals its declared `meta.degree`.
//!
//! All group-B tables read the current row only (offset 0) and are entirely
//! base-field, so `eval_program_base` (single `main_row`, row 0) is the
//! interpreter entry point.

use math::field::element::FieldElement;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintSet, ProverEvalFolder, RootKind, VerifierEvalFolder,
    num_base_from_meta,
};
use stark::frame::Frame;
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

/// Run the folder-vs-capture-interpret differential + meta invariants for one
/// table's [`ConstraintSet`]. All three interpretations derive from the ONE
/// single-source body, so agreement across them (on random off-trace rows) is a
/// permanent regression guard that capture/interpretation stays faithful to the
/// compiled folder.
///
/// * `set` — the table's [`ConstraintSet`].
/// * `num_cols` — the table's `cols::NUM_COLUMNS`.
fn check_table<CS: ConstraintSet<Gl, Gl3>>(label: &str, set: &CS, num_cols: usize) {
    let meta = set.meta();
    let n = meta.len();

    // --- meta invariants: dense, idx-ordered, all-base (group-B tables). ---
    assert_eq!(
        num_base_from_meta(&meta),
        n,
        "[{label}] all-base num_base (group-B tables are entirely base-field)"
    );
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Base, "[{label}] meta kind {i}");
    }

    // --- capture once; tree-measured degree <= the table's declared max ---
    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    set.eval(&mut cb);
    let (prog, degrees) = cb.finish(n);
    assert_eq!(degrees.len(), n, "[{label}] one emit per constraint");
    // Release-safe exact-once check: the emitted indices must be exactly
    // 0..n. The per-emit EmitTracker only exists under debug_assertions,
    // which CI's --release test build compiles out; this assert catches a
    // double-emit/skip typo (count still == n) in any build profile.
    let mut emitted: Vec<usize> = degrees.iter().map(|&(idx, _)| idx).collect();
    emitted.sort_unstable();
    assert!(
        emitted.iter().enumerate().all(|(i, &idx)| i == idx),
        "[{label}] emitted constraint indices are not exactly 0..{n}: {emitted:?}"
    );
    let max_degree = set.max_degree();
    for &(idx, measured) in &degrees {
        assert!(
            measured <= max_degree,
            "[{label}] constraint {idx}: tree degree {measured} EXCEEDS max_degree() {max_degree}"
        );
    }
    let no_ch: Vec<Fp3> = vec![];
    let offset_e = Fp3::zero();

    let mut rng = SplitMix64(0x5EED_0000_0000_0000 ^ label.len() as u64);
    for trial in 0..TRIALS {
        let row: Vec<FE> = (0..num_cols).map(|_| FE::from(rng.next_u64())).collect();
        let row_e: Vec<Fp3> = row.iter().map(|x| x.to_extension()).collect();

        // --- ProverEvalFolder (base) ---
        let frame = Frame::<Gl, Gl3>::new(vec![TableView::new(vec![row.clone()], vec![vec![]])]);
        let ctx = TransitionEvaluationContext::new_prover(
            frame.as_row_frame(),
            &no_ch,
            &no_ch,
            &offset_e,
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
            &frame_e, &no_ch, &no_ch, &offset_e,
        );
        let mut vext_out = vec![Fp3::zero(); n];
        let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
        set.eval(&mut vfolder);
        vfolder.assert_all_emitted();

        // Prover folder (promoted) == verifier folder: the same body over the
        // same row in base vs extension must agree.
        for (i, (b, v)) in base_out.iter().zip(vext_out.iter()).enumerate() {
            assert_eq!(
                &b.to_extension(),
                v,
                "[{label}] prover-vs-verifier folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- capture → flatten → interpret == ProverEvalFolder (base) ---
        for (i, want) in base_out.iter().enumerate() {
            assert_eq!(
                &eval_program_base(&prog, i, &row),
                want,
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
    use crate::tables::eq::{EqConstraints, cols};

    #[test]
    fn eq_constraint_set_folder_capture_agree() {
        check_table("eq", &EqConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// store.rs
// =============================================================================

mod store {
    use super::*;
    use crate::tables::store::{StoreConstraints, cols};

    #[test]
    fn store_constraint_set_folder_capture_agree() {
        check_table("store", &StoreConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// memw.rs
// =============================================================================

mod memw {
    use super::*;
    use crate::tables::memw::{MemwConstraints, cols};

    #[test]
    fn memw_constraint_set_folder_capture_agree() {
        check_table("memw", &MemwConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// memw_aligned.rs
// =============================================================================

mod memw_aligned {
    use super::*;
    use crate::tables::memw_aligned::{MemwAlignedConstraints, cols};

    #[test]
    fn memw_aligned_constraint_set_folder_capture_agree() {
        check_table("memw_aligned", &MemwAlignedConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// memw_register.rs
// =============================================================================

mod memw_register {
    use super::*;
    use crate::tables::memw_register::{MemwRegisterConstraints, cols};

    #[test]
    fn memw_register_constraint_set_folder_capture_agree() {
        check_table("memw_register", &MemwRegisterConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// branch.rs
// =============================================================================

mod branch {
    use super::*;
    use crate::tables::branch::{BranchConstraints, cols};

    #[test]
    fn branch_constraint_set_folder_capture_agree() {
        check_table("branch", &BranchConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// commit.rs
// =============================================================================

mod commit {
    use super::*;
    use crate::tables::commit::{CommitConstraints, cols};

    #[test]
    fn commit_constraint_set_folder_capture_agree() {
        check_table("commit", &CommitConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// dma.rs
// =============================================================================

mod memmove {
    use super::*;
    use crate::tables::memmove::{MemmoveConstraints, cols};

    #[test]
    fn memmove_constraint_set_folder_capture_agree() {
        check_table("memmove", &MemmoveConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// keccak.rs
// =============================================================================

mod keccak {
    use super::*;
    use crate::tables::keccak::{KeccakConstraints, cols};

    #[test]
    fn keccak_constraint_set_folder_capture_agree() {
        check_table("keccak", &KeccakConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// keccak_rnd.rs
// =============================================================================

mod keccak_rnd {
    use super::*;
    use crate::tables::keccak_rnd::{KeccakRndConstraints, cols};

    #[test]
    fn keccak_rnd_constraint_set_folder_capture_agree() {
        check_table("keccak_rnd", &KeccakRndConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// cpu32.rs
// =============================================================================

mod cpu32 {
    use super::*;
    use crate::tables::cpu32::{Cpu32Constraints, cols};

    #[test]
    fn cpu32_constraint_set_folder_capture_agree() {
        check_table("cpu32", &Cpu32Constraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// cpu.rs (CpuConstraints lives in constraints/cpu.rs, not a
// prover/src/tables/*.rs conversion)
// =============================================================================

mod cpu {
    use super::*;
    use crate::constraints::cpu::{CpuConstraints, NUM_CPU_CONSTRAINTS};
    use crate::tables::cpu::cols;

    #[test]
    fn cpu_constraint_set_folder_capture_agree() {
        assert_eq!(CpuConstraints.meta().len(), NUM_CPU_CONSTRAINTS);
        check_table("cpu", &CpuConstraints, cols::NUM_COLUMNS);
    }
}

// =============================================================================
// hint.rs
// =============================================================================

mod hint {
    use super::*;
    use crate::tables::hint::{HintConstraints, cols};

    #[test]
    fn hint_constraint_set_folder_capture_agree() {
        // The one constraint is IS_BIT(mu): a single dense, idx-0, base-field root.
        assert_eq!(HintConstraints.meta().len(), 1);
        check_table("hint", &HintConstraints, cols::NUM_COLUMNS);
    }
}
