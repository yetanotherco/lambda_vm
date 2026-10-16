//! Folder-vs-capture-interpret regression tests for the single-body `emit_*`
//! constraint functions in `constraints::{templates, cpu}`.
//!
//! Each `emit_*` body is run three ways — the `ProverEvalFolder` (base), the
//! `VerifierEvalFolder` (extension), and the `CaptureBuilder` → flat IR →
//! `eval_program_base` interpreter — and asserted to agree on [`TRIALS`] random
//! off-trace rows. All three derive from the ONE body, so this pins that
//! capture/interpretation stays faithful to the compiled folder. Per constraint
//! we also assert the meta invariants (dense, idx-ordered, all-base) and that
//! the tree-measured degree equals the declared `meta.degree`.

use math::field::element::FieldElement;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintBuilder, MetaBuilder, ProverEvalFolder, RootKind, VerifierEvalFolder,
    num_base_from_meta,
};
use stark::frame::Frame;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::constraints::cpu::{
    emit_arg2, emit_arg2_exclusive, emit_branch_cond, emit_branch_rvd_pair, emit_mem_flags_bit,
    emit_next_pc_add_pair, emit_product_zero, emit_reg_not_read_is_zero, emit_rvd_eq_res,
};
use crate::constraints::templates::{AddLinearTerm, AddOperand, emit_add_pair, emit_is_bit};
use crate::tables::cpu::cols;
use crate::tables::types::{FE, GoldilocksExtension, GoldilocksField};

type Gl = GoldilocksField;
type Gl3 = GoldilocksExtension;
type Fp3 = FieldElement<Gl3>;

const TRIALS: usize = 1000;
const NUM_COLS: usize = cols::NUM_COLUMNS;

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

/// An emit body under test: emits `n` base constraints at indices `0..n`.
trait EmitBody {
    fn n(&self) -> usize;
    fn eval<B: ConstraintBuilder<Gl, Gl3>>(&self, b: &mut B);
}

macro_rules! emit_body {
    ($name:ident, $n:expr, |$b:ident| $body:block) => {
        struct $name;
        impl EmitBody for $name {
            fn n(&self) -> usize {
                $n
            }
            fn eval<B: ConstraintBuilder<Gl, Gl3>>(&self, $b: &mut B) {
                $body
            }
        }
    };
}

/// Folder-vs-capture-interpret check for one emit body. The body is run three
/// ways (prover folder, verifier folder, captured-IR interpreter) and asserted
/// to agree on random off-trace rows — all derive from the ONE emit body, so
/// this pins that capture/interpretation stays faithful to the compiled folder.
/// `meta` must be dense, idx-ordered, all-base, with each declared degree equal
/// to the tree-measured degree.
fn check_emit<T: EmitBody>(label: &str, body: &T, max_degree: usize) {
    let n = body.n();

    // --- meta invariants (DERIVED from the body): dense, idx-ordered, all-base ---
    let meta = {
        let mut mb = MetaBuilder::new();
        body.eval(&mut mb);
        mb.into_meta()
    };
    assert_eq!(meta.len(), n, "[{label}] meta length");
    assert_eq!(num_base_from_meta(&meta), n, "[{label}] all-base num_base");
    for (i, m) in meta.iter().enumerate() {
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Base, "[{label}] meta kind {i}");
    }

    // --- capture once; tree-measured degree matches the declared max ---
    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    body.eval(&mut cb);
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
    let mut max_measured = 0;
    for &(_, measured) in &degrees {
        assert!(
            measured <= max_degree,
            "[{label}] tree degree {measured} EXCEEDS declared max {max_degree}"
        );
        max_measured = max_measured.max(measured);
    }
    assert_eq!(
        max_measured, max_degree,
        "[{label}] max tree-measured degree {max_measured} != declared {max_degree}"
    );
    let no_ch: Vec<Fp3> = vec![];
    let offset_e = Fp3::zero();

    let mut rng = SplitMix64(0x5EED_0000_0000_0000 ^ label.len() as u64);
    for trial in 0..TRIALS {
        let row: Vec<FE> = (0..NUM_COLS).map(|_| FE::from(rng.next_u64())).collect();
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
        body.eval(&mut folder);
        folder.assert_all_emitted();

        // --- VerifierEvalFolder (ext) ---
        let frame_e =
            Frame::<Gl3, Gl3>::new(vec![TableView::new(vec![row_e.clone()], vec![vec![]])]);
        let vctx = TransitionEvaluationContext::<Gl, Gl3>::new_verifier(
            &frame_e, &no_ch, &no_ch, &offset_e,
        );
        let mut vext_out = vec![Fp3::zero(); n];
        let mut vfolder = VerifierEvalFolder::new(&vctx, &mut vext_out);
        body.eval(&mut vfolder);
        vfolder.assert_all_emitted();

        // Prover folder (promoted) == verifier folder == interpreter.
        for i in 0..n {
            assert_eq!(
                base_out[i].to_extension(),
                vext_out[i],
                "[{label}] prover-vs-verifier folder mismatch, constraint {i}, trial {trial}"
            );
            assert_eq!(
                eval_program_base(&prog, i, &row),
                base_out[i],
                "[{label}] interpreter mismatch, constraint {i}, trial {trial}"
            );
        }
    }
}

// =============================================================================
// templates.rs: IS_BIT
// =============================================================================

#[test]
fn emit_is_bit_folder_capture_agree() {
    emit_body!(Uncond, 1, |b| { emit_is_bit(b, 0, 7, None) });
    check_emit("is_bit_unconditional", &Uncond, 2);

    emit_body!(Cond, 1, |b| { emit_is_bit(b, 0, 5, Some(3)) });
    check_emit("is_bit_conditional", &Cond, 3);
}

// =============================================================================
// templates.rs: ADD pair
// =============================================================================

/// Run the pair check for one `conditional` flag.
fn check_add_pair_case<T: EmitBody>(label: &str, body: &T, conditional: bool) {
    let max_degree = if conditional { 3 } else { 2 };
    check_emit(label, body, max_degree);
}

#[test]
fn emit_add_pair_conditional_dword() {
    emit_body!(Body, 2, |b| {
        emit_add_pair(
            b,
            0,
            &[0],
            &AddOperand::dword(1),
            &AddOperand::dword(3),
            &AddOperand::dword(5),
        )
    });
    check_add_pair_case("add_pair_conditional_dword", &Body, true);
}

#[test]
fn emit_add_pair_linear_unconditional() {
    // DWordHL repack lhs; negative-coefficient + constant linear rhs —
    // exercises const_signed on both signs.
    fn rhs() -> AddOperand {
        AddOperand::linear(
            &[
                AddLinearTerm::Column {
                    coefficient: -2,
                    column: 2,
                },
                AddLinearTerm::Constant(4),
            ],
            &[],
        )
    }
    emit_body!(Body, 2, |b| {
        emit_add_pair(
            b,
            0,
            &[],
            &AddOperand::from_dword_hl(8),
            &rhs(),
            &AddOperand::dword(5),
        )
    });
    check_add_pair_case("add_pair_linear_unconditional", &Body, false);
}

#[test]
fn emit_add_pair_multi_cond_bytes() {
    // Multi-column condition (flag sum), Word + Constant operands, and a
    // DWordBL byte-repacked sum — the remaining AddOperand variants.
    emit_body!(Body, 2, |b| {
        emit_add_pair(
            b,
            0,
            &[0, 2],
            &AddOperand::from_word(4),
            &AddOperand::constant(300),
            &AddOperand::from_dword_bl(20),
        )
    });
    check_add_pair_case("add_pair_multi_cond_bytes", &Body, true);
}

// =============================================================================
// cpu.rs: decode / assumption constraints
// =============================================================================

#[test]
fn emit_product_zero_folder_capture_agree() {
    emit_body!(Body, 1, |b| { emit_product_zero(b, 0, 12, 17) });
    check_emit("product_zero", &Body, 2);
}

#[test]
fn emit_arg2_exclusive_folder_capture_agree() {
    emit_body!(Body0, 1, |b| { emit_arg2_exclusive(b, 0, cols::IMM_0) });
    check_emit("arg2_exclusive_imm0", &Body0, 3);

    emit_body!(Body1, 1, |b| { emit_arg2_exclusive(b, 0, cols::IMM_1) });
    check_emit("arg2_exclusive_imm1", &Body1, 3);
}

#[test]
fn emit_mem_flags_bit_folder_capture_agree() {
    emit_body!(Body, 1, |b| { emit_mem_flags_bit(b, 0) });
    check_emit("mem_flags_bit", &Body, 3);
}

#[test]
fn emit_reg_not_read_is_zero_folder_capture_agree() {
    emit_body!(Body, 1, |b| {
        emit_reg_not_read_is_zero(b, 0, cols::READ_REGISTER1, cols::RV1_0)
    });
    check_emit("reg_not_read_is_zero_rv1", &Body, 2);

    emit_body!(Body2, 1, |b| {
        emit_reg_not_read_is_zero(b, 0, cols::READ_REGISTER2, cols::RV2_1)
    });
    check_emit("reg_not_read_is_zero_rv2", &Body2, 2);
}

// =============================================================================
// cpu.rs: alu / mem / branch groups
// =============================================================================

#[test]
fn emit_arg2_folder_capture_agree() {
    emit_body!(Body0, 1, |b| { emit_arg2(b, 0, 0) });
    check_emit("arg2_word0", &Body0, 2);
    emit_body!(Body1, 1, |b| { emit_arg2(b, 0, 1) });
    check_emit("arg2_word1", &Body1, 2);
}

#[test]
fn emit_rvd_eq_res_folder_capture_agree() {
    emit_body!(Body0, 1, |b| { emit_rvd_eq_res(b, 0, 0) });
    check_emit("rvd_eq_res_word0", &Body0, 2);
    emit_body!(Body1, 1, |b| { emit_rvd_eq_res(b, 0, 1) });
    check_emit("rvd_eq_res_word1", &Body1, 2);
}

#[test]
fn emit_branch_rvd_pair_folder_capture_agree() {
    emit_body!(Body, 2, |b| { emit_branch_rvd_pair(b, 0) });
    check_emit("branch_rvd_pair", &Body, 3);
}

#[test]
fn emit_branch_cond_folder_capture_agree() {
    emit_body!(Body, 1, |b| { emit_branch_cond(b, 0) });
    check_emit("branch_cond", &Body, 3);
}

#[test]
fn emit_next_pc_add_pair_folder_capture_agree() {
    emit_body!(Body, 2, |b| { emit_next_pc_add_pair(b, 0) });
    check_emit("next_pc_add_pair", &Body, 3);
}
