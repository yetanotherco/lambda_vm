//! Differential tests for the single-body `emit_*` constraint functions.
//!
//! Each `emit_*` function in `constraints::{templates, cpu}` is checked
//! against the OLD boxed constraint struct it transcribes (the structs stay
//! in-branch as this oracle until the final deletion phase), on
//! [`TRIALS`] random rows — off-trace points, where a weakened or slipped
//! transcription diverges with overwhelming probability:
//!
//! 1. `ProverEvalFolder` output == old `evaluate::<Gl, Gl3>`;
//! 2. `VerifierEvalFolder` output == old `evaluate::<Gl3, Gl3>` (embedded);
//! 3. `CaptureBuilder` → flatten → interpret == old `evaluate::<Gl, Gl3>`;
//!
//! plus per constraint: tree-measured degree == declared `meta.degree` ==
//! old `degree()`, and `*_meta` zerofier parameters == the old struct's
//! `period`/`offset`/`exemptions_period`/`periodic_exemptions_offset`/
//! `end_exemptions`.

use math::field::element::FieldElement;
use stark::constraint_ir::eval_program_base;
use stark::constraints::builder::{
    CaptureBuilder, ConstraintBuilder, ConstraintMeta, ProverEvalFolder, RootKind,
    VerifierEvalFolder, num_base_from_meta,
};
use stark::constraints::transition::TransitionConstraint;
use stark::frame::Frame;
use stark::lookup::PackingShifts;
use stark::table::TableView;
use stark::traits::TransitionEvaluationContext;

use crate::constraints::cpu::{
    Arg2Constraint, Arg2ExclusiveConstraint, BranchCondConstraint, BranchRvdConstraint,
    MemFlagsBitConstraint, NextPcAddConstraint, ProductZeroConstraint, RegNotReadIsZeroConstraint,
    RvdEqResConstraint, emit_arg2, emit_arg2_exclusive, emit_branch_cond, emit_branch_rvd_pair,
    emit_mem_flags_bit, emit_next_pc_add_pair, emit_product_zero, emit_reg_not_read_is_zero,
    emit_rvd_eq_res,
};
use crate::constraints::cpu::{
    arg2_exclusive_meta, arg2_meta, branch_cond_meta, branch_rvd_meta, mem_flags_bit_meta,
    next_pc_add_meta, product_zero_meta, reg_not_read_is_zero_meta, rvd_eq_res_meta,
};
use crate::constraints::templates::{
    AddConstraint, AddLinearTerm, AddOperand, IsBitConstraint, add_pair_meta, emit_add_pair,
    emit_is_bit, is_bit_meta,
};
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

/// Old-struct evaluator at the prover instantiation `<Gl, Gl3>`.
type OldEval<'a> = &'a dyn Fn(&TableView<Gl, Gl3>) -> FE;
/// Old-struct evaluator at the verifier instantiation `<Gl3, Gl3>`.
type OldEvalExt<'a> = &'a dyn Fn(&TableView<Gl3, Gl3>) -> Fp3;

/// Zerofier/degree parameters read off an old constraint struct.
struct OldParams {
    degree: usize,
    period: usize,
    offset: usize,
    exemptions_period: Option<usize>,
    periodic_exemptions_offset: Option<usize>,
    end_exemptions: usize,
}

fn old_params<T: TransitionConstraint<Gl, Gl3>>(c: &T) -> OldParams {
    OldParams {
        degree: c.degree(),
        period: c.period(),
        offset: c.offset(),
        exemptions_period: c.exemptions_period(),
        periodic_exemptions_offset: c.periodic_exemptions_offset(),
        end_exemptions: c.end_exemptions(),
    }
}

/// The full differential check for one emit body against its old structs.
fn check_emit_vs_old<T: EmitBody>(
    label: &str,
    body: &T,
    meta: &[ConstraintMeta],
    olds: &[OldEval<'_>],
    olds_ext: &[OldEvalExt<'_>],
    old_params: &[OldParams],
) {
    let n = body.n();
    assert_eq!(meta.len(), n, "[{label}] meta length");
    assert_eq!(olds.len(), n);
    assert_eq!(olds_ext.len(), n);
    assert_eq!(old_params.len(), n);

    // --- meta parity vs the old structs ---
    assert_eq!(num_base_from_meta(meta), n, "[{label}] all-base num_base");
    for (i, (m, p)) in meta.iter().zip(old_params.iter()).enumerate() {
        assert_eq!(m.constraint_idx, i, "[{label}] meta idx {i}");
        assert_eq!(m.kind, RootKind::Base, "[{label}] meta kind {i}");
        assert_eq!(m.degree, p.degree, "[{label}] degree {i}");
        assert_eq!(m.period, p.period, "[{label}] period {i}");
        assert_eq!(m.offset, p.offset, "[{label}] offset {i}");
        assert_eq!(
            m.exemptions_period, p.exemptions_period,
            "[{label}] exemptions_period {i}"
        );
        assert_eq!(
            m.periodic_exemptions_offset, p.periodic_exemptions_offset,
            "[{label}] periodic_exemptions_offset {i}"
        );
        assert_eq!(
            m.end_exemptions, p.end_exemptions,
            "[{label}] end_exemptions {i}"
        );
    }

    // --- capture once; tree-measured degree == declared == old ---
    let mut cb = CaptureBuilder::<Gl, Gl3>::new();
    body.eval(&mut cb);
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
        let row: Vec<FE> = (0..NUM_COLS).map(|_| FE::from(rng.next_u64())).collect();
        let row_e: Vec<Fp3> = row.iter().map(|x| x.to_extension()).collect();

        let step: TableView<Gl, Gl3> = TableView::new(vec![row.clone()], vec![Vec::new()]);
        let step_e: TableView<Gl3, Gl3> = TableView::new(vec![row_e.clone()], vec![Vec::new()]);

        // --- 1. ProverEvalFolder == old evaluate::<Gl, Gl3> ---
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
        body.eval(&mut folder);
        folder.assert_all_emitted();
        for (i, old) in olds.iter().enumerate() {
            assert_eq!(
                base_out[i],
                old(&step),
                "[{label}] prover folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- 2. VerifierEvalFolder == old evaluate::<Gl3, Gl3> ---
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
        body.eval(&mut vfolder);
        vfolder.assert_all_emitted();
        for (i, old) in olds_ext.iter().enumerate() {
            assert_eq!(
                vext_out[i],
                old(&step_e),
                "[{label}] verifier folder mismatch, constraint {i}, trial {trial}"
            );
        }

        // --- 3. capture → flatten → interpret == old evaluate ---
        for (i, old) in olds.iter().enumerate() {
            assert_eq!(
                eval_program_base(&prog, i, &row),
                old(&step),
                "[{label}] interpreter mismatch, constraint {i}, trial {trial}"
            );
        }
    }
}

// =============================================================================
// templates.rs: IS_BIT
// =============================================================================

#[test]
fn emit_is_bit_matches_old() {
    emit_body!(Uncond, 1, |b| { emit_is_bit(b, 0, 7, None) });
    let old = IsBitConstraint::unconditional(7, 0);
    check_emit_vs_old(
        "is_bit_unconditional",
        &Uncond,
        &[is_bit_meta(0, false)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );

    emit_body!(Cond, 1, |b| { emit_is_bit(b, 0, 5, Some(3)) });
    let old = IsBitConstraint::new(3, 5, 0);
    check_emit_vs_old(
        "is_bit_conditional",
        &Cond,
        &[is_bit_meta(0, true)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );
}

// =============================================================================
// templates.rs: ADD pair
// =============================================================================

/// Run the pair check for one (cond, lhs, rhs, sum) configuration.
fn check_add_pair_case<T: EmitBody>(
    label: &str,
    body: &T,
    cond_cols: Vec<usize>,
    lhs: AddOperand,
    rhs: AddOperand,
    sum: AddOperand,
) {
    let conditional = !cond_cols.is_empty();
    let (old0, old1) = AddConstraint::new_pair(cond_cols, lhs, rhs, sum, 0);
    check_emit_vs_old(
        label,
        body,
        &add_pair_meta(0, conditional),
        &[&|s| old0.evaluate::<Gl, Gl3>(s), &|s| {
            old1.evaluate::<Gl, Gl3>(s)
        }],
        &[&|s| old0.evaluate::<Gl3, Gl3>(s), &|s| {
            old1.evaluate::<Gl3, Gl3>(s)
        }],
        &[old_params(&old0), old_params(&old1)],
    );
}

#[test]
fn emit_add_pair_matches_old_conditional_dword() {
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
    check_add_pair_case(
        "add_pair_conditional_dword",
        &Body,
        vec![0],
        AddOperand::dword(1),
        AddOperand::dword(3),
        AddOperand::dword(5),
    );
}

#[test]
fn emit_add_pair_matches_old_linear_unconditional() {
    // DWordHL repack lhs; negative-coefficient + constant linear rhs —
    // exercises const_signed on both signs.
    fn rhs() -> AddOperand {
        AddOperand::linear(
            vec![
                AddLinearTerm::Column {
                    coefficient: -2,
                    column: 2,
                },
                AddLinearTerm::Constant(4),
            ],
            vec![],
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
    check_add_pair_case(
        "add_pair_linear_unconditional",
        &Body,
        vec![],
        AddOperand::from_dword_hl(8),
        rhs(),
        AddOperand::dword(5),
    );
}

#[test]
fn emit_add_pair_matches_old_multi_cond_bytes() {
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
    check_add_pair_case(
        "add_pair_multi_cond_bytes",
        &Body,
        vec![0, 2],
        AddOperand::from_word(4),
        AddOperand::constant(300),
        AddOperand::from_dword_bl(20),
    );
}

// =============================================================================
// cpu.rs: decode / assumption constraints
// =============================================================================

#[test]
fn emit_product_zero_matches_old() {
    emit_body!(Body, 1, |b| { emit_product_zero(b, 0, 12, 17) });
    let old = ProductZeroConstraint::new(12, 17, 0);
    check_emit_vs_old(
        "product_zero",
        &Body,
        &[product_zero_meta(0)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );
}

#[test]
fn emit_arg2_exclusive_matches_old() {
    for imm_col in [cols::IMM_0, cols::IMM_1] {
        let old = Arg2ExclusiveConstraint::new(imm_col, 0);
        // Body reads `imm_col` from the environment via two fixed cases.
        if imm_col == cols::IMM_0 {
            emit_body!(Body0, 1, |b| { emit_arg2_exclusive(b, 0, cols::IMM_0) });
            check_emit_vs_old(
                "arg2_exclusive_imm0",
                &Body0,
                &[arg2_exclusive_meta(0)],
                &[&|s| old.evaluate::<Gl, Gl3>(s)],
                &[&|s| old.evaluate::<Gl3, Gl3>(s)],
                &[old_params(&old)],
            );
        } else {
            emit_body!(Body1, 1, |b| { emit_arg2_exclusive(b, 0, cols::IMM_1) });
            check_emit_vs_old(
                "arg2_exclusive_imm1",
                &Body1,
                &[arg2_exclusive_meta(0)],
                &[&|s| old.evaluate::<Gl, Gl3>(s)],
                &[&|s| old.evaluate::<Gl3, Gl3>(s)],
                &[old_params(&old)],
            );
        }
    }
}

#[test]
fn emit_mem_flags_bit_matches_old() {
    emit_body!(Body, 1, |b| { emit_mem_flags_bit(b, 0) });
    let old = MemFlagsBitConstraint::new(0);
    check_emit_vs_old(
        "mem_flags_bit",
        &Body,
        &[mem_flags_bit_meta(0)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );
}

#[test]
fn emit_reg_not_read_is_zero_matches_old() {
    emit_body!(Body, 1, |b| {
        emit_reg_not_read_is_zero(b, 0, cols::READ_REGISTER1, cols::RV1_0)
    });
    let old = RegNotReadIsZeroConstraint::new(cols::READ_REGISTER1, cols::RV1_0, 0);
    check_emit_vs_old(
        "reg_not_read_is_zero_rv1",
        &Body,
        &[reg_not_read_is_zero_meta(0)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );

    emit_body!(Body2, 1, |b| {
        emit_reg_not_read_is_zero(b, 0, cols::READ_REGISTER2, cols::RV2_1)
    });
    let old = RegNotReadIsZeroConstraint::new(cols::READ_REGISTER2, cols::RV2_1, 0);
    check_emit_vs_old(
        "reg_not_read_is_zero_rv2",
        &Body2,
        &[reg_not_read_is_zero_meta(0)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );
}

// =============================================================================
// cpu.rs: alu / mem / branch groups
// =============================================================================

#[test]
fn emit_arg2_matches_old() {
    emit_body!(Body0, 1, |b| { emit_arg2(b, 0, 0) });
    emit_body!(Body1, 1, |b| { emit_arg2(b, 0, 1) });
    let old0 = Arg2Constraint::new(0, 0);
    check_emit_vs_old(
        "arg2_word0",
        &Body0,
        &[arg2_meta(0)],
        &[&|s| old0.evaluate::<Gl, Gl3>(s)],
        &[&|s| old0.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old0)],
    );
    let old1 = Arg2Constraint::new(1, 0);
    check_emit_vs_old(
        "arg2_word1",
        &Body1,
        &[arg2_meta(0)],
        &[&|s| old1.evaluate::<Gl, Gl3>(s)],
        &[&|s| old1.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old1)],
    );
}

#[test]
fn emit_rvd_eq_res_matches_old() {
    emit_body!(Body0, 1, |b| { emit_rvd_eq_res(b, 0, 0) });
    emit_body!(Body1, 1, |b| { emit_rvd_eq_res(b, 0, 1) });
    let old0 = RvdEqResConstraint::new(0, 0);
    check_emit_vs_old(
        "rvd_eq_res_word0",
        &Body0,
        &[rvd_eq_res_meta(0)],
        &[&|s| old0.evaluate::<Gl, Gl3>(s)],
        &[&|s| old0.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old0)],
    );
    let old1 = RvdEqResConstraint::new(1, 0);
    check_emit_vs_old(
        "rvd_eq_res_word1",
        &Body1,
        &[rvd_eq_res_meta(0)],
        &[&|s| old1.evaluate::<Gl, Gl3>(s)],
        &[&|s| old1.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old1)],
    );
}

#[test]
fn emit_branch_rvd_pair_matches_old() {
    emit_body!(Body, 2, |b| { emit_branch_rvd_pair(b, 0) });
    let (old0, old1) = BranchRvdConstraint::new_pair(0);
    check_emit_vs_old(
        "branch_rvd_pair",
        &Body,
        &branch_rvd_meta(0),
        &[&|s| old0.evaluate::<Gl, Gl3>(s), &|s| {
            old1.evaluate::<Gl, Gl3>(s)
        }],
        &[&|s| old0.evaluate::<Gl3, Gl3>(s), &|s| {
            old1.evaluate::<Gl3, Gl3>(s)
        }],
        &[old_params(&old0), old_params(&old1)],
    );
}

#[test]
fn emit_branch_cond_matches_old() {
    emit_body!(Body, 1, |b| { emit_branch_cond(b, 0) });
    let old = BranchCondConstraint::new(0);
    check_emit_vs_old(
        "branch_cond",
        &Body,
        &[branch_cond_meta(0)],
        &[&|s| old.evaluate::<Gl, Gl3>(s)],
        &[&|s| old.evaluate::<Gl3, Gl3>(s)],
        &[old_params(&old)],
    );
}

#[test]
fn emit_next_pc_add_pair_matches_old() {
    emit_body!(Body, 2, |b| { emit_next_pc_add_pair(b, 0) });
    let (old0, old1) = NextPcAddConstraint::new_pair(0);
    check_emit_vs_old(
        "next_pc_add_pair",
        &Body,
        &next_pc_add_meta(0),
        &[&|s| old0.evaluate::<Gl, Gl3>(s), &|s| {
            old1.evaluate::<Gl, Gl3>(s)
        }],
        &[&|s| old0.evaluate::<Gl3, Gl3>(s), &|s| {
            old1.evaluate::<Gl3, Gl3>(s)
        }],
        &[old_params(&old0), old_params(&old1)],
    );
}
