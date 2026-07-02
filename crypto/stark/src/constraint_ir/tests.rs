//! Unit tests for the field-generic constraint IR: hand-built programs checked
//! against direct `FieldElement` arithmetic, the prover/verifier entry points
//! against hand-constructed contexts, and a non-Goldilocks tower (`E = F`) that
//! exercises the reflexive `IsSubFieldOf` path — the point of the genericity.

use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField as Ext;
use math::field::goldilocks::GoldilocksField as Fp;
use math::field::test_fields::u32_test_field::U32TestField;

use super::builder::IrBuilder;
use super::interp::{eval_program, eval_program_base, eval_program_verifier};
use super::ir::{ConstraintProgram, Dim, Op};
use crate::frame::Frame;
use crate::lookup::PackingShifts;
use crate::table::TableView;
use crate::traits::TransitionEvaluationContext;

type FpE = FieldElement<Fp>;
type ExtE = FieldElement<Ext>;

fn fp(v: u64) -> FpE {
    FpE::from(v)
}

/// Build a degree-3 Goldilocks extension element from three `u64` components.
fn ext3(a: u64, b: u64, c: u64) -> ExtE {
    ExtE::from_raw([fp(a), fp(b), fp(c)])
}

// ------------------------------------------------------------------------
// id-0 convention + const dedup
// ------------------------------------------------------------------------

#[test]
fn id_zero_is_base_const_zero() {
    let b = IrBuilder::<Fp, Ext>::new();
    let prog = b.finish(0);
    // Node 0 is ConstBase(0); base_consts[0] is the base-field zero.
    assert_eq!(prog.nodes[0], Op::ConstBase(0));
    assert_eq!(prog.dims[0], Dim::Base);
    assert_eq!(prog.base_consts[0], FpE::zero());
    assert_eq!(prog.len(), 1);
    assert!(!prog.is_empty());
}

#[test]
fn const_base_zero_dedups_to_id_zero() {
    let mut b = IrBuilder::<Fp, Ext>::new();
    let z = b.const_base(0);
    assert_eq!(z.dim(), Dim::Base);
    let prog = b.finish(0);
    // No new node or const slot: reuses the reserved id-0 zero.
    assert_eq!(prog.nodes.len(), 1);
    assert_eq!(prog.base_consts.len(), 1);
}

#[test]
fn const_dedup_same_value_interned_once() {
    let mut b = IrBuilder::<Fp, Ext>::new();
    b.const_base(7);
    b.const_base(7);
    let prog = b.finish(0);
    // base_consts: [0, 7] only; nodes: ConstBase(0), ConstBase(1) only.
    assert_eq!(prog.base_consts, vec![fp(0), fp(7)]);
    assert_eq!(prog.nodes.len(), 2);
}

#[test]
fn const_signed_negative_reduces_and_dedups() {
    let mut b = IrBuilder::<Fp, Ext>::new();
    let neg = b.const_signed(-1);
    assert_eq!(neg.dim(), Dim::Base);
    let prog = b.finish(0);
    // -1 in the field is p - 1; matches FieldElement::from(-1i64).
    assert_eq!(prog.base_consts[1], FpE::from(-1i64));

    // Interning the same negative twice uses one slot and one node.
    let mut b2 = IrBuilder::<Fp, Ext>::new();
    b2.const_signed(-5);
    b2.const_signed(-5);
    let prog2 = b2.finish(0);
    assert_eq!(prog2.base_consts, vec![fp(0), FpE::from(-5i64)]);
    assert_eq!(prog2.nodes.len(), 2);

    // A positive i64 dedups against the same value interned via const_base.
    let mut b3 = IrBuilder::<Fp, Ext>::new();
    b3.const_base(9);
    b3.const_signed(9);
    let prog3 = b3.finish(0);
    assert_eq!(prog3.base_consts, vec![fp(0), fp(9)]);
    assert_eq!(prog3.nodes.len(), 2);
}

#[test]
fn const_ext_dedups_by_value() {
    let mut b = IrBuilder::<Fp, Ext>::new();
    let e1 = b.const_ext(ext3(1, 2, 3));
    b.const_ext(ext3(1, 2, 3));
    b.const_ext(ext3(4, 5, 6));
    assert_eq!(e1.dim(), Dim::Ext);
    let prog = b.finish(0);
    // ext_consts: two distinct values.
    assert_eq!(prog.ext_consts, vec![ext3(1, 2, 3), ext3(4, 5, 6)]);
    // nodes: ConstBase(0) [id-0] + ConstExt(0) + ConstExt(1).
    assert_eq!(prog.nodes.len(), 3);
    assert_eq!(prog.nodes[1], Op::ConstExt(0));
    assert_eq!(prog.nodes[2], Op::ConstExt(1));
}

// ------------------------------------------------------------------------
// CSE on (Op, Dim) still works with side-table constants.
// ------------------------------------------------------------------------

#[test]
fn cse_shares_structurally_identical_subexpressions() {
    let mut b = IrBuilder::<Fp, Ext>::new();
    let x = b.main(0, 0);
    let y = b.main(0, 1);
    let s1 = b.add(x, y);
    let s2 = b.add(x, y); // structurally identical: no new node
    let nodes_so_far = 4; // zero, x, y, add
    let m = b.mul(s1, s2); // Mul(add, add): one new node
    b.emit(0, m);
    let prog = b.finish(1);
    assert_eq!(prog.nodes.len(), nodes_so_far + 1);

    let row = vec![fp(3), fp(4)];
    let got = eval_program_base(&prog, 0, &row);
    let s = fp(3) + fp(4);
    assert_eq!(got, s * s);
}

// ------------------------------------------------------------------------
// Every arithmetic Op over base-field leaves, checked against direct math.
// ------------------------------------------------------------------------

#[test]
fn base_arithmetic_add_sub_mul_neg() {
    // Roots: idx 0 = (x + y) - (x * y); idx 1 = its negation.
    let mut b = IrBuilder::<Fp, Ext>::new();
    let x = b.main(0, 0);
    let y = b.main(0, 1);
    let sum = b.add(x, y);
    let prod = b.mul(x, y);
    let diff = b.sub(sum, prod);
    let negd = b.neg(diff);
    assert_eq!(sum.dim(), Dim::Base);
    assert_eq!(prod.dim(), Dim::Base);
    assert_eq!(diff.dim(), Dim::Base);
    assert_eq!(negd.dim(), Dim::Base);
    b.emit(0, diff);
    b.emit(1, negd);
    let prog = b.finish(2);

    for (px, py) in [(3u64, 5u64), (0, 9), (100, 7), (1, 1)] {
        let row = vec![fp(px), fp(py)];
        let expected = (fp(px) + fp(py)) - (fp(px) * fp(py));
        assert_eq!(eval_program_base(&prog, 0, &row), expected);
        assert_eq!(eval_program_base(&prog, 1, &row), -expected);
    }
}

#[test]
fn base_const_arithmetic() {
    // 2 * x - 1
    let mut b = IrBuilder::<Fp, Ext>::new();
    let x = b.main(0, 0);
    let two = b.const_base(2);
    let one = b.one();
    let twox = b.mul(two, x);
    let res = b.sub(twox, one);
    b.emit(0, res);
    let prog = b.finish(1);

    for xv in [0u64, 1, 2, 42, 1_000_000] {
        let got = eval_program_base(&prog, 0, &[fp(xv)]);
        assert_eq!(got, fp(2) * fp(xv) - fp(1));
    }
}

// ------------------------------------------------------------------------
// Frame offsets: reading the next row (offset 1).
// ------------------------------------------------------------------------

#[test]
fn frame_offset_reads_next_step() {
    // next - cur over main column 0.
    let mut b = IrBuilder::<Fp, Ext>::new();
    let cur = b.main(0, 0);
    let next = b.main(1, 0);
    let res = b.sub(next, cur);
    b.emit(0, res);
    let prog = b.finish(1);

    let step0 = TableView::<Fp, Ext>::new(vec![vec![fp(10)]], vec![vec![]]);
    let step1 = TableView::<Fp, Ext>::new(vec![vec![fp(17)]], vec![vec![]]);
    let frame = Frame::<Fp, Ext>::new(vec![step0, step1]);
    let rap: Vec<ExtE> = vec![];
    let alpha: Vec<ExtE> = vec![];
    let offset = ExtE::zero();
    let shifts = PackingShifts::<Fp>::new();
    let ctx = TransitionEvaluationContext::new_prover(&frame, &rap, &alpha, &offset, &shifts);

    let mut base_evals = vec![FpE::zero()];
    let mut ext_evals: Vec<ExtE> = vec![];
    eval_program(&prog, &ctx, &mut base_evals, &mut ext_evals);
    assert_eq!(base_evals[0], fp(17) - fp(10));
}

// ------------------------------------------------------------------------
// Mixed Base×Ext arithmetic with auto-embed, and the explicit Embed op.
// ------------------------------------------------------------------------

#[test]
fn mixed_base_ext_auto_embeds() {
    // aux (Ext) + main (Base) and main * aux: result Ext, base auto-embedded.
    let mut b = IrBuilder::<Fp, Ext>::new();
    let m = b.main(0, 0); // Base
    let a = b.aux(0, 0); // Ext
    let sum = b.add(a, m);
    let prod = b.mul(m, a);
    assert_eq!(sum.dim(), Dim::Ext);
    assert_eq!(prod.dim(), Dim::Ext);
    b.emit(0, sum);
    b.emit(1, prod);
    let prog = b.finish(0); // both roots are Ext

    let main_val = fp(5);
    let aux_val = ext3(2, 3, 4);
    let step = TableView::<Fp, Ext>::new(vec![vec![main_val]], vec![vec![aux_val]]);
    let frame = Frame::<Fp, Ext>::new(vec![step]);
    let rap: Vec<ExtE> = vec![];
    let alpha: Vec<ExtE> = vec![];
    let offset = ExtE::zero();
    let shifts = PackingShifts::<Fp>::new();
    let ctx = TransitionEvaluationContext::new_prover(&frame, &rap, &alpha, &offset, &shifts);

    let mut base_evals: Vec<FpE> = vec![];
    let mut ext_evals = vec![ExtE::zero(), ExtE::zero()];
    eval_program(&prog, &ctx, &mut base_evals, &mut ext_evals);
    // Mixed operators put the subfield on the left: F op E -> E.
    assert_eq!(ext_evals[0], main_val + aux_val);
    assert_eq!(ext_evals[1], main_val * aux_val);
}

#[test]
fn explicit_embed_and_ext_neg() {
    // Embed(main) and Neg over an Ext value: embed(m) + (-aux).
    let mut b = IrBuilder::<Fp, Ext>::new();
    let m = b.main(0, 0);
    let e = b.embed(m);
    assert_eq!(m.dim(), Dim::Base);
    assert_eq!(e.dim(), Dim::Ext);
    let a = b.aux(0, 0);
    let na = b.neg(a);
    assert_eq!(na.dim(), Dim::Ext);
    let res = b.add(e, na);
    b.emit(0, res);
    let prog = b.finish(0);
    assert!(prog.nodes.iter().any(|op| matches!(op, Op::Embed(_))));

    let aux_val = ext3(1, 2, 3);
    let step = TableView::<Fp, Ext>::new(vec![vec![fp(9)]], vec![vec![aux_val]]);
    let frame = Frame::<Fp, Ext>::new(vec![step]);
    let rap: Vec<ExtE> = vec![];
    let alpha: Vec<ExtE> = vec![];
    let offset = ExtE::zero();
    let shifts = PackingShifts::<Fp>::new();
    let ctx = TransitionEvaluationContext::new_prover(&frame, &rap, &alpha, &offset, &shifts);

    let mut base_evals: Vec<FpE> = vec![];
    let mut ext_evals = vec![ExtE::zero()];
    eval_program(&prog, &ctx, &mut base_evals, &mut ext_evals);
    assert_eq!(ext_evals[0], fp(9).to_extension::<Ext>() - aux_val);
}

// ------------------------------------------------------------------------
// Every leaf kind: main, challenge, alpha_power, table_offset, aux.
// ------------------------------------------------------------------------

#[test]
fn all_leaf_kinds_logup_shaped() {
    // A LogUp-shaped expression touching every leaf variety:
    //   main(0,0) * challenge(0) + alpha_pow(1) * aux(0,3) - table_offset()
    let mut b = IrBuilder::<Fp, Ext>::new();
    let m = b.main(0, 0); // Base
    let ch = b.challenge(0); // Ext
    let ap = b.alpha_power(1); // Ext
    let au = b.aux(0, 3); // Ext
    let off = b.table_offset(); // Ext
    assert_eq!(m.dim(), Dim::Base);
    assert_eq!(ch.dim(), Dim::Ext);
    assert_eq!(ap.dim(), Dim::Ext);
    assert_eq!(au.dim(), Dim::Ext);
    assert_eq!(off.dim(), Dim::Ext);
    let t1 = b.mul(m, ch); // Base×Ext → Ext
    let t2 = b.mul(ap, au); // Ext×Ext → Ext
    let s = b.add(t1, t2);
    let res = b.sub(s, off);
    assert_eq!(res.dim(), Dim::Ext);
    b.emit(0, res);
    let prog = b.finish(0);

    let main_row = vec![fp(6)];
    let rap = vec![ext3(1, 0, 0), ext3(2, 2, 2)];
    let alpha = vec![ext3(9, 9, 9), ext3(3, 1, 4)];
    let offset = ext3(7, 7, 7);
    let aux_row = vec![ext3(0, 0, 0), ext3(0, 0, 0), ext3(0, 0, 0), ext3(5, 5, 5)];

    let expected = {
        let t1 = main_row[0] * rap[0]; // main(0,0) * challenge(0)
        let t2 = alpha[1] * aux_row[3];
        (t1 + t2) - offset
    };

    let step = TableView::<Fp, Ext>::new(vec![main_row], vec![aux_row]);
    let frame = Frame::<Fp, Ext>::new(vec![step]);
    let shifts = PackingShifts::<Fp>::new();
    let ctx = TransitionEvaluationContext::new_prover(&frame, &rap, &alpha, &offset, &shifts);

    let mut base_evals: Vec<FpE> = vec![];
    let mut ext_evals = vec![ExtE::zero()];
    eval_program(&prog, &ctx, &mut base_evals, &mut ext_evals);
    assert_eq!(ext_evals[0], expected);
}

// ------------------------------------------------------------------------
// Prover & verifier full entry points on hand-built contexts (both variants).
// ------------------------------------------------------------------------

/// One base constraint (idx 0: `a - b`) and one ext constraint
/// (idx 1: `aux0 * alpha0`); `num_base = 1`.
fn two_constraint_program() -> ConstraintProgram<Fp, Ext> {
    let mut b = IrBuilder::<Fp, Ext>::new();
    let a = b.main(0, 0);
    let bb = b.main(0, 1);
    let base_c = b.sub(a, bb);
    b.emit(0, base_c);
    let au = b.aux(0, 0);
    let al = b.alpha_power(0);
    let ext_c = b.mul(au, al);
    b.emit(1, ext_c);
    b.finish(1)
}

#[test]
fn prover_entry_point_splits_base_and_ext() {
    let prog = two_constraint_program();
    let aux_val = ext3(2, 0, 1);
    let step = TableView::<Fp, Ext>::new(vec![vec![fp(30), fp(12)]], vec![vec![aux_val]]);
    let frame = Frame::<Fp, Ext>::new(vec![step]);
    let rap: Vec<ExtE> = vec![];
    let alpha = vec![ext3(3, 3, 3)];
    let offset = ExtE::zero();
    let shifts = PackingShifts::<Fp>::new();
    let ctx = TransitionEvaluationContext::new_prover(&frame, &rap, &alpha, &offset, &shifts);

    let mut base_evals = vec![FpE::zero()];
    let mut ext_evals = vec![ExtE::zero(), ExtE::zero()];
    eval_program(&prog, &ctx, &mut base_evals, &mut ext_evals);

    // Base root lands in base_evals[0]; ext root in ext_evals[1] (absolute idx).
    assert_eq!(base_evals[0], fp(30) - fp(12));
    assert_eq!(ext_evals[1], aux_val * alpha[0]);
}

#[test]
fn verifier_entry_point_promotes_base_roots() {
    let prog = two_constraint_program();
    // Verifier frame holds extension elements only (Frame<E, E>).
    let aux_val = ext3(2, 0, 1);
    let step = TableView::<Ext, Ext>::new(
        vec![vec![ext3(30, 0, 0), ext3(12, 0, 0)]],
        vec![vec![aux_val]],
    );
    let frame = Frame::<Ext, Ext>::new(vec![step]);
    let rap: Vec<ExtE> = vec![];
    let alpha = vec![ext3(3, 3, 3)];
    let offset = ExtE::zero();
    let shifts = PackingShifts::<Ext>::new();
    let ctx = TransitionEvaluationContext::<Fp, Ext>::new_verifier(
        &frame, &rap, &alpha, &offset, &shifts,
    );

    let mut ext_evals = vec![ExtE::zero(), ExtE::zero()];
    eval_program_verifier(&prog, &ctx, &mut ext_evals);

    // The base-rooted constraint is promoted into the extension on write.
    assert_eq!(ext_evals[0], ext3(30, 0, 0) - ext3(12, 0, 0));
    assert_eq!(ext_evals[1], aux_val * alpha[0]);
}

// ------------------------------------------------------------------------
// roots indexed by emit(constraint_idx), in any emission order.
// ------------------------------------------------------------------------

#[test]
fn roots_indexed_by_constraint_idx_any_order() {
    let mut b = IrBuilder::<Fp, Ext>::new();
    let x = b.main(0, 0);
    // Emit idx 2 before idx 0 — roots must still land in the right slots.
    let x2 = b.mul(x, x);
    b.emit(2, x2);
    b.emit(0, x);
    let one = b.one();
    let xp1 = b.add(x, one);
    b.emit(1, xp1);
    let prog = b.finish(3);
    assert_eq!(prog.roots.len(), 3);

    let row = vec![fp(4)];
    assert_eq!(eval_program_base(&prog, 0, &row), fp(4));
    assert_eq!(eval_program_base(&prog, 1, &row), fp(4) + fp(1));
    assert_eq!(eval_program_base(&prog, 2, &row), fp(4) * fp(4));
}

// ------------------------------------------------------------------------
// complete flag plumbing.
// ------------------------------------------------------------------------

#[test]
fn complete_flag_defaults_true_and_mark_unsupported_clears_it() {
    let b = IrBuilder::<Fp, Ext>::new();
    assert!(b.finish(0).complete);

    let mut b = IrBuilder::<Fp, Ext>::new();
    b.mark_unsupported();
    assert!(!b.finish(0).complete);
}

// ------------------------------------------------------------------------
// Non-Goldilocks tower: E = F over the Baby-Bear-prime U32 test field.
// Exercises the reflexive IsSubFieldOf<F> impl and proves the module is
// genuinely field-generic. (This trimmed math crate has no Stark252-style
// big field; U32TestField has a different modulus AND a different BaseType
// (u32), so it is a strict genericity check.)
// ------------------------------------------------------------------------

#[test]
fn non_goldilocks_reflexive_tower_builds_and_interprets() {
    type G = U32TestField;
    type GE = FieldElement<G>;
    fn g(v: u64) -> GE {
        GE::from(v)
    }

    // Base-only program for eval_program_base (which walks every node and
    // accepts main leaves only): x * y + 3.
    let mut b0 = IrBuilder::<G, G>::new();
    let x = b0.main(0, 0);
    let y = b0.main(0, 1);
    let prod = b0.mul(x, y);
    let three = b0.const_base(3);
    let base_res = b0.add(prod, three);
    b0.emit(0, base_res);
    let base_prog = b0.finish(1);
    let row = vec![g(6), g(7)];
    assert_eq!(eval_program_base(&base_prog, 0, &row), g(6) * g(7) + g(3));

    // Program: idx 0 (base) = x * y + 3; idx 1 (ext = same field) = aux0 + 10.
    let mut b = IrBuilder::<G, G>::new();
    let x = b.main(0, 0);
    let y = b.main(0, 1);
    let prod = b.mul(x, y);
    let three = b.const_base(3);
    let base_res = b.add(prod, three);
    b.emit(0, base_res);

    let au = b.aux(0, 0); // "Ext" (= G here)
    let ec = b.const_ext(g(10));
    let ext_res = b.add(au, ec);
    assert_eq!(ext_res.dim(), Dim::Ext);
    b.emit(1, ext_res);

    let prog = b.finish(1);
    // Const dedup with a non-u64 BaseType (u32) still works.
    assert_eq!(prog.base_consts, vec![g(0), g(3)]);
    assert_eq!(prog.ext_consts, vec![g(10)]);

    // Full prover entry point with F = E = G.
    let step = TableView::<G, G>::new(vec![vec![g(6), g(7)]], vec![vec![g(4)]]);
    let frame = Frame::<G, G>::new(vec![step]);
    let rap: Vec<GE> = vec![];
    let alpha: Vec<GE> = vec![];
    let offset = g(0);
    let shifts = PackingShifts::<G>::new();
    let ctx =
        TransitionEvaluationContext::<G, G>::new_prover(&frame, &rap, &alpha, &offset, &shifts);
    let mut base_evals = vec![GE::zero()];
    let mut ext_evals = vec![GE::zero(), GE::zero()];
    eval_program(&prog, &ctx, &mut base_evals, &mut ext_evals);
    assert_eq!(base_evals[0], g(6) * g(7) + g(3));
    assert_eq!(ext_evals[1], g(4) + g(10));

    // Verifier entry point too (the frame is Frame<G, G> either way here).
    let vctx =
        TransitionEvaluationContext::<G, G>::new_verifier(&frame, &rap, &alpha, &offset, &shifts);
    let mut v_evals = vec![GE::zero(), GE::zero()];
    eval_program_verifier(&prog, &vctx, &mut v_evals);
    assert_eq!(v_evals[0], g(6) * g(7) + g(3));
    assert_eq!(v_evals[1], g(4) + g(10));
}

// ------------------------------------------------------------------------
// Random-row differential fuzz: a nontrivial base program vs direct math.
// ------------------------------------------------------------------------

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

#[test]
fn random_rows_match_direct_arithmetic() {
    // ((a + b) * c - a) * (b - c) + 5
    let mut bld = IrBuilder::<Fp, Ext>::new();
    let a = bld.main(0, 0);
    let b = bld.main(0, 1);
    let c = bld.main(0, 2);
    let ab = bld.add(a, b);
    let abc = bld.mul(ab, c);
    let abca = bld.sub(abc, a);
    let bc = bld.sub(b, c);
    let t = bld.mul(abca, bc);
    let five = bld.const_base(5);
    let res = bld.add(t, five);
    bld.emit(0, res);
    let prog = bld.finish(1);

    let mut rng = SplitMix64(0xDEAD_BEEF_CAFE_F00D);
    for _ in 0..1000 {
        let av = fp(rng.next_u64());
        let bv = fp(rng.next_u64());
        let cv = fp(rng.next_u64());
        let row = vec![av, bv, cv];
        let got = eval_program_base(&prog, 0, &row);
        let expected = ((av + bv) * cv - av) * (bv - cv) + fp(5);
        assert_eq!(got, expected);
    }
}
