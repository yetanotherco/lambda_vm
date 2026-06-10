//! Micro-benchmark: per-constraint `Box<dyn>` dispatch vs static dispatch in
//! the transition-constraint hot loop. Run with:
//! `cargo test -p stark --release dispatch_bench -- --ignored --nocapture`
//!
//! Three paths over the same constraints and data:
//! - A: `Vec<Box<dyn TransitionConstraintEvaluator>>`, one virtual call per
//!   constraint per LDE point (what production does today).
//! - B: enum + match dispatch (what a per-AIR `compute_transition_prover`
//!   override would use for heterogeneous constraint lists).
//! - C: hand-inlined straight-line evaluation (the P3-style upper bound).

use std::time::Instant;

use math::field::element::FieldElement;
use math::field::goldilocks::GoldilocksField;

use crate::constraints::transition::TransitionConstraintEvaluator;
use crate::frame::Frame;
use crate::lookup::PackingShifts;
use crate::trace::LDETraceTable;
use crate::traits::TransitionEvaluationContext;

type F = GoldilocksField;
type FE = FieldElement<F>;

const NUM_ROWS: usize = 1 << 18;
const NUM_COLS: usize = 32;
const NUM_CONSTRAINTS: usize = 32;
const REPS: usize = 3;

/// `flag · (a ⊕ b − c)` with ⊕ as the arithmetic XOR `a + b − 2ab`
/// (mirrors `BranchCondConstraint`-style bodies).
struct XorishConstraint {
    idx: usize,
    flag_col: usize,
    a_col: usize,
    b_col: usize,
    c_col: usize,
}

impl XorishConstraint {
    #[inline]
    fn compute(&self, ctx: &TransitionEvaluationContext<F, F>) -> FE {
        let TransitionEvaluationContext::Prover { frame, .. } = ctx else {
            unreachable!()
        };
        let step = frame.get_evaluation_step(0);
        let flag = step.get_main_evaluation_element(0, self.flag_col);
        let a = step.get_main_evaluation_element(0, self.a_col);
        let b = step.get_main_evaluation_element(0, self.b_col);
        let c = step.get_main_evaluation_element(0, self.c_col);
        let two = FE::from(2u64);
        let xor = a + b - two * a * b;
        flag * &(xor - c)
    }
}

impl TransitionConstraintEvaluator<F, F> for XorishConstraint {
    fn degree(&self) -> usize {
        3
    }
    fn constraint_idx(&self) -> usize {
        self.idx
    }
    fn evaluate_verifier(
        &self,
        ctx: &TransitionEvaluationContext<F, F>,
        evals: &mut [FieldElement<F>],
    ) {
        evals[self.idx] = self.compute(ctx);
    }
    fn evaluate_prover(
        &self,
        ctx: &TransitionEvaluationContext<F, F>,
        base_evals: &mut [FE],
        _ext_evals: &mut [FE],
    ) {
        base_evals[self.idx] = self.compute(ctx);
    }
}

/// `sel · (next − cur·cur − k)` reading both offsets (transition-style).
struct MulAccConstraint {
    idx: usize,
    sel_col: usize,
    cur_col: usize,
    next_col: usize,
}

impl MulAccConstraint {
    #[inline]
    fn compute(&self, ctx: &TransitionEvaluationContext<F, F>) -> FE {
        let TransitionEvaluationContext::Prover { frame, .. } = ctx else {
            unreachable!()
        };
        let cur = frame.get_evaluation_step(0);
        let next = frame.get_evaluation_step(1);
        let sel = cur.get_main_evaluation_element(0, self.sel_col);
        let x = cur.get_main_evaluation_element(0, self.cur_col);
        let y = next.get_main_evaluation_element(0, self.next_col);
        let k = FE::from(7u64);
        sel * &(y - x * x - k)
    }
}

impl TransitionConstraintEvaluator<F, F> for MulAccConstraint {
    fn degree(&self) -> usize {
        3
    }
    fn constraint_idx(&self) -> usize {
        self.idx
    }
    fn evaluate_verifier(
        &self,
        ctx: &TransitionEvaluationContext<F, F>,
        evals: &mut [FieldElement<F>],
    ) {
        evals[self.idx] = self.compute(ctx);
    }
    fn evaluate_prover(
        &self,
        ctx: &TransitionEvaluationContext<F, F>,
        base_evals: &mut [FE],
        _ext_evals: &mut [FE],
    ) {
        base_evals[self.idx] = self.compute(ctx);
    }
}

/// Packs four byte-columns into a word and checks against a word column
/// (mirrors `pack_bytes_to_word`-style bodies).
struct PackBytesConstraint {
    idx: usize,
    b0: usize,
    b1: usize,
    b2: usize,
    b3: usize,
    word_col: usize,
}

impl PackBytesConstraint {
    #[inline]
    fn compute(&self, ctx: &TransitionEvaluationContext<F, F>) -> FE {
        let TransitionEvaluationContext::Prover { frame, .. } = ctx else {
            unreachable!()
        };
        let step = frame.get_evaluation_step(0);
        let b0 = step.get_main_evaluation_element(0, self.b0);
        let b1 = step.get_main_evaluation_element(0, self.b1);
        let b2 = step.get_main_evaluation_element(0, self.b2);
        let b3 = step.get_main_evaluation_element(0, self.b3);
        let w = step.get_main_evaluation_element(0, self.word_col);
        let s8 = FE::from(1u64 << 8);
        let s16 = FE::from(1u64 << 16);
        let s24 = FE::from(1u64 << 24);
        b0 + &s8 * b1 + &s16 * b2 + &s24 * b3 - w
    }
}

impl TransitionConstraintEvaluator<F, F> for PackBytesConstraint {
    fn degree(&self) -> usize {
        1
    }
    fn constraint_idx(&self) -> usize {
        self.idx
    }
    fn evaluate_verifier(
        &self,
        ctx: &TransitionEvaluationContext<F, F>,
        evals: &mut [FieldElement<F>],
    ) {
        evals[self.idx] = self.compute(ctx);
    }
    fn evaluate_prover(
        &self,
        ctx: &TransitionEvaluationContext<F, F>,
        base_evals: &mut [FE],
        _ext_evals: &mut [FE],
    ) {
        base_evals[self.idx] = self.compute(ctx);
    }
}

/// Static-dispatch variant of the same heterogeneous constraint list.
enum StaticConstraint {
    Xorish(XorishConstraint),
    MulAcc(MulAccConstraint),
    PackBytes(PackBytesConstraint),
}

impl StaticConstraint {
    #[inline]
    fn evaluate_static(&self, ctx: &TransitionEvaluationContext<F, F>, base_evals: &mut [FE]) {
        match self {
            StaticConstraint::Xorish(c) => base_evals[c.idx] = c.compute(ctx),
            StaticConstraint::MulAcc(c) => base_evals[c.idx] = c.compute(ctx),
            StaticConstraint::PackBytes(c) => base_evals[c.idx] = c.compute(ctx),
        }
    }
}

#[allow(clippy::type_complexity)]
fn build_constraints() -> (
    Vec<Box<dyn TransitionConstraintEvaluator<F, F>>>,
    Vec<StaticConstraint>,
) {
    let mut boxed: Vec<Box<dyn TransitionConstraintEvaluator<F, F>>> = Vec::new();
    let mut statics: Vec<StaticConstraint> = Vec::new();
    for idx in 0..NUM_CONSTRAINTS {
        let c0 = idx % NUM_COLS;
        let c = |k: usize| (c0 + k) % NUM_COLS;
        match idx % 3 {
            0 => {
                boxed.push(Box::new(XorishConstraint {
                    idx,
                    flag_col: c(0),
                    a_col: c(1),
                    b_col: c(2),
                    c_col: c(3),
                }));
                statics.push(StaticConstraint::Xorish(XorishConstraint {
                    idx,
                    flag_col: c(0),
                    a_col: c(1),
                    b_col: c(2),
                    c_col: c(3),
                }));
            }
            1 => {
                boxed.push(Box::new(MulAccConstraint {
                    idx,
                    sel_col: c(0),
                    cur_col: c(1),
                    next_col: c(2),
                }));
                statics.push(StaticConstraint::MulAcc(MulAccConstraint {
                    idx,
                    sel_col: c(0),
                    cur_col: c(1),
                    next_col: c(2),
                }));
            }
            _ => {
                boxed.push(Box::new(PackBytesConstraint {
                    idx,
                    b0: c(0),
                    b1: c(1),
                    b2: c(2),
                    b3: c(3),
                    word_col: c(4),
                }));
                statics.push(StaticConstraint::PackBytes(PackBytesConstraint {
                    idx,
                    b0: c(0),
                    b1: c(1),
                    b2: c(2),
                    b3: c(3),
                    word_col: c(4),
                }));
            }
        }
    }
    (boxed, statics)
}

fn build_lde() -> LDETraceTable<F, F> {
    // Pseudo-random but deterministic data; LCG over u64 mapped into the field.
    let mut state: u64 = 0x243F_6A88_85A3_08D3;
    let data: Vec<FE> = (0..NUM_ROWS * NUM_COLS)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            FE::from(state >> 1)
        })
        .collect();
    LDETraceTable::from_row_major(data, NUM_COLS, Vec::new(), 0, 1, 2)
}

#[test]
#[ignore]
#[allow(clippy::needless_range_loop)]
fn dispatch_bench() {
    let lde = build_lde();
    let (boxed, statics) = build_constraints();
    let offsets = [0usize, 1usize];
    let packing_shifts = PackingShifts::<F>::new();
    let rap: Vec<FE> = Vec::new();
    let alphas: Vec<FE> = Vec::new();
    let table_offset = FE::from(0u64);

    let mut frame = Frame::<F, F>::preallocate(2, 1, NUM_COLS, 0);
    let mut base_buf = vec![FE::from(0u64); NUM_CONSTRAINTS];
    let mut ext_buf: Vec<FE> = Vec::new();

    // Checksums guard that all three paths compute the same thing (and stop
    // the optimizer from deleting the loops).
    let mut sums = [FE::from(0u64); 3];
    let mut times = [0.0f64; 3];

    for rep in 0..REPS {
        for (path, time_slot) in times.iter_mut().enumerate() {
            let t = Instant::now();
            let mut acc = FE::from(0u64);
            for row in 0..NUM_ROWS - 2 {
                frame.fill_from_lde(&lde, row, &offsets);
                let ctx = TransitionEvaluationContext::new_prover(
                    &frame,
                    &[],
                    &rap,
                    &alphas,
                    &table_offset,
                    &packing_shifts,
                );
                match path {
                    0 => {
                        boxed
                            .iter()
                            .for_each(|c| c.evaluate_prover(&ctx, &mut base_buf, &mut ext_buf));
                    }
                    1 => {
                        statics
                            .iter()
                            .for_each(|c| c.evaluate_static(&ctx, &mut base_buf));
                    }
                    _ => {
                        let TransitionEvaluationContext::Prover { frame, .. } = &ctx else {
                            unreachable!()
                        };
                        let cur = frame.get_evaluation_step(0);
                        let next = frame.get_evaluation_step(1);
                        let two = FE::from(2u64);
                        let k = FE::from(7u64);
                        let s8 = FE::from(1u64 << 8);
                        let s16 = FE::from(1u64 << 16);
                        let s24 = FE::from(1u64 << 24);
                        for idx in 0..NUM_CONSTRAINTS {
                            let c0 = idx % NUM_COLS;
                            let c = |kk: usize| (c0 + kk) % NUM_COLS;
                            base_buf[idx] = match idx % 3 {
                                0 => {
                                    let flag = cur.get_main_evaluation_element(0, c(0));
                                    let a = cur.get_main_evaluation_element(0, c(1));
                                    let b = cur.get_main_evaluation_element(0, c(2));
                                    let cc = cur.get_main_evaluation_element(0, c(3));
                                    let xor = a + b - &two * a * b;
                                    flag * &(xor - cc)
                                }
                                1 => {
                                    let sel = cur.get_main_evaluation_element(0, c(0));
                                    let x = cur.get_main_evaluation_element(0, c(1));
                                    let y = next.get_main_evaluation_element(0, c(2));
                                    sel * &(y - x * x - &k)
                                }
                                _ => {
                                    let b0 = cur.get_main_evaluation_element(0, c(0));
                                    let b1 = cur.get_main_evaluation_element(0, c(1));
                                    let b2 = cur.get_main_evaluation_element(0, c(2));
                                    let b3 = cur.get_main_evaluation_element(0, c(3));
                                    let w = cur.get_main_evaluation_element(0, c(4));
                                    b0 + &s8 * b1 + &s16 * b2 + &s24 * b3 - w
                                }
                            };
                        }
                    }
                }
                for v in &base_buf {
                    acc = acc + v;
                }
            }
            *time_slot += t.elapsed().as_secs_f64();
            if rep == 0 {
                sums[path] = acc;
            } else {
                assert_eq!(sums[path], acc, "non-deterministic path {path}");
            }
        }
    }

    assert_eq!(sums[0], sums[1], "static path diverges from boxed path");
    assert_eq!(sums[0], sums[2], "inlined path diverges from boxed path");

    let per = |t: f64| t / REPS as f64;
    println!("rows={NUM_ROWS} cols={NUM_COLS} constraints={NUM_CONSTRAINTS} reps={REPS}");
    println!("A boxed dyn   : {:>8.3} ms/pass", per(times[0]) * 1e3);
    println!(
        "B enum match  : {:>8.3} ms/pass  ({:.2}x vs A)",
        per(times[1]) * 1e3,
        per(times[0]) / per(times[1])
    );
    println!(
        "C hand inline : {:>8.3} ms/pass  ({:.2}x vs A)",
        per(times[2]) * 1e3,
        per(times[0]) / per(times[2])
    );
}
