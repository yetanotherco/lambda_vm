//! The `FIELD_VM` chip: one row per state.
//!
//! Spec: `spec/src/field_vm.toml`. Every `ExtField` variable takes three base
//! columns, and every `ExtField` constraint is emitted as three base
//! constraints with the `x³ = 2` reduction.

use core::ops::{Add, Mul, Sub};

use stark::constraints::boundary::BoundaryConstraint;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet, RowDomain};
use stark::lookup::{
    BoundaryConstraintBuilder, BusInteraction, BusValue, LinearTerm, Multiplicity, Packing,
};
use stark::trace::TraceTable;

use super::executor::{Execution, State, Step};
use super::isa::{Instr, N, NUM_REGS, Program};
use super::mux::{D, MuxTables, Split, T, split_powers};
use crate::tables::types::{BusId, FE, FEE, GoldilocksExtension, GoldilocksField, zeroed_fe_vec};

type F = GoldilocksField;
type E = GoldilocksExtension;

pub mod cols {
    use super::super::isa::N;
    use super::super::mux::T;

    pub const PC: usize = 0;
    /// `argument_registers[4]`, for `d, a, b, c`.
    pub const ARG_REG: usize = 1;
    /// `argument_scalars[4]`.
    pub const ARG_SCALE: usize = ARG_REG + 4;
    /// `argument_offsets[4]`.
    pub const ARG_OFFSET: usize = ARG_SCALE + 4;
    /// `mem_flags[4]`.
    pub const MEM_FLAG: usize = ARG_OFFSET + 4;
    /// `hint_input[N]`.
    pub const HINT_IN: usize = MEM_FLAG + 4;
    pub const HINT_OUT: usize = HINT_IN + N;
    /// `registers[N]`, three columns each.
    pub const REGS: usize = HINT_OUT + 1;
    pub const ZERO: usize = REGS + 3 * N;
    pub const OUT_INV: usize = ZERO + 1;
    /// `arg_reg_pows_computed[4][T]`.
    pub const POWS: usize = OUT_INV + 3;
    pub const ARGS_PREMEM: usize = POWS + 4 * T;
    pub const ARGS: usize = ARGS_PREMEM + 12;
    pub const NUM_COLUMNS: usize = ARGS + 12;

    pub const fn reg(g: usize) -> usize {
        REGS + 3 * g
    }
    /// `x_i^{split_exponent(k)}` for `k = 1..=T`.
    pub const fn pow(i: usize, k: usize) -> usize {
        POWS + i * T + (k - 1)
    }
    pub const fn premem(i: usize) -> usize {
        ARGS_PREMEM + 3 * i
    }
    pub const fn arg(i: usize) -> usize {
        ARGS + 3 * i
    }
}

/// Bit `i` of the packed flags: `mem_flags` (0..4), `hint_output` (4), `hint_input` (5..).
pub const FLAG_HINT_OUT: u32 = 4;
pub const FLAG_HINT_IN: u32 = 5;
/// Bits per register index in the packed argument registers.
pub const REG_BITS: u32 = 8;

const _: () = assert!(FLAG_HINT_IN as usize + N <= 64 && NUM_REGS <= 1 << REG_BITS);

/// The eleven base values of one `FIELD_VM_DECODE` tuple: `pc`, the leftover
/// `imm0`/`imm1` of `c`, the other `imm0`s, the other `imm1`s, the flags and
/// the packed register indices.
pub fn decode_tuple(pc: u64, instr: &Instr) -> [FE; 11] {
    let args = instr.args();
    let mut flags = 0u64;
    for (i, a) in args.iter().enumerate() {
        flags |= (a.mem as u64) << i;
    }
    flags |= (instr.hint_out as u64) << FLAG_HINT_OUT;
    for g in 0..N {
        flags |= (instr.hint_in[g] as u64) << (FLAG_HINT_IN as usize + g);
    }
    let regs = args.iter().enumerate().fold(0u64, |acc, (i, a)| {
        acc | (a.reg as u64) << (REG_BITS as usize * i)
    });
    [
        FE::from(pc),
        args[3].scale,
        args[3].offset,
        args[0].scale,
        args[1].scale,
        args[2].scale,
        args[0].offset,
        args[1].offset,
        args[2].offset,
        FE::from(flags),
        FE::from(regs),
    ]
}

// =========================================================================
// Trace generation
// =========================================================================

fn set_ext(t: &mut stark::table::Table<F>, row: usize, col: usize, v: &FEE) {
    for (k, c) in v.value().iter().enumerate() {
        t.set(row, col + k, *c);
    }
}

fn fill_row(t: &mut stark::table::Table<F>, row: usize, program: &Program, step: &Step) {
    let State { zero, pc, regs } = step.state;
    let instr = &program.instrs[pc as usize];
    t.set(row, cols::PC, FE::from(pc));
    for (i, a) in instr.args().iter().enumerate() {
        let x = FE::from(a.reg as u64);
        t.set(row, cols::ARG_REG + i, x);
        t.set(row, cols::ARG_SCALE + i, a.scale);
        t.set(row, cols::ARG_OFFSET + i, a.offset);
        t.set(row, cols::MEM_FLAG + i, FE::from(a.mem as u64));
        let pows = split_powers(x);
        for (k, p) in pows.iter().enumerate().skip(1) {
            t.set(row, cols::pow(i, k), *p);
        }
        set_ext(t, row, cols::premem(i), &step.args_premem[i]);
        set_ext(t, row, cols::arg(i), &step.args[i]);
    }
    for (g, r) in regs.iter().enumerate() {
        t.set(row, cols::HINT_IN + g, FE::from(instr.hint_in[g] as u64));
        set_ext(t, row, cols::reg(g), r);
    }
    t.set(row, cols::HINT_OUT, FE::from(instr.hint_out as u64));
    t.set(row, cols::ZERO, FE::from(zero as u64));
    let d = step.args[0];
    let inv = if d == FEE::zero() {
        FEE::zero()
    } else {
        d.inv().expect("nonzero")
    };
    set_ext(t, row, cols::OUT_INV, &inv);
}

/// The FIELD_VM trace, padded to a power of two with halt steps, and the
/// DECODE multiplicities including the padding.
pub fn generate_trace(
    program: &Program,
    exec: &Execution,
    min_rows: usize,
) -> (TraceTable<F, E>, Vec<u64>) {
    let num_rows = exec.steps.len().next_power_of_two().max(min_rows);
    let halt = *exec
        .steps
        .last()
        .expect("an execution ends with a halt step");
    let mut trace = TraceTable::new_main(
        zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    for row in 0..num_rows {
        let step = exec.steps.get(row).unwrap_or(&halt);
        fill_row(&mut trace.main_table, row, program, step);
    }
    let mut decode_mult = exec.decode_mult.clone();
    decode_mult[halt.state.pc as usize] += (num_rows - exec.steps.len()) as u64;
    (trace, decode_mult)
}

// =========================================================================
// Bus interactions
// =========================================================================

fn direct(c: usize) -> BusValue {
    BusValue::Packed {
        start_column: c,
        packing: Packing::Direct,
    }
}

/// - **Sends** `FIELD_VM_DECODE[decode_tuple]` once per row.
/// - **Sends** `FIELD_VM_MEM[args_premem[i]] -> args[i]` when `mem_flags[i]`.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut flags: Vec<LinearTerm> = (0..4)
        .map(|i| LinearTerm::Column {
            coefficient: 1 << i,
            column: cols::MEM_FLAG + i,
        })
        .collect();
    flags.push(LinearTerm::Column {
        coefficient: 1 << FLAG_HINT_OUT,
        column: cols::HINT_OUT,
    });
    flags.extend((0..N).map(|g| LinearTerm::Column {
        coefficient: 1 << (FLAG_HINT_IN as usize + g),
        column: cols::HINT_IN + g,
    }));
    let regs = (0..4)
        .map(|i| LinearTerm::Column {
            coefficient: 1 << (REG_BITS as usize * i),
            column: cols::ARG_REG + i,
        })
        .collect();

    let mut decode = vec![
        direct(cols::PC),
        direct(cols::ARG_SCALE + 3),
        direct(cols::ARG_OFFSET + 3),
    ];
    decode.extend((0..3).map(|i| direct(cols::ARG_SCALE + i)));
    decode.extend((0..3).map(|i| direct(cols::ARG_OFFSET + i)));
    decode.push(BusValue::linear(flags));
    decode.push(BusValue::linear(regs));

    let mut interactions = vec![BusInteraction::sender(
        BusId::FieldVmDecode,
        Multiplicity::One,
        decode,
    )];
    for i in 0..4 {
        let values = (0..3)
            .map(|k| direct(cols::premem(i) + k))
            .chain((0..3).map(|k| direct(cols::arg(i) + k)))
            .collect();
        interactions.push(BusInteraction::sender(
            BusId::FieldVmMem,
            Multiplicity::Column(cols::MEM_FLAG + i),
            values,
        ));
    }
    interactions
}

// =========================================================================
// Constraints
// =========================================================================

type Ext3<X> = [X; 3];

fn ext_add<X: Clone + Add<Output = X>>(a: &Ext3<X>, b: &Ext3<X>) -> Ext3<X> {
    core::array::from_fn(|k| a[k].clone() + b[k].clone())
}

fn ext_sub<X: Clone + Sub<Output = X>>(a: &Ext3<X>, b: &Ext3<X>) -> Ext3<X> {
    core::array::from_fn(|k| a[k].clone() - b[k].clone())
}

fn ext_scale<X: Clone + Mul<Output = X>>(s: &X, a: &Ext3<X>) -> Ext3<X> {
    core::array::from_fn(|k| s.clone() * a[k].clone())
}

/// `a · b` in `F[w]/(w³ − 2)`.
fn ext_mul<X: Clone + Add<Output = X> + Mul<Output = X>>(
    a: &Ext3<X>,
    b: &Ext3<X>,
    two: &X,
) -> Ext3<X> {
    let m = |i: usize, j: usize| a[i].clone() * b[j].clone();
    [
        m(0, 0) + two.clone() * (m(1, 2) + m(2, 1)),
        m(0, 1) + m(1, 0) + two.clone() * m(2, 2),
        m(0, 2) + m(1, 1) + m(2, 0),
    ]
}

#[derive(Clone, Default)]
pub struct FieldVmConstraints {
    tables: MuxTables,
}

impl FieldVmConstraints {
    pub fn new() -> Self {
        Self::default()
    }
}

struct Emitter {
    idx: usize,
}

impl Emitter {
    fn all<B: ConstraintBuilder<F, E>>(&mut self, b: &mut B, e: B::Expr) {
        b.emit_base(self.idx, e);
        self.idx += 1;
    }

    fn next<B: ConstraintBuilder<F, E>>(&mut self, b: &mut B, e: B::Expr) {
        b.emit_base_rows(self.idx, RowDomain::except_last(1), e);
        self.idx += 1;
    }
}

fn ext_col<B: ConstraintBuilder<F, E>>(b: &B, off: usize, col: usize) -> Ext3<B::Expr> {
    core::array::from_fn(|k| b.main(off, col + k))
}

fn all_regs<B: ConstraintBuilder<F, E>>(b: &B, off: usize) -> Vec<Ext3<B::Expr>> {
    let mut regs = vec![
        [b.main(off, cols::ZERO), b.zero(), b.zero()],
        [b.main(off, cols::PC), b.zero(), b.zero()],
    ];
    regs.extend((0..N).map(|g| ext_col(b, off, cols::reg(g))));
    regs
}

/// `Σ_k pow_{i,k} Σ_l s[k][l] x_i^l`, skipping zero coefficients so the
/// captured degree is the real one.
fn eval_split<B: ConstraintBuilder<F, E>>(b: &B, s: &Split, i: usize) -> B::Expr {
    let x = b.main(0, cols::ARG_REG + i);
    let mut acc: Option<B::Expr> = None;
    for (k, split) in s.iter().enumerate() {
        let mut inner: Option<B::Expr> = None;
        let mut x_pow: Option<B::Expr> = None;
        for c in split.iter() {
            if *c != FE::zero() {
                let coeff = b.const_base(c.canonical());
                let term = match &x_pow {
                    None => coeff,
                    Some(p) => coeff * p.clone(),
                };
                inner = Some(match inner {
                    None => term,
                    Some(acc) => acc + term,
                });
            }
            x_pow = Some(match x_pow {
                None => x.clone(),
                Some(p) => p * x.clone(),
            });
        }
        let Some(inner) = inner else { continue };
        let term = if k == 0 {
            inner
        } else {
            b.main(0, cols::pow(i, k)) * inner
        };
        acc = Some(match acc {
            None => term,
            Some(a) => a + term,
        });
    }
    acc.unwrap_or_else(|| b.zero())
}

impl ConstraintSet<F, E> for FieldVmConstraints {
    fn max_degree(&self) -> usize {
        D
    }

    fn eval<B: ConstraintBuilder<F, E>>(&self, b: &mut B) {
        let mut em = Emitter { idx: 0 };
        let one = b.one();
        let two = b.const_base(2);
        let bit = |b: &B, c: usize| {
            let x = b.main(0, c);
            x.clone() * (x - b.one())
        };

        // decode
        for i in 0..4 {
            let e = bit(b, cols::MEM_FLAG + i);
            em.all(b, e);
        }
        for g in 0..N {
            let e = bit(b, cols::HINT_IN + g);
            em.all(b, e);
        }
        let e = bit(b, cols::HINT_OUT);
        em.all(b, e);
        for i in 0..4 {
            let e = eval_split(b, &self.tables.range, i);
            em.all(b, e);
        }

        // mux
        for i in 0..4 {
            let x = b.main(0, cols::ARG_REG + i);
            let mut prev: Option<B::Expr> = None;
            for k in 1..=T {
                let step = if k == 1 { D - 1 } else { D - 2 };
                let mut p = prev.clone().unwrap_or_else(|| b.one());
                for _ in 0..step {
                    p = p * x.clone();
                }
                let e = b.main(0, cols::pow(i, k)) - p;
                em.all(b, e);
                prev = Some(b.main(0, cols::pow(i, k)));
            }
        }
        let premem = |b: &B, i: usize, off: usize| -> Ext3<B::Expr> {
            let regs = all_regs(b, off);
            let mut sum: Ext3<B::Expr> = [b.zero(), b.zero(), b.zero()];
            for (j, r) in regs.iter().enumerate() {
                let sel = eval_split(b, &self.tables.mux[j], i);
                sum = ext_add(&sum, &ext_scale(&sel, r));
            }
            let scaled = ext_scale(&b.main(0, cols::ARG_SCALE + i), &sum);
            [
                scaled[0].clone() + b.main(0, cols::ARG_OFFSET + i),
                scaled[1].clone(),
                scaled[2].clone(),
            ]
        };
        for i in 1..4 {
            let expected = premem(b, i, 0);
            let diff = ext_sub(&ext_col(b, 0, cols::premem(i)), &expected);
            for e in diff {
                em.all(b, e);
            }
        }
        let expected = premem(b, 0, 1);
        let diff = ext_sub(&ext_col(b, 0, cols::premem(0)), &expected);
        for e in diff {
            em.next(b, e);
        }

        // memory
        for i in 0..4 {
            let not_mem = one.clone() - b.main(0, cols::MEM_FLAG + i);
            let diff = ext_sub(
                &ext_col(b, 0, cols::arg(i)),
                &ext_col(b, 0, cols::premem(i)),
            );
            for e in diff {
                em.all(b, not_mem.clone() * e);
            }
        }

        // fma
        let args: Vec<Ext3<B::Expr>> = (0..4).map(|i| ext_col(b, 0, cols::arg(i))).collect();
        let rhs = ext_add(&ext_mul(&args[1], &args[2], &two), &args[3]);
        for e in ext_sub(&args[0], &rhs) {
            em.all(b, e);
        }

        // transition
        let not_hint_out = one.clone() - b.main(0, cols::HINT_OUT);
        for g in 0..N {
            let not_hinted_next = one.clone() - b.main(1, cols::HINT_IN + g);
            let not_out_reg = one.clone() - eval_split(b, &self.tables.mux[g + 2], 0);
            let diff = ext_sub(&ext_col(b, 1, cols::reg(g)), &ext_col(b, 0, cols::reg(g)));
            for e in diff.iter() {
                em.next(
                    b,
                    not_hinted_next.clone() * not_hint_out.clone() * e.clone(),
                );
            }
            for e in diff {
                em.next(b, not_hinted_next.clone() * not_out_reg.clone() * e);
            }
        }
        let pc_step = b.main(1, cols::PC) - b.main(0, cols::PC) - one.clone();
        em.next(b, not_hint_out * pc_step.clone());
        let not_out_pc = one.clone() - eval_split(b, &self.tables.mux[1], 0);
        em.next(b, not_out_pc * pc_step);
        let e = bit(b, cols::ZERO);
        em.all(b, e);
        let zero_next = b.main(1, cols::ZERO);
        for e in args[0].iter() {
            em.next(b, zero_next.clone() * e.clone());
        }
        let not_zero_next = one.clone() - zero_next;
        let prod = ext_mul(&args[0], &ext_col(b, 0, cols::OUT_INV), &two);
        let target = [one.clone(), b.zero(), b.zero()];
        for e in ext_sub(&prod, &target) {
            em.next(b, not_zero_next.clone() * e);
        }
    }
}

// =========================================================================
// Boundary constraints
// =========================================================================

#[derive(
    Clone, Debug, Default, PartialEq, Eq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
pub struct FvmPublicInputs {
    /// Index of the last FIELD_VM row; checked against the proof's trace length.
    pub last_row: usize,
}

/// First row: `PC = 1`, `ZERO = 0`, registers `0`. Last row: the halt state.
pub struct FieldVmBoundary;

impl BoundaryConstraintBuilder<F, E, FvmPublicInputs> for FieldVmBoundary {
    fn boundary_constraints(pi: &FvmPublicInputs, _: &[FEE]) -> Vec<BoundaryConstraint<E>> {
        let mut out = Vec::with_capacity(2 * (2 + 3 * N));
        for (step, pc, zero) in [(0, 1u64, 0u64), (pi.last_row, 0, 1)] {
            out.push(BoundaryConstraint::new_main(cols::PC, step, FEE::from(pc)));
            out.push(BoundaryConstraint::new_main(
                cols::ZERO,
                step,
                FEE::from(zero),
            ));
            for c in cols::REGS..cols::REGS + 3 * N {
                out.push(BoundaryConstraint::new_main(c, step, FEE::zero()));
            }
        }
        out
    }
}
