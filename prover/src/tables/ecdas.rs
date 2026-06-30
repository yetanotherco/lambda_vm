//! ECDAS chip — one double/add step of the scalar-multiplication sequence.
//!
//! Each row receives an accumulator `(A, G, round, op)` on the self-referential `Ecdas`
//! bus, computes `R = 2A` (op=0) or `R = A + G` (op=1) via three byte-limb convolution
//! relations (`λ`, `xR`, `yR`, each with a 33-byte quotient + 64-entry carry array and the
//! offset `r = 3p`), and sends the updated accumulator back with `round − (1 − next_op)`
//! and `next_op`. When `next_op = 1` it consumes the scalar bit at `round` on the `Bit`
//! bus (an add follows). ECSM seeds and drains the bus; interior rows telescope.
//!
//! See `spec/src/ecdas.toml`. Constraints are **unconditional**; padding rows set the quotients
//! to `r` and `op = 0`, which makes every relation hold with zero carries.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraint_ir::{Capture, Expr, IrBuilder};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::IsBitConstraint;
use crate::tables::ecsm::ecdas_tuple;
use ecsm::{EcdasStep, P_BYTES};

pub(crate) use ecsm::R_BYTES;

// Bias signed convolution carries into IsHalfword [0, 2^16); see spec ecsm.typ "Carry offsets" (@ecsm-limb_carry).
pub(crate) const CARRY_OFFSET_LAMBDA: i64 = 32636;
pub(crate) const CARRY_OFFSET_XR: i64 = 8161;
pub(crate) const CARRY_OFFSET_YR: i64 = 16320;

// =========================================================================
// Column indices (~521 columns)
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const XG: usize = 2; // U256BL (32)
    pub const YG: usize = 34;
    pub const XA: usize = 66;
    pub const YA: usize = 98;
    pub const ROUND: usize = 130; // Byte
    pub const OP: usize = 131; // Bit
    pub const XR: usize = 132; // U256BL (32)
    pub const YR: usize = 164;
    pub const LAMBDA: usize = 196; // U256BL (32)
    pub const Q0: usize = 228; // Byte[33]
    pub const C0: usize = 261; // BaseField[64]
    pub const Q1: usize = 325; // Byte[33]
    pub const C1: usize = 358; // BaseField[64]
    pub const Q2: usize = 422; // Byte[33]
    pub const C2: usize = 455; // BaseField[64]
    pub const NEXT_OP: usize = 519; // Bit
    pub const MU: usize = 520;

    pub const NUM_COLUMNS: usize = 521;

    #[inline]
    pub const fn c0(i: usize) -> usize {
        C0 + i
    }
    #[inline]
    pub const fn c1(i: usize) -> usize {
        C1 + i
    }
    #[inline]
    pub const fn c2(i: usize) -> usize {
        C2 + i
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One ECDAS row: a double/add step witness plus its ECALL timestamp.
#[derive(Debug, Clone)]
pub struct EcdasOperation {
    pub timestamp: u64,
    pub step: EcdasStep,
}

// =========================================================================
// Trace generation
// =========================================================================

fn fe_from_i64(c: i64) -> FE {
    if c >= 0 {
        FE::from(c as u64)
    } else {
        FE::zero() - FE::from((-c) as u64)
    }
}

pub fn generate_ecdas_trace(
    ops: &[EcdasOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        let s = &op.step;

        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_bytes(row_idx, cols::XG, &s.x_g);
        table.set_bytes(row_idx, cols::YG, &s.y_g);
        table.set_bytes(row_idx, cols::XA, &s.x_a);
        table.set_bytes(row_idx, cols::YA, &s.y_a);
        table.set_byte(row_idx, cols::ROUND, s.round);
        table.set_byte(row_idx, cols::OP, s.op);
        table.set_bytes(row_idx, cols::XR, &s.x_r);
        table.set_bytes(row_idx, cols::YR, &s.y_r);
        table.set_bytes(row_idx, cols::LAMBDA, &s.lambda);
        table.set_bytes(row_idx, cols::Q0, &s.q0);
        table.set_bytes(row_idx, cols::Q1, &s.q1);
        table.set_bytes(row_idx, cols::Q2, &s.q2);
        for i in 0..64 {
            debug_assert!((0..1 << 16).contains(&(s.c0[i] + CARRY_OFFSET_LAMBDA)));
            debug_assert!((0..1 << 16).contains(&(s.c1[i] + CARRY_OFFSET_XR)));
            debug_assert!((0..1 << 16).contains(&(s.c2[i] + CARRY_OFFSET_YR)));
            table.set_fe(row_idx, cols::c0(i), fe_from_i64(s.c0[i]));
            table.set_fe(row_idx, cols::c1(i), fe_from_i64(s.c1[i]));
            table.set_fe(row_idx, cols::c2(i), fe_from_i64(s.c2[i]));
        }
        table.set_byte(row_idx, cols::NEXT_OP, s.next_op);
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows: q0 = q1 = q2 = r, op = 0, everything else 0. This makes every
    // (unconditional) convolution relation hold with zero carries.
    for row_idx in n..num_rows {
        table.set_bytes(row_idx, cols::Q0, &R_BYTES);
        table.set_bytes(row_idx, cols::Q1, &R_BYTES);
        table.set_bytes(row_idx, cols::Q2, &R_BYTES);
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

fn packed(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let ts_lo = || packed(cols::TIMESTAMP_0);
    let ts_hi = || packed(cols::TIMESTAMP_1);
    let mut out = Vec::new();

    // Receive [ts, xA, yA, xG, yG, round, op].
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        mu(),
        ecdas_tuple(
            cols::XA,
            cols::YA,
            cols::XG,
            cols::YG,
            packed(cols::ROUND),
            packed(cols::OP),
            ts_lo(),
            ts_hi(),
        ),
    ));

    // IS_BYTE range checks (single byte → AreBytes[x, 0]).
    let is_byte = |col: usize, len: usize, out: &mut Vec<BusInteraction>| {
        for i in 0..len {
            out.push(BusInteraction::sender(
                BusId::AreBytes,
                Multiplicity::Column(cols::MU),
                vec![packed(col + i), BusValue::constant(0)],
            ));
        }
    };
    is_byte(cols::ROUND, 1, &mut out);
    is_byte(cols::LAMBDA, 32, &mut out);
    is_byte(cols::Q0, 33, &mut out);
    is_byte(cols::XR, 32, &mut out);
    is_byte(cols::Q1, 33, &mut out);
    is_byte(cols::YR, 32, &mut out);
    is_byte(cols::Q2, 33, &mut out);

    // IS_HALF range checks on the carries (offsets keep them in [0, 2^16)).
    let half = |col: usize, off: i64| {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: col,
            },
            LinearTerm::Constant(off),
        ])
    };
    for (base, off) in [
        (cols::C0, CARRY_OFFSET_LAMBDA),
        (cols::C1, CARRY_OFFSET_XR),
        (cols::C2, CARRY_OFFSET_YR),
    ] {
        for i in 0..63 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                mu(),
                vec![half(base + i, off)],
            ));
        }
    }

    // Send Bit[ts, round] when adding next (mult = next_op).
    out.push(BusInteraction::sender(
        BusId::Bit,
        Multiplicity::Column(cols::NEXT_OP),
        vec![ts_lo(), ts_hi(), packed(cols::ROUND)],
    ));

    // Send the updated accumulator: [ts, xR, yR, xG, yG, round - 1 + next_op, next_op].
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        mu(),
        ecdas_tuple(
            cols::XR,
            cols::YR,
            cols::XG,
            cols::YG,
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::ROUND,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::NEXT_OP,
                },
                LinearTerm::Constant(-1),
            ]),
            packed(cols::NEXT_OP),
            ts_lo(),
            ts_hi(),
        ),
    ));

    out
}

// =========================================================================
// Constraints
// =========================================================================

fn p_byte<F: IsField>(m: usize) -> FieldElement<F> {
    if m < 32 {
        FieldElement::from(P_BYTES[m] as u64)
    } else {
        FieldElement::zero()
    }
}

fn r_byte<F: IsField>(m: usize) -> FieldElement<F> {
    if m < 33 {
        FieldElement::from(R_BYTES[m] as u64)
    } else {
        FieldElement::zero()
    }
}

#[derive(Clone, Copy)]
pub enum Relation {
    Lambda,
    Xr,
    Yr,
}

/// Unconditional convolution carry constraint at limb `i`: `2^8·c_i − c_{i-1} − S_i = 0`.
pub struct ConvCarry {
    pub relation: Relation,
    pub i: usize,
    pub constraint_idx: usize,
}

impl ConvCarry {
    fn s_i<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let i = self.i;
        let col = |c: usize| -> FieldElement<F> { step.get_main_evaluation_element(0, c).clone() };
        // bytes (zero beyond the stored length)
        let b = |base: usize, len: usize, j: usize| -> FieldElement<F> {
            if j < len {
                col(base + j)
            } else {
                FieldElement::zero()
            }
        };
        let lam = |j: usize| b(cols::LAMBDA, 32, j);
        let xg = |j: usize| b(cols::XG, 32, j);
        let xa = |j: usize| b(cols::XA, 32, j);
        let ya = |j: usize| b(cols::YA, 32, j);
        let yg = |j: usize| b(cols::YG, 32, j);
        let xr = |j: usize| b(cols::XR, 32, j);
        let yr = |j: usize| b(cols::YR, 32, j);
        let op = col(cols::OP);
        let one = FieldElement::<F>::one();

        // r·P − q·P convolution (shared structure across all three relations).
        let rq = |qbase: usize| -> FieldElement<F> {
            let mut s = FieldElement::<F>::zero();
            for j in 0..=i {
                s += (r_byte::<F>(j) - b(qbase, 33, j)) * p_byte::<F>(i - j);
            }
            s
        };

        match self.relation {
            Relation::Lambda => {
                // op·(Σ λ_j(xG-xA)_{i-j} + (yA_i - yG_i))
                let mut op_branch = ya(i) - yg(i);
                for j in 0..=i {
                    op_branch += lam(j) * (xg(i - j) - xa(i - j));
                }
                // (1-op)·Σ (2 λ_j yA_{i-j} - 3 xA_j xA_{i-j})
                let mut notop_branch = FieldElement::<F>::zero();
                for j in 0..=i {
                    notop_branch = notop_branch
                        + FieldElement::<F>::from(2u64) * lam(j) * ya(i - j)
                        - FieldElement::<F>::from(3u64) * xa(j) * xa(i - j);
                }
                op.clone() * op_branch + (one - op) * notop_branch + rq(cols::Q0)
            }
            Relation::Xr => {
                // Σ λ_j λ_{i-j} − xA_i − xG_i − xR_i − (1-op)(xA_i − xG_i) + rq
                let mut s = FieldElement::<F>::zero();
                for j in 0..=i {
                    s += lam(j) * lam(i - j);
                }
                s - xa(i) - xg(i) - xr(i) - (one - op) * (xa(i) - xg(i)) + rq(cols::Q1)
            }
            Relation::Yr => {
                // Σ λ_j(xA-xR)_{i-j} − yA_i − yR_i + rq
                let mut s = FieldElement::<F>::zero();
                for j in 0..=i {
                    s += lam(j) * (xa(i - j) - xr(i - j));
                }
                s - ya(i) - yr(i) + rq(cols::Q2)
            }
        }
    }

    /// Capture `s_i`, mirroring [`Self::s_i`].
    fn capture_s_i(&self, b: &mut IrBuilder) -> Expr {
        let i = self.i;
        let col = |c: usize, b: &mut IrBuilder| -> Expr { b.main(0, c) };
        // bytes (zero beyond the stored length)
        let byte = |base: usize, len: usize, j: usize, b: &mut IrBuilder| -> Expr {
            if j < len {
                col(base + j, b)
            } else {
                b.const_base(0)
            }
        };
        let lam = |j: usize, b: &mut IrBuilder| byte(cols::LAMBDA, 32, j, b);
        let xg = |j: usize, b: &mut IrBuilder| byte(cols::XG, 32, j, b);
        let xa = |j: usize, b: &mut IrBuilder| byte(cols::XA, 32, j, b);
        let ya = |j: usize, b: &mut IrBuilder| byte(cols::YA, 32, j, b);
        let yg = |j: usize, b: &mut IrBuilder| byte(cols::YG, 32, j, b);
        let xr = |j: usize, b: &mut IrBuilder| byte(cols::XR, 32, j, b);
        let yr = |j: usize, b: &mut IrBuilder| byte(cols::YR, 32, j, b);
        let p_byte_e = |m: usize, b: &mut IrBuilder| -> Expr {
            if m < 32 {
                b.const_base(P_BYTES[m] as u64)
            } else {
                b.const_base(0)
            }
        };
        let r_byte_e = |m: usize, b: &mut IrBuilder| -> Expr {
            if m < 33 {
                b.const_base(R_BYTES[m] as u64)
            } else {
                b.const_base(0)
            }
        };

        // r·P − q·P convolution (shared structure across all three relations).
        let rq = |qbase: usize, i: usize, b: &mut IrBuilder| -> Expr {
            let mut s = b.const_base(0);
            for j in 0..=i {
                let r_j = r_byte_e(j, b);
                let q_j = byte(qbase, 33, j, b);
                let diff = b.sub(r_j, q_j);
                let p_ij = p_byte_e(i - j, b);
                let term = b.mul(diff, p_ij);
                s = b.add(s, term);
            }
            s
        };

        match self.relation {
            Relation::Lambda => {
                let op = col(cols::OP, b);
                let one = b.one();
                // op·(Σ λ_j(xG-xA)_{i-j} + (yA_i - yG_i))
                let ya_i = ya(i, b);
                let yg_i = yg(i, b);
                let mut op_branch = b.sub(ya_i, yg_i);
                for j in 0..=i {
                    let lam_j = lam(j, b);
                    let xg_ij = xg(i - j, b);
                    let xa_ij = xa(i - j, b);
                    let diff = b.sub(xg_ij, xa_ij);
                    let term = b.mul(lam_j, diff);
                    op_branch = b.add(op_branch, term);
                }
                // (1-op)·Σ (2 λ_j yA_{i-j} - 3 xA_j xA_{i-j})
                let mut notop_branch = b.const_base(0);
                for j in 0..=i {
                    let two = b.const_base(2);
                    let lam_j = lam(j, b);
                    let ya_ij = ya(i - j, b);
                    let two_lam = b.mul(two, lam_j);
                    let term_pos = b.mul(two_lam, ya_ij);

                    let three = b.const_base(3);
                    let xa_j = xa(j, b);
                    let xa_ij = xa(i - j, b);
                    let three_xa = b.mul(three, xa_j);
                    let term_neg = b.mul(three_xa, xa_ij);

                    let sum_pos = b.add(notop_branch, term_pos);
                    notop_branch = b.sub(sum_pos, term_neg);
                }
                let op_term = b.mul(op, op_branch);
                let one_minus_op = b.sub(one, op);
                let notop_term = b.mul(one_minus_op, notop_branch);
                let rq_term = rq(cols::Q0, i, b);
                let s = b.add(op_term, notop_term);
                b.add(s, rq_term)
            }
            Relation::Xr => {
                let op = col(cols::OP, b);
                let one = b.one();
                // Σ λ_j λ_{i-j} − xA_i − xG_i − xR_i − (1-op)(xA_i − xG_i) + rq
                let mut s = b.const_base(0);
                for j in 0..=i {
                    let lam_j = lam(j, b);
                    let lam_ij = lam(i - j, b);
                    let term = b.mul(lam_j, lam_ij);
                    s = b.add(s, term);
                }
                let xa_i = xa(i, b);
                let xg_i = xg(i, b);
                let xr_i = xr(i, b);
                let s = b.sub(s, xa_i);
                let s = b.sub(s, xg_i);
                let s = b.sub(s, xr_i);
                let xa_i2 = xa(i, b);
                let xg_i2 = xg(i, b);
                let diff = b.sub(xa_i2, xg_i2);
                let one_minus_op = b.sub(one, op);
                let term2 = b.mul(one_minus_op, diff);
                let s = b.sub(s, term2);
                let rq_term = rq(cols::Q1, i, b);
                b.add(s, rq_term)
            }
            Relation::Yr => {
                // Σ λ_j(xA-xR)_{i-j} − yA_i − yR_i + rq
                let mut s = b.const_base(0);
                for j in 0..=i {
                    let lam_j = lam(j, b);
                    let xa_ij = xa(i - j, b);
                    let xr_ij = xr(i - j, b);
                    let diff = b.sub(xa_ij, xr_ij);
                    let term = b.mul(lam_j, diff);
                    s = b.add(s, term);
                }
                let ya_i = ya(i, b);
                let yr_i = yr(i, b);
                let s = b.sub(s, ya_i);
                let s = b.sub(s, yr_i);
                let rq_term = rq(cols::Q2, i, b);
                b.add(s, rq_term)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for ConvCarry {
    fn degree(&self) -> usize {
        match self.relation {
            Relation::Lambda => 3, // op · (λ · Δx)
            Relation::Xr | Relation::Yr => 2,
        }
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let c_base = match self.relation {
            Relation::Lambda => cols::C0,
            Relation::Xr => cols::C1,
            Relation::Yr => cols::C2,
        };
        let c_i = step.get_main_evaluation_element(0, c_base + self.i).clone();
        let c_prev = if self.i == 0 {
            FieldElement::<F>::zero()
        } else {
            step.get_main_evaluation_element(0, c_base + self.i - 1)
                .clone()
        };
        FieldElement::<F>::from(256u64) * c_i - c_prev - self.s_i(step)
    }
}

impl Capture for ConvCarry {
    fn capture(&self, b: &mut IrBuilder) {
        let c_base = match self.relation {
            Relation::Lambda => cols::C0,
            Relation::Xr => cols::C1,
            Relation::Yr => cols::C2,
        };
        let c_i = b.main(0, c_base + self.i);
        let c_prev = if self.i == 0 {
            b.const_base(0)
        } else {
            b.main(0, c_base + self.i - 1)
        };
        let s_i = self.capture_s_i(b);

        // 256·c_i − c_prev − s_i
        let two_five_six = b.const_base(256);
        let scaled = b.mul(two_five_six, c_i);
        let s = b.sub(scaled, c_prev);
        let root = b.sub(s, s_i);
        b.emit(self.constraint_idx, root);
    }
}

/// `col = 0` (unconditional, degree 1). Used for the closing `c_63 = 0`.
pub struct ColIsZero {
    pub col: usize,
    pub constraint_idx: usize,
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for ColIsZero {
    fn degree(&self) -> usize {
        1
    }
    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }
    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        step.get_main_evaluation_element(0, self.col).clone()
    }
}

impl Capture for ColIsZero {
    fn capture(&self, b: &mut IrBuilder) {
        let root = b.main(0, self.col);
        b.emit(self.constraint_idx, root);
    }
}

/// `a · b = 0` or `a · (1 - b) = 0` (degree 2).
pub struct MulZero {
    pub a: usize,
    pub b: usize,
    pub b_complement: bool,
    pub constraint_idx: usize,
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MulZero {
    fn degree(&self) -> usize {
        2
    }
    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }
    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let a = step.get_main_evaluation_element(0, self.a).clone();
        let b = step.get_main_evaluation_element(0, self.b).clone();
        if self.b_complement {
            a * (FieldElement::<F>::one() - b)
        } else {
            a * b
        }
    }
}

impl Capture for MulZero {
    fn capture(&self, builder: &mut IrBuilder) {
        let a = builder.main(0, self.a);
        let bval = builder.main(0, self.b);
        let root = if self.b_complement {
            let one = builder.one();
            let one_minus_b = builder.sub(one, bval);
            builder.mul(a, one_minus_b)
        } else {
            builder.mul(a, bval)
        };
        builder.emit(self.constraint_idx, root);
    }
}

/// Creates all ECDAS transition constraints (200 total).
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::new();
    let mut idx = constraint_idx_start;

    // IS_BIT on μ, op and next_op (the spec range-checks op: ecdas:c:range_op).
    for col in [cols::MU, cols::OP, cols::NEXT_OP] {
        constraints.push(IsBitConstraint::unconditional(col, idx).boxed());
        idx += 1;
    }

    // op · next_op = 0
    constraints.push(
        MulZero {
            a: cols::OP,
            b: cols::NEXT_OP,
            b_complement: false,
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;
    // next_op · (1 - mu) = 0
    constraints.push(
        MulZero {
            a: cols::NEXT_OP,
            b: cols::MU,
            b_complement: true,
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    // λ, xR, yR convolution carries + closings.
    for (relation, c_base) in [
        (Relation::Lambda, cols::C0),
        (Relation::Xr, cols::C1),
        (Relation::Yr, cols::C2),
    ] {
        for i in 0..64 {
            constraints.push(
                ConvCarry {
                    relation,
                    i,
                    constraint_idx: idx,
                }
                .boxed(),
            );
            idx += 1;
        }
        constraints.push(
            ColIsZero {
                col: c_base + 63,
                constraint_idx: idx,
            }
            .boxed(),
        );
        idx += 1;
    }

    (constraints, idx)
}
