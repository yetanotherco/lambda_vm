//! ECDAS chip — one double/add step of the scalar-multiplication sequence.
//!
//! Each row receives an accumulator `(A, G, round, op)` on the self-referential `Ecdas`
//! bus, computes `R = 2A` (op=0) or `R = A + G` (op=1) via three byte-limb convolution
//! relations (`λ`, `xR`, `yR`, each with a 33-byte quotient + 64-entry carry array and the
//! offset `r = 3p`), and sends the updated accumulator back with `round − (1 − next_op)`
//! and `next_op`. When `next_op = 1` it consumes the scalar bit at `round` on the `Bit`
//! bus (an add follows). ECSM seeds and drains the bus; interior rows telescope.
//!
//! See `spec/ecdas.toml`. Constraints are **unconditional**; padding rows set the quotients
//! to `r` and `op = 0`, which makes every relation hold with zero carries.

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::constraints::templates::IsBitConstraint;
use crate::tables::ecsm::ecdas_tuple;
use ecsm::{EcdasStep, P_BYTES};

/// `r = 3·p` as 33 little-endian bytes (the spec offset that keeps all quotients positive).
const R_BYTES: [u8; 33] = [
    0x8D, 0xF4, 0xFF, 0xFF, 0xFC, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
    0x02,
];

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

fn write_bytes(data: &mut [FE], base: usize, col: usize, bytes: &[u8]) {
    for (i, &b) in bytes.iter().enumerate() {
        data[base + col + i] = FE::from(b as u64);
    }
}

pub fn generate_ecdas_trace(
    ops: &[EcdasOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let s = &op.step;

        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);
        write_bytes(&mut data, base, cols::XG, &s.x_g);
        write_bytes(&mut data, base, cols::YG, &s.y_g);
        write_bytes(&mut data, base, cols::XA, &s.x_a);
        write_bytes(&mut data, base, cols::YA, &s.y_a);
        data[base + cols::ROUND] = FE::from(s.round as u64);
        data[base + cols::OP] = FE::from(s.op as u64);
        write_bytes(&mut data, base, cols::XR, &s.x_r);
        write_bytes(&mut data, base, cols::YR, &s.y_r);
        write_bytes(&mut data, base, cols::LAMBDA, &s.lambda);
        write_bytes(&mut data, base, cols::Q0, &s.q0);
        write_bytes(&mut data, base, cols::Q1, &s.q1);
        write_bytes(&mut data, base, cols::Q2, &s.q2);
        for i in 0..64 {
            data[base + cols::c0(i)] = fe_from_i64(s.c0[i]);
            data[base + cols::c1(i)] = fe_from_i64(s.c1[i]);
            data[base + cols::c2(i)] = fe_from_i64(s.c2[i]);
        }
        data[base + cols::NEXT_OP] = FE::from(s.next_op as u64);
        data[base + cols::MU] = FE::one();
    }

    // Padding rows: q0 = q1 = q2 = r, op = 0, everything else 0. This makes every
    // (unconditional) convolution relation hold with zero carries.
    for row_idx in n..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        write_bytes(&mut data, base, cols::Q0, &R_BYTES);
        write_bytes(&mut data, base, cols::Q1, &R_BYTES);
        write_bytes(&mut data, base, cols::Q2, &R_BYTES);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
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
    for (base, off) in [(cols::C0, 32636i64), (cols::C1, 8161), (cols::C2, 16320)] {
        for i in 0..63 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                mu(),
                vec![half(base + i, off)],
            ));
        }
    }

    // Receive Bit[ts, round] when adding next (mult = next_op).
    out.push(BusInteraction::receiver(
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
enum Relation {
    Lambda,
    Xr,
    Yr,
}

/// Unconditional convolution carry constraint at limb `i`: `2^8·c_i − c_{i-1} − S_i = 0`.
struct ConvCarry {
    relation: Relation,
    i: usize,
    constraint_idx: usize,
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

/// `col = 0` (unconditional, degree 1). Used for the closing `c_63 = 0`.
struct ColIsZero {
    col: usize,
    constraint_idx: usize,
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

/// `a · b = 0` or `a · (1 - b) = 0` (degree 2).
struct MulZero {
    a: usize,
    b: usize,
    b_complement: bool,
    constraint_idx: usize,
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

/// Creates all ECDAS transition constraints (199 total).
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

    // IS_BIT(mu), IS_BIT(op), IS_BIT(next_op). `op` is also boolean transitively (it arrives
    // on the Ecdas bus from a constant-0 seed or a prior IS_BIT'd next_op); the direct check
    // makes the constraint layer self-contained (the λ relation blends op·add + (1−op)·double).
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

#[cfg(test)]
mod tests {
    use super::*;
    use ecsm::compute_witness;

    fn gx_le() -> [u8; 32] {
        let mut be = [
            0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87,
            0x0B, 0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B,
            0x16, 0xF8, 0x17, 0x98,
        ];
        be.reverse();
        be
    }

    fn k_le(v: u64) -> [u8; 32] {
        let mut k = [0u8; 32];
        k[..8].copy_from_slice(&v.to_le_bytes());
        k
    }

    fn ops_for(k: u64) -> Vec<EcdasOperation> {
        let w = compute_witness(&k_le(k), &gx_le()).unwrap();
        w.steps
            .into_iter()
            .map(|step| EcdasOperation {
                timestamp: 444,
                step,
            })
            .collect()
    }

    fn row_view(
        trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
        row: usize,
    ) -> TableView<GoldilocksField, GoldilocksExtension> {
        let main: Vec<FE> = (0..cols::NUM_COLUMNS)
            .map(|c| *trace.main_table.get(row, c))
            .collect();
        TableView::new(vec![main], vec![])
    }

    #[test]
    fn r_bytes_is_three_p() {
        // 3·p as 33 little-endian bytes, cross-checked against the ecsm field modulus.
        let p = ecsm::p();
        let three_p = &p * 3u32;
        let mut bytes = three_p.to_bytes_le();
        bytes.resize(33, 0);
        assert_eq!(&bytes[..], &R_BYTES[..]);
    }

    /// Every ECDAS constraint evaluates to zero on a generated trace across many scalars
    /// (which exercise both double and add steps), including padding rows.
    #[test]
    fn constraints_hold_on_generated_trace() {
        for k in [2u64, 3, 5, 7, 0xFF, 0xABCD, 1_000_003] {
            let ops = ops_for(k);
            assert!(!ops.is_empty(), "k={k} should have steps");
            let trace = generate_ecdas_trace(&ops);

            for row in 0..trace.num_rows() {
                let view = row_view(&trace, row);
                assert_eq!(
                    IsBitConstraint::unconditional(cols::MU, 0).evaluate(&view),
                    FE::zero(),
                    "is_bit(mu) k={k} row {row}"
                );
                assert_eq!(
                    IsBitConstraint::unconditional(cols::NEXT_OP, 0).evaluate(&view),
                    FE::zero()
                );
                assert_eq!(
                    IsBitConstraint::unconditional(cols::OP, 0).evaluate(&view),
                    FE::zero(),
                    "is_bit(op) k={k} row {row}"
                );
                assert_eq!(
                    MulZero {
                        a: cols::OP,
                        b: cols::NEXT_OP,
                        b_complement: false,
                        constraint_idx: 0
                    }
                    .evaluate(&view),
                    FE::zero(),
                    "op·next_op k={k} row {row}"
                );
                assert_eq!(
                    MulZero {
                        a: cols::NEXT_OP,
                        b: cols::MU,
                        b_complement: true,
                        constraint_idx: 0
                    }
                    .evaluate(&view),
                    FE::zero()
                );
                for relation in [Relation::Lambda, Relation::Xr, Relation::Yr] {
                    for i in 0..64 {
                        let v = ConvCarry {
                            relation,
                            i,
                            constraint_idx: 0,
                        }
                        .evaluate(&view);
                        assert_eq!(v, FE::zero(), "conv k={k} i={i} row {row}");
                    }
                }
                for c_base in [cols::C0, cols::C1, cols::C2] {
                    assert_eq!(
                        ColIsZero {
                            col: c_base + 63,
                            constraint_idx: 0
                        }
                        .evaluate(&view),
                        FE::zero()
                    );
                }
            }
        }
    }

    #[test]
    fn create_constraints_count() {
        let (constraints, next) = create_constraints(0);
        assert_eq!(constraints.len(), 200);
        assert_eq!(next, 200);
    }
}
