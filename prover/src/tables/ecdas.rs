//! ECDAS chip — one double/add step of the scalar-multiplication sequence.
//!
//! Each row receives an accumulator `(A, G, round, op)` on the self-referential `Ecdas`
//! bus, computes `R = 2A` (op=0) or `R = A + G` (op=1) via three byte-limb convolution
//! relations (`λ`, `xR`, `yR`, each with a 33-byte quotient + 64-entry carry array and the
//! offset `r = 3p`), and sends the updated accumulator back with `round − (1 − next_op)`
//! and `next_op`. When `next_op = 1` it consumes the scalar bit at `round` on the `Bit`
//! bus (an add follows). ECSM seeds and drains the bus; interior rows telescope.
//!
//! See `spec/src/ecdas.toml`. Constraints are **unconditional**; padding rows set quotients
//! to 0 and `op = 1`. The `R·P` term in the λ, xR, and yR relations is gated with `μ`, so it
//! vanishes on padding rows (μ=0) and all relations hold with zero carries.

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
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
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
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

    // Padding rows: q0 = q1 = q2 = 0, op = 1 (add), everything else 0. The μ-gated R·P
    // term vanishes (μ=0), so all convolution relations hold with zero carries.
    for row_idx in n..num_rows {
        table.set_byte(row_idx, cols::OP, 1);
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

    // Receive [id, ts, xA, yA, xG, yG, round, op].
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

    // ARE_BYTES range checks, paired: `ARE_BYTES[X, Y]` checks BOTH elements
    // (bitwise.rs `generate_bitwise_row` enumerates every (x, y) byte pair), so
    // adjacent bytes share one send — 196 single-byte `[b, 0]` sends become 98.
    // Layout: the 32-byte prefixes of LAMBDA, XR, YR, Q0, Q1, Q2 pair internally
    // as (2i, 2i+1); the four odd bytes pair as (ROUND, Q0[32]) and
    // (Q1[32], Q2[32]). `collect_bitwise_from_ecdas` mirrors this layout exactly
    // (sends and BITWISE multiplicities must move together).
    let pair = |col_x: usize, col_y: usize, out: &mut Vec<BusInteraction>| {
        out.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU),
            vec![packed(col_x), packed(col_y)],
        ));
    };
    for base in [
        cols::LAMBDA,
        cols::XR,
        cols::YR,
        cols::Q0,
        cols::Q1,
        cols::Q2,
    ] {
        for i in 0..16 {
            pair(base + 2 * i, base + 2 * i + 1, &mut out);
        }
    }
    pair(cols::ROUND, cols::Q0 + 32, &mut out);
    pair(cols::Q1 + 32, cols::Q2 + 32, &mut out);

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

    // Send the updated accumulator: [id, ts, xR, yR, xG, yG, round - 1 + next_op, next_op].
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

/// Which convolution relation an ECDAS carry constraint enforces.
#[derive(Clone, Copy)]
pub enum Relation {
    Lambda,
    Xr,
    Yr,
}

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..200:
//   0,1,2 : IS_BIT(MU), IS_BIT(OP), IS_BIT(NEXT_OP)
//   3     : OP · NEXT_OP
//   4     : NEXT_OP · (1 − MU)
//   then for (Lambda,C0),(Xr,C1),(Yr,C2): 64 ConvCarry (i=0..64) + 1 ColIsZero.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// ECDAS transition constraints as a single-source [`ConstraintSet`] (200
/// total). No column configuration needed (the layout is fixed via `cols`).
pub struct EcdasConstraints;

impl EcdasConstraints {
    /// Byte `m` of the base-point order `P` (zero beyond 32 bytes).
    fn p_byte_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        m: usize,
    ) -> B::Expr {
        if m < 32 {
            b.const_base(P_BYTES[m] as u64)
        } else {
            b.zero()
        }
    }

    /// Byte `m` of `R` (zero beyond 33 bytes).
    fn r_byte_expr<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        m: usize,
    ) -> B::Expr {
        if m < 33 {
            b.const_base(R_BYTES[m] as u64)
        } else {
            b.zero()
        }
    }

    /// `bytes[base + j]` for `j < len`, else zero (the `b` closure in `s_i`).
    fn byte_at<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        base: usize,
        len: usize,
        j: usize,
    ) -> B::Expr {
        if j < len {
            b.main(0, base + j)
        } else {
            b.zero()
        }
    }

    /// The `μ·R·P − q·P` convolution term
    /// `μ·Σ_{j=0..=i} r_byte(j)·p_byte(i−j) − Σ_{j=0..=i} q[j]·p_byte(i−j)`
    /// (shared structure across all three relations). R and P are constants, so
    /// `μ·R·P` is degree 1. The `μ`-gate makes the term vanish on padding rows
    /// (`μ=0`, `q=0`), keeping all relations at zero carries.
    fn rq<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
        qbase: usize,
    ) -> B::Expr {
        let mu = b.main(0, cols::MU);
        let mut r_p = b.zero();
        let mut q_p = b.zero();
        for j in 0..=i {
            r_p = r_p + Self::r_byte_expr(b, j) * Self::p_byte_expr(b, i - j);
            q_p = q_p + Self::byte_at(b, qbase, 33, j) * Self::p_byte_expr(b, i - j);
        }
        mu * r_p - q_p
    }

    /// `S_i` for `relation` at limb `i`.
    fn s_i<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let lam = |j: usize| Self::byte_at(b, cols::LAMBDA, 32, j);
        let xg = |j: usize| Self::byte_at(b, cols::XG, 32, j);
        let xa = |j: usize| Self::byte_at(b, cols::XA, 32, j);
        let ya = |j: usize| Self::byte_at(b, cols::YA, 32, j);
        let yg = |j: usize| Self::byte_at(b, cols::YG, 32, j);
        let xr = |j: usize| Self::byte_at(b, cols::XR, 32, j);
        let yr = |j: usize| Self::byte_at(b, cols::YR, 32, j);
        let op = b.main(0, cols::OP);
        let one = b.one();

        match relation {
            Relation::Lambda => {
                // op·(Σ λ_j(xG-xA)_{i-j} + (yA_i - yG_i))
                let mut op_branch = ya(i) - yg(i);
                for j in 0..=i {
                    op_branch = op_branch + lam(j) * (xg(i - j) - xa(i - j));
                }
                // (1-op)·Σ (2 λ_j yA_{i-j} - 3 xA_j xA_{i-j})
                let mut notop_branch = b.zero();
                for j in 0..=i {
                    let two = b.const_base(2);
                    let three = b.const_base(3);
                    notop_branch =
                        notop_branch + two * lam(j) * ya(i - j) - three * xa(j) * xa(i - j);
                }
                op.clone() * op_branch + (one - op) * notop_branch + Self::rq(b, i, cols::Q0)
            }
            Relation::Xr => {
                // Σ λ_j λ_{i-j} − xA_i − xG_i − xR_i − (1-op)(xA_i − xG_i) + rq
                let mut s = b.zero();
                for j in 0..=i {
                    s = s + lam(j) * lam(i - j);
                }
                s - xa(i) - xg(i) - xr(i) - (one - op) * (xa(i) - xg(i)) + Self::rq(b, i, cols::Q1)
            }
            Relation::Yr => {
                // Σ λ_j(xA-xR)_{i-j} − yA_i − yR_i + rq
                let mut s = b.zero();
                for j in 0..=i {
                    s = s + lam(j) * (xa(i - j) - xr(i - j));
                }
                s - ya(i) - yr(i) + Self::rq(b, i, cols::Q2)
            }
        }
    }

    /// `256·c_i − c_{i-1} − S_i`.
    fn conv_carry<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let c_base = match relation {
            Relation::Lambda => cols::C0,
            Relation::Xr => cols::C1,
            Relation::Yr => cols::C2,
        };
        let c_i = b.main(0, c_base + i);
        let c_prev = if i == 0 {
            b.zero()
        } else {
            b.main(0, c_base + i - 1)
        };
        let two_pow_8 = b.const_base(256);
        two_pow_8 * c_i - c_prev - Self::s_i(b, relation, i)
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EcdasConstraints {
    // The Lambda ConvCarry has the op·(λ·Δx) term, making it degree 3.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0,1,2: unconditional IS_BIT `x·(1−x)` on [MU, OP, NEXT_OP].
        for (i, col) in [cols::MU, cols::OP, cols::NEXT_OP].into_iter().enumerate() {
            let x = b.main(0, col);
            let one = b.one();
            b.emit_base(i, x.clone() * (one - x));
        }

        // idx 3: OP · NEXT_OP = 0.
        let op = b.main(0, cols::OP);
        let next_op = b.main(0, cols::NEXT_OP);
        b.emit_base(3, op * next_op);

        // idx 4: NEXT_OP · (1 − MU) = 0.
        let next_op = b.main(0, cols::NEXT_OP);
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(4, next_op * (one - mu));

        // Per relation: 64 ConvCarry (i=0..64) + 1 ColIsZero(c_63).
        let mut idx = 5;
        for (relation, c_base) in [
            (Relation::Lambda, cols::C0),
            (Relation::Xr, cols::C1),
            (Relation::Yr, cols::C2),
        ] {
            for i in 0..64 {
                let root = Self::conv_carry(b, relation, i);
                b.emit_base(idx, root);
                idx += 1;
            }
            let c_last = b.main(0, c_base + 63);
            b.emit_base(idx, c_last); // ColIsZero c_63
            idx += 1;
        }
    }
}
