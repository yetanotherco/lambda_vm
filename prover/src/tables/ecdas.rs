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

use std::borrow::Cow;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::gpack::{TraceForm, WidthHint, generate_main};
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

/// The `op` byte of the one row a walk lists in place of a deferred ECREC
/// call's ECDAS rows ([`EcdasOperation::deferred_ecrec`]). No real row has it:
/// a step doubles (0) or adds (1).
const DEFERRED_ECREC_OP: u8 = u8::MAX;

impl EcdasOperation {
    /// The row that stands, in the walk's ECDAS list, for the rows of the ECREC
    /// call at `timestamp` whose witness is computed later: the finish puts
    /// the call's rows in its place before any ECDAS table is built
    /// (`trace_builder::resolve_deferred_ecrec`).
    pub(crate) fn deferred_ecrec(timestamp: u64) -> Self {
        Self {
            timestamp,
            step: EcdasStep {
                x_a: [0; 32],
                y_a: [0; 32],
                x_g: [0; 32],
                y_g: [0; 32],
                round: 0,
                op: DEFERRED_ECREC_OP,
                next_op: 0,
                lambda: [0; 32],
                x_r: [0; 32],
                y_r: [0; 32],
                q0: [0; 33],
                q1: [0; 33],
                q2: [0; 33],
                c0: [0; 64],
                c1: [0; 64],
                c2: [0; 64],
            },
        }
    }
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
    generate_ecdas_trace_as(ops, TraceForm::Wide)
}

/// The widths ECDAS traces needed so far in this process (`tables::gpack`).
pub(crate) static WIDTHS: WidthHint = WidthHint::new();

/// [`generate_ecdas_trace`] in `form` (`tables::gpack`).
pub fn generate_ecdas_trace_as(
    ops: &[EcdasOperation],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_ecdas_rows_as(ops, ops.len().next_power_of_two().max(4), form)
}

/// Whether [`generate_ecdas_rows_as`] in `form` writes its rows packed, with no
/// 64-bit table on the way: in [`TraceForm::Narrow`] once an earlier ECDAS
/// trace of this process left its widths (`tables::gpack`).
pub(crate) fn rows_written_packed(form: TraceForm) -> bool {
    matches!(
        super::gpack::Plan::new(form, &WIDTHS, cols::NUM_COLUMNS),
        super::gpack::Plan::Write(_)
    )
}

/// A ECDAS table of `num_rows` rows in `form`: `ops` (at most `num_rows`, a row
/// each, one double/add step a row), then the padding rows. A row reads its own op alone and
/// the padding rows are constants, so rows `[k·R, (k+1)·R)` of the whole padded
/// table are this over the ops in that range with `num_rows = R`: the block's
/// cut of the whole table, built on its own.
pub fn generate_ecdas_rows_as(
    ops: &[EcdasOperation],
    num_rows: usize,
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_ecdas_rows_of(ops, num_rows, form, borrowed)
}

/// An op read as itself.
fn borrowed(op: &EcdasOperation) -> Cow<'_, EcdasOperation> {
    Cow::Borrowed(op)
}

/// [`generate_ecdas_rows_as`] over rows held in another form, each read as
/// its op by `op` (a [`CompactEcdasOp`] widened back) as its row is filled.
pub(crate) fn generate_ecdas_rows_of<R: Sync>(
    rows: &[R],
    num_rows: usize,
    form: TraceForm,
    op: impl for<'r> Fn(&'r R) -> Cow<'r, EcdasOperation> + Sync,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = rows.len();
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row_idx, row) in rows.iter().enumerate() {
            let op = op(row);
            let s = &op.step;
            debug_assert_ne!(
                s.op, DEFERRED_ECREC_OP,
                "a deferred ECREC call's row reached the ECDAS table"
            );

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
    })
}

/// The carry offsets of `c0`, `c1` and `c2`, in that order.
const CARRY_OFFSETS: [i64; 3] = [CARRY_OFFSET_LAMBDA, CARRY_OFFSET_XR, CARRY_OFFSET_YR];

/// An ECDAS row as a builder keeps it until its table is built. Its carries —
/// `c0`, `c1`, `c2`, three `[i64; 64]`, 1.5 KB of a 1.87 KB row — are held as
/// their offset 16-bit values, the range the trace checks them in: ≈ 0.72 KB a
/// row. A row any of whose carries does not fit keeps all three as they are,
/// so a row widens back to exactly the op it was made from whatever the input
/// ([`Self::to_op`]); the release build has no range assert to rely on.
#[derive(Clone, Debug)]
pub(crate) struct CompactEcdasOp {
    timestamp: u64,
    x_a: [u8; 32],
    y_a: [u8; 32],
    x_g: [u8; 32],
    y_g: [u8; 32],
    round: u8,
    op: u8,
    next_op: u8,
    lambda: [u8; 32],
    x_r: [u8; 32],
    y_r: [u8; 32],
    q0: [u8; 33],
    q1: [u8; 33],
    q2: [u8; 33],
    carries: Carries,
}

/// A row's carries, `c0`, `c1`, `c2` in that order. The common row is held
/// inline; boxing it would cost an allocation a row, the rare one is boxed.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
enum Carries {
    /// Each carry plus its array's offset ([`CARRY_OFFSETS`]), every one in
    /// `0..2^16`.
    Offset([[u16; 64]; 3]),
    /// As they are: some carry's offset value is outside `0..2^16`.
    Raw(Box<[[i64; 64]; 3]>),
}

impl CompactEcdasOp {
    /// `op`, its carries offset when they all fit 16 bits.
    pub(crate) fn from_op(op: &EcdasOperation) -> Self {
        let s = &op.step;
        let raw = [s.c0, s.c1, s.c2];
        let mut offset = [[0u16; 64]; 3];
        let mut fits = true;
        for ((carries, packed), base) in raw.iter().zip(&mut offset).zip(CARRY_OFFSETS) {
            for (&c, slot) in carries.iter().zip(packed.iter_mut()) {
                match c.checked_add(base).and_then(|v| u16::try_from(v).ok()) {
                    Some(v) => *slot = v,
                    None => fits = false,
                }
            }
        }
        Self {
            timestamp: op.timestamp,
            x_a: s.x_a,
            y_a: s.y_a,
            x_g: s.x_g,
            y_g: s.y_g,
            round: s.round,
            op: s.op,
            next_op: s.next_op,
            lambda: s.lambda,
            x_r: s.x_r,
            y_r: s.y_r,
            q0: s.q0,
            q1: s.q1,
            q2: s.q2,
            carries: if fits {
                Carries::Offset(offset)
            } else {
                Carries::Raw(Box::new(raw))
            },
        }
    }

    /// The op this row was made from, exactly.
    pub(crate) fn to_op(&self) -> EcdasOperation {
        let [c0, c1, c2] = match &self.carries {
            Carries::Offset(offset) => {
                let mut raw = [[0i64; 64]; 3];
                for ((carries, packed), base) in raw.iter_mut().zip(offset).zip(CARRY_OFFSETS) {
                    for (c, &v) in carries.iter_mut().zip(packed) {
                        *c = i64::from(v) - base;
                    }
                }
                raw
            }
            Carries::Raw(raw) => **raw,
        };
        EcdasOperation {
            timestamp: self.timestamp,
            step: EcdasStep {
                x_a: self.x_a,
                y_a: self.y_a,
                x_g: self.x_g,
                y_g: self.y_g,
                round: self.round,
                op: self.op,
                next_op: self.next_op,
                lambda: self.lambda,
                x_r: self.x_r,
                y_r: self.y_r,
                q0: self.q0,
                q1: self.q1,
                q2: self.q2,
                c0,
                c1,
                c2,
            },
        }
    }

    /// The call's timestamp when the row stands for a deferred ECREC call's
    /// rows ([`EcdasOperation::deferred_ecrec`]), `None` for a real row.
    pub(crate) fn deferred_ecrec(&self) -> Option<u64> {
        (self.op == DEFERRED_ECREC_OP).then_some(self.timestamp)
    }

    /// Whether the row keeps its carries as they are.
    #[cfg(test)]
    pub(crate) fn is_raw(&self) -> bool {
        matches!(self.carries, Carries::Raw(_))
    }
}

/// A [`CompactEcdasOp`] read as its op.
pub(crate) fn widen(row: &CompactEcdasOp) -> Cow<'_, EcdasOperation> {
    Cow::Owned(row.to_op())
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
#[derive(Clone, Copy)]
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

// =========================================================================
// The limb form: the same three relations over 16-bit limb positions
// =========================================================================
//
// [`EcdasForm::Halves`] proves the integer identities of the byte form with
// the carry chain at radix 2^16. A 16-bit limb is the linear form
// `L_k = b_{2k} + 256·b_{2k+1}` of two byte columns (no limb column is
// committed); every byte column of the byte form stays, range-checked two to
// a lookup. Per relation and limb position `k = 0..32`:
//
//   2^16·C_k − C_{k−1} − S'_k = 0,   C_{−1} = 0,   C_31 = 0 (no column),
//
// where `S'_k` is the byte form's `S` written over limbs, and the carry
// `C_k = LO_k + 2^16·HI_k − 2^23` (k < 31) is held as a 16-bit `LO_k`
// (`IS_HALF`) and an 8-bit `HI_k` (`ARE_BYTES`, two to a lookup).
//
// Why it is sound: every limb is below 2^16 (its bytes are checked), so every
// `|S'_k| < 2^39`; every carry is in `[−2^23, 2^23)` by its checks; so each
// constraint's terms are below 2^40 in size, far below the field's half, and
// each congruence is an equality over the integers. The equalities telescope:
// `Σ_k 2^{16k}·S'_k = 2^512·C_31 − C_{−1} = 0`, and that sum is the byte
// form's `Σ_i 2^{8i}·S_i` (the same products, regrouped), so each relation
// holds over the integers exactly as in the byte form. The honest carries fit
// the offset with a bit to spare (`the_carry_bound_is_computed`, in the tests).

/// Which relations an ECDAS table proves its steps with: each a format
/// constant the prover and the verifier of one block share, never read from a
/// proof.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum EcdasForm {
    /// The byte-position relations: 64 carries a relation, each checked as a
    /// 16-bit value, and one byte check a lookup (521 columns, 200
    /// constraints, 388 interactions). Today's table.
    #[default]
    Bytes,
    /// The same relations at 16-bit limb positions: 31 carries a relation,
    /// each a 16-bit and an 8-bit part, and two byte checks a lookup (515
    /// columns, 101 constraints, 242 interactions).
    Halves,
}

impl EcdasForm {
    /// The table's columns.
    pub const fn num_columns(self) -> usize {
        match self {
            Self::Bytes => cols::NUM_COLUMNS,
            Self::Halves => halves_cols::NUM_COLUMNS,
        }
    }

    /// The multiplicity column (1 on a real row, 0 on a padding row).
    pub const fn mu(self) -> usize {
        match self {
            Self::Bytes => cols::MU,
            Self::Halves => halves_cols::MU,
        }
    }

    /// The table's constraints.
    pub const fn num_constraints(self) -> usize {
        match self {
            Self::Bytes => 200,
            Self::Halves => 5 + 3 * HALVES_POSITIONS,
        }
    }
}

/// Limb positions a relation's identity spans (`q·P` reaches position 31).
pub const HALVES_POSITIONS: usize = 32;
/// Carries a relation holds under [`EcdasForm::Halves`] (`C_31` is zero).
pub const HALVES_CARRIES: usize = HALVES_POSITIONS - 1;
/// The offset that makes every carry a 24-bit number: `C + 2^23` is
/// `LO + 2^16·HI` with `LO < 2^16`, `HI < 2^8`.
pub const HALVES_CARRY_OFFSET: i64 = 1 << 23;

/// Column indices under [`EcdasForm::Halves`] (515 columns). Every column
/// before `Q0`'s end is the byte form's; the carry arrays are each relation's
/// `LO` (31) then `HI` (31).
pub mod halves_cols {
    pub use super::cols::{
        LAMBDA, OP, Q0, ROUND, TIMESTAMP_0, TIMESTAMP_1, XA, XG, XR, YA, YG, YR,
    };

    pub const LO0: usize = 261; // [16-bit; 31]
    pub const HI0: usize = 292; // [byte; 31]
    pub const Q1: usize = 323; // Byte[33]
    pub const LO1: usize = 356;
    pub const HI1: usize = 387;
    pub const Q2: usize = 418; // Byte[33]
    pub const LO2: usize = 451;
    pub const HI2: usize = 482;
    pub const NEXT_OP: usize = 513; // Bit
    pub const MU: usize = 514;

    pub const NUM_COLUMNS: usize = 515;

    /// `(LO, HI, Q)` of each relation, λ's, xR's and yR's.
    pub const RELATIONS: [(usize, usize, usize); 3] =
        [(LO0, HI0, Q0), (LO1, HI1, Q1), (LO2, HI2, Q2)];
}

const _: () = {
    use halves_cols::*;
    assert!(Q0 + 33 == LO0 && LO0 + 31 == HI0 && HI0 + 31 == Q1);
    assert!(Q1 + 33 == LO1 && LO1 + 31 == HI1 && HI1 + 31 == Q2);
    assert!(Q2 + 33 == LO2 && LO2 + 31 == HI2 && HI2 + 31 == NEXT_OP);
    assert!(NEXT_OP + 1 == MU && MU + 1 == NUM_COLUMNS);
};

/// `bytes` (little-endian) as `N` 16-bit limbs, zero past them.
const fn limbs_of<const N: usize>(bytes: &[u8]) -> [u64; N] {
    let mut out = [0u64; N];
    let mut i = 0;
    while i < bytes.len() {
        out[i / 2] += (bytes[i] as u64) << (8 * (i % 2));
        i += 1;
    }
    out
}

/// `P`'s 16 limbs.
const P_LIMBS: [u64; 16] = limbs_of::<16>(&P_BYTES);
/// `R = 3P`'s 17 limbs (the 17th is its 33rd byte).
const R_LIMBS: [u64; 17] = limbs_of::<17>(&R_BYTES);
/// `Σ_j R_j·P_{k−j}` at each limb position: the `μ·R·P` term's constant.
const RP_LIMBS: [u64; HALVES_POSITIONS] = {
    let mut out = [0u64; HALVES_POSITIONS];
    let mut j = 0;
    while j < 17 {
        let mut m = 0;
        while m < 16 {
            out[j + m] += R_LIMBS[j] * P_LIMBS[m];
            m += 1;
        }
        j += 1;
    }
    out
};

/// A step's operands as integer limbs: what the limb form's relations read.
/// Each entry is an integer (a test may put one out of its range).
#[derive(Clone, Debug)]
pub(crate) struct HalvesLimbs {
    pub(crate) xg: [i64; 16],
    pub(crate) yg: [i64; 16],
    pub(crate) xa: [i64; 16],
    pub(crate) ya: [i64; 16],
    pub(crate) xr: [i64; 16],
    pub(crate) yr: [i64; 16],
    pub(crate) lambda: [i64; 16],
    /// `q0`, `q1`, `q2`, 17 limbs each.
    pub(crate) q: [[i64; 17]; 3],
    pub(crate) op: i64,
    pub(crate) mu: i64,
}

impl HalvesLimbs {
    /// A real row's limbs (`μ = 1`).
    pub(crate) fn of(s: &EcdasStep) -> Self {
        let l16 = |b: &[u8; 32]| limbs_of::<16>(b).map(|v| v as i64);
        let l17 = |b: &[u8; 33]| limbs_of::<17>(b).map(|v| v as i64);
        Self {
            xg: l16(&s.x_g),
            yg: l16(&s.y_g),
            xa: l16(&s.x_a),
            ya: l16(&s.y_a),
            xr: l16(&s.x_r),
            yr: l16(&s.y_r),
            lambda: l16(&s.lambda),
            q: [l17(&s.q0), l17(&s.q1), l17(&s.q2)],
            op: i64::from(s.op),
            mu: 1,
        }
    }

    /// `S'_k` of each relation (λ, xR, yR) at each limb position, as the
    /// constraints compute it, over the integers.
    pub(crate) fn terms(&self) -> [[i64; HALVES_POSITIONS]; 3] {
        let conv = |a: &[i64; 16], b: &[i64; 16], k: usize| -> i64 {
            (k.saturating_sub(15)..=k.min(15))
                .map(|j| a[j] * b[k - j])
                .sum()
        };
        let at = |a: &[i64; 16], k: usize| if k < 16 { a[k] } else { 0 };
        let rq = |q: &[i64; 17], k: usize| -> i64 {
            let qp: i64 = (k.saturating_sub(15)..=k.min(16))
                .map(|j| q[j] * P_LIMBS[k - j] as i64)
                .sum();
            self.mu * RP_LIMBS[k] as i64 - qp
        };
        let dx: [i64; 16] = std::array::from_fn(|i| self.xg[i] - self.xa[i]);
        let ax: [i64; 16] = std::array::from_fn(|i| self.xa[i] - self.xr[i]);
        let mut out = [[0i64; HALVES_POSITIONS]; 3];
        #[allow(clippy::needless_range_loop)]
        for k in 0..HALVES_POSITIONS {
            let add = conv(&self.lambda, &dx, k) + at(&self.ya, k) - at(&self.yg, k);
            let dbl = 2 * conv(&self.lambda, &self.ya, k) - 3 * conv(&self.xa, &self.xa, k);
            out[0][k] = self.op * add + (1 - self.op) * dbl + rq(&self.q[0], k);
            out[1][k] = conv(&self.lambda, &self.lambda, k)
                - at(&self.xa, k)
                - at(&self.xg, k)
                - at(&self.xr, k)
                - (1 - self.op) * (at(&self.xa, k) - at(&self.xg, k))
                + rq(&self.q[1], k);
            out[2][k] =
                conv(&self.lambda, &ax, k) - at(&self.ya, k) - at(&self.yr, k) + rq(&self.q[2], k);
        }
        out
    }
}

/// The carry chain of one relation's terms: `C_k = (C_{k−1} + S'_k) / 2^16`
/// (floored) for `k < 31`, and whether the chain is exact — every division
/// exact and `C_30 + S'_31 = 0`, which an honest step's terms always are.
pub(crate) fn halves_chain(terms: &[i64; HALVES_POSITIONS]) -> ([i64; HALVES_CARRIES], bool) {
    let mut carries = [0i64; HALVES_CARRIES];
    let mut prev = 0i64;
    let mut exact = true;
    for (k, carry) in carries.iter_mut().enumerate() {
        let s = prev + terms[k];
        exact &= s.rem_euclid(1 << 16) == 0;
        prev = s.div_euclid(1 << 16);
        *carry = prev;
    }
    exact &= prev + terms[HALVES_POSITIONS - 1] == 0;
    (carries, exact)
}

/// A step's carries under [`EcdasForm::Halves`], λ's, xR's and yR's, from its
/// bytes (the byte form's carries are not read).
pub fn halves_carries(s: &EcdasStep) -> [[i64; HALVES_CARRIES]; 3] {
    let terms = HalvesLimbs::of(s).terms();
    terms.map(|t| {
        let (carries, exact) = halves_chain(&t);
        debug_assert!(exact, "an ECDAS step whose relations do not hold");
        carries
    })
}

/// A carry's `(LO, HI)`: `C + 2^23 = LO + 2^16·HI`, `LO` in `0..2^16`. A carry
/// outside `[−2^23, 2^23)` (no honest step has one) gives an `HI` outside
/// `0..2^8`, which its check refuses.
pub(crate) fn split_carry(c: i64) -> (u64, i64) {
    let v = c + HALVES_CARRY_OFFSET;
    let lo = v.rem_euclid(1 << 16);
    (lo as u64, (v - lo) >> 16)
}

/// The widths ECDAS traces under [`EcdasForm::Halves`] needed so far in this
/// process (`tables::gpack`).
pub(crate) static WIDTHS_HALVES: WidthHint = WidthHint::new();

/// [`rows_written_packed`] for `ecdas`'s table.
pub(crate) fn rows_written_packed_in(form: TraceForm, ecdas: EcdasForm) -> bool {
    let (hint, columns) = match ecdas {
        EcdasForm::Bytes => (&WIDTHS, cols::NUM_COLUMNS),
        EcdasForm::Halves => (&WIDTHS_HALVES, halves_cols::NUM_COLUMNS),
    };
    matches!(
        super::gpack::Plan::new(form, hint, columns),
        super::gpack::Plan::Write(_)
    )
}

/// [`rows_written_packed_in`] under [`EcdasForm::Halves`], as a function.
pub(crate) fn halves_rows_written_packed(form: TraceForm) -> bool {
    rows_written_packed_in(form, EcdasForm::Halves)
}

/// [`generate_ecdas_rows_of`] in `ecdas`'s form.
pub(crate) fn generate_ecdas_rows_in<R: Sync>(
    rows: &[R],
    num_rows: usize,
    form: TraceForm,
    ecdas: EcdasForm,
    op: impl for<'r> Fn(&'r R) -> Cow<'r, EcdasOperation> + Sync,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    match ecdas {
        EcdasForm::Bytes => generate_ecdas_rows_of(rows, num_rows, form, op),
        EcdasForm::Halves => generate_halves_rows_of(rows, num_rows, form, op),
    }
}

/// An ECDAS table of `num_rows` rows under [`EcdasForm::Halves`]: `rows` (a
/// step a row), then the padding rows, whose carries are zero (`LO = 0`,
/// `HI = 2^23 / 2^16`).
fn generate_halves_rows_of<R: Sync>(
    rows: &[R],
    num_rows: usize,
    form: TraceForm,
    op: impl for<'r> Fn(&'r R) -> Cow<'r, EcdasOperation> + Sync,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use halves_cols as h;
    let n = rows.len();
    let (_, hi_zero) = split_carry(0);
    generate_main!(form, &WIDTHS_HALVES, num_rows, h::NUM_COLUMNS, |table| {
        for (row_idx, row) in rows.iter().enumerate() {
            let op = op(row);
            let s = &op.step;
            debug_assert_ne!(
                s.op, DEFERRED_ECREC_OP,
                "a deferred ECREC call's row reached the ECDAS table"
            );
            table.set_dword_wl(row_idx, h::TIMESTAMP_0, op.timestamp);
            table.set_bytes(row_idx, h::XG, &s.x_g);
            table.set_bytes(row_idx, h::YG, &s.y_g);
            table.set_bytes(row_idx, h::XA, &s.x_a);
            table.set_bytes(row_idx, h::YA, &s.y_a);
            table.set_byte(row_idx, h::ROUND, s.round);
            table.set_byte(row_idx, h::OP, s.op);
            table.set_bytes(row_idx, h::XR, &s.x_r);
            table.set_bytes(row_idx, h::YR, &s.y_r);
            table.set_bytes(row_idx, h::LAMBDA, &s.lambda);
            table.set_bytes(row_idx, h::Q0, &s.q0);
            table.set_bytes(row_idx, h::Q1, &s.q1);
            table.set_bytes(row_idx, h::Q2, &s.q2);
            for (carries, &(lo, hi, _)) in halves_carries(s).iter().zip(&h::RELATIONS) {
                for (k, &c) in carries.iter().enumerate() {
                    let (l, high) = split_carry(c);
                    debug_assert!((0..1 << 8).contains(&high), "carry {c} out of range");
                    table.set_u64(row_idx, lo + k, l);
                    table.set_fe(row_idx, hi + k, fe_from_i64(high));
                }
            }
            table.set_byte(row_idx, h::NEXT_OP, s.next_op);
            table.set_fe(row_idx, h::MU, FE::one());
        }
        // Padding rows: q = 0, op = 1, μ = 0, every carry zero.
        for row_idx in n..num_rows {
            table.set_byte(row_idx, h::OP, 1);
            for &(_, hi, _) in &h::RELATIONS {
                for k in 0..HALVES_CARRIES {
                    table.set_fe(row_idx, hi + k, fe_from_i64(hi_zero));
                }
            }
        }
    })
}

/// The bytes the limb form checks, in the order they are paired: `ROUND`, λ,
/// `q0`, xR, `q1`, yR, `q2` (196, the byte form's single checks).
pub(crate) fn halves_checked_bytes() -> Vec<usize> {
    use halves_cols as h;
    let mut out = vec![h::ROUND];
    for (start, len) in [
        (h::LAMBDA, 32),
        (h::Q0, 33),
        (h::XR, 32),
        (h::Q1, 33),
        (h::YR, 32),
        (h::Q2, 33),
    ] {
        out.extend(start..start + len);
    }
    out
}

/// ECDAS's interactions in `form`.
pub fn bus_interactions_for(form: EcdasForm) -> Vec<BusInteraction> {
    match form {
        EcdasForm::Bytes => bus_interactions(),
        EcdasForm::Halves => halves_bus_interactions(),
    }
}

/// The limb form's 242 interactions: the Ecdas bus receive, 98 paired byte
/// checks, per relation 31 `IS_HALF` on `LO` and 16 paired byte checks on
/// `HI` (the last `HI` with the constant 0), the `Bit` send and the Ecdas bus
/// send.
fn halves_bus_interactions() -> Vec<BusInteraction> {
    use halves_cols as h;
    let mu = || Multiplicity::Column(h::MU);
    let ts_lo = || packed(h::TIMESTAMP_0);
    let ts_hi = || packed(h::TIMESTAMP_1);
    let are_bytes =
        |a: BusValue, b: BusValue| BusInteraction::sender(BusId::AreBytes, mu(), vec![a, b]);
    let mut out = Vec::new();

    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        mu(),
        ecdas_tuple(
            h::XA,
            h::YA,
            h::XG,
            h::YG,
            packed(h::ROUND),
            packed(h::OP),
            ts_lo(),
            ts_hi(),
        ),
    ));

    let bytes = halves_checked_bytes();
    debug_assert_eq!(bytes.len() % 2, 0);
    for pair in bytes.chunks(2) {
        out.push(are_bytes(packed(pair[0]), packed(pair[1])));
    }

    for &(lo, hi, _) in &h::RELATIONS {
        for k in 0..HALVES_CARRIES {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                mu(),
                vec![packed(lo + k)],
            ));
        }
        for m in 0..HALVES_CARRIES.div_ceil(2) {
            let second = if 2 * m + 1 < HALVES_CARRIES {
                packed(hi + 2 * m + 1)
            } else {
                BusValue::constant(0)
            };
            out.push(are_bytes(packed(hi + 2 * m), second));
        }
    }

    out.push(BusInteraction::sender(
        BusId::Bit,
        Multiplicity::Column(h::NEXT_OP),
        vec![ts_lo(), ts_hi(), packed(h::ROUND)],
    ));

    out.push(BusInteraction::sender(
        BusId::Ecdas,
        mu(),
        ecdas_tuple(
            h::XR,
            h::YR,
            h::XG,
            h::YG,
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: h::ROUND,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: h::NEXT_OP,
                },
                LinearTerm::Constant(-1),
            ]),
            packed(h::NEXT_OP),
            ts_lo(),
            ts_hi(),
        ),
    ));

    out
}

/// The limb form's constraints ([`EcdasForm::Halves`], 101): today's five
/// flag constraints, then for λ, xR and yR one carry identity a limb
/// position, `2^16·C_k − C_{k−1} − S'_k` for `k = 0..32`.
#[derive(Clone, Copy)]
pub struct EcdasHalvesConstraints;

impl EcdasHalvesConstraints {
    /// The body. The top carry `C_31` is zero (`top_columns` `None`, the
    /// constraints'); a test's mutant reads relation `r`'s from column
    /// `first + r` instead (`Some(first)`), a column of its own.
    pub(crate) fn eval_with<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &mut B,
        top_columns: Option<usize>,
    ) {
        use halves_cols as h;
        for (i, col) in [h::MU, h::OP, h::NEXT_OP].into_iter().enumerate() {
            let x = b.main(0, col);
            let one = b.one();
            b.emit_base(i, x.clone() * (one - x));
        }
        let op = b.main(0, h::OP);
        let next_op = b.main(0, h::NEXT_OP);
        b.emit_base(3, op * next_op);
        let next_op = b.main(0, h::NEXT_OP);
        let mu = b.main(0, h::MU);
        let one = b.one();
        b.emit_base(4, next_op * (one - mu));

        // The limbs, each built once.
        let limbs = |b: &B, start: usize, n_bytes: usize| -> Vec<B::Expr> {
            (0..n_bytes.div_ceil(2))
                .map(|k| {
                    let low = b.main(0, start + 2 * k);
                    if 2 * k + 1 < n_bytes {
                        low + b.const_base(256) * b.main(0, start + 2 * k + 1)
                    } else {
                        low
                    }
                })
                .collect()
        };
        let lam = limbs(b, h::LAMBDA, 32);
        let xg = limbs(b, h::XG, 32);
        let yg = limbs(b, h::YG, 32);
        let xa = limbs(b, h::XA, 32);
        let ya = limbs(b, h::YA, 32);
        let xr = limbs(b, h::XR, 32);
        let yr = limbs(b, h::YR, 32);
        let qs: Vec<Vec<B::Expr>> = h::RELATIONS
            .iter()
            .map(|&(_, _, q)| limbs(b, q, 33))
            .collect();
        let dx: Vec<B::Expr> = (0..16).map(|i| xg[i].clone() - xa[i].clone()).collect();
        let ax: Vec<B::Expr> = (0..16).map(|i| xa[i].clone() - xr[i].clone()).collect();
        let op = b.main(0, h::OP);
        let mu = b.main(0, h::MU);

        // `Σ_j a_j·b_{k−j}` over 16 limbs each, `None` when no product lands at `k`.
        let conv = |a: &[B::Expr], c: &[B::Expr], k: usize| -> Option<B::Expr> {
            (k.saturating_sub(15)..=k.min(15))
                .map(|j| a[j].clone() * c[k - j].clone())
                .reduce(|s, t| s + t)
        };
        // `Σ_j a_j·a_{k−j}`, each cross product once and doubled.
        let square = |a: &[B::Expr], k: usize| -> Option<B::Expr> {
            let lo = k.saturating_sub(15);
            let cross = (lo..=k.min(15))
                .filter(|&j| j < k - j)
                .map(|j| a[j].clone() * a[k - j].clone())
                .reduce(|s, t| s + t)
                .map(|s| b.const_base(2) * s);
            let middle =
                (k.is_multiple_of(2) && k / 2 < 16).then(|| a[k / 2].clone() * a[k / 2].clone());
            match (cross, middle) {
                (Some(c), Some(m)) => Some(c + m),
                (c, m) => c.or(m),
            }
        };
        let at = |a: &[B::Expr], k: usize| (k < a.len()).then(|| a[k].clone());
        // `μ·Σ R_j P_{k−j} − Σ q_j P_{k−j}`.
        let rq = |q: &[B::Expr], k: usize| -> B::Expr {
            let qp = (k.saturating_sub(15)..=k.min(16))
                .map(|j| b.const_base(P_LIMBS[k - j]) * q[j].clone())
                .reduce(|s, t| s + t)
                .unwrap_or_else(|| b.zero());
            mu.clone() * b.const_base(RP_LIMBS[k]) - qp
        };
        let or_zero = |e: Option<B::Expr>| e.unwrap_or_else(|| b.zero());

        let terms: [Vec<B::Expr>; 3] = [
            // λ: op·(Σ λ_j·Δx_{k−j} + yA_k − yG_k)
            //    + (1−op)·(2·Σ λ_j·yA_{k−j} − 3·Σ xA_j·xA_{k−j}) + μ·RP − q0·P
            (0..HALVES_POSITIONS)
                .map(|k| {
                    let add = match (conv(&lam, &dx, k), at(&ya, k), at(&yg, k)) {
                        (Some(c), Some(y), Some(g)) => Some(c + y - g),
                        (c, _, _) => c,
                    };
                    let dbl = match (conv(&lam, &ya, k), square(&xa, k)) {
                        (Some(l), Some(s)) => Some(b.const_base(2) * l - b.const_base(3) * s),
                        (l, s) => l
                            .map(|l| b.const_base(2) * l)
                            .or(s.map(|s| -(b.const_base(3) * s))),
                    };
                    // op·add + (1−op)·dbl, as dbl + op·(add − dbl).
                    let branch = match (add, dbl) {
                        (Some(a), Some(d)) => Some(d.clone() + op.clone() * (a - d)),
                        (Some(a), None) => Some(op.clone() * a),
                        (None, Some(d)) => Some(d.clone() - op.clone() * d),
                        (None, None) => None,
                    };
                    or_zero(branch) + rq(&qs[0], k)
                })
                .collect(),
            // xR: Σ λ_j·λ_{k−j} − xA_k − xG_k − xR_k − (1−op)·(xA_k − xG_k) + μ·RP − q1·P,
            // the linear part as op·(xA_k − xG_k) − 2·xA_k − xR_k.
            (0..HALVES_POSITIONS)
                .map(|k| {
                    let linear = match (at(&dx, k), at(&xa, k), at(&xr, k)) {
                        (Some(d), Some(a), Some(r)) => {
                            Some(-(op.clone() * d) - b.const_base(2) * a - r)
                        }
                        _ => None,
                    };
                    let s = match (square(&lam, k), linear) {
                        (Some(s), Some(l)) => s + l,
                        (s, l) => or_zero(s.or(l)),
                    };
                    s + rq(&qs[1], k)
                })
                .collect(),
            // yR: Σ λ_j·(xA − xR)_{k−j} − yA_k − yR_k + μ·RP − q2·P
            (0..HALVES_POSITIONS)
                .map(|k| {
                    let s = match (conv(&lam, &ax, k), at(&ya, k), at(&yr, k)) {
                        (Some(c), Some(y), Some(r)) => c - y - r,
                        (c, _, _) => or_zero(c),
                    };
                    s + rq(&qs[2], k)
                })
                .collect(),
        ];

        let offset = b.const_signed(HALVES_CARRY_OFFSET);
        let shift = b.const_base(1 << 16);
        let mut idx = 5;
        for (r, (relation, &(lo, hi, _))) in terms.iter().zip(&h::RELATIONS).enumerate() {
            let carry = |b: &B, k: usize| -> B::Expr {
                b.main(0, lo + k) + shift.clone() * b.main(0, hi + k) - offset.clone()
            };
            let mut prev: Option<B::Expr> = None;
            for (k, s) in relation.iter().enumerate() {
                let c_k = if k < HALVES_CARRIES {
                    Some(carry(b, k))
                } else {
                    top_columns.map(|first| b.main(0, first + r))
                };
                let mut root = match &c_k {
                    Some(c) => shift.clone() * c.clone() - s.clone(),
                    None => -s.clone(),
                };
                if let Some(p) = prev {
                    root = root - p;
                }
                b.emit_base(idx, root);
                idx += 1;
                prev = c_k;
            }
        }
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EcdasHalvesConstraints {
    // λ's identity has the op·(λ·Δx) term, making it degree 3.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        Self::eval_with(b, None);
    }
}

#[cfg(test)]
mod compact_tests {
    use super::*;

    /// A step whose every field differs from its neighbours', with these carries.
    fn op(i: u64, c: [[i64; 64]; 3]) -> EcdasOperation {
        let bytes = |f: u64| -> [u8; 32] {
            std::array::from_fn(|b| ((i + 1) * 31 + f * 7 + b as u64) as u8)
        };
        let wide = |f: u64| -> [u8; 33] {
            std::array::from_fn(|b| ((i + 3) * 17 + f * 5 + b as u64) as u8)
        };
        EcdasOperation {
            timestamp: 4 * i + 3,
            step: EcdasStep {
                x_a: bytes(1),
                y_a: bytes(2),
                x_g: bytes(3),
                y_g: bytes(4),
                round: (i % 256) as u8,
                op: (i % 2) as u8,
                next_op: ((i + 1) % 2) as u8,
                lambda: bytes(5),
                x_r: bytes(6),
                y_r: bytes(7),
                q0: wide(8),
                q1: wide(9),
                q2: wide(10),
                c0: c[0],
                c1: c[1],
                c2: c[2],
            },
        }
    }

    /// Carries at and around each array's 16-bit bounds, and far outside them;
    /// `out` puts one carry of array `k` outside its range.
    fn ops() -> Vec<(EcdasOperation, bool)> {
        let mut ops = Vec::new();
        for (i, base) in CARRY_OFFSETS.iter().enumerate() {
            let low = -base;
            let high = (1 << 16) - 1 - base;
            let fit: [[i64; 64]; 3] = std::array::from_fn(|k| {
                std::array::from_fn(|j| {
                    let b = CARRY_OFFSETS[k];
                    match j % 4 {
                        0 => -b,
                        1 => (1 << 16) - 1 - b,
                        2 => (j as i64 * 37) % 200 - 100,
                        _ => 0,
                    }
                })
            });
            ops.push((op(i as u64, fit), false));
            for (n, bad) in [low - 1, high + 1, i64::MIN, i64::MAX, -(1 << 40)]
                .into_iter()
                .enumerate()
            {
                let mut c = fit;
                c[i][(n * 13) % 64] = bad;
                ops.push((op((10 + 10 * i + n) as u64, c), true));
            }
        }
        ops
    }

    fn same(a: &EcdasOperation, b: &EcdasOperation) -> bool {
        format!("{a:?}") == format!("{b:?}")
    }

    /// Every row widens back to exactly the op it was made from: carries at
    /// both ends of their 16-bit range held offset, and a row with any carry
    /// outside it (by one, or at either end of i64) held as it is.
    #[test]
    fn a_compact_row_widens_to_the_op_it_was_made_from() {
        for (op, out) in ops() {
            let row = CompactEcdasOp::from_op(&op);
            assert_eq!(row.is_raw(), out, "{op:?}");
            assert!(same(&row.to_op(), &op), "{op:?}");
        }
        // The compact row is well under half the op.
        assert!(
            std::mem::size_of::<CompactEcdasOp>() * 5 < std::mem::size_of::<EcdasOperation>() * 2,
            "{} vs {}",
            std::mem::size_of::<CompactEcdasOp>(),
            std::mem::size_of::<EcdasOperation>()
        );
    }

    fn words(t: &TraceTable<GoldilocksField, GoldilocksExtension>) -> Vec<u64> {
        match t.narrow_main() {
            Some(p) => (0..p.cols()).flat_map(|c| p.column(c)).collect(),
            None => {
                let m = &t.main_table;
                (0..m.width)
                    .flat_map(|c| (0..m.height).map(move |r| *m.get(r, c).value()))
                    .collect()
            }
        }
    }

    /// A table built from compact rows, each widened as its row is filled, is
    /// the table of the ops, wide and packed, the fallback rows included.
    #[test]
    fn a_table_of_compact_rows_is_the_table_of_the_ops() {
        // The generator's debug checks reject a carry outside its range, so the
        // fallback rows are built only without them.
        let wide: Vec<EcdasOperation> = ops()
            .into_iter()
            .filter(|(_, out)| !out || !cfg!(debug_assertions))
            .map(|(op, _)| op)
            .collect();
        let compact: Vec<CompactEcdasOp> = wide.iter().map(CompactEcdasOp::from_op).collect();
        for form in [TraceForm::Wide, TraceForm::Narrow, TraceForm::Narrow] {
            for rows in [wide.len().next_power_of_two(), 64] {
                let want = generate_ecdas_rows_as(&wide, rows, form);
                let got = generate_ecdas_rows_of(&compact, rows, form, widen);
                assert_eq!(words(&got), words(&want), "{form:?} {rows}");
            }
        }
    }
}
