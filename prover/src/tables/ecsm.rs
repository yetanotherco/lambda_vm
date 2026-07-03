//! ECSM core chip — orchestrates one secp256k1 scalar multiplication `k·G`.
//!
//! One row per `ECALL(-11)`. It reads `xG` and `k` from memory, witnesses `yG` and proves
//! `yG² ≡ xG³ + b mod p` (via two byte-limb convolution relations with quotients `q0,q1`
//! and 64-entry carry arrays `c0,c1`), enforces `0 < k < N` and `xR < p`, writes `xR` back,
//! triggers EC_SCALAR to serve `k` bit-by-bit, and delegates the double-and-add to ECDAS over
//! the `Ecdas`/`ServeK`/`Bit` buses.
//!
//! See `spec/src/ecsm.toml`. All multi-limb arithmetic uses 8-bit limbs; the witness is built
//! by `ecsm::compute_witness`, which reproduces these exact recurrences.
//!
//! ## Padding
//! Padding rows have `mu = 0`, all columns zero **except `q1`, which pads to `p`**. This makes
//! both carry relations close on padding without gating the whole recurrence: the x² relation
//! has no standalone constant (closes at all-zero), and the yG relation closes because the
//! `p² − q1·p` offset cancels (`q1 = p`) and the curve constant `b` is multiplied by `µ` (so it
//! drops when `µ = 0`). Only that single `µ·b` term is µ-gated. The range checks /
//! virtual-carry checks remain µ-gated as before.

use executor::vm::instruction::execution::ECSM_SYSCALL_NUMBER;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::INV_SHIFT_32;
use ecsm::{B, EcsmWitness, N_BYTES, P_BYTES};

// Bias signed convolution carries into IsHalfword [0, 2^16); see spec ecsm.typ "Carry offset" (@ecsm-limb_carry).
pub(crate) const CARRY_OFFSET_X2: i64 = 8160;
pub(crate) const CARRY_OFFSET_YG: i64 = 16319;

// =========================================================================
// Column indices (~427 columns)
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const ADDR_XG_0: usize = 2;
    pub const ADDR_XG_1: usize = 3;
    pub const ADDR_K_0: usize = 4;
    pub const ADDR_K_1: usize = 5;
    pub const ADDR_XR_0: usize = 6;
    pub const ADDR_XR_1: usize = 7;

    pub const XR: usize = 8; // U256BL (32)
    pub const YR: usize = 40; // U256BL (32)
    pub const K: usize = 72; // U256BL (32)
    pub const LEN_K: usize = 104; // Byte
    pub const XG: usize = 105; // U256BL (32)
    pub const YG: usize = 137; // U256BL (32)
    pub const X2: usize = 169; // U256BL (32)
    pub const Q0: usize = 201; // U256BL (32)
    pub const C0: usize = 233; // BaseField[64]
    pub const Q1: usize = 297; // Byte[33]
    pub const C1: usize = 330; // BaseField[64]
    pub const K_SUB_N: usize = 394; // U256HL (16 halfwords)
    pub const XR_SUB_P: usize = 410; // U256HL (16 halfwords)
    pub const MU: usize = 426;

    pub const NUM_COLUMNS: usize = 427;

    #[inline]
    pub const fn xr(i: usize) -> usize {
        XR + i
    }
    #[inline]
    pub const fn k(i: usize) -> usize {
        K + i
    }
    #[inline]
    pub const fn xg(i: usize) -> usize {
        XG + i
    }
    #[inline]
    pub const fn yg(i: usize) -> usize {
        YG + i
    }
    #[inline]
    pub const fn x2(i: usize) -> usize {
        X2 + i
    }
    #[inline]
    pub const fn q0(i: usize) -> usize {
        Q0 + i
    }
    #[inline]
    pub const fn c0(i: usize) -> usize {
        C0 + i
    }
    #[inline]
    pub const fn q1(i: usize) -> usize {
        Q1 + i
    }
    #[inline]
    pub const fn c1(i: usize) -> usize {
        C1 + i
    }
    #[inline]
    pub const fn k_sub_n(i: usize) -> usize {
        K_SUB_N + i
    }
    #[inline]
    pub const fn xr_sub_p(i: usize) -> usize {
        XR_SUB_P + i
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One ECSM ecall: the math witness plus the three memory addresses and timestamp.
#[derive(Debug, Clone)]
pub struct EcsmOperation {
    pub timestamp: u64,
    pub addr_xg: u64,
    pub addr_k: u64,
    pub addr_xr: u64,
    pub witness: EcsmWitness,
}

// =========================================================================
// Trace generation
// =========================================================================

/// Converts a signed carry to a field element (negatives wrap to `p − |c|`).
fn fe_from_i64(c: i64) -> FE {
    if c >= 0 {
        FE::from(c as u64)
    } else {
        FE::zero() - FE::from((-c) as u64)
    }
}

/// Writes a 32-byte little-endian value as 16 halfwords (U256HL).
fn write_halfwords(table: &mut impl VmTable, row: usize, col: usize, bytes: &[u8; 32]) {
    let mut halfwords = [0u16; 16];
    for j in 0..16 {
        halfwords[j] = u16::from_le_bytes([bytes[2 * j], bytes[2 * j + 1]]);
    }
    table.set_halves(row, col, &halfwords);
}

pub fn generate_ecsm_trace(
    ops: &[EcsmOperation],
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
        let w = &op.witness;

        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row_idx, cols::ADDR_XG_0, op.addr_xg);
        table.set_dword_wl(row_idx, cols::ADDR_K_0, op.addr_k);
        table.set_dword_wl(row_idx, cols::ADDR_XR_0, op.addr_xr);

        table.set_bytes(row_idx, cols::XR, &w.x_r);
        table.set_bytes(row_idx, cols::YR, &w.y_r);
        table.set_bytes(row_idx, cols::K, &w.k);
        table.set_u64(row_idx, cols::LEN_K, w.len_k as u64);
        table.set_bytes(row_idx, cols::XG, &w.x_g);
        table.set_bytes(row_idx, cols::YG, &w.y_g);
        table.set_bytes(row_idx, cols::X2, &w.x2);
        table.set_bytes(row_idx, cols::Q0, &w.q0);
        table.set_bytes(row_idx, cols::Q1, &w.q1);
        write_halfwords(table, row_idx, cols::K_SUB_N, &w.k_sub_n);
        write_halfwords(table, row_idx, cols::XR_SUB_P, &w.x_r_sub_p);

        for i in 0..64 {
            debug_assert!((0..1 << 16).contains(&(w.c0[i] + CARRY_OFFSET_X2)));
            debug_assert!((0..1 << 16).contains(&(w.c1[i] + CARRY_OFFSET_YG)));
            table.set_fe(row_idx, cols::c0(i), fe_from_i64(w.c0[i]));
            table.set_fe(row_idx, cols::c1(i), fe_from_i64(w.c1[i]));
        }

        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows (`mu = 0`) must carry `q1 = p` so the yG carry relation closes: the
    // `p² − q1·p` offset cancels and the µ-gated `b` term drops. Bytes 0..31 hold p; byte 32
    // stays 0 (a valid IS_BIT value).
    for row_idx in n..num_rows {
        table.set_bytes(row_idx, cols::Q1, &P_BYTES);
    }

    trace
}

// =========================================================================
// Bus value helpers
// =========================================================================

fn packed(col: usize) -> BusValue {
    BusValue::Packed {
        start_column: col,
        packing: Packing::Direct,
    }
}

/// `[old[8], is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]` —
/// a 24-element MEMW **read** tuple (`old == value`).
#[allow(clippy::too_many_arguments)]
fn memw_read(
    value: [BusValue; 8],
    is_register: u64,
    base_lo: BusValue,
    base_hi: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
    w2: u64,
    w8: u64,
) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(24);
    v.extend(value.clone()); // old == value (read)
    v.push(BusValue::constant(is_register));
    v.push(base_lo);
    v.push(base_hi);
    v.extend(value);
    v.push(ts_lo);
    v.push(ts_hi);
    v.push(BusValue::constant(w2));
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(w8));
    v
}

/// `[is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]` —
/// a 16-element MEMW **write** tuple (MEMW table supplies `old`).
fn memw_write(
    value: [BusValue; 8],
    base_lo: BusValue,
    base_hi: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
    w8: u64,
) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(16);
    v.push(BusValue::constant(0)); // is_register = 0 (memory)
    v.push(base_lo);
    v.push(base_hi);
    v.extend(value);
    v.push(ts_lo);
    v.push(ts_hi);
    v.push(BusValue::constant(0)); // w2
    v.push(BusValue::constant(0)); // w4
    v.push(BusValue::constant(w8));
    v
}

/// The eight bytes of a 256-bit value at `col + 8*chunk` as MEMW value elements.
fn dword_bytes(col: usize, chunk: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| packed(col + 8 * chunk + b))
}

/// A register value `[lo, hi, 0, 0, 0, 0, 0, 0]` as MEMW value elements.
fn register_value(lo_col: usize, hi_col: usize) -> [BusValue; 8] {
    let mut v: [BusValue; 8] = std::array::from_fn(|_| BusValue::constant(0));
    v[0] = packed(lo_col);
    v[1] = packed(hi_col);
    v
}

/// The 32 bytes of a U256BL coordinate as bus elements (shared shape for the ECDAS bus,
/// used identically by ECSM and ECDAS).
pub fn point_coord_busvalues(col: usize) -> Vec<BusValue> {
    (0..32).map(|b| packed(col + b)).collect()
}

// =========================================================================
// Bus interactions
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let ts_lo = || packed(cols::TIMESTAMP_0);
    let ts_hi = || packed(cols::TIMESTAMP_1);
    let mut out = Vec::new();

    // ECALL receiver (mult = mu): [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            ts_lo(),
            ts_hi(),
            BusValue::constant(ECSM_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(ECSM_SYSCALL_NUMBER >> 32),
        ],
    ));

    // read x11 -> addr_xG (register read at ts).
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_read(
            register_value(cols::ADDR_XG_0, cols::ADDR_XG_1),
            1,
            BusValue::constant(2 * 11),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
            1,
            0,
        ),
    ));
    // read xG: 4 doublewords at addr_xG + 8i (ts).
    for i in 0..4 {
        let base_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ADDR_XG_0,
            },
            LinearTerm::Constant((8 * i) as i64),
        ]);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                dword_bytes(cols::XG, i),
                0,
                base_lo,
                packed(cols::ADDR_XG_1),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // read x12 -> addr_k (register read at ts).
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_read(
            register_value(cols::ADDR_K_0, cols::ADDR_K_1),
            1,
            BusValue::constant(2 * 12),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
            1,
            0,
        ),
    ));
    // read k: 4 doublewords at addr_k + 8i (ts).
    for i in 0..4 {
        let base_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ADDR_K_0,
            },
            LinearTerm::Constant((8 * i) as i64),
        ]);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                dword_bytes(cols::K, i),
                0,
                base_lo,
                packed(cols::ADDR_K_1),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // read x10 -> addr_xR (register read at ts + 1).
    let ts_lo_plus = |d: i64| {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Constant(d),
        ])
    };
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_read(
            register_value(cols::ADDR_XR_0, cols::ADDR_XR_1),
            1,
            BusValue::constant(2 * 10),
            BusValue::constant(0),
            ts_lo_plus(1),
            ts_hi(),
            1,
            0,
        ),
    ));
    // write xR: 4 doublewords at addr_xR + 8i (ts + 2).
    for i in 0..4 {
        let base_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ADDR_XR_0,
            },
            LinearTerm::Constant((8 * i) as i64),
        ]);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_write(
                dword_bytes(cols::XR, i),
                base_lo,
                packed(cols::ADDR_XR_1),
                ts_lo_plus(2),
                ts_hi(),
                1,
            ),
        ));
    }

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
    is_byte(cols::X2, 32, &mut out);
    is_byte(cols::Q0, 32, &mut out);
    is_byte(cols::YG, 32, &mut out);
    is_byte(cols::Q1, 32, &mut out); // q1[0..31]; q1[32] is an IS_BIT constraint
    // xG and k are byte-checked at memory write time (store.rs AreBytes), not re-checked here.

    // IS_HALF range checks on shifted carries, then k_sub_N / xR_sub_p.
    let half_offset = |col: usize, off: i64| {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: col,
            },
            LinearTerm::Constant(off),
        ])
    };
    for i in 0..63 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu(),
            vec![half_offset(cols::c0(i), CARRY_OFFSET_X2)],
        ));
    }
    for i in 0..63 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu(),
            vec![half_offset(cols::c1(i), CARRY_OFFSET_YG)],
        ));
    }
    for i in 0..16 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu(),
            vec![packed(cols::k_sub_n(i))],
        ));
    }
    for i in 0..16 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu(),
            vec![packed(cols::xr_sub_p(i))],
        ));
    }

    // ZERO bus: assert k != 0 (sum of k's 32 bytes is nonzero).
    out.push(BusInteraction::sender(
        BusId::Zero,
        mu(),
        vec![
            BusValue::linear(
                (0..32)
                    .map(|i| LinearTerm::Column {
                        coefficient: 1,
                        column: cols::k(i),
                    })
                    .collect(),
            ),
            BusValue::constant(0), // expected ZERO output = 0  ⇒  input is nonzero
        ],
    ));

    // Delegation buses.
    // SERVE_K send: [ts, addr_k, 31].
    out.push(BusInteraction::sender(
        BusId::ServeK,
        mu(),
        vec![
            ts_lo(),
            ts_hi(),
            packed(cols::ADDR_K_0),
            packed(cols::ADDR_K_1),
            BusValue::constant(31),
        ],
    ));
    // BIT sender: the MSB at position len_k.
    out.push(BusInteraction::sender(
        BusId::Bit,
        mu(),
        vec![ts_lo(), ts_hi(), packed(cols::LEN_K)],
    ));
    // ECDAS start: [ts, xG, yG, xG, yG, len_k - 1, 0].
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        mu(),
        ecdas_tuple(
            cols::XG,
            cols::YG,
            cols::XG,
            cols::YG,
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::LEN_K,
                },
                LinearTerm::Constant(-1),
            ]),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
        ),
    ));
    // ECDAS final receiver: [ts, xR, yR, xG, yG, -1, 0].
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        mu(),
        ecdas_tuple(
            cols::XR,
            cols::YR,
            cols::XG,
            cols::YG,
            BusValue::linear(vec![LinearTerm::Constant(-1)]),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
        ),
    ));

    out
}

/// Builds the ECDAS bus tuple `[ts_lo, ts_hi, accX(32), accY(32), genX(32), genY(32),
/// round, op]`. Shared so the ECSM sender and the ECDAS receiver/sender pack it identically.
#[allow(clippy::too_many_arguments)]
pub fn ecdas_tuple(
    acc_x: usize,
    acc_y: usize,
    gen_x: usize,
    gen_y: usize,
    round: BusValue,
    op: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(2 + 4 * 32 + 2);
    v.push(ts_lo);
    v.push(ts_hi);
    v.extend(point_coord_busvalues(acc_x));
    v.extend(point_coord_busvalues(acc_y));
    v.extend(point_coord_busvalues(gen_x));
    v.extend(point_coord_busvalues(gen_y));
    v.push(round);
    v.push(op);
    v
}

/// Which convolution relation a carry constraint enforces.
#[derive(Clone, Copy)]
pub enum Relation {
    /// `xG² − x2 − q0·p = 0`
    X2,
    /// `yG² + p² − xG·x2 − b − q1·p = 0`
    Yg,
}

/// A range-check overflow addition: `p + xR_sub_p = xR + 2^256` (`k<N` / `xR<p`).
#[derive(Clone, Copy)]
pub enum OverflowKind {
    KLtN,
    XrLtP,
}

impl OverflowKind {
    /// The constant addend's 32-bit word `i` (`N` for `k<N`, `p` for `xR<p`).
    fn const_word(self, i: usize) -> u64 {
        let bytes = match self {
            OverflowKind::KLtN => &N_BYTES,
            OverflowKind::XrLtP => &P_BYTES,
        };
        let mut w = 0u64;
        for b in 0..4 {
            w += (bytes[4 * i + b] as u64) << (8 * b);
        }
        w
    }
    /// Column base of the witnessed halfword addend (`k_sub_N` / `xR_sub_p`).
    fn addend_hl_base(self) -> usize {
        match self {
            OverflowKind::KLtN => cols::K_SUB_N,
            OverflowKind::XrLtP => cols::XR_SUB_P,
        }
    }
    /// Column base of the byte sum (`k` / `xR`).
    fn sum_bl_base(self) -> usize {
        match self {
            OverflowKind::KLtN => cols::K,
            OverflowKind::XrLtP => cols::XR,
        }
    }
}

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..148:
//   0        : IS_BIT(MU)
//   1..65    : ConvCarry(X2, 0..64)
//   65       : ColIsZero(c0(63))
//   66..130  : ConvCarry(Yg, 0..64)
//   130      : ColIsZero(c1(63))
//   131      : IS_BIT(q1(32))
//   132..139 : CarryBit(KLtN, 0..7)
//   139      : OverflowRequired(KLtN)
//   140..147 : CarryBit(XrLtP, 0..7)
//   147      : OverflowRequired(XrLtP)

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// ECSM transition constraints as a single-source [`ConstraintSet`] (148
/// total). No column configuration needed (the layout is fixed via `cols`).
pub struct EcsmConstraints;

impl EcsmConstraints {
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

    /// `bytes[base + j]` for `j < len`, else zero (the `byte` closure in `s_i`).
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

    /// `S_i` for `relation` at limb `i`.
    fn s_i<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let byte = |base: usize, len: usize, j: usize| Self::byte_at(b, base, len, j);
        let mut s = b.zero();
        match relation {
            Relation::X2 => {
                // Σ xG_j·xG_{i-j} − x2_i − Σ q0_j·P_{i-j}
                for j in 0..=i {
                    s = s + byte(cols::XG, 32, j) * byte(cols::XG, 32, i - j);
                    s = s - byte(cols::Q0, 32, j) * Self::p_byte_expr(b, i - j);
                }
                s = s - byte(cols::X2, 32, i);
            }
            Relation::Yg => {
                // Σ (yG_j·yG_{i-j} + P_j·P_{i-j} − x2_j·xG_{i-j} − q1_j·P_{i-j}) − b_i
                for j in 0..=i {
                    s = s + byte(cols::YG, 32, j) * byte(cols::YG, 32, i - j);
                    s = s + Self::p_byte_expr(b, j) * Self::p_byte_expr(b, i - j);
                    s = s - byte(cols::X2, 32, j) * byte(cols::XG, 32, i - j);
                    s = s - byte(cols::Q1, 33, j) * Self::p_byte_expr(b, i - j);
                }
                if i == 0 {
                    // Only the curve constant `b` is µ-gated (µ·B); B_i = 0 for i ≥ 1.
                    let mu = b.main(0, cols::MU);
                    let curve_b = b.const_base(B);
                    s = s - mu * curve_b;
                }
            }
        }
        s
    }

    /// `256·c_i − c_{i-1} − S_i`.
    fn conv_carry<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let c_base = match relation {
            Relation::X2 => cols::C0,
            Relation::Yg => cols::C1,
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

    /// The 8 word-carries of the `kind` addition.
    fn carry_chain<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        kind: OverflowKind,
    ) -> [B::Expr; 8] {
        let hl = kind.addend_hl_base();
        let bl = kind.sum_bl_base();
        let mut c: [B::Expr; 8] = std::array::from_fn(|_| b.zero());
        let mut prev = b.zero();
        for (i, slot) in c.iter_mut().enumerate() {
            // addend1 word i (from halfwords): hl[2i] + 2^16·hl[2i+1]
            let shift_16 = b.const_base(1u64 << 16);
            let addend1 = b.main(0, hl + 2 * i) + b.main(0, hl + 2 * i + 1) * shift_16;
            // sum word i (from bytes): Σ bl[4i+b]·2^{8b}
            let mut sum = b.zero();
            for byte in 0..4 {
                let shift = b.const_base(1u64 << (8 * byte));
                sum = sum + b.main(0, bl + 4 * i + byte) * shift;
            }
            let addend0 = b.const_base(kind.const_word(i));
            let inv = b.const_base(INV_SHIFT_32);
            let ci = (addend0 + addend1 + prev.clone() - sum) * inv;
            *slot = ci.clone();
            prev = ci;
        }
        c
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for EcsmConstraints {
    // The k<N / xR<p carry-bit constraints (µ·c·(1−c)) are degree 3.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: IS_BIT(MU): mu·(1−mu). (deg 2)
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(0, mu.clone() * (one - mu));

        let mut idx = 1;

        // X2 convolution: 64 carries (deg 2) + closing c0(63) (deg 1).
        for i in 0..64 {
            let root = Self::conv_carry(b, Relation::X2, i);
            b.emit_base(idx, root);
            idx += 1;
        }
        let c0_last = b.main(0, cols::c0(63));
        b.emit_base(idx, c0_last);
        idx += 1;

        // Yg convolution: 64 carries (deg 2) + closing c1(63) (deg 1).
        for i in 0..64 {
            let root = Self::conv_carry(b, Relation::Yg, i);
            b.emit_base(idx, root);
            idx += 1;
        }
        let c1_last = b.main(0, cols::c1(63));
        b.emit_base(idx, c1_last);
        idx += 1;

        // idx 131: IS_BIT(q1[32]): x·(1−x). (deg 2)
        let q1_32 = b.main(0, cols::q1(32));
        let one = b.one();
        b.emit_base(idx, q1_32.clone() * (one - q1_32));
        idx += 1;

        // k < N and xR < p: 7 carry bits (deg 3) + overflow-required (deg 2) each.
        for kind in [OverflowKind::KLtN, OverflowKind::XrLtP] {
            let c = Self::carry_chain(b, kind);
            for ci in c.iter().take(7) {
                // µ · c_i · (1 − c_i)
                let mu = b.main(0, cols::MU);
                let one = b.one();
                b.emit_base(idx, mu * ci.clone() * (one - ci.clone()));
                idx += 1;
            }
            // µ · (1 − c_7)
            let mu = b.main(0, cols::MU);
            let one = b.one();
            b.emit_base(idx, mu * (one - c[7].clone()));
            idx += 1;
        }

        debug_assert_eq!(idx, 148);
    }
}
