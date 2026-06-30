//! ECSM core chip — orchestrates one secp256k1 scalar multiplication `k·G`.
//!
//! One row per `ECALL(-11)`. It reads `xG` and `k` from memory, witnesses `yG` and proves
//! `yG² ≡ xG³ + b mod p` (via two byte-limb convolution relations with quotients `q0,q1`
//! and 64-entry carry arrays `c0,c1`), enforces `0 < k < N` and `xR < p`, writes `xR` back,
//! serves the scalar bits directly via the `Bit` bus, and delegates the double-and-add to ECDAS
//! over the `Ecdas`/`Bit` buses.
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
use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::{INV_SHIFT_32, IsBitConstraint, new_is_bit_constraints};
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
    pub const K: usize = 72; // Bit[256] — scalar bits, k[0] is LSB
    pub const LEN_K: usize = 328; // Byte
    pub const XG: usize = 329; // U256BL (32)
    pub const YG: usize = 361; // U256BL (32)
    pub const X2: usize = 393; // U256BL (32)
    pub const Q0: usize = 425; // U256BL (32)
    pub const C0: usize = 457; // BaseField[64]
    pub const Q1: usize = 521; // Byte[33]
    pub const C1: usize = 554; // BaseField[64]
    pub const K_SUB_N: usize = 618; // U256HL (16 halfwords)
    pub const XR_SUB_P: usize = 634; // U256HL (16 halfwords)
    pub const MU: usize = 650;

    pub const NUM_COLUMNS: usize = 651;

    #[inline]
    pub const fn xr(i: usize) -> usize {
        XR + i
    }
    /// Bit `i` of the scalar `k` (0 = LSB, 255 = MSB).
    #[inline]
    pub const fn k_bit(i: usize) -> usize {
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
        for b in 0..256 {
            let bit = (w.k[b / 8] >> (b % 8)) & 1;
            table.set_fe(row_idx, cols::k_bit(b), FE::from(bit as u64));
        }
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

/// `byte_k[byte_idx]` as a MEMW bus value: linear combination of 8 bit columns
/// `k_bit[8*byte_idx .. 8*byte_idx+7]` with coefficients 2^0..2^7.
fn k_byte_busvalue(byte_idx: usize) -> BusValue {
    BusValue::linear(
        (0..8)
            .map(|j| LinearTerm::Column {
                coefficient: 1i64 << j,
                column: cols::k_bit(8 * byte_idx + j),
            })
            .collect(),
    )
}

/// One 8-byte MEMW dword chunk of k (bytes `8*dword_idx .. 8*dword_idx+7`).
fn k_dword_busvalues(dword_idx: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| k_byte_busvalue(8 * dword_idx + b))
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
                k_dword_busvalues(i),
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

    // ZERO bus: assert k != 0 (sum of byte_k[0..31] is nonzero).
    // byte_k[i] = Σ_{j=0}^{7} 2^j · k[8i+j], so Σ byte_k = Σ_{b=0}^{255} 2^(b%8) · k[b].
    out.push(BusInteraction::sender(
        BusId::Zero,
        mu(),
        vec![
            BusValue::linear(
                (0..256)
                    .map(|b| LinearTerm::Column {
                        coefficient: 1i64 << (b % 8),
                        column: cols::k_bit(b),
                    })
                    .collect(),
            ),
            BusValue::constant(0), // expected ZERO output = 0  ⇒  input is nonzero
        ],
    ));

    // Delegation buses.
    // BIT receivers: receive Bit[ts, i] from ECDAS for each scalar bit i=0..255.
    for i in 0..256 {
        out.push(BusInteraction::receiver(
            BusId::Bit,
            Multiplicity::Column(cols::k_bit(i)),
            vec![ts_lo(), ts_hi(), BusValue::constant(i as u64)],
        ));
    }
    // BIT sender: the MSB at position len_k (always 1).
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

// =========================================================================
// Constraints
// =========================================================================

/// Which convolution relation a carry constraint enforces.
#[derive(Clone, Copy)]
pub enum Relation {
    /// `xG² − x2 − q0·p = 0`
    X2,
    /// `yG² + p² − xG·x2 − b − q1·p = 0`
    Yg,
}

fn p_byte<F: IsField>(m: usize) -> FieldElement<F> {
    if m < 32 {
        FieldElement::from(P_BYTES[m] as u64)
    } else {
        FieldElement::zero()
    }
}

/// Convolution carry constraint at limb `i`: `2^8·c_i − c_{i-1} − S_i = 0`, with `c_{-1} = 0`.
/// Unconditional (degree 2); the only µ-gated term is the curve constant `µ·b` inside `S_i`
/// for the yG relation at limb 0 (see [`ConvCarry::s_i`]).
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
        let byte = |base: usize, len: usize, j: usize| -> FieldElement<F> {
            if j < len {
                col(base + j)
            } else {
                FieldElement::zero()
            }
        };
        let mut s = FieldElement::<F>::zero();
        match self.relation {
            Relation::X2 => {
                // Σ xG_j·xG_{i-j} − x2_i − Σ q0_j·P_{i-j}
                for j in 0..=i {
                    s += byte(cols::XG, 32, j) * byte(cols::XG, 32, i - j);
                    s = s - byte(cols::Q0, 32, j) * p_byte::<F>(i - j);
                }
                s = s - byte(cols::X2, 32, i);
            }
            Relation::Yg => {
                // Σ (yG_j·yG_{i-j} + P_j·P_{i-j} − x2_j·xG_{i-j} − q1_j·P_{i-j}) − b_i
                for j in 0..=i {
                    s += byte(cols::YG, 32, j) * byte(cols::YG, 32, i - j);
                    s += p_byte::<F>(j) * p_byte::<F>(i - j);
                    s = s - byte(cols::X2, 32, j) * byte(cols::XG, 32, i - j);
                    s = s - byte(cols::Q1, 33, j) * p_byte::<F>(i - j);
                }
                if i == 0 {
                    // Only the curve constant `b` is gated by `µ`: it vanishes on padding
                    // (µ=0) and equals `b` on real rows (µ=1). `B` is the zero-extension of
                    // `b`, so `B_i = 0` for i ≥ 1 — nothing to gate there. The rest of the
                    // relation stays unconditional.
                    let mu = step.get_main_evaluation_element(0, cols::MU).clone();
                    s = s - mu * FieldElement::<F>::from(B);
                }
            }
        }
        s
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for ConvCarry {
    fn degree(&self) -> usize {
        2 // degree-2 convolution; the only µ-gated term (µ·b) is degree 1
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
            Relation::X2 => cols::C0,
            Relation::Yg => cols::C1,
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

/// The two 256-bit addition-overflow checks (`k < N` and `xR < p`), whose 8 word-carries
/// `c` are virtual. Each `c_i = 2^-32·(addend0_i + addend1_i + c_{i-1} − sum_i)`. The addition
/// must overflow `2^256` (carry-out `c_7 = 1`), which proves the strict inequality:
/// `k < N` is `N + k_sub_N = k + 2^256` (with `k_sub_N = k − N mod 2^256`); `xR < p` is
/// `p + xR_sub_p = xR + 2^256` (with `xR_sub_p = xR − p mod 2^256`).
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
    /// Column base of the sum.
    fn sum_col_base(self) -> usize {
        match self {
            OverflowKind::KLtN => cols::K,
            OverflowKind::XrLtP => cols::XR,
        }
    }
    /// Whether the sum is stored as individual bits (k) rather than bytes (xR).
    fn sum_is_bits(self) -> bool {
        matches!(self, OverflowKind::KLtN)
    }
}

/// Computes the 8 word-carries of the addition for `kind`.
fn carry_chain<F, E>(kind: OverflowKind, step: &TableView<F, E>) -> [FieldElement<F>; 8]
where
    F: IsSubFieldOf<E>,
    E: IsField,
{
    let inv = FieldElement::<F>::from(INV_SHIFT_32);
    let hl = kind.addend_hl_base();
    let base = kind.sum_col_base();
    let mut c: [FieldElement<F>; 8] = std::array::from_fn(|_| FieldElement::zero());
    let mut prev = FieldElement::<F>::zero();
    for (i, slot) in c.iter_mut().enumerate() {
        // addend1 word i (from halfwords): hl[2i] + 2^16·hl[2i+1]
        let addend1 = step.get_main_evaluation_element(0, hl + 2 * i).clone()
            + step.get_main_evaluation_element(0, hl + 2 * i + 1).clone()
                * FieldElement::<F>::from(1u64 << 16);
        // sum word i: computed from individual bits (k) or bytes (xR).
        let mut sum = FieldElement::<F>::zero();
        if kind.sum_is_bits() {
            // k is stored as 256 individual bits; word i = bits 32i..32i+31.
            for bit in 0..32 {
                sum += step
                    .get_main_evaluation_element(0, base + 32 * i + bit)
                    .clone()
                    * FieldElement::<F>::from(1u64 << bit);
            }
        } else {
            // xR is stored as 32 bytes; word i = bytes 4i..4i+3.
            for b in 0..4 {
                sum += step
                    .get_main_evaluation_element(0, base + 4 * i + b)
                    .clone()
                    * FieldElement::<F>::from(1u64 << (8 * b));
            }
        }
        let addend0 = FieldElement::<F>::from(kind.const_word(i));
        let ci = (addend0 + addend1 + prev.clone() - sum) * inv.clone();
        *slot = ci.clone();
        prev = ci;
    }
    c
}

/// `µ · c_i · (1 - c_i) = 0` for a virtual carry bit (degree 3, since `c_i` is linear).
pub struct CarryBit {
    pub kind: OverflowKind,
    pub i: usize,
    pub constraint_idx: usize,
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for CarryBit {
    fn degree(&self) -> usize {
        3
    }
    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }
    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let c = carry_chain(self.kind, step);
        let mu = step.get_main_evaluation_element(0, cols::MU).clone();
        let one = FieldElement::<F>::one();
        mu * c[self.i].clone() * (one - c[self.i].clone())
    }
}

/// `µ · (1 - c_7) = 0`: the top carry must be 1 (the addition overflows).
pub struct OverflowRequired {
    pub kind: OverflowKind,
    pub constraint_idx: usize,
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for OverflowRequired {
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
        let c = carry_chain(self.kind, step);
        let mu = step.get_main_evaluation_element(0, cols::MU).clone();
        mu * (FieldElement::<F>::one() - c[7].clone())
    }
}

/// Creates all ECSM transition constraints (405 total: 1 mu + 256 k bits + 148 others).
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

    // IS_BIT(mu)
    constraints.push(IsBitConstraint::unconditional(cols::MU, idx).boxed());
    idx += 1;

    // IS_BIT(k[i]) for all 256 scalar bits.
    let k_bit_cols: Vec<usize> = (0..256).map(cols::k_bit).collect();
    let (k_bit_constraints, next_idx) = new_is_bit_constraints(&k_bit_cols, idx);
    for c in k_bit_constraints {
        constraints.push(c.boxed());
    }
    idx = next_idx;

    // x2 convolution: 64 carries + closing.
    for i in 0..64 {
        constraints.push(
            ConvCarry {
                relation: Relation::X2,
                i,
                constraint_idx: idx,
            }
            .boxed(),
        );
        idx += 1;
    }
    constraints.push(
        ColIsZero {
            col: cols::c0(63),
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    // yG convolution: 64 carries + closing.
    for i in 0..64 {
        constraints.push(
            ConvCarry {
                relation: Relation::Yg,
                i,
                constraint_idx: idx,
            }
            .boxed(),
        );
        idx += 1;
    }
    constraints.push(
        ColIsZero {
            col: cols::c1(63),
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    // IS_BIT(q1[32])
    constraints.push(IsBitConstraint::unconditional(cols::q1(32), idx).boxed());
    idx += 1;

    // k < N: 7 carry bits + overflow-required.
    for i in 0..7 {
        constraints.push(
            CarryBit {
                kind: OverflowKind::KLtN,
                i,
                constraint_idx: idx,
            }
            .boxed(),
        );
        idx += 1;
    }
    constraints.push(
        OverflowRequired {
            kind: OverflowKind::KLtN,
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    // xR < p: 7 carry bits + overflow-required.
    for i in 0..7 {
        constraints.push(
            CarryBit {
                kind: OverflowKind::XrLtP,
                i,
                constraint_idx: idx,
            }
            .boxed(),
        );
        idx += 1;
    }
    constraints.push(
        OverflowRequired {
            kind: OverflowKind::XrLtP,
            constraint_idx: idx,
        }
        .boxed(),
    );
    idx += 1;

    (constraints, idx)
}
