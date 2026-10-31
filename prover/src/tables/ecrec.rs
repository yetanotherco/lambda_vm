//! ECREC chip — one ECDSA public-key recovery on secp256k1 per row.
//!
//! One row per `ECALL(-12)`. It reads the signature `(r, s)`, the message hash `z`
//! (big-endian, as the guest holds them) and the parity of `R`'s y, and either writes
//! the public key `Q = u1·G + u2·R` (`u1 = −z·r⁻¹`, `u2 = s·r⁻¹ mod N`) with status 0,
//! or *bails*: status 1 and a zero key, after which the guest recovers the key in
//! software. A bail proves nothing and needs nothing proven: the guest's software path
//! decides the answer. Everything below is gated on `OK` (the row computed the key).
//!
//! Under `OK` the row proves:
//! - `0 < r < N`, `0 < s < N` (overflow chains, `Zero` lookups);
//! - `w = r⁻¹`: `r·w − 1 − qw·N = 0`; `u1`: `z·w + u1 − qz·N = 0`; `u2`:
//!   `s·w − u2 − qu·N = 0`, with `u1, u2` as 256 bit columns each, `0 < u1, u2 < N`;
//! - `R = (r, yR)` on the curve (ECSM's two relations on `(r, yR)`), `yR < p`, and
//!   `yR[0] = 2·H + PAR` with `PAR` the `x13` value: the requested root;
//! - `A = u1·G` through the `Ecdas`/`Bit` chain keyed at `TS`, `B = u2·R` keyed at
//!   `TS + 1` (ECDAS unchanged), canonical `xA, xB < p` and `xA ≠ xB` (an inverse
//!   witness over the eight 32-bit word differences);
//! - `Q = A + B` as one ECDAS addition row keyed at `TS + 2`, with `xQ, yQ < p`.
//!
//! Every ECALL timestamp is a multiple of 4: the CPU's preprocessed `TIMESTAMP`
//! column is `4·(i + 1)`, and each CPU table adds its base `4·S_k`, four times the
//! rows of the tables before it (`cpu::timestamp_bases`, computed by the verifier
//! from the heights). So `TS`, `TS + 1` and `TS + 2` never meet another call's keys
//! on the two buses only ECSM, ECDAS and ECREC use (the offsets are tied to the
//! stride by a compile-time check below).
//!
//! The witness is `ecsm::compute_ecrec_witness`; padding rows are all zero.

use executor::vm::instruction::execution::ECREC_SYSCALL_NUMBER;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::ecsm::{CARRY_OFFSET_X2, CARRY_OFFSET_YG, point_coord_busvalues};
use super::gpack::{TraceForm, WidthHint, generate_main};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::INV_SHIFT_32;
use ecsm::{B, EcdasStep, EcrecWitness, N_BYTES, P_BYTES};

/// The second chain's key on the `Ecdas` and `Bit` buses, as an offset from the
/// call's timestamp: the first chain is keyed at `TS`, this one at `TS + 1`, the
/// final addition at `TS + ADD_KEY`. Every ECALL timestamp is a multiple of 4, so
/// these never meet another call's keys; and the two chains must not share a key,
/// or their `Bit` messages become one multiset and their scalars can be swapped.
pub(crate) const CHAIN2_KEY: i64 = 1;

/// The final addition's key offset (see [`CHAIN2_KEY`]).
pub(crate) const ADD_KEY: i64 = 2;

// The three keys of one call, `TS`, `TS + CHAIN2_KEY` and `TS + ADD_KEY`, are
// distinct and stay below the next CPU row's timestamp `TS + TIMESTAMP_STRIDE`,
// so no two calls share a key; ECALL timestamps are multiples of the stride
// because every CPU table's base is too (`cpu::timestamp_bases`). A smaller
// stride or larger offsets would let one call's keys meet the next call's.
const _: () = assert!(
    0 < CHAIN2_KEY && CHAIN2_KEY < ADD_KEY && ADD_KEY < super::cpu::TIMESTAMP_STRIDE as i64,
    "ECREC's key offsets must be distinct, nonzero and below the CPU timestamp stride"
);

/// Offset of the three mod-N product relations' carries into `IsHalfword`:
/// their honest carries are within `±8161` (the `x2` relation's bound), and
/// `2^14` leaves room on both sides.
pub(crate) const CARRY_OFFSET_MODN: i64 = 1 << 14;

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    pub const ADDR_OUT_0: usize = 2;
    pub const ADDR_OUT_1: usize = 3;
    pub const ADDR_SIG_0: usize = 4;
    pub const ADDR_SIG_1: usize = 5;
    pub const ADDR_MSG_0: usize = 6;
    pub const ADDR_MSG_1: usize = 7;
    /// The `x13` value: the requested parity of `R`'s y.
    pub const PAR: usize = 8;
    pub const MU: usize = 9;
    /// The row computed the key (status 0); `MU − OK` is the bail status.
    pub const OK: usize = 10;
    /// `yR[0] = 2·H + PAR`.
    pub const H: usize = 11;
    pub const LEN1: usize = 12;
    pub const LEN2: usize = 13;
    pub const R: usize = 14; // 32 bytes, little-endian limbs
    pub const S: usize = R + 32;
    pub const Z: usize = S + 32;
    pub const W: usize = Z + 32;
    pub const QW: usize = W + 32;
    pub const CW: usize = QW + 32; // 64 carries
    pub const U1: usize = CW + 64; // 256 bits, LSB first
    pub const QZ: usize = U1 + 256;
    pub const CZ: usize = QZ + 32;
    pub const U2: usize = CZ + 64; // 256 bits
    pub const QU: usize = U2 + 256;
    pub const CU: usize = QU + 32;
    pub const YR: usize = CU + 64;
    pub const X2: usize = YR + 32;
    pub const Q0: usize = X2 + 32;
    pub const C0: usize = Q0 + 32;
    pub const Q1: usize = C0 + 64; // 33 bytes
    pub const C1: usize = Q1 + 33;
    pub const XA: usize = C1 + 64;
    pub const YA: usize = XA + 32;
    pub const XB: usize = YA + 32;
    pub const YB: usize = XB + 32;
    pub const XQ: usize = YB + 32;
    pub const YQ: usize = XQ + 32;
    /// Inverse witnesses of the eight 32-bit word differences `xB − xA`.
    pub const DINV: usize = YQ + 32;
    pub const R_SUB_N: usize = DINV + 8; // 16 halfwords each
    pub const S_SUB_N: usize = R_SUB_N + 16;
    pub const U1_SUB_N: usize = S_SUB_N + 16;
    pub const U2_SUB_N: usize = U1_SUB_N + 16;
    pub const YR_SUB_P: usize = U2_SUB_N + 16;
    pub const XA_SUB_P: usize = YR_SUB_P + 16;
    pub const XB_SUB_P: usize = XA_SUB_P + 16;
    pub const XQ_SUB_P: usize = XB_SUB_P + 16;
    pub const YQ_SUB_P: usize = XQ_SUB_P + 16;

    pub const NUM_COLUMNS: usize = YQ_SUB_P + 16;
}

// =========================================================================
// Operation
// =========================================================================

/// One ECREC ecall: its witness, addresses and timestamp.
#[derive(Debug, Clone)]
pub struct EcrecOperation {
    pub timestamp: u64,
    pub addr_out: u64,
    pub addr_sig: u64,
    pub addr_msg: u64,
    pub witness: EcrecWitness,
}

impl EcrecOperation {
    /// The heap its witness owns beyond the op: the ECDAS steps, which no ECREC table reads.
    pub(crate) fn steps_heap_bytes(&self) -> usize {
        (self.witness.chain1.capacity() + self.witness.chain2.capacity())
            * std::mem::size_of::<EcdasStep>()
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

fn write_halfwords(table: &mut impl VmTable, row: usize, col: usize, bytes: &[u8; 32]) {
    let halfwords: [u16; 16] =
        std::array::from_fn(|j| u16::from_le_bytes([bytes[2 * j], bytes[2 * j + 1]]));
    table.set_halves(row, col, &halfwords);
}

fn write_bits(table: &mut impl VmTable, row: usize, col: usize, bytes: &[u8; 32]) {
    for b in 0..256 {
        table.set_fe(
            row,
            col + b,
            FE::from(((bytes[b / 8] >> (b % 8)) & 1) as u64),
        );
    }
}

fn write_carries(table: &mut impl VmTable, row: usize, col: usize, c: &[i64; 64], offset: i64) {
    for (i, &v) in c.iter().enumerate() {
        debug_assert!((0..1 << 16).contains(&(v + offset)), "carry {i} = {v}");
        table.set_fe(row, col + i, fe_from_i64(v));
    }
}

/// The eight 32-bit word differences `xB − xA` of a witness, as integers.
pub(crate) fn word_differences(w: &EcrecWitness) -> [i64; 8] {
    std::array::from_fn(|c| {
        let word = |b: &[u8; 32]| u32::from_le_bytes(b[4 * c..4 * c + 4].try_into().unwrap());
        i64::from(word(&w.x_b)) - i64::from(word(&w.x_a))
    })
}

pub fn generate_ecrec_trace(
    ops: &[EcrecOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_ecrec_trace_as(ops, TraceForm::Wide)
}

static WIDTHS: WidthHint = WidthHint::new();

pub fn generate_ecrec_trace_as(
    ops: &[EcrecOperation],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_ecrec_rows_as(ops, ops.len().next_power_of_two().max(4), form)
}

/// An ECREC table of `num_rows` rows: `ops`, then all-zero padding rows.
pub fn generate_ecrec_rows_as(
    ops: &[EcrecOperation],
    num_rows: usize,
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row, op) in ops.iter().enumerate() {
            let w = &op.witness;
            table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
            table.set_dword_wl(row, cols::ADDR_OUT_0, op.addr_out);
            table.set_dword_wl(row, cols::ADDR_SIG_0, op.addr_sig);
            table.set_dword_wl(row, cols::ADDR_MSG_0, op.addr_msg);
            table.set_u64(row, cols::PAR, w.parity as u64);
            table.set_fe(row, cols::MU, FE::one());
            table.set_bytes(row, cols::R, &w.r);
            table.set_bytes(row, cols::S, &w.s);
            table.set_bytes(row, cols::Z, &w.z);
            if !w.ok {
                continue;
            }
            table.set_fe(row, cols::OK, FE::one());
            table.set_u64(row, cols::H, w.h as u64);
            table.set_u64(row, cols::LEN1, w.len1 as u64);
            table.set_u64(row, cols::LEN2, w.len2 as u64);
            table.set_bytes(row, cols::W, &w.w);
            table.set_bytes(row, cols::QW, &w.qw);
            write_carries(table, row, cols::CW, &w.cw, CARRY_OFFSET_MODN);
            write_bits(table, row, cols::U1, &w.u1);
            table.set_bytes(row, cols::QZ, &w.qz);
            write_carries(table, row, cols::CZ, &w.cz, CARRY_OFFSET_MODN);
            write_bits(table, row, cols::U2, &w.u2);
            table.set_bytes(row, cols::QU, &w.qu);
            write_carries(table, row, cols::CU, &w.cu, CARRY_OFFSET_MODN);
            table.set_bytes(row, cols::YR, &w.y_r);
            table.set_bytes(row, cols::X2, &w.x2);
            table.set_bytes(row, cols::Q0, &w.q0);
            write_carries(table, row, cols::C0, &w.c0, CARRY_OFFSET_X2);
            table.set_bytes(row, cols::Q1, &w.q1);
            write_carries(table, row, cols::C1, &w.c1, CARRY_OFFSET_YG);
            table.set_bytes(row, cols::XA, &w.x_a);
            table.set_bytes(row, cols::YA, &w.y_a);
            table.set_bytes(row, cols::XB, &w.x_b);
            table.set_bytes(row, cols::YB, &w.y_b);
            table.set_bytes(row, cols::XQ, &w.x_q);
            table.set_bytes(row, cols::YQ, &w.y_q);
            // One nonzero word difference gets its inverse; the sum Σ d·inv is then 1.
            let d = word_differences(w);
            let c = d
                .iter()
                .position(|&v| v != 0)
                .expect("xA ≠ xB on a computed row");
            table.set_fe(
                row,
                cols::DINV + c,
                fe_from_i64(d[c]).inv().expect("nonzero"),
            );
            for (col, bytes) in [
                (cols::R_SUB_N, &w.r_sub_n),
                (cols::S_SUB_N, &w.s_sub_n),
                (cols::U1_SUB_N, &w.u1_sub_n),
                (cols::U2_SUB_N, &w.u2_sub_n),
                (cols::YR_SUB_P, &w.y_r_sub_p),
                (cols::XA_SUB_P, &w.x_a_sub_p),
                (cols::XB_SUB_P, &w.x_b_sub_p),
                (cols::XQ_SUB_P, &w.x_q_sub_p),
                (cols::YQ_SUB_P, &w.y_q_sub_p),
            ] {
                write_halfwords(table, row, col, bytes);
            }
        }
    })
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

fn col_plus(col: usize, k: i64) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: col,
        },
        LinearTerm::Constant(k),
    ])
}

/// `MU − OK`: the bail status (and the multiplicity of nothing).
fn bail_value() -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: cols::MU,
        },
        LinearTerm::Column {
            coefficient: -1,
            column: cols::OK,
        },
    ])
}

/// `[old[8], is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]`, a read.
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
    v.extend(value.clone());
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

/// `[is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]`, a write.
fn memw_write(
    value: [BusValue; 8],
    base_lo: BusValue,
    base_hi: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(16);
    v.push(BusValue::constant(0));
    v.push(base_lo);
    v.push(base_hi);
    v.extend(value);
    v.push(ts_lo);
    v.push(ts_hi);
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(1));
    v
}

fn register_value(lo: BusValue, hi: BusValue) -> [BusValue; 8] {
    let mut v: [BusValue; 8] = std::array::from_fn(|_| BusValue::constant(0));
    v[0] = lo;
    v[1] = hi;
    v
}

/// Memory byte `m` (`0..32`) of a big-endian 256-bit value held as little-endian limbs
/// at `col`.
fn be_byte(col: usize, m: usize) -> BusValue {
    packed(col + 31 - m)
}

/// The `Ecdas` tuple `[0, ts_lo, ts_hi, acc_x, acc_y, gen_x, gen_y, round, op]` with
/// any bus values (ECSM's [`super::ecsm::ecdas_tuple`] takes columns only).
#[allow(clippy::too_many_arguments)]
fn ecdas_tuple(
    acc_x: Vec<BusValue>,
    acc_y: Vec<BusValue>,
    gen_x: Vec<BusValue>,
    gen_y: Vec<BusValue>,
    round: BusValue,
    op: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(1 + 2 + 4 * 32 + 2);
    v.push(BusValue::constant(0)); // curve id 0 = secp256k1, as ECSM and ECDAS
    v.push(ts_lo);
    v.push(ts_hi);
    v.extend(acc_x);
    v.extend(acc_y);
    v.extend(gen_x);
    v.extend(gen_y);
    v.push(round);
    v.push(op);
    v
}

/// A constant 256-bit value as 32 little-endian byte constants.
fn const_coord(le: &[u8; 32]) -> Vec<BusValue> {
    le.iter().map(|&b| BusValue::constant(b as u64)).collect()
}

/// The generator's coordinates, little-endian.
pub(crate) fn generator_le() -> ([u8; 32], [u8; 32]) {
    let mut x = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    let mut y = [
        0x48, 0x3A, 0xDA, 0x77, 0x26, 0xA3, 0xC4, 0x65, 0x5D, 0xA4, 0xFB, 0xFC, 0x0E, 0x11, 0x08,
        0xA8, 0xFD, 0x17, 0xB4, 0x48, 0xA6, 0x85, 0x54, 0x19, 0x9C, 0x47, 0xD0, 0x8F, 0xFB, 0x10,
        0xD4, 0xB8,
    ];
    x.reverse();
    y.reverse();
    (x, y)
}

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let ok = || Multiplicity::Column(cols::OK);
    let ts = |k: i64| col_plus(cols::TIMESTAMP_0, k);
    let ts_hi = || packed(cols::TIMESTAMP_1);
    let mut out = Vec::new();

    // ECALL receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            ts(0),
            ts_hi(),
            BusValue::constant(ECREC_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(ECREC_SYSCALL_NUMBER >> 32),
        ],
    ));

    // A register read of `reg` with value `[lo, hi]` at ts + k.
    let reg_read = |reg: u64, lo: BusValue, hi: BusValue, k: i64| {
        BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                register_value(lo, hi),
                1,
                BusValue::constant(2 * reg),
                BusValue::constant(0),
                ts(k),
                ts_hi(),
                1,
                0,
            ),
        )
    };

    // T: x11 -> addr_sig; the signature r ‖ s (8 doublewords, big-endian values).
    out.push(reg_read(
        11,
        packed(cols::ADDR_SIG_0),
        packed(cols::ADDR_SIG_1),
        0,
    ));
    for i in 0..8usize {
        let value: [BusValue; 8] = std::array::from_fn(|j| {
            let m = 8 * i + j;
            if m < 32 {
                be_byte(cols::R, m)
            } else {
                be_byte(cols::S, m - 32)
            }
        });
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                value,
                0,
                col_plus(cols::ADDR_SIG_0, (8 * i) as i64),
                packed(cols::ADDR_SIG_1),
                ts(0),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // T + 1: x12 -> addr_msg, x13 = parity (a value), the hash z (4 doublewords).
    out.push(reg_read(
        12,
        packed(cols::ADDR_MSG_0),
        packed(cols::ADDR_MSG_1),
        1,
    ));
    out.push(reg_read(13, packed(cols::PAR), BusValue::constant(0), 1));
    for i in 0..4usize {
        let value: [BusValue; 8] = std::array::from_fn(|j| be_byte(cols::Z, 8 * i + j));
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                value,
                0,
                col_plus(cols::ADDR_MSG_0, (8 * i) as i64),
                packed(cols::ADDR_MSG_1),
                ts(1),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // T + 2: x10 -> addr_out; the key Q.x ‖ Q.y (big-endian) and the status doubleword.
    out.push(reg_read(
        10,
        packed(cols::ADDR_OUT_0),
        packed(cols::ADDR_OUT_1),
        2,
    ));
    for i in 0..9usize {
        let value: [BusValue; 8] = std::array::from_fn(|j| {
            let m = 8 * i + j;
            match m {
                0..32 => be_byte(cols::XQ, m),
                32..64 => be_byte(cols::YQ, m - 32),
                64 => bail_value(),
                _ => BusValue::constant(0),
            }
        });
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_write(
                value,
                col_plus(cols::ADDR_OUT_0, (8 * i) as i64),
                packed(cols::ADDR_OUT_1),
                ts(2),
                ts_hi(),
            ),
        ));
    }

    // Byte checks on the computed row's witness bytes.
    let is_byte = |col: usize, len: usize, out: &mut Vec<BusInteraction>| {
        for i in 0..len {
            out.push(BusInteraction::sender(
                BusId::AreBytes,
                ok(),
                vec![packed(col + i), BusValue::constant(0)],
            ));
        }
    };
    for col in [
        cols::W,
        cols::QW,
        cols::QZ,
        cols::QU,
        cols::YR,
        cols::X2,
        cols::Q0,
    ] {
        is_byte(col, 32, &mut out);
    }
    is_byte(cols::Q1, 33, &mut out);
    is_byte(cols::H, 1, &mut out);

    // Halfword checks: the five relations' carries (offset), the nine subtraction witnesses.
    for (col, off) in [
        (cols::CW, CARRY_OFFSET_MODN),
        (cols::CZ, CARRY_OFFSET_MODN),
        (cols::CU, CARRY_OFFSET_MODN),
        (cols::C0, CARRY_OFFSET_X2),
        (cols::C1, CARRY_OFFSET_YG),
    ] {
        for i in 0..63 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                ok(),
                vec![col_plus(col + i, off)],
            ));
        }
    }
    for col in [
        cols::R_SUB_N,
        cols::S_SUB_N,
        cols::U1_SUB_N,
        cols::U2_SUB_N,
        cols::YR_SUB_P,
        cols::XA_SUB_P,
        cols::XB_SUB_P,
        cols::XQ_SUB_P,
        cols::YQ_SUB_P,
    ] {
        for i in 0..16 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                ok(),
                vec![packed(col + i)],
            ));
        }
    }

    // Zero: r, s, u1, u2 nonzero (the sums of their bytes).
    let byte_sum = |col: usize, bits: bool| {
        if bits {
            BusValue::linear(
                (0..256)
                    .map(|b| LinearTerm::Column {
                        coefficient: 1i64 << (b % 8),
                        column: col + b,
                    })
                    .collect(),
            )
        } else {
            BusValue::linear(
                (0..32)
                    .map(|b| LinearTerm::Column {
                        coefficient: 1,
                        column: col + b,
                    })
                    .collect(),
            )
        }
    };
    for (col, bits) in [
        (cols::R, false),
        (cols::S, false),
        (cols::U1, true),
        (cols::U2, true),
    ] {
        out.push(BusInteraction::sender(
            BusId::Zero,
            ok(),
            vec![byte_sum(col, bits), BusValue::constant(0)],
        ));
    }

    // The first chain, u1·G, keyed at ts.
    let (gx, gy) = generator_le();
    for i in 0..256 {
        out.push(BusInteraction::receiver(
            BusId::Bit,
            Multiplicity::Column(cols::U1 + i),
            vec![ts(0), ts_hi(), BusValue::constant(i as u64)],
        ));
    }
    out.push(BusInteraction::sender(
        BusId::Bit,
        ok(),
        vec![ts(0), ts_hi(), packed(cols::LEN1)],
    ));
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        ok(),
        ecdas_tuple(
            const_coord(&gx),
            const_coord(&gy),
            const_coord(&gx),
            const_coord(&gy),
            col_plus(cols::LEN1, -1),
            BusValue::constant(0),
            ts(0),
            ts_hi(),
        ),
    ));
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        ok(),
        ecdas_tuple(
            point_coord_busvalues(cols::XA),
            point_coord_busvalues(cols::YA),
            const_coord(&gx),
            const_coord(&gy),
            BusValue::linear(vec![LinearTerm::Constant(-1)]),
            BusValue::constant(0),
            ts(0),
            ts_hi(),
        ),
    ));

    // The second chain, u2·R with R = (r, yR), keyed at ts + 1.
    for i in 0..256 {
        out.push(BusInteraction::receiver(
            BusId::Bit,
            Multiplicity::Column(cols::U2 + i),
            vec![ts(CHAIN2_KEY), ts_hi(), BusValue::constant(i as u64)],
        ));
    }
    out.push(BusInteraction::sender(
        BusId::Bit,
        ok(),
        vec![ts(CHAIN2_KEY), ts_hi(), packed(cols::LEN2)],
    ));
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        ok(),
        ecdas_tuple(
            point_coord_busvalues(cols::R),
            point_coord_busvalues(cols::YR),
            point_coord_busvalues(cols::R),
            point_coord_busvalues(cols::YR),
            col_plus(cols::LEN2, -1),
            BusValue::constant(0),
            ts(CHAIN2_KEY),
            ts_hi(),
        ),
    ));
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        ok(),
        ecdas_tuple(
            point_coord_busvalues(cols::XB),
            point_coord_busvalues(cols::YB),
            point_coord_busvalues(cols::R),
            point_coord_busvalues(cols::YR),
            BusValue::linear(vec![LinearTerm::Constant(-1)]),
            BusValue::constant(0),
            ts(CHAIN2_KEY),
            ts_hi(),
        ),
    ));

    // The final addition Q = A + B, one ECDAS add row keyed at ts + 2.
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        ok(),
        ecdas_tuple(
            point_coord_busvalues(cols::XA),
            point_coord_busvalues(cols::YA),
            point_coord_busvalues(cols::XB),
            point_coord_busvalues(cols::YB),
            BusValue::constant(0),
            BusValue::constant(1),
            ts(ADD_KEY),
            ts_hi(),
        ),
    ));
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        ok(),
        ecdas_tuple(
            point_coord_busvalues(cols::XQ),
            point_coord_busvalues(cols::YQ),
            point_coord_busvalues(cols::XB),
            point_coord_busvalues(cols::YB),
            BusValue::linear(vec![LinearTerm::Constant(-1)]),
            BusValue::constant(0),
            ts(ADD_KEY),
            ts_hi(),
        ),
    ));

    out
}

// =========================================================================
// Constraints
// =========================================================================

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// The `< m` checks, as ECSM's overflow chains: `m + sub = v + 2^256` with a carry out.
/// Each is load-bearing (`tests::ecrec_tests::*_is_refused_only_by_its_bound`).
#[derive(Clone, Copy, Debug)]
pub enum Bound {
    RLtN,
    /// Without it `(r, s + N)`, which Ethereum refuses, recovers the same key.
    SLtN,
    /// `u1 < N` and `u2 < N` keep each chain off ECDAS's chord branch at `acc = gen`,
    /// where λ is free: a prover who makes the true scalar 2 (`r = −z/2`, or
    /// `s = 2r`) and claims `2 + N` would end that chain in `G + G` by the chord.
    U1LtN,
    /// See [`Bound::U1LtN`].
    U2LtN,
    YrLtP,
    /// Canonical `xA`, `xB` make the `xA ≠ xB` witness mean `xA ≢ xB (mod p)`.
    XaLtP,
    /// See [`Bound::XaLtP`].
    XbLtP,
    XqLtP,
    YqLtP,
}

impl Bound {
    pub const ALL: [Bound; 9] = [
        Bound::RLtN,
        Bound::SLtN,
        Bound::U1LtN,
        Bound::U2LtN,
        Bound::YrLtP,
        Bound::XaLtP,
        Bound::XbLtP,
        Bound::XqLtP,
        Bound::YqLtP,
    ];

    fn modulus(self) -> &'static [u8; 32] {
        match self {
            Bound::RLtN | Bound::SLtN | Bound::U1LtN | Bound::U2LtN => &N_BYTES,
            _ => &P_BYTES,
        }
    }

    fn sub_col(self) -> usize {
        match self {
            Bound::RLtN => cols::R_SUB_N,
            Bound::SLtN => cols::S_SUB_N,
            Bound::U1LtN => cols::U1_SUB_N,
            Bound::U2LtN => cols::U2_SUB_N,
            Bound::YrLtP => cols::YR_SUB_P,
            Bound::XaLtP => cols::XA_SUB_P,
            Bound::XbLtP => cols::XB_SUB_P,
            Bound::XqLtP => cols::XQ_SUB_P,
            Bound::YqLtP => cols::YQ_SUB_P,
        }
    }

    /// The bounded value's column and whether it is held as bits (else bytes).
    fn value_col(self) -> (usize, bool) {
        match self {
            Bound::RLtN => (cols::R, false),
            Bound::SLtN => (cols::S, false),
            Bound::U1LtN => (cols::U1, true),
            Bound::U2LtN => (cols::U2, true),
            Bound::YrLtP => (cols::YR, false),
            Bound::XaLtP => (cols::XA, false),
            Bound::XbLtP => (cols::XB, false),
            Bound::XqLtP => (cols::XQ, false),
            Bound::YqLtP => (cols::YQ, false),
        }
    }
}

/// Which product relation a carry chain enforces.
#[derive(Clone, Copy, Debug)]
pub enum Relation {
    /// `r·w − 1 − qw·N = 0`
    W,
    /// `z·w + u1 − qz·N = 0`
    U1,
    /// `s·w − u2 − qu·N = 0`
    U2,
    /// `r² − x2 − q0·p = 0`
    X2,
    /// `yR² + p² − r·x2 − b − q1·p = 0`
    Yr,
}

/// ECREC constraints. Index map:
///   0, 1, 2   : IS_BIT(MU), IS_BIT(OK), IS_BIT(PAR)
///   3         : OK·(1 − MU)
///   4..516    : IS_BIT of the 512 scalar bits
///   516       : (Σ u1 bits + Σ u2 bits)·(1 − OK)
///   517..842  : for W, U1, U2, X2, Yr: 64 OK-gated carry steps + OK·c₆₃
///   842       : OK·IS_BIT(q1[32])
///   843       : OK·(yR[0] − 2·H − PAR)
///   844..916  : for the nine bounds: 7 OK·carry-bit + OK·(1 − c₇)
///   916       : Σ_c d_c·DINV_c − OK
///   917..981  : (MU − OK)·xQ_i, (MU − OK)·yQ_i
#[derive(Clone, Copy)]
pub struct EcrecConstraints;

pub const NUM_CONSTRAINTS: usize = 981;

type Gl = GoldilocksField;
type Ext = GoldilocksExtension;

impl EcrecConstraints {
    fn byte<B: ConstraintBuilder<Gl, Ext>>(b: &B, base: usize, len: usize, j: usize) -> B::Expr {
        if j < len {
            b.main(0, base + j)
        } else {
            b.zero()
        }
    }

    /// Byte `j` of a scalar held as 256 bits at `base`.
    fn bits_byte<B: ConstraintBuilder<Gl, Ext>>(b: &B, base: usize, j: usize) -> B::Expr {
        let mut e = b.zero();
        if j < 32 {
            for k in 0..8 {
                e = e + b.main(0, base + 8 * j + k) * b.const_base(1u64 << k);
            }
        }
        e
    }

    fn const_byte<B: ConstraintBuilder<Gl, Ext>>(b: &B, bytes: &[u8; 32], j: usize) -> B::Expr {
        if j < 32 {
            b.const_base(bytes[j] as u64)
        } else {
            b.zero()
        }
    }

    /// `S_i` of `relation` at limb `i`.
    fn s_i<B: ConstraintBuilder<Gl, Ext>>(b: &B, relation: Relation, i: usize) -> B::Expr {
        let by = |base: usize, j: usize| Self::byte(b, base, 32, j);
        let n_ = |j: usize| Self::const_byte(b, &N_BYTES, j);
        let p_ = |j: usize| Self::const_byte(b, &P_BYTES, j);
        let mut s = b.zero();
        match relation {
            Relation::W => {
                for j in 0..=i {
                    s = s + by(cols::R, j) * by(cols::W, i - j) - by(cols::QW, j) * n_(i - j);
                }
                if i == 0 {
                    s = s - b.one();
                }
            }
            Relation::U1 => {
                for j in 0..=i {
                    s = s + by(cols::Z, j) * by(cols::W, i - j) - by(cols::QZ, j) * n_(i - j);
                }
                s = s + Self::bits_byte(b, cols::U1, i);
            }
            Relation::U2 => {
                for j in 0..=i {
                    s = s + by(cols::S, j) * by(cols::W, i - j) - by(cols::QU, j) * n_(i - j);
                }
                s = s - Self::bits_byte(b, cols::U2, i);
            }
            Relation::X2 => {
                for j in 0..=i {
                    s = s + by(cols::R, j) * by(cols::R, i - j) - by(cols::Q0, j) * p_(i - j);
                }
                s = s - by(cols::X2, i);
            }
            Relation::Yr => {
                let mut p2 = b.zero();
                for j in 0..=i {
                    s = s + by(cols::YR, j) * by(cols::YR, i - j);
                    p2 = p2 + p_(j) * p_(i - j);
                    s = s - by(cols::X2, j) * by(cols::R, i - j);
                    s = s - Self::byte(b, cols::Q1, 33, j) * p_(i - j);
                }
                s = s + p2;
                if i == 0 {
                    s = s - b.const_base(B);
                }
            }
        }
        s
    }

    fn carry_base(relation: Relation) -> usize {
        match relation {
            Relation::W => cols::CW,
            Relation::U1 => cols::CZ,
            Relation::U2 => cols::CU,
            Relation::X2 => cols::C0,
            Relation::Yr => cols::C1,
        }
    }

    /// The eight virtual word carries of `bound`'s addition `m + sub = v + 2^256`.
    fn bound_carries<B: ConstraintBuilder<Gl, Ext>>(b: &B, bound: Bound) -> [B::Expr; 8] {
        let m = bound.modulus();
        let hl = bound.sub_col();
        let (vcol, bits) = bound.value_col();
        let mut c: [B::Expr; 8] = std::array::from_fn(|_| b.zero());
        let mut prev = b.zero();
        for (i, slot) in c.iter_mut().enumerate() {
            let sub = b.main(0, hl + 2 * i) + b.main(0, hl + 2 * i + 1) * b.const_base(1 << 16);
            let mut v = b.zero();
            if bits {
                for k in 0..32 {
                    v = v + b.main(0, vcol + 32 * i + k) * b.const_base(1u64 << k);
                }
            } else {
                for k in 0..4 {
                    v = v + b.main(0, vcol + 4 * i + k) * b.const_base(1u64 << (8 * k));
                }
            }
            let word = (0..4).fold(0u64, |acc, k| acc + ((m[4 * i + k] as u64) << (8 * k)));
            let ci = (b.const_base(word) + sub + prev.clone() - v) * b.const_base(INV_SHIFT_32);
            *slot = ci.clone();
            prev = ci;
        }
        c
    }
}

impl ConstraintSet<Gl, Ext> for EcrecConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<Gl, Ext>>(&self, b: &mut B) {
        let mut idx = 0;
        let mut emit = |b: &mut B, e: B::Expr| {
            b.emit_base(idx, e);
            idx += 1;
        };
        let main = |b: &B, c: usize| b.main(0, c);

        for c in [cols::MU, cols::OK, cols::PAR] {
            let x = main(b, c);
            let e = x.clone() * (b.one() - x);
            emit(b, e);
        }
        let e = main(b, cols::OK) * (b.one() - main(b, cols::MU));
        emit(b, e);

        let mut bit_sum = b.zero();
        for base in [cols::U1, cols::U2] {
            for i in 0..256 {
                let x = main(b, base + i);
                bit_sum = bit_sum + x.clone();
                let e = x.clone() * (b.one() - x);
                emit(b, e);
            }
        }
        let e = bit_sum * (b.one() - main(b, cols::OK));
        emit(b, e);

        for relation in [
            Relation::W,
            Relation::U1,
            Relation::U2,
            Relation::X2,
            Relation::Yr,
        ] {
            let base = Self::carry_base(relation);
            for i in 0..64 {
                let prev = if i == 0 {
                    b.zero()
                } else {
                    main(b, base + i - 1)
                };
                let step = main(b, base + i) * b.const_base(256) - prev - Self::s_i(b, relation, i);
                let e = main(b, cols::OK) * step;
                emit(b, e);
            }
            let e = main(b, cols::OK) * main(b, base + 63);
            emit(b, e);
        }

        let q = main(b, cols::Q1 + 32);
        let e = main(b, cols::OK) * q.clone() * (b.one() - q);
        emit(b, e);

        let e = main(b, cols::OK)
            * (main(b, cols::YR) - main(b, cols::H) * b.const_base(2) - main(b, cols::PAR));
        emit(b, e);

        for bound in Bound::ALL {
            let c = Self::bound_carries(b, bound);
            for ci in c.iter().take(7) {
                let e = main(b, cols::OK) * ci.clone() * (b.one() - ci.clone());
                emit(b, e);
            }
            let e = main(b, cols::OK) * (b.one() - c[7].clone());
            emit(b, e);
        }

        // xA ≠ xB: Σ_c (xB − xA)_c · DINV_c = OK over the eight 32-bit word differences.
        let mut sum = b.zero();
        for c in 0..8 {
            let mut d = b.zero();
            for k in 0..4 {
                d = d
                    + (main(b, cols::XB + 4 * c + k) - main(b, cols::XA + 4 * c + k))
                        * b.const_base(1u64 << (8 * k));
            }
            sum = sum + d * main(b, cols::DINV + c);
        }
        let e = sum - main(b, cols::OK);
        emit(b, e);

        // A bail writes a zero key.
        for base in [cols::XQ, cols::YQ] {
            for i in 0..32 {
                let e = (main(b, cols::MU) - main(b, cols::OK)) * main(b, base + i);
                emit(b, e);
            }
        }

        debug_assert_eq!(idx, NUM_CONSTRAINTS);
    }
}
