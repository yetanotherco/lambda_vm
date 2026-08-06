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
//! ## Operand addresses
//! Each of the twelve doubleword accesses carries its own address column, derived from the
//! operand base by a real 64-bit addition and range-checked halfword by halfword — spec
//! `ec:c:range_addr_*` and `ec:c:extrapolate_addr_*`. The table therefore needs no precondition
//! from the caller: an operand whose bytes straddle `2^32` is expressed exactly, and one whose
//! last address would wrap `2^64` is rejected by `µ·carry_1 = 0`. Deriving the bases inline as
//! `ADDR_*_0 + 8i` instead, which is what this table used to do, made the AIR's accepted set
//! depend on the executor's `ecsm_addr_ok` and the two drifted apart (#902).
//!
//! One band is worth naming because this table does not own it: for a base in
//! `[u64::MAX-30, u64::MAX-24]` the additions here are all satisfied, and the trace fails only
//! because MEMW's per-byte `hi = base_1 + carry` is never reduced mod `2^32`, so the last byte's
//! token sits at `2^32` and no PAGE token supplies it. Sound, and in-circuit — but a cross-table
//! argument rather than a constraint of ours, and not reachable through the executor, so no test
//! covers it; forging it needs a hand-built trace.
//!
//! ## Padding
//! Padding rows have `mu = 0`, all columns zero. The yG carry relation closes because both the
//! `µ·p²` and `µ·b` terms vanish when `µ = 0`, leaving the trivial `0 = 0` recurrence. The x²
//! relation has no standalone constant and also closes at all-zero. The range checks, the
//! address additions and the virtual-carry checks are all µ-gated.

use executor::vm::instruction::execution::ECSM_SYSCALL_NUMBER;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::{AddOperand, INV_SHIFT_32, emit_add_pair};
use ecsm::{B, EcsmWitness, N_BYTES, P_BYTES};

// Bias signed convolution carries into IsHalfword [0, 2^16); see spec ecsm.typ "Carry offset" (@ecsm-limb_carry).
pub(crate) const CARRY_OFFSET_X2: i64 = 8160;
pub(crate) const CARRY_OFFSET_YG: i64 = 16319;

// =========================================================================
// Column indices (703 columns; keep in sync with NUM_COLUMNS below)
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
    pub const XG_SUB_P: usize = 618; // U256HL (16 halfwords)
    pub const K_SUB_N: usize = 634; // U256HL (16 halfwords)
    pub const XR_SUB_P: usize = 650; // U256HL (16 halfwords)
    pub const MU: usize = 666;

    /// Per-access operand addresses, `addr_*[i]` for `i = 1..=3`, each a `DWordHL`
    /// (4 halfwords) — spec `ec:c:extrapolate_addr_{xG,k,xR}`.
    ///
    /// `addr_*[0]` is the `ADDR_*_0` / `ADDR_*_1` pair above, which the spec allows to stay a
    /// `DWordWL` ("`addr_xG[0]`, `addr_k[0]` and `addr_xR[0]` could be `DWordWL`s rather than
    /// `HL`s"). It carries no range check of its own, and the reason is NOT that the REGISTER
    /// table range-checks register words — it does not; `register.rs` has no constraint set at
    /// all and pushes `init`/`fini` raw. What binds it is the `i = 0` MEMW send: it puts the
    /// pair straight onto the Memory bus, where the only tokens available come from PAGE and
    /// REGISTER at canonical `(lo, hi)` addresses, so a non-canonical limb matches nothing and
    /// the bus does not balance. Every other chip that takes an address from a register leans
    /// on the same argument (see the note in `memw.rs` about `base_address_1`).
    pub const ADDR_XG_ACC: usize = 667; // DWordHL[3] (12)
    pub const ADDR_K_ACC: usize = 679; // DWordHL[3] (12)
    pub const ADDR_XR_ACC: usize = 691; // DWordHL[3] (12)

    pub const NUM_COLUMNS: usize = 703;

    /// Halfword `hw` of the `i`-th per-access address in the block at `base`.
    ///
    /// The assert is load-bearing, not defensive: `i = 0` would underflow `i - 1` and, in
    /// release, land on `xr_sub_p(13..15)` and `MU`. Access 0 is the `ADDR_*_0`/`ADDR_*_1`
    /// pair, which has no halfword columns — the `0..4` loops over the MEMW sends sit right
    /// next to the `1..4` loops over these, so the wrong bound is the natural typo here.
    #[inline]
    const fn acc_hw(base: usize, i: usize, hw: usize) -> usize {
        assert!(matches!(i, 1..=3), "per-access address index must be 1..=3");
        assert!(hw < 4, "a DWordHL has four halfwords");
        base + (i - 1) * 4 + hw
    }

    /// Halfword `hw` of `addr_xG[i]`, for `i = 1..=3`.
    #[inline]
    pub const fn addr_xg_acc(i: usize, hw: usize) -> usize {
        acc_hw(ADDR_XG_ACC, i, hw)
    }
    /// Halfword `hw` of `addr_k[i]`, for `i = 1..=3`.
    #[inline]
    pub const fn addr_k_acc(i: usize, hw: usize) -> usize {
        acc_hw(ADDR_K_ACC, i, hw)
    }
    /// Halfword `hw` of `addr_xR[i]`, for `i = 1..=3`.
    #[inline]
    pub const fn addr_xr_acc(i: usize, hw: usize) -> usize {
        acc_hw(ADDR_XR_ACC, i, hw)
    }

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
    pub const fn xg_sub_p(i: usize) -> usize {
        XG_SUB_P + i
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
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
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

        // addr_*[i] = addr_*[0] + 8i, as real 64-bit additions (spec
        // `ec:c:extrapolate_addr_*`). The executor rejects the operand before the trace is
        // built if any of these would pass u64::MAX, which is what the top-carry constraint
        // enforces in-circuit.
        for i in 1..4 {
            let off = (8 * i) as u64;
            table.set_dword_hl(
                row_idx,
                cols::addr_xg_acc(i, 0),
                op.addr_xg.wrapping_add(off),
            );
            table.set_dword_hl(row_idx, cols::addr_k_acc(i, 0), op.addr_k.wrapping_add(off));
            table.set_dword_hl(
                row_idx,
                cols::addr_xr_acc(i, 0),
                op.addr_xr.wrapping_add(off),
            );
        }

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
        write_halfwords(table, row_idx, cols::XG_SUB_P, &w.x_g_sub_p);
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

/// The `(lo, hi)` words of `addr_*[i]`, the address of the operand's `i`-th doubleword.
///
/// `i = 0` is the `DWordWL` base bound to the register read; `i = 1..=3` are the per-access
/// `DWordHL` columns that `ec:c:extrapolate_addr_*` derives from it, repacked into words.
/// Nothing here adds an offset: the carry lives in the columns, so an address whose bytes
/// cross `2^32` is expressed exactly.
fn access_addr(
    base_lo: usize,
    base_hi: usize,
    acc: fn(usize, usize) -> usize,
    i: usize,
) -> (BusValue, BusValue) {
    if i == 0 {
        return (packed(base_lo), packed(base_hi));
    }
    let word = |hw: usize| {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: hw,
            },
            LinearTerm::Column {
                coefficient: 1 << 16,
                column: hw + 1,
            },
        ])
    };
    (word(acc(i, 0)), word(acc(i, 2)))
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
    // read xG: 4 doublewords at addr_xG[i] (ts).
    for i in 0..4 {
        let (base_lo, base_hi) =
            access_addr(cols::ADDR_XG_0, cols::ADDR_XG_1, cols::addr_xg_acc, i);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                dword_bytes(cols::XG, i),
                0,
                base_lo,
                base_hi,
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    let ts_lo_plus = |d: i64| {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Constant(d),
        ])
    };

    // read x12 -> addr_k (register read at ts + 1).
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_read(
            register_value(cols::ADDR_K_0, cols::ADDR_K_1),
            1,
            BusValue::constant(2 * 12),
            BusValue::constant(0),
            ts_lo_plus(1),
            ts_hi(),
            1,
            0,
        ),
    ));
    // read k: 4 doublewords at addr_k[i] (ts + 1).
    for i in 0..4 {
        let (base_lo, base_hi) = access_addr(cols::ADDR_K_0, cols::ADDR_K_1, cols::addr_k_acc, i);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                k_dword_busvalues(i),
                0,
                base_lo,
                base_hi,
                ts_lo_plus(1),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // read x10 -> addr_xR (register read at ts + 2, grouped with xR writes).
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_read(
            register_value(cols::ADDR_XR_0, cols::ADDR_XR_1),
            1,
            BusValue::constant(2 * 10),
            BusValue::constant(0),
            ts_lo_plus(2),
            ts_hi(),
            1,
            0,
        ),
    ));
    // write xR: 4 doublewords at addr_xR[i] (ts + 2).
    for i in 0..4 {
        let (base_lo, base_hi) =
            access_addr(cols::ADDR_XR_0, cols::ADDR_XR_1, cols::addr_xr_acc, i);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_write(
                dword_bytes(cols::XR, i),
                base_lo,
                base_hi,
                ts_lo_plus(2),
                ts_hi(),
                1,
            ),
        ));
    }

    // IS_HALF on every halfword of every per-access address (spec `ec:c:range_addr_*`).
    // addr_*[0] is excluded: its canonicality comes from the i = 0 MEMW send reaching the
    // Memory bus, where only canonical PAGE/REGISTER tokens exist (see the `cols` note).
    for acc in [cols::addr_xg_acc, cols::addr_k_acc, cols::addr_xr_acc] {
        for i in 1..4 {
            for hw in 0..4 {
                out.push(BusInteraction::sender(
                    BusId::IsHalfword,
                    mu(),
                    vec![packed(acc(i, hw))],
                ));
            }
        }
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
    is_byte(cols::Q1, 33, &mut out); // q1[0..=32] (all 33 bytes)
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
            vec![packed(cols::xg_sub_p(i))],
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

/// Builds the ECDAS bus tuple `[id, ts_lo, ts_hi, accX(32), accY(32), genX(32), genY(32),
/// round, op]`. `id` is the curve identifier (0 = secp256k1). Shared so the ECSM sender and
/// the ECDAS receiver/sender pack it identically.
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
    let mut v = Vec::with_capacity(1 + 2 + 4 * 32 + 2);
    v.push(BusValue::constant(0)); // id = 0 (secp256k1)
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
    /// `yG² + µ·p² − xG·x2 − µ·b − q1·p = 0`
    Yg,
}

/// The addition-overflow range checks (`xG < p`, `k < N`, `xR < p`), whose 8 word-carries
/// `c` are virtual. Each `c_i = 2^-32·(addend0_i + addend1_i + c_{i-1} − sum_i)`. The addition
/// must overflow `2^256` (carry-out `c_7 = 1`), which proves the strict inequality:
/// `xG < p` is `p + xg_sub_p = xG + 2^256`; `k < N` is `N + k_sub_N = k + 2^256`;
/// `xR < p` is `p + xR_sub_p = xR + 2^256`.
#[derive(Clone, Copy)]
pub enum OverflowKind {
    XgLtP,
    KLtN,
    XrLtP,
}

impl OverflowKind {
    /// The constant addend's 32-bit word `i` (`p` for `xG<p`/`xR<p`, `N` for `k<N`).
    fn const_word(self, i: usize) -> u64 {
        let bytes = match self {
            OverflowKind::XgLtP => &P_BYTES,
            OverflowKind::KLtN => &N_BYTES,
            OverflowKind::XrLtP => &P_BYTES,
        };
        let mut w = 0u64;
        for b in 0..4 {
            w += (bytes[4 * i + b] as u64) << (8 * b);
        }
        w
    }
    /// Column base of the witnessed halfword addend (`xg_sub_p` / `k_sub_N` / `xR_sub_p`).
    fn addend_hl_base(self) -> usize {
        match self {
            OverflowKind::XgLtP => cols::XG_SUB_P,
            OverflowKind::KLtN => cols::K_SUB_N,
            OverflowKind::XrLtP => cols::XR_SUB_P,
        }
    }
    /// Column base of the sum.
    fn sum_col_base(self) -> usize {
        match self {
            OverflowKind::XgLtP => cols::XG,
            OverflowKind::KLtN => cols::K,
            OverflowKind::XrLtP => cols::XR,
        }
    }
    /// Whether the sum is stored as individual bits (k) rather than bytes (xG/xR).
    fn sum_is_bits(self) -> bool {
        matches!(self, OverflowKind::KLtN)
    }
}

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..434:
//   0        : IS_BIT(MU)
//   1..257   : IS_BIT(k[i]) for the 256 scalar bits
//   257      : KBitsZeroOnPadding — (Σ k_bit[i])·(1−µ)
//   258..322 : ConvCarry(X2, 0..64)
//   322      : ColIsZero(c0(63))
//   323..387 : ConvCarry(Yg, 0..64)
//   387      : ColIsZero(c1(63))
//   388      : IS_BIT(q1(32))
//   389..396 : CarryBit(XgLtP, 0..7)
//   396      : OverflowRequired(XgLtP)
//   397..404 : CarryBit(KLtN, 0..7)
//   404      : OverflowRequired(KLtN)
//   405..412 : CarryBit(XrLtP, 0..7)
//   412      : OverflowRequired(XrLtP)
//   413..431 : AddCarryPair(addr_*[i] = addr_*[0] + 8i), 3 operands x i in 1..=3
//   431..434 : µ·carry_1 = 0 on addr_*[3] (the 64-bit addition must not wrap)

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// ECSM transition constraints as a single-source [`ConstraintSet`] (434
/// total). No column configuration needed (the layout is fixed via `cols`).
#[derive(Clone, Copy)]
pub struct EcsmConstraints;

impl EcsmConstraints {
    /// Byte `m` of the field prime `P` (zero beyond 32 bytes).
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
                // Σ (yG_j·yG_{i-j} + µ·P_j·P_{i-j} − x2_j·xG_{i-j} − q1_j·P_{i-j}) − µ·b_i
                // Both the p² offset and the curve constant b are µ-gated: they vanish on
                // padding rows (µ=0), so all columns (including q1) can pad to zero.
                // Factor µ out of the p² sum (µ·ΣP_j·P_{i-j}) as ECDAS `rq()` does, so µ
                // is applied once per limb instead of once per term.
                let mu = b.main(0, cols::MU);
                let mut p2 = b.zero();
                for j in 0..=i {
                    s = s + byte(cols::YG, 32, j) * byte(cols::YG, 32, i - j);
                    p2 = p2 + Self::p_byte_expr(b, j) * Self::p_byte_expr(b, i - j);
                    s = s - byte(cols::X2, 32, j) * byte(cols::XG, 32, i - j);
                    s = s - byte(cols::Q1, 33, j) * Self::p_byte_expr(b, i - j);
                }
                s = s + mu.clone() * p2;
                if i == 0 {
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

    /// The 8 word-carries of the `kind` addition. `k` is summed from its 256 individual
    /// bit columns; `xG`/`xR` from their 32 byte columns.
    fn carry_chain<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        kind: OverflowKind,
    ) -> [B::Expr; 8] {
        let hl = kind.addend_hl_base();
        let base = kind.sum_col_base();
        let mut c: [B::Expr; 8] = std::array::from_fn(|_| b.zero());
        let mut prev = b.zero();
        for (i, slot) in c.iter_mut().enumerate() {
            // addend1 word i (from halfwords): hl[2i] + 2^16·hl[2i+1]
            let shift_16 = b.const_base(1u64 << 16);
            let addend1 = b.main(0, hl + 2 * i) + b.main(0, hl + 2 * i + 1) * shift_16;
            // sum word i: from individual bits (k) or bytes (xG/xR).
            let mut sum = b.zero();
            if kind.sum_is_bits() {
                // k is stored as 256 individual bits; word i = bits 32i..32i+31.
                for bit in 0..32 {
                    let shift = b.const_base(1u64 << bit);
                    sum = sum + b.main(0, base + 32 * i + bit) * shift;
                }
            } else {
                // xG/xR stored as 32 bytes; word i = bytes 4i..4i+3.
                for byte in 0..4 {
                    let shift = b.const_base(1u64 << (8 * byte));
                    sum = sum + b.main(0, base + 4 * i + byte) * shift;
                }
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
    // The xG<p / k<N / xR<p carry-bit constraints (µ·c·(1−c)) are degree 3.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: IS_BIT(MU): mu·(1−mu). (deg 2)
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(0, mu.clone() * (one - mu));

        let mut idx = 1;

        // idx 1..257: IS_BIT(k[i]) for the 256 scalar bits: k·(1−k). (deg 2)
        for i in 0..256 {
            let k = b.main(0, cols::k_bit(i));
            let one = b.one();
            b.emit_base(idx, k.clone() * (one - k));
            idx += 1;
        }

        // idx 257: KBitsZeroOnPadding: (Σ k_bit[i])·(1−µ). All scalar bits must be zero on
        // padding rows (µ=0), else a prover could fire phantom `Bit` bus receives. (deg 2)
        let mut k_sum = b.zero();
        for i in 0..256 {
            k_sum = k_sum + b.main(0, cols::k_bit(i));
        }
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(idx, k_sum * (one - mu));
        idx += 1;

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

        // idx 388: IS_BIT(q1[32]): x·(1−x). (deg 2)
        let q1_32 = b.main(0, cols::q1(32));
        let one = b.one();
        b.emit_base(idx, q1_32.clone() * (one - q1_32));
        idx += 1;

        // xG < p, k < N and xR < p: 7 carry bits (deg 3) + overflow-required (deg 2) each.
        for kind in [OverflowKind::XgLtP, OverflowKind::KLtN, OverflowKind::XrLtP] {
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

        // addr_*[i] = addr_*[0] + 8i for i = 1..=3, as real 64-bit additions with the carry
        // propagating into the high limb (spec `ec:c:extrapolate_addr_*`). Two carry-bit
        // constraints per addition, gated on µ so padding rows close at all-zero.
        for (base, acc) in [
            (
                cols::ADDR_XG_0,
                cols::addr_xg_acc as fn(usize, usize) -> usize,
            ),
            (cols::ADDR_K_0, cols::addr_k_acc),
            (cols::ADDR_XR_0, cols::addr_xr_acc),
        ] {
            for i in 1..4 {
                emit_add_pair(
                    b,
                    idx,
                    &[cols::MU],
                    &AddOperand::dword(base),
                    &AddOperand::constant((8 * i) as i64),
                    &AddOperand::from_dword_hl(acc(i, 0)),
                );
                idx += 2;
            }
        }

        // µ · carry_1 = 0 on the last address of each operand: the 64-bit addition must not
        // wrap. `emit_add_pair` only constrains its carries to be bits, so without this a
        // prover could take addr_*[3] = addr_*[0] + 24 − 2^64. Only i = 3 needs it: if
        // addr + 24 does not wrap, neither does addr + 8 or addr + 16. Mirrors the top-lane
        // constraint in `keccak.rs`.
        for (base, acc) in [
            (
                cols::ADDR_XG_0,
                cols::addr_xg_acc as fn(usize, usize) -> usize,
            ),
            (cols::ADDR_K_0, cols::addr_k_acc),
            (cols::ADDR_XR_0, cols::addr_xr_acc),
        ] {
            let c65536 = b.const_base(65536);
            let inv_2_32 = b.const_base(INV_SHIFT_32);
            let base_lo = b.main(0, base);
            let base_hi = b.main(0, base + 1);
            let sum_lo = b.main(0, acc(3, 0)) + b.main(0, acc(3, 1)) * c65536.clone();
            let sum_hi = b.main(0, acc(3, 2)) + b.main(0, acc(3, 3)) * c65536;
            let c24 = b.const_base(24);
            let carry_0 = (base_lo + c24 - sum_lo) * inv_2_32.clone();
            let carry_1 = (base_hi + carry_0 - sum_hi) * inv_2_32;
            let mu = b.main(0, cols::MU);
            b.emit_base(idx, mu * carry_1);
            idx += 1;
        }

        debug_assert_eq!(idx, 434);
    }
}
