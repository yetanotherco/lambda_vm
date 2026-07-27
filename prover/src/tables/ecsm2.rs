//! ECSM2 core chip — orchestrates one `Q = u1·P1 + u2·P2` (lincomb2) evaluation.
//!
//! One row per `ECALL(-12)`. It receives the ECALL, reads the three 64-byte
//! operands and writes the 64-byte result plus the status word, proves `P2` is
//! on the curve and canonical in `y`, proves both scalars are in `[1, N)`,
//! publishes the three addends and the NUMS correction constant on the
//! [`Addend`](BusId::Addend) bus, serves the 512 joint scalar digits on the
//! [`JointBit`](BusId::JointBit) bus, and seeds and drains the three segments of
//! the joint double-add chain that [`ecdas2`](super::ecdas2) executes.
//!
//! The witness is built by `ecsm::witness::lincomb2_witness`, which is the spec:
//! every column block below is one of its fields.
//!
//! # `P1` is pinned to the generator `G`
//!
//! `Lincomb2Witness` carries `mem_p2` but **no `mem_p1`**, so a general `P1` is
//! not provable. Instead the 8 doubleword reads at `a1` carry *constant* values
//! (`GENERATOR_LE`), which asserts "memory at `a1` contains exactly G" at zero
//! columns and zero constraints, and makes `P1`'s on-curve-ness a compile-time
//! fact. The executor agrees by construction: `lincomb2_outcome` returns
//! `LINCOMB2_STATUS_P1_NOT_GENERATOR` when the bytes at `a1` are not `G`, so a
//! non-ecrecover caller degrades to the software fallback instead of making the
//! block unprovable.
//!
//! # Two flags, not one: `MU` and `OK`
//!
//! | flag | meaning | gates |
//! |---|---|---|
//! | `MU` | a real lincomb2 ecall happened at this timestamp | the `Ecall` receive and the `x10` read+write (status) |
//! | `OK` | `status == 0`, i.e. the chain is proven | *everything else* — all operand reads, the result write, every range check, every relation, every chain/addend/digit bus |
//!
//! with `IS_BIT` on both and `OK·(1 − MU) = 0`.
//!
//! Two reasons the split is required rather than stylistic:
//!
//! 1. **Bus 19 must balance on the error path.** The CPU sends on `Ecall` for
//!    every ecall, so an unmatched syscall unbalances the bus. There is no chain
//!    to prove when `status != 0`, and the padding trick (`μ = 0`, all columns
//!    zero) would kill the receive as well.
//! 2. **`status = 7` (`P1 != G`) must stay provable.** If the `a1` reads were
//!    `MU`-gated they would assert, on exactly the path where it is false, that
//!    memory at `a1` holds `G`. Gating them by `OK` makes the error row claim
//!    nothing about memory beyond the status write.
//!
//! Soundness of the converse direction — `status == 0` must *oblige* the proof,
//! or a prover sets `OK = 0`, writes `status = 0`, and the guest reads a
//! fabricated `Q` — is carried by two constraints:
//!
//! ```text
//!   OK · STATUS = 0                          (OK = 1  ⇒  status is 0)
//!   MU · (STATUS · S_INV − (1 − OK)) = 0     (OK = 0  ⇒  status is non-zero)
//! ```
//!
//! The witnessed inverse `S_INV` keeps the per-variant error codes distinguishable.
//!
//! # Error and padding rows
//!
//! An error row sets `OK = 0` and every math column to zero, so all convolution
//! and carry-chain relations close at zero carries by exactly the argument
//! padding rows already use. It differs from a padding row only in `MU = 1` and
//! in carrying the real `ADDR_Q`/`STATUS` that the `x10` access binds.
//!
//! # What a dead row can still emit
//!
//! "The columns are zero as generated" is **not** an argument: a malicious
//! prover fills padding rows freely, so every interaction has to be inert by
//! *constraint*. The question to ask of each one is not "is it gated?" but
//! "which column supplies its multiplicity, and what forces that column to
//! zero?". `ecdas2` had a live hole of exactly this shape — its digit sends take
//! their multiplicity from raw `D1`/`D2` columns, and a `MU = 0` row could still
//! fire them, which is worth an arbitrary chosen recovered public key.
//!
//! Audited here interaction by interaction:
//!
//! | interaction | multiplicity | what makes a dead row inert |
//! |---|---|---|
//! | `Ecall` receive | `MU` | balance-forced: the CPU sends exactly one per real ecall, so a spurious `MU = 1` has no matching send |
//! | `x10` read+write | `MU` | same |
//! | all other MEMW, `AreBytes`, `IsHalfword`, `Zero`, `EcT0`, the `Ecdas` seeds/drains, the `sel = 4` Addend publish | `OK` | `OK` is `IS_BIT` and `OK·(1 − MU) = 0`, so it inherits the row above |
//! | 512 `JointBit` receives | `2·u1_bit(i)` / `2·u2_bit(i)` — **raw columns, not `OK`** | idx 517, 518: `(Σ u_bit)·(1 − OK) = 0` |
//! | 3 Addend publishes | `N1`/`N2`/`N3` — **raw columns, not `OK`** | idx 519..=521: `N·(1 − OK) = 0` |
//!
//! The last two rows are the ones that would otherwise be live. Recorded here
//! because this reasoning is expensive to reconstruct and cheap to write down —
//! and because the bug in the sibling chip existed precisely because nobody had.
//!
//! # `len` needs no consumer range check
//!
//! `LEN_M1 = len − 1` keys the `EcT0` send as `LEN_M1 + 1`, and that table has
//! exactly 256 unpadded rows spanning `len ∈ [1, 256]`. A send outside that range
//! matches no row and the LogUp argument cannot balance, so the bound holds by
//! construction — see the `ec_t0` module header, which asks consumers *not* to
//! add a redundant check.

use executor::vm::instruction::execution::{ECSM_LINCOMB2_SYSCALL_NUMBER, GENERATOR_LE};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::ecdas2::{addend_tuple, coord, joint_tuple};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};
use crate::constraints::templates::INV_SHIFT_32;
use ecsm::witness::{JointSel, Lincomb2Witness, T0_X_LE, T0_Y_LE};
use ecsm::{B, N_BYTES, P_BYTES};

// Bias signed convolution carries into IsHalfword [0, 2^16). Same relations as
// the ECSM membership proof, so the same offsets.
pub(crate) use super::ecsm::{CARRY_OFFSET_X2, CARRY_OFFSET_YG};

/// Addend-bus selector values. Never 0: a zero element is skipped by the
/// fingerprint, so a `sel = 0` addend would alias a shorter tuple.
pub const SEL_P1: u64 = 1;
pub const SEL_P2: u64 = 2;
pub const SEL_P12: u64 = 3;
pub const SEL_CORRECTION: u64 = 4;

/// `JointBit` stream tags. Also never 0, for the same reason plus the old-chain
/// `Bit[ts, round]` aliasing described on [`BusId::JointBit`].
pub const STREAM_U1: u64 = 1;
pub const STREAM_U2: u64 = 2;

// =========================================================================
// Column indices (1155 columns; keep in sync with NUM_COLUMNS below)
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;
    /// `a0`: where `xQ‖yQ` is written. This is x10's value *before* the ecall —
    /// the status clobbers it, so both ride the one `x10` read+write access.
    pub const ADDR_Q_0: usize = 2;
    pub const ADDR_Q_1: usize = 3;
    /// `a1`: address of `xP1‖yP1` (asserted to hold `G`).
    pub const ADDR_P1_0: usize = 4;
    pub const ADDR_P1_1: usize = 5;
    /// `a2`: address of `xP2‖yP2`.
    pub const ADDR_P2_0: usize = 6;
    pub const ADDR_P2_1: usize = 7;
    /// `a3`: address of `u1‖u2`.
    pub const ADDR_U_0: usize = 8;
    pub const ADDR_U_1: usize = 9;

    pub const X_P2: usize = 10; // U256BL (32)
    pub const Y_P2: usize = 42; // U256BL (32)

    // `P2` curve-membership sub-witness — the same two convolutions ECSM proves
    // for its generator, applied to the variable point.
    /// `x2 = xP2² mod p`
    pub const MEM_X2: usize = 74; // U256BL (32)
    /// quotient of the `x2` relation
    pub const MEM_Q0: usize = 106; // U256BL (32)
    pub const MEM_C0: usize = 138; // BaseField[64]
    /// quotient of the `y²` relation
    pub const MEM_Q1: usize = 202; // Byte[33]
    pub const MEM_C1: usize = 235; // BaseField[64]

    /// `yP2 < p`
    pub const Y_P2_SUB_P: usize = 299; // U256HL (16 halfwords)

    /// `P12 = P1 + P2`, drained from the phase-0 chain row.
    pub const X_P12: usize = 315; // U256BL (32)
    pub const Y_P12: usize = 347; // U256BL (32)

    /// `u1` as 256 bits, LSB first.
    pub const U1: usize = 379;
    /// `u2` as 256 bits, LSB first.
    pub const U2: usize = 635;
    /// `u1 < N`
    pub const U1_SUB_N: usize = 891; // U256HL (16 halfwords)
    /// `u2 < N`
    pub const U2_SUB_N: usize = 907; // U256HL (16 halfwords)

    /// `len − 1`, the schedule length. Keys the `EcT0` lookup as `LEN_M1 + 1`
    /// and seeds the main chain's first round.
    pub const LEN_M1: usize = 923;

    /// `−2^len·T₀`, received from the preprocessed `EC_T0` table. This is the
    /// NEGATED point, matching both the table and the witness's correction-row
    /// addend — never `Lincomb2Witness::x_t0_pow`/`y_t0_pow`, which hold the
    /// positive `2^len·T₀` and differ from these by a modular negation of `y`.
    pub const X_T0N: usize = 924; // U256BL (32)
    pub const Y_T0N: usize = 956; // U256BL (32)

    /// The accumulator handed from chain phase 1 to chain phase 2. ECSM2
    /// receives the phase-1 drain into these columns and re-sends them as the
    /// phase-2 seed, so the hand-off is a literal relay.
    pub const ACC_X: usize = 988; // U256BL (32)
    pub const ACC_Y: usize = 1020; // U256BL (32)

    /// The result, drained from the phase-2 (correction) chain row.
    pub const X_Q: usize = 1052; // U256BL (32)
    pub const Y_Q: usize = 1084; // U256BL (32)
    /// `xQ < p`, `yQ < p` — load-bearing: the guest keccaks these bytes, so a
    /// `+p`-shifted coordinate hashes to a different address.
    pub const X_Q_SUB_P: usize = 1116; // U256HL (16 halfwords)
    pub const Y_Q_SUB_P: usize = 1132; // U256HL (16 halfwords)

    /// Addend publish counts. Needs no range check: an inflated count leaves an
    /// unmatched send and a "negative" one is unrepresentable, so LogUp balance
    /// pins each to the exact number of receives.
    pub const N1: usize = 1148;
    pub const N2: usize = 1149;
    pub const N3: usize = 1150;

    /// The word written back to `x10`.
    pub const STATUS: usize = 1151;
    /// Witnessed inverse of `STATUS` on error rows.
    pub const S_INV: usize = 1152;
    /// `status == 0`: the full chain is proven on this row.
    pub const OK: usize = 1153;
    /// A real lincomb2 ecall happened at this timestamp.
    pub const MU: usize = 1154;

    pub const NUM_COLUMNS: usize = 1155;

    /// Bit `i` of `u1` (0 = LSB, 255 = MSB).
    #[inline]
    pub const fn u1_bit(i: usize) -> usize {
        U1 + i
    }
    /// Bit `i` of `u2`.
    #[inline]
    pub const fn u2_bit(i: usize) -> usize {
        U2 + i
    }
    #[inline]
    pub const fn mem_c0(i: usize) -> usize {
        MEM_C0 + i
    }
    #[inline]
    pub const fn mem_c1(i: usize) -> usize {
        MEM_C1 + i
    }
    #[inline]
    pub const fn mem_q1(i: usize) -> usize {
        MEM_Q1 + i
    }
    #[inline]
    pub const fn x_q(i: usize) -> usize {
        X_Q + i
    }
    #[inline]
    pub const fn y_q(i: usize) -> usize {
        Y_Q + i
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One lincomb2 ecall: the four operand addresses, the status word written back
/// to `x10`, and — only when the status is `0` — the chip witness.
#[derive(Debug, Clone)]
pub struct Ecsm2Operation {
    pub timestamp: u64,
    pub addr_q: u64,
    pub addr_p1: u64,
    pub addr_p2: u64,
    pub addr_u: u64,
    pub status: u64,
    /// `None` exactly when `status != 0`. The row then proves only the `Ecall`
    /// receive and the status write.
    pub witness: Option<Box<Lincomb2Witness>>,
}

/// How many times each of the three point addends is consumed by the chain.
///
/// The precompute row genuinely adds `P2`, so it counts towards `n2` — the
/// counts are witnessed and balance-forced, so that is not a special case.
pub fn addend_counts(w: &Lincomb2Witness) -> (u64, u64, u64) {
    let mut counts = (0u64, 0u64, 0u64);
    for step in &w.steps {
        match step.sel {
            JointSel::AddP1 => counts.0 += 1,
            JointSel::AddP2 | JointSel::Precompute => counts.1 += 1,
            JointSel::AddP12 => counts.2 += 1,
            JointSel::Double | JointSel::Correction => {}
        }
    }
    counts
}

/// The correction row's addend, i.e. `−2^len·T₀` as the `EC_T0` table stores it.
///
/// Read off the emitted row rather than reconstructed from
/// `x_t0_pow`/`y_t0_pow`, which hold the *positive* blind: `x` agrees but `y` is
/// a modular negation apart, and mixing the two is a silent sign flip.
pub fn correction_addend(w: &Lincomb2Witness) -> ([u8; 32], [u8; 32]) {
    let last = w
        .steps
        .last()
        .expect("lincomb2 witness always emits a correction row");
    debug_assert_eq!(last.sel, JointSel::Correction);
    (last.step.x_g, last.step.y_g)
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

pub fn generate_ecsm2_trace(
    ops: &[Ecsm2Operation],
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
        // Present on every row, error or not: these are what the MU-gated Ecall
        // receive and x10 read+write bind.
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row_idx, cols::ADDR_Q_0, op.addr_q);
        table.set_u64(row_idx, cols::STATUS, op.status);
        table.set_fe(row_idx, cols::MU, FE::one());

        let Some(w) = op.witness.as_deref() else {
            // Error row: OK = 0 and every math column stays zero, so all
            // relations close at zero carries. Only the status inverse is live.
            let inv = FE::from(op.status)
                .inv()
                .expect("error rows carry a non-zero status by construction");
            table.set_fe(row_idx, cols::S_INV, inv);
            continue;
        };

        table.set_fe(row_idx, cols::OK, FE::one());
        table.set_dword_wl(row_idx, cols::ADDR_P1_0, op.addr_p1);
        table.set_dword_wl(row_idx, cols::ADDR_P2_0, op.addr_p2);
        table.set_dword_wl(row_idx, cols::ADDR_U_0, op.addr_u);

        table.set_bytes(row_idx, cols::X_P2, &w.x_p2);
        table.set_bytes(row_idx, cols::Y_P2, &w.y_p2);
        table.set_bytes(row_idx, cols::MEM_X2, &w.mem_p2.x2);
        table.set_bytes(row_idx, cols::MEM_Q0, &w.mem_p2.q0);
        table.set_bytes(row_idx, cols::MEM_Q1, &w.mem_p2.q1);
        for i in 0..64 {
            debug_assert!((0..1 << 16).contains(&(w.mem_p2.c0[i] + CARRY_OFFSET_X2)));
            debug_assert!((0..1 << 16).contains(&(w.mem_p2.c1[i] + CARRY_OFFSET_YG)));
            table.set_fe(row_idx, cols::mem_c0(i), fe_from_i64(w.mem_p2.c0[i]));
            table.set_fe(row_idx, cols::mem_c1(i), fe_from_i64(w.mem_p2.c1[i]));
        }
        write_halfwords(table, row_idx, cols::Y_P2_SUB_P, &w.y_p2_sub_p);

        table.set_bytes(row_idx, cols::X_P12, &w.x_p12);
        table.set_bytes(row_idx, cols::Y_P12, &w.y_p12);

        for b in 0..256 {
            let bit1 = (w.u1[b / 8] >> (b % 8)) & 1;
            let bit2 = (w.u2[b / 8] >> (b % 8)) & 1;
            table.set_fe(row_idx, cols::u1_bit(b), FE::from(bit1 as u64));
            table.set_fe(row_idx, cols::u2_bit(b), FE::from(bit2 as u64));
        }
        write_halfwords(table, row_idx, cols::U1_SUB_N, &w.u1_sub_n);
        write_halfwords(table, row_idx, cols::U2_SUB_N, &w.u2_sub_n);

        debug_assert!((1..=256).contains(&w.len), "len out of the EC_T0 range");
        table.set_u64(row_idx, cols::LEN_M1, (w.len - 1) as u64);

        let (x_t0n, y_t0n) = correction_addend(w);
        table.set_bytes(row_idx, cols::X_T0N, &x_t0n);
        table.set_bytes(row_idx, cols::Y_T0N, &y_t0n);

        // The accumulator entering the correction row: the phase-1 drain that
        // ECSM2 relays into the phase-2 seed.
        let correction = w.steps.last().expect("correction row");
        table.set_bytes(row_idx, cols::ACC_X, &correction.step.x_a);
        table.set_bytes(row_idx, cols::ACC_Y, &correction.step.y_a);

        table.set_bytes(row_idx, cols::X_Q, &w.x_q);
        table.set_bytes(row_idx, cols::Y_Q, &w.y_q);
        write_halfwords(table, row_idx, cols::X_Q_SUB_P, &w.x_q_sub_p);
        write_halfwords(table, row_idx, cols::Y_Q_SUB_P, &w.y_q_sub_p);

        let (n1, n2, n3) = addend_counts(w);
        table.set_u64(row_idx, cols::N1, n1);
        table.set_u64(row_idx, cols::N2, n2);
        table.set_u64(row_idx, cols::N3, n3);
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

/// 32 constant bus elements — a compile-time point coordinate, packed exactly as
/// [`coord`] packs a column-backed one.
fn const_coord(bytes: &[u8]) -> Vec<BusValue> {
    debug_assert_eq!(bytes.len(), 32);
    bytes
        .iter()
        .map(|&b| BusValue::constant(b as u64))
        .collect()
}

/// `[old[8], is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]`
/// — the 24-element MEMW tuple. A plain read passes `old == value`; a combined
/// read+write (the `x10` status access) passes the pre-ecall value as `old`.
#[allow(clippy::too_many_arguments)]
fn memw_read(
    old: [BusValue; 8],
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
    v.extend(old);
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

/// `[is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]` — the
/// 16-element MEMW **write** tuple (MEMW supplies `old`).
fn memw_write(
    value: [BusValue; 8],
    base_lo: BusValue,
    base_hi: BusValue,
    ts_lo: BusValue,
    ts_hi: BusValue,
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
    v.push(BusValue::constant(1)); // w8
    v
}

/// A register value `[lo, hi, 0, 0, 0, 0, 0, 0]` as MEMW value elements.
fn register_value(lo: BusValue, hi: BusValue) -> [BusValue; 8] {
    let mut v: [BusValue; 8] = std::array::from_fn(|_| BusValue::constant(0));
    v[0] = lo;
    v[1] = hi;
    v
}

/// The eight bytes of a 64-byte operand's doubleword `chunk`, taken from two
/// 32-byte column blocks laid out back to back.
fn operand_dword(lo_col: usize, hi_col: usize, chunk: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| {
        let byte = 8 * chunk + b;
        if byte < 32 {
            packed(lo_col + byte)
        } else {
            packed(hi_col + byte - 32)
        }
    })
}

/// The same, for a compile-time 64-byte constant operand.
fn const_operand_dword(bytes: &[u8; 64], chunk: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| BusValue::constant(bytes[8 * chunk + b] as u64))
}

/// Byte `byte_idx` of a bit-decomposed 256-bit scalar: `Σ 2^j · bit[8·idx + j]`.
fn scalar_byte(base: usize, byte_idx: usize) -> BusValue {
    BusValue::linear(
        (0..8)
            .map(|j| LinearTerm::Column {
                coefficient: 1i64 << j,
                column: base + 8 * byte_idx + j,
            })
            .collect(),
    )
}

/// The eight bytes of `u1‖u2`'s doubleword `chunk`, from the bit columns.
fn scalar_dword(chunk: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| {
        let byte = 8 * chunk + b;
        if byte < 32 {
            scalar_byte(cols::U1, byte)
        } else {
            scalar_byte(cols::U2, byte - 32)
        }
    })
}

/// `col + offset` as a bus element (an operand's per-doubleword base address).
fn addr_plus(col: usize, offset: i64) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: col,
        },
        LinearTerm::Constant(offset),
    ])
}

// =========================================================================
// Bus interactions
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let ok = || Multiplicity::Column(cols::OK);
    let ts_lo = || packed(cols::TIMESTAMP_0);
    let ts_hi = || packed(cols::TIMESTAMP_1);
    let mut out = Vec::new();

    // --- MU-gated: the ECALL receive and the status write -------------------

    // ECALL receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            ts_lo(),
            ts_hi(),
            BusValue::constant(ECSM_LINCOMB2_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(ECSM_LINCOMB2_SYSCALL_NUMBER >> 32),
        ],
    ));

    // Combined read+write of x10: old = the result address `a0`, new = STATUS.
    // The COMMIT chip sets the precedent for an accelerator writing a register
    // during its ecall (`commit.rs`, "read+write x10"); ECALL decode's
    // `write_register = false` governs the CPU row's own write path only.
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_read(
            register_value(packed(cols::ADDR_Q_0), packed(cols::ADDR_Q_1)),
            register_value(packed(cols::STATUS), BusValue::constant(0)),
            1,
            BusValue::constant(2 * 10),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
            1,
            0,
        ),
    ));

    // --- OK-gated: everything else ------------------------------------------

    // Register reads of a1/a2/a3.
    for (reg, lo, hi) in [
        (11usize, cols::ADDR_P1_0, cols::ADDR_P1_1),
        (12, cols::ADDR_P2_0, cols::ADDR_P2_1),
        (13, cols::ADDR_U_0, cols::ADDR_U_1),
    ] {
        let value = register_value(packed(lo), packed(hi));
        out.push(BusInteraction::sender(
            BusId::Memw,
            ok(),
            memw_read(
                value.clone(),
                value,
                1,
                BusValue::constant(2 * reg as u64),
                BusValue::constant(0),
                ts_lo(),
                ts_hi(),
                1,
                0,
            ),
        ));
    }

    // Read P1 (8 doublewords at a1 + 8i) with CONSTANT values: this is what
    // pins `P1 = G` with zero columns.
    for i in 0..8 {
        let value = const_operand_dword(&GENERATOR_LE, i);
        out.push(BusInteraction::sender(
            BusId::Memw,
            ok(),
            memw_read(
                value.clone(),
                value,
                0,
                addr_plus(cols::ADDR_P1_0, (8 * i) as i64),
                packed(cols::ADDR_P1_1),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // Read P2 (8 doublewords at a2 + 8i).
    for i in 0..8 {
        let value = operand_dword(cols::X_P2, cols::Y_P2, i);
        out.push(BusInteraction::sender(
            BusId::Memw,
            ok(),
            memw_read(
                value.clone(),
                value,
                0,
                addr_plus(cols::ADDR_P2_0, (8 * i) as i64),
                packed(cols::ADDR_P2_1),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // Read u1‖u2 (8 doublewords at a3 + 8i), reassembled from the bit columns.
    for i in 0..8 {
        let value = scalar_dword(i);
        out.push(BusInteraction::sender(
            BusId::Memw,
            ok(),
            memw_read(
                value.clone(),
                value,
                0,
                addr_plus(cols::ADDR_U_0, (8 * i) as i64),
                packed(cols::ADDR_U_1),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // Write xQ‖yQ (8 doublewords at a0 + 8i). OK-gated: the executor writes
    // nothing on the error path, so there must be no claimed write either.
    for i in 0..8 {
        out.push(BusInteraction::sender(
            BusId::Memw,
            ok(),
            memw_write(
                operand_dword(cols::X_Q, cols::Y_Q, i),
                addr_plus(cols::ADDR_Q_0, (8 * i) as i64),
                packed(cols::ADDR_Q_1),
                ts_lo(),
                ts_hi(),
            ),
        ));
    }

    // ARE_BYTES range checks, paired: one send checks BOTH elements. Only the
    // membership sub-witness needs them — `X_P2`/`Y_P2` are byte-checked at
    // store time (the authority today's `xG`/`k` also rely on), and every other
    // point block is inherited through a keyed tuple.
    // `collect_bitwise_from_ecsm2` mirrors this layout exactly.
    for base in [cols::MEM_X2, cols::MEM_Q0, cols::MEM_Q1] {
        for i in 0..16 {
            out.push(BusInteraction::sender(
                BusId::AreBytes,
                ok(),
                vec![packed(base + 2 * i), packed(base + 2 * i + 1)],
            ));
        }
    }
    out.push(BusInteraction::sender(
        BusId::AreBytes,
        ok(),
        vec![packed(cols::mem_q1(32)), BusValue::constant(0)],
    ));

    // IS_HALF on the shifted membership carries, then the five overflow blocks.
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
            ok(),
            vec![half_offset(cols::mem_c0(i), CARRY_OFFSET_X2)],
        ));
    }
    for i in 0..63 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            ok(),
            vec![half_offset(cols::mem_c1(i), CARRY_OFFSET_YG)],
        ));
    }
    for base in [
        cols::Y_P2_SUB_P,
        cols::U1_SUB_N,
        cols::U2_SUB_N,
        cols::X_Q_SUB_P,
        cols::Y_Q_SUB_P,
    ] {
        for i in 0..16 {
            out.push(BusInteraction::sender(
                BusId::IsHalfword,
                ok(),
                vec![packed(base + i)],
            ));
        }
    }

    // ZERO bus: assert u1 != 0 and u2 != 0. The `< N` overflow witness admits
    // zero (`N + (2^256 − N)` overflows), so this is the check that excludes it.
    for base in [cols::U1, cols::U2] {
        out.push(BusInteraction::sender(
            BusId::Zero,
            ok(),
            vec![
                BusValue::linear(
                    (0..256)
                        .map(|b| LinearTerm::Column {
                            coefficient: 1i64 << (b % 8),
                            column: base + b,
                        })
                        .collect(),
                ),
                BusValue::constant(0), // expected ZERO output = 0 ⇒ input is nonzero
            ],
        ));
    }

    // JointBit receivers, one per (bit position, stream). Multiplicity `2·bit`,
    // NOT `bit`: a set digit is carried by both the round's doubling and its
    // add, and both send. The 2× is what forces the add to exist at all — with
    // only the doubling available the total can never reach 2.
    for (base, stream) in [(cols::U1, STREAM_U1), (cols::U2, STREAM_U2)] {
        for i in 0..256 {
            out.push(BusInteraction::receiver(
                BusId::JointBit,
                Multiplicity::Linear(vec![LinearTerm::Column {
                    coefficient: 2,
                    column: base + i,
                }]),
                vec![
                    ts_lo(),
                    ts_hi(),
                    BusValue::constant(i as u64),
                    BusValue::constant(stream),
                ],
            ));
        }
    }

    // Addend publishes. Every coordinate is pinned somewhere else: G is a
    // constant, P2 is MEMW-bound, P12 is the phase-0 drain, and the correction
    // constant is the EC_T0 lookup. So balance on this bus leaves the chain no
    // free addend anywhere.
    let g_x = || const_coord(&GENERATOR_LE[..32]);
    let g_y = || const_coord(&GENERATOR_LE[32..]);
    for (sel, x, y, mult) in [
        (SEL_P1, g_x(), g_y(), Multiplicity::Column(cols::N1)),
        (
            SEL_P2,
            coord(cols::X_P2),
            coord(cols::Y_P2),
            Multiplicity::Column(cols::N2),
        ),
        (
            SEL_P12,
            coord(cols::X_P12),
            coord(cols::Y_P12),
            Multiplicity::Column(cols::N3),
        ),
        // Exactly one correction row per proven ecall, so its count IS `OK`.
        (SEL_CORRECTION, coord(cols::X_T0N), coord(cols::Y_T0N), ok()),
    ] {
        out.push(BusInteraction::sender(
            BusId::Addend,
            mult,
            addend_tuple(ts_lo(), ts_hi(), BusValue::constant(sel), x, y),
        ));
    }

    // EC_T0 lookup: send a plain `len`; the table's receive key is `LEN_M1 + 1`,
    // so the −1 storage encoding stays entirely on its side.
    {
        let mut values = vec![addr_plus(cols::LEN_M1, 1)];
        values.extend(coord(cols::X_T0N));
        values.extend(coord(cols::Y_T0N));
        out.push(BusInteraction::sender(BusId::EcT0, ok(), values));
    }

    // The three chain segments, each pinned at both ends at multiplicity OK.
    // A chain row can only execute in phase q if some sender published phase q,
    // and the drains are received on the very columns the range checks and the
    // output write bind — so both telescoping breaks are closed at both ends.
    let seed_round = |r: BusValue| r;
    let drain_round = || BusValue::linear(vec![LinearTerm::Constant(-1)]);

    // phase 0 — precompute: a = P1 = G, addend = P2, result = P12.
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        ok(),
        joint_tuple(
            g_x(),
            g_y(),
            BusValue::constant(0),
            seed_round(BusValue::constant(0)),
            BusValue::constant(1), // op = add
            ts_lo(),
            ts_hi(),
        ),
    ));
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        ok(),
        joint_tuple(
            coord(cols::X_P12),
            coord(cols::Y_P12),
            BusValue::constant(0),
            drain_round(),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
        ),
    ));

    // phase 1 — main chain: seeded at the NUMS blind T₀ and round `len − 1`.
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        ok(),
        joint_tuple(
            const_coord(&T0_X_LE),
            const_coord(&T0_Y_LE),
            BusValue::constant(1),
            seed_round(packed(cols::LEN_M1)),
            BusValue::constant(0), // op = double
            ts_lo(),
            ts_hi(),
        ),
    ));
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        ok(),
        joint_tuple(
            coord(cols::ACC_X),
            coord(cols::ACC_Y),
            BusValue::constant(1),
            drain_round(),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
        ),
    ));

    // phase 2 — correction: the phase-1 drain relayed straight back out. A
    // direct chain hand-off is not expressible, because the outgoing tuple pins
    // the successor's `op` to `NB`, which is 0 on the last main row while the
    // correction row is an add.
    out.push(BusInteraction::sender(
        BusId::Ecdas,
        ok(),
        joint_tuple(
            coord(cols::ACC_X),
            coord(cols::ACC_Y),
            BusValue::constant(2),
            seed_round(BusValue::constant(0)),
            BusValue::constant(1), // op = add
            ts_lo(),
            ts_hi(),
        ),
    ));
    out.push(BusInteraction::receiver(
        BusId::Ecdas,
        ok(),
        joint_tuple(
            coord(cols::X_Q),
            coord(cols::Y_Q),
            BusValue::constant(2),
            drain_round(),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
        ),
    ));

    out
}

// =========================================================================
// Constraints
// =========================================================================

/// Which membership convolution relation a carry constraint enforces.
#[derive(Clone, Copy)]
pub enum Relation {
    /// `xP2² − x2 − q0·p = 0`
    X2,
    /// `yP2² + OK·p² − xP2·x2 − OK·b − q1·p = 0`
    Yg,
}

/// The addition-overflow range checks, whose 8 word-carries `c` are virtual:
/// `c_i = 2^-32·(addend0_i + addend1_i + c_{i-1} − sum_i)`. The addition must
/// overflow `2^256` (carry-out `c_7 = 1`), which proves the strict inequality.
#[derive(Clone, Copy)]
pub enum OverflowKind {
    /// `p + yP2_sub_p = yP2 + 2^256`
    Yp2LtP,
    /// `N + u1_sub_N = u1 + 2^256`
    U1LtN,
    U2LtN,
    /// `p + xQ_sub_p = xQ + 2^256`
    XqLtP,
    YqLtP,
}

impl OverflowKind {
    /// The constant addend's 32-bit word `i`.
    fn const_word(self, i: usize) -> u64 {
        let bytes = match self {
            OverflowKind::U1LtN | OverflowKind::U2LtN => &N_BYTES,
            _ => &P_BYTES,
        };
        let mut w = 0u64;
        for b in 0..4 {
            w += (bytes[4 * i + b] as u64) << (8 * b);
        }
        w
    }
    /// Column base of the witnessed halfword addend.
    fn addend_hl_base(self) -> usize {
        match self {
            OverflowKind::Yp2LtP => cols::Y_P2_SUB_P,
            OverflowKind::U1LtN => cols::U1_SUB_N,
            OverflowKind::U2LtN => cols::U2_SUB_N,
            OverflowKind::XqLtP => cols::X_Q_SUB_P,
            OverflowKind::YqLtP => cols::Y_Q_SUB_P,
        }
    }
    /// Column base of the sum.
    fn sum_col_base(self) -> usize {
        match self {
            OverflowKind::Yp2LtP => cols::Y_P2,
            OverflowKind::U1LtN => cols::U1,
            OverflowKind::U2LtN => cols::U2,
            OverflowKind::XqLtP => cols::X_Q,
            OverflowKind::YqLtP => cols::Y_Q,
        }
    }
    /// Whether the sum is stored as 256 individual bits rather than 32 bytes.
    fn sum_is_bits(self) -> bool {
        matches!(self, OverflowKind::U1LtN | OverflowKind::U2LtN)
    }
}

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// Constraint indices 0..=692 (693 total):
//   0        : IS_BIT(MU)
//   1        : IS_BIT(OK)
//   2        : OK · (1 − MU)                        (OK ⇒ MU)
//   3        : OK · STATUS                          (OK ⇒ status is 0)
//   4        : MU · (STATUS·S_INV − (1 − OK))       (¬OK ⇒ status is non-zero)
//   5..=260  : IS_BIT(u1[i])
//   261..=516: IS_BIT(u2[i])
//   517, 518 : (Σ u_bit)·(1 − OK)                   (no phantom JointBit receives)
//   519..=521: N1/N2/N3 · (1 − OK)                  (no phantom Addend publishes)
//   522..=586: ConvCarry(X2, 0..64) + ColIsZero(mem_c0(63))
//   587..=651: ConvCarry(Yg, 0..64) + ColIsZero(mem_c1(63))
//   652      : IS_BIT(mem_q1(32))
//   653..=692: 5 × (7 CarryBit + 1 OverflowRequired)

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// ECSM2 transition constraints as a single-source [`ConstraintSet`] (693
/// total). No column configuration needed (the layout is fixed via `cols`).
pub struct Ecsm2Constraints;

impl Ecsm2Constraints {
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

    /// `bytes[base + j]` for `j < len`, else zero.
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

    /// `S_i` for `relation` at limb `i`. Structurally identical to the ECSM
    /// generator-membership body, with `OK` in place of `µ` as the gate for the
    /// `p²` and `b` constants — so an error row zeroes to `0 = 0` exactly like a
    /// padding row.
    fn s_i<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        relation: Relation,
        i: usize,
    ) -> B::Expr {
        let byte = |base: usize, len: usize, j: usize| Self::byte_at(b, base, len, j);
        let mut s = b.zero();
        match relation {
            Relation::X2 => {
                // Σ xP2_j·xP2_{i-j} − x2_i − Σ q0_j·P_{i-j}
                for j in 0..=i {
                    s = s + byte(cols::X_P2, 32, j) * byte(cols::X_P2, 32, i - j);
                    s = s - byte(cols::MEM_Q0, 32, j) * Self::p_byte_expr(b, i - j);
                }
                s = s - byte(cols::MEM_X2, 32, i);
            }
            Relation::Yg => {
                // Σ (yP2_j·yP2_{i-j} + OK·P_j·P_{i-j} − x2_j·xP2_{i-j} − q1_j·P_{i-j}) − OK·b_i
                let ok = b.main(0, cols::OK);
                let mut p2 = b.zero();
                for j in 0..=i {
                    s = s + byte(cols::Y_P2, 32, j) * byte(cols::Y_P2, 32, i - j);
                    p2 = p2 + Self::p_byte_expr(b, j) * Self::p_byte_expr(b, i - j);
                    s = s - byte(cols::MEM_X2, 32, j) * byte(cols::X_P2, 32, i - j);
                    s = s - byte(cols::MEM_Q1, 33, j) * Self::p_byte_expr(b, i - j);
                }
                s = s + ok.clone() * p2;
                if i == 0 {
                    let curve_b = b.const_base(B);
                    s = s - ok * curve_b;
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
            Relation::X2 => cols::MEM_C0,
            Relation::Yg => cols::MEM_C1,
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

    /// The 8 word-carries of the `kind` addition. Scalars are summed from their
    /// 256 individual bit columns; coordinates from their 32 byte columns.
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
            let mut sum = b.zero();
            if kind.sum_is_bits() {
                for bit in 0..32 {
                    let shift = b.const_base(1u64 << bit);
                    sum = sum + b.main(0, base + 32 * i + bit) * shift;
                }
            } else {
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

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for Ecsm2Constraints {
    // The overflow carry-bit constraints (OK·c·(1−c)) and the status-inverse
    // binding are degree 3.
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0, 1: IS_BIT(MU), IS_BIT(OK).
        for (i, col) in [cols::MU, cols::OK].into_iter().enumerate() {
            let x = b.main(0, col);
            let one = b.one();
            b.emit_base(i, x.clone() * (one - x));
        }

        // idx 2: OK·(1 − MU) = 0. A proven chain implies a real ecall.
        let ok = b.main(0, cols::OK);
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(2, ok * (one - mu));

        // idx 3: OK·STATUS = 0. Claiming the chain forces the status to 0.
        let ok = b.main(0, cols::OK);
        let status = b.main(0, cols::STATUS);
        b.emit_base(3, ok * status);

        // idx 4: MU·(STATUS·S_INV − (1 − OK)) = 0. On a real ecall that does NOT
        // claim the chain, the status must be invertible, i.e. non-zero. This is
        // the constraint that makes `status == 0` *oblige* the proof: without it
        // a prover sets OK = 0, writes status 0, and the guest reads a fabricated
        // Q out of memory. MU-gated so all-zero padding rows are free; the
        // witnessed inverse keeps the per-variant error codes distinguishable.
        let mu = b.main(0, cols::MU);
        let status = b.main(0, cols::STATUS);
        let s_inv = b.main(0, cols::S_INV);
        let one = b.one();
        let ok = b.main(0, cols::OK);
        b.emit_base(4, mu * (status * s_inv - (one - ok)));

        let mut idx = 5;

        // idx 5..=516: IS_BIT on both scalars' 512 bits.
        for base in [cols::U1, cols::U2] {
            for i in 0..256 {
                let x = b.main(0, base + i);
                let one = b.one();
                b.emit_base(idx, x.clone() * (one - x));
                idx += 1;
            }
        }

        // idx 517, 518: (Σ u_bit)·(1 − OK) = 0. Scalar bits are the JointBit
        // receive multiplicities, so a non-OK row with live bits would fire
        // phantom receives.
        for base in [cols::U1, cols::U2] {
            let mut sum = b.zero();
            for i in 0..256 {
                sum = sum + b.main(0, base + i);
            }
            let ok = b.main(0, cols::OK);
            let one = b.one();
            b.emit_base(idx, sum * (one - ok));
            idx += 1;
        }

        // idx 519..=521: the addend publish counts vanish on non-OK rows, for
        // the same reason.
        for col in [cols::N1, cols::N2, cols::N3] {
            let n = b.main(0, col);
            let ok = b.main(0, cols::OK);
            let one = b.one();
            b.emit_base(idx, n * (one - ok));
            idx += 1;
        }

        // P2 membership: x2 convolution (64 carries + closing), then y².
        for (relation, c_base) in [(Relation::X2, cols::MEM_C0), (Relation::Yg, cols::MEM_C1)] {
            for i in 0..64 {
                let root = Self::conv_carry(b, relation, i);
                b.emit_base(idx, root);
                idx += 1;
            }
            let c_last = b.main(0, c_base + 63);
            b.emit_base(idx, c_last);
            idx += 1;
        }

        // idx 652: IS_BIT(mem_q1[32]) — the 33rd quotient byte is a single bit.
        let q1_32 = b.main(0, cols::mem_q1(32));
        let one = b.one();
        b.emit_base(idx, q1_32.clone() * (one - q1_32));
        idx += 1;

        // The five overflow checks: 7 carry bits (deg 3) + overflow-required
        // (deg 2) each, all OK-gated so error and padding rows close at zero.
        for kind in [
            OverflowKind::Yp2LtP,
            OverflowKind::U1LtN,
            OverflowKind::U2LtN,
            OverflowKind::XqLtP,
            OverflowKind::YqLtP,
        ] {
            let c = Self::carry_chain(b, kind);
            for ci in c.iter().take(7) {
                let ok = b.main(0, cols::OK);
                let one = b.one();
                b.emit_base(idx, ok * ci.clone() * (one - ci.clone()));
                idx += 1;
            }
            let ok = b.main(0, cols::OK);
            let one = b.one();
            b.emit_base(idx, ok * (one - c[7].clone()));
            idx += 1;
        }

        debug_assert_eq!(idx, 693);
    }
}
