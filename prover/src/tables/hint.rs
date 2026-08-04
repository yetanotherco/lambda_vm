//! HINT table — receiver for the non-constraining `hint` ecall.
//!
//! The `hint` ecall (syscall `u64::MAX - 30`) lets the executor hand the guest a
//! value that is expensive to compute but cheap to verify (modular inverse, sqrt,
//! …); the guest verifies it with ordinary constrained instructions. Unlike a
//! normal `STORE`, the ecall writes the 32-byte output to guest memory *directly*
//! (not through the CPU load/store decode), so those writes are invisible to the
//! CPU op stream — this table is what puts them into the memory argument.
//!
//! The table therefore does exactly four things, and constrains **nothing** about
//! *which* value was hinted (that is the point — soundness lives in the guest's
//! verify). It does constrain *where* the value lands and that it is 32 bytes:
//!
//! 1. **Receives** the `Hint` ecall on the `Ecall` bus (balances the CPU's send;
//!    a syscall with no receiver leaves the LogUp argument unbalanced).
//! 2. **Reads `x12`** (`a2`) through the memory argument, which pins `out_addr` to
//!    the value the CPU had in that register. The writes below take their base from
//!    an ordinary trace column, so without this read that column is free and the
//!    witness chooses *where* the 32 bytes land — an arbitrary memory write, which
//!    is a strictly larger hole than the unconstrained value.
//! 3. **Sends** the four 8-byte MEMW writes of the output at `out_addr` +0/8/16/24
//!    (received by the MEMW table). Without these the output's initial→final
//!    memory chain is unexplained and the memory argument fails to balance.
//! 4. **Range-checks** the 32 output cells as bytes (`AreBytes`). MEMW does not
//!    range-check what it receives, so each table that writes fresh values into
//!    memory checks its own cells; skipping it lets the witness put arbitrary field
//!    elements where loads and the ALU expect bytes.
//!
//! The input read (the ecall also reads `in_addr`) is intentionally **not** modeled:
//! a read leaves the value unchanged, the guest supplies the input via ordinary
//! stores, and nothing depends on the ecall having re-read it — so omitting it is
//! sound and avoids the mixed-timestamp bookkeeping of a partial-buffer read.
//!
//! `mu` is constrained to a bit (`IS_BIT`, the table's only algebraic constraint) —
//! the same guard every other multiplicity-column table carries (ECSM/ECDAS/COMMIT/
//! STORE/MEMW_R). The `Ecall` bus alone does not establish it: its tuple carries the
//! timestamp, a free column, so the LogUp identity pins only the *sum* of `mu` over
//! the rows sharing a `(ts, syscall)` tuple to the CPU's send — it does not rule out
//! a witness that spreads `mu` across rows with integer weights summing to 1 (a `+1`
//! row plus a `+1`/`-1` pair, each keeping its own `out_addr`, the base the four
//! output writes take). MEMW does NOT catch this: it only ever receives the legal
//! `+1`, while the `-1` cancels an honest STORE on the sender side, so MEMW's own
//! multiplicity constraints stay satisfied and nothing downstream rejects it. The
//! `IS_BIT` on `mu` here is therefore load-bearing -- not a redundant restatement of
//! a check some other table performs.
//!
//! ## Columns (41)
//! - `timestamp[0..1]` (DWordWL): the ecall timestamp `T`
//! - `out_addr[0..1]` (DWordWL): base address of the 32-byte output buffer
//! - `out_bytes[0..31]`: the 32 output bytes (the hint) — **unconstrained**
//! - `mu`: multiplicity flag (1 = real hint call, 0 = padding) — gates every bus
//! - `selector[0..1]` (DWordWL): `a0`, bound to `x10` and range-checked `< 3`
//! - `in_addr[0..1]` (DWordWL): `a1`, bound to `x11`; its low limb is range-checked
//!   so the ecall's input range cannot straddle the 32-bit limb boundary
//!
//! Both address low limbs are range-checked against [`HINT_ADDR_LIMB_BOUND`]; see that
//! constant for why the memory bus alone does not bound `out_addr` tightly enough.

use executor::vm::instruction::execution::HINT_SYSCALL_NUMBER;
use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use crate::constraints::templates::emit_is_bit;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

/// One past the largest valid hint selector (`a0 ∈ {0, 1, 2}` = FIELD_INV / SCALAR_INV /
/// FIELD_SQRT). Re-exported from the executor, which const-asserts that the bound and its
/// `is_valid_hint_selector` set coincide — so the AIR's range-check cannot drift from the
/// set the executor accepts.
pub use executor::vm::instruction::execution::HINT_SELECTOR_BOUND;

/// Bound the low 32-bit limb of `in_addr` and `out_addr` must stay under so the
/// ecall's 32-byte range (`+0..+31`) cannot straddle the 2^32 limb boundary. Mirrors
/// the executor's `addr_limb_ok(addr, 31)`: `(addr % 2^32) + 31 < 2^32`, i.e. the
/// largest accepted limb is `2^32 - 32`.
///
/// Both operands need this explicitly. `in_addr` because it is not on the memory bus
/// at all (the input read is not modelled). `out_addr` because the bus bounds it only
/// to `2^32 - 25`: the write bases are `out_addr_lo + 8i`, so the largest one
/// (`+24`) stops being a canonical limb at `2^32 - 24`, while MEMW's `carry`
/// columns resolve the *bytes* past it correctly. That left a seven-value window
/// (`2^32-31 ..= 2^32-25`) the AIR accepted and the executor rejected with
/// `HintAddressOverflow` — a prover could prove a hint call the VM halts on.
pub const HINT_ADDR_LIMB_BOUND: u64 = (1 << 32) - 31;

pub mod cols {
    /// timestamp[0]: lower 32 bits of the ecall timestamp
    pub const TIMESTAMP_0: usize = 0;
    /// timestamp[1]: upper 32 bits (always 0 — timestamps fit u32)
    pub const TIMESTAMP_1: usize = 1;
    /// out_addr[0]: lower 32 bits of the output base address
    pub const ADDR_OUT_0: usize = 2;
    /// out_addr[1]: upper 32 bits of the output base address
    pub const ADDR_OUT_1: usize = 3;
    /// out_bytes[0..31]: the 32 output bytes, one per column
    pub const OUT: usize = 4;
    /// multiplicity flag (1 = real hint call, 0 = padding)
    pub const MU: usize = 36;
    /// selector[0]: lower 32 bits of `a0` (the hint id)
    pub const SEL_0: usize = 37;
    /// selector[1]: upper 32 bits of `a0`
    pub const SEL_1: usize = 38;
    /// in_addr[0]: lower 32 bits of `a1` (the input base address)
    pub const ADDR_IN_0: usize = 39;
    /// in_addr[1]: upper 32 bits of `a1`
    pub const ADDR_IN_1: usize = 40;

    pub const NUM_COLUMNS: usize = 41;

    /// Column of output byte `i` (0..32).
    #[inline]
    pub const fn out(i: usize) -> usize {
        OUT + i
    }
}

/// One `hint` ecall: the timestamp, the output base address, and the 32 output
/// bytes the executor wrote to guest memory (recomputed by the trace builder).
#[derive(Debug, Clone)]
pub struct HintOperation {
    pub timestamp: u64,
    pub out_addr: u64,
    pub out_bytes: [u8; 32],
    /// `a0` — the hint selector, bound to `x10` and range-checked `< 3`.
    pub hint_id: u64,
    /// `a1` — the input base address, bound to `x11` and low-limb range-checked.
    pub in_addr: u64,
}

/// Generates the HINT trace: one row per hint-ecall call (in program order),
/// `mu = 1`; padding rows are all-zero (`mu = 0`, inert on the bus). Empty (all
/// padding) for programs that make no hint calls.
pub fn generate_hint_trace(
    ops: &[HintOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row, op) in ops.iter().enumerate() {
        debug_assert!(
            op.timestamp <= u32::MAX as u64,
            "HINT timestamp {} exceeds u32",
            op.timestamp
        );
        table.set_dword_wl(row, cols::TIMESTAMP_0, op.timestamp);
        table.set_dword_wl(row, cols::ADDR_OUT_0, op.out_addr);
        table.set_bytes(row, cols::OUT, &op.out_bytes);
        table.set_dword_wl(row, cols::SEL_0, op.hint_id);
        table.set_dword_wl(row, cols::ADDR_IN_0, op.in_addr);
        table.set_fe(row, cols::MU, FE::one());
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

/// The eight output bytes of doubleword `chunk` (`out_bytes[8*chunk .. 8*chunk+7]`)
/// as MEMW value elements.
fn out_dword_bytes(chunk: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| packed(cols::out(8 * chunk + b)))
}

/// A 16-element MEMW **write** tuple (CO25): `[is_register=0, base_lo, base_hi,
/// value[8], ts_lo, ts_hi, w2=0, w4=0, w8=1]`. The MEMW table supplies `old`.
fn memw_write(value: [BusValue; 8], base_lo: BusValue, base_hi: BusValue) -> Vec<BusValue> {
    let mut v = Vec::with_capacity(16);
    v.push(BusValue::constant(0)); // is_register = 0 (memory)
    v.push(base_lo);
    v.push(base_hi);
    v.extend(value);
    v.push(packed(cols::TIMESTAMP_0)); // ts_lo
    v.push(packed(cols::TIMESTAMP_1)); // ts_hi
    v.push(BusValue::constant(0)); // w2
    v.push(BusValue::constant(0)); // w4
    v.push(BusValue::constant(1)); // w8 = 1 (8-byte write)
    v
}

/// A 24-element MEMW **read** tuple (CO24) for a register: `[old[8], is_register=1,
/// base_lo=2*reg, base_hi=0, value[8], ts_lo, ts_hi, w2=1, w4=0, w8=0]`, with
/// `old == value` because a read leaves the register unchanged. Binds `x{reg}` to
/// the `(lo, hi)` column pair at the ecall timestamp.
fn memw_register_read(reg: u64, lo_col: usize, hi_col: usize) -> Vec<BusValue> {
    let value = || [packed(lo_col), packed(hi_col)];
    let mut v = Vec::with_capacity(24);
    v.extend(value()); // old[0..2]
    v.extend(std::iter::repeat_n(BusValue::constant(0), 6)); // old[2..8]
    v.push(BusValue::constant(1)); // is_register = 1
    v.push(BusValue::constant(2 * reg)); // base_address lo
    v.push(BusValue::constant(0)); // base_address hi
    v.extend(value()); // value[0..2] == old
    v.extend(std::iter::repeat_n(BusValue::constant(0), 6)); // value[2..8]
    v.push(packed(cols::TIMESTAMP_0));
    v.push(packed(cols::TIMESTAMP_1));
    v.push(BusValue::constant(1)); // w2 = 1 (register = 2 words)
    v.push(BusValue::constant(0)); // w4
    v.push(BusValue::constant(0)); // w8
    v
}

/// Bus interactions:
/// - **`Ecall` receiver** (mult `mu`): `[timestamp, cast(HINT_SYSCALL_NUMBER,
///   DWordWL)]` — HALT-shaped, balances the CPU's ECALL send.
/// - **MEMW register-read sender** (mult `mu`): binds `out_addr` to `x12`, the
///   ecall's `a2`. Without it the write addresses below are free columns, so a
///   witness could place the output bytes at any address it likes — an arbitrary
///   memory write, independent of whether the hinted *value* is constrained.
/// - **MEMW write senders** (mult `mu`, ×4): the four 8-byte writes of the output
///   at `out_addr` +0/8/16/24, timestamp `T`. Received by the MEMW table.
/// - **`AreBytes` senders** (mult `mu`, ×16): range-check the 32 output cells.
///
/// - **MEMW register-read senders** (mult `mu`, ×2): bind `a0` (`x10`, the selector)
///   and `a1` (`x11`, the input address) to their register columns.
/// - **ALU `LT` senders** (mult `mu`, ×3): assert `selector < 3` and that both
///   `in_addr`'s and `out_addr`'s low limbs are `< 2^32 − 31`, matching the executor's
///   up-front rejections (`HintUnknownSelector`, `HintAddressOverflow`). Without them
///   the AIR would accept hints the executor rejects — a malicious prover could prove
///   an execution the VM would halt on. The value stays unconstrained (the guest
///   verifies it); this only pins the *operands* to the executor's accepted set.
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mu = || Multiplicity::Column(cols::MU);
    let mut out = Vec::with_capacity(27);

    // ECALL receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        mu(),
        vec![
            packed(cols::TIMESTAMP_0),
            packed(cols::TIMESTAMP_1),
            BusValue::constant(HINT_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(HINT_SYSCALL_NUMBER >> 32),
        ],
    ));

    // Bind out_addr to x12 (a2): without this the write base below is a free column.
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_register_read(12, cols::ADDR_OUT_0, cols::ADDR_OUT_1),
    ));

    // Bind a0 (x10 = selector) and a1 (x11 = in_addr). Without these the range-checks
    // below would constrain free columns instead of the registers the CPU held.
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_register_read(10, cols::SEL_0, cols::SEL_1),
    ));
    out.push(BusInteraction::sender(
        BusId::Memw,
        mu(),
        memw_register_read(11, cols::ADDR_IN_0, cols::ADDR_IN_1),
    ));

    // ALU LT: selector < 3 (full 64-bit value), asserting the result is 1. A witness
    // with an out-of-range selector has no matching LT row and unbalances the bus.
    // ALU LT tuple (matching the LT table's receiver): `[lhs_lo, lhs_hi, rhs_lo,
    // rhs_hi, op_encoding, result, 0]` — both operands are two elements (low, high
    // 32-bit words), `op_encoding = LT` for an unsigned non-inverted compare, and
    // `result = 1` asserts the strict inequality holds.
    //
    // selector < 3 (full 64-bit value: SEL_0/SEL_1).
    out.push(BusInteraction::sender(
        BusId::Alu,
        mu(),
        vec![
            BusValue::Packed {
                start_column: cols::SEL_0,
                packing: Packing::DWordWL,
            },
            BusValue::constant(HINT_SELECTOR_BOUND),
            BusValue::constant(0),
            BusValue::constant(alu_op::LT as u64),
            BusValue::constant(1),
            BusValue::constant(0),
        ],
    ));

    // in_addr's and out_addr's low limbs < 2^32 - 31, matching addr_limb_ok(addr, 31).
    // The lhs high word is a literal 0, so only the low limb is compared — exactly the
    // executor's check, which ignores the high limb. `out_addr` needs its own check even
    // though it is on the memory bus: the bus only bounds it to 2^32 - 25 (see
    // HINT_ADDR_LIMB_BOUND), leaving a window the executor rejects.
    for addr_lo in [cols::ADDR_IN_0, cols::ADDR_OUT_0] {
        out.push(BusInteraction::sender(
            BusId::Alu,
            mu(),
            vec![
                packed(addr_lo),
                BusValue::constant(0),
                BusValue::constant(HINT_ADDR_LIMB_BOUND),
                BusValue::constant(0),
                BusValue::constant(alu_op::LT as u64),
                BusValue::constant(1),
                BusValue::constant(0),
            ],
        ));
    }

    // write output: 4 doublewords at out_addr + 8i (timestamp T).
    for i in 0..4 {
        let base_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::ADDR_OUT_0,
            },
            LinearTerm::Constant((8 * i) as i64),
        ]);
        out.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_write(out_dword_bytes(i), base_lo, packed(cols::ADDR_OUT_1)),
        ));
    }

    // ARE_BYTES[out_bytes[2i], out_bytes[2i+1]]: the output cells are free columns
    // that enter memory as MEMW write values, and MEMW range-checks nothing it
    // receives. Every other table that puts fresh values into memory (STORE, KECCAK,
    // ECSM, PAGE) range-checks its own cells for this reason: the value is allowed to
    // be *wrong* here, but it must still be 32 bytes, or the witness can smuggle
    // arbitrary field elements into memory and break the byte decomposition that
    // loads and the ALU depend on. 16 sends, pairing cells as ECSM/KECCAK do.
    for i in 0..16 {
        out.push(BusInteraction::sender(
            BusId::AreBytes,
            mu(),
            vec![packed(cols::out(2 * i)), packed(cols::out(2 * i + 1))],
        ));
    }

    out
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The HINT table's single transition constraint: `mu·(1−mu) = 0`.
///
/// `mu` is the multiplicity gating every one of this table's bus interactions
/// (the `Ecall` receive, the `x12` register read, the four output writes, the
/// 16 byte range-checks). It must be boolean, or a witness could put a non-`{0,1}`
/// value on the `AreBytes`/MEMW sends. The LogUp argument already fixes `mu`'s value
/// via the timestamp-unique `Ecall` tuple, but every other multiplicity-column table
/// bit-constrains its column in-circuit; HINT does the same rather than being the
/// lone exception that relies solely on bus balance.
#[derive(Clone, Copy)]
pub struct HintConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for HintConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: IS_BIT for mu.
        emit_is_bit(b, 0, cols::MU, None);
    }
}
