//! KECCAK core chip — handles ECALL, memory I/O, and delegation to the round chip.
//!
//! One row per keccak permutation. The chip serves two syscalls, selected per
//! row by two mutually exclusive mode bits, `MU = MU_P + MU_S`:
//!
//! * **`MU_P` — permute** ([`KECCAK_SYSCALL_NUMBER`]). One ecall, one row: reads
//!   the 25 lanes of the state at `x10`, sends them to the round chip on the
//!   Keccak bus, receives the output after 24 rounds and writes it back.
//! * **`MU_S` — Keccak-256 of a buffer** ([`KECCAK256_SYSCALL_NUMBER`]). One
//!   ecall of `len` bytes is `⌊len / 136⌋ + 1` rows, one per absorbed block: row
//!   `j` reads block `j` from memory, XORs the padded block into the rate of the
//!   state the previous row's permutation ended in, and sends that to the round
//!   chip; the last row writes the first four lanes of its own output (the
//!   digest) to `x12`.
//!
//! ## The sponge mode
//!
//! The rows of one call are linked by the [`BusId::KeccakSponge`] bus: a row
//! that is not the call's last sends `(T, j + 1, count − 136, src + 136,
//! src_hi, out, output state)` and the next row receives it as `(T, j, count,
//! src, src_hi, out, previous state)`. The previous state's rate lands in
//! `PREV`, its capacity directly in the capacity bytes of `INPUT_STATE`. The
//! first row receives the call on the Ecall bus instead and reads `x10`, `x11`,
//! `x12` (input address, length, output address) through the memory argument.
//!
//! On a sponge row `TIMESTAMP_0` is the call's timestamp `T` and `TIMESTAMP_1`
//! is the block index `j`, so the Keccak-bus key `(T, j)` is distinct for every
//! permutation of every call (a permute row's key is `(T', 0)` at its own
//! ecall's timestamp). Nothing but that key binds a round-chip request to its
//! reply, so its uniqueness is what makes the reply this row's permutation.
//! Every memory access of a sponge row uses the call's own timestamps
//! `(T, 0)` (reads) and `(T + 1, 0)` (the digest writes), never `TIMESTAMP_1`.
//!
//! Which bytes of a block are input is a prefix mask `s`: `s[8k + b] =
//! FULL[k] + PART[k][b]` for `b < 7` and `s[8k + 7] = FULL[k]`. A full lane is
//! one 8-byte read (multiplicity `FULL[k]`), a lane the input ends inside is
//! read a byte at a time (`PART[k][b]`). No byte past the input is read, and
//! its `MSG` cell is forced to 0. The padded block is then a linear form of
//! the columns, `m'[i] = MSG[i] + (s[i−1] − s[i]) + [i = 135]·0x80·LAST`
//! (`s[−1] = 1`): the `0x01` lands right after the input and the `0x80` on
//! byte 135 (`0x81` when the two meet).
//!
//! The byte count `COUNT` starts at `x11` (its high word forced 0), falls by
//! 136 per row, and the last row's mask must hold exactly `COUNT` bytes, at
//! most 135. In integers `len − 136·J` is in `(−2^48, 2^32)` for any chain of
//! `J <` [`SPONGE_MAX_ROWS`] rows, far inside the field, so `COUNT = Σ s` on the
//! last row forces `J = ⌊len/136⌋`. The same bound keeps the block index `j`
//! from wrapping, which the keys' uniqueness and the chain's acyclicity rest on.
//! Both proof shapes stay below it: a block proof holds at most
//! `BLOCK_MAX_KECCAK` tables of at most `2^BLOCK_KECCAK_MAX_VARS` rows (2^29,
//! asserted at compile time in `block_whir`), and a STARK proof one KECCAK table
//! (`TableCounts::validate`) of at most 2^32 rows (the field's two-adicity).
//!
//! ## Column layout (907 columns)
//!
//! | Group          | Size | Description                                    |
//! |----------------|------|------------------------------------------------|
//! | timestamp      |    2 | DWordWL; sponge: `(T, j)`                      |
//! | addr           |    8 | State address as DWordBL (permute)             |
//! | input_state    |  200 | Permutation input bytes [5][5][8]              |
//! | output_state   |  200 | Permutation output bytes [5][5][8]             |
//! | state_ptr      |   75 | Per-lane DWordWHH addresses [25][3] (permute)  |
//! | mu             |    1 | Real row, `MU_P + MU_S`                        |
//! | mu_p, mu_s     |    2 | Mode bits                                      |
//! | first, last    |    2 | First / last row of a sponge call              |
//! | count          |    1 | Input bytes left before this row               |
//! | src, out       |    4 | Block address and digest address, DWordWL      |
//! | src_end        |    2 | Halves of `src_lo + 135`                       |
//! | out_end        |    2 | Halves of `out_lo + 31`                        |
//! | prev           |  136 | Rate of the previous permutation's output      |
//! | msg            |  136 | Block bytes read (0 past the input)            |
//! | full           |   17 | Lane read as one 8-byte access                 |
//! | part           |  119 | Byte `b < 7` of a lane read as a 1-byte access |

use executor::vm::instruction::execution::{
    KECCAK_SYSCALL_NUMBER, KECCAK256_RATE_BYTES, KECCAK256_SYSCALL_NUMBER,
};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::gpack::{TraceForm, WidthHint, generate_main};
use super::is_wh::{IsWhKey, is_whh};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::{AddOperand, INV_SHIFT_32};

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    // addr[8] — state address as 8 bytes (DWordBL)
    pub const ADDR: usize = 2;

    // input_state[5][5][8] = 200 bytes
    pub const INPUT_STATE: usize = ADDR + 8; // 10

    // output_state[5][5][8] = 200 bytes
    pub const OUTPUT_STATE: usize = INPUT_STATE + 200; // 210

    // state_ptr[25][3] = 75 columns (DWordWHH per lane: half_lo, half_mid, word_hi)
    pub const STATE_PTR: usize = OUTPUT_STATE + 200; // 410

    pub const MU: usize = STATE_PTR + 75; // 485

    // ---------------------------------------------------------------------
    // Mode bits (`MU = MU_P + MU_S`)
    // ---------------------------------------------------------------------
    /// Permute row (`keccak_permute`).
    pub const MU_P: usize = MU + 1; // 486
    /// Sponge row (`keccak256`).
    pub const MU_S: usize = MU_P + 1; // 487

    // ---------------------------------------------------------------------
    // Sponge mode
    // ---------------------------------------------------------------------
    /// First row of a call: receives the ecall and reads x10..x12.
    pub const FIRST: usize = MU_S + 1; // 488
    /// Last row of a call: writes the digest, sends no link.
    pub const LAST: usize = FIRST + 1; // 489
    /// Input bytes left before this row's block.
    pub const COUNT: usize = LAST + 1; // 490
    /// Address of this row's block, DWordWL.
    pub const SRC_LO: usize = COUNT + 1; // 491
    pub const SRC_HI: usize = SRC_LO + 1; // 492
    /// Digest address, DWordWL.
    pub const OUT_LO: usize = SRC_HI + 1; // 493
    pub const OUT_HI: usize = OUT_LO + 1; // 494
    /// `SRC_LO + 135` as two halves (the block's last byte stays below 2^32).
    pub const SRC_END: usize = OUT_HI + 1; // 495
    /// `OUT_LO + 31` as two halves (the digest's last byte stays below 2^32).
    pub const OUT_END: usize = SRC_END + 2; // 497
    /// Rate bytes (lanes 0..17) of the previous permutation's output.
    pub const PREV: usize = OUT_END + 2; // 499
    /// The block's bytes as read (0 past the input).
    pub const MSG: usize = PREV + RATE; // 635
    /// `FULL[k]`: lane `k` of the block is input and read as one 8-byte access.
    pub const FULL: usize = MSG + RATE; // 771
    /// `PART[k][b]` (`b < 7`): byte `b` of lane `k` is input and read alone.
    pub const PART: usize = FULL + RATE_LANES; // 788

    pub const NUM_COLUMNS: usize = PART + RATE_LANES * 7; // 907

    /// Bytes of a block (the rate).
    pub const RATE: usize = 136;
    /// Lanes of a block.
    pub const RATE_LANES: usize = 17;

    #[inline]
    pub const fn prev(i: usize) -> usize {
        PREV + i
    }

    #[inline]
    pub const fn msg(i: usize) -> usize {
        MSG + i
    }

    #[inline]
    pub const fn full(lane: usize) -> usize {
        FULL + lane
    }

    /// `PART[lane][byte]`, `byte < 7`.
    #[inline]
    pub const fn part(lane: usize, byte: usize) -> usize {
        PART + lane * 7 + byte
    }

    /// Byte `i` (`0..200`, lane-major: lane `i / 8`, byte `i % 8`) of the
    /// permutation input. Lane `L = x + 5y`, so this is `input_state(x, y, b)`.
    #[inline]
    pub const fn input_byte(i: usize) -> usize {
        INPUT_STATE + i
    }

    /// Byte `i` of the permutation output, as [`input_byte`].
    #[inline]
    pub const fn output_byte(i: usize) -> usize {
        OUTPUT_STATE + i
    }

    // -------------------------------------------------------------------------
    // Index helpers
    // -------------------------------------------------------------------------

    #[inline]
    pub const fn addr(byte: usize) -> usize {
        ADDR + byte
    }

    /// Index into input_state[x][y][byte]
    #[inline]
    pub const fn input_state(x: usize, y: usize, byte: usize) -> usize {
        INPUT_STATE + (x + 5 * y) * 8 + byte
    }

    /// Index into output_state[x][y][byte]
    #[inline]
    pub const fn output_state(x: usize, y: usize, byte: usize) -> usize {
        OUTPUT_STATE + (x + 5 * y) * 8 + byte
    }

    /// Index into state_ptr[lane_idx][limb] (DWordWHH = [half_lo, half_mid, word_hi])
    #[inline]
    pub const fn state_ptr(lane_idx: usize, limb: usize) -> usize {
        STATE_PTR + lane_idx * 3 + limb
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One KECCAK row: one permutation.
///
/// `timestamp` is the row's Keccak-bus key as a `DWordWL`: the ecall's
/// timestamp for a permute row, `T | j << 32` for block `j` of a Keccak-256
/// call at `T` (`TIMESTAMP_0 = T`, `TIMESTAMP_1 = j`). The round chip's rows are
/// built from the same field, so they carry the same key.
#[derive(Debug, Clone)]
pub struct KeccakOperation {
    pub timestamp: u64,
    /// The state address of a permute row; 0 on a sponge row.
    pub state_addr: u64,
    pub input: [u64; 25],
    pub output: [u64; 25],
    /// `Some` on a sponge row (block of a `keccak256` call).
    pub sponge: Option<SpongeRow>,
}

/// What a sponge row holds beyond the permutation.
#[derive(Debug, Clone)]
pub struct SpongeRow {
    /// Address of this row's block (`ptr + 136·j`).
    pub src: u64,
    /// Digest address (`x12`).
    pub out: u64,
    /// Input bytes left before this block (`len − 136·j`).
    pub count: u64,
    pub first: bool,
    pub last: bool,
    /// The block's input bytes, zero past the input.
    pub msg: [u8; 136],
    /// The read mask `s`: `s[i]` = byte `i` of the block is input and read. All
    /// 136 on a row that is not the last, the first `count` on the last.
    pub mask: [bool; 136],
}

impl SpongeRow {
    /// The read mask of a block holding `taken` input bytes.
    pub fn prefix(taken: usize) -> [bool; 136] {
        std::array::from_fn(|i| i < taken)
    }

    /// Input bytes the block reads.
    pub fn taken(&self) -> usize {
        self.mask.iter().filter(|&&s| s).count()
    }

    /// `FULL[k]`: the whole lane is read as one access.
    pub fn full(&self, lane: usize) -> bool {
        self.mask[8 * lane..8 * lane + 8].iter().all(|&s| s)
    }

    /// `PART[k][b]`: byte `b < 7` of lane `k` is read alone (the lane is not whole).
    pub fn part(&self, lane: usize, byte: usize) -> bool {
        !self.full(lane) && self.mask[8 * lane + byte]
    }

    /// The padded block, the chip's linear form `m'_i = msg_i + s_{i−1} − s_i
    /// + [i = 135]·0x80·last` (`s_{−1} = 1`): on a prefix mask, the input bytes,
    /// then on the last row `0x01` right after them and `0x80` on byte 135.
    pub fn padded(&self) -> [u8; 136] {
        std::array::from_fn(|i| {
            let before = if i == 0 { true } else { self.mask[i - 1] };
            let mut v = i32::from(self.msg[i]) + i32::from(before) - i32::from(self.mask[i]);
            if i == cols::RATE - 1 && self.last {
                v += 0x80;
            }
            v as u8
        })
    }
}

impl KeccakOperation {
    /// The previous permutation's output rate on a sponge row: this row's input
    /// with the padded block XORed back out (zero on the first row).
    pub fn sponge_prev(&self, sponge: &SpongeRow) -> [u8; 136] {
        let padded = sponge.padded();
        std::array::from_fn(|i| ((self.input[i / 8] >> (8 * (i % 8))) as u8) ^ padded[i])
    }
}

/// The KECCAK rows of one `keccak256` call at timestamp `t` over `input`, at
/// `ptr` with the digest at `out`: one per permutation, from the executor's own
/// [`executor::vm::instruction::execution::keccak256_permutations`].
pub fn sponge_ops(t: u64, ptr: u64, out: u64, input: &[u8]) -> Vec<KeccakOperation> {
    use executor::vm::instruction::execution::keccak256_permutations;
    let rate = KECCAK256_RATE_BYTES as usize;
    let perms = keccak256_permutations(input);
    let n = perms.len();
    perms
        .into_iter()
        .enumerate()
        .map(|(j, (start, end))| {
            let begin = j * rate;
            let take = input.len().saturating_sub(begin).min(rate);
            let mut msg = [0u8; 136];
            msg[..take].copy_from_slice(&input[begin..begin + take]);
            KeccakOperation {
                timestamp: t | ((j as u64) << 32),
                state_addr: 0,
                input: start,
                output: end,
                sponge: Some(SpongeRow {
                    src: ptr.wrapping_add((j * rate) as u64),
                    out,
                    count: (input.len() - begin) as u64,
                    first: j == 0,
                    last: j + 1 == n,
                    msg,
                    mask: SpongeRow::prefix(take),
                }),
            }
        })
        .collect()
}

/// The `IS_WH` sends of the KECCAK table: one per lane of every permute row, the top
/// 48 bits of `state_ptr[lane] = state_addr + 8·lane` (`keccak:c:range_state_ptr`).
pub fn is_wh_keys(ops: &[KeccakOperation]) -> impl Iterator<Item = IsWhKey> + '_ {
    ops.iter().filter(|op| op.sponge.is_none()).flat_map(|op| {
        (0..25u64).map(move |lane_idx| {
            let ptr = op
                .state_addr
                .checked_add(lane_idx * 8)
                .expect("keccak state address range must be validated by the executor");
            IsWhKey::of_dword(ptr)
        })
    })
}

// =========================================================================
// Trace generation
// =========================================================================

pub fn generate_keccak_trace(
    ops: &[KeccakOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_keccak_trace_as(ops, TraceForm::Wide)
}

/// The widths KECCAK traces needed so far in this process (`tables::gpack`).
static WIDTHS: WidthHint = WidthHint::new();

/// [`generate_keccak_trace`] in `form` (`tables::gpack`).
pub fn generate_keccak_trace_as(
    ops: &[KeccakOperation],
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    generate_keccak_rows_as(ops, ops.len().next_power_of_two().max(4), form)
}

/// Whether [`generate_keccak_rows_as`] in `form` writes its rows packed, with no
/// 64-bit table on the way: in [`TraceForm::Narrow`] once an earlier KECCAK
/// trace of this process left its widths (`tables::gpack`).
pub(crate) fn rows_written_packed(form: TraceForm) -> bool {
    matches!(
        super::gpack::Plan::new(form, &WIDTHS, cols::NUM_COLUMNS),
        super::gpack::Plan::Write(_)
    )
}

/// A KECCAK table of `num_rows` rows in `form`: `ops` (at most `num_rows`, a row
/// each, one permutation call a row), then the padding rows. A row reads its own op alone and
/// the padding rows are constants, so rows `[k·R, (k+1)·R)` of the whole padded
/// table are this over the ops in that range with `num_rows = R`: the block's
/// cut of the whole table, built on its own.
pub fn generate_keccak_rows_as(
    ops: &[KeccakOperation],
    num_rows: usize,
    form: TraceForm,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    generate_main!(form, &WIDTHS, num_rows, cols::NUM_COLUMNS, |table| {
        for (row_idx, op) in ops.iter().enumerate() {
            // Timestamp (sponge: `(T, j)`)
            table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

            // Input and output state as bytes
            for x in 0..5 {
                for y in 0..5 {
                    table.set_dword_bl(row_idx, cols::input_state(x, y, 0), op.input[x + 5 * y]);
                    table.set_dword_bl(row_idx, cols::output_state(x, y, 0), op.output[x + 5 * y]);
                }
            }
            table.set_fe(row_idx, cols::MU, FE::one());

            let Some(sponge) = &op.sponge else {
                // Permute row.
                table.set_dword_bl(row_idx, cols::addr(0), op.state_addr);
                // State pointers: state_ptr[lane] = addr + 8 * lane_idx
                for lane_idx in 0..25 {
                    let ptr = op
                        .state_addr
                        .checked_add(lane_idx as u64 * 8)
                        .expect("keccak state address range must be validated by the executor");
                    table.set_dword_whh(row_idx, cols::state_ptr(lane_idx, 0), ptr);
                }
                table.set_fe(row_idx, cols::MU_P, FE::one());
                continue;
            };

            // Sponge row. `ADDR` and `STATE_PTR` stay 0: their constraints and
            // sends are gated by `MU_P`.
            table.set_fe(row_idx, cols::MU_S, FE::one());
            table.set_bool(row_idx, cols::FIRST, sponge.first);
            table.set_bool(row_idx, cols::LAST, sponge.last);
            table.set_u64(row_idx, cols::COUNT, sponge.count);
            table.set_dword_wl(row_idx, cols::SRC_LO, sponge.src);
            table.set_dword_wl(row_idx, cols::OUT_LO, sponge.out);
            let src_end = (sponge.src & 0xFFFF_FFFF) + 135;
            table.set_u64(row_idx, cols::SRC_END, src_end & 0xFFFF);
            table.set_u64(row_idx, cols::SRC_END + 1, src_end >> 16);
            if sponge.last {
                let out_end = (sponge.out & 0xFFFF_FFFF) + 31;
                table.set_u64(row_idx, cols::OUT_END, out_end & 0xFFFF);
                table.set_u64(row_idx, cols::OUT_END + 1, out_end >> 16);
            }
            table.set_bytes(row_idx, cols::PREV, &op.sponge_prev(sponge));
            table.set_bytes(row_idx, cols::MSG, &sponge.msg);
            for lane in 0..cols::RATE_LANES {
                table.set_bool(row_idx, cols::full(lane), sponge.full(lane));
                for byte in 0..7 {
                    table.set_bool(row_idx, cols::part(lane, byte), sponge.part(lane, byte));
                }
            }
        }

        // Padding rows: state_ptr[lane][0] = 8 * lane_idx (per spec keccak.toml pad).
        // half_mid and word_hi stay zero since 8*24 = 192 fits in half_lo.
        // mu = 0 gates all bus interactions and the ADD constraint, so these values
        // only need to satisfy the pad requirement, not reconstruct a real address.
        for row_idx in n..num_rows {
            for lane_idx in 0..25 {
                table.set_u64(row_idx, cols::state_ptr(lane_idx, 0), (lane_idx as u64) * 8);
            }
        }
    })
}

// =========================================================================
// Bus interactions
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = KECCAK_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_SYSCALL_NUMBER >> 32;
    let mut interactions = Vec::with_capacity(84 + SPONGE_INTERACTIONS);

    // 1. ECALL receiver (shared bus, per spec keccak:c:output)
    // Payload: [ts_lo, ts_hi, syscall_lo32, syscall_hi32] in DWordWL [lo, hi]
    // ordering, matching the CPU ECALL sender shared with HALT/COPY/ECSM/HINT.
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::Column(cols::MU_P),
        vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::constant(syscall_lo),
            BusValue::constant(syscall_hi),
        ],
    ));

    // 2. MEMW read_addr: read register x10 to bind addr (per spec keccak:c:read_addr)
    // Format: [old[8], is_register=1, base_addr=[20,0], value[8], ts, ts_hi, write2=1, write4=0, write8=0]
    // For register read: old = value = addr as WL + 6 zeros
    {
        // addr as DWordWL from DWordBL bytes: lo32 = sum(addr[0..4] * 256^i), hi32 = sum(addr[4..8] * 256^i)
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::addr(0),
            },
            LinearTerm::Column {
                coefficient: 256,
                column: cols::addr(1),
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: cols::addr(2),
            },
            LinearTerm::Column {
                coefficient: 16777216,
                column: cols::addr(3),
            },
        ]);
        let addr_hi = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::addr(4),
            },
            LinearTerm::Column {
                coefficient: 256,
                column: cols::addr(5),
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: cols::addr(6),
            },
            LinearTerm::Column {
                coefficient: 16777216,
                column: cols::addr(7),
            },
        ]);
        let mut values = Vec::with_capacity(24);
        // old[0..7] = addr as WL + 6 zeros
        values.push(addr_lo.clone());
        values.push(addr_hi.clone());
        for _ in 2..8 {
            values.push(BusValue::constant(0));
        }
        // is_register = 1
        values.push(BusValue::constant(1));
        // base_address = 2*10 = 20 (register x10)
        values.push(BusValue::constant(20));
        values.push(BusValue::constant(0));
        // value[0..7] = same as old (read)
        values.push(addr_lo);
        values.push(addr_hi);
        for _ in 2..8 {
            values.push(BusValue::constant(0));
        }
        // timestamp
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        });
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        // write2=1, write4=0, write8=0 (register access)
        values.push(BusValue::constant(1));
        values.push(BusValue::constant(0));
        values.push(BusValue::constant(0));
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU_P),
            values,
        ));
    }

    // 2. Keccak bus: send (timestamp, 0, input_state[200])
    // Per spec keccak.toml: input = ["timestamp", 0, "input_state"] where
    // input_state is [[[Byte, 8], 5], 5] — 200 Byte elements, each its own
    // bus element (no packing).
    {
        let mut values = vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::constant(0), // round = 0
        ];
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    values.push(BusValue::Packed {
                        start_column: cols::input_state(x, y, b),
                        packing: Packing::Direct,
                    });
                }
            }
        }
        interactions.push(BusInteraction::sender(
            BusId::Keccak,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // 3. Keccak bus: receive (timestamp, 24, output_state[200])
    {
        let mut values = vec![
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            BusValue::constant(24), // round = 24
        ];
        for x in 0..5 {
            for y in 0..5 {
                for b in 0..8 {
                    values.push(BusValue::Packed {
                        start_column: cols::output_state(x, y, b),
                        packing: Packing::Direct,
                    });
                }
            }
        }
        interactions.push(BusInteraction::receiver(
            BusId::Keccak,
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    // 4. IS_WHH range checks on state_ptr (keccak:c:range_state_ptr): IS_WH on the
    // top 48 bits and IS_HALF on half_lo, 50 interactions. `MU_P` is a bit, as the
    // template's `cond: Bit` requires.
    for lane_idx in 0..25 {
        interactions.extend(is_whh(
            cols::state_ptr(lane_idx, 0),
            Multiplicity::Column(cols::MU_P),
        ));
    }

    // 5. Alignment: addr[0] & 7 = 0, which enforces addr % 8 == 0.
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Column(cols::MU_P),
        vec![
            BusValue::constant(alu_op::AND as u64),
            BusValue::Packed {
                start_column: cols::addr(0),
                packing: Packing::Direct,
            },
            BusValue::constant(7),
            BusValue::constant(0),
        ],
    ));

    // 6. Range-check every addr byte (4 ARE_BYTES pairs). The addr columns are
    // reconstructed as a linear combination (addr_lo = b0 + 256*b1 + 65536*b2 +
    // 2^24*b3, etc.) for the MEMW lookup and the no-overflow / alignment
    // constraints. Without an explicit byte range check on each cell, an
    // attacker can keep the field-element value of that linear combination
    // correct while encoding arbitrary non-byte values in the individual cells
    // (e.g. addr[0]=0, addr[1]=V_lo * 256^{-1} mod p), bypassing the alignment
    // check. Spec emits 8 IS_BYTE templates; we merge `(addr[2i], addr[2i+1])`.
    for i in 0..4 {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU_P),
            vec![
                BusValue::Packed {
                    start_column: cols::addr(2 * i),
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::addr(2 * i + 1),
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // 7. MEMW interactions: 25 combined read+write per lane (per spec)
    // Format: [old[8], is_register, addr_lo32, addr_hi32, value[8], ts[2], w2, w4, w8] = 24
    // old = input_state (read), value = output_state (write)
    for lane_idx in 0..25 {
        let x = lane_idx % 5;
        let y = lane_idx / 5;

        // Address as DWordWL: lo32 = half_lo + 2^16*half_mid, hi32 = word_hi
        let addr_lo = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::state_ptr(lane_idx, 0),
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: cols::state_ptr(lane_idx, 1),
            },
        ]);
        let addr_hi = BusValue::Packed {
            start_column: cols::state_ptr(lane_idx, 2),
            packing: Packing::Direct,
        };

        let mut values = Vec::with_capacity(24);
        // old[0..8] = input_state bytes (the value being read)
        for b in 0..8 {
            values.push(BusValue::Packed {
                start_column: cols::input_state(x, y, b),
                packing: Packing::Direct,
            });
        }
        // is_register = 0
        values.push(BusValue::constant(0));
        // address as DWordWL
        values.push(addr_lo);
        values.push(addr_hi);
        // value[0..8] = output_state bytes (the value being written)
        for b in 0..8 {
            values.push(BusValue::Packed {
                start_column: cols::output_state(x, y, b),
                packing: Packing::Direct,
            });
        }
        // timestamp
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        });
        values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        // write2=0, write4=0, write8=1
        values.push(BusValue::constant(0));
        values.push(BusValue::constant(0));
        values.push(BusValue::constant(1));

        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU_P),
            values,
        ));
    }

    interactions.extend(sponge_bus_interactions());
    interactions
}

/// The most KECCAK rows one proof may hold for the sponge mode's counting
/// argument (module docs): with `N` rows, `len − 136·J` stays in
/// `(−136·N, 2^32)`, which must not reach `p ≈ 2^64`; and the block index must
/// not wrap. 2^40 leaves a margin of 2^16 over what either needs. A cap that
/// lets one proof hold more KECCAK rows has to stay under this; `block_whir`
/// asserts its own at compile time.
pub const SPONGE_MAX_ROWS: u128 = 1 << 40;

/// Interactions of the sponge mode, after the 84 of the permute mode.
pub const SPONGE_INTERACTIONS: usize = 1 + 3 + 2 + RATE_READS + cols::RATE + 4 + 4;

/// Memory reads of a block: 17 lane reads and 7 byte reads per lane.
const RATE_READS: usize = cols::RATE_LANES * 8;

fn direct(column: usize) -> BusValue {
    BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    }
}

fn column_term(column: usize, coefficient: i64) -> LinearTerm {
    LinearTerm::Column {
        coefficient,
        column,
    }
}

/// `column + constant`.
fn offset(column: usize, constant: i64) -> BusValue {
    BusValue::linear(vec![column_term(column, 1), LinearTerm::Constant(constant)])
}

/// The read mask `s_i` as linear terms with `coefficient` (`s_{8k+b} =
/// FULL_k + PART_{k,b}` for `b < 7`, `s_{8k+7} = FULL_k`). `i = -1` is the
/// constant 1.
fn mask_terms(i: isize, coefficient: i64) -> Vec<LinearTerm> {
    if i < 0 {
        return vec![LinearTerm::Constant(coefficient)];
    }
    let i = i as usize;
    let (lane, byte) = (i / 8, i % 8);
    let mut terms = vec![column_term(cols::full(lane), coefficient)];
    if byte < 7 {
        terms.push(column_term(cols::part(lane, byte), coefficient));
    }
    terms
}

/// The padded message byte `m'_i = MSG_i + (s_{i-1} − s_i) + [i = 135]·0x80·LAST`.
pub(crate) fn padded_byte(i: usize) -> BusValue {
    let mut terms = vec![column_term(cols::msg(i), 1)];
    terms.extend(mask_terms(i as isize - 1, 1));
    terms.extend(mask_terms(i as isize, -1));
    if i == cols::RATE - 1 {
        terms.push(column_term(cols::LAST, 0x80));
    }
    BusValue::linear(terms)
}

/// A MEMW register read of `x{reg}` at `(T, 0)` binding `(lo, hi)`.
fn sponge_register_read(reg: u64, lo: BusValue, hi: BusValue) -> Vec<BusValue> {
    let zeros = || std::iter::repeat_n(BusValue::constant(0), 6);
    let mut v = Vec::with_capacity(24);
    v.push(lo.clone());
    v.push(hi.clone());
    v.extend(zeros()); // old[2..8]
    v.push(BusValue::constant(1)); // is_register
    v.push(BusValue::constant(2 * reg));
    v.push(BusValue::constant(0));
    v.push(lo);
    v.push(hi);
    v.extend(zeros()); // value[2..8]
    v.push(direct(cols::TIMESTAMP_0));
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(1)); // w2: a register
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(0));
    v
}

/// A MEMW memory read at `(T, 0)` of `bytes` (1 or 8 of them) at `address_lo`
/// (`SRC_LO + offset`) and `SRC_HI`: old = value = the bytes.
fn sponge_memory_read(offset_lo: i64, bytes: &[usize]) -> Vec<BusValue> {
    let value = || {
        let mut v: Vec<BusValue> = bytes.iter().map(|&c| direct(c)).collect();
        v.resize_with(8, || BusValue::constant(0));
        v
    };
    let mut v = Vec::with_capacity(24);
    v.extend(value()); // old
    v.push(BusValue::constant(0)); // is_register
    v.push(offset(cols::SRC_LO, offset_lo));
    v.push(direct(cols::SRC_HI));
    v.extend(value()); // value == old
    v.push(direct(cols::TIMESTAMP_0));
    v.push(BusValue::constant(0));
    v.push(BusValue::constant(0)); // w2
    v.push(BusValue::constant(0)); // w4
    v.push(BusValue::constant(u64::from(bytes.len() == 8))); // w8
    v
}

/// The digest address on the link between rows (`SpongeCheck::LinkOut` leaves it
/// out in a test).
fn link_out() -> [BusValue; 2] {
    if dropped(SpongeCheck::LinkOut) {
        [BusValue::constant(0), BusValue::constant(0)]
    } else {
        [direct(cols::OUT_LO), direct(cols::OUT_HI)]
    }
}

/// The sponge mode's checks by what they stop, so a test can leave one out and
/// show the forgery it stops then passes (`keccak256_tests`). Outside the tests
/// [`dropped`] is always `false`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SpongeCheck {
    /// `μ = MU_P + MU_S` (idx 56).
    ModeSum,
    /// `(FIRST + LAST)(1 − MU_S) = 0` (idx 57).
    FlagsOnSponge,
    /// `FULL`, `PART` bits.
    MaskBits,
    /// `FULL_k · PART_{k,b} = 0`.
    OneRead,
    /// The mask is a prefix.
    Prefix,
    /// No read off a sponge row.
    ReadsOnSponge,
    /// A row that is not last reads a whole block.
    FullUnlessLast,
    /// The last block holds at most 135 bytes.
    LastNotFull,
    /// The last block holds exactly `COUNT` bytes.
    CountOnLast,
    /// Bytes past the input are 0.
    MsgZero,
    /// The first row is block 0.
    FirstBlock,
    /// The first row starts from the zero state.
    FirstState,
    /// `SRC_END` and `OUT_END` hold the addresses' ends.
    AddressHalves,
    /// The digest address rides the link between rows.
    LinkOut,
    /// Lane 0's 8-byte memory read.
    LaneRead,
}

#[cfg(test)]
thread_local! {
    /// The check a test leaves out ([`SpongeCheck`]).
    pub(crate) static DROPPED: std::cell::Cell<Option<SpongeCheck>> =
        const { std::cell::Cell::new(None) };
    /// A test's forgery of a call's rows before the walk emits their memory
    /// traffic (`trace_builder::collect_keccak256_ops`).
    #[allow(clippy::type_complexity)]
    pub(crate) static FORGE: std::cell::RefCell<Option<Box<dyn Fn(&mut Vec<KeccakOperation>)>>> =
        const { std::cell::RefCell::new(None) };
}

/// Whether a test left `check` out; never outside the tests.
#[inline(always)]
pub(crate) fn dropped(check: SpongeCheck) -> bool {
    #[cfg(test)]
    {
        DROPPED.with(|d| d.get() == Some(check))
    }
    #[cfg(not(test))]
    {
        let _ = check;
        false
    }
}

/// A test's forgery of one call's rows (see [`FORGE`]).
#[cfg(test)]
pub(crate) fn forge(rows: &mut Vec<KeccakOperation>) {
    FORGE.with(|f| {
        if let Some(f) = f.borrow().as_ref() {
            f(rows)
        }
    });
}

/// Recomputes a call's permutations from its rows' masks and messages, starting
/// from `start`: a test's forgery made consistent with the round chip.
#[cfg(test)]
pub(crate) fn rechain(rows: &mut [KeccakOperation], start: [u64; 25]) {
    use executor::vm::instruction::execution::keccak_f1600;
    let mut state = start;
    for row in rows {
        let padded = row.sponge.as_ref().expect("a sponge row").padded();
        for (lane, chunk) in state.iter_mut().zip(padded.chunks_exact(8)) {
            *lane ^= u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
        }
        row.input = state;
        keccak_f1600(&mut state);
        row.output = state;
    }
}

/// The sponge mode's interactions (see the module docs).
///
/// - Ecall receive (`FIRST`): `[T, 0, KECCAK256 lo, hi]`.
/// - x10, x11, x12 read at `(T, 0)` (`FIRST`): `(SRC_LO, SRC_HI)`, `(COUNT, 0)`,
///   `(OUT_LO, OUT_HI)`. The 0 high word of x11 bounds the length below 2^32.
/// - KeccakSponge send (`MU_S − LAST`) and receive (`MU_S − FIRST`).
/// - 17 lane reads (`FULL_k`) and 119 byte reads (`PART_{k,b}`) at `(T, 0)`.
/// - 136 XORs (`MU_S`): `INPUT_STATE_i = PREV_i ⊕ m'_i` over the rate.
/// - IS_HALF on `SRC_END` (`MU_S`) and `OUT_END` (`LAST`).
/// - The digest: lanes 0..4 of `OUTPUT_STATE` written at `OUT + 8k`, `(T + 1, 0)`
///   (`LAST`).
pub fn sponge_bus_interactions() -> Vec<BusInteraction> {
    let mut out = Vec::with_capacity(SPONGE_INTERACTIONS);
    let first = || Multiplicity::Column(cols::FIRST);
    let last = || Multiplicity::Column(cols::LAST);
    let mu_s = || Multiplicity::Column(cols::MU_S);

    // Ecall receive on the call's first row.
    out.push(BusInteraction::receiver(
        BusId::Ecall,
        first(),
        vec![
            direct(cols::TIMESTAMP_0),
            BusValue::constant(0),
            BusValue::constant(KECCAK256_SYSCALL_NUMBER & 0xFFFF_FFFF),
            BusValue::constant(KECCAK256_SYSCALL_NUMBER >> 32),
        ],
    ));

    // The operands, read on the first row.
    for (reg, lo, hi) in [
        (10, direct(cols::SRC_LO), direct(cols::SRC_HI)),
        (11, direct(cols::COUNT), BusValue::constant(0)),
        (12, direct(cols::OUT_LO), direct(cols::OUT_HI)),
    ] {
        out.push(BusInteraction::sender(
            BusId::Memw,
            first(),
            sponge_register_read(reg, lo, hi),
        ));
    }

    // The link to the next row: (T, j + 1, COUNT − 136, SRC_END + 1, SRC_HI,
    // OUT_LO, OUT_HI, OUTPUT_STATE).
    {
        let mut v = Vec::with_capacity(7 + 200);
        v.push(direct(cols::TIMESTAMP_0));
        v.push(offset(cols::TIMESTAMP_1, 1));
        v.push(offset(cols::COUNT, -(cols::RATE as i64)));
        v.push(BusValue::linear(vec![
            column_term(cols::SRC_END, 1),
            column_term(cols::SRC_END + 1, 1 << 16),
            LinearTerm::Constant(1),
        ]));
        v.push(direct(cols::SRC_HI));
        v.extend(link_out());
        v.extend((0..200).map(|i| direct(cols::output_byte(i))));
        out.push(BusInteraction::sender(
            BusId::KeccakSponge,
            Multiplicity::Diff(cols::MU_S, cols::LAST),
            v,
        ));
    }
    // ... and from the previous one: the previous output's rate into PREV, its
    // capacity straight into this permutation's input.
    {
        let mut v = Vec::with_capacity(7 + 200);
        v.push(direct(cols::TIMESTAMP_0));
        v.push(direct(cols::TIMESTAMP_1));
        v.push(direct(cols::COUNT));
        v.push(direct(cols::SRC_LO));
        v.push(direct(cols::SRC_HI));
        v.extend(link_out());
        v.extend((0..cols::RATE).map(|i| direct(cols::prev(i))));
        v.extend((cols::RATE..200).map(|i| direct(cols::input_byte(i))));
        out.push(BusInteraction::receiver(
            BusId::KeccakSponge,
            Multiplicity::Diff(cols::MU_S, cols::FIRST),
            v,
        ));
    }

    // The block's reads: a whole lane at once, or the bytes of the lane the
    // input ends inside one at a time. Byte 7 of a lane is never read alone.
    for lane in 0..cols::RATE_LANES {
        let bytes: Vec<usize> = (0..8).map(|b| cols::msg(8 * lane + b)).collect();
        // A test leaves lane 0's read out with `SpongeCheck::LaneRead` (a zero
        // multiplicity, so the interaction count stays).
        let multiplicity = if lane == 0 && dropped(SpongeCheck::LaneRead) {
            Multiplicity::Linear(vec![LinearTerm::Constant(0)])
        } else {
            Multiplicity::Column(cols::full(lane))
        };
        out.push(BusInteraction::sender(
            BusId::Memw,
            multiplicity,
            sponge_memory_read(8 * lane as i64, &bytes),
        ));
        for byte in 0..7 {
            out.push(BusInteraction::sender(
                BusId::Memw,
                Multiplicity::Column(cols::part(lane, byte)),
                sponge_memory_read((8 * lane + byte) as i64, &[cols::msg(8 * lane + byte)]),
            ));
        }
    }

    // The absorb: INPUT_STATE_i = PREV_i XOR m'_i over the rate.
    for i in 0..cols::RATE {
        out.push(BusInteraction::sender(
            BusId::ByteAlu,
            mu_s(),
            vec![
                BusValue::constant(alu_op::XOR as u64),
                direct(cols::prev(i)),
                padded_byte(i),
                direct(cols::input_byte(i)),
            ],
        ));
    }

    // Address range checks: the block's last byte and the digest's last byte
    // stay below 2^32 in the low address word.
    for half in 0..2 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu_s(),
            vec![direct(cols::SRC_END + half)],
        ));
    }
    for half in 0..2 {
        out.push(BusInteraction::sender(
            BusId::IsHalfword,
            last(),
            vec![direct(cols::OUT_END + half)],
        ));
    }

    // The digest: lanes 0..4 of this row's output, at T + 1.
    for lane in 0..4 {
        let mut v = Vec::with_capacity(16);
        v.push(BusValue::constant(0)); // is_register
        v.push(offset(cols::OUT_LO, 8 * lane as i64));
        v.push(direct(cols::OUT_HI));
        v.extend((0..8).map(|b| direct(cols::output_byte(8 * lane + b))));
        v.push(offset(cols::TIMESTAMP_0, 1));
        v.push(BusValue::constant(0));
        v.push(BusValue::constant(0)); // w2
        v.push(BusValue::constant(0)); // w4
        v.push(BusValue::constant(1)); // w8
        out.push(BusInteraction::sender(BusId::Memw, last(), v));
    }

    debug_assert_eq!(out.len(), SPONGE_INTERACTIONS);
    out
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The KECCAK core table's transition constraints as a single [`ConstraintSet`].
///
/// Permute mode (as before the sponge mode, now gated on `MU_P`):
/// - idx 0-49: for `lane_idx ∈ 0..25`, the `ADD` carry pair (gated on `MU_P`)
///   enforcing `state_ptr[lane] = addr + 8·lane_idx` (`addr` DWordBL,
///   `state_ptr` DWordWHH);
/// - idx 50:   `MU_P · carry_1 = 0` (top-lane no-overflow), where `carry_1` is
///   the high carry of `addr + 192 = state_ptr[24]`;
/// - idx 51:   `μ · (1 − μ) = 0` (MuIsBit).
///
/// Modes (idx 52-57): `MU_P`, `MU_S`, `FIRST`, `LAST` bits; `μ = MU_P + MU_S`
/// (with idx 51 a row is in at most one mode); `FIRST` and `LAST` only on
/// sponge rows. Every multiplicity of the permute mode is `MU_P` and every one
/// of the sponge mode is `FIRST`, `LAST`, `MU_S` (or a difference of these with
/// `MU_S`) or a mask bit that idx 58+ forces to 0 off a sponge row, so a row
/// sends nothing of the mode it is not in.
///
/// Sponge mode, from idx 58 (see [`SPONGE_CONSTRAINTS`] for the count):
/// - `FULL_k`, `PART_{k,b}` bits; `FULL_k · PART_{k,b} = 0` (each `s_i` a bit);
/// - prefix: `(1 − s_i) · s_{i+1} = 0`;
/// - no read off a sponge row: `(1 − MU_S) · s_0 = 0`;
/// - a row that is not last reads a whole block: `(MU_S − LAST)(1 − FULL_16) = 0`;
/// - the last block holds at most 135 input bytes: `LAST · FULL_16 = 0`;
/// - the last block holds exactly `COUNT` bytes: `LAST · (COUNT − Σ s_i) = 0`;
/// - bytes past the input are 0: `(1 − s_i) · MSG_i = 0`;
/// - the first row starts from the zero state at block 0:
///   `FIRST · TIMESTAMP_1 = 0`, `FIRST · PREV_i = 0`, `FIRST · INPUT_i = 0` on
///   the 64 capacity bytes;
/// - `MU_S · (SRC_LO + 135 − SRC_END_0 − 2^16 SRC_END_1) = 0`,
///   `LAST · (OUT_LO + 31 − OUT_END_0 − 2^16 OUT_END_1) = 0`.
#[derive(Clone, Copy)]
pub struct KeccakConstraints;

/// Constraints of the permute mode and the mode bits (idx 0-57).
const MODE_CONSTRAINTS: usize = 58;

/// Constraints of the sponge mode (idx 58..).
pub const SPONGE_CONSTRAINTS: usize = cols::RATE_LANES * 8 // FULL, PART bits
    + cols::RATE_LANES * 7 // FULL · PART
    + (cols::RATE - 1) // prefix
    + 1 // no read off a sponge row
    + 1 // full block unless last
    + 1 // last block < 136
    + 1 // count on the last row
    + cols::RATE // MSG zero past the input
    + 1 + cols::RATE + (200 - cols::RATE) // first row: j, PREV, capacity
    + 2; // address halves

/// All of KECCAK's constraints.
pub const NUM_CONSTRAINTS: usize = MODE_CONSTRAINTS + SPONGE_CONSTRAINTS;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for KeccakConstraints {
    fn max_degree(&self) -> usize {
        3
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        use crate::constraints::templates::{emit_add_pair, emit_is_bit};

        // idx 0-49: state_ptr[lane] = addr + 8*lane_idx (gated on MU_P).
        for lane_idx in 0..25 {
            let offset = (lane_idx * 8) as i64;
            emit_add_pair(
                b,
                lane_idx * 2,
                &[cols::MU_P],
                &AddOperand::from_dword_bl(cols::ADDR),
                &AddOperand::constant(offset),
                &AddOperand::from_dword_whh(cols::state_ptr(lane_idx, 0)),
            );
        }

        // idx 50: MU_P · carry_1 (top-lane no-overflow).
        let c256 = b.const_base(256);
        let c65536 = b.const_base(65536);
        let c16777216 = b.const_base(16777216);
        let addr_lo = b.main(0, cols::addr(0))
            + b.main(0, cols::addr(1)) * c256.clone()
            + b.main(0, cols::addr(2)) * c65536.clone()
            + b.main(0, cols::addr(3)) * c16777216.clone();
        let addr_hi = b.main(0, cols::addr(4))
            + b.main(0, cols::addr(5)) * c256
            + b.main(0, cols::addr(6)) * c65536.clone()
            + b.main(0, cols::addr(7)) * c16777216;
        let ptr_lo = b.main(0, cols::state_ptr(24, 0)) + b.main(0, cols::state_ptr(24, 1)) * c65536;
        let ptr_hi = b.main(0, cols::state_ptr(24, 2));

        let inv_2_32 = b.const_base(INV_SHIFT_32);
        let c192 = b.const_base(192);
        let carry_0 = (addr_lo + c192 - ptr_lo) * inv_2_32.clone();
        let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
        let mu_p = b.main(0, cols::MU_P);
        b.emit_base(50, mu_p * carry_1);

        // idx 51: MuIsBit — μ·(1−μ). Every bus send is weighted by μ or by a
        // mode's own bits, so μ ≠ 0 is indistinguishable from μ = 1 at the
        // constraint level, while on the bus a μ = −1 row inverts a range check
        // (carrier). KECCAK does not deduplicate (one row per permutation, μ ∈
        // {0,1}), so the bit constraint is honest and forbids μ = −1.
        let mu = b.main(0, cols::MU);
        let one = b.one();
        b.emit_base(51, mu.clone() * (one.clone() - mu.clone()));

        // idx 52-55: the mode and boundary bits.
        emit_is_bit(b, 52, cols::MU_P, None);
        emit_is_bit(b, 53, cols::MU_S, None);
        emit_is_bit(b, 54, cols::FIRST, None);
        emit_is_bit(b, 55, cols::LAST, None);
        // idx 56: μ = MU_P + MU_S. With μ, MU_P, MU_S bits, at most one mode.
        let mu_p = b.main(0, cols::MU_P);
        let mu_s = b.main(0, cols::MU_S);
        let e = if dropped(SpongeCheck::ModeSum) {
            b.zero()
        } else {
            mu - mu_p - mu_s.clone()
        };
        b.emit_base(56, e);
        // idx 57: FIRST and LAST only on sponge rows: (FIRST + LAST)(1 − MU_S).
        // Both are bits and both products are ≥ 0 in the integers, so the sum
        // vanishing over the bits {0,1} means each does (2·1 ≠ 0 in the field).
        let first = b.main(0, cols::FIRST);
        let last = b.main(0, cols::LAST);
        let e = if dropped(SpongeCheck::FlagsOnSponge) {
            b.zero()
        } else {
            (first.clone() + last.clone()) * (one.clone() - mu_s.clone())
        };
        b.emit_base(57, e);

        // ---------------- sponge mode ----------------
        let mut idx = MODE_CONSTRAINTS;
        // Every sponge constraint names its check, so a test can leave one out
        // (`SpongeCheck`); outside the tests none is ever left out.
        let mut emit = |b: &mut B, check: SpongeCheck, e: B::Expr| {
            let e = if dropped(check) { b.zero() } else { e };
            b.emit_base(idx, e);
            idx += 1;
        };

        // FULL_k, PART_{k,b} bits.
        for lane in 0..cols::RATE_LANES {
            let f = b.main(0, cols::full(lane));
            let e = f.clone() * (b.one() - f);
            emit(b, SpongeCheck::MaskBits, e);
            for byte in 0..7 {
                let p = b.main(0, cols::part(lane, byte));
                let e = p.clone() * (b.one() - p);
                emit(b, SpongeCheck::MaskBits, e);
            }
        }
        // FULL_k · PART_{k,b} = 0: each s_i is a bit.
        for lane in 0..cols::RATE_LANES {
            for byte in 0..7 {
                let f = b.main(0, cols::full(lane));
                let p = b.main(0, cols::part(lane, byte));
                emit(b, SpongeCheck::OneRead, f * p);
            }
        }
        // s_i as an expression.
        let s = |b: &B, i: usize| -> B::Expr {
            let (lane, byte) = (i / 8, i % 8);
            let f = b.main(0, cols::full(lane));
            if byte < 7 {
                f + b.main(0, cols::part(lane, byte))
            } else {
                f
            }
        };
        // Prefix: (1 − s_i) · s_{i+1} = 0.
        for i in 0..cols::RATE - 1 {
            let e = (b.one() - s(b, i)) * s(b, i + 1);
            emit(b, SpongeCheck::Prefix, e);
        }
        // No read off a sponge row: (1 − MU_S) · s_0 = 0.
        let e = (b.one() - mu_s.clone()) * s(b, 0);
        emit(b, SpongeCheck::ReadsOnSponge, e);
        // A row that is not last reads a whole block: (MU_S − LAST)(1 − FULL_16).
        let full_last = b.main(0, cols::full(cols::RATE_LANES - 1));
        let e = (mu_s.clone() - last.clone()) * (b.one() - full_last.clone());
        emit(b, SpongeCheck::FullUnlessLast, e);
        // The last block holds at most 135 input bytes: LAST · FULL_16 = 0.
        emit(b, SpongeCheck::LastNotFull, last.clone() * full_last);
        // The last block holds exactly COUNT bytes: LAST · (COUNT − Σ s_i) = 0.
        let mut taken = b.zero();
        for i in 0..cols::RATE {
            taken = taken + s(b, i);
        }
        let count = b.main(0, cols::COUNT);
        emit(b, SpongeCheck::CountOnLast, last.clone() * (count - taken));
        // Bytes past the input are 0: (1 − s_i) · MSG_i = 0.
        for i in 0..cols::RATE {
            let e = (b.one() - s(b, i)) * b.main(0, cols::msg(i));
            emit(b, SpongeCheck::MsgZero, e);
        }
        // The first row is block 0 and starts from the zero state.
        let e = first.clone() * b.main(0, cols::TIMESTAMP_1);
        emit(b, SpongeCheck::FirstBlock, e);
        for i in 0..cols::RATE {
            let e = first.clone() * b.main(0, cols::prev(i));
            emit(b, SpongeCheck::FirstState, e);
        }
        for i in cols::RATE..200 {
            let e = first.clone() * b.main(0, cols::input_byte(i));
            emit(b, SpongeCheck::FirstState, e);
        }
        // SRC_LO + 135 and OUT_LO + 31 as halves (each half range-checked).
        let c65536 = b.const_base(1 << 16);
        let src_end = b.main(0, cols::SRC_END) + b.main(0, cols::SRC_END + 1) * c65536.clone();
        let e = mu_s * (b.main(0, cols::SRC_LO) + b.const_base(135) - src_end);
        emit(b, SpongeCheck::AddressHalves, e);
        let out_end = b.main(0, cols::OUT_END) + b.main(0, cols::OUT_END + 1) * c65536;
        let e = last * (b.main(0, cols::OUT_LO) + b.const_base(31) - out_end);
        emit(b, SpongeCheck::AddressHalves, e);

        debug_assert_eq!(idx, NUM_CONSTRAINTS);
    }
}
