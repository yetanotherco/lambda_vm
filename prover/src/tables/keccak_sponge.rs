//! KECCAK_SPONGE chip — the sponge-absorb accelerator (`ECALL -4`).
//!
//! One row per absorbed 136-byte rate block. An absorb call over `n` blocks
//! occupies `n` rows that all share the ecall's CPU timestamp: the first row
//! receives the ECALL, reads the three operand registers and MEMW-reads the
//! 25-lane state; every row MEMW-reads its 17 message dwords, XORs the block
//! into lanes 0..17 via `ByteAlu[XOR]` lookups and round-trips the absorbed
//! state through the shared KECCAK_RND chip over the `Keccak` bus; the last
//! row MEMW-writes the final state back. Between rows, the running state and
//! the call's registers travel over the self-referential `KeccakSponge` chain
//! bus (same shape as COMMIT's `CommitNextByte` and the ECDAS sequence bus).
//!
//! ## Why every permutation is keyed by `(timestamp, seq)` — the swap attack
//!
//! All `n` permutations of one call share ONE CPU timestamp (the ecall is a
//! single cycle). The `Keccak` bus tuple used to be `(ts, round, state)` and
//! the round chip echoes `ts` through its 24-round chain. With two blocks A
//! and B at the same `ts`, a malicious prover could feed the round chip
//! `perm-input(A)` and `perm-input(B)` and hand A's permutation output to B's
//! row and vice versa: every tuple still appears exactly once per side, so
//! **the bus balances** and the forged sponge "absorbs" the blocks against
//! swapped intermediate states. The chain equality between rows does not save
//! this on its own — it only forces *some* consistent assignment of outputs
//! to rows, not the right one — unless the keys on the Keccak-bus legs are
//! unique per permutation.
//!
//! The fix: the `Keccak` bus carries an extra `seq` element. The classic
//! KECCAK core chip (one permutation per ecall) sends `seq = 0`; sponge row
//! `k` sends `seq = SEQ = k`; KECCAK_RND carries `seq` through untouched,
//! exactly like it carries `ts`. `SEQ` itself is pinned by the chain:
//!
//! - `μ_first · SEQ = 0` anchors the first row of a call at `SEQ = 0`;
//! - the chain sender emits `SEQ + 1` and the receiver consumes `SEQ`, so
//!   every non-first row's `SEQ` is its predecessor's `SEQ + 1`. A chain can
//!   never wrap the field (that would need `p ≈ 2^64` rows), so `SEQ` values
//!   along one call are exactly `0, 1, …, n−1` — distinct, hence every
//!   permutation of the call has a unique `(ts, seq)` key on both Keccak-bus
//!   legs, and the swap above unbalances the bus.
//!
//! Chain-shape soundness (why the rows of one call form a simple path):
//! - exactly one `μ_first` row per call: the CPU sends ONE Ecall token per
//!   ecall; two first rows would consume it twice and unbalance the bus;
//! - no forks/merges: every chain token is emitted once (`μ − μ_last`) and
//!   consumed once (`μ − μ_first`); duplicating a link propagates back to a
//!   duplicated Ecall consumption (and forward to a duplicated state write on
//!   one `(address, timestamp)` pair, which the memory argument rejects);
//! - exactly `n = x12` rows: the first row reads `x12` into `(N_LO, N_HI)`,
//!   the chain carries them unchanged, and the last row pins
//!   `N_LO = SEQ + 1`, `N_HI = 0`. Pinning the WORDS (not the recombined
//!   field value) closes the mod-p alias `N = n + p`: a register value of
//!   `n + p` has `N_HI = 2^32 − 1 ≠ 0`. This bounds provable calls to
//!   `n < 2^32`, which is vacuous — `n` real rows must exist in this table,
//!   so `n` is bounded by the trace size long before `2^32`;
//! - a call can never end early or run forever: a non-last row's chain token
//!   must be consumed and a last row must satisfy `N_LO = SEQ + 1`, so
//!   `n = 0` (which the executor also rejects) admits no witness at all.
//!
//! ## Addressing (ECSM low-limb idiom, NOT the KECCAK pointer apparatus)
//!
//! The classic KECCAK core chip materializes one DWordHL pointer per lane
//! (100 columns + 100 IS_HALF sends per row). That is affordable at one row
//! per *call* but would double this chip's per-block cost, so the sponge uses
//! the ECSM operand idiom instead: per-access addresses go on the Memw bus as
//! `base_lo + offset` with `base_hi` unchanged — no carry into the low limb —
//! and the executor guarantees the room (`(base % 2^32) + last_offset <
//! 2^32`, see `KECCAK_ABSORB_SYSCALL_NUMBER`). Soundness is fail-closed: a
//! block base whose low limb has drifted out of range yields Memw/Memory
//! tokens with no matching PAGE/REGISTER cell, unbalancing the bus (the same
//! argument `memw.rs` makes for its virtual `address_add` carries).
//!
//! - `state_ptr` is materialized as 8 range-checked bytes (`S_ADDR`, DWordBL)
//!   so the `addr & 7 = 0` alignment lookup has the low byte; lane `i` of the
//!   state lives at `(s_lo + 8i, s_hi)` with `s_lo/s_hi` the byte recombines.
//! - the current block base is carried as two words `(D_LO, D_HI)`; message
//!   dword `j` lives at `(D_LO + 8j, D_HI)`; the chain sender advances the
//!   base by one rate block as the *linear* element `D_LO + 136` (sound
//!   because the executor's low-limb guarantee covers the whole data region,
//!   and a lying prover only produces unmatchable Memw tokens, per the
//!   fail-closed argument above).
//!
//! ## Memory model (must mirror `collect_keccak_sponge_memw_ops` op-for-op)
//!
//! - first row, at `ts`: register reads x10/x11/x12 (24-element read tuples)
//!   and 25 pure lane reads of the state (`old = value = STATE_IN`);
//! - every row, at `ts`: 17 pure dword reads of the block (`old = value`);
//! - last row, at `ts + 1`: 25 write-only lane writes (16-element tuples; the
//!   MEMW table materializes `old` itself — the pre-write content is the
//!   first row's `STATE_IN`, re-written at `ts` by the lane reads). Reads at
//!   `ts` / write at `ts + 1` keeps every `(address, timestamp)` pair unique,
//!   which the memory argument's strict `old_ts < ts` ordering requires; the
//!   executor's region-overlap rejection guarantees the state and data
//!   regions never collide at `ts`.
//!
//! ## Byte range checks
//!
//! `STATE_IN[0..136]` and `BLOCK` are operands of the `ByteAlu[XOR]` lookups,
//! which simultaneously range-check both operands and pin the output —
//! `XORED` needs no extra check. `STATE_IN[136..200]` is bound element-wise
//! either to memory bytes (first row, MEMW read) or to the previous row's
//! `STATE_OUT` (chain), and `STATE_OUT` is bound element-wise to KECCAK_RND's
//! χ/ι columns, themselves XOR-lookup outputs — so every state byte is
//! transitively range-checked without further sends. `S_ADDR` gets explicit
//! `AreBytes` pairs (its cells feed linear address recombines).
//!
//! ## Column layout (690 columns)
//!
//! | Group       | Size | Description                                        |
//! |-------------|------|----------------------------------------------------|
//! | timestamp   |    2 | DWordWL, the ecall's CPU timestamp                 |
//! | seq         |    1 | Block index within the call (0-based)              |
//! | n           |    2 | x12 (n_blocks) as DWordWL words                    |
//! | s_addr      |    8 | state_ptr as DWordBL bytes                         |
//! | d           |    2 | current block base (data_ptr + 136·seq) as words   |
//! | state_in    |  200 | running state entering this block [lane][byte]     |
//! | block       |  136 | message block bytes [lane][byte]                   |
//! | xored       |  136 | state_in[i] ^ block[i] for the absorbed region     |
//! | state_out   |  200 | permuted state [lane][byte]                        |
//! | μ, μ_first, μ_last | 3 | multiplicity / bookend flags                |

use executor::vm::instruction::execution::{KECCAK_ABSORB_SYSCALL_NUMBER, KECCAK_RATE_BYTES};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::emit_is_bit;

/// Bytes of one rate block (17 lanes × 8).
pub const RATE_BYTES: usize = KECCAK_RATE_BYTES as usize;
/// Lanes of one rate block.
pub const RATE_LANES: usize = 17;

// =========================================================================
// Column indices
// =========================================================================

pub mod cols {
    use super::RATE_BYTES;

    pub const TIMESTAMP_0: usize = 0;
    pub const TIMESTAMP_1: usize = 1;

    /// Block index within the call (0-based).
    pub const SEQ: usize = 2;

    /// x12 (n_blocks) as DWordWL words, carried unchanged along the chain.
    pub const N_LO: usize = 3;
    pub const N_HI: usize = 4;

    /// state_ptr as DWordBL (8 bytes), carried along the chain.
    pub const S_ADDR: usize = 5;

    /// Current block base address (data_ptr + 136·seq) as DWordWL words.
    pub const D_LO: usize = S_ADDR + 8; // 13
    pub const D_HI: usize = D_LO + 1; // 14

    /// state_in[25][8] — running state entering this block.
    pub const STATE_IN: usize = D_HI + 1; // 15

    /// block[17][8] — the message block.
    pub const BLOCK: usize = STATE_IN + 200; // 215

    /// xored[17][8] — state_in ^ block over the absorbed region.
    pub const XORED: usize = BLOCK + RATE_BYTES; // 351

    /// state_out[25][8] — the permuted state.
    pub const STATE_OUT: usize = XORED + RATE_BYTES; // 487

    /// μ: 1 on real rows.
    pub const MU: usize = STATE_OUT + 200; // 687
    /// μ_first: 1 on the first row of a call (receives the ECALL).
    pub const MU_FIRST: usize = MU + 1; // 688
    /// μ_last: 1 on the last row of a call (writes the state back).
    pub const MU_LAST: usize = MU_FIRST + 1; // 689

    pub const NUM_COLUMNS: usize = MU_LAST + 1; // 690

    // -------------------------------------------------------------------------
    // Index helpers (lane = x + 5y, matching the KECCAK core chip layout)
    // -------------------------------------------------------------------------

    #[inline]
    pub const fn s_addr(byte: usize) -> usize {
        S_ADDR + byte
    }

    #[inline]
    pub const fn state_in(lane: usize, byte: usize) -> usize {
        STATE_IN + lane * 8 + byte
    }

    #[inline]
    pub const fn block(lane: usize, byte: usize) -> usize {
        BLOCK + lane * 8 + byte
    }

    #[inline]
    pub const fn xored(lane: usize, byte: usize) -> usize {
        XORED + lane * 8 + byte
    }

    #[inline]
    pub const fn state_out(lane: usize, byte: usize) -> usize {
        STATE_OUT + lane * 8 + byte
    }
}

// =========================================================================
// Operation struct
// =========================================================================

/// One absorbed block (= one row) of a keccak sponge-absorb call.
#[derive(Debug, Clone)]
pub struct KeccakSpongeOperation {
    /// The ecall's CPU timestamp (shared by every block of the call).
    pub timestamp: u64,
    /// Block index within the call (0-based).
    pub seq: u64,
    /// Total blocks of the call (the x12 register value).
    pub n_blocks: u64,
    /// state_ptr (the x10 register value).
    pub state_addr: u64,
    /// This block's base address: data_ptr + 136·seq.
    pub block_addr: u64,
    /// Running state entering this block.
    pub state_in: [u64; 25],
    /// The 136 message bytes of this block.
    pub block: [u8; RATE_BYTES],
    /// The permuted state leaving this block.
    pub state_out: [u64; 25],
    /// First row of the call.
    pub first: bool,
    /// Last row of the call.
    pub last: bool,
}

// =========================================================================
// Trace generation
// =========================================================================

pub fn generate_keccak_sponge_trace(
    ops: &[KeccakSpongeOperation],
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
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);
        table.set_u64(row_idx, cols::SEQ, op.seq);
        table.set_dword_wl(row_idx, cols::N_LO, op.n_blocks);
        table.set_dword_bl(row_idx, cols::s_addr(0), op.state_addr);
        table.set_dword_wl(row_idx, cols::D_LO, op.block_addr);

        for (lane, &v) in op.state_in.iter().enumerate() {
            table.set_dword_bl(row_idx, cols::state_in(lane, 0), v);
        }
        table.set_bytes(row_idx, cols::block(0, 0), &op.block);
        for i in 0..RATE_BYTES {
            let state_byte = ((op.state_in[i / 8] >> ((i % 8) * 8)) & 0xFF) as u8;
            table.set_byte(row_idx, cols::XORED + i, state_byte ^ op.block[i]);
        }
        for (lane, &v) in op.state_out.iter().enumerate() {
            table.set_dword_bl(row_idx, cols::state_out(lane, 0), v);
        }

        table.set_fe(row_idx, cols::MU, FE::one());
        table.set_bool(row_idx, cols::MU_FIRST, op.first);
        table.set_bool(row_idx, cols::MU_LAST, op.last);
    }

    // Padding rows stay all-zero: μ = μ_first = μ_last = 0 gates every bus
    // interaction, and all seven transition constraints hold at zero.
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

/// `s_addr`'s low word as a linear byte recombine, plus a constant offset.
fn s_lo_plus(offset: i64) -> BusValue {
    let mut terms: Vec<LinearTerm> = (0..4)
        .map(|i| LinearTerm::Column {
            coefficient: 1i64 << (8 * i),
            column: cols::s_addr(i),
        })
        .collect();
    if offset != 0 {
        terms.push(LinearTerm::Constant(offset));
    }
    BusValue::linear(terms)
}

/// `s_addr`'s high word as a linear byte recombine.
fn s_hi() -> BusValue {
    BusValue::linear(
        (0..4)
            .map(|i| LinearTerm::Column {
                coefficient: 1i64 << (8 * i),
                column: cols::s_addr(4 + i),
            })
            .collect(),
    )
}

/// `[old[8], is_register, base_lo, base_hi, value[8], ts_lo, ts_hi, w2, w4, w8]`
/// — a 24-element MEMW **read** tuple (`old == value`), as in `ecsm.rs`.
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
/// a 16-element MEMW **write** tuple (the MEMW table supplies `old`).
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

/// The 8 bytes of trace lane `col + 8*lane .. +8` as MEMW value elements.
fn lane_bytes(base_col: usize, lane: usize) -> [BusValue; 8] {
    std::array::from_fn(|b| packed(base_col + lane * 8 + b))
}

/// The call-state elements shared by the chain receive/send:
/// `[n_lo, n_hi, s_lo, s_hi, d_lo(+offset), d_hi]`.
fn chain_registers(d_lo_offset: i64) -> Vec<BusValue> {
    let d_lo = if d_lo_offset == 0 {
        packed(cols::D_LO)
    } else {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::D_LO,
            },
            LinearTerm::Constant(d_lo_offset),
        ])
    };
    vec![
        packed(cols::N_LO),
        packed(cols::N_HI),
        s_lo_plus(0),
        s_hi(),
        d_lo,
        packed(cols::D_HI),
    ]
}

// =========================================================================
// Bus interactions (216 total)
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = KECCAK_ABSORB_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_ABSORB_SYSCALL_NUMBER >> 32;
    let mu = || Multiplicity::Column(cols::MU);
    let mu_first = || Multiplicity::Column(cols::MU_FIRST);
    let mu_last = || Multiplicity::Column(cols::MU_LAST);
    let ts_lo = || packed(cols::TIMESTAMP_0);
    let ts_hi = || packed(cols::TIMESTAMP_1);
    let ts_lo_plus_1 = || {
        BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::TIMESTAMP_0,
            },
            LinearTerm::Constant(1),
        ])
    };

    let mut interactions = Vec::with_capacity(216);

    // 1. ECALL receiver (mult = μ_first): [ts_lo, ts_hi, syscall_lo32, syscall_hi32].
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        mu_first(),
        vec![
            ts_lo(),
            ts_hi(),
            BusValue::constant(syscall_lo),
            BusValue::constant(syscall_hi),
        ],
    ));

    // 2-4. Register reads at ts (mult = μ_first): x10 = state_ptr,
    // x11 = data_ptr (= this row's block base, since SEQ = 0 on first rows),
    // x12 = n_blocks. All pure 24-element reads (old == value).
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        mu_first(),
        memw_read(
            register_value(s_lo_plus(0), s_hi()),
            1,
            BusValue::constant(2 * 10),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
            1,
            0,
        ),
    ));
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        mu_first(),
        memw_read(
            register_value(packed(cols::D_LO), packed(cols::D_HI)),
            1,
            BusValue::constant(2 * 11),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
            1,
            0,
        ),
    ));
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        mu_first(),
        memw_read(
            register_value(packed(cols::N_LO), packed(cols::N_HI)),
            1,
            BusValue::constant(2 * 12),
            BusValue::constant(0),
            ts_lo(),
            ts_hi(),
            1,
            0,
        ),
    ));

    // 5. Chain receive (mult = μ − μ_first):
    //    [ts, seq, n, s_addr, block_base, state_in[200]].
    {
        let mut values = vec![ts_lo(), ts_hi(), packed(cols::SEQ)];
        values.extend(chain_registers(0));
        for i in 0..200 {
            values.push(packed(cols::STATE_IN + i));
        }
        interactions.push(BusInteraction::receiver(
            BusId::KeccakSponge,
            Multiplicity::Diff(cols::MU, cols::MU_FIRST),
            values,
        ));
    }

    // 6. Chain send (mult = μ − μ_last):
    //    [ts, seq + 1, n, s_addr, block_base + 136, state_out[200]].
    {
        let mut values = vec![
            ts_lo(),
            ts_hi(),
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::SEQ,
                },
                LinearTerm::Constant(1),
            ]),
        ];
        values.extend(chain_registers(RATE_BYTES as i64));
        for i in 0..200 {
            values.push(packed(cols::STATE_OUT + i));
        }
        interactions.push(BusInteraction::sender(
            BusId::KeccakSponge,
            Multiplicity::Diff(cols::MU, cols::MU_LAST),
            values,
        ));
    }

    // 7. Keccak bus: send (ts, round = 0, seq, absorbed_state[200]).
    // The absorbed state is XORED over lanes 0..17 and STATE_IN pass-through
    // over lanes 17..25. Element order must match KECCAK_RND's receiver:
    // x outer, y inner, lane = x + 5y.
    {
        let mut values = vec![ts_lo(), ts_hi(), BusValue::constant(0), packed(cols::SEQ)];
        for x in 0..5 {
            for y in 0..5 {
                let lane = x + 5 * y;
                for b in 0..8 {
                    let col = if lane < RATE_LANES {
                        cols::xored(lane, b)
                    } else {
                        cols::state_in(lane, b)
                    };
                    values.push(packed(col));
                }
            }
        }
        interactions.push(BusInteraction::sender(BusId::Keccak, mu(), values));
    }

    // 8. Keccak bus: receive (ts, round = 24, seq, state_out[200]).
    {
        let mut values = vec![ts_lo(), ts_hi(), BusValue::constant(24), packed(cols::SEQ)];
        for x in 0..5 {
            for y in 0..5 {
                let lane = x + 5 * y;
                for b in 0..8 {
                    values.push(packed(cols::state_out(lane, b)));
                }
            }
        }
        interactions.push(BusInteraction::receiver(BusId::Keccak, mu(), values));
    }

    // 9. Absorb XORs (136, mult = μ): XORED[i] = STATE_IN[i] ^ BLOCK[i].
    // The lookup simultaneously range-checks both operands and pins the output.
    for i in 0..RATE_BYTES {
        interactions.push(BusInteraction::sender(
            BusId::ByteAlu,
            mu(),
            vec![
                BusValue::constant(alu_op::XOR as u64),
                packed(cols::STATE_IN + i),
                packed(cols::BLOCK + i),
                packed(cols::XORED + i),
            ],
        ));
    }

    // 10. Alignment: s_addr[0] & 7 = 0 (mult = μ).
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        mu(),
        vec![
            BusValue::constant(alu_op::AND as u64),
            packed(cols::s_addr(0)),
            BusValue::constant(7),
            BusValue::constant(0),
        ],
    ));

    // 11. Range-check the s_addr bytes (4 ARE_BYTES pairs, mult = μ): the
    // cells feed the linear s_lo/s_hi recombines, so without per-byte checks
    // a prover could encode non-byte values that keep the recombined field
    // value (and hence the MEMW tuples) intact while dodging the alignment
    // lookup (same rationale as the KECCAK core chip's addr checks).
    for i in 0..4 {
        interactions.push(BusInteraction::sender(
            BusId::AreBytes,
            mu(),
            vec![packed(cols::s_addr(2 * i)), packed(cols::s_addr(2 * i + 1))],
        ));
    }

    // 12. Message dword reads (17, mult = μ): pure reads of block dword j at
    // (D_LO + 8j, D_HI), timestamp ts.
    for j in 0..RATE_LANES {
        let base_lo = if j == 0 {
            packed(cols::D_LO)
        } else {
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_LO,
                },
                LinearTerm::Constant((8 * j) as i64),
            ])
        };
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            mu(),
            memw_read(
                lane_bytes(cols::BLOCK, j),
                0,
                base_lo,
                packed(cols::D_HI),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // 13. State lane reads (25, mult = μ_first): pure reads of the pre-call
    // state at (s_lo + 8·lane, s_hi), timestamp ts.
    for lane in 0..25 {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            mu_first(),
            memw_read(
                lane_bytes(cols::STATE_IN, lane),
                0,
                s_lo_plus((8 * lane) as i64),
                s_hi(),
                ts_lo(),
                ts_hi(),
                0,
                1,
            ),
        ));
    }

    // 14. State lane writes (25, mult = μ_last): write-only tuples of the
    // final state at (s_lo + 8·lane, s_hi), timestamp ts + 1 (the MEMW table
    // materializes old = the value the μ_first lane reads re-wrote at ts).
    for lane in 0..25 {
        interactions.push(BusInteraction::sender(
            BusId::Memw,
            mu_last(),
            memw_write(
                lane_bytes(cols::STATE_OUT, lane),
                s_lo_plus((8 * lane) as i64),
                s_hi(),
                ts_lo_plus_1(),
                ts_hi(),
            ),
        ));
    }

    interactions
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The KECCAK_SPONGE table's 7 transition constraints as a single
/// [`ConstraintSet`]:
/// - idx 0-2: `IS_BIT` on `μ`, `μ_first`, `μ_last` (unconditional; padding
///   rows are all-zero);
/// - idx 3:   `(μ_first + μ_last)·(1 − μ) = 0` (bookends imply μ; a row may
///   be both when n = 1);
/// - idx 4:   `μ_first · SEQ = 0` (a call's chain starts at block 0 — the
///   anchor of the `(ts, seq)` permutation keying, see the module docs);
/// - idx 5:   `μ_last · (N_LO − SEQ − 1) = 0` (the call has exactly
///   `n_blocks` rows);
/// - idx 6:   `μ_last · N_HI = 0` (pins the x12 WORDS, not the recombined
///   field value — closes the `N = n + p` mod-p alias).
///
/// Everything else is bus-enforced: XOR/range checks via ByteAlu/AreBytes,
/// the chain increment via the `KeccakSponge` sender's `SEQ + 1` /
/// `D_LO + 136` linear elements, and the permutation via the Keccak bus.
#[derive(Clone, Copy)]
pub struct KeccakSpongeConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for KeccakSpongeConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0-2: IS_BIT on μ, μ_first, μ_last.
        emit_is_bit(b, 0, cols::MU, None);
        emit_is_bit(b, 1, cols::MU_FIRST, None);
        emit_is_bit(b, 2, cols::MU_LAST, None);

        // idx 3: (μ_first + μ_last) · (1 − μ) = 0.
        let one = b.one();
        let first = b.main(0, cols::MU_FIRST);
        let last = b.main(0, cols::MU_LAST);
        let mu = b.main(0, cols::MU);
        b.emit_base(3, (first + last) * (one - mu));

        // idx 4: μ_first · SEQ = 0.
        let first = b.main(0, cols::MU_FIRST);
        let seq = b.main(0, cols::SEQ);
        b.emit_base(4, first * seq);

        // idx 5: μ_last · (N_LO − SEQ − 1) = 0.
        let last = b.main(0, cols::MU_LAST);
        let n_lo = b.main(0, cols::N_LO);
        let seq = b.main(0, cols::SEQ);
        let one = b.one();
        b.emit_base(5, last * (n_lo - seq - one));

        // idx 6: μ_last · N_HI = 0.
        let last = b.main(0, cols::MU_LAST);
        let n_hi = b.main(0, cols::N_HI);
        b.emit_base(6, last * n_hi);
    }
}
