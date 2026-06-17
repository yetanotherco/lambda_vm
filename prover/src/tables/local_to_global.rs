//! Local-to-global memory boundary claims for cross-epoch continuations.
//!
//! Each epoch, for every memory cell it touches,
//! makes an `init` claim (the cell's value when first touched this epoch, which
//! earlier epoch last wrote it, and that write's timestamp) and a `fini` claim
//! (the cell's value at this epoch's end, this epoch's index, and the last
//! access timestamp). A final LogUp matches each `fini` against the `init` of the
//! next epoch that touches the same cell, proving global memory consistency.
//!
//! ## Range-checked columns
//!
//! Every column is range-checked so it cannot hold an out-of-range field
//! element: the two value columns are bytes (checked via the `AreBytes` bus) and
//! every other quantity (address, epoch, timestamp) is stored as 16-bit
//! halfword columns checked via the `IsHalfword` bus. The wider 32-bit values
//! the buses match on are never stored directly — they are reconstructed from
//! the (range-checked) halfwords by a linear combination, so a malicious prover
//! cannot smuggle a non-canonical value past the lookup. The range-check
//! lookups are emitted on the epoch-local table (which has the BITWISE provider);
//! the global proof commits the identical trace (the commitment binding compares
//! their roots), so it inherits the same guarantee.

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::paged_mem::PagedMem;

/// Per-cell provenance: `(last_writer_epoch, value, timestamp)`. Unset cells read
/// back as the genesis default `(GENESIS_EPOCH, 0, 0)`.
type Provenance = PagedMem<(u64, u64, u64)>;

/// Sentinel `originating_epoch` for cells whose value comes from the program's
/// initial memory — no prior epoch wrote them.
pub const GENESIS_EPOCH: u64 = u64::MAX;

/// A cell's state when an epoch first touches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InitClaim {
    /// Value the cell held when this epoch first touched it.
    pub value: u64,
    /// Epoch that last wrote the cell (or [`GENESIS_EPOCH`]).
    pub originating_epoch: u64,
    /// Timestamp of that originating write.
    pub timestamp: u64,
}

/// A cell's state at the end of the epoch that touched it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FiniClaim {
    /// Value the cell holds at this epoch's end.
    pub value: u64,
    /// This epoch's index.
    pub epoch: u64,
    /// Last access timestamp for the cell this epoch.
    pub timestamp: u64,
}

/// The init/fini boundary claims for a single touched cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CellBoundary {
    pub address: u64,
    pub init: InitClaim,
    pub fini: FiniClaim,
}

/// One epoch's touched cells, each as `(address, end_value, end_timestamp)`.
pub type EpochTouches = Vec<(u64, u64, u64)>;

/// Compute the sparse per-epoch boundary claims.
///
/// `initial_memory` maps each address to its program-start value (originating
/// epoch [`GENESIS_EPOCH`], timestamp 0). `epochs[e]` lists the cells touched in
/// epoch `e` with their end value and end timestamp. Returns, per epoch, the
/// boundary claims for exactly the cells that epoch touched (sparse): each
/// cell's `init` is taken from the previous epoch that wrote it, and its `fini`
/// records this epoch as the new writer.
pub fn epoch_boundaries(
    initial_memory: &HashMap<u64, u64>,
    epochs: &[EpochTouches],
) -> Vec<Vec<CellBoundary>> {
    // provenance[addr] = (last_writer_epoch, value, timestamp)
    let mut provenance = genesis_provenance(initial_memory);

    let mut result = Vec::with_capacity(epochs.len());
    for (epoch, touched) in epochs.iter().enumerate() {
        result.push(epoch_boundary(&mut provenance, epoch as u64, touched));
    }
    result
}

/// One epoch's boundaries, taking `init` from the running `provenance` (the cell's
/// last writer) and updating `provenance` with this epoch's `fini`. This is the
/// per-epoch step of [`epoch_boundaries`], exposed so the streaming continuation
/// prover can build each epoch's table incrementally without all epochs at once.
pub fn epoch_boundary(
    provenance: &mut Provenance,
    epoch: u64,
    touched: &[(u64, u64, u64)],
) -> Vec<CellBoundary> {
    let mut boundaries = Vec::with_capacity(touched.len());
    for &(address, end_value, end_timestamp) in touched {
        // Unset cells read back as the genesis default `(GENESIS_EPOCH, 0, 0)`.
        let (originating_epoch, init_value, init_timestamp) = provenance.get(address);
        boundaries.push(CellBoundary {
            address,
            init: InitClaim {
                value: init_value,
                originating_epoch,
                timestamp: init_timestamp,
            },
            fini: FiniClaim {
                value: end_value,
                epoch,
                timestamp: end_timestamp,
            },
        });
        provenance.set(address, (epoch, end_value, end_timestamp));
    }
    boundaries
}

/// Seed the provenance store from the program's initial memory (genesis cells).
pub fn genesis_provenance(initial_memory: &HashMap<u64, u64>) -> Provenance {
    let mut provenance = Provenance::new((GENESIS_EPOCH, 0, 0));
    for (&addr, &value) in initial_memory {
        provenance.set(addr, (GENESIS_EPOCH, value, 0));
    }
    provenance
}

// =========================================================================
// AIR trace columns
// =========================================================================

/// Column indices for the local-to-global table: one row per touched cell.
///
/// Every wide quantity is stored as 16-bit halfword columns so each one can be
/// `IsHalfword`-range-checked directly; the 32-bit values the buses match on
/// (`address_lo/hi`, `epoch`, `timestamp_lo/hi`) are reconstructed from those
/// halfwords via a linear combination (see [`word`]). The two value columns are
/// single bytes (RAM is byte-granular), range-checked via `AreBytes`.
pub mod cols {
    // Address (64-bit) as four 16-bit halfwords: `address_lo = A0 + 2^16·A1`,
    // `address_hi = A2 + 2^16·A3`.
    pub const ADDR_0: usize = 0;
    pub const ADDR_1: usize = 1;
    pub const ADDR_2: usize = 2;
    pub const ADDR_3: usize = 3;

    /// Init value: a single byte, like PAGE's `value`.
    pub const INIT_VALUE: usize = 4;

    // Init epoch (fits 32 bits; `GENESIS_EPOCH` reduces to `2^32-2`): two halfwords.
    pub const INIT_EPOCH_0: usize = 5;
    pub const INIT_EPOCH_1: usize = 6;

    // Init timestamp (64-bit) as four halfwords (`ts_lo = T0 + 2^16·T1`, etc.).
    pub const INIT_TS_0: usize = 7;
    pub const INIT_TS_1: usize = 8;
    pub const INIT_TS_2: usize = 9;
    pub const INIT_TS_3: usize = 10;

    /// Fini value: a single byte.
    pub const FINI_VALUE: usize = 11;

    // Fini epoch: two halfwords.
    pub const FINI_EPOCH_0: usize = 12;
    pub const FINI_EPOCH_1: usize = 13;

    // Fini timestamp: four halfwords.
    pub const FINI_TS_0: usize = 14;
    pub const FINI_TS_1: usize = 15;
    pub const FINI_TS_2: usize = 16;
    pub const FINI_TS_3: usize = 17;

    pub const NUM_COLUMNS: usize = 18;

    /// The halfword columns, in order — every column that is `IsHalfword`-checked.
    pub const HALFWORD_COLUMNS: [usize; 16] = [
        ADDR_0,
        ADDR_1,
        ADDR_2,
        ADDR_3,
        INIT_EPOCH_0,
        INIT_EPOCH_1,
        INIT_TS_0,
        INIT_TS_1,
        INIT_TS_2,
        INIT_TS_3,
        FINI_EPOCH_0,
        FINI_EPOCH_1,
        FINI_TS_0,
        FINI_TS_1,
        FINI_TS_2,
        FINI_TS_3,
    ];
}

/// Little-endian 16-bit halfwords of a 64-bit value: `[bits 0-15, 16-31, 32-47, 48-63]`.
fn halfwords64(v: u64) -> [u64; 4] {
    [
        v & 0xFFFF,
        (v >> 16) & 0xFFFF,
        (v >> 32) & 0xFFFF,
        (v >> 48) & 0xFFFF,
    ]
}

/// Canonical 32-bit field value of an epoch index, matching `FE::from(epoch)`.
///
/// Real epoch indices are small (< 2^32) and map to themselves; the genesis
/// sentinel [`GENESIS_EPOCH`] (`u64::MAX`) reduces to `2^32 - 2` modulo the
/// Goldilocks prime, which is exactly what `global_memory` emits via
/// `FE::from(GENESIS_EPOCH)`, so the two sides match on the bus.
fn epoch_field_low32(epoch: u64) -> u64 {
    if epoch == GENESIS_EPOCH {
        (1 << 32) - 2
    } else {
        debug_assert!(epoch < (1 << 32), "epoch index exceeds 32 bits");
        epoch
    }
}

/// The two halfwords of an epoch index (its canonical 32-bit field value).
fn epoch_halfwords(epoch: u64) -> [u64; 2] {
    let v = epoch_field_low32(epoch);
    [v & 0xFFFF, (v >> 16) & 0xFFFF]
}

// =========================================================================
// Trace generation
// =========================================================================

/// Build the local-to-global trace: one row per touched cell's boundary claims,
/// padded up to a power of two (padding rows are all zero).
pub fn generate_local_to_global_trace(
    boundaries: &[CellBoundary],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = boundaries.len().next_power_of_two().max(1);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row, b) in boundaries.iter().enumerate() {
        let base = row * cols::NUM_COLUMNS;
        let addr = halfwords64(b.address);
        let init_ts = halfwords64(b.init.timestamp);
        let fini_ts = halfwords64(b.fini.timestamp);
        let init_epoch = epoch_halfwords(b.init.originating_epoch);
        let fini_epoch = epoch_halfwords(b.fini.epoch);

        data[base + cols::ADDR_0] = FE::from(addr[0]);
        data[base + cols::ADDR_1] = FE::from(addr[1]);
        data[base + cols::ADDR_2] = FE::from(addr[2]);
        data[base + cols::ADDR_3] = FE::from(addr[3]);
        data[base + cols::INIT_VALUE] = FE::from(b.init.value & 0xFF);
        data[base + cols::INIT_EPOCH_0] = FE::from(init_epoch[0]);
        data[base + cols::INIT_EPOCH_1] = FE::from(init_epoch[1]);
        data[base + cols::INIT_TS_0] = FE::from(init_ts[0]);
        data[base + cols::INIT_TS_1] = FE::from(init_ts[1]);
        data[base + cols::INIT_TS_2] = FE::from(init_ts[2]);
        data[base + cols::INIT_TS_3] = FE::from(init_ts[3]);
        data[base + cols::FINI_VALUE] = FE::from(b.fini.value & 0xFF);
        data[base + cols::FINI_EPOCH_0] = FE::from(fini_epoch[0]);
        data[base + cols::FINI_EPOCH_1] = FE::from(fini_epoch[1]);
        data[base + cols::FINI_TS_0] = FE::from(fini_ts[0]);
        data[base + cols::FINI_TS_1] = FE::from(fini_ts[1]);
        data[base + cols::FINI_TS_2] = FE::from(fini_ts[2]);
        data[base + cols::FINI_TS_3] = FE::from(fini_ts[3]);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

/// A 32-bit value reconstructed from its two halfword columns: `lo + 2^16·hi`.
fn word(lo_col: usize, hi_col: usize) -> BusValue {
    BusValue::linear(vec![
        LinearTerm::Column {
            coefficient: 1,
            column: lo_col,
        },
        LinearTerm::Column {
            coefficient: 1 << 16,
            column: hi_col,
        },
    ])
}

fn byte(column: usize) -> BusValue {
    BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    }
}

/// Cross-epoch memory bus interactions, two per row (one touched cell):
/// - **receive** the `init` token `(address, value, originating_epoch, timestamp)`
///   left by the epoch that last wrote the cell;
/// - **send** the `fini` token `(address, value, current_epoch, timestamp)` for
///   the next epoch that touches the cell.
///
/// These tokens are matched ACROSS epochs by the final aggregation LogUp (step 4),
/// so within a single epoch's table the GlobalMemory bus is deliberately
/// unbalanced (real rows have `init != fini`). All-zero padding rows self-cancel
/// because their init and fini tokens are identical.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // init: receive the token left by the originating epoch.
        BusInteraction::receiver(
            BusId::GlobalMemory,
            Multiplicity::One,
            vec![
                word(cols::ADDR_0, cols::ADDR_1),
                word(cols::ADDR_2, cols::ADDR_3),
                byte(cols::INIT_VALUE),
                word(cols::INIT_EPOCH_0, cols::INIT_EPOCH_1),
                word(cols::INIT_TS_0, cols::INIT_TS_1),
                word(cols::INIT_TS_2, cols::INIT_TS_3),
            ],
        ),
        // fini: send the token for the next epoch to consume.
        BusInteraction::sender(
            BusId::GlobalMemory,
            Multiplicity::One,
            vec![
                word(cols::ADDR_0, cols::ADDR_1),
                word(cols::ADDR_2, cols::ADDR_3),
                byte(cols::FINI_VALUE),
                word(cols::FINI_EPOCH_0, cols::FINI_EPOCH_1),
                word(cols::FINI_TS_0, cols::FINI_TS_1),
                word(cols::FINI_TS_2, cols::FINI_TS_3),
            ],
        ),
    ]
}

/// Epoch-LOCAL memory bus interactions, mirroring PAGE-C3/C4 (`page.rs`).
///
/// Inside an epoch proof the L2G table bookends the epoch's `Memory` bus for the
/// RAM bytes it touches: it receives each cell's initial token at timestamp 0
/// (the epoch-start seed, matching the first MEMW read's `old_timestamp`) and
/// sends its final token at the last access timestamp. This replaces PAGE's
/// init/fini bookend for touched bytes. The `Memory` token layout is
/// `[is_register, address_lo, address_hi, timestamp_lo, timestamp_hi, value]`;
/// RAM only, so `is_register = 0`, and the byte value is the LO column.
pub fn memory_bus_interactions() -> Vec<BusInteraction> {
    vec![
        // init: receive the cell's initial token at the epoch-start seed (ts = 0).
        BusInteraction::receiver(
            BusId::Memory,
            Multiplicity::One,
            vec![
                BusValue::constant(0),
                word(cols::ADDR_0, cols::ADDR_1),
                word(cols::ADDR_2, cols::ADDR_3),
                BusValue::constant(0),
                BusValue::constant(0),
                byte(cols::INIT_VALUE),
            ],
        ),
        // fini: send the cell's final token at the last access timestamp.
        BusInteraction::sender(
            BusId::Memory,
            Multiplicity::One,
            vec![
                BusValue::constant(0),
                word(cols::ADDR_0, cols::ADDR_1),
                word(cols::ADDR_2, cols::ADDR_3),
                word(cols::FINI_TS_0, cols::FINI_TS_1),
                word(cols::FINI_TS_2, cols::FINI_TS_3),
                byte(cols::FINI_VALUE),
            ],
        ),
    ]
}

/// Range-check bus interactions: one `AreBytes` lookup for the two value bytes
/// and one `IsHalfword` lookup per halfword column. They fire on every row
/// (Multiplicity::One), so the matching multiplicities — including the all-zero
/// padding rows — are emitted by [`collect_bitwise_from_l2g`].
///
/// These are committed only on the epoch-local table (`l2g_memory_air`), whose
/// proof carries the BITWISE provider; the global proof commits the identical
/// trace, so its columns inherit the same range guarantee via the commitment
/// binding. Keep this in sync with [`collect_bitwise_from_l2g`].
pub fn range_check_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(1 + cols::HALFWORD_COLUMNS.len());
    interactions.push(BusInteraction::sender(
        BusId::AreBytes,
        Multiplicity::One,
        vec![byte(cols::INIT_VALUE), byte(cols::FINI_VALUE)],
    ));
    for &column in &cols::HALFWORD_COLUMNS {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::One,
            vec![byte(column)],
        ));
    }
    interactions
}

/// The BITWISE lookups the L2G range checks send, so the BITWISE table's
/// multiplicities balance the [`range_check_interactions`] senders. Emits one
/// `AreBytes` and 16 `IsHalfword` ops per row, padded to a power of two with
/// all-zero rows (which still fire, since the senders are unconditional).
pub fn collect_bitwise_from_l2g(boundaries: &[CellBoundary]) -> Vec<BitwiseOperation> {
    let num_rows = boundaries.len().next_power_of_two().max(1);
    let mut ops = Vec::with_capacity(num_rows * (1 + cols::HALFWORD_COLUMNS.len()));

    let push_halfword = |ops: &mut Vec<BitwiseOperation>, v16: u64| {
        ops.push(BitwiseOperation::halfword(
            BitwiseOperationType::IsHalf,
            (v16 & 0xFF) as u8,
            ((v16 >> 8) & 0xFF) as u8,
        ));
    };

    for b in boundaries {
        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::AreBytes,
            (b.init.value & 0xFF) as u8,
            (b.fini.value & 0xFF) as u8,
        ));
        let addr = halfwords64(b.address);
        let init_ts = halfwords64(b.init.timestamp);
        let fini_ts = halfwords64(b.fini.timestamp);
        let init_epoch = epoch_halfwords(b.init.originating_epoch);
        let fini_epoch = epoch_halfwords(b.fini.epoch);
        for v in addr {
            push_halfword(&mut ops, v);
        }
        for v in init_epoch {
            push_halfword(&mut ops, v);
        }
        for v in init_ts {
            push_halfword(&mut ops, v);
        }
        for v in fini_epoch {
            push_halfword(&mut ops, v);
        }
        for v in fini_ts {
            push_halfword(&mut ops, v);
        }
    }

    // Padding rows are all zero: AreBytes(0, 0) + 16 × IsHalfword(0).
    for _ in boundaries.len()..num_rows {
        ops.push(BitwiseOperation::byte_op(
            BitwiseOperationType::AreBytes,
            0,
            0,
        ));
        for _ in 0..cols::HALFWORD_COLUMNS.len() {
            push_halfword(&mut ops, 0);
        }
    }

    ops
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(epoch: &[CellBoundary], address: u64) -> &CellBoundary {
        epoch
            .iter()
            .find(|b| b.address == address)
            .expect("address not found in epoch boundaries")
    }

    #[test]
    fn test_sparse_only_touched_cells() {
        let initial_memory = HashMap::from([(10, 5)]);
        let epochs = vec![
            vec![(10, 7, 3), (20, 9, 4)], // epoch 0 touches 10 and 20
            vec![(10, 8, 10)],            // epoch 1 touches only 10
            vec![(20, 9, 20)],            // epoch 2 touches only 20
        ];
        let boundaries = epoch_boundaries(&initial_memory, &epochs);

        assert_eq!(boundaries.len(), 3);
        // Only touched cells appear, nothing else.
        assert_eq!(boundaries[0].len(), 2);
        assert_eq!(boundaries[1].len(), 1);
        assert_eq!(boundaries[2].len(), 1);
        assert_eq!(boundaries[1][0].address, 10);
        assert_eq!(boundaries[2][0].address, 20);
    }

    #[test]
    fn test_genesis_init_for_first_touch() {
        let initial_memory = HashMap::from([(10, 5)]);
        let epochs = vec![vec![(10, 7, 3), (20, 9, 4)]];
        let boundaries = epoch_boundaries(&initial_memory, &epochs);

        // Cell 10 starts from program memory: value 5, genesis epoch, ts 0.
        let c10 = find(&boundaries[0], 10);
        assert_eq!(
            c10.init,
            InitClaim {
                value: 5,
                originating_epoch: GENESIS_EPOCH,
                timestamp: 0,
            }
        );
        // Cell 20 was never in initial memory: genesis, value 0.
        let c20 = find(&boundaries[0], 20);
        assert_eq!(
            c20.init,
            InitClaim {
                value: 0,
                originating_epoch: GENESIS_EPOCH,
                timestamp: 0,
            }
        );
    }

    #[test]
    fn test_fini_records_current_epoch_value_and_timestamp() {
        let initial_memory = HashMap::from([(10, 5)]);
        let epochs = vec![vec![(10, 7, 3)], vec![(10, 8, 10)]];
        let boundaries = epoch_boundaries(&initial_memory, &epochs);

        assert_eq!(
            find(&boundaries[0], 10).fini,
            FiniClaim {
                value: 7,
                epoch: 0,
                timestamp: 3,
            }
        );
        assert_eq!(
            find(&boundaries[1], 10).fini,
            FiniClaim {
                value: 8,
                epoch: 1,
                timestamp: 10,
            }
        );
    }

    #[test]
    fn test_telescoping_consecutive_epochs() {
        let initial_memory = HashMap::from([(10, 5)]);
        let epochs = vec![vec![(10, 7, 3)], vec![(10, 8, 10)]];
        let boundaries = epoch_boundaries(&initial_memory, &epochs);

        // Epoch 0's fini for cell 10 is consumed as epoch 1's init.
        let fini0 = find(&boundaries[0], 10).fini;
        let init1 = find(&boundaries[1], 10).init;
        assert_eq!(fini0.value, init1.value);
        assert_eq!(fini0.epoch, init1.originating_epoch);
        assert_eq!(fini0.timestamp, init1.timestamp);
        // Concretely: epoch 0 left (7, epoch 0, ts 3).
        assert_eq!(
            init1,
            InitClaim {
                value: 7,
                originating_epoch: 0,
                timestamp: 3,
            }
        );
    }

    #[test]
    fn test_telescoping_skips_untouched_epochs() {
        // Cell 20 is touched in epoch 0, skipped in epoch 1, touched again in 2.
        let initial_memory = HashMap::new();
        let epochs = vec![
            vec![(20, 9, 4)],  // epoch 0 writes 20
            vec![(10, 1, 5)],  // epoch 1 does not touch 20
            vec![(20, 9, 20)], // epoch 2 touches 20 again
        ];
        let boundaries = epoch_boundaries(&initial_memory, &epochs);

        // Epoch 2's init for cell 20 links straight back to epoch 0 (no cost
        // incurred for the epoch that did not touch it).
        let fini0 = find(&boundaries[0], 20).fini;
        let init2 = find(&boundaries[2], 20).init;
        assert_eq!(init2.originating_epoch, 0);
        assert_eq!(init2.value, fini0.value);
        assert_eq!(init2.timestamp, fini0.timestamp);
    }

    fn sample_boundary(address: u64) -> CellBoundary {
        CellBoundary {
            address,
            init: InitClaim {
                value: 0x1_0000_0005,
                originating_epoch: GENESIS_EPOCH,
                timestamp: 0,
            },
            fini: FiniClaim {
                value: 0x2_0000_0007,
                epoch: 1,
                timestamp: 0x3_0000_0009,
            },
        }
    }

    /// Reconstruct a 32-bit value from its two halfword columns, as the bus does.
    fn word_value(
        trace: &TraceTable<GoldilocksField, GoldilocksExtension>,
        lo: usize,
        hi: usize,
    ) -> FE {
        *trace.main_table.get(0, lo) + FE::from(1u64 << 16) * *trace.main_table.get(0, hi)
    }

    #[test]
    fn test_num_columns() {
        assert_eq!(cols::NUM_COLUMNS, 18);
        assert_eq!(cols::HALFWORD_COLUMNS.len(), 16);
    }

    #[test]
    fn test_halfword_columns_reconstruct_to_the_split_values() {
        let b = sample_boundary(0x4_0000_0001);
        let trace = generate_local_to_global_trace(&[b]);

        assert_eq!(trace.num_rows(), 1);

        let lo32 = |v: u64| FE::from(v & 0xFFFF_FFFF);
        let hi32 = |v: u64| FE::from(v >> 32);
        let byte = |v: u64| FE::from(v & 0xFF);

        // Address halfwords reconstruct to address_lo / address_hi.
        assert_eq!(
            word_value(&trace, cols::ADDR_0, cols::ADDR_1),
            lo32(b.address)
        );
        assert_eq!(
            word_value(&trace, cols::ADDR_2, cols::ADDR_3),
            hi32(b.address)
        );
        // Values are stored as single bytes.
        assert_eq!(
            *trace.main_table.get(0, cols::INIT_VALUE),
            byte(b.init.value)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::FINI_VALUE),
            byte(b.fini.value)
        );
        // Timestamps reconstruct to lo / hi 32-bit words.
        assert_eq!(
            word_value(&trace, cols::INIT_TS_0, cols::INIT_TS_1),
            lo32(b.init.timestamp)
        );
        assert_eq!(
            word_value(&trace, cols::INIT_TS_2, cols::INIT_TS_3),
            hi32(b.init.timestamp)
        );
        assert_eq!(
            word_value(&trace, cols::FINI_TS_0, cols::FINI_TS_1),
            lo32(b.fini.timestamp)
        );
        assert_eq!(
            word_value(&trace, cols::FINI_TS_2, cols::FINI_TS_3),
            hi32(b.fini.timestamp)
        );
        // Fini epoch reconstructs to the epoch index.
        assert_eq!(
            word_value(&trace, cols::FINI_EPOCH_0, cols::FINI_EPOCH_1),
            FE::from(b.fini.epoch)
        );
    }

    #[test]
    fn test_genesis_epoch_halfwords_match_global_memory_encoding() {
        // The genesis init-epoch halfwords must reconstruct to FE::from(GENESIS_EPOCH),
        // the exact value global_memory sends on the GlobalMemory bus.
        let b = sample_boundary(0x1000);
        let trace = generate_local_to_global_trace(&[b]);
        assert_eq!(
            word_value(&trace, cols::INIT_EPOCH_0, cols::INIT_EPOCH_1),
            FE::from(GENESIS_EPOCH)
        );
        // And every halfword is genuinely a halfword (< 2^16).
        for &col in &cols::HALFWORD_COLUMNS {
            let raw = *trace.main_table.get(0, col).value();
            assert!(raw < (1 << 16), "column {col} is not a halfword: {raw}");
        }
    }

    #[test]
    fn test_trace_padded_to_power_of_two_with_zero_rows() {
        // 3 boundaries pad up to 4 rows; the padding row is all zero.
        let boundaries: Vec<CellBoundary> = (0..3).map(sample_boundary).collect();
        let trace = generate_local_to_global_trace(&boundaries);
        assert_eq!(trace.num_rows(), 4);
        for col in 0..cols::NUM_COLUMNS {
            assert_eq!(*trace.main_table.get(3, col), FE::zero());
        }
    }

    #[test]
    fn test_empty_trace_is_padded_to_one_row() {
        let trace = generate_local_to_global_trace(&[]);
        assert_eq!(trace.num_rows(), 1);
        for col in 0..cols::NUM_COLUMNS {
            assert_eq!(*trace.main_table.get(0, col), FE::zero());
        }
    }

    #[test]
    fn test_bus_interactions() {
        let interactions = bus_interactions();
        assert_eq!(interactions.len(), 2); // init (receive) + fini (send)

        let global_memory = u64::from(BusId::GlobalMemory);
        let init = &interactions[0];
        let fini = &interactions[1];

        // init consumes the originating epoch's token; fini produces this epoch's.
        assert!(!init.is_sender);
        assert!(fini.is_sender);
        assert_eq!(init.bus_id, global_memory);
        assert_eq!(fini.bus_id, global_memory);

        // Both tokens have the same 6-element shape so they can match across
        // epochs: address(lo,hi), value(byte), epoch, timestamp(lo,hi).
        assert_eq!(init.values.len(), 6);
        assert_eq!(fini.values.len(), 6);
    }

    #[test]
    fn test_range_check_interactions_cover_every_column() {
        let interactions = range_check_interactions();
        // 1 AreBytes (two value bytes) + one IsHalfword per halfword column.
        assert_eq!(interactions.len(), 1 + cols::HALFWORD_COLUMNS.len());
        let are_bytes = u64::from(BusId::AreBytes);
        let is_halfword = u64::from(BusId::IsHalfword);
        assert_eq!(interactions[0].bus_id, are_bytes);
        assert_eq!(interactions[0].values.len(), 2);
        for interaction in &interactions[1..] {
            assert!(interaction.is_sender);
            assert_eq!(interaction.bus_id, is_halfword);
            assert_eq!(interaction.values.len(), 1);
        }
    }

    #[test]
    fn test_collect_bitwise_matches_sender_count() {
        // One AreBytes + 16 IsHalfword per row, padded to a power of two.
        let boundaries: Vec<CellBoundary> = (0..3).map(sample_boundary).collect();
        let ops = collect_bitwise_from_l2g(&boundaries);
        let num_rows = 4; // 3 padded to 4
        let per_row = 1 + cols::HALFWORD_COLUMNS.len();
        assert_eq!(ops.len(), num_rows * per_row);

        let are_bytes = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::AreBytes)
            .count();
        let is_half = ops
            .iter()
            .filter(|o| o.lookup_type == BitwiseOperationType::IsHalf)
            .count();
        assert_eq!(are_bytes, num_rows);
        assert_eq!(is_half, num_rows * cols::HALFWORD_COLUMNS.len());
    }

    #[test]
    fn test_collect_bitwise_values_match_the_committed_halfword_columns() {
        // Each IsHalfword op the collector emits must carry the same value as the
        // corresponding halfword column the range-check sender reads, so the bus
        // balances on the right BITWISE rows (not just the right counts). Use a
        // boundary with distinct values in every quantity.
        let b = CellBoundary {
            address: 0x1234_5678_9abc_def0,
            init: InitClaim {
                value: 0xAB,
                originating_epoch: 0x0011_2233,
                timestamp: 0x4455_6677_8899_aabb,
            },
            fini: FiniClaim {
                value: 0xCD,
                epoch: 0x00aa_00bb,
                timestamp: 0xccdd_eeff_0011_2233,
            },
        };
        let trace = generate_local_to_global_trace(&[b]);
        let ops = collect_bitwise_from_l2g(&[b]);

        // The single AreBytes op carries the two value bytes.
        assert_eq!(ops[0].lookup_type, BitwiseOperationType::AreBytes);
        assert_eq!(ops[0].x as u64, b.init.value & 0xFF);
        assert_eq!(ops[0].y as u64, b.fini.value & 0xFF);

        // The 16 IsHalfword ops follow, in HALFWORD_COLUMNS order, each matching
        // the value committed in that column.
        for (i, &col) in cols::HALFWORD_COLUMNS.iter().enumerate() {
            let op = &ops[1 + i];
            assert_eq!(op.lookup_type, BitwiseOperationType::IsHalf);
            let op_value = op.x as u64 + ((op.y as u64) << 8);
            assert_eq!(
                FE::from(op_value),
                *trace.main_table.get(0, col),
                "IsHalfword op {i} value disagrees with column {col}"
            );
        }
    }
}
