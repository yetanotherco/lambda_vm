//! Local-to-global memory boundary claims for cross-epoch continuations.
//!
//! Each epoch, for every memory cell it touches,
//! makes an `init` claim (the cell's value when first touched this epoch, which
//! earlier epoch last wrote it, and that write's timestamp) and a `fini` claim
//! (the cell's value at this epoch's end, this epoch's index, and the last
//! access timestamp). A final LogUp matches each `fini` against the `init` of the
//! next epoch that touches the same cell, proving global memory consistency.
//!
//! ## Epoch labels
//!
//! Epochs are labelled 1-based (epoch index `i` → label `i+1`) and the genesis
//! sentinel is `0` ([`GENESIS_EPOCH`]). This makes "the originating epoch is
//! strictly earlier" a plain `init_epoch < fini_epoch` — genesis (`0`) is below
//! every real epoch, so it needs no special case.
//!
//! ## Ordering constraint
//!
//! The GlobalMemory LogUp only proves the init/fini tokens *match as a set*; it
//! does not by itself force the chain to be consumed in increasing-epoch order.
//! Without that, a prover could let an init consume a *later* epoch's fini (a
//! backward/self edge), seeding a cell with an unjustified value. So each real
//! row also proves `init_epoch < fini_epoch` via an `IsB20` lookup on
//! `fini_epoch − 1 − init_epoch` (it must be a valid 20-bit value). This bounds
//! the number of epochs to `< 2^20` (~1M) — unreachable in practice (optimal
//! epochs are millions of cycles, so thousands of epochs) and fails closed.
//!
//! ## Range-checked columns
//!
//! A column needs an explicit range check only if nothing else already pins it.
//! Most L2G columns travel on the epoch-local `Memory` bus and are matched there
//! against MEMW, which already range/order-checks address, timestamp and value —
//! exactly how PAGE relies on MEMW in the monolithic prover. So `address` and
//! `fini_timestamp` are plain 32-bit columns with no extra check, and the value
//! bytes get the same batched `AreBytes` check PAGE uses (the `init` value is a
//! trusted source, so it must be checked). `fini_epoch` is the same constant for
//! every row of an epoch's table, so it is supplied as a per-table constant (not
//! a column) by [`bus_interactions`].
//!
//! The only column that lives ONLY on the cross-epoch `GlobalMemory` bus has no
//! MEMW partner: `init_epoch`. It is stored as two 16-bit halfword columns, each
//! checked via `IsHalfword`, and the 32-bit bus value is rebuilt from them by a
//! linear combination (see [`word`]). The checks are emitted on the epoch-local
//! table (which has the BITWISE provider); the global proof commits the identical
//! trace (the commitment binding compares their roots), so it inherits the same
//! guarantee. There is no `init_timestamp` column: timestamps are epoch-local, and
//! the cross-epoch chain is ordered by epoch.
//!
//! ## Padding
//!
//! Real rows carry `MU = 1`; the power-of-two padding rows carry `MU = 0`. Every
//! interaction uses `Multiplicity::Column(MU)`, so padding rows fire nothing —
//! we never rely on token self-cancellation (this is the standard pattern used
//! by every variable-length table). `MU` is self-enforced: dropping a real row
//! (`MU = 0`) breaks its telescoping link → bus imbalance.

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
/// initial memory — no prior epoch wrote them. Chosen as `0`, below every real
/// (1-based) epoch label, so `init_epoch < fini_epoch` holds for genesis cells.
pub const GENESIS_EPOCH: u64 = 0;

/// Maximum number of epochs a continuation run may have.
///
/// The cross-epoch ordering check proves `init_epoch < fini_epoch` via an `IsB20`
/// (20-bit) lookup on `fini_epoch - 1 - init_epoch`. A genesis-sourced cell
/// finalized in epoch `index` (0-based) has gap `index`, so every epoch must
/// satisfy `index < 2^20`. A run needing more epochs cannot be proved — the
/// IsB20 bus would not balance — so the driver rejects it up front (see
/// `prove_continuation`).
pub const MAX_EPOCHS: u64 = 1 << 20;

/// A cell's state when an epoch first touches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct InitClaim {
    /// Value the cell held when this epoch first touched it.
    pub value: u64,
    /// Epoch that last wrote the cell (or [`GENESIS_EPOCH`]).
    pub originating_epoch: u64,
    /// Timestamp of that originating write.
    pub timestamp: u64,
}

/// A cell's state at the end of the epoch that touched it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct FiniClaim {
    /// Value the cell holds at this epoch's end.
    pub value: u64,
    /// This epoch's label (1-based).
    pub epoch: u64,
    /// Last access timestamp for the cell this epoch.
    pub timestamp: u64,
}

/// The init/fini boundary claims for a single touched cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CellBoundary {
    pub address: u64,
    pub init: InitClaim,
    pub fini: FiniClaim,
}

/// One epoch's touched cells, each as `(address, end_value, end_timestamp)`.
pub type EpochTouches = Vec<(u64, u64, u64)>;

/// Convert a 0-based epoch index into its 1-based table label.
pub fn epoch_label(epoch_index: u64) -> u64 {
    epoch_index + 1
}

/// Compute the sparse per-epoch boundary claims.
///
/// `initial_memory` maps each address to its program-start value (originating
/// epoch [`GENESIS_EPOCH`], timestamp 0). `epochs[e]` lists the cells touched in
/// epoch `e` with their end value and end timestamp. Returns, per epoch, the
/// boundary claims for exactly the cells that epoch touched (sparse): each
/// cell's `init` is taken from the previous epoch that wrote it, and its `fini`
/// records this epoch (1-based label) as the new writer.
pub fn epoch_boundaries(
    initial_memory: &HashMap<u64, u64>,
    epochs: &[EpochTouches],
) -> Vec<Vec<CellBoundary>> {
    // provenance[addr] = (last_writer_epoch, value, timestamp)
    let mut provenance = genesis_provenance(initial_memory.iter().map(|(&a, &v)| (a, v)));

    let mut result = Vec::with_capacity(epochs.len());
    for (epoch, touched) in epochs.iter().enumerate() {
        result.push(epoch_boundary(
            &mut provenance,
            epoch_label(epoch as u64),
            touched,
        ));
    }
    result
}

/// One epoch's boundaries, taking `init` from the running `provenance` (the cell's
/// last writer) and updating `provenance` with this epoch's `fini`. `epoch` is the
/// 1-based label. This is the per-epoch step of [`epoch_boundaries`], exposed so
/// the streaming continuation prover can build each epoch's table incrementally
/// without all epochs at once.
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

/// Seed the provenance store from the program's initial memory (genesis cells),
/// supplied as an `(address, value)` iterator. The continuation prover feeds the
/// paged genesis image directly, avoiding an intermediate address→value map.
pub fn genesis_provenance(genesis: impl IntoIterator<Item = (u64, u64)>) -> Provenance {
    let mut provenance = Provenance::new((GENESIS_EPOCH, 0, 0));
    for (addr, value) in genesis {
        provenance.set(addr, (GENESIS_EPOCH, value, 0));
    }
    provenance
}

// =========================================================================
// AIR trace columns
// =========================================================================

/// Column indices for the local-to-global table: one row per touched cell.
///
/// `address` and `fini_timestamp` are plain 32-bit columns (matched on the Memory
/// bus against MEMW). The cross-epoch-only `init_epoch` is stored as 16-bit
/// halfword columns ([`RANGE_CHECKED_HALFWORDS`]), checked via `IsHalfword`, and
/// rebuilt into its 32-bit bus value via [`word`]. The value bytes get the
/// batched `AreBytes` check. `fini_epoch` is a per-table constant (not a column).
/// `MU` is the real-row selector / multiplicity.
pub mod cols {
    /// address_lo: 32-bit; matched on the Memory bus against MEMW.
    pub const ADDRESS_LO: usize = 0;
    /// address_hi: 32-bit; matched on the Memory bus against MEMW.
    pub const ADDRESS_HI: usize = 1;

    /// Init value: a single byte, like PAGE's `value`.
    pub const INIT_VALUE: usize = 2;

    // Init epoch — GlobalMemory-bus only, range-checked: two halfwords
    // (`init_epoch = INIT_EPOCH_0 + 2^16·INIT_EPOCH_1`).
    pub const INIT_EPOCH_0: usize = 3;
    pub const INIT_EPOCH_1: usize = 4;

    // Note: there is no init-timestamp column. Timestamps are epoch-local ordering
    // scratch (the Memory-bus init token is seeded at ts=0); across epochs the chain
    // is ordered by `init_epoch < fini_epoch`, so the GlobalMemory bus carries no
    // timestamp at all (see `bus_interactions`).

    /// Fini value: a single byte.
    pub const FINI_VALUE: usize = 5;

    /// fini_timestamp_lo: 32-bit; matched on the Memory bus against MEMW.
    pub const FINI_TIMESTAMP_LO: usize = 6;
    /// fini_timestamp_hi: 32-bit; matched on the Memory bus against MEMW.
    pub const FINI_TIMESTAMP_HI: usize = 7;

    /// MU: real-row selector / LogUp multiplicity (1 on real rows, 0 on padding).
    pub const MU: usize = 8;

    pub const NUM_COLUMNS: usize = 9;

    /// The halfword columns (cross-epoch-only quantities), in order — every column
    /// that is `IsHalfword`-checked.
    pub const RANGE_CHECKED_HALFWORDS: [usize; 2] = [INIT_EPOCH_0, INIT_EPOCH_1];
}

/// The two halfwords of an epoch label (genesis `0` or a small 1-based index, all
/// well under 2^32).
fn epoch_halfwords(epoch: u64) -> [u64; 2] {
    debug_assert!(epoch < (1 << 32), "epoch label exceeds 32 bits");
    [epoch & 0xFFFF, (epoch >> 16) & 0xFFFF]
}

// =========================================================================
// Trace generation
// =========================================================================

/// Build the local-to-global trace: one row per touched cell's boundary claims,
/// padded up to a power of two. Real rows set `MU = 1`; padding rows stay all-zero
/// (`MU = 0`), so they fire no interactions.
pub fn generate_local_to_global_trace(
    boundaries: &[CellBoundary],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = boundaries.len().next_power_of_two().max(1);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row, b) in boundaries.iter().enumerate() {
        let base = row * cols::NUM_COLUMNS;
        let init_epoch = epoch_halfwords(b.init.originating_epoch);

        // Plain 32-bit columns (MEMW-checked on the Memory bus).
        data[base + cols::ADDRESS_LO] = FE::from(b.address & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_HI] = FE::from(b.address >> 32);
        data[base + cols::FINI_TIMESTAMP_LO] = FE::from(b.fini.timestamp & 0xFFFF_FFFF);
        data[base + cols::FINI_TIMESTAMP_HI] = FE::from(b.fini.timestamp >> 32);
        // Byte values (AreBytes-checked).
        data[base + cols::INIT_VALUE] = FE::from(b.init.value & 0xFF);
        data[base + cols::FINI_VALUE] = FE::from(b.fini.value & 0xFF);
        // Cross-epoch-only quantity as IsHalfword-checked halfwords.
        data[base + cols::INIT_EPOCH_0] = FE::from(init_epoch[0]);
        data[base + cols::INIT_EPOCH_1] = FE::from(init_epoch[1]);
        // Real-row selector.
        data[base + cols::MU] = FE::one();
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

/// A column read directly as a single field element (a 32-bit word or a byte).
pub(crate) fn direct(column: usize) -> BusValue {
    BusValue::Packed {
        start_column: column,
        packing: Packing::Direct,
    }
}

fn mu() -> Multiplicity {
    Multiplicity::Column(cols::MU)
}

/// Cross-epoch memory bus interactions, two per row (one touched cell):
/// - **receive** the `init` token `(address, value, originating_epoch)` left by the
///   epoch that last wrote the cell;
/// - **send** the `fini` token `(address, value, epoch_label)` for the next epoch
///   that touches the cell.
///
/// `fini_epoch` is the per-table constant `epoch_label`; `init_epoch` comes from the
/// range-checked halfword columns via [`word`]; `address` is direct 32-bit columns.
/// No timestamp is carried: the chain is ordered by epoch, and timestamps are
/// epoch-local (only the Memory bus, not this one, uses them).
///
/// These tokens are matched ACROSS epochs by the final aggregation LogUp (step 4),
/// so within a single epoch's table the GlobalMemory bus is deliberately
/// unbalanced (real rows have `init != fini`). Padding rows fire nothing (`MU = 0`).
pub fn bus_interactions(epoch_label: u64) -> Vec<BusInteraction> {
    vec![
        // init: receive the token left by the originating epoch. No timestamp: the
        // chain is ordered by epoch, and timestamps are epoch-local (see cols).
        BusInteraction::receiver(
            BusId::GlobalMemory,
            mu(),
            vec![
                direct(cols::ADDRESS_LO),
                direct(cols::ADDRESS_HI),
                direct(cols::INIT_VALUE),
                word(cols::INIT_EPOCH_0, cols::INIT_EPOCH_1),
            ],
        ),
        // fini: send the token for the next epoch to consume.
        BusInteraction::sender(
            BusId::GlobalMemory,
            mu(),
            vec![
                direct(cols::ADDRESS_LO),
                direct(cols::ADDRESS_HI),
                direct(cols::FINI_VALUE),
                BusValue::constant(epoch_label),
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
///
/// Address, fini timestamp and the values appear here, so MEMW range-checks them
/// for us — they need no L2G range check (see [`range_check_interactions`]).
pub fn memory_bus_interactions() -> Vec<BusInteraction> {
    vec![
        // init: receive the cell's initial token at the epoch-start seed (ts = 0).
        BusInteraction::receiver(
            BusId::Memory,
            mu(),
            vec![
                BusValue::constant(0),
                direct(cols::ADDRESS_LO),
                direct(cols::ADDRESS_HI),
                BusValue::constant(0),
                BusValue::constant(0),
                direct(cols::INIT_VALUE),
            ],
        ),
        // fini: send the cell's final token at the last access timestamp.
        BusInteraction::sender(
            BusId::Memory,
            mu(),
            vec![
                BusValue::constant(0),
                direct(cols::ADDRESS_LO),
                direct(cols::ADDRESS_HI),
                direct(cols::FINI_TIMESTAMP_LO),
                direct(cols::FINI_TIMESTAMP_HI),
                direct(cols::FINI_VALUE),
            ],
        ),
    ]
}

/// Range-check + ordering bus interactions for the columns nothing else
/// constrains, all with multiplicity `MU` (so padding fires none):
/// - one `AreBytes` for the two value bytes (the `init` value is a trusted source);
/// - one `IsHalfword` per cross-epoch-only halfword column;
/// - one `IsB20` proving `init_epoch < fini_epoch` (the ordering constraint), via
///   `fini_epoch − 1 − init_epoch` being a valid 20-bit value. With genesis epoch
///   `0` this also covers genesis cells (`0 < fini_epoch`) with no special case.
///
/// Address and fini timestamp are NOT here — MEMW checks them on the Memory bus.
/// These are committed only on the epoch-local table (`l2g_memory_air`), whose
/// proof carries the BITWISE provider; the global proof commits the identical
/// trace, so its columns inherit the same guarantee via the commitment binding.
/// Keep this in sync with [`collect_bitwise_from_l2g`].
pub fn range_check_interactions(epoch_label: u64) -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(2 + cols::RANGE_CHECKED_HALFWORDS.len());
    interactions.push(BusInteraction::sender(
        BusId::AreBytes,
        mu(),
        vec![direct(cols::INIT_VALUE), direct(cols::FINI_VALUE)],
    ));
    for &column in &cols::RANGE_CHECKED_HALFWORDS {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            mu(),
            vec![direct(column)],
        ));
    }
    // Ordering: IsB20[epoch_label - 1 - init_epoch], where
    // init_epoch = INIT_EPOCH_0 + 2^16·INIT_EPOCH_1.
    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        mu(),
        vec![BusValue::linear(vec![
            LinearTerm::Constant(epoch_label as i64 - 1),
            LinearTerm::Column {
                coefficient: -1,
                column: cols::INIT_EPOCH_0,
            },
            LinearTerm::Column {
                coefficient: -(1 << 16),
                column: cols::INIT_EPOCH_1,
            },
        ])],
    ));
    interactions
}

/// The BITWISE lookups the L2G range checks + ordering check send, so the BITWISE
/// table's multiplicities balance the [`range_check_interactions`] senders. Emits,
/// per real row, one `AreBytes`, one `IsHalfword` per cross-epoch halfword, and one
/// `IsB20` for the ordering difference. Padding rows fire nothing (`MU = 0`), so
/// none are emitted for them.
pub fn collect_bitwise_from_l2g(boundaries: &[CellBoundary]) -> Vec<BitwiseOperation> {
    let per_row = 2 + cols::RANGE_CHECKED_HALFWORDS.len();
    let mut ops = Vec::with_capacity(boundaries.len() * per_row);

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
        let init_epoch = epoch_halfwords(b.init.originating_epoch);
        for v in init_epoch {
            push_halfword(&mut ops, v);
        }
        // Ordering: IsB20[fini_epoch - 1 - init_epoch]. Honest rows have
        // init_epoch < fini_epoch, so the difference is a small non-negative value.
        let diff = b.fini.epoch - 1 - b.init.originating_epoch;
        debug_assert!(diff < MAX_EPOCHS, "epoch gap exceeds IsB20 range");
        ops.push(BitwiseOperation::b20(
            (diff & 0xFF) as u8,
            ((diff >> 8) & 0xFF) as u8,
            ((diff >> 16) & 0xF) as u8,
        ));
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
    fn test_fini_records_current_epoch_label_and_timestamp() {
        let initial_memory = HashMap::from([(10, 5)]);
        let epochs = vec![vec![(10, 7, 3)], vec![(10, 8, 10)]];
        let boundaries = epoch_boundaries(&initial_memory, &epochs);

        // Labels are 1-based: epoch index 0 → label 1, index 1 → label 2.
        assert_eq!(
            find(&boundaries[0], 10).fini,
            FiniClaim {
                value: 7,
                epoch: 1,
                timestamp: 3,
            }
        );
        assert_eq!(
            find(&boundaries[1], 10).fini,
            FiniClaim {
                value: 8,
                epoch: 2,
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
        // Concretely: epoch 0 (label 1) left (7, label 1, ts 3).
        assert_eq!(
            init1,
            InitClaim {
                value: 7,
                originating_epoch: 1,
                timestamp: 3,
            }
        );
        // And init_epoch (1) < fini_epoch (2), the ordering invariant.
        assert!(init1.originating_epoch < find(&boundaries[1], 10).fini.epoch);
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

        // Epoch 2's init for cell 20 links straight back to epoch 0 (label 1).
        let fini0 = find(&boundaries[0], 20).fini;
        let init2 = find(&boundaries[2], 20).init;
        assert_eq!(init2.originating_epoch, 1);
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
        assert_eq!(cols::NUM_COLUMNS, 9);
        assert_eq!(cols::RANGE_CHECKED_HALFWORDS.len(), 2);
    }

    #[test]
    fn test_columns_hold_the_split_values() {
        let b = sample_boundary(0x4_0000_0001);
        let trace = generate_local_to_global_trace(&[b]);

        assert_eq!(trace.num_rows(), 1);

        let lo32 = |v: u64| FE::from(v & 0xFFFF_FFFF);
        let hi32 = |v: u64| FE::from(v >> 32);
        let byte = |v: u64| FE::from(v & 0xFF);
        let at = |c: usize| *trace.main_table.get(0, c);

        // Address and fini timestamp are plain 32-bit columns (MEMW-checked).
        assert_eq!(at(cols::ADDRESS_LO), lo32(b.address));
        assert_eq!(at(cols::ADDRESS_HI), hi32(b.address));
        assert_eq!(at(cols::FINI_TIMESTAMP_LO), lo32(b.fini.timestamp));
        assert_eq!(at(cols::FINI_TIMESTAMP_HI), hi32(b.fini.timestamp));
        // Values are stored as single bytes.
        assert_eq!(at(cols::INIT_VALUE), byte(b.init.value));
        assert_eq!(at(cols::FINI_VALUE), byte(b.fini.value));
        // The cross-epoch-only quantity reconstructs from its halfwords.
        // Genesis init epoch reconstructs to 0 (== GENESIS_EPOCH).
        assert_eq!(
            word_value(&trace, cols::INIT_EPOCH_0, cols::INIT_EPOCH_1),
            FE::from(GENESIS_EPOCH)
        );
        // Real row carries MU = 1.
        assert_eq!(at(cols::MU), FE::one());
    }

    #[test]
    fn test_padding_rows_are_zero_including_mu() {
        // 3 boundaries pad up to 4 rows; the padding row is all zero, MU = 0.
        let boundaries: Vec<CellBoundary> = (0..3).map(sample_boundary).collect();
        let trace = generate_local_to_global_trace(&boundaries);
        assert_eq!(trace.num_rows(), 4);
        for col in 0..cols::NUM_COLUMNS {
            assert_eq!(*trace.main_table.get(3, col), FE::zero());
        }
        // And real rows have MU = 1.
        for row in 0..3 {
            assert_eq!(*trace.main_table.get(row, cols::MU), FE::one());
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
        let interactions = bus_interactions(1);
        assert_eq!(interactions.len(), 2); // init (receive) + fini (send)

        let global_memory = u64::from(BusId::GlobalMemory);
        let init = &interactions[0];
        let fini = &interactions[1];

        // init consumes the originating epoch's token; fini produces this epoch's.
        assert!(!init.is_sender);
        assert!(fini.is_sender);
        assert_eq!(init.bus_id, global_memory);
        assert_eq!(fini.bus_id, global_memory);

        // Both tokens have the same 4-element shape so they can match across
        // epochs: address(lo,hi), value(byte), epoch. No timestamp — the chain is
        // ordered by epoch, and timestamps are epoch-local.
        assert_eq!(init.values.len(), 4);
        assert_eq!(fini.values.len(), 4);
    }

    #[test]
    fn test_range_check_interactions_cover_every_column() {
        let interactions = range_check_interactions(1);
        // 1 AreBytes + one IsHalfword per cross-epoch halfword + 1 IsB20 ordering.
        assert_eq!(interactions.len(), 2 + cols::RANGE_CHECKED_HALFWORDS.len());
        let are_bytes = u64::from(BusId::AreBytes);
        let is_halfword = u64::from(BusId::IsHalfword);
        let is_b20 = u64::from(BusId::IsB20);
        assert_eq!(interactions[0].bus_id, are_bytes);
        assert_eq!(interactions[0].values.len(), 2);
        for interaction in &interactions[1..1 + cols::RANGE_CHECKED_HALFWORDS.len()] {
            assert!(interaction.is_sender);
            assert_eq!(interaction.bus_id, is_halfword);
            assert_eq!(interaction.values.len(), 1);
        }
        let ordering = interactions.last().unwrap();
        assert!(ordering.is_sender);
        assert_eq!(ordering.bus_id, is_b20);
    }

    #[test]
    fn test_collect_bitwise_matches_sender_count() {
        // Per row: 1 AreBytes + one IsHalfword per cross-epoch halfword + 1 IsB20.
        // No padding ops (padding has MU = 0 and fires nothing).
        let boundaries: Vec<CellBoundary> = (0..3).map(sample_boundary).collect();
        let ops = collect_bitwise_from_l2g(&boundaries);
        let per_row = 2 + cols::RANGE_CHECKED_HALFWORDS.len();
        assert_eq!(ops.len(), boundaries.len() * per_row);

        let count = |t: BitwiseOperationType| ops.iter().filter(|o| o.lookup_type == t).count();
        assert_eq!(count(BitwiseOperationType::AreBytes), boundaries.len());
        assert_eq!(
            count(BitwiseOperationType::IsHalf),
            boundaries.len() * cols::RANGE_CHECKED_HALFWORDS.len()
        );
        assert_eq!(count(BitwiseOperationType::IsB20), boundaries.len());
    }

    #[test]
    fn test_collect_bitwise_values_match_the_committed_halfword_columns() {
        // Each IsHalfword op the collector emits must carry the same value as the
        // corresponding halfword column the range-check sender reads. Use a
        // boundary with distinct values, and a real (>=1) originating epoch.
        let b = CellBoundary {
            address: 0x1234_5678_9abc_def0,
            init: InitClaim {
                value: 0xAB,
                originating_epoch: 3,
                timestamp: 0x4455_6677_8899_aabb,
            },
            fini: FiniClaim {
                value: 0xCD,
                epoch: 9,
                timestamp: 0xccdd_eeff_0011_2233,
            },
        };
        let trace = generate_local_to_global_trace(&[b]);
        let ops = collect_bitwise_from_l2g(&[b]);

        // The single AreBytes op carries the two value bytes.
        assert_eq!(ops[0].lookup_type, BitwiseOperationType::AreBytes);
        assert_eq!(ops[0].x as u64, b.init.value & 0xFF);
        assert_eq!(ops[0].y as u64, b.fini.value & 0xFF);

        // The IsHalfword ops follow, in RANGE_CHECKED_HALFWORDS order, each
        // matching the value committed in that column.
        for (i, &col) in cols::RANGE_CHECKED_HALFWORDS.iter().enumerate() {
            let op = &ops[1 + i];
            assert_eq!(op.lookup_type, BitwiseOperationType::IsHalf);
            let op_value = op.x as u64 + ((op.y as u64) << 8);
            assert_eq!(
                FE::from(op_value),
                *trace.main_table.get(0, col),
                "IsHalfword op {i} value disagrees with column {col}"
            );
        }

        // The last op is the ordering IsB20 of `fini_epoch - 1 - init_epoch`.
        let ordering = ops.last().unwrap();
        assert_eq!(ordering.lookup_type, BitwiseOperationType::IsB20);
        let value = ordering.x as u64 + ((ordering.y as u64) << 8) + ((ordering.z as u64) << 16);
        assert_eq!(value, b.fini.epoch - 1 - b.init.originating_epoch);
    }

    #[test]
    fn test_ordering_rejects_future_reference() {
        // The ordering sender computes the field value `fini_epoch - 1 - init_epoch`.
        // For an honest row (init_epoch < fini_epoch) it's a small valid IsB20 value;
        // for a forged FUTURE reference (init_epoch >= fini_epoch) it underflows in
        // the field to a value far outside [0, 2^20), so no IsB20 row matches and the
        // bus cannot balance.
        let order_value = |fini_label: u64, init_epoch: u64| -> FE {
            FE::from(fini_label - 1) - FE::from(init_epoch)
        };

        // Honest: epoch 5 consuming epoch 2's fini → 5 - 1 - 2 = 2, in range.
        let honest = order_value(5, 2);
        assert!(*honest.value() < (1 << 20));

        // Forged future reference: epoch 5's init claims originating epoch 9.
        let forged = order_value(5, 9);
        assert!(
            *forged.value() >= (1 << 20),
            "a future-epoch reference must fall outside the IsB20 range"
        );

        // Forged self reference: epoch 5's init claims originating epoch 5.
        // 5 - 1 - 5 = -1 in the field → also out of range.
        let self_ref = order_value(5, 5);
        assert!(*self_ref.value() >= (1 << 20));
    }
}
