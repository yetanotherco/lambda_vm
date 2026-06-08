//! Local-to-global memory boundary claims for cross-epoch continuations.
//!
//! Each epoch, for every memory cell it touches,
//! makes an `init` claim (the cell's value when first touched this epoch, which
//! earlier epoch last wrote it, and that write's timestamp) and a `fini` claim
//! (the cell's value at this epoch's end, this epoch's index, and the last
//! access timestamp). A final LogUp matches each `fini` against the `init` of the
//! next epoch that touches the same cell, proving global memory consistency.
//!
//! This module currently provides only the boundary-claim data and the
//! provenance/telescoping logic. The AIR table, bus, and prover wiring come in
//! later steps.

use std::collections::HashMap;

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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
    let mut provenance: HashMap<u64, (u64, u64, u64)> = initial_memory
        .iter()
        .map(|(&addr, &value)| (addr, (GENESIS_EPOCH, value, 0)))
        .collect();

    let mut result = Vec::with_capacity(epochs.len());
    for (epoch, touched) in epochs.iter().enumerate() {
        let epoch = epoch as u64;
        let mut boundaries = Vec::with_capacity(touched.len());
        for &(address, end_value, end_timestamp) in touched {
            let (originating_epoch, init_value, init_timestamp) = provenance
                .get(&address)
                .copied()
                .unwrap_or((GENESIS_EPOCH, 0, 0));
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
            provenance.insert(address, (epoch, end_value, end_timestamp));
        }
        result.push(boundaries);
    }
    result
}

// =========================================================================
// AIR trace columns
// =========================================================================

/// Column indices for the local-to-global table: one row per touched cell.
/// Each `u64` field is split into lo/hi 32-bit words (a full `u64` does not fit
/// a single Goldilocks element).
pub mod cols {
    pub const ADDRESS_LO: usize = 0;
    pub const ADDRESS_HI: usize = 1;
    pub const INIT_VALUE_LO: usize = 2;
    pub const INIT_VALUE_HI: usize = 3;
    /// Epoch is a small counter — a single column (no hi word).
    pub const INIT_EPOCH: usize = 4;
    pub const INIT_TIMESTAMP_LO: usize = 5;
    pub const INIT_TIMESTAMP_HI: usize = 6;
    pub const FINI_VALUE_LO: usize = 7;
    pub const FINI_VALUE_HI: usize = 8;
    pub const FINI_EPOCH: usize = 9;
    pub const FINI_TIMESTAMP_LO: usize = 10;
    pub const FINI_TIMESTAMP_HI: usize = 11;

    pub const NUM_COLUMNS: usize = 12;
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
        data[base + cols::ADDRESS_LO] = FE::from(b.address & 0xFFFF_FFFF);
        data[base + cols::ADDRESS_HI] = FE::from(b.address >> 32);
        data[base + cols::INIT_VALUE_LO] = FE::from(b.init.value & 0xFFFF_FFFF);
        data[base + cols::INIT_VALUE_HI] = FE::from(b.init.value >> 32);
        data[base + cols::INIT_EPOCH] = FE::from(b.init.originating_epoch);
        data[base + cols::INIT_TIMESTAMP_LO] = FE::from(b.init.timestamp & 0xFFFF_FFFF);
        data[base + cols::INIT_TIMESTAMP_HI] = FE::from(b.init.timestamp >> 32);
        data[base + cols::FINI_VALUE_LO] = FE::from(b.fini.value & 0xFFFF_FFFF);
        data[base + cols::FINI_VALUE_HI] = FE::from(b.fini.value >> 32);
        data[base + cols::FINI_EPOCH] = FE::from(b.fini.epoch);
        data[base + cols::FINI_TIMESTAMP_LO] = FE::from(b.fini.timestamp & 0xFFFF_FFFF);
        data[base + cols::FINI_TIMESTAMP_HI] = FE::from(b.fini.timestamp >> 32);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

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
                BusValue::Packed {
                    start_column: cols::ADDRESS_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_HI,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::INIT_VALUE_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::INIT_VALUE_HI,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::INIT_EPOCH,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::INIT_TIMESTAMP_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::INIT_TIMESTAMP_HI,
                    packing: Packing::Direct,
                },
            ],
        ),
        // fini: send the token for the next epoch to consume.
        BusInteraction::sender(
            BusId::GlobalMemory,
            Multiplicity::One,
            vec![
                BusValue::Packed {
                    start_column: cols::ADDRESS_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::ADDRESS_HI,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::FINI_VALUE_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::FINI_VALUE_HI,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::FINI_EPOCH,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::FINI_TIMESTAMP_LO,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::FINI_TIMESTAMP_HI,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
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

    #[test]
    fn test_num_columns() {
        assert_eq!(cols::NUM_COLUMNS, 12);
    }

    #[test]
    fn test_trace_columns_hold_lo_hi_split_values() {
        let b = sample_boundary(0x4_0000_0001);
        let trace = generate_local_to_global_trace(&[b]);

        // One row padded up to the next power of two (1).
        assert_eq!(trace.num_rows(), 1);

        let lo = |v: u64| FE::from(v & 0xFFFF_FFFF);
        let hi = |v: u64| FE::from(v >> 32);

        assert_eq!(*trace.main_table.get(0, cols::ADDRESS_LO), lo(b.address));
        assert_eq!(*trace.main_table.get(0, cols::ADDRESS_HI), hi(b.address));
        assert_eq!(
            *trace.main_table.get(0, cols::INIT_VALUE_LO),
            lo(b.init.value)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::INIT_VALUE_HI),
            hi(b.init.value)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::INIT_EPOCH),
            FE::from(b.init.originating_epoch)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::INIT_TIMESTAMP_LO),
            lo(b.init.timestamp)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::INIT_TIMESTAMP_HI),
            hi(b.init.timestamp)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::FINI_VALUE_LO),
            lo(b.fini.value)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::FINI_VALUE_HI),
            hi(b.fini.value)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::FINI_EPOCH),
            FE::from(b.fini.epoch)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::FINI_TIMESTAMP_LO),
            lo(b.fini.timestamp)
        );
        assert_eq!(
            *trace.main_table.get(0, cols::FINI_TIMESTAMP_HI),
            hi(b.fini.timestamp)
        );
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

        // Both tokens have the same 7-element shape so they can match across
        // epochs: address(lo,hi), value(lo,hi), epoch, timestamp(lo,hi).
        assert_eq!(init.values.len(), 7);
        assert_eq!(fini.values.len(), 7);
    }
}
