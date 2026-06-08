//! Local-to-global memory boundary claims for cross-epoch continuations.
//!
//! Per `spec/streaming` Approach 2: each epoch, for every memory cell it touches,
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
}
