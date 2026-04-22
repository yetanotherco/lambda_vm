//! KECCAK core chip — handles ECALL, memory I/O, and delegation to the round chip.
//!
//! One row per keccak permutation call. Reads/writes 25 u64 lanes from/to memory,
//! sends input state to the round chip via the Keccak bus, and receives the output
//! state after 24 rounds.
//!
//! ## Column layout (~511 columns)
//!
//! | Group          | Size | Description                                    |
//! |----------------|------|------------------------------------------------|
//! | timestamp      |    2 | DWordWL                                        |
//! | addr           |    8 | State address as DWordBL (8 bytes)             |
//! | input_state    |  200 | Input state bytes [5][5][8]                    |
//! | output_state   |  200 | Output state bytes [5][5][8]                   |
//! | state_ptr      |  100 | Per-lane DWordHL addresses [25][4]             |
//! | mu             |    1 | Multiplicity flag                              |

use executor::vm::instruction::execution::KECCAK_SYSCALL_NUMBER;
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};
use crate::constraints::templates::{AddConstraint, AddOperand};

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

    // state_ptr[25][4] = 100 halfwords (DWordHL per lane)
    pub const STATE_PTR: usize = OUTPUT_STATE + 200; // 410

    pub const MU: usize = STATE_PTR + 100; // 510

    pub const NUM_COLUMNS: usize = MU + 1; // 511

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

    /// Index into state_ptr[lane_idx][halfword] (DWordHL = 4 halfwords)
    #[inline]
    pub const fn state_ptr(lane_idx: usize, hw: usize) -> usize {
        STATE_PTR + lane_idx * 4 + hw
    }
}

// =========================================================================
// Operation struct
// =========================================================================

#[derive(Debug, Clone)]
pub struct KeccakOperation {
    pub timestamp: u64,
    pub state_addr: u64,
    pub input: [u64; 25],
    pub output: [u64; 25],
}

// =========================================================================
// Trace generation
// =========================================================================

fn byte_of(val: u64, b: usize) -> u8 {
    ((val >> (b * 8)) & 0xFF) as u8
}

pub fn generate_keccak_trace(
    ops: &[KeccakOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, op) in ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        // Timestamp
        data[base + cols::TIMESTAMP_0] = FE::from(op.timestamp & 0xFFFF_FFFF);
        data[base + cols::TIMESTAMP_1] = FE::from(op.timestamp >> 32);

        // Address as 8 bytes
        for b in 0..8 {
            data[base + cols::addr(b)] = FE::from(byte_of(op.state_addr, b) as u64);
        }

        // Input state as bytes
        for x in 0..5 {
            for y in 0..5 {
                let lane = op.input[x + 5 * y];
                for b in 0..8 {
                    data[base + cols::input_state(x, y, b)] = FE::from(byte_of(lane, b) as u64);
                }
            }
        }

        // Output state as bytes
        for x in 0..5 {
            for y in 0..5 {
                let lane = op.output[x + 5 * y];
                for b in 0..8 {
                    data[base + cols::output_state(x, y, b)] = FE::from(byte_of(lane, b) as u64);
                }
            }
        }

        // State pointers: state_ptr[lane] = addr + 8 * lane_idx
        for lane_idx in 0..25 {
            let ptr = op.state_addr.wrapping_add(lane_idx as u64 * 8);
            data[base + cols::state_ptr(lane_idx, 0)] = FE::from(ptr & 0xFFFF);
            data[base + cols::state_ptr(lane_idx, 1)] = FE::from((ptr >> 16) & 0xFFFF);
            data[base + cols::state_ptr(lane_idx, 2)] = FE::from((ptr >> 32) & 0xFFFF);
            data[base + cols::state_ptr(lane_idx, 3)] = FE::from((ptr >> 48) & 0xFFFF);
        }

        // mu = 1 (real row)
        data[base + cols::MU] = FE::one();
    }

    // Padding rows: state_ptr[lane][0] = 8 * lane_idx (per spec keccak.toml pad).
    // Halfwords 1..3 stay zero since 8*24 = 192 fits in the low halfword.
    // mu = 0 gates all bus interactions and the ADD constraint, so these values
    // only need to satisfy the pad requirement, not reconstruct a real address.
    for row_idx in n..num_rows {
        let base = row_idx * cols::NUM_COLUMNS;
        for lane_idx in 0..25 {
            data[base + cols::state_ptr(lane_idx, 0)] = FE::from((lane_idx as u64) * 8);
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = KECCAK_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_SYSCALL_NUMBER >> 32;
    let mut interactions = Vec::with_capacity(160);

    // 1. ECALL receiver (shared bus, per spec keccak:c:output)
    // Format: [ts_lo, ts_hi, syscall_lo32, syscall_hi32] (DWordWL convention).
    //
    // Spec keccak.toml:51 has `["arr", 2^32-1, 2^32-2]` which flattens to
    // [hi, lo] — inconsistent with HALT/COMMIT which use `["cast", N, "DWordWL"]`
    // → [lo, hi]. The CPU ECALL sender (cpu.rs) is shared across all three
    // receivers and uses [lo, hi], so applying the spec's keccak ordering
    // literally desbalances the LogUp bus.
    //
    // Upstream spec needs to change keccak.toml:51 to `["cast", -2, "DWordWL"]`.
    // See docs/keccak-spec-deviations.md #7.
    interactions.push(BusInteraction::receiver(
        BusId::Ecall,
        Multiplicity::Column(cols::MU),
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
            Multiplicity::Column(cols::MU),
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

    // 4. IS_HALF range checks on state_ptr (100 interactions)
    for lane_idx in 0..25 {
        for hw in 0..4 {
            interactions.push(BusInteraction::sender(
                BusId::IsHalfword,
                Multiplicity::Column(cols::MU),
                vec![BusValue::Packed {
                    start_column: cols::state_ptr(lane_idx, hw),
                    packing: Packing::Direct,
                }],
            ));
        }
    }

    // 5. MEMW interactions: 25 combined read+write per lane (per spec)
    // Format: [old[8], is_register, addr_lo32, addr_hi32, value[8], ts[2], w2, w4, w8] = 24
    // old = input_state (read), value = output_state (write)
    for lane_idx in 0..25 {
        let x = lane_idx % 5;
        let y = lane_idx / 5;

        // Address as DWordWL: lo32 = h0 + 2^16*h1, hi32 = h2 + 2^16*h3
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
        let addr_hi = BusValue::linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::state_ptr(lane_idx, 2),
            },
            LinearTerm::Column {
                coefficient: 65536,
                column: cols::state_ptr(lane_idx, 3),
            },
        ]);

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
            Multiplicity::Column(cols::MU),
            values,
        ));
    }

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// Create constraints for the KECCAK core chip.
///
/// Per spec (keccak:c:state_ptr): ADD template for each lane:
///   state_ptr[lane] = addr + 8 * lane_idx
///
/// 25 lane pointers × 2 constraints per ADD = 50 constraints total.
/// Conditional on mu (only real rows).
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::with_capacity(50);
    let mut idx = constraint_idx_start;

    // state_ptr[lane] = addr + 8*lane_idx
    // addr is DWordBL (8 bytes), state_ptr is DWordHL (4 halfwords)
    // ADD: lhs = addr (DWordBL→DWordWL), rhs = 8*lane_idx (constant), sum = state_ptr (DWordHL→DWordWL)
    for lane_idx in 0..25 {
        let offset = (lane_idx * 8) as i64;
        let (c0, c1) = AddConstraint::new_pair(
            vec![cols::MU], // conditional on mu
            AddOperand::from_dword_bl(cols::ADDR),
            AddOperand::constant(offset),
            AddOperand::from_dword_hl(cols::state_ptr(lane_idx, 0)),
            idx,
        );
        constraints.push(c0.boxed());
        constraints.push(c1.boxed());
        idx += 2;
    }

    (constraints, idx)
}
