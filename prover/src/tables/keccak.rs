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
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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
        INPUT_STATE + (x * 5 + y) * 8 + byte
    }

    /// Index into output_state[x][y][byte]
    #[inline]
    pub const fn output_state(x: usize, y: usize, byte: usize) -> usize {
        OUTPUT_STATE + (x * 5 + y) * 8 + byte
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

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus interactions
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = KECCAK_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_SYSCALL_NUMBER >> 32;
    let mut interactions = Vec::with_capacity(160);

    // 1. EcallKeccak receiver: [ts_lo, ts_hi, syscall_lo32, syscall_hi32, addr_lo32, addr_hi32]
    interactions.push(BusInteraction::receiver(
        BusId::EcallKeccak,
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
            // state_addr as DWordWL from DWordBL bytes
            BusValue::linear(vec![
                LinearTerm::Column { coefficient: 1, column: cols::addr(0) },
                LinearTerm::Column { coefficient: 256, column: cols::addr(1) },
                LinearTerm::Column { coefficient: 65536, column: cols::addr(2) },
                LinearTerm::Column { coefficient: 16777216, column: cols::addr(3) },
            ]),
            BusValue::linear(vec![
                LinearTerm::Column { coefficient: 1, column: cols::addr(4) },
                LinearTerm::Column { coefficient: 256, column: cols::addr(5) },
                LinearTerm::Column { coefficient: 65536, column: cols::addr(6) },
                LinearTerm::Column { coefficient: 16777216, column: cols::addr(7) },
            ]),
        ],
    ));

    // 2. Keccak bus: send (timestamp, 0, input_state[200])
    {
        let mut values = vec![
            BusValue::Packed { start_column: cols::TIMESTAMP_0, packing: Packing::Direct },
            BusValue::Packed { start_column: cols::TIMESTAMP_1, packing: Packing::Direct },
            BusValue::constant(0), // round = 0
        ];
        for x in 0..5 {
            for y in 0..5 {
                values.push(BusValue::Packed {
                    start_column: cols::input_state(x, y, 0),
                    packing: Packing::Word4L,
                });
                values.push(BusValue::Packed {
                    start_column: cols::input_state(x, y, 4),
                    packing: Packing::Word4L,
                });
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
            BusValue::Packed { start_column: cols::TIMESTAMP_0, packing: Packing::Direct },
            BusValue::Packed { start_column: cols::TIMESTAMP_1, packing: Packing::Direct },
            BusValue::constant(24), // round = 24
        ];
        for x in 0..5 {
            for y in 0..5 {
                values.push(BusValue::Packed {
                    start_column: cols::output_state(x, y, 0),
                    packing: Packing::Word4L,
                });
                values.push(BusValue::Packed {
                    start_column: cols::output_state(x, y, 4),
                    packing: Packing::Word4L,
                });
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

    // 5. MEMW interactions for 25 lane reads (on mu) + 25 lane writes (on mu)
    // Format: [old[8], is_register, address[DWordHL=2], value[8], ts[2], w2, w4, w8]
    for lane_idx in 0..25 {
        let x = lane_idx % 5;
        let y = lane_idx / 5;
        let addr_start = cols::state_ptr(lane_idx, 0);

        // Read: old = input, value = input (read doesn't change)
        let mut read_values = Vec::with_capacity(24);
        // old[0..8] = input bytes
        for b in 0..8 {
            read_values.push(BusValue::Packed {
                start_column: cols::input_state(x, y, b),
                packing: Packing::Direct,
            });
        }
        // is_register = 0
        read_values.push(BusValue::constant(0));
        // address as DWordHL (2 bus elements packed from 4 halfword columns)
        read_values.push(BusValue::Packed {
            start_column: addr_start,
            packing: Packing::DWordHL,
        });
        // value[0..8] = same as old (read)
        for b in 0..8 {
            read_values.push(BusValue::Packed {
                start_column: cols::input_state(x, y, b),
                packing: Packing::Direct,
            });
        }
        // timestamp
        read_values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_0,
            packing: Packing::Direct,
        });
        read_values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        // write2=0, write4=0, write8=1
        read_values.push(BusValue::constant(0));
        read_values.push(BusValue::constant(0));
        read_values.push(BusValue::constant(1));

        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            read_values,
        ));

        // Write: new value = output, timestamp = ts + 1
        let mut write_values = Vec::with_capacity(16);
        // is_register = 0
        write_values.push(BusValue::constant(0));
        // address as DWordHL
        write_values.push(BusValue::Packed {
            start_column: addr_start,
            packing: Packing::DWordHL,
        });
        // value[0..8] = output bytes
        for b in 0..8 {
            write_values.push(BusValue::Packed {
                start_column: cols::output_state(x, y, b),
                packing: Packing::Direct,
            });
        }
        // timestamp + 1
        write_values.push(BusValue::linear(vec![
            LinearTerm::Column { coefficient: 1, column: cols::TIMESTAMP_0 },
            LinearTerm::Constant(1),
        ]));
        write_values.push(BusValue::Packed {
            start_column: cols::TIMESTAMP_1,
            packing: Packing::Direct,
        });
        // write2=0, write4=0, write8=1
        write_values.push(BusValue::constant(0));
        write_values.push(BusValue::constant(0));
        write_values.push(BusValue::constant(1));

        interactions.push(BusInteraction::sender(
            BusId::Memw,
            Multiplicity::Column(cols::MU),
            write_values,
        ));
    }

    interactions
}
