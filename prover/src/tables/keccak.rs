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
use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use stark::constraints::builder::{ConstraintBuilder, ConstraintMeta, ConstraintSet};

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};
use crate::constraints::templates::{AddConstraint, AddOperand, INV_SHIFT_32};

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

pub fn generate_keccak_trace(
    ops: &[KeccakOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let n = ops.len();
    let num_rows = n.next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in ops.iter().enumerate() {
        // Timestamp
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

        // Address as 8 bytes
        table.set_dword_bl(row_idx, cols::addr(0), op.state_addr);

        // Input state as bytes
        for x in 0..5 {
            for y in 0..5 {
                let lane = op.input[x + 5 * y];
                table.set_dword_bl(row_idx, cols::input_state(x, y, 0), lane);
            }
        }

        // Output state as bytes
        for x in 0..5 {
            for y in 0..5 {
                let lane = op.output[x + 5 * y];
                table.set_dword_bl(row_idx, cols::output_state(x, y, 0), lane);
            }
        }

        // State pointers: state_ptr[lane] = addr + 8 * lane_idx
        for lane_idx in 0..25 {
            let ptr = op
                .state_addr
                .checked_add(lane_idx as u64 * 8)
                .expect("keccak state address range must be validated by the executor");
            table.set_dword_hl(row_idx, cols::state_ptr(lane_idx, 0), ptr);
        }

        // mu = 1 (real row)
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    // Padding rows: state_ptr[lane][0] = 8 * lane_idx (per spec keccak.toml pad).
    // Halfwords 1..3 stay zero since 8*24 = 192 fits in the low halfword.
    // mu = 0 gates all bus interactions and the ADD constraint, so these values
    // only need to satisfy the pad requirement, not reconstruct a real address.
    for row_idx in n..num_rows {
        for lane_idx in 0..25 {
            table.set_u64(row_idx, cols::state_ptr(lane_idx, 0), (lane_idx as u64) * 8);
        }
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

pub fn bus_interactions() -> Vec<BusInteraction> {
    let syscall_lo = KECCAK_SYSCALL_NUMBER & 0xFFFF_FFFF;
    let syscall_hi = KECCAK_SYSCALL_NUMBER >> 32;
    let mut interactions = Vec::with_capacity(160);

    // 1. ECALL receiver (shared bus, per spec keccak:c:output)
    // Payload: [ts_lo, ts_hi, syscall_lo32, syscall_hi32] in DWordWL [lo, hi]
    // ordering, matching the CPU ECALL sender shared with HALT/COMMIT.
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

    // 5. Alignment: addr[0] & 7 = 0, which enforces addr % 8 == 0.
    interactions.push(BusInteraction::sender(
        BusId::ByteAlu,
        Multiplicity::Column(cols::MU),
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
            Multiplicity::Column(cols::MU),
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

struct KeccakAddressNoOverflowConstraint {
    constraint_idx: usize,
}

impl KeccakAddressNoOverflowConstraint {
    fn new(constraint_idx: usize) -> Self {
        Self { constraint_idx }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let addr_lo = step.get_main_evaluation_element(0, cols::addr(0)).clone()
            + step.get_main_evaluation_element(0, cols::addr(1)) * FieldElement::<F>::from(256)
            + step.get_main_evaluation_element(0, cols::addr(2)) * FieldElement::<F>::from(65536)
            + step.get_main_evaluation_element(0, cols::addr(3))
                * FieldElement::<F>::from(16777216);
        let addr_hi = step.get_main_evaluation_element(0, cols::addr(4)).clone()
            + step.get_main_evaluation_element(0, cols::addr(5)) * FieldElement::<F>::from(256)
            + step.get_main_evaluation_element(0, cols::addr(6)) * FieldElement::<F>::from(65536)
            + step.get_main_evaluation_element(0, cols::addr(7))
                * FieldElement::<F>::from(16777216);

        let ptr_lo = step
            .get_main_evaluation_element(0, cols::state_ptr(24, 0))
            .clone()
            + step.get_main_evaluation_element(0, cols::state_ptr(24, 1))
                * FieldElement::<F>::from(65536);
        let ptr_hi = step
            .get_main_evaluation_element(0, cols::state_ptr(24, 2))
            .clone()
            + step.get_main_evaluation_element(0, cols::state_ptr(24, 3))
                * FieldElement::<F>::from(65536);

        let inv_2_32 = FieldElement::<F>::from(INV_SHIFT_32);
        let carry_0 = (addr_lo + FieldElement::<F>::from(192) - ptr_lo) * inv_2_32.clone();
        let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
        step.get_main_evaluation_element(0, cols::MU).clone() * carry_1
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension>
    for KeccakAddressNoOverflowConstraint
{
    fn degree(&self) -> usize {
        2
    }

    fn constraint_idx(&self) -> usize {
        self.constraint_idx
    }

    fn evaluate<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        self.compute(step)
    }
}

/// Create constraints for the KECCAK core chip.
///
/// Per spec (keccak:c:state_ptr): ADD template for each lane:
///   state_ptr[lane] = addr + 8 * lane_idx
///
/// 25 lane pointers × 2 constraints per ADD + 1 top-lane no-overflow
/// constraint = 51 constraints total.
/// Conditional on mu (only real rows).
pub fn create_constraints(
    constraint_idx_start: usize,
) -> (
    Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>>,
    usize,
) {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::with_capacity(51);
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

    constraints.push(KeccakAddressNoOverflowConstraint::new(idx).boxed());
    idx += 1;

    (constraints, idx)
}

// =========================================================================
// Single-source constraint set (ConstraintBuilder front-end)
// =========================================================================

/// The KECCAK core table's transition constraints as a single [`ConstraintSet`],
/// mirroring [`create_constraints`] index-for-index (51 constraints):
/// - idx 0-49: for `lane_idx ∈ 0..25`, the `ADD` carry pair (gated on `μ`)
///   enforcing `state_ptr[lane] = addr + 8·lane_idx` (`addr` DWordBL,
///   `state_ptr` DWordHL);
/// - idx 50:   `μ · carry_1 = 0` (top-lane no-overflow), where `carry_1` is the
///   high carry of `addr + 192 = state_ptr[24]`.
pub struct KeccakConstraints;

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for KeccakConstraints {
    fn meta(&self) -> Vec<ConstraintMeta> {
        let mut m = Vec::with_capacity(51);
        for lane_idx in 0..25 {
            // ADD pair is conditional on μ → degree 3.
            m.extend(crate::constraints::templates::add_pair_meta(
                lane_idx * 2,
                true,
            ));
        }
        m.push(ConstraintMeta::base(50, 2)); // μ · carry_1
        m
    }

    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        use crate::constraints::templates::emit_add_pair;

        // idx 0-49: state_ptr[lane] = addr + 8*lane_idx (gated on μ).
        for lane_idx in 0..25 {
            let offset = (lane_idx * 8) as i64;
            emit_add_pair(
                b,
                lane_idx * 2,
                &[cols::MU],
                &AddOperand::from_dword_bl(cols::ADDR),
                &AddOperand::constant(offset),
                &AddOperand::from_dword_hl(cols::state_ptr(lane_idx, 0)),
            );
        }

        // idx 50: μ · carry_1 (top-lane no-overflow).
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
        let ptr_lo =
            b.main(0, cols::state_ptr(24, 0)) + b.main(0, cols::state_ptr(24, 1)) * c65536.clone();
        let ptr_hi = b.main(0, cols::state_ptr(24, 2)) + b.main(0, cols::state_ptr(24, 3)) * c65536;

        let inv_2_32 = b.const_base(INV_SHIFT_32);
        let c192 = b.const_base(192);
        let carry_0 = (addr_lo + c192 - ptr_lo) * inv_2_32.clone();
        let carry_1 = (addr_hi + carry_0 - ptr_hi) * inv_2_32;
        let mu = b.main(0, cols::MU);
        b.emit_base(50, mu * carry_1);
    }
}
