//! BYTEWISE ALU table.
//!
//! Computes a full-word bitwise `AND`/`OR`/`XOR` of two 64-bit values by
//! decomposing them into bytes and delegating each byte to the `BYTE_ALU`
//! lookup. The CPU dispatches here on the unified `ALU` bus for `alu_op`
//! `AND`(0)/`OR`(1)/`XOR`(2); `alu_flags` for these ops equals just the opcode.
//!
//! Spec: `spec/src/bytewise.toml`. The chip has no polynomial constraints —
//! correctness is entirely enforced by the lookups (the `BYTE_ALU` lookup also
//! range-checks each input byte).
//!
//! ## Columns
//! - `a`: DWordBL (8 bytes)   — first input
//! - `b`: DWordBL (8 bytes)   — second input
//! - `op`: Byte               — the `alu_op` opcode (AND/OR/XOR)
//! - `res`: DWordBL (8 bytes) — output
//! - `μ`: multiplicity

use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::trace::TraceTable;

use super::types::{BusId, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

// =========================================================================
// Column indices for BYTEWISE table
// =========================================================================

/// Column definitions for the BYTEWISE table.
pub mod cols {
    /// a as 8 bytes (DWordBL), little-endian.
    pub const A: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
    /// b as 8 bytes (DWordBL), little-endian.
    pub const B: [usize; 8] = [8, 9, 10, 11, 12, 13, 14, 15];
    /// op: Byte (alu_op opcode: AND/OR/XOR)
    pub const OP: usize = 16;
    /// res as 8 bytes (DWordBL), little-endian.
    pub const RES: [usize; 8] = [17, 18, 19, 20, 21, 22, 23, 24];
    /// μ: multiplicity
    pub const MU: usize = 25;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 26;
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single BYTEWISE operation. `op` is an [`alu_op`] opcode in {AND, OR, XOR}.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct BytewiseOperation {
    pub a: u64,
    pub b: u64,
    pub op: u8,
}

impl BytewiseOperation {
    /// Create a new BYTEWISE operation.
    pub fn new(a: u64, b: u64, op: u8) -> Self {
        Self { a, b, op }
    }

    /// The result of applying `op` to `a` and `b` (byte-wise == full-word).
    pub fn compute_res(&self) -> u64 {
        match self.op {
            alu_op::AND => self.a & self.b,
            alu_op::OR => self.a | self.b,
            alu_op::XOR => self.a ^ self.b,
            other => panic!("BYTEWISE only handles AND/OR/XOR, got opcode {other}"),
        }
    }

    /// The 8 `BYTE_ALU` lookups this op sends, for the BITWISE table's
    /// multiplicity bookkeeping (one per byte, keyed by opsel).
    pub fn collect_bitwise_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};
        let kind = match self.op {
            alu_op::AND => BitwiseOperationType::ByteAluAnd,
            alu_op::OR => BitwiseOperationType::ByteAluOr,
            alu_op::XOR => BitwiseOperationType::ByteAluXor,
            other => panic!("BYTEWISE only handles AND/OR/XOR, got opcode {other}"),
        };
        (0..8)
            .map(|i| {
                let a = ((self.a >> (i * 8)) & 0xFF) as u8;
                let b = ((self.b >> (i * 8)) & 0xFF) as u8;
                BitwiseOperation::byte_op(kind, a, b)
            })
            .collect()
    }
}

/// Generates the BYTEWISE trace from a list of operations.
///
/// Duplicate operations are merged with summed multiplicities, then padded to
/// the next power of two (minimum 4).
pub fn generate_bytewise_trace(
    operations: &[BytewiseOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use std::collections::HashMap;

    let mut op_map: HashMap<BytewiseOperation, u64> = HashMap::new();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    // Canonical row order: HashMap iteration order is per-process random, so
    // sort to keep the committed trace deterministic across runs.
    let mut unique_ops: Vec<_> = op_map.into_iter().collect();
    unique_ops.sort_unstable_by_key(|(op, _)| (op.a, op.b, op.op));
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        let res = op.compute_res();

        table.set_dword_bl(row_idx, cols::A[0], op.a);
        table.set_dword_bl(row_idx, cols::B[0], op.b);
        table.set_dword_bl(row_idx, cols::RES[0], res);
        table.set_byte(row_idx, cols::OP, op.op);
        table.set_u64(row_idx, cols::MU, *multiplicity);
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// All bus interactions for the BYTEWISE table:
/// - **Sends** `BYTE_ALU[op, a[i], b[i]] -> res[i]` for each of the 8 bytes.
/// - **Receives** `ALU[a, b, op] -> res` (operands packed DWordBL -> 2 words).
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::with_capacity(9);

    for i in 0..8 {
        interactions.push(BusInteraction::sender(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU),
            vec![
                BusValue::Packed {
                    start_column: cols::OP,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::A[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::B[i],
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RES[i],
                    packing: Packing::Direct,
                },
            ],
        ));
    }

    // ALU[a, b, op] -> res (receiver). a/b/res are DWordBL (8 bytes) packed
    // into 2 words each, matching the CPU's DWordWL operands.
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU),
        vec![
            BusValue::Packed {
                start_column: cols::A[0],
                packing: Packing::DWordBL,
            },
            BusValue::Packed {
                start_column: cols::B[0],
                packing: Packing::DWordBL,
            },
            BusValue::Packed {
                start_column: cols::OP,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
        ],
    ));

    interactions
}
