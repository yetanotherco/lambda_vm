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

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, alu_op};

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

    // CPU-side dedup: HashMap merge with summed multiplicities. The
    // BYTEWISE key is (a:u64, b:u64, op:u8) = 136 bits → doesn't fit in
    // the u128-keyed `multiplicity_count_multifield` primitive. We could
    // hash + tie-break but for the small dedup sizes here the HashMap
    // pass is already cheap; keep it on CPU and only run the row layout
    // on device.
    let mut op_map: HashMap<BytewiseOperation, u64> = HashMap::new();
    for op in operations {
        *op_map.entry(op.clone()).or_insert(0) += 1;
    }

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);

    // GPU fast path: pack the deduped (a, b, res, op, mu) into five
    // parallel u64 arrays and kernel-byte-decompose into 26 columns.
    #[cfg(feature = "cuda")]
    if let Some(table) = try_generate_bytewise_trace_gpu(&unique_ops, num_rows) {
        return table;
    }

    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, (op, multiplicity)) in unique_ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;
        let res = op.compute_res();

        for i in 0..8 {
            data[base + cols::A[i]] = FE::from((op.a >> (8 * i)) & 0xFF);
            data[base + cols::B[i]] = FE::from((op.b >> (8 * i)) & 0xFF);
            data[base + cols::RES[i]] = FE::from((res >> (8 * i)) & 0xFF);
        }
        data[base + cols::OP] = FE::from(op.op as u64);
        data[base + cols::MU] = FE::from(*multiplicity);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// CUDA fast path for `generate_bytewise_trace`. Flattens the already-deduped
/// unique ops into five parallel u64 arrays (padded to num_rows) and runs
/// the row-layout kernel.
#[cfg(feature = "cuda")]
fn try_generate_bytewise_trace_gpu(
    unique_ops: &[(BytewiseOperation, u64)],
    num_rows: usize,
) -> Option<TraceTable<GoldilocksField, GoldilocksExtension>> {
    let mut a_values = vec![0u64; num_rows];
    let mut b_values = vec![0u64; num_rows];
    let mut res_values = vec![0u64; num_rows];
    let mut ops = vec![0u64; num_rows];
    let mut multiplicities = vec![0u64; num_rows];

    for (i, (op, mu)) in unique_ops.iter().enumerate() {
        a_values[i] = op.a;
        b_values[i] = op.b;
        res_values[i] = op.compute_res();
        ops[i] = op.op as u64;
        multiplicities[i] = *mu;
    }

    let raw = stark::gpu_lde::try_generate_bytewise_trace_gpu_raw(
        num_rows,
        &a_values,
        &b_values,
        &res_values,
        &ops,
        &multiplicities,
        cols::NUM_COLUMNS,
    )?;
    let data: Vec<FE> = raw.into_iter().map(FE::from).collect();
    Some(TraceTable::new_main(data, cols::NUM_COLUMNS, 1))
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
