//! LOAD (Memory Load with Extension) table.
//!
//! This table handles memory load operations with sign/zero extension for
//! RISC-V load instructions (LB, LH, LW, LD, LBU, LHU, LWU).
//!
//! ## Inputs
//! - `base_address`: DWordWL (64-bit address)
//! - `timestamp`: DWordWL (64-bit timestamp)
//! - `read2/4/8`: Bit (access width flags)
//! - `signed`: Bit (1 = sign-extend, 0 = zero-extend)
//!
//! ## Output
//! - `res[8]`: DWordBL (8 bytes result, properly extended)
//!
//! ## Auxiliary
//! - `sign_bit`: Bit (MSB of the read data, for sign extension)
//!
//! ## Virtual (computed inline)
//! - `read1`: μ - read2 - read4 - read8 (reading exactly 1 byte)
//!
//! ## Bus Interactions
//! - Receiver: LOAD (from CPU for load operations)
//! - Sender: MEMW (to read from memory)
//! - Sender: MSB8 (for sign bit extraction)

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraint_ir::{Capture, IrBuilder};
use stark::constraints::transition::{TransitionConstraint, TransitionConstraintEvaluator};
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable};

// =========================================================================
// Column indices for LOAD table
// =========================================================================

/// Column definitions for the LOAD table.
pub mod cols {
    // Input columns
    /// base_address: DWordWL (2 words = 2 columns)
    pub const BASE_ADDRESS_0: usize = 0;
    pub const BASE_ADDRESS_1: usize = 1;

    /// timestamp: DWordWL (2 words = 2 columns)
    pub const TIMESTAMP_0: usize = 2;
    pub const TIMESTAMP_1: usize = 3;

    /// read2, read4, read8: access width flags
    /// Similar to MEMW write flags - see MEMW for encoding explanation
    pub const READ2: usize = 4;
    pub const READ4: usize = 5;
    pub const READ8: usize = 6;

    /// signed: Bit (1 = sign-extend, 0 = zero-extend)
    pub const SIGNED: usize = 7;

    // Output columns
    /// res[8]: 8 bytes of result (DWordBL = 8 columns)
    pub const RES: [usize; 8] = [8, 9, 10, 11, 12, 13, 14, 15];

    // Auxiliary columns
    /// sign_bit: Bit (MSB of the relevant byte for sign extension)
    pub const SIGN_BIT: usize = 16;

    // Multiplicity column
    /// μ: Whether this row is active
    pub const MU: usize = 17;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 18;
}

// =========================================================================
// Trace generation
// =========================================================================

/// A single LOAD operation to be added to the trace.
#[derive(Debug, Clone)]
pub struct LoadOperation {
    /// Base address (64-bit)
    pub base_address: u64,
    /// Timestamp of this access
    pub timestamp: u64,
    /// Access width: 1, 2, 4, or 8 bytes
    pub width: u8,
    /// Whether to sign-extend (true) or zero-extend (false)
    pub signed: bool,
    /// Result bytes (8 bytes, extended)
    pub res: [u64; 8],
}

impl LoadOperation {
    /// Create a new LOAD operation.
    pub fn new(base_address: u64, timestamp: u64, width: u8, signed: bool, res: [u64; 8]) -> Self {
        Self {
            base_address,
            timestamp,
            width,
            signed,
            res,
        }
    }

    /// Convert access width to the spec's flag representation (read2, read4, read8).
    ///
    /// The spec uses three flags to encode access width:
    /// - `read2`: set if accessing 2+ bytes (width >= 2)
    /// - `read4`: set if accessing 4+ bytes (width >= 4)
    /// - `read8`: set if accessing 8 bytes (width == 8)
    ///
    /// | Width | read2 | read4 | read8 |
    /// |-------|-------|-------|-------|
    /// |   1   |   0   |   0   |   0   |
    /// |   2   |   1   |   0   |   0   |
    /// |   4   |   0   |   1   |   0   |
    /// |   8   |   0   |   0   |   1   |
    ///
    /// Note: These are "exactly N" semantics per spec, not cumulative.
    /// Virtual column read1 = μ - read2 - read4 - read8 computes "exactly 1 byte".
    pub fn read_flags(&self) -> (bool, bool, bool) {
        match self.width {
            1 => (false, false, false),
            2 => (true, false, false),
            4 => (false, true, false),
            8 => (false, false, true),
            _ => (false, false, false),
        }
    }

    /// Get the sign bit from the result based on width.
    ///
    /// The sign bit is the MSB of the highest byte being read:
    /// - width 1: MSB of res[0] (bit 7)
    /// - width 2: MSB of res[1] (bit 7 of byte 1 = bit 15 of halfword)
    /// - width 4: MSB of res[3] (bit 7 of byte 3 = bit 31 of word)
    /// - width 8: MSB of res[7] (bit 7 of byte 7 = bit 63 of dword)
    pub fn compute_sign_bit(&self) -> bool {
        let byte_idx = match self.width {
            1 => 0,
            2 => 1,
            4 => 3,
            8 => 7,
            _ => 0,
        };
        (self.res[byte_idx] >> 7) & 1 == 1
    }

    /// Collect MSB8 bitwise lookups for sign bit extraction.
    ///
    /// Per spec constraints #3-#5:
    /// - read1: MSB8[res[0]] -> sign_bit
    /// - read2: MSB8[res[1]] -> sign_bit
    /// - read4: MSB8[res[3]] -> sign_bit
    /// - read8: no MSB8 lookup needed (all 8 bytes are used)
    pub fn collect_bitwise_ops(&self) -> Vec<super::bitwise::BitwiseOperation> {
        use super::bitwise::{BitwiseOperation, BitwiseOperationType};

        // For width 8, no sign extension is needed
        if self.width == 8 {
            return Vec::new();
        }

        // Get the byte index for the MSB8 lookup based on width
        let byte_idx = match self.width {
            1 => 0, // res[0] for read1
            2 => 1, // res[1] for read2
            4 => 3, // res[3] for read4
            _ => return Vec::new(),
        };

        let input_byte = self.res[byte_idx] as u8;
        vec![BitwiseOperation::single_byte(
            BitwiseOperationType::Msb8,
            input_byte,
        )]
    }
}

/// Generates the LOAD trace table from a list of operations.
pub fn generate_load_trace(
    operations: &[LoadOperation],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let num_rows = operations.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, op) in operations.iter().enumerate() {
        // Input columns
        table.set_dword_wl(row_idx, cols::BASE_ADDRESS_0, op.base_address);
        table.set_dword_wl(row_idx, cols::TIMESTAMP_0, op.timestamp);

        // read flags
        let (r2, r4, r8) = op.read_flags();
        table.set_bool(row_idx, cols::READ2, r2);
        table.set_bool(row_idx, cols::READ4, r4);
        table.set_bool(row_idx, cols::READ8, r8);

        // signed
        table.set_bool(row_idx, cols::SIGNED, op.signed);

        // Output: res[8]
        for i in 0..8 {
            table.set_u64(row_idx, cols::RES[i], op.res[i]);
        }

        // Auxiliary: sign_bit
        table.set_bool(row_idx, cols::SIGN_BIT, op.compute_sign_bit());

        // Multiplicity: active row
        table.set_fe(row_idx, cols::MU, FE::one());
    }

    trace
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the LOAD table.
///
/// The LOAD table:
/// - **Receives** LOAD lookups from CPU
/// - **Sends** MEMW lookups to read memory
/// - **Sends** MSB8 lookups for sign bit extraction
#[allow(clippy::vec_init_then_push)]
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

    // -------------------------------------------------------------------------
    // MEMW sender (to read memory) - ENABLED
    // -------------------------------------------------------------------------
    // LOAD calls MEMW with is_register=0, passing res as both value and old
    // (since we're reading, value=old=the read data)
    // RES columns contain individual bytes, sent as Direct elements
    // to match the unified MEMW Read receiver format.
    interactions.push(BusInteraction::sender(
        BusId::Memw,
        Multiplicity::Column(cols::MU),
        vec![
            // old[0..7] = 8 individual bytes (Direct elements)
            // For reads, old == value (same data read back)
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[7],
                packing: Packing::Direct,
            },
            // is_register = 0 (constant)
            BusValue::constant(0),
            // base_address (DWordWL = 2 words)
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_1,
                packing: Packing::Direct,
            },
            // value[0..7] = 8 individual bytes (Direct elements)
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[2],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[4],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[5],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[6],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RES[7],
                packing: Packing::Direct,
            },
            // timestamp (DWordWL = 2 words)
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::TIMESTAMP_1,
                packing: Packing::Direct,
            },
            // read flags (same as write flags for MEMW)
            BusValue::Packed {
                start_column: cols::READ2,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::READ4,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::READ8,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // MSB8 lookups for sign bit extraction
    // -------------------------------------------------------------------------
    // Need to extract MSB from the relevant byte based on width
    // For read1: MSB8[res[0]] -> sign_bit, multiplicity = read1 = μ - read2 - read4 - read8
    // For read2: MSB8[res[1]] -> sign_bit, multiplicity = read2
    // For read4: MSB8[res[3]] -> sign_bit, multiplicity = read4
    // (For read8, no extension needed - all 8 bytes are used)

    // MSB8[res[0]] -> sign_bit (for read1)
    // read1 = μ - read2 - read4 - read8 (reading exactly 1 byte)
    interactions.push(BusInteraction::sender(
        BusId::Msb8,
        Multiplicity::Linear(vec![
            LinearTerm::Column {
                coefficient: 1,
                column: cols::MU,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::READ2,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::READ4,
            },
            LinearTerm::Column {
                coefficient: -1,
                column: cols::READ8,
            },
        ]),
        vec![
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGN_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // MSB8[res[1]] -> sign_bit (for read2)
    interactions.push(BusInteraction::sender(
        BusId::Msb8,
        Multiplicity::Column(cols::READ2),
        vec![
            BusValue::Packed {
                start_column: cols::RES[1],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGN_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // MSB8[res[3]] -> sign_bit (for read4)
    interactions.push(BusInteraction::sender(
        BusId::Msb8,
        Multiplicity::Column(cols::READ4),
        vec![
            BusValue::Packed {
                start_column: cols::RES[3],
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGN_BIT,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // MEMORY receiver (from CPU) — unified high-level memory op.
    // -------------------------------------------------------------------------
    // MEMORY[out=res::DWordWL; timestamp, address, value, mem_flags] | -μ
    // The CPU dispatches LOAD here (mem_flags bit 0 = memory_op = 0). The `value`
    // field carries the store value and is 0 for loads; `out` is the loaded res.
    // mem_flags = 2*signed + 4*read2 + 8*read4 + 16*read8 (memory_op = 0).
    interactions.push(BusInteraction::receiver(
        BusId::MemoryOp,
        Multiplicity::Column(cols::MU),
        vec![
            // timestamp (DWordWL = 2 words)
            BusValue::Packed {
                start_column: cols::TIMESTAMP_0,
                packing: Packing::DWordWL,
            },
            // address = base_address (DWordWL = 2 words)
            BusValue::Packed {
                start_column: cols::BASE_ADDRESS_0,
                packing: Packing::DWordWL,
            },
            // value (store value) = 0 for loads
            BusValue::constant(0),
            BusValue::constant(0),
            // mem_flags byte
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 2,
                    column: cols::SIGNED,
                },
                LinearTerm::Column {
                    coefficient: 4,
                    column: cols::READ2,
                },
                LinearTerm::Column {
                    coefficient: 8,
                    column: cols::READ4,
                },
                LinearTerm::Column {
                    coefficient: 16,
                    column: cols::READ8,
                },
            ]),
            // out = res::DWordWL (8 bytes packed as 2 words) — the loaded value
            BusValue::Packed {
                start_column: cols::RES[0],
                packing: Packing::DWordBL,
            },
        ],
    ));

    interactions
}
// =========================================================================
// Constraints
// =========================================================================

/// LOAD table constraint kinds.
#[derive(Debug, Clone, Copy)]
pub enum LoadConstraintKind {
    /// (read2 + read4 + read8) => μ: if reading 2+ bytes, row must be active
    ReadImpliesMu,
    /// Extension constraint for res[i] when not reading those bytes
    /// !read8 => res[i] = signed * sign_bit * 255 for i in 4..8
    ExtensionHigh(usize),
    /// !read4 && !read8 => res[i] = signed * sign_bit * 255 for i in 2..4
    ExtensionMid(usize),
    /// !read2 && !read4 && !read8 => res[1] = signed * sign_bit * 255
    ExtensionLow,
    /// `IS_BIT<flag>`: `flag * (1 - flag) = 0` for a boolean flag used as a bus
    /// multiplicity / extension selector (`load.toml` `signed`/`read2`/`read4`/
    /// `read8`). `usize` is the flag column.
    FlagIsBit(usize),
    /// `IS_BIT<read2 + read4 + read8>`: the width selector sum is boolean, so
    /// `read1 = μ − sum` is well-formed (`load.toml:107-109`).
    WidthSumIsBit,
}

/// LOAD table constraint.
pub struct LoadConstraint {
    constraint_idx: usize,
    kind: LoadConstraintKind,
}

impl LoadConstraint {
    pub fn new(kind: LoadConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let one = FieldElement::<F>::one();
        let ff = FieldElement::<F>::from(255u64); // 0xFF for sign extension

        let mu = step.get_main_evaluation_element(0, cols::MU).clone();
        let read2 = step.get_main_evaluation_element(0, cols::READ2).clone();
        let read4 = step.get_main_evaluation_element(0, cols::READ4).clone();
        let read8 = step.get_main_evaluation_element(0, cols::READ8).clone();
        let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
        let sign_bit = step.get_main_evaluation_element(0, cols::SIGN_BIT).clone();

        match self.kind {
            LoadConstraintKind::ReadImpliesMu => {
                // (read2 + read4 + read8) * (1 - μ) = 0
                let read_sum = &read2 + &read4 + &read8;
                &read_sum * (&one - &mu)
            }
            LoadConstraintKind::ExtensionHigh(i) => {
                // (1 - read8) * (res[i] - signed * sign_bit * 255) = 0
                // i should be in 4..8
                let res_i = step.get_main_evaluation_element(0, cols::RES[i]).clone();
                let expected = &signed * &sign_bit * &ff;
                (&one - &read8) * (&res_i - &expected)
            }
            LoadConstraintKind::ExtensionMid(i) => {
                // (1 - read4 - read8) * (res[i] - signed * sign_bit * 255) = 0
                // i should be in 2..4
                let res_i = step.get_main_evaluation_element(0, cols::RES[i]).clone();
                let expected = &signed * &sign_bit * &ff;
                (&one - &read4 - &read8) * (&res_i - &expected)
            }
            LoadConstraintKind::ExtensionLow => {
                // (1 - read2 - read4 - read8) * (res[1] - signed * sign_bit * 255) = 0
                let res_1 = step.get_main_evaluation_element(0, cols::RES[1]).clone();
                let expected = &signed * &sign_bit * &ff;
                (&one - &read2 - &read4 - &read8) * (&res_1 - &expected)
            }
            LoadConstraintKind::FlagIsBit(col) => {
                // flag * (1 - flag) = 0
                let flag = step.get_main_evaluation_element(0, col).clone();
                &flag * (&one - &flag)
            }
            LoadConstraintKind::WidthSumIsBit => {
                // sum * (1 - sum) = 0, sum = read2 + read4 + read8
                let sum = &read2 + &read4 + &read8;
                &sum * (&one - &sum)
            }
        }
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for LoadConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            LoadConstraintKind::ReadImpliesMu => 2,
            // Extension constraints: (1 - readX) * (res[i] - signed * sign_bit * 255)
            // = degree 1 * (degree 1 - degree 2) = degree 3
            LoadConstraintKind::ExtensionHigh(_) => 3,
            LoadConstraintKind::ExtensionMid(_) => 3,
            LoadConstraintKind::ExtensionLow => 3,
            // flag * (1 - flag) and sum * (1 - sum)
            LoadConstraintKind::FlagIsBit(_) => 2,
            LoadConstraintKind::WidthSumIsBit => 2,
        }
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

impl Capture for LoadConstraint {
    fn capture(&self, b: &mut IrBuilder) {
        let one = b.one();
        let ff = b.const_base(255);

        let mu = b.main(0, cols::MU);
        let read2 = b.main(0, cols::READ2);
        let read4 = b.main(0, cols::READ4);
        let read8 = b.main(0, cols::READ8);
        let signed = b.main(0, cols::SIGNED);
        let sign_bit = b.main(0, cols::SIGN_BIT);

        let root = match self.kind {
            LoadConstraintKind::ReadImpliesMu => {
                // (read2 + read4 + read8) * (1 - mu)
                let read_sum = b.add(read2, read4);
                let read_sum = b.add(read_sum, read8);
                let one_minus_mu = b.sub(one, mu);
                b.mul(read_sum, one_minus_mu)
            }
            LoadConstraintKind::ExtensionHigh(i) => {
                // (1 - read8) * (res[i] - signed * sign_bit * 255)
                let res_i = b.main(0, cols::RES[i]);
                let signed_sign_bit = b.mul(signed, sign_bit);
                let expected = b.mul(signed_sign_bit, ff);
                let diff = b.sub(res_i, expected);
                let one_minus_read8 = b.sub(one, read8);
                b.mul(one_minus_read8, diff)
            }
            LoadConstraintKind::ExtensionMid(i) => {
                // (1 - read4 - read8) * (res[i] - signed * sign_bit * 255)
                let res_i = b.main(0, cols::RES[i]);
                let signed_sign_bit = b.mul(signed, sign_bit);
                let expected = b.mul(signed_sign_bit, ff);
                let diff = b.sub(res_i, expected);
                let coeff = b.sub(one, read4);
                let coeff = b.sub(coeff, read8);
                b.mul(coeff, diff)
            }
            LoadConstraintKind::ExtensionLow => {
                // (1 - read2 - read4 - read8) * (res[1] - signed * sign_bit * 255)
                let res_1 = b.main(0, cols::RES[1]);
                let signed_sign_bit = b.mul(signed, sign_bit);
                let expected = b.mul(signed_sign_bit, ff);
                let diff = b.sub(res_1, expected);
                let coeff = b.sub(one, read2);
                let coeff = b.sub(coeff, read4);
                let coeff = b.sub(coeff, read8);
                b.mul(coeff, diff)
            }
            LoadConstraintKind::FlagIsBit(col) => {
                // flag * (1 - flag)
                let flag = b.main(0, col);
                let one_minus_flag = b.sub(one, flag);
                b.mul(flag, one_minus_flag)
            }
            LoadConstraintKind::WidthSumIsBit => {
                // sum * (1 - sum), sum = read2 + read4 + read8
                let sum = b.add(read2, read4);
                let sum = b.add(sum, read8);
                let one_minus_sum = b.sub(one, sum);
                b.mul(sum, one_minus_sum)
            }
        };

        b.emit(self.constraint_idx, root);
    }
}

/// Creates all constraints for the LOAD table.
pub fn constraints()
-> Vec<Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>> {
    let mut constraints: Vec<
        Box<dyn TransitionConstraintEvaluator<GoldilocksField, GoldilocksExtension>>,
    > = Vec::new();

    let mut idx = 0;

    // IS_BIT on the width/sign flags (used as bus multiplicities + extension
    // selectors): signed, read2, read4, read8 (`load.toml` `all` group).
    for flag_col in [cols::SIGNED, cols::READ2, cols::READ4, cols::READ8] {
        constraints.push(LoadConstraint::new(LoadConstraintKind::FlagIsBit(flag_col), idx).boxed());
        idx += 1;
    }
    // IS_BIT on the width-selector sum (so read1 = μ − sum is well-formed).
    constraints.push(LoadConstraint::new(LoadConstraintKind::WidthSumIsBit, idx).boxed());
    idx += 1;

    // (read2 + read4 + read8) => μ
    constraints.push(LoadConstraint::new(LoadConstraintKind::ReadImpliesMu, idx).boxed());
    idx += 1;

    // Extension constraints for high bytes (4..8): !read8 => res[i] = extended
    for i in 4..8 {
        constraints.push(LoadConstraint::new(LoadConstraintKind::ExtensionHigh(i), idx).boxed());
        idx += 1;
    }

    // Extension constraints for mid bytes (2..4): !(read4 + read8) => res[i] = extended
    for i in 2..4 {
        constraints.push(LoadConstraint::new(LoadConstraintKind::ExtensionMid(i), idx).boxed());
        idx += 1;
    }

    // Extension constraint for low byte (1): !(read2 + read4 + read8) => res[1] = extended
    constraints.push(LoadConstraint::new(LoadConstraintKind::ExtensionLow, idx).boxed());

    constraints
}
