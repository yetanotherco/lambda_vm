//! MUL table for 64x64 -> 128-bit multiplication.
//!
//! This table computes the full 128-bit product of two 64-bit operands,
//! supporting both signed and unsigned multiplication.
//!
//! ## Inputs
//! - `lhs`: DWordHL (64-bit as 4 halfwords) - left operand
//! - `lhs_signed`: Bit - whether to treat lhs as signed
//! - `rhs`: DWordHL (64-bit as 4 halfwords) - right operand
//! - `rhs_signed`: Bit - whether to treat rhs as signed
//!
//! ## Outputs
//! - `lo`: DWordHL (64-bit) - lower 64 bits of 128-bit product
//! - `hi`: DWordHL (64-bit) - upper 64 bits of 128-bit product
//!
//! ## Auxiliary
//! - `lhs_is_negative`: Bit - sign of lhs (when signed)
//! - `rhs_is_negative`: Bit - sign of rhs (when signed)
//! - `raw_product[0..4]`: B51 - intermediate convolution values
//!
//! ## Virtual (embedded in constraints)
//! - `lhs_ext[0..8]`: Sign-extended lhs (8 halfwords)
//! - `rhs_ext[0..8]`: Sign-extended rhs (8 halfwords)
//! - `carry[0..4]`: Carries from reduce-and-carry operation
//!
//! ## Bus Interactions
//! - Sender: MSB16 (×2 for sign extraction)
//! - Sender: IS_HALF (×16 for lhs/rhs input and lo/hi output range checks)
//! - Sender: IS_B20 (×4 for carry range checks)
//! - Receiver: ALU (×2 for lo and hi results — every MUL lookup, CPU
//!   MUL/MULH dispatch and dvrm's internal `d*q` consistency)

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use std::collections::HashMap;

use super::types::{
    BusId, GoldilocksExtension, GoldilocksField, INV_2_32, INV_2_64, INV_2_96, INV_2_128,
    NEG_INV_2_16, NEG_INV_2_32, NEG_INV_2_48, NEG_INV_2_64, NEG_INV_2_80, NEG_INV_2_96,
    NEG_INV_2_112, NEG_INV_2_128, SHIFT_16, VmTable, alu_op,
};

/// Total row multiplicity (`ALU` bus, lo + hi), used by the internal
/// range-check sends so they fire once per row-instance.
fn row_mult() -> Multiplicity {
    Multiplicity::Sum(cols::MU_LO, cols::MU_HI)
}

// =========================================================================
// Column indices for MUL table
// =========================================================================

/// Column definitions for the MUL table.
pub mod cols {
    // Input columns: lhs (DWordHL = 4 halfwords)
    /// lhs[0]: Half (bits 0-15)
    pub const LHS_0: usize = 0;
    /// lhs[1]: Half (bits 16-31)
    pub const LHS_1: usize = 1;
    /// lhs[2]: Half (bits 32-47)
    pub const LHS_2: usize = 2;
    /// lhs[3]: Half (bits 48-63)
    pub const LHS_3: usize = 3;

    /// lhs_signed: Bit (1 = signed, 0 = unsigned)
    pub const LHS_SIGNED: usize = 4;

    // Input columns: rhs (DWordHL = 4 halfwords)
    /// rhs[0]: Half (bits 0-15)
    pub const RHS_0: usize = 5;
    /// rhs[1]: Half (bits 16-31)
    pub const RHS_1: usize = 6;
    /// rhs[2]: Half (bits 32-47)
    pub const RHS_2: usize = 7;
    /// rhs[3]: Half (bits 48-63)
    pub const RHS_3: usize = 8;

    /// rhs_signed: Bit (1 = signed, 0 = unsigned)
    pub const RHS_SIGNED: usize = 9;

    // Output columns: lo (DWordHL = 4 halfwords)
    /// lo[0]: Half (bits 0-15 of lower 64-bit product)
    pub const LO_0: usize = 10;
    /// lo[1]: Half (bits 16-31)
    pub const LO_1: usize = 11;
    /// lo[2]: Half (bits 32-47)
    pub const LO_2: usize = 12;
    /// lo[3]: Half (bits 48-63)
    pub const LO_3: usize = 13;

    // Output columns: hi (DWordHL = 4 halfwords)
    /// hi[0]: Half (bits 0-15 of upper 64-bit product)
    pub const HI_0: usize = 14;
    /// hi[1]: Half (bits 16-31)
    pub const HI_1: usize = 15;
    /// hi[2]: Half (bits 32-47)
    pub const HI_2: usize = 16;
    /// hi[3]: Half (bits 48-63)
    pub const HI_3: usize = 17;

    // Auxiliary columns
    /// lhs_is_negative: Bit (1 if lhs is negative when signed)
    pub const LHS_IS_NEGATIVE: usize = 18;
    /// rhs_is_negative: Bit (1 if rhs is negative when signed)
    pub const RHS_IS_NEGATIVE: usize = 19;

    // Raw product columns (B51 = fits in 51 bits)
    /// raw_product[0]: Intermediate convolution value
    pub const RAW_PRODUCT_0: usize = 20;
    /// raw_product[1]: Intermediate convolution value
    pub const RAW_PRODUCT_1: usize = 21;
    /// raw_product[2]: Intermediate convolution value
    pub const RAW_PRODUCT_2: usize = 22;
    /// raw_product[3]: Intermediate convolution value
    pub const RAW_PRODUCT_3: usize = 23;

    // Multiplicity columns. All MUL lookups (CPU MUL/MULH dispatch and dvrm's
    // internal `d*q` consistency checks) go through the unified `ALU` bus.
    /// μ_lo: `ALU` bus multiplicity for lo result lookups
    pub const MU_LO: usize = 24;
    /// μ_hi: `ALU` bus multiplicity for hi result lookups
    pub const MU_HI: usize = 25;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 26;
}

// =========================================================================
// Constants
// =========================================================================

/// Sign extension fill value: 0xFFFF (all 1s for 16 bits)
const SIGN_FILL: u64 = 0xFFFF;

// =========================================================================
// MulOperation struct
// =========================================================================

/// A single MUL operation to be added to the trace.
///
/// Every operation is dispatched on the unified `ALU` bus (CPU MUL/MULH and
/// dvrm's internal `d*q` consistency checks); the lo/hi half is selected by
/// the sender's `flags` byte at lookup time.
///
/// Derives Hash and Eq for HashMap-based deduplication.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct MulOperation {
    /// Left operand (64-bit)
    pub lhs: u64,
    /// Whether lhs is treated as signed
    pub lhs_signed: bool,
    /// Right operand (64-bit)
    pub rhs: u64,
    /// Whether rhs is treated as signed
    pub rhs_signed: bool,
}

/// Multiplicities for a MUL operation, split by lo/hi result lookup.
#[derive(Debug, Clone, Default)]
pub struct MulMultiplicities {
    /// `ALU` bus count requesting lo result
    pub mu_lo: u64,
    /// `ALU` bus count requesting hi result
    pub mu_hi: u64,
}

impl MulOperation {
    /// Create a new MUL operation.
    pub fn new(lhs: u64, lhs_signed: bool, rhs: u64, rhs_signed: bool) -> Self {
        Self {
            lhs,
            lhs_signed,
            rhs,
            rhs_signed,
        }
    }

    /// Compute the full 128-bit product.
    ///
    /// Returns (lo, hi) where:
    /// - lo = lower 64 bits of product
    /// - hi = upper 64 bits of product
    pub fn compute_product(&self) -> (u64, u64) {
        // Convert to 128-bit based on signedness
        let a: i128 = if self.lhs_signed {
            self.lhs as i64 as i128
        } else {
            self.lhs as u128 as i128
        };

        let b: i128 = if self.rhs_signed {
            self.rhs as i64 as i128
        } else {
            self.rhs as u128 as i128
        };

        let product = a.wrapping_mul(b);
        let lo = product as u64;
        let hi = (product >> 64) as u64;

        (lo, hi)
    }

    /// Check if lhs is negative (when treated as signed).
    pub fn lhs_is_negative(&self) -> bool {
        self.lhs_signed && (self.lhs as i64) < 0
    }

    /// Check if rhs is negative (when treated as signed).
    pub fn rhs_is_negative(&self) -> bool {
        self.rhs_signed && (self.rhs as i64) < 0
    }

    /// Get sign-extended lhs as 8 halfwords.
    ///
    /// lhs_ext[0..4] = lhs halfwords
    /// lhs_ext[4..8] = 0xFFFF if negative, 0 otherwise
    pub fn lhs_extended(&self) -> [u64; 8] {
        let fill = if self.lhs_is_negative() { SIGN_FILL } else { 0 };
        [
            self.lhs & 0xFFFF,
            (self.lhs >> 16) & 0xFFFF,
            (self.lhs >> 32) & 0xFFFF,
            (self.lhs >> 48) & 0xFFFF,
            fill,
            fill,
            fill,
            fill,
        ]
    }

    /// Get sign-extended rhs as 8 halfwords.
    pub fn rhs_extended(&self) -> [u64; 8] {
        let fill = if self.rhs_is_negative() { SIGN_FILL } else { 0 };
        [
            self.rhs & 0xFFFF,
            (self.rhs >> 16) & 0xFFFF,
            (self.rhs >> 32) & 0xFFFF,
            (self.rhs >> 48) & 0xFFFF,
            fill,
            fill,
            fill,
            fill,
        ]
    }

    /// Compute the raw_product values (convolution intermediates).
    ///
    /// raw_product[i] = Σ_k=0^1 2^(16k) × Σ_j=0^(2i+k) lhs_ext[j] × rhs_ext[2i+k-j]
    #[allow(clippy::needless_range_loop)]
    pub fn compute_raw_products(&self) -> [u64; 4] {
        let lhs_ext = self.lhs_extended();
        let rhs_ext = self.rhs_extended();

        let mut raw = [0u64; 4];

        for i in 0..4 {
            let mut sum: u128 = 0;

            for k in 0..=1 {
                let idx = 2 * i + k;
                if idx < 8 {
                    for j in 0..=idx {
                        if j < 8 && (idx - j) < 8 {
                            let term = (lhs_ext[j] as u128) * (rhs_ext[idx - j] as u128);
                            sum += term << (16 * k);
                        }
                    }
                }
            }

            raw[i] = sum as u64;
        }

        raw
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the MUL trace table from a list of operations.
///
/// Operations are deduplicated by (lhs, lhs_signed, rhs, rhs_signed).
/// Each unique operation tracks separate multiplicities for lo and hi lookups.
///
/// # Arguments
/// * `operations` - List of (MulOperation, wants_hi) pairs
pub fn generate_mul_trace(
    operations: &[(MulOperation, bool)],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // Deduplicate: (lhs, lhs_signed, rhs, rhs_signed) -> (mu_lo, mu_hi)
    let mut op_map: HashMap<MulOperation, MulMultiplicities> = HashMap::new();

    for (op, wants_hi) in operations {
        let entry = op_map.entry(op.clone()).or_default();
        if *wants_hi {
            entry.mu_hi += 1;
        } else {
            entry.mu_lo += 1;
        }
    }

    // Canonical row order: HashMap iteration order is per-process random, so
    // sort to keep the committed trace deterministic across runs.
    let mut unique_ops: Vec<_> = op_map.into_iter().collect();
    unique_ops.sort_unstable_by_key(|(op, _)| (op.lhs, op.lhs_signed, op.rhs, op.rhs_signed));
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(num_rows * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (op, multiplicities)) in unique_ops.iter().enumerate() {
        // Compute product
        let (lo, hi) = op.compute_product();

        // Fill lhs as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::LHS_0, op.lhs);
        table.set_bool(row_idx, cols::LHS_SIGNED, op.lhs_signed);

        // Fill rhs as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::RHS_0, op.rhs);
        table.set_bool(row_idx, cols::RHS_SIGNED, op.rhs_signed);

        // Fill lo as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::LO_0, lo);

        // Fill hi as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::HI_0, hi);

        // Fill auxiliary columns
        table.set_bool(row_idx, cols::LHS_IS_NEGATIVE, op.lhs_is_negative());
        table.set_bool(row_idx, cols::RHS_IS_NEGATIVE, op.rhs_is_negative());

        // Fill raw_product columns
        let raw = op.compute_raw_products();
        table.set_u64(row_idx, cols::RAW_PRODUCT_0, raw[0]);
        table.set_u64(row_idx, cols::RAW_PRODUCT_1, raw[1]);
        table.set_u64(row_idx, cols::RAW_PRODUCT_2, raw[2]);
        table.set_u64(row_idx, cols::RAW_PRODUCT_3, raw[3]);

        // Fill multiplicities (ALU bus, lo/hi)
        table.set_u64(row_idx, cols::MU_LO, multiplicities.mu_lo);
        table.set_u64(row_idx, cols::MU_HI, multiplicities.mu_hi);
    }

    trace
}

// =========================================================================
// Bus Interactions
// =========================================================================

/// Creates all bus interactions for the MUL table.
///
/// The MUL table:
/// - **Sends** MSB16 lookups for sign bit extraction (×2)
/// - **Sends** IS_HALF lookups for lhs/rhs input and lo/hi output range checks (×16)
/// - **Sends** IS_B20 lookups for carry range checks (×4)
/// - **Receives** MUL lookups from CPU table (×2: lo and hi)
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

    // -------------------------------------------------------------------------
    // MSB16 lookups for sign bit extraction
    // -------------------------------------------------------------------------
    // MSB16[lhs[3]] -> lhs_is_negative (when lhs_signed=1)
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::LHS_SIGNED),
        vec![
            BusValue::Packed {
                start_column: cols::LHS_3,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::LHS_IS_NEGATIVE,
                packing: Packing::Direct,
            },
        ],
    ));

    // MSB16[rhs[3]] -> rhs_is_negative (when rhs_signed=1)
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::RHS_SIGNED),
        vec![
            BusValue::Packed {
                start_column: cols::RHS_3,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::RHS_IS_NEGATIVE,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // IS_HALF lookups for lhs/rhs INPUT range checks (multiplicity: mu_lo + mu_hi).
    // The bus binds only the packed 32-bit words, so without these the input
    // half-limbs are free (non-canonical halves re-packing to the same word).
    // -------------------------------------------------------------------------
    for col in [
        cols::LHS_0,
        cols::LHS_1,
        cols::LHS_2,
        cols::LHS_3,
        cols::RHS_0,
        cols::RHS_1,
        cols::RHS_2,
        cols::RHS_3,
    ] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_LO, cols::MU_HI),
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // IS_HALF lookups for lo range checks (multiplicity: mu_lo + mu_hi)
    // -------------------------------------------------------------------------
    for col in [cols::LO_0, cols::LO_1, cols::LO_2, cols::LO_3] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            row_mult(),
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // IS_HALF lookups for hi range checks (multiplicity: mu_lo + mu_hi)
    // -------------------------------------------------------------------------
    for col in [cols::HI_0, cols::HI_1, cols::HI_2, cols::HI_3] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            row_mult(),
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // IS_B20 lookups for carry range checks (multiplicity: mu_lo + mu_hi)
    // Carries are virtual (computed inline) as linear combinations:
    //   carry[0] = 2^-32 * (raw_product[0] - res[0])
    //   carry[i] = 2^-32 * (raw_product[i] + carry[i-1] - res[i])
    // where res = [lo_word0, lo_word1, hi_word0, hi_word1]
    // -------------------------------------------------------------------------

    // carry[0] = 2^-32 * raw_product[0] - 2^-32 * lo[0] - 2^-16 * lo[1]
    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        row_mult(),
        vec![BusValue::linear(vec![
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_32,
                column: cols::RAW_PRODUCT_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_32,
                column: cols::LO_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_16,
                column: cols::LO_1,
            },
        ])],
    ));

    // carry[1] = 2^-32 * raw_product[1] + 2^-64 * raw_product[0]
    //          - 2^-64 * lo[0] - 2^-48 * lo[1] - 2^-32 * lo[2] - 2^-16 * lo[3]
    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        row_mult(),
        vec![BusValue::linear(vec![
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_32,
                column: cols::RAW_PRODUCT_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_64,
                column: cols::RAW_PRODUCT_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_64,
                column: cols::LO_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_48,
                column: cols::LO_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_32,
                column: cols::LO_2,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_16,
                column: cols::LO_3,
            },
        ])],
    ));

    // carry[2] = 2^-32 * raw_product[2] + 2^-64 * raw_product[1] + 2^-96 * raw_product[0]
    //          - 2^-96 * lo[0] - 2^-80 * lo[1] - 2^-64 * lo[2] - 2^-48 * lo[3]
    //          - 2^-32 * hi[0] - 2^-16 * hi[1]
    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        row_mult(),
        vec![BusValue::linear(vec![
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_32,
                column: cols::RAW_PRODUCT_2,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_64,
                column: cols::RAW_PRODUCT_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_96,
                column: cols::RAW_PRODUCT_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_96,
                column: cols::LO_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_80,
                column: cols::LO_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_64,
                column: cols::LO_2,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_48,
                column: cols::LO_3,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_32,
                column: cols::HI_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_16,
                column: cols::HI_1,
            },
        ])],
    ));

    // carry[3] = 2^-32 * raw_product[3] + 2^-64 * raw_product[2] + 2^-96 * raw_product[1] + 2^-128 * raw_product[0]
    //          - 2^-128 * lo[0] - 2^-112 * lo[1] - 2^-96 * lo[2] - 2^-80 * lo[3]
    //          - 2^-64 * hi[0] - 2^-48 * hi[1] - 2^-32 * hi[2] - 2^-16 * hi[3]
    interactions.push(BusInteraction::sender(
        BusId::IsB20,
        row_mult(),
        vec![BusValue::linear(vec![
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_32,
                column: cols::RAW_PRODUCT_3,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_64,
                column: cols::RAW_PRODUCT_2,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_96,
                column: cols::RAW_PRODUCT_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: INV_2_128,
                column: cols::RAW_PRODUCT_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_128,
                column: cols::LO_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_112,
                column: cols::LO_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_96,
                column: cols::LO_2,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_80,
                column: cols::LO_3,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_64,
                column: cols::HI_0,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_48,
                column: cols::HI_1,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_32,
                column: cols::HI_2,
            },
            LinearTerm::ColumnUnsigned {
                coefficient: NEG_INV_2_16,
                column: cols::HI_3,
            },
        ])],
    ));

    // -------------------------------------------------------------------------
    // ALU receivers: every MUL lookup arrives here — CPU
    // MUL/MULH/MULHSU/MULHU dispatch and dvrm's internal `d*q` consistency.
    // ALU[lhs, rhs, flags, result] where flags =
    //   opsel(MUL) + 32*lhs_signed + 64*rhs_signed (+128 for the hi result).
    // -------------------------------------------------------------------------
    let mul_flags = |hi: i64| {
        BusValue::linear(vec![
            LinearTerm::Constant(alu_op::MUL as i64 + hi),
            LinearTerm::Column {
                coefficient: 32,
                column: cols::LHS_SIGNED,
            },
            LinearTerm::Column {
                coefficient: 64,
                column: cols::RHS_SIGNED,
            },
        ])
    };
    // ALU lo (muldiv bit 7 = 0)
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU_LO),
        vec![
            BusValue::Packed {
                start_column: cols::LHS_0,
                packing: Packing::DWordHL,
            },
            BusValue::Packed {
                start_column: cols::RHS_0,
                packing: Packing::DWordHL,
            },
            mul_flags(0),
            BusValue::Packed {
                start_column: cols::LO_0,
                packing: Packing::DWordHL,
            },
        ],
    ));
    // ALU hi (muldiv bit 7 = 1 => +128)
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU_HI),
        vec![
            BusValue::Packed {
                start_column: cols::LHS_0,
                packing: Packing::DWordHL,
            },
            BusValue::Packed {
                start_column: cols::RHS_0,
                packing: Packing::DWordHL,
            },
            mul_flags(128),
            BusValue::Packed {
                start_column: cols::HI_0,
                packing: Packing::DWordHL,
            },
        ],
    ));

    interactions
}

// =========================================================================
// Single-body constraint set (ConstraintSet front-end)
// =========================================================================
//
// One body against the generic `ConstraintBuilder` serves the compiled prover
// folder, the verifier folder and IR capture. Constraint indices 0..8:
//   0: SignedIsBit(LHS_SIGNED)  1: SignedIsBit(RHS_SIGNED)
//   2: LhsSign                  3: RhsSign
//   4..8: RawProduct(0..4)

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// MUL table constraints as a single-source [`ConstraintSet`]. No column
/// configuration is needed (the MUL layout is fixed via `cols`).
pub struct MulConstraints;

impl MulConstraints {
    /// `x · (1 − x)` IS_BIT check for a sign-flag column.
    fn signed_is_bit<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        col: usize,
    ) -> B::Expr {
        let x = b.main(0, col);
        let one = b.one();
        x.clone() * (one - x)
    }

    /// `raw_product[i] − Σ_k 2^(16k)·Σ_j lhs_ext[j]·rhs_ext[idx−j]` (idx = 2i+k).
    fn raw_product<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
    ) -> B::Expr {
        let lhs = [
            b.main(0, cols::LHS_0),
            b.main(0, cols::LHS_1),
            b.main(0, cols::LHS_2),
            b.main(0, cols::LHS_3),
        ];
        let rhs = [
            b.main(0, cols::RHS_0),
            b.main(0, cols::RHS_1),
            b.main(0, cols::RHS_2),
            b.main(0, cols::RHS_3),
        ];
        let lhs_is_neg = b.main(0, cols::LHS_IS_NEGATIVE);
        let rhs_is_neg = b.main(0, cols::RHS_IS_NEGATIVE);

        // Sign-extended values: [0..4] = halfwords, [4..8] = sign_fill * is_neg.
        // Known redundancy: the two sign-fill products are rebuilt in each of
        // the four raw_product constraints. Hoisting them was tried and showed
        // no measurable speedup (ABBA), so the body keeps the declarative form.
        let sign_fill = b.const_base(SIGN_FILL);
        let lhs_hi = sign_fill.clone() * lhs_is_neg;
        let rhs_hi = sign_fill * rhs_is_neg;
        let lhs_ext: [B::Expr; 8] = [
            lhs[0].clone(),
            lhs[1].clone(),
            lhs[2].clone(),
            lhs[3].clone(),
            lhs_hi.clone(),
            lhs_hi.clone(),
            lhs_hi.clone(),
            lhs_hi,
        ];
        let rhs_ext: [B::Expr; 8] = [
            rhs[0].clone(),
            rhs[1].clone(),
            rhs[2].clone(),
            rhs[3].clone(),
            rhs_hi.clone(),
            rhs_hi.clone(),
            rhs_hi.clone(),
            rhs_hi,
        ];

        // Convolution sum.
        let shift_16 = b.const_base(SHIFT_16);
        let mut sum = b.zero();
        for k in 0..=1usize {
            let idx = 2 * i + k;
            if idx < 8 {
                let mut inner_sum = b.zero();
                for j in 0..=idx {
                    if j < 8 && (idx - j) < 8 {
                        inner_sum = inner_sum + lhs_ext[j].clone() * rhs_ext[idx - j].clone();
                    }
                }
                if k == 0 {
                    sum = sum + inner_sum;
                } else {
                    sum = sum + inner_sum * shift_16.clone();
                }
            }
        }

        let raw_col = match i {
            0 => cols::RAW_PRODUCT_0,
            1 => cols::RAW_PRODUCT_1,
            2 => cols::RAW_PRODUCT_2,
            3 => cols::RAW_PRODUCT_3,
            _ => unreachable!(),
        };
        let raw_product = b.main(0, raw_col);
        raw_product - sum
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for MulConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0,1: IS_BIT range checks on the sign-flag multiplicities.
        let is_bit_lhs = Self::signed_is_bit(b, cols::LHS_SIGNED);
        b.emit_base(0, is_bit_lhs);
        let is_bit_rhs = Self::signed_is_bit(b, cols::RHS_SIGNED);
        b.emit_base(1, is_bit_rhs);

        // idx 2: LhsSign: (1 - lhs_signed) * lhs_is_negative
        let lhs_signed = b.main(0, cols::LHS_SIGNED);
        let lhs_is_neg = b.main(0, cols::LHS_IS_NEGATIVE);
        let one = b.one();
        b.emit_base(2, (one - lhs_signed) * lhs_is_neg);

        // idx 3: RhsSign: (1 - rhs_signed) * rhs_is_negative
        let rhs_signed = b.main(0, cols::RHS_SIGNED);
        let rhs_is_neg = b.main(0, cols::RHS_IS_NEGATIVE);
        let one = b.one();
        b.emit_base(3, (one - rhs_signed) * rhs_is_neg);

        // idx 4..8: raw_product convolution for i = 0..4.
        for i in 0..4 {
            let root = Self::raw_product(b, i);
            b.emit_base(4 + i, root);
        }
    }
}

// =========================================================================
// Helper functions
// =========================================================================

/// Compute the virtual carry values for verification.
///
/// This is used by tests to verify the reduce-and-carry logic.
pub fn compute_carries(lo: u64, hi: u64, raw_products: &[u64; 4]) -> [u64; 4] {
    // res[0..4] = [lo_word0, lo_word1, hi_word0, hi_word1]
    let res = [lo & 0xFFFF_FFFF, lo >> 32, hi & 0xFFFF_FFFF, hi >> 32];

    let mut carries = [0u64; 4];

    // carry[0] = (raw_product[0] - res[0]) / 2^32
    let diff0 = raw_products[0].wrapping_sub(res[0]);
    carries[0] = diff0 >> 32;

    // carry[i] = (raw_product[i] + carry[i-1] - res[i]) / 2^32
    for i in 1..4 {
        let sum = raw_products[i]
            .wrapping_add(carries[i - 1])
            .wrapping_sub(res[i]);
        carries[i] = sum >> 32;
    }

    carries
}
