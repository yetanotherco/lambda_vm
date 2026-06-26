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

use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "prove")]
use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, INV_2_32, INV_2_64, INV_2_96, INV_2_128,
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
#[cfg(feature = "prove")]
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

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
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
    // Carries are virtual columns computed as linear combinations:
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
// Constraints
// =========================================================================

/// MUL table constraint kinds.
#[derive(Debug, Clone, Copy)]
pub enum MulConstraintKind {
    /// SIGN constraint for lhs: (1 - lhs_signed) * lhs_is_negative = 0
    LhsSign,
    /// SIGN constraint for rhs: (1 - rhs_signed) * rhs_is_negative = 0
    RhsSign,
    /// IS_BIT range check on a sign flag column: `x * (1 - x) = 0`. Required
    /// because `lhs_signed`/`rhs_signed` are used as bus multiplicities, so an
    /// out-of-range value (e.g. `lhs_signed = 3`) would otherwise be accepted.
    SignedIsBit(usize),
    /// Raw product convolution formula for index i
    RawProduct(usize),
}

/// MUL table constraint.
pub struct MulConstraint {
    constraint_idx: usize,
    kind: MulConstraintKind,
}

impl MulConstraint {
    /// Create a new MUL constraint.
    pub fn new(kind: MulConstraintKind, constraint_idx: usize) -> Self {
        Self {
            constraint_idx,
            kind,
        }
    }

    /// Compute the constraint value.
    fn compute<F, E>(&self, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        match self.kind {
            MulConstraintKind::LhsSign => {
                // (1 - lhs_signed) * lhs_is_negative = 0
                let lhs_signed = step
                    .get_main_evaluation_element(0, cols::LHS_SIGNED)
                    .clone();
                let lhs_is_neg = step
                    .get_main_evaluation_element(0, cols::LHS_IS_NEGATIVE)
                    .clone();
                let one = FieldElement::<F>::one();
                (&one - &lhs_signed) * &lhs_is_neg
            }
            MulConstraintKind::RhsSign => {
                // (1 - rhs_signed) * rhs_is_negative = 0
                let rhs_signed = step
                    .get_main_evaluation_element(0, cols::RHS_SIGNED)
                    .clone();
                let rhs_is_neg = step
                    .get_main_evaluation_element(0, cols::RHS_IS_NEGATIVE)
                    .clone();
                let one = FieldElement::<F>::one();
                (&one - &rhs_signed) * &rhs_is_neg
            }
            MulConstraintKind::SignedIsBit(col) => {
                // x * (1 - x) = 0
                let x = step.get_main_evaluation_element(0, col).clone();
                let one = FieldElement::<F>::one();
                &x * &(&one - &x)
            }
            MulConstraintKind::RawProduct(i) => {
                // raw_product[i] = convolution formula
                // This requires computing the sign-extended values and convolution
                self.compute_raw_product_constraint(i, step)
            }
        }
    }

    /// Compute raw_product constraint for index i.
    ///
    /// raw_product[i] = Σ_k=0^1 2^(16k) × Σ_j=0^(2i+k) lhs_ext[j] × rhs_ext[2i+k-j]
    fn compute_raw_product_constraint<F, E>(
        &self,
        i: usize,
        step: &TableView<F, E>,
    ) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        // Get lhs halfwords
        let lhs: [FieldElement<F>; 4] = [
            step.get_main_evaluation_element(0, cols::LHS_0).clone(),
            step.get_main_evaluation_element(0, cols::LHS_1).clone(),
            step.get_main_evaluation_element(0, cols::LHS_2).clone(),
            step.get_main_evaluation_element(0, cols::LHS_3).clone(),
        ];

        // Get rhs halfwords
        let rhs: [FieldElement<F>; 4] = [
            step.get_main_evaluation_element(0, cols::RHS_0).clone(),
            step.get_main_evaluation_element(0, cols::RHS_1).clone(),
            step.get_main_evaluation_element(0, cols::RHS_2).clone(),
            step.get_main_evaluation_element(0, cols::RHS_3).clone(),
        ];

        // Get sign bits
        let lhs_is_neg = step
            .get_main_evaluation_element(0, cols::LHS_IS_NEGATIVE)
            .clone();
        let rhs_is_neg = step
            .get_main_evaluation_element(0, cols::RHS_IS_NEGATIVE)
            .clone();

        // Build sign-extended values
        let sign_fill = FieldElement::<F>::from(SIGN_FILL);
        let mut lhs_ext: [FieldElement<F>; 8] = core::array::from_fn(|_| FieldElement::zero());
        let mut rhs_ext: [FieldElement<F>; 8] = core::array::from_fn(|_| FieldElement::zero());

        lhs_ext[..4].clone_from_slice(&lhs);
        rhs_ext[..4].clone_from_slice(&rhs);
        for j in 4..8 {
            lhs_ext[j] = &sign_fill * &lhs_is_neg;
            rhs_ext[j] = &sign_fill * &rhs_is_neg;
        }

        // Compute convolution sum
        let shift_16 = FieldElement::<F>::from(SHIFT_16);
        let mut sum = FieldElement::<F>::zero();

        for k in 0..=1u32 {
            let idx = 2 * i + k as usize;
            if idx < 8 {
                let mut inner_sum = FieldElement::<F>::zero();
                for j in 0..=idx {
                    if j < 8 && (idx - j) < 8 {
                        inner_sum = &inner_sum + &(&lhs_ext[j] * &rhs_ext[idx - j]);
                    }
                }
                // Multiply by 2^(16*k)
                if k == 0 {
                    sum = &sum + &inner_sum;
                } else {
                    sum = &sum + &(&inner_sum * &shift_16);
                }
            }
        }

        // Constraint: raw_product[i] - sum = 0
        let raw_col = match i {
            0 => cols::RAW_PRODUCT_0,
            1 => cols::RAW_PRODUCT_1,
            2 => cols::RAW_PRODUCT_2,
            3 => cols::RAW_PRODUCT_3,
            _ => unreachable!(),
        };
        let raw_product = step.get_main_evaluation_element(0, raw_col).clone();

        raw_product - sum
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for MulConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            // (1 - signed) * is_negative is degree 2
            MulConstraintKind::LhsSign | MulConstraintKind::RhsSign => 2,
            // x * (1 - x) is degree 2
            MulConstraintKind::SignedIsBit(_) => 2,
            // Raw product: lhs_ext[j] * rhs_ext[idx-j] where each may involve
            // sign_fill * is_negative (degree 1), so product is degree 2
            // But we're summing many degree-2 terms, still degree 2
            MulConstraintKind::RawProduct(_) => 2,
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

/// Creates all constraints for the MUL table.
///
/// Returns: (constraints, next_constraint_idx)
pub fn mul_constraints(constraint_idx_start: usize) -> (Vec<MulConstraint>, usize) {
    let mut idx = constraint_idx_start;
    let mut constraints = Vec::new();

    // IS_BIT range checks on the sign flags (used as bus multiplicities).
    constraints.push(MulConstraint::new(
        MulConstraintKind::SignedIsBit(cols::LHS_SIGNED),
        idx,
    ));
    idx += 1;
    constraints.push(MulConstraint::new(
        MulConstraintKind::SignedIsBit(cols::RHS_SIGNED),
        idx,
    ));
    idx += 1;

    // SIGN constraints
    constraints.push(MulConstraint::new(MulConstraintKind::LhsSign, idx));
    idx += 1;
    constraints.push(MulConstraint::new(MulConstraintKind::RhsSign, idx));
    idx += 1;

    // Raw product constraints for i in 0..4
    for i in 0..4 {
        constraints.push(MulConstraint::new(MulConstraintKind::RawProduct(i), idx));
        idx += 1;
    }

    (constraints, idx)
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
