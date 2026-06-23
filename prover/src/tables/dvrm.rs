//! DVRM table for 64-bit division/remainder.
//!
//! This table computes quotient and remainder for 64-bit division,
//! supporting both signed and unsigned operations.
//!
//! ## Inputs
//! - `n`: DWordHL (64-bit as 4 halfwords) - numerator
//! - `d`: DWordHL (64-bit as 4 halfwords) - denominator
//! - `signed`: Bit - whether to treat operands as signed
//!
//! ## Outputs
//! - `q`: DWordHL (64-bit) - quotient
//! - `r`: DWordHL (64-bit) - remainder
//!
//! ## Auxiliary
//! - `div_by_zero`: Bit - denominator is zero
//! - `overflow`: Bit - signed overflow (MIN / -1)
//! - `abs_r`: DWordWL (2 words) - absolute value of remainder
//! - `abs_d`: DWordWL (2 words) - absolute value of denominator
//! - `n_sub_r`: DWordHL (4 halfwords) - numerator minus remainder
//! - `sign_n_sub_r`: Bit - sign of (n - r)
//! - `sign_n`, `sign_d`, `sign_q`, `sign_r`: Bit - sign bits
//!
//! ## Bus Interactions
//! - Sender: IS_HALF (×16: n, d, r, n_sub_r, q)
//! - Sender: MSB16 (×3 for sign extraction: n, d, r)
//! - Sender: LT (×1 for abs_r < abs_d)
//! - Sender: MUL (×2 for n_sub_r = d * q verification)
//! - Sender: ZERO (×5 for div_by_zero, overflow, NEG template)
//! - Receiver: DVRM (×2 for quotient and remainder results)

use alloc::vec;
use alloc::vec::Vec;
#[cfg(feature = "prove")]
use std::collections::HashMap;

use math::field::element::FieldElement;
use math::field::traits::{IsField, IsSubFieldOf};
use smallvec::smallvec;
use stark::constraints::transition::TransitionConstraint;
use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::table::TableView;
use stark::trace::TraceTable;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, NEG_INV_2_16, NEG_INV_2_32, NEG_INV_2_48,
    NEG_INV_2_64, SHIFT_16,
};

// =========================================================================
// Column indices for DVRM table
// =========================================================================

/// Column definitions for the DVRM table.
pub mod cols {
    // Input columns: n (numerator) as DWordHL (4 halfwords)
    /// n[0]: Half (bits 0-15)
    pub const N_0: usize = 0;
    /// n[1]: Half (bits 16-31)
    pub const N_1: usize = 1;
    /// n[2]: Half (bits 32-47)
    pub const N_2: usize = 2;
    /// n[3]: Half (bits 48-63)
    pub const N_3: usize = 3;

    // Input columns: d (denominator) as DWordHL (4 halfwords)
    /// d[0]: Half (bits 0-15)
    pub const D_0: usize = 4;
    /// d[1]: Half (bits 16-31)
    pub const D_1: usize = 5;
    /// d[2]: Half (bits 32-47)
    pub const D_2: usize = 6;
    /// d[3]: Half (bits 48-63)
    pub const D_3: usize = 7;

    /// signed: Bit (1 = signed, 0 = unsigned)
    pub const SIGNED: usize = 8;

    // Output columns: q (quotient) as DWordHL (4 halfwords)
    /// q[0]: Half (bits 0-15)
    pub const Q_0: usize = 9;
    /// q[1]: Half (bits 16-31)
    pub const Q_1: usize = 10;
    /// q[2]: Half (bits 32-47)
    pub const Q_2: usize = 11;
    /// q[3]: Half (bits 48-63)
    pub const Q_3: usize = 12;

    // Output columns: r (remainder) as DWordHL (4 halfwords)
    /// r[0]: Half (bits 0-15)
    pub const R_0: usize = 13;
    /// r[1]: Half (bits 16-31)
    pub const R_1: usize = 14;
    /// r[2]: Half (bits 32-47)
    pub const R_2: usize = 15;
    /// r[3]: Half (bits 48-63)
    pub const R_3: usize = 16;

    // Auxiliary columns
    /// div_by_zero: Bit (1 if denominator is 0)
    pub const DIV_BY_ZERO: usize = 17;
    /// overflow: Bit (1 if signed overflow: MIN / -1)
    pub const OVERFLOW: usize = 18;

    // abs_r: DWordWL (2 words) - absolute value of remainder
    /// abs_r[0]: Word (bits 0-31)
    pub const ABS_R_0: usize = 19;
    /// abs_r[1]: Word (bits 32-63)
    pub const ABS_R_1: usize = 20;

    // abs_d: DWordWL (2 words) - absolute value of denominator
    /// abs_d[0]: Word (bits 0-31)
    pub const ABS_D_0: usize = 21;
    /// abs_d[1]: Word (bits 32-63)
    pub const ABS_D_1: usize = 22;

    // n_sub_r: DWordHL (4 halfwords) - n - r
    /// n_sub_r[0]: Half (bits 0-15)
    pub const N_SUB_R_0: usize = 23;
    /// n_sub_r[1]: Half (bits 16-31)
    pub const N_SUB_R_1: usize = 24;
    /// n_sub_r[2]: Half (bits 32-47)
    pub const N_SUB_R_2: usize = 25;
    /// n_sub_r[3]: Half (bits 48-63)
    pub const N_SUB_R_3: usize = 26;

    /// sign_n_sub_r: Bit (sign of n - r)
    pub const SIGN_N_SUB_R: usize = 27;

    /// sign_n: Bit (sign of numerator)
    pub const SIGN_N: usize = 28;
    /// sign_d: Bit (sign of denominator)
    pub const SIGN_D: usize = 29;
    /// sign_q: Bit (sign of quotient)
    pub const SIGN_Q: usize = 30;
    /// sign_r: Bit (sign of remainder)
    pub const SIGN_R: usize = 31;

    // Multiplicity columns
    /// μ_q: multiplicity for quotient result lookups
    pub const MU_Q: usize = 32;
    /// μ_r: multiplicity for remainder result lookups
    pub const MU_R: usize = 33;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 34;
}

// =========================================================================
// Constants
// =========================================================================

/// Sign extension fill value: 0xFFFF (all 1s for 16 bits)
const SIGN_FILL: u64 = 0xFFFF;

// =========================================================================
// DvrmOperation struct
// =========================================================================

/// A single DVRM operation to be added to the trace.
///
/// Derives Hash and Eq for HashMap-based deduplication.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DvrmOperation {
    /// Numerator (64-bit)
    pub n: u64,
    /// Denominator (64-bit)
    pub d: u64,
    /// Whether to treat operands as signed
    pub signed: bool,
}

/// Multiplicities for a DVRM operation (separate for quotient and remainder lookups).
#[derive(Debug, Clone, Default)]
pub struct DvrmMultiplicities {
    /// Count of lookups requesting quotient result
    pub mu_q: u64,
    /// Count of lookups requesting remainder result
    pub mu_r: u64,
}

impl DvrmOperation {
    /// Create a new DVRM operation.
    pub fn new(n: u64, d: u64, signed: bool) -> Self {
        Self { n, d, signed }
    }

    /// Check if this is a division by zero.
    pub fn is_div_by_zero(&self) -> bool {
        self.d == 0
    }

    /// Check if this is a signed overflow case (MIN / -1).
    pub fn is_overflow(&self) -> bool {
        self.signed && self.n == (i64::MIN as u64) && self.d == (u64::MAX) // -1 as u64
    }

    /// Compute the quotient per RISC-V spec.
    pub fn compute_quotient(&self) -> u64 {
        if self.is_div_by_zero() {
            // Division by zero: quotient = all 1s
            u64::MAX
        } else if self.is_overflow() {
            // Signed overflow (MIN / -1): quotient = MIN
            self.n // i64::MIN
        } else if self.signed {
            ((self.n as i64).wrapping_div(self.d as i64)) as u64
        } else {
            self.n / self.d
        }
    }

    /// Compute the remainder per RISC-V spec.
    pub fn compute_remainder(&self) -> u64 {
        if self.is_div_by_zero() {
            // Division by zero: remainder = numerator
            self.n
        } else if self.is_overflow() {
            // Signed overflow (MIN / -1): remainder = 0
            0
        } else if self.signed {
            ((self.n as i64).wrapping_rem(self.d as i64)) as u64
        } else {
            self.n % self.d
        }
    }

    /// Get the sign bit of the numerator (bit 63).
    pub fn sign_n(&self) -> bool {
        self.signed && (self.n >> 63) == 1
    }

    /// Get the sign bit of the denominator (bit 63).
    pub fn sign_d(&self) -> bool {
        self.signed && (self.d >> 63) == 1
    }

    /// Get the sign_q flag: whether q should be treated as signed in MUL verification.
    /// Per spec DVRM-C7: sign_q = signed * (1 - overflow)
    pub fn sign_q(&self) -> bool {
        self.signed && !self.is_overflow()
    }

    /// Get the sign bit of the remainder (bit 63).
    pub fn sign_r(&self) -> bool {
        let r = self.compute_remainder();
        self.signed && (r >> 63) == 1
    }

    /// Compute the absolute value of a signed 64-bit number.
    fn abs_value(val: u64, is_negative: bool) -> u64 {
        if is_negative {
            (val as i64).unsigned_abs()
        } else {
            val
        }
    }

    /// Compute absolute value of remainder.
    pub fn abs_r(&self) -> u64 {
        Self::abs_value(self.compute_remainder(), self.sign_r())
    }

    /// Compute absolute value of denominator.
    pub fn abs_d(&self) -> u64 {
        Self::abs_value(self.d, self.sign_d())
    }

    /// Compute n - r (numerator minus remainder).
    pub fn n_sub_r(&self) -> u64 {
        self.n.wrapping_sub(self.compute_remainder())
    }

    /// Sign of n_sub_r.
    pub fn sign_n_sub_r(&self) -> bool {
        self.signed && (self.n_sub_r() >> 63) == 1
    }
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the DVRM trace table from a list of operations.
///
/// Operations are deduplicated by (n, d, signed).
/// Each unique operation tracks separate multiplicities for quotient and remainder lookups.
///
/// # Arguments
/// * `operations` - List of (DvrmOperation, wants_remainder) pairs
#[cfg(feature = "prove")]
pub fn generate_dvrm_trace(
    operations: &[(DvrmOperation, bool)],
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    // Deduplicate: (n, d, signed) -> (mu_q, mu_r)
    let mut op_map: HashMap<DvrmOperation, DvrmMultiplicities> = HashMap::new();

    for (op, wants_remainder) in operations {
        let entry = op_map.entry(op.clone()).or_default();
        if *wants_remainder {
            entry.mu_r += 1;
        } else {
            entry.mu_q += 1;
        }
    }

    let unique_ops: Vec<_> = op_map.into_iter().collect();
    let num_rows = unique_ops.len().next_power_of_two().max(4);
    let mut data = vec![FE::zero(); num_rows * cols::NUM_COLUMNS];

    for (row_idx, (op, multiplicities)) in unique_ops.iter().enumerate() {
        let base = row_idx * cols::NUM_COLUMNS;

        let q = op.compute_quotient();
        let r = op.compute_remainder();
        let n_sub_r = op.n_sub_r();
        let abs_r = op.abs_r();
        let abs_d = op.abs_d();

        // Fill n as DWordHL (4 halfwords)
        data[base + cols::N_0] = FE::from(op.n & 0xFFFF);
        data[base + cols::N_1] = FE::from((op.n >> 16) & 0xFFFF);
        data[base + cols::N_2] = FE::from((op.n >> 32) & 0xFFFF);
        data[base + cols::N_3] = FE::from((op.n >> 48) & 0xFFFF);

        // Fill d as DWordHL (4 halfwords)
        data[base + cols::D_0] = FE::from(op.d & 0xFFFF);
        data[base + cols::D_1] = FE::from((op.d >> 16) & 0xFFFF);
        data[base + cols::D_2] = FE::from((op.d >> 32) & 0xFFFF);
        data[base + cols::D_3] = FE::from((op.d >> 48) & 0xFFFF);

        data[base + cols::SIGNED] = FE::from(op.signed as u64);

        // Fill q as DWordHL (4 halfwords)
        data[base + cols::Q_0] = FE::from(q & 0xFFFF);
        data[base + cols::Q_1] = FE::from((q >> 16) & 0xFFFF);
        data[base + cols::Q_2] = FE::from((q >> 32) & 0xFFFF);
        data[base + cols::Q_3] = FE::from((q >> 48) & 0xFFFF);

        // Fill r as DWordHL (4 halfwords)
        data[base + cols::R_0] = FE::from(r & 0xFFFF);
        data[base + cols::R_1] = FE::from((r >> 16) & 0xFFFF);
        data[base + cols::R_2] = FE::from((r >> 32) & 0xFFFF);
        data[base + cols::R_3] = FE::from((r >> 48) & 0xFFFF);

        // Fill auxiliary columns
        data[base + cols::DIV_BY_ZERO] = FE::from(op.is_div_by_zero() as u64);
        data[base + cols::OVERFLOW] = FE::from(op.is_overflow() as u64);

        data[base + cols::ABS_R_0] = FE::from(abs_r & 0xFFFF_FFFF);
        data[base + cols::ABS_R_1] = FE::from(abs_r >> 32);

        data[base + cols::ABS_D_0] = FE::from(abs_d & 0xFFFF_FFFF);
        data[base + cols::ABS_D_1] = FE::from(abs_d >> 32);

        // Fill n_sub_r as DWordHL (4 halfwords)
        data[base + cols::N_SUB_R_0] = FE::from(n_sub_r & 0xFFFF);
        data[base + cols::N_SUB_R_1] = FE::from((n_sub_r >> 16) & 0xFFFF);
        data[base + cols::N_SUB_R_2] = FE::from((n_sub_r >> 32) & 0xFFFF);
        data[base + cols::N_SUB_R_3] = FE::from((n_sub_r >> 48) & 0xFFFF);

        data[base + cols::SIGN_N_SUB_R] = FE::from(op.sign_n_sub_r() as u64);
        data[base + cols::SIGN_N] = FE::from(op.sign_n() as u64);
        data[base + cols::SIGN_D] = FE::from(op.sign_d() as u64);
        data[base + cols::SIGN_Q] = FE::from(op.sign_q() as u64);
        data[base + cols::SIGN_R] = FE::from(op.sign_r() as u64);

        // Multiplicities
        data[base + cols::MU_Q] = FE::from(multiplicities.mu_q);
        data[base + cols::MU_R] = FE::from(multiplicities.mu_r);
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

// =========================================================================
// Bus Interactions
// =========================================================================

/// Creates all bus interactions for the DVRM table.
///
/// The DVRM table:
/// - **Sends** IS_HALF lookups for n, d, q, r, n_sub_r range checks (×20)
/// - **Sends** MSB16 lookups for sign extraction (×3: n, d, r)
/// - **Sends** LT lookup for |r| < |d| (×1)
/// - **Sends** MUL lookups for n_sub_r = d * q verification (×2: lo and hi)
/// - **Sends** ZERO lookups for div_by_zero, overflow, NEG carries (×5)
/// - **Receives** DVRM lookups from CPU table (×2: quotient and remainder)
pub fn bus_interactions() -> Vec<BusInteraction> {
    let mut interactions = Vec::new();

    // DVRM-A1.i (IS_HALF[n[i]]) and DVRM-A2.i (IS_HALF[d[i]]) are assumptions:
    // the CPU (sender) is responsible for range-checking n and d before sending
    // to DVRM. The DVRM table does NOT send these IS_HALF lookups.

    // -------------------------------------------------------------------------
    // DVRM-C13.i: IS_HALF[r[i]] (×4), multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    for col in [cols::R_0, cols::R_1, cols::R_2, cols::R_3] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_Q, cols::MU_R),
            smallvec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // DVRM-C14.i: IS_HALF[n_sub_r[i]] (×4), multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    for col in [
        cols::N_SUB_R_0,
        cols::N_SUB_R_1,
        cols::N_SUB_R_2,
        cols::N_SUB_R_3,
    ] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_Q, cols::MU_R),
            smallvec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // DVRM-C11.i: IS_HALF[q[i]] (×4), multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    for col in [cols::Q_0, cols::Q_1, cols::Q_2, cols::Q_3] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_Q, cols::MU_R),
            smallvec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // DVRM-C18 (SIGN): MSB16[sign_n; n[3]] when signed=1
    // Multiplicity: Column(SIGNED) = 0 or 1 per unique row.
    // The trace builder deduplicates MSB16 lookups per unique op.
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::SIGNED),
        smallvec![
            BusValue::Packed {
                start_column: cols::N_3,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGN_N,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C19 (SIGN): MSB16[sign_r; r[3]] when signed=1
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::SIGNED),
        smallvec![
            BusValue::Packed {
                start_column: cols::R_3,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGN_R,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C20 (SIGN): MSB16[sign_d; d[3]] when signed=1
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Msb16,
        Multiplicity::Column(cols::SIGNED),
        smallvec![
            BusValue::Packed {
                start_column: cols::D_3,
                packing: Packing::Direct,
            },
            BusValue::Packed {
                start_column: cols::SIGN_D,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C2: LT[1-div_by_zero; abs_r, abs_d, 0]
    // Verify |r| < |d| when d != 0
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Lt,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        smallvec![
            // abs_r as DWordWL (2 words → 2 elements)
            BusValue::Packed {
                start_column: cols::ABS_R_0,
                packing: Packing::DWordWL,
            },
            // abs_d as DWordWL (2 words → 2 elements)
            BusValue::Packed {
                start_column: cols::ABS_D_0,
                packing: Packing::DWordWL,
            },
            // signed = 0 (unsigned comparison of absolute values)
            BusValue::constant(0),
            // lt_result = 1 - div_by_zero
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::DIV_BY_ZERO,
                },
            ]),
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C9: MUL[n_sub_r::DWordWL; d, signed, q, sign_q, 0]
    // Verify n - r = d * q (lower 64 bits)
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Mul,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        smallvec![
            // d as DWordHL (lhs)
            BusValue::Packed {
                start_column: cols::D_0,
                packing: Packing::DWordHL,
            },
            // lhs_signed = signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // q as DWordHL (rhs)
            BusValue::Packed {
                start_column: cols::Q_0,
                packing: Packing::DWordHL,
            },
            // rhs_signed = sign_q
            BusValue::Packed {
                start_column: cols::SIGN_Q,
                packing: Packing::Direct,
            },
            // result: n_sub_r as DWordHL (lower 64 bits of d*q)
            BusValue::Packed {
                start_column: cols::N_SUB_R_0,
                packing: Packing::DWordHL,
            },
            // muldiv_selector = 0 (lo)
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C10: MUL[extension_n_sub_r::DWordWL; d, signed, q, sign_q, 1]
    // Verify upper 64 bits of d * q = sign extension of n_sub_r
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Mul,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        smallvec![
            // d as DWordHL (lhs)
            BusValue::Packed {
                start_column: cols::D_0,
                packing: Packing::DWordHL,
            },
            // lhs_signed = signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // q as DWordHL (rhs)
            BusValue::Packed {
                start_column: cols::Q_0,
                packing: Packing::DWordHL,
            },
            // rhs_signed = sign_q
            BusValue::Packed {
                start_column: cols::SIGN_Q,
                packing: Packing::Direct,
            },
            // result: sign extension of n_sub_r as DWordHL
            // Each halfword = sign_n_sub_r * 65535
            // lo32 = sign_n_sub_r * (65535 + 65535 * 2^16) = sign_n_sub_r * 0xFFFFFFFF
            // hi32 = same
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: (SIGN_FILL + SIGN_FILL * SHIFT_16) as i64,
                column: cols::SIGN_N_SUB_R,
            }]),
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: (SIGN_FILL + SIGN_FILL * SHIFT_16) as i64,
                column: cols::SIGN_N_SUB_R,
            }]),
            // muldiv_selector = 1 (hi)
            BusValue::constant(1),
        ],
    ));

    // =========================================================================
    // ZERO interactions (C3, C5, C8, C20)
    // =========================================================================

    // -------------------------------------------------------------------------
    // DVRM-C3: sign_r ⇒ NEG<abs_r; r>
    // carry[0] = 2^-32 * ((r::DWordWL)[0] + abs_r[0])
    // carry[1] = 2^-32 * ((r::DWordWL)[1] + abs_r[1] + carry[0])
    // ZERO[1-carry[0]; r[0]+r[1]] with multiplicity sign_r
    // ZERO[1-carry[1]; r[0]+r[1]+r[2]+r[3]] with multiplicity sign_r
    // -------------------------------------------------------------------------

    // C3a: 1 - carry[0] = 1 - 2^-32*r[0] - 2^-16*r[1] - 2^-32*abs_r[0]
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::SIGN_R),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::R_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::R_1,
                },
            ]),
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::ABS_R_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::R_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_16,
                    column: cols::R_1,
                },
            ]),
        ],
    ));

    // C3b: 1 - carry[1] = 1 - 2^-32*r[2] - 2^-16*r[3] - 2^-32*abs_r[1]
    //                        - 2^-64*r[0] - 2^-48*r[1] - 2^-64*abs_r[0]
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::SIGN_R),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::R_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::R_1,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::R_2,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::R_3,
                },
            ]),
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                // Current-level terms (carry[1] direct)
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::ABS_R_1,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::R_2,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_16,
                    column: cols::R_3,
                },
                // carry[0]-dependent terms (shifted by additional 2^-32)
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_64,
                    column: cols::ABS_R_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_64,
                    column: cols::R_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_48,
                    column: cols::R_1,
                },
            ]),
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C5: sign_d ⇒ NEG<abs_d; d>
    // carry[0] = 2^-32 * ((d::DWordWL)[0] + abs_d[0])
    // carry[1] = 2^-32 * ((d::DWordWL)[1] + abs_d[1] + carry[0])
    // -------------------------------------------------------------------------

    // C5a: 1 - carry[0] = 1 - 2^-32*d[0] - 2^-16*d[1] - 2^-32*abs_d[0]
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::SIGN_D),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_1,
                },
            ]),
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::ABS_D_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::D_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_16,
                    column: cols::D_1,
                },
            ]),
        ],
    ));

    // C5b: 1 - carry[1] = 1 - 2^-32*d[2] - 2^-16*d[3] - 2^-32*abs_d[1]
    //                        - 2^-64*d[0] - 2^-48*d[1] - 2^-64*abs_d[0]
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Column(cols::SIGN_D),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_1,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_2,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_3,
                },
            ]),
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                // Current-level terms (carry[1] direct)
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::ABS_D_1,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_32,
                    column: cols::D_2,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_16,
                    column: cols::D_3,
                },
                // carry[0]-dependent terms (shifted by additional 2^-32)
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_64,
                    column: cols::ABS_D_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_64,
                    column: cols::D_0,
                },
                LinearTerm::ColumnUnsigned {
                    coefficient: NEG_INV_2_48,
                    column: cols::D_1,
                },
            ]),
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C8: ZERO[overflow; overflow_sum] multiplicity: μ_q + μ_r
    // overflow_sum = n[0]+n[1]+n[2]+(n[3]-2^15*sign_n)+(1-sign_n)+(65535-d[0])+...+(65535-d[3])
    // Each term ≥ 0, total ≤ 2^19. Sum is 0 iff overflow condition holds.
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::N_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::N_1,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::N_2,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::N_3,
                },
                LinearTerm::Column {
                    coefficient: -32769, // -(2^15 + 1) * sign_n
                    column: cols::SIGN_N,
                },
                LinearTerm::Constant(1 + 4 * 65535), // 262141
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::D_0,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::D_1,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::D_2,
                },
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::D_3,
                },
            ]),
            BusValue::Packed {
                start_column: cols::OVERFLOW,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C17: ZERO[div_by_zero; d[0]+d[1]+d[2]+d[3]]
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Zero,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        smallvec![
            BusValue::linear(vec![
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_0,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_1,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_2,
                },
                LinearTerm::Column {
                    coefficient: 1,
                    column: cols::D_3,
                },
            ]),
            BusValue::Packed {
                start_column: cols::DIV_BY_ZERO,
                packing: Packing::Direct,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C21: Receiver for quotient result
    // DVRM[q::DWordWL; n, d, signed, 0] with multiplicity -μ_q
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Dvrm,
        Multiplicity::Column(cols::MU_Q),
        smallvec![
            // n as DWordHL (4 halfwords → 2 words)
            BusValue::Packed {
                start_column: cols::N_0,
                packing: Packing::DWordHL,
            },
            // d as DWordHL
            BusValue::Packed {
                start_column: cols::D_0,
                packing: Packing::DWordHL,
            },
            // signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // q as DWordHL (result)
            BusValue::Packed {
                start_column: cols::Q_0,
                packing: Packing::DWordHL,
            },
            // muldiv_selector = 0 (quotient)
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C22: Receiver for remainder result
    // DVRM[r::DWordWL; n, d, signed, 1] with multiplicity -μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Dvrm,
        Multiplicity::Column(cols::MU_R),
        smallvec![
            // n as DWordHL
            BusValue::Packed {
                start_column: cols::N_0,
                packing: Packing::DWordHL,
            },
            // d as DWordHL
            BusValue::Packed {
                start_column: cols::D_0,
                packing: Packing::DWordHL,
            },
            // signed
            BusValue::Packed {
                start_column: cols::SIGNED,
                packing: Packing::Direct,
            },
            // r as DWordHL (result)
            BusValue::Packed {
                start_column: cols::R_0,
                packing: Packing::DWordHL,
            },
            // muldiv_selector = 1 (remainder)
            BusValue::constant(1),
        ],
    ));

    interactions
}

// =========================================================================
// Constraints
// =========================================================================

/// DVRM table constraint kinds.
#[derive(Debug, Clone, Copy)]
pub enum DvrmConstraintKind {
    /// DVRM-A3: signed * (1 - signed) = 0
    SignedIsBit,
    /// DVRM-C1: (r[0]+r[1]+r[2]+r[3]) * (sign_r - sign_n) = 0
    RemainderSignMatchesNumerator,
    /// DVRM-C4.i: (1-sign_r) * (abs_r[i] - (r::DWordWL)[i]) = 0
    AbsRFormula(usize),
    /// DVRM-C6.i: (1-sign_d) * (abs_d[i] - (d::DWordWL)[i]) = 0
    AbsDFormula(usize),
    /// DVRM-C7: signed * (1-overflow) - sign_q = 0
    SignQFormula,
    /// DVRM-C12.i: carry[i] * (1 - carry[i]) = 0 (virtual carries from n = n_sub_r + r)
    CarryIsBit(usize),
    /// DVRM-C15: sign_n_sub_r * (1-sign_n_sub_r) = 0
    SignNSubRIsBit,
    /// DVRM-C18b: (1-signed) * sign_n = 0
    UnsignedSignN,
    /// DVRM-C19b: (1-signed) * sign_r = 0
    UnsignedSignR,
    /// DVRM-C20b: (1-signed) * sign_d = 0
    UnsignedSignD,
    /// DVRM-C16.i: div_by_zero * (q[i] - 65535) = 0
    DivByZeroQ(usize),
}

/// DVRM table constraint.
pub struct DvrmConstraint {
    constraint_idx: usize,
    kind: DvrmConstraintKind,
}

impl DvrmConstraint {
    /// Create a new DVRM constraint.
    pub fn new(kind: DvrmConstraintKind, constraint_idx: usize) -> Self {
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
        let one = FieldElement::<F>::one();

        match self.kind {
            DvrmConstraintKind::SignedIsBit => {
                // signed * (1 - signed) = 0
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                &signed * (&one - &signed)
            }
            DvrmConstraintKind::RemainderSignMatchesNumerator => {
                // (r[0]+r[1]+r[2]+r[3]) * (sign_r - sign_n) = 0
                let r0 = step.get_main_evaluation_element(0, cols::R_0).clone();
                let r1 = step.get_main_evaluation_element(0, cols::R_1).clone();
                let r2 = step.get_main_evaluation_element(0, cols::R_2).clone();
                let r3 = step.get_main_evaluation_element(0, cols::R_3).clone();
                let sign_r = step.get_main_evaluation_element(0, cols::SIGN_R).clone();
                let sign_n = step.get_main_evaluation_element(0, cols::SIGN_N).clone();
                let r_sum = &r0 + &r1 + &r2 + &r3;
                &r_sum * (&sign_r - &sign_n)
            }
            DvrmConstraintKind::AbsRFormula(i) => {
                // (1-sign_r) * (abs_r[i] - (r::DWordWL)[i]) = 0
                let sign_r = step.get_main_evaluation_element(0, cols::SIGN_R).clone();
                let abs_r_col = if i == 0 { cols::ABS_R_0 } else { cols::ABS_R_1 };
                let abs_r = step.get_main_evaluation_element(0, abs_r_col).clone();

                // r::DWordWL[i]: lo32 = r[0] + r[1]*2^16, hi32 = r[2] + r[3]*2^16
                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let r_wl = if i == 0 {
                    let r0 = step.get_main_evaluation_element(0, cols::R_0).clone();
                    let r1 = step.get_main_evaluation_element(0, cols::R_1).clone();
                    &r0 + &r1 * &shift_16
                } else {
                    let r2 = step.get_main_evaluation_element(0, cols::R_2).clone();
                    let r3 = step.get_main_evaluation_element(0, cols::R_3).clone();
                    &r2 + &r3 * &shift_16
                };

                (&one - &sign_r) * (&abs_r - &r_wl)
            }
            DvrmConstraintKind::AbsDFormula(i) => {
                // (1-sign_d) * (abs_d[i] - (d::DWordWL)[i]) = 0
                let sign_d = step.get_main_evaluation_element(0, cols::SIGN_D).clone();
                let abs_d_col = if i == 0 { cols::ABS_D_0 } else { cols::ABS_D_1 };
                let abs_d = step.get_main_evaluation_element(0, abs_d_col).clone();

                let shift_16 = FieldElement::<F>::from(SHIFT_16);
                let d_wl = if i == 0 {
                    let d0 = step.get_main_evaluation_element(0, cols::D_0).clone();
                    let d1 = step.get_main_evaluation_element(0, cols::D_1).clone();
                    &d0 + &d1 * &shift_16
                } else {
                    let d2 = step.get_main_evaluation_element(0, cols::D_2).clone();
                    let d3 = step.get_main_evaluation_element(0, cols::D_3).clone();
                    &d2 + &d3 * &shift_16
                };

                (&one - &sign_d) * (&abs_d - &d_wl)
            }
            DvrmConstraintKind::SignQFormula => {
                // signed * (1-overflow) - sign_q = 0
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                let overflow = step.get_main_evaluation_element(0, cols::OVERFLOW).clone();
                let sign_q = step.get_main_evaluation_element(0, cols::SIGN_Q).clone();
                &signed * (&one - &overflow) - &sign_q
            }
            DvrmConstraintKind::CarryIsBit(i) => {
                // Virtual carry from n = n_sub_r + r
                // carry[i] * (1 - carry[i]) = 0
                let carry = self.compute_carry(i, step);
                &carry * (&one - &carry)
            }
            DvrmConstraintKind::SignNSubRIsBit => {
                // sign_n_sub_r * (1 - sign_n_sub_r) = 0
                let sign = step
                    .get_main_evaluation_element(0, cols::SIGN_N_SUB_R)
                    .clone();
                &sign * (&one - &sign)
            }
            DvrmConstraintKind::UnsignedSignN => {
                // (1-signed) * sign_n = 0
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                let sign_n = step.get_main_evaluation_element(0, cols::SIGN_N).clone();
                (&one - &signed) * &sign_n
            }
            DvrmConstraintKind::UnsignedSignR => {
                // (1-signed) * sign_r = 0
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                let sign_r = step.get_main_evaluation_element(0, cols::SIGN_R).clone();
                (&one - &signed) * &sign_r
            }
            DvrmConstraintKind::UnsignedSignD => {
                // (1-signed) * sign_d = 0
                let signed = step.get_main_evaluation_element(0, cols::SIGNED).clone();
                let sign_d = step.get_main_evaluation_element(0, cols::SIGN_D).clone();
                (&one - &signed) * &sign_d
            }
            DvrmConstraintKind::DivByZeroQ(i) => {
                // div_by_zero * (q[i] - 65535) = 0
                let dbz = step
                    .get_main_evaluation_element(0, cols::DIV_BY_ZERO)
                    .clone();
                let q_col = match i {
                    0 => cols::Q_0,
                    1 => cols::Q_1,
                    2 => cols::Q_2,
                    3 => cols::Q_3,
                    _ => unreachable!(),
                };
                let q = step.get_main_evaluation_element(0, q_col).clone();
                let fill = FieldElement::<F>::from(SIGN_FILL);
                &dbz * (&q - &fill)
            }
        }
    }

    /// Compute virtual carry[i] for the addition n_sub_r + r = n.
    ///
    /// The carries verify that n = n_sub_r + r by checking the carry chain.
    /// We use sign-extended versions for signed arithmetic.
    fn compute_carry<F, E>(&self, i: usize, step: &TableView<F, E>) -> FieldElement<F>
    where
        F: IsSubFieldOf<E>,
        E: IsField,
    {
        let shift_16 = FieldElement::<F>::from(SHIFT_16);
        let inv_2_32 = FieldElement::<F>::from(crate::constraints::templates::INV_SHIFT_32);
        let sign_fill = FieldElement::<F>::from(SIGN_FILL);

        // Get n, n_sub_r, r halfwords
        let n: [FieldElement<F>; 4] = [
            step.get_main_evaluation_element(0, cols::N_0).clone(),
            step.get_main_evaluation_element(0, cols::N_1).clone(),
            step.get_main_evaluation_element(0, cols::N_2).clone(),
            step.get_main_evaluation_element(0, cols::N_3).clone(),
        ];
        let nsr: [FieldElement<F>; 4] = [
            step.get_main_evaluation_element(0, cols::N_SUB_R_0).clone(),
            step.get_main_evaluation_element(0, cols::N_SUB_R_1).clone(),
            step.get_main_evaluation_element(0, cols::N_SUB_R_2).clone(),
            step.get_main_evaluation_element(0, cols::N_SUB_R_3).clone(),
        ];
        let r: [FieldElement<F>; 4] = [
            step.get_main_evaluation_element(0, cols::R_0).clone(),
            step.get_main_evaluation_element(0, cols::R_1).clone(),
            step.get_main_evaluation_element(0, cols::R_2).clone(),
            step.get_main_evaluation_element(0, cols::R_3).clone(),
        ];

        let sign_n = step.get_main_evaluation_element(0, cols::SIGN_N).clone();
        let sign_r = step.get_main_evaluation_element(0, cols::SIGN_R).clone();
        let sign_nsr = step
            .get_main_evaluation_element(0, cols::SIGN_N_SUB_R)
            .clone();

        // Build extended QuadWL values (4 words each)
        // extended_n[0] = n[0] + n[1]*2^16
        // extended_n[1] = n[2] + n[3]*2^16
        // extended_n[2] = sign_n * 0xFFFFFFFF
        // extended_n[3] = sign_n * 0xFFFFFFFF
        let ext_n = self.build_extended_quad(&n, &sign_n, &shift_16, &sign_fill);
        let ext_r = self.build_extended_quad(&r, &sign_r, &shift_16, &sign_fill);
        let ext_nsr = self.build_extended_quad(&nsr, &sign_nsr, &shift_16, &sign_fill);

        // carry[0] = (ext_nsr[0] + ext_r[0] - ext_n[0]) / 2^32
        // carry[i] = (ext_nsr[i] + ext_r[i] + carry[i-1] - ext_n[i]) / 2^32
        if i == 0 {
            (&ext_nsr[0] + &ext_r[0] - &ext_n[0]) * &inv_2_32
        } else {
            let prev_carry = self.compute_carry(i - 1, step);
            (&ext_nsr[i] + &ext_r[i] + &prev_carry - &ext_n[i]) * &inv_2_32
        }
    }

    /// Build sign-extended QuadWL representation.
    fn build_extended_quad<F: IsSubFieldOf<E>, E: IsField>(
        &self,
        halfwords: &[FieldElement<F>; 4],
        sign: &FieldElement<F>,
        shift_16: &FieldElement<F>,
        sign_fill: &FieldElement<F>,
    ) -> [FieldElement<F>; 4] {
        let ext_word = sign * sign_fill + sign * sign_fill * shift_16;
        [
            &halfwords[0] + &halfwords[1] * shift_16,
            &halfwords[2] + &halfwords[3] * shift_16,
            ext_word.clone(),
            ext_word,
        ]
    }
}

impl TransitionConstraint<GoldilocksField, GoldilocksExtension> for DvrmConstraint {
    fn degree(&self) -> usize {
        match self.kind {
            DvrmConstraintKind::SignedIsBit => 2,
            DvrmConstraintKind::RemainderSignMatchesNumerator => 2,
            DvrmConstraintKind::AbsRFormula(_) => 2,
            DvrmConstraintKind::AbsDFormula(_) => 2,
            DvrmConstraintKind::SignQFormula => 2,
            DvrmConstraintKind::CarryIsBit(_) => 2,
            DvrmConstraintKind::SignNSubRIsBit => 2,
            DvrmConstraintKind::UnsignedSignN => 2,
            DvrmConstraintKind::UnsignedSignR => 2,
            DvrmConstraintKind::UnsignedSignD => 2,
            DvrmConstraintKind::DivByZeroQ(_) => 2,
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

/// Creates all constraints for the DVRM table.
///
/// Returns: (constraints, next_constraint_idx)
pub fn dvrm_constraints(constraint_idx_start: usize) -> (Vec<DvrmConstraint>, usize) {
    let mut idx = constraint_idx_start;
    let mut constraints = Vec::new();

    // DVRM-A3: signed is bit
    constraints.push(DvrmConstraint::new(DvrmConstraintKind::SignedIsBit, idx));
    idx += 1;

    // DVRM-C1: remainder sign matches numerator sign
    constraints.push(DvrmConstraint::new(
        DvrmConstraintKind::RemainderSignMatchesNumerator,
        idx,
    ));
    idx += 1;

    // DVRM-C4: abs_r formula (×2)
    for i in 0..2 {
        constraints.push(DvrmConstraint::new(DvrmConstraintKind::AbsRFormula(i), idx));
        idx += 1;
    }

    // DVRM-C6: abs_d formula (×2)
    for i in 0..2 {
        constraints.push(DvrmConstraint::new(DvrmConstraintKind::AbsDFormula(i), idx));
        idx += 1;
    }

    // DVRM-C7: sign_q formula
    constraints.push(DvrmConstraint::new(DvrmConstraintKind::SignQFormula, idx));
    idx += 1;

    // DVRM-C12.i: carry is bit (×4)
    for i in 0..4 {
        constraints.push(DvrmConstraint::new(DvrmConstraintKind::CarryIsBit(i), idx));
        idx += 1;
    }

    // DVRM-C15: sign_n_sub_r is bit
    constraints.push(DvrmConstraint::new(DvrmConstraintKind::SignNSubRIsBit, idx));
    idx += 1;

    // DVRM-C18b: unsigned sign_n = 0
    constraints.push(DvrmConstraint::new(DvrmConstraintKind::UnsignedSignN, idx));
    idx += 1;

    // DVRM-C19b: unsigned sign_r = 0
    constraints.push(DvrmConstraint::new(DvrmConstraintKind::UnsignedSignR, idx));
    idx += 1;

    // DVRM-C20b: unsigned sign_d = 0
    constraints.push(DvrmConstraint::new(DvrmConstraintKind::UnsignedSignD, idx));
    idx += 1;

    // DVRM-C16.i: div_by_zero implies q = all 1s (×4)
    for i in 0..4 {
        constraints.push(DvrmConstraint::new(DvrmConstraintKind::DivByZeroQ(i), idx));
        idx += 1;
    }

    (constraints, idx)
}
