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
//! - Sender: IS_HALF (×20: n, d, r, n_sub_r, q)
//! - Sender: MSB16 (×3 for sign extraction: n, d, r)
//! - Sender: ALU (×3, on the unified bus: ×1 LT-flavored for `|r| < |d|`,
//!   ×2 MUL-flavored for `n - r = d * q` lo/hi)
//! - Sender: ZERO (×5 for div_by_zero, overflow, NEG template)
//! - Receiver: ALU (×2, on the unified bus, for quotient and remainder results)

use stark::lookup::{BusInteraction, BusValue, LinearTerm, Multiplicity, Packing};
use stark::trace::TraceTable;

use std::collections::HashMap;

use super::types::{
    BusId, FE, GoldilocksExtension, GoldilocksField, NEG_INV_2_16, NEG_INV_2_32, NEG_INV_2_48,
    NEG_INV_2_64, SHIFT_16, VmTable, alu_op,
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
    let mut trace = TraceTable::new_main(
        vec![FE::zero(); num_rows * cols::NUM_COLUMNS],
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for (row_idx, (op, multiplicities)) in unique_ops.iter().enumerate() {
        let q = op.compute_quotient();
        let r = op.compute_remainder();
        let n_sub_r = op.n_sub_r();
        let abs_r = op.abs_r();
        let abs_d = op.abs_d();

        // Fill n as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::N_0, op.n);

        // Fill d as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::D_0, op.d);

        table.set_bool(row_idx, cols::SIGNED, op.signed);

        // Fill q as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::Q_0, q);

        // Fill r as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::R_0, r);

        // Fill auxiliary columns
        table.set_bool(row_idx, cols::DIV_BY_ZERO, op.is_div_by_zero());
        table.set_bool(row_idx, cols::OVERFLOW, op.is_overflow());

        table.set_dword_wl(row_idx, cols::ABS_R_0, abs_r);
        table.set_dword_wl(row_idx, cols::ABS_D_0, abs_d);

        // Fill n_sub_r as DWordHL (4 halfwords)
        table.set_dword_hl(row_idx, cols::N_SUB_R_0, n_sub_r);

        table.set_bool(row_idx, cols::SIGN_N_SUB_R, op.sign_n_sub_r());
        table.set_bool(row_idx, cols::SIGN_N, op.sign_n());
        table.set_bool(row_idx, cols::SIGN_D, op.sign_d());
        table.set_bool(row_idx, cols::SIGN_Q, op.sign_q());
        table.set_bool(row_idx, cols::SIGN_R, op.sign_r());

        // Multiplicities
        table.set_u64(row_idx, cols::MU_Q, multiplicities.mu_q);
        table.set_u64(row_idx, cols::MU_R, multiplicities.mu_r);
    }

    trace
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

    // -------------------------------------------------------------------------
    // DVRM-A1.i: IS_HALF[n[i]] (×4) and DVRM-A2.i: IS_HALF[d[i]] (×4),
    // multiplicity: μ_q + μ_r.
    // The bus binds only the packed 32-bit words (DWordHL/DWordBL emit two
    // words, not the four halves), so without these the input halves are free:
    // a prover could supply non-canonical halves that re-pack to the same word
    // yet sum to 0 in the field, forging div_by_zero (DVRM-C17 keys on the
    // half-sum) for a nonzero denominator. Range-checking each half closes that.
    // -------------------------------------------------------------------------
    for col in [
        cols::N_0,
        cols::N_1,
        cols::N_2,
        cols::N_3,
        cols::D_0,
        cols::D_1,
        cols::D_2,
        cols::D_3,
    ] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_Q, cols::MU_R),
            vec![BusValue::Packed {
                start_column: col,
                packing: Packing::Direct,
            }],
        ));
    }

    // -------------------------------------------------------------------------
    // DVRM-C13.i: IS_HALF[r[i]] (×4), multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    for col in [cols::R_0, cols::R_1, cols::R_2, cols::R_3] {
        interactions.push(BusInteraction::sender(
            BusId::IsHalfword,
            Multiplicity::Sum(cols::MU_Q, cols::MU_R),
            vec![BusValue::Packed {
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
            vec![BusValue::Packed {
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
            vec![BusValue::Packed {
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
        vec![
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
        vec![
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
        vec![
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
    // DVRM-C2: ALU[abs_r, abs_d, opsel(LT), 1-div_by_zero, 0]
    // Verify |r| < |d| when d != 0 (the ALU output is 1 iff abs_r < abs_d).
    // This lookup is dispatched on the unified ALU bus with signed=0/invert=0
    // (there is no dedicated `Lt` bus).
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        vec![
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
            // flags = opsel(LT) (signed=0, invert=0)
            BusValue::constant(alu_op::LT as u64),
            // out_lo = 1 - div_by_zero (LT result fits in the low word)
            BusValue::linear(vec![
                LinearTerm::Constant(1),
                LinearTerm::Column {
                    coefficient: -1,
                    column: cols::DIV_BY_ZERO,
                },
            ]),
            // out_hi = 0
            BusValue::constant(0),
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C9: ALU[d, q, opsel(MUL)+32*signed+64*sign_q, n_sub_r]
    // Verify n - r = d * q (lower 64 bits). The lookup is dispatched on the
    // unified ALU bus with the lo selector (flags `+0`); there is no dedicated
    // `Mul` bus.
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    let mul_flags = |hi: i64| {
        BusValue::linear(vec![
            LinearTerm::Constant(alu_op::MUL as i64 + hi),
            LinearTerm::Column {
                coefficient: 32,
                column: cols::SIGNED,
            },
            LinearTerm::Column {
                coefficient: 64,
                column: cols::SIGN_Q,
            },
        ])
    };
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        vec![
            // lhs = d as DWordHL
            BusValue::Packed {
                start_column: cols::D_0,
                packing: Packing::DWordHL,
            },
            // rhs = q as DWordHL
            BusValue::Packed {
                start_column: cols::Q_0,
                packing: Packing::DWordHL,
            },
            // flags = opsel(MUL) + 32*signed + 64*sign_q (lo half)
            mul_flags(0),
            // result = n_sub_r as DWordHL (lower 64 bits of d*q)
            BusValue::Packed {
                start_column: cols::N_SUB_R_0,
                packing: Packing::DWordHL,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C10: ALU[d, q, opsel(MUL)+32*signed+64*sign_q+128, sign_ext(n_sub_r)]
    // Verify upper 64 bits of d * q = sign extension of n_sub_r.
    // Dispatched on the unified ALU bus with the hi selector (flags `+128`).
    // multiplicity: μ_q + μ_r
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::sender(
        BusId::Alu,
        Multiplicity::Sum(cols::MU_Q, cols::MU_R),
        vec![
            // lhs = d as DWordHL
            BusValue::Packed {
                start_column: cols::D_0,
                packing: Packing::DWordHL,
            },
            // rhs = q as DWordHL
            BusValue::Packed {
                start_column: cols::Q_0,
                packing: Packing::DWordHL,
            },
            // flags = opsel(MUL) + 32*signed + 64*sign_q + 128 (hi half)
            mul_flags(128),
            // result: sign extension of n_sub_r.
            // The MUL Alu receiver consumes the result as `Packed{HI_0, DWordHL}`
            // → 2 elements `[HI_0 + 2^16*HI_1, HI_2 + 2^16*HI_3]`. Both equal
            // SIGN_N_SUB_R * 0xFFFFFFFF (each halfword is SIGN_FILL when negative).
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: (SIGN_FILL + SIGN_FILL * SHIFT_16) as i64,
                column: cols::SIGN_N_SUB_R,
            }]),
            BusValue::linear(vec![LinearTerm::Column {
                coefficient: (SIGN_FILL + SIGN_FILL * SHIFT_16) as i64,
                column: cols::SIGN_N_SUB_R,
            }]),
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
        vec![
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
        vec![
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
        vec![
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
        vec![
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
        vec![
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
        vec![
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
    // DVRM-C21: Quotient result on the unified ALU bus.
    // ALU[q::DWordWL; n, d, opsel(DIVREM) + 32*signed] | μ_q  (muldiv bit 7 = 0)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU_Q),
        vec![
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
            // flags = DIVREM + 32*signed (quotient: muldiv selector = 0)
            BusValue::linear(vec![
                LinearTerm::Constant(alu_op::DIVREM as i64),
                LinearTerm::Column {
                    coefficient: 32,
                    column: cols::SIGNED,
                },
            ]),
            // q as DWordHL (result)
            BusValue::Packed {
                start_column: cols::Q_0,
                packing: Packing::DWordHL,
            },
        ],
    ));

    // -------------------------------------------------------------------------
    // DVRM-C22: Remainder result on the unified ALU bus.
    // ALU[r::DWordWL; n, d, opsel(DIVREM) + 32*signed + 128] | μ_r  (muldiv bit 7 = 1)
    // -------------------------------------------------------------------------
    interactions.push(BusInteraction::receiver(
        BusId::Alu,
        Multiplicity::Column(cols::MU_R),
        vec![
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
            // flags = DIVREM + 32*signed + 128 (remainder: muldiv selector = 1)
            BusValue::linear(vec![
                LinearTerm::Constant(alu_op::DIVREM as i64 + 128),
                LinearTerm::Column {
                    coefficient: 32,
                    column: cols::SIGNED,
                },
            ]),
            // r as DWordHL (result)
            BusValue::Packed {
                start_column: cols::R_0,
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
// folder, the verifier folder and IR capture. Constraint indices 0..19.

use stark::constraints::builder::{ConstraintBuilder, ConstraintSet};

/// DVRM table constraints as a single-source [`ConstraintSet`]. No column
/// configuration is needed (the DVRM layout is fixed via `cols`).
pub struct DvrmConstraints;

impl DvrmConstraints {
    /// Sign-extended QuadWL word `k` (0..4) of a halfword group:
    /// `[hw0 + hw1·2^16, hw2 + hw3·2^16, ext, ext]`, where
    /// `ext = sign·SIGN_FILL + sign·SIGN_FILL·2^16`.
    fn ext_quad<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        hw: [usize; 4],
        sign_col: usize,
        k: usize,
    ) -> B::Expr {
        let shift_16 = b.const_base(SHIFT_16);
        match k {
            0 => {
                let hw0 = b.main(0, hw[0]);
                let hw1 = b.main(0, hw[1]);
                hw0 + hw1 * shift_16
            }
            1 => {
                let hw2 = b.main(0, hw[2]);
                let hw3 = b.main(0, hw[3]);
                hw2 + hw3 * shift_16
            }
            _ => {
                // ext = sign * SIGN_FILL + sign * SIGN_FILL * 2^16
                let sign = b.main(0, sign_col);
                let sign_fill = b.const_base(SIGN_FILL);
                let sign_fill2 = b.const_base(SIGN_FILL);
                let shift_16b = b.const_base(SHIFT_16);
                sign.clone() * sign_fill + sign * sign_fill2 * shift_16b
            }
        }
    }

    /// Virtual carry[i] for `n = n_sub_r + r` (extended QuadWL, recursive chain).
    fn carry<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        i: usize,
    ) -> B::Expr {
        const N: [usize; 4] = [cols::N_0, cols::N_1, cols::N_2, cols::N_3];
        const NSR: [usize; 4] = [
            cols::N_SUB_R_0,
            cols::N_SUB_R_1,
            cols::N_SUB_R_2,
            cols::N_SUB_R_3,
        ];
        const R: [usize; 4] = [cols::R_0, cols::R_1, cols::R_2, cols::R_3];

        let ext_n = Self::ext_quad(b, N, cols::SIGN_N, i);
        let ext_r = Self::ext_quad(b, R, cols::SIGN_R, i);
        let ext_nsr = Self::ext_quad(b, NSR, cols::SIGN_N_SUB_R, i);
        let inv_2_32 = b.const_base(crate::constraints::templates::INV_SHIFT_32);

        if i == 0 {
            // carry[0] = (ext_nsr[0] + ext_r[0] - ext_n[0]) / 2^32
            (ext_nsr + ext_r - ext_n) * inv_2_32
        } else {
            // carry[i] = (ext_nsr[i] + ext_r[i] + carry[i-1] - ext_n[i]) / 2^32
            let prev = Self::carry(b, i - 1);
            (ext_nsr + ext_r + prev - ext_n) * inv_2_32
        }
    }

    /// `r::DWordWL[i]` (i = 0 → lo32, else hi32); used generically for r or d
    /// halfword groups.
    fn dword_wl<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(
        b: &B,
        lo: usize,
        hi: usize,
    ) -> B::Expr {
        let shift_16 = b.const_base(SHIFT_16);
        let a = b.main(0, lo);
        let c = b.main(0, hi);
        a + c * shift_16
    }
}

impl ConstraintSet<GoldilocksField, GoldilocksExtension> for DvrmConstraints {
    fn eval<B: ConstraintBuilder<GoldilocksField, GoldilocksExtension>>(&self, b: &mut B) {
        // idx 0: SignedIsBit — signed * (1 - signed)
        let signed = b.main(0, cols::SIGNED);
        let one = b.one();
        b.emit_base(0, signed.clone() * (one - signed));

        // idx 1: RemainderSignMatchesNumerator —
        // (r[0]+r[1]+r[2]+r[3]) * (sign_r - sign_n)
        let r0 = b.main(0, cols::R_0);
        let r1 = b.main(0, cols::R_1);
        let r2 = b.main(0, cols::R_2);
        let r3 = b.main(0, cols::R_3);
        let sign_r = b.main(0, cols::SIGN_R);
        let sign_n = b.main(0, cols::SIGN_N);
        let r_sum = r0 + r1 + r2 + r3;
        b.emit_base(1, r_sum * (sign_r - sign_n));

        // idx 2,3: AbsRFormula(0,1) — (1-sign_r) * (abs_r[i] - r::DWordWL[i])
        for (off, (abs_col, lo, hi)) in [
            (cols::ABS_R_0, cols::R_0, cols::R_1),
            (cols::ABS_R_1, cols::R_2, cols::R_3),
        ]
        .into_iter()
        .enumerate()
        {
            let sign_r = b.main(0, cols::SIGN_R);
            let one = b.one();
            let abs_r = b.main(0, abs_col);
            let r_wl = Self::dword_wl(b, lo, hi);
            b.emit_base(2 + off, (one - sign_r) * (abs_r - r_wl));
        }

        // idx 4,5: AbsDFormula(0,1) — (1-sign_d) * (abs_d[i] - d::DWordWL[i])
        for (off, (abs_col, lo, hi)) in [
            (cols::ABS_D_0, cols::D_0, cols::D_1),
            (cols::ABS_D_1, cols::D_2, cols::D_3),
        ]
        .into_iter()
        .enumerate()
        {
            let sign_d = b.main(0, cols::SIGN_D);
            let one = b.one();
            let abs_d = b.main(0, abs_col);
            let d_wl = Self::dword_wl(b, lo, hi);
            b.emit_base(4 + off, (one - sign_d) * (abs_d - d_wl));
        }

        // idx 6: SignQFormula — signed * (1-overflow) - sign_q
        let signed = b.main(0, cols::SIGNED);
        let overflow = b.main(0, cols::OVERFLOW);
        let sign_q = b.main(0, cols::SIGN_Q);
        let one = b.one();
        b.emit_base(6, signed * (one - overflow) - sign_q);

        // idx 7..11: CarryIsBit(0..4) — carry[i] * (1 - carry[i])
        for i in 0..4 {
            let carry = Self::carry(b, i);
            let one = b.one();
            b.emit_base(7 + i, carry.clone() * (one - carry));
        }

        // idx 11: SignNSubRIsBit — sign_n_sub_r * (1 - sign_n_sub_r)
        let sign = b.main(0, cols::SIGN_N_SUB_R);
        let one = b.one();
        b.emit_base(11, sign.clone() * (one - sign));

        // idx 12: UnsignedSignN — (1-signed) * sign_n
        let signed = b.main(0, cols::SIGNED);
        let sign_n = b.main(0, cols::SIGN_N);
        let one = b.one();
        b.emit_base(12, (one - signed) * sign_n);

        // idx 13: UnsignedSignR — (1-signed) * sign_r
        let signed = b.main(0, cols::SIGNED);
        let sign_r = b.main(0, cols::SIGN_R);
        let one = b.one();
        b.emit_base(13, (one - signed) * sign_r);

        // idx 14: UnsignedSignD — (1-signed) * sign_d
        let signed = b.main(0, cols::SIGNED);
        let sign_d = b.main(0, cols::SIGN_D);
        let one = b.one();
        b.emit_base(14, (one - signed) * sign_d);

        // idx 15..19: DivByZeroQ(0..4) — div_by_zero * (q[i] - 65535)
        let q_cols = [cols::Q_0, cols::Q_1, cols::Q_2, cols::Q_3];
        for (i, &q_col) in q_cols.iter().enumerate() {
            let dbz = b.main(0, cols::DIV_BY_ZERO);
            let q = b.main(0, q_col);
            let fill = b.const_base(SIGN_FILL);
            b.emit_base(15 + i, dbz * (q - fill));
        }
    }
}
