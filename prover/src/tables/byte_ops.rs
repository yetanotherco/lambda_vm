//! BYTE_OPS precomputed lookup table for byte-pair operations.
//!
//! Holds every `(X, Y) ∈ [0, 256)²` across four `OP_ID` slices and serves
//! the byte-pair lookups that BITWISE used to multiplex into its 2²⁰ rows.
//!
//! ## Layout (4 slices × 256² = 2¹⁸ rows, power of two)
//!
//! | Slice | OP_ID | Role                           | RESULT col holds |
//! |-------|------:|--------------------------------|------------------|
//! | 0     | 0     | non-bitwise (MSB8/MSB16/range) | 0 (unused)       |
//! | 1     | 1     | AND                            | X & Y            |
//! | 2     | 2     | OR                             | X \| Y           |
//! | 3     | 4     | XOR                            | X ^ Y            |
//!
//! `OP_ID = 1 / 2 / 4` is the disjoint-bit encoding `AND + 2*OR + 4*XOR`.
//! Senders emit `(op_id, X, Y, RESULT)` against `BusId::Bitwise`; the row
//! at the matching slice has the right RESULT precomputed.
//!
//! Non-bitwise multiplicities (MU_IS_BYTE, MU_IS_HALF, MU_MSB8, MU_MSB16)
//! live exclusively on the slice-0 rows so each (X, Y) has a single
//! canonical home; the senders for those buses don't include an op_id.

use std::sync::OnceLock;

use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::bitwise::{BitwiseOperation, BitwiseOperationType};
use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

// =========================================================================
// Column indices for BYTE_OPS table
// =========================================================================

pub mod cols {
    /// X: Byte input (0-255)
    pub const X: usize = 0;
    /// Y: Byte input (0-255)
    pub const Y: usize = 1;
    /// OP_ID: ∈ {0, 1, 2, 4}. 0 = non-bitwise slice, 1/2/4 = AND/OR/XOR.
    pub const OP_ID: usize = 2;
    /// RESULT: precomputed result for the slice's op (0 on slice 0).
    pub const RESULT: usize = 3;
    /// MSB of byte X: (X >> 7) & 1
    pub const MSB8: usize = 4;
    /// MSB of halfword (X + 256*Y): ((X + 256*Y) >> 15) & 1
    pub const MSB16: usize = 5;

    /// Multiplicity for the unified Bitwise lookup (AND/OR/XOR).
    pub const MU_BITWISE: usize = 6;
    /// Multiplicity for IS_BYTE lookups (only fired on slice 0).
    pub const MU_IS_BYTE: usize = 7;
    /// Multiplicity for IS_HALF lookups (only fired on slice 0).
    pub const MU_IS_HALF: usize = 8;
    /// Multiplicity for MSB8 lookups (only fired on slice 0).
    pub const MU_MSB8: usize = 9;
    /// Multiplicity for MSB16 lookups (only fired on slice 0).
    pub const MU_MSB16: usize = 10;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 11;
}

/// 256² × 4 slices = 262,144 rows (2¹⁸).
pub const NUM_ROWS: usize = 256 * 256 * 4;

/// Number of precomputed (non-multiplicity) columns.
pub const NUM_PRECOMPUTED_COLS: usize = 6;

/// Number of slices in the table (op_id discriminator).
pub const NUM_SLICES: usize = 4;

/// Maps a slice index `s ∈ [0, 4)` to its `OP_ID` value.
const fn slice_to_op_id(s: usize) -> u32 {
    match s {
        0 => 0,
        1 => 1,
        2 => 2,
        3 => 4,
        _ => 0,
    }
}

/// Inverse: given an op_id ∈ {1, 2, 4}, return its slice index.
const fn op_id_to_slice(op_id: u8) -> usize {
    match op_id {
        1 => 1,
        2 => 2,
        4 => 3,
        _ => 0,
    }
}

// =========================================================================
// Compile-time row generation
// =========================================================================

/// Generate one row of the byte_ops table.
///
/// Index encoding: `index = x + 256 * y + 65536 * slice` where
/// `x, y ∈ [0, 255]` and `slice ∈ [0, 4)`.
#[inline]
pub const fn generate_byte_ops_row(index: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    let x = (index & 0xFF) as u32;
    let y = ((index >> 8) & 0xFF) as u32;
    let slice = (index >> 16) & 0x3;
    let op_id = slice_to_op_id(slice);

    let result = match op_id {
        1 => x & y,
        2 => x | y,
        4 => x ^ y,
        _ => 0,
    };

    let msb8 = (x >> 7) & 1;
    let halfword = x + y * 256;
    let msb16 = (halfword >> 15) & 1;

    [
        x as u64,
        y as u64,
        op_id as u64,
        result as u64,
        msb8 as u64,
        msb16 as u64,
    ]
}

pub const fn is_preprocessed() -> bool {
    true
}

// =========================================================================
// Preprocessed commitment (computed once, cached)
// =========================================================================

static BYTE_OPS_COMMITMENT: OnceLock<Commitment> = OnceLock::new();

/// Computes the Merkle commitment over the precomputed byte_ops columns.
///
/// Mirrors [`bitwise::compute_preprocessed_commitment`] — the LDE-rooted
/// commitment is required so FRI queries at any blow-up index can be opened
/// against this preprocessed table.
fn compute_preprocessed_commitment(options: &ProofOptions) -> Commitment {
    #[cfg(feature = "parallel")]
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .into_par_iter()
        .map(|col_idx| {
            (0..NUM_ROWS)
                .map(|idx| {
                    let row = generate_byte_ops_row(idx);
                    FE::from(row[col_idx])
                })
                .collect()
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let columns: Vec<Vec<FE>> = {
        let mut cols: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
            .map(|_| Vec::with_capacity(NUM_ROWS))
            .collect();
        for idx in 0..NUM_ROWS {
            let row = generate_byte_ops_row(idx);
            for (col_idx, &value) in row.iter().enumerate() {
                cols[col_idx].push(FE::from(value));
            }
        }
        cols
    };

    #[cfg(feature = "parallel")]
    let polys: Vec<Polynomial<FE>> = columns
        .par_iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for byte_ops column")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for byte_ops column")
        })
        .collect();

    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    #[cfg(feature = "parallel")]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .par_iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for byte_ops polynomial")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for byte_ops polynomial")
        })
        .collect();

    #[cfg(feature = "parallel")]
    lde_columns.par_iter_mut().for_each(|col| {
        in_place_bit_reverse_permute(col);
    });

    #[cfg(not(feature = "parallel"))]
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    let lde_rows = columns2rows(lde_columns);

    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for byte_ops LDE");

    tree.root
}

#[inline]
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    *BYTE_OPS_COMMITMENT.get_or_init(|| compute_preprocessed_commitment(options))
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generate the precomputed BYTE_OPS trace table (multiplicities zeroed).
pub fn generate_byte_ops_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut data = vec![FE::zero(); NUM_ROWS * cols::NUM_COLUMNS];

    for slice in 0..NUM_SLICES {
        let op_id = slice_to_op_id(slice);
        for x in 0u32..256 {
            for y in 0u32..256 {
                let row_idx = (x as usize) + (y as usize) * 256 + slice * 65536;
                let base = row_idx * cols::NUM_COLUMNS;

                data[base + cols::X] = FE::from(x as u64);
                data[base + cols::Y] = FE::from(y as u64);
                data[base + cols::OP_ID] = FE::from(op_id as u64);

                let result = match op_id {
                    1 => x & y,
                    2 => x | y,
                    4 => x ^ y,
                    _ => 0,
                };
                data[base + cols::RESULT] = FE::from(result as u64);

                let msb8 = (x >> 7) & 1;
                let halfword = x + y * 256;
                let msb16 = (halfword >> 15) & 1;
                data[base + cols::MSB8] = FE::from(msb8 as u64);
                data[base + cols::MSB16] = FE::from(msb16 as u64);

                // Multiplicity columns are zero-initialized by `vec!` above.
            }
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
}

/// Slice-0 row index for a given (x, y), used by non-bitwise senders that
/// don't include an op_id.
#[inline]
pub fn row_index(x: u8, y: u8) -> usize {
    (x as usize) + (y as usize) * 256
}

/// Row index for a bitwise op (AND/OR/XOR) at the matching op_id slice.
#[inline]
pub fn bitwise_row_index(x: u8, y: u8, op_id: u8) -> usize {
    let slice = op_id_to_slice(op_id);
    (x as usize) + (y as usize) * 256 + slice * 65536
}

/// Apply lookups to multiplicity columns.
///
/// Routes the byte-pair operations into their canonical rows:
/// - AndByte/OrByte/XorByte → MU_BITWISE on the matching op_id slice
/// - Msb8/Msb16/IsByte/IsHalf → MU_* on slice 0 (single canonical home per (X, Y))
/// - 20-bit ops (Zero/IsB20/Hwsl) stay on the BITWISE table.
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ops: &[BitwiseOperation],
) {
    for op in ops {
        let (mu_col, row) = match op.lookup_type {
            BitwiseOperationType::AndByte => (cols::MU_BITWISE, bitwise_row_index(op.x, op.y, 1)),
            BitwiseOperationType::OrByte => (cols::MU_BITWISE, bitwise_row_index(op.x, op.y, 2)),
            BitwiseOperationType::XorByte => (cols::MU_BITWISE, bitwise_row_index(op.x, op.y, 4)),
            BitwiseOperationType::Msb8 => (cols::MU_MSB8, row_index(op.x, op.y)),
            BitwiseOperationType::Msb16 => (cols::MU_MSB16, row_index(op.x, op.y)),
            BitwiseOperationType::IsByte => (cols::MU_IS_BYTE, row_index(op.x, op.y)),
            BitwiseOperationType::IsHalf => (cols::MU_IS_HALF, row_index(op.x, op.y)),
            BitwiseOperationType::Zero
            | BitwiseOperationType::IsB20
            | BitwiseOperationType::Hwsl => continue,
        };
        let current = trace.main_table.get_row(row)[mu_col];
        trace.set_main(row, mu_col, current + FE::one());
    }
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Receivers for byte-pair lookups. The unified Bitwise receiver matches
/// `(op_id, X, Y, RESULT)`; non-bitwise receivers stay separate but only
/// receive on slice-0 rows where their multiplicity is non-zero.
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
        // Bitwise[op_id, X, Y, RESULT]
        BusInteraction::receiver(
            BusId::Bitwise,
            Multiplicity::Column(cols::MU_BITWISE),
            vec![
                BusValue::Packed {
                    start_column: cols::OP_ID,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::RESULT,
                    packing: Packing::Direct,
                },
            ],
        ),
        // MSB8[X] -> MSB8
        BusInteraction::receiver(
            BusId::Msb8,
            Multiplicity::Column(cols::MU_MSB8),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::MSB8,
                    packing: Packing::Direct,
                },
            ],
        ),
        // MSB16[X + 256*Y] -> MSB16
        BusInteraction::receiver(
            BusId::Msb16,
            Multiplicity::Column(cols::MU_MSB16),
            vec![
                BusValue::linear(vec![
                    stark::lookup::LinearTerm::Column {
                        coefficient: 1,
                        column: cols::X,
                    },
                    stark::lookup::LinearTerm::Column {
                        coefficient: 256,
                        column: cols::Y,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::MSB16,
                    packing: Packing::Direct,
                },
            ],
        ),
        // IS_BYTE[X, Y] - range check two byte values, no output.
        BusInteraction::receiver(
            BusId::IsByte,
            Multiplicity::Column(cols::MU_IS_BYTE),
            vec![
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
            ],
        ),
        // IS_HALF[X + 256*Y] - range check for halfword
        BusInteraction::receiver(
            BusId::IsHalfword,
            Multiplicity::Column(cols::MU_IS_HALF),
            vec![BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::X,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 256,
                    column: cols::Y,
                },
            ])],
        ),
    ]
}
