//! BITWISE precomputed lookup table.
//!
//! This table provides byte/range lookup types used by other tables:
//!
//! ## Range Checks
//! - `ARE_BYTES[X, Y]` - X and Y are valid bytes [0, 256). Spec template
//!   `IS_BYTE<X>` is implemented by sending `ARE_BYTES[X, 0]`.
//! - `IS_HALF[X]` - X is a valid halfword [0, 2^16)
//! - `IS_B20[X]` - X is a valid 20-bit value [0, 2^20)
//!
//! ## Bitwise Operations
//! - `BYTE_ALU[opsel, X, Y] -> out` for byte AND/OR/XOR
//! - `MSB8[X]` -> most significant bit of byte
//! - `MSB16[X]` -> most significant bit of halfword
//! - `ZERO[X]` -> whether X is zero
//!
//! ## Shift Helpers
//! - `HWSL[X, Z]` -> [(X << Z) mod 2^16, X >> (16 - Z)]
//!
//! ## Table Structure
//!
//! The table has 2^20 rows (256 * 256 * 16) with precomputed values.
//! Each row is indexed by (X: Byte, Y: Byte, Z: B4).
//!
//! All lookups are provided as receivers with negative multiplicity,
//! meaning other tables send to this table.

use math::polynomial::Polynomial;
use stark::commitment::{ROWS_PER_LEAF, commit_bit_reversed};
use stark::config::Commitment;
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::proof::options::ProofOptions;
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::TraceTable;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField, VmTable, alu_op};

// =========================================================================
// Column indices for BITWISE table
// =========================================================================

/// Input columns (precomputed)
pub mod cols {
    /// X: Byte input (0-255)
    pub const X: usize = 0;
    /// Y: Byte input (0-255)
    pub const Y: usize = 1;
    /// Z: 4-bit input (0-15) for shift amount
    pub const Z: usize = 2;

    /// AND result: X & Y
    pub const AND: usize = 3;
    /// OR result: X | Y
    pub const OR: usize = 4;
    /// XOR result: X ^ Y
    pub const XOR: usize = 5;
    /// MSB of byte X: (X >> 7) & 1
    pub const MSB8: usize = 6;
    /// MSB of halfword (X + 256*Y): ((X + 256*Y) >> 15) & 1
    pub const MSB16: usize = 7;
    /// Zero check: (X == 0 && Y == 0) ? 1 : 0
    pub const ZERO: usize = 8;
    /// Shift left result: ((X + 256*Y) << Z) & 0xFFFF
    pub const SLL: usize = 9;
    /// Shift left carry: (X + 256*Y) >> (16 - Z)
    pub const SLLC: usize = 10;

    // Multiplicity columns for each lookup type
    /// Multiplicity for MSB8 lookups
    pub const MU_MSB8: usize = 11;
    /// Multiplicity for MSB16 lookups
    pub const MU_MSB16: usize = 12;
    /// Multiplicity for ZERO lookups
    pub const MU_ZERO: usize = 13;
    /// Multiplicity for ARE_BYTES lookups. Each lookup checks X and Y; pass Y=0
    /// for a single-byte range check (spec template `IS_BYTE<X>`).
    pub const MU_ARE_BYTES: usize = 14;
    /// Multiplicity for IS_HALF lookups
    pub const MU_IS_HALF: usize = 15;
    /// Multiplicity for IS_B20 lookups
    pub const MU_IS_B20: usize = 16;
    /// Multiplicity for HWSL lookups
    pub const MU_HWSL: usize = 17;
    /// Multiplicity for `BYTE_ALU[opsel=AND]` lookups
    pub const MU_BYTE_ALU_AND: usize = 18;
    /// Multiplicity for `BYTE_ALU[opsel=OR]` lookups
    pub const MU_BYTE_ALU_OR: usize = 19;
    /// Multiplicity for `BYTE_ALU[opsel=XOR]` lookups
    pub const MU_BYTE_ALU_XOR: usize = 20;
    /// Total number of columns
    pub const NUM_COLUMNS: usize = 21;
}

/// Number of rows in the BITWISE table: 256 * 256 * 16 = 2^20
pub const NUM_ROWS: usize = 256 * 256 * 16;

/// Number of precomputed (non-multiplicity) columns
pub const NUM_PRECOMPUTED_COLS: usize = 11;

// =========================================================================
// Compile-time row generation
// =========================================================================

/// Generate bitwise table row values at compile time.
///
/// This is a `const fn` that can be evaluated at compile time, allowing
/// the verifier to compute table values without runtime overhead.
///
/// Index encoding: `index = x + y * 256 + z * 65536`
/// where x, y ∈ [0, 255] and z ∈ [0, 15]
///
/// Returns the 11 precomputed columns: [X, Y, Z, AND, OR, XOR, MSB8, MSB16, ZERO, SLL, SLLC]
#[inline]
pub const fn generate_bitwise_row(index: usize) -> [u64; NUM_PRECOMPUTED_COLS] {
    let x = (index & 0xFF) as u32;
    let y = ((index >> 8) & 0xFF) as u32;
    let z = ((index >> 16) & 0xF) as u32;

    // Bitwise operations on bytes
    let and_val = x & y;
    let or_val = x | y;
    let xor_val = x ^ y;

    // MSB extractions
    let msb8 = (x >> 7) & 1;
    let halfword = x + y * 256;
    let msb16 = (halfword >> 15) & 1;

    // Zero check (X + 256*Y + 65536*Z must be zero)
    let is_zero = if x == 0 && y == 0 && z == 0 { 1 } else { 0 };

    // Shift operations on halfword
    let sll = if z == 0 {
        halfword
    } else {
        (halfword << z) & 0xFFFF
    };
    let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };

    [
        x as u64,       // X
        y as u64,       // Y
        z as u64,       // Z
        and_val as u64, // AND
        or_val as u64,  // OR
        xor_val as u64, // XOR
        msb8 as u64,    // MSB8
        msb16 as u64,   // MSB16
        is_zero as u64, // ZERO
        sll as u64,     // SLL
        sllc as u64,    // SLLC
    ]
}

/// Whether this table is preprocessed (commitment is static).
///
/// Preprocessed tables have their commitment known at compile time,
/// so it's not included in proofs - both prover and verifier use the
/// static value in the Fiat-Shamir transcript.
pub const fn is_preprocessed() -> bool {
    true
}

// =========================================================================
// Preprocessed commitment
// =========================================================================

/// Returns the static BITWISE preprocessed commitment for `blowup_factor`,
/// or `None` if no value is shipped for it. Values were generated by the
/// `compute_static_commitments` binary at the project's standard
/// `coset_offset = 3` (the value every in-tree `ProofOptions` constructor
/// pins) and pinned by `bitwise_static_matches_recompute_*` tests so any
/// drift in the AIR or FFT pipeline is caught at test time. The verifier
/// reads these from its compiled binary — no input data is trusted.
///
/// # Regenerating
///
/// Only regenerate these match arms after a *deliberate, reviewed* change
/// to the BITWISE table layout, the AIR's preprocessed column count, or
/// the FFT / LDE / Merkle pipeline. Run:
///
/// ```text
/// cargo run --bin compute_static_commitments --release
/// ```
///
/// and paste the printed match arms over the ones below.
///
/// **If a drift test failed, do not regenerate first.** The drift tests
/// exist to force a human to ask "why did this change?" before the new
/// bytes get blessed. Re-pasting on a drift failure silently launders an
/// unintended table change into the verifier's compiled-in trust anchor.
fn static_commitment(blowup_factor: u8) -> Option<Commitment> {
    match blowup_factor {
        2 => Some([
            0xfa, 0x3e, 0xcf, 0x80, 0xfd, 0x95, 0xe5, 0x09, 0x74, 0xd4, 0x55, 0x23, 0xf6, 0x42,
            0xb6, 0x4b, 0x05, 0xc4, 0xf9, 0x66, 0xc2, 0x4d, 0xff, 0xda, 0x31, 0x47, 0xab, 0x7b,
            0x0c, 0x6d, 0xc4, 0xcf,
        ]),
        4 => Some([
            0xff, 0x76, 0x8e, 0x85, 0x4b, 0xdc, 0x32, 0x61, 0x96, 0x16, 0x15, 0x19, 0x73, 0x70,
            0xf0, 0x64, 0x81, 0xfd, 0x4f, 0x5c, 0xbd, 0x9c, 0x30, 0x26, 0xd5, 0xc0, 0x81, 0xf3,
            0xce, 0x38, 0x50, 0x3e,
        ]),
        8 => Some([
            0x0e, 0x1b, 0xc1, 0x0d, 0xae, 0x64, 0xe7, 0xca, 0xe0, 0x2a, 0x3b, 0xab, 0xd7, 0xd2,
            0xbb, 0x80, 0xd5, 0x24, 0x5a, 0xce, 0x25, 0xb6, 0x84, 0x77, 0x9c, 0xb5, 0xeb, 0x67,
            0x61, 0x82, 0x78, 0x3d,
        ]),
        _ => None,
    }
}

/// Computes the Merkle commitment over the precomputed bitwise table columns.
///
/// This builds a Merkle tree over the LDE (Low Degree Extension) of the precomputed
/// columns, matching exactly how the prover commits to traces. The tree has
/// NUM_ROWS * blowup_factor leaves, enabling FRI queries at any index
/// in the extended domain.
///
/// Critical for security: the commitment must be over LDE values (not raw values)
/// because FRI queries can target any index in [0, N*blowup). A raw-value commitment
/// would only have N leaves, unable to verify queries at indices >= N.
///
/// Exposed for the `compute_static_commitments` binary and the
/// drift-detection tests in `static_commitments_tests`. Production callers
/// should go through [`preprocessed_commitment`] so the static const-table
/// shortcut is used when applicable.
#[doc(hidden)]
pub fn compute_preprocessed_commitment(options: &ProofOptions) -> Commitment {
    // Step 1: Generate precomputed columns in parallel
    // Each column is generated independently by iterating over all row indices
    #[cfg(feature = "parallel")]
    let columns: Vec<Vec<FE>> = (0..NUM_PRECOMPUTED_COLS)
        .into_par_iter()
        .map(|col_idx| {
            (0..NUM_ROWS)
                .map(|idx| {
                    let row = generate_bitwise_row(idx);
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
            let row = generate_bitwise_row(idx);
            for (col_idx, &value) in row.iter().enumerate() {
                cols[col_idx].push(FE::from(value));
            }
        }
        cols
    };

    // Step 2: Interpolate each column to a polynomial (parallel)
    #[cfg(feature = "parallel")]
    let polys: Vec<Polynomial<FE>> = columns
        .par_iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for bitwise column")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let polys: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for bitwise column")
        })
        .collect();

    // Step 3: Evaluate polynomials on LDE domain (parallel)
    let blowup_factor = options.blowup_factor as usize;
    let coset_offset = FE::from(options.coset_offset);

    #[cfg(feature = "parallel")]
    let lde_columns: Vec<Vec<FE>> = polys
        .par_iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let lde_columns: Vec<Vec<FE>> = polys
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, blowup_factor, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    let (_, root) = commit_bit_reversed(&lde_columns, ROWS_PER_LEAF)
        .expect("Failed to build Merkle tree for bitwise LDE");
    root
}

/// Returns the preprocessed commitment for the bitwise table.
///
/// Looks up `blowup_factor` via [`static_commitment`] when `coset_offset == 3`
/// (the value every in-tree `ProofOptions` constructor pins, and the offset
/// the static bytes were generated for); on miss — either a non-3 coset or a
/// `blowup_factor` outside `STATIC_BLOWUP_FACTORS` — recomputes from scratch.
pub fn preprocessed_commitment(options: &ProofOptions) -> Commitment {
    if options.coset_offset == 3
        && let Some(commitment) = static_commitment(options.blowup_factor)
    {
        return commitment;
    }
    log::warn!(
        "bitwise preprocessed commitment not static for (blowup={}, coset={}); \
         falling back to recompute. Add a match arm to `static_commitment` by running \
         `cargo run --bin compute_static_commitments --release`.",
        options.blowup_factor,
        options.coset_offset,
    );
    compute_preprocessed_commitment(options)
}

// =========================================================================
// Trace generation
// =========================================================================

/// Generates the precomputed BITWISE trace table.
///
/// This creates a table with 2^20 rows, one for each combination of:
/// - X: 0..256 (byte)
/// - Y: 0..256 (byte)
/// - Z: 0..16 (4-bit shift amount)
///
/// All output columns are precomputed. Multiplicity columns are initialized
/// to zero and will be updated when other tables send lookups.
pub fn generate_bitwise_trace() -> TraceTable<GoldilocksField, GoldilocksExtension> {
    let mut trace = TraceTable::new_main(
        crate::tables::types::zeroed_fe_vec(NUM_ROWS * cols::NUM_COLUMNS),
        cols::NUM_COLUMNS,
        1,
    );
    let table = &mut trace.main_table;

    for x in 0u32..256 {
        for y in 0u32..256 {
            for z in 0u32..16 {
                let row_idx = (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256;

                // Input columns
                table.set_byte(row_idx, cols::X, x as u8);
                table.set_byte(row_idx, cols::Y, y as u8);
                table.set_byte(row_idx, cols::Z, z as u8);

                // Bitwise operation results
                table.set_byte(row_idx, cols::AND, (x & y) as u8);
                table.set_byte(row_idx, cols::OR, (x | y) as u8);
                table.set_byte(row_idx, cols::XOR, (x ^ y) as u8);

                // MSB extractions
                let msb8 = (x >> 7) & 1;
                let halfword = x + y * 256;
                let msb16 = (halfword >> 15) & 1;
                table.set_bool(row_idx, cols::MSB8, msb8 == 1);
                table.set_bool(row_idx, cols::MSB16, msb16 == 1);

                // Zero check (X + 256*Y + 65536*Z must be zero)
                table.set_bool(row_idx, cols::ZERO, x == 0 && y == 0 && z == 0);

                // Shift operations on halfword
                let sll = if z == 0 {
                    halfword
                } else {
                    (halfword << z) & 0xFFFF
                };
                let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };
                table.set_half(row_idx, cols::SLL, sll as u16);
                table.set_half(row_idx, cols::SLLC, sllc as u16);

                // Multiplicity columns start at zero. They are filled by
                // `BitwiseHistogram::fill_multiplicities`; `update_multiplicities`
                // only tops up the continuation L2G lookups afterward.
            }
        }
    }

    trace
}

/// Computes the row index for a given (X, Y, Z) tuple.
#[inline]
pub fn row_index(x: u8, y: u8, z: u8) -> usize {
    debug_assert!(z < 16, "Z must be in range [0, 16)");
    (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256
}

/// Updates multiplicity columns based on lookups from other tables.
///
/// This function should be called after all other tables have recorded their lookups.
///
/// # Arguments
/// * `trace` - The BITWISE trace table to update
/// * `ops` - Vector of BitwiseOperation requests
pub fn update_multiplicities(
    trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ops: &[BitwiseOperation],
) {
    // A pre-uploaded device copy of the main trace would go stale with the
    // in-place edits below; drop it so the commit re-uploads fresh data.
    #[cfg(feature = "cuda")]
    trace.clear_main_rowmajor_dev();
    for op in ops {
        let row = row_index(op.x, op.y, op.z);
        let mu_col = mu_column(op.lookup_type);

        // Increment multiplicity
        let current = trace.main_table.get_row(row)[mu_col];
        trace.main_table.set_fe(row, mu_col, current + FE::one());
    }
}

/// Number of distinct BITWISE lookup types (one multiplicity column each).
/// Derived from [`BitwiseOperationType::ALL`], which the compile-time guard
/// below keeps in lockstep with [`lookup_type_index`].
pub(crate) const NUM_LOOKUP_TYPES: usize = BitwiseOperationType::ALL.len();

/// Dense index in `[0, NUM_LOOKUP_TYPES)` for a lookup type. Ordering is an
/// internal detail of the histogram; [`BitwiseOperationType::ALL`] is its
/// inverse, enforced at compile time.
#[inline]
pub(crate) const fn lookup_type_index(t: BitwiseOperationType) -> usize {
    match t {
        BitwiseOperationType::Msb8 => 0,
        BitwiseOperationType::Msb16 => 1,
        BitwiseOperationType::Zero => 2,
        BitwiseOperationType::AreBytes => 3,
        BitwiseOperationType::IsHalf => 4,
        BitwiseOperationType::IsB20 => 5,
        BitwiseOperationType::Hwsl => 6,
        BitwiseOperationType::ByteAluAnd => 7,
        BitwiseOperationType::ByteAluOr => 8,
        BitwiseOperationType::ByteAluXor => 9,
    }
}

/// The MU_* multiplicity column for each lookup type, in [`lookup_type_index`]
/// order. This is the single source of truth for the type→column mapping: both
/// the per-op path ([`mu_column`]) and the histogram fill ([`type_mu_column`])
/// index into this one array. The compile-time block below checks the entries
/// are pairwise distinct, so a duplicate column is a build error rather than a
/// silent overwrite in [`BitwiseHistogram::fill_multiplicities`].
const MU_COLUMNS: [usize; NUM_LOOKUP_TYPES] = [
    cols::MU_MSB8,         // Msb8
    cols::MU_MSB16,        // Msb16
    cols::MU_ZERO,         // Zero
    cols::MU_ARE_BYTES,    // AreBytes
    cols::MU_IS_HALF,      // IsHalf
    cols::MU_IS_B20,       // IsB20
    cols::MU_HWSL,         // Hwsl
    cols::MU_BYTE_ALU_AND, // ByteAluAnd
    cols::MU_BYTE_ALU_OR,  // ByteAluOr
    cols::MU_BYTE_ALU_XOR, // ByteAluXor
];

/// Multiplicity column for a lookup type. Used by the per-op path
/// ([`update_multiplicities`]), which is still live production code: continuation
/// epochs add their L2G lookups through it on top of the histogram-filled trace.
#[inline]
pub(crate) const fn mu_column(t: BitwiseOperationType) -> usize {
    MU_COLUMNS[lookup_type_index(t)]
}

/// Multiplicity column for the histogram lane at dense index `type_idx`
/// (inverse of [`lookup_type_index`]). Used by [`BitwiseHistogram::fill_multiplicities`].
///
/// Reads directly from [`MU_COLUMNS`], the single type→column source of truth.
#[inline]
const fn type_mu_column(type_idx: usize) -> usize {
    MU_COLUMNS[type_idx]
}

// Compile-time guards on the type↔column bookkeeping.
//
// 1. `ALL` must list every lookup type exactly once, in `lookup_type_index`
//    order (i.e. it is the exact inverse of that mapping). Adding a variant
//    forces the `lookup_type_index` match to be extended (exhaustiveness), and
//    this assert then forces `ALL` — and with it `NUM_LOOKUP_TYPES` — to follow.
// 2. The type→column map is now derived from the single `MU_COLUMNS` array, and
//    its entries are checked pairwise distinct (injective). A wrong or duplicated
//    MU column would silently unbalance the BITWISE bus, so both are compile
//    errors, not test failures.
const _: () = {
    let mut i = 0;
    while i < NUM_LOOKUP_TYPES {
        assert!(lookup_type_index(BitwiseOperationType::ALL[i]) == i);
        let mut j = i + 1;
        while j < NUM_LOOKUP_TYPES {
            assert!(
                MU_COLUMNS[i] != MU_COLUMNS[j],
                "MU_COLUMNS entries must map distinct lookup types to distinct columns"
            );
            j += 1;
        }
        i += 1;
    }
};

/// "Histogram-on-the-fly" accumulator for BITWISE lookup multiplicities.
///
/// Replaces materializing the giant `Vec<BitwiseOperation>` (whose only consumer
/// is the multiplicity count) with a dense counter array. Each lookup increments
/// `counters[type_idx * NUM_ROWS + row_index(x, y, z)]`.
///
/// The histogram is a commutative monoid: increments and [`merge`](Self::merge)
/// are order-independent, so per-thread histograms can be tree-reduced and the
/// resulting multiplicities are byte-identical to the serial per-op count that
/// [`update_multiplicities`] produces (both just sum the same lookups per cell).
///
/// Memory: `NUM_ROWS * NUM_LOOKUP_TYPES * 8` bytes = 2^20 * 10 * 8 = 80 MiB.
pub(crate) struct BitwiseHistogram {
    counters: Box<[u64]>,
}

impl BitwiseHistogram {
    /// Allocate a zeroed histogram (80 MiB).
    // No `Default` impl on purpose: `new()` allocates 80 MiB, so a stray
    // `..Default::default()` / `#[derive(Default)]` must not silently do that.
    #[allow(clippy::new_without_default)]
    pub(crate) fn new() -> Self {
        Self {
            counters: vec![0u64; NUM_ROWS * NUM_LOOKUP_TYPES].into_boxed_slice(),
        }
    }

    /// Increment the counter for one lookup.
    #[inline]
    pub(crate) fn bump(&mut self, op: BitwiseOperation) {
        self.bump_n(op, 1);
    }

    /// Add `n` occurrences of one lookup in a single step (e.g. CPU padding rows,
    /// which all send identical all-zero lookups).
    #[inline]
    pub(crate) fn bump_n(&mut self, op: BitwiseOperation, n: u64) {
        let idx = lookup_type_index(op.lookup_type) * NUM_ROWS + row_index(op.x, op.y, op.z);
        // (x, y) are u8, and row_index debug-asserts z < 16, so in debug builds a
        // corrupt op fails loudly here. In release an out-of-domain z would NOT
        // panic: the flat index can land in another type's lane and silently
        // mis-count both cells — the proof then fails verification instead of the
        // prover crashing. What actually upholds the invariant is that every
        // `BitwiseOperation` constructor masks or debug-asserts z < 16.
        self.counters[idx] += n;
    }

    /// Fold a slice of lookups into the histogram.
    #[inline]
    pub(crate) fn add_ops(&mut self, ops: &[BitwiseOperation]) {
        for &op in ops {
            self.bump(op);
        }
    }

    /// Merge another histogram into this one (commutative, order-independent).
    pub(crate) fn merge(&mut self, other: &BitwiseHistogram) {
        for (a, b) in self.counters.iter_mut().zip(other.counters.iter()) {
            *a += *b;
        }
    }

    /// Write the accumulated multiplicities into the BITWISE trace's MU columns.
    ///
    /// OVERWRITES each nonzero cell with its count (it does not add to what is
    /// there), so it assumes the MU columns are still zero — true for a fresh
    /// [`generate_bitwise_trace`] output, where it produces exactly the same MU
    /// columns as calling [`update_multiplicities`] with the full op vector.
    /// Callers that layer additional lookups on top (continuation epochs add
    /// their L2G lookups via `update_multiplicities`, which increments) must do
    /// so strictly AFTER this fill, never before.
    pub(crate) fn fill_multiplicities(
        &self,
        trace: &mut TraceTable<GoldilocksField, GoldilocksExtension>,
    ) {
        for type_idx in 0..NUM_LOOKUP_TYPES {
            let mu_col = type_mu_column(type_idx);
            let base = type_idx * NUM_ROWS;
            for row in 0..NUM_ROWS {
                let count = self.counters[base + row];
                if count != 0 {
                    trace.main_table.set_fe(row, mu_col, FE::from(count));
                }
            }
        }
    }
}

/// Types of lookups the BITWISE table provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitwiseOperationType {
    Msb8,
    Msb16,
    Zero,
    AreBytes,
    IsHalf,
    IsB20,
    Hwsl,
    ByteAluAnd,
    ByteAluOr,
    ByteAluXor,
}

impl BitwiseOperationType {
    /// Every lookup type exactly once, in [`lookup_type_index`] order (the
    /// compile-time guard next to [`type_mu_column`] enforces this). The array
    /// length is the single origin of [`NUM_LOOKUP_TYPES`].
    pub(crate) const ALL: [Self; 10] = [
        Self::Msb8,
        Self::Msb16,
        Self::Zero,
        Self::AreBytes,
        Self::IsHalf,
        Self::IsB20,
        Self::Hwsl,
        Self::ByteAluAnd,
        Self::ByteAluOr,
        Self::ByteAluXor,
    ];
}

/// A lookup request to the BITWISE precomputed table.
///
/// The BITWISE table has 2^20 rows indexed by `(x, y, z)`.
/// Each row contains precomputed results for various operations.
///
/// # Fields (matching spec column names)
/// - `lookup_type`: Which operation result to look up
/// - `x`: Byte input (0-255)
/// - `y`: Byte input (0-255)
/// - `z`: 4-bit value (0-15), shift amount for HWSL
///
/// # How inputs map to operations
/// - AND/OR/XOR: `x OP y`
/// - MSB8: MSB of `x`
/// - MSB16: MSB of halfword `x + y * 256`
/// - ARE_BYTES: Range check both `x` and `y`; use `y = 0` for a single byte
/// - IS_HALF: Range check on `x + y * 256`
/// - HWSL: Shift `x + y * 256` by `z` bits, returning [SLL, SLLC] as a pair
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BitwiseOperation {
    pub lookup_type: BitwiseOperationType,
    pub x: u8,
    pub y: u8,
    pub z: u8,
}

impl BitwiseOperation {
    /// Create a new bitwise operation.
    pub fn new(lookup_type: BitwiseOperationType, x: u8, y: u8, z: u8) -> Self {
        debug_assert!(z < 16, "z must be in range [0, 16)");
        Self {
            lookup_type,
            x,
            y,
            z,
        }
    }

    /// Create an operation for byte ops (AND, OR, XOR) where z is unused.
    pub fn byte_op(lookup_type: BitwiseOperationType, x: u8, y: u8) -> Self {
        Self::new(lookup_type, x, y, 0)
    }

    /// Create an operation for single-byte ops (MSB8, ARE_BYTES with y=0).
    pub fn single_byte(lookup_type: BitwiseOperationType, x: u8) -> Self {
        Self::new(lookup_type, x, 0, 0)
    }

    /// Create an operation for halfword ops (MSB16, IS_HALF).
    pub fn halfword(lookup_type: BitwiseOperationType, x: u8, y: u8) -> Self {
        Self::new(lookup_type, x, y, 0)
    }

    /// Create a ZERO lookup for a value up to 20 bits.
    /// Value is decomposed as: x + 256*y + 65536*z.
    pub fn zero(value: u32) -> Self {
        assert!(value < (1 << 20), "ZERO value must fit in 20 bits");
        let x = (value & 0xFF) as u8;
        let y = ((value >> 8) & 0xFF) as u8;
        let z = ((value >> 16) & 0xF) as u8;
        Self::new(BitwiseOperationType::Zero, x, y, z)
    }

    /// Create an operation for HWSL shift lookups.
    pub fn shift_op(lookup_type: BitwiseOperationType, x: u8, y: u8, z: u8) -> Self {
        Self::new(lookup_type, x, y, z)
    }

    /// Create an IS_B20 operation for 20-bit range checks.
    /// Value is packed as: x + 256*y + 65536*z (where z is 4 bits).
    pub fn b20(x: u8, y: u8, z: u8) -> Self {
        debug_assert!(z < 16, "z must be 4-bit value for IS_B20");
        Self::new(BitwiseOperationType::IsB20, x, y, z)
    }
}

// =========================================================================
// Bus interactions
// =========================================================================

/// Creates all bus interactions for the BITWISE table.
///
/// The BITWISE table is a **receiver** for all lookups (negative multiplicity
/// in the spec corresponds to receiving lookups from other tables).
pub fn bus_interactions() -> Vec<BusInteraction> {
    vec![
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
        // Input is packed as Word2L (X + 2^8 * Y would need custom, but spec says X + 256*Y)
        // Since X and Y are bytes, we use a linear combination
        BusInteraction::receiver(
            BusId::Msb16,
            Multiplicity::Column(cols::MU_MSB16),
            vec![
                // X + 256*Y as linear combination
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
        // ZERO[X + 256*Y + 65536*Z] -> ZERO
        BusInteraction::receiver(
            BusId::Zero,
            Multiplicity::Column(cols::MU_ZERO),
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
                    stark::lookup::LinearTerm::Column {
                        coefficient: 65536,
                        column: cols::Z,
                    },
                ]),
                BusValue::Packed {
                    start_column: cols::ZERO,
                    packing: Packing::Direct,
                },
            ],
        ),
        // ARE_BYTES[X, Y] - range check two byte values, no output.
        // Single-byte checks (spec template `IS_BYTE<X>`) send Y=0.
        BusInteraction::receiver(
            BusId::AreBytes,
            Multiplicity::Column(cols::MU_ARE_BYTES),
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
        // IS_B20[X + 256*Y + 65536*Z] - range check for 20-bit
        BusInteraction::receiver(
            BusId::IsB20,
            Multiplicity::Column(cols::MU_IS_B20),
            vec![BusValue::linear(vec![
                stark::lookup::LinearTerm::Column {
                    coefficient: 1,
                    column: cols::X,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 256,
                    column: cols::Y,
                },
                stark::lookup::LinearTerm::Column {
                    coefficient: 65536,
                    column: cols::Z,
                },
            ])],
        ),
        // HWSL[X + 256*Y, Z] -> [SLL, SLLC]
        BusInteraction::receiver(
            BusId::Hwsl,
            Multiplicity::Column(cols::MU_HWSL),
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
                    start_column: cols::Z,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::SLL,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::SLLC,
                    packing: Packing::Direct,
                },
            ],
        ),
        // BYTE_ALU[opsel, X, Y] -> out.
        // Unifies AND/OR/XOR into one bus keyed by the `alu_op` descriptor.
        // Implemented as one receiver per opsel, reusing the precomputed
        // AND/OR/XOR result columns (the "single 2^20 column" in bitwise.typ is
        // an optimization note, not a requirement).
        BusInteraction::receiver(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU_BYTE_ALU_AND),
            vec![
                BusValue::constant(alu_op::AND as u64),
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::AND,
                    packing: Packing::Direct,
                },
            ],
        ),
        BusInteraction::receiver(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU_BYTE_ALU_OR),
            vec![
                BusValue::constant(alu_op::OR as u64),
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::OR,
                    packing: Packing::Direct,
                },
            ],
        ),
        BusInteraction::receiver(
            BusId::ByteAlu,
            Multiplicity::Column(cols::MU_BYTE_ALU_XOR),
            vec![
                BusValue::constant(alu_op::XOR as u64),
                BusValue::Packed {
                    start_column: cols::X,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::Y,
                    packing: Packing::Direct,
                },
                BusValue::Packed {
                    start_column: cols::XOR,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}
