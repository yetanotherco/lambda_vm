//! BITWISE precomputed lookup table.
//!
//! This table provides 11 different lookup types used by other tables:
//!
//! ## Range Checks
//! - `IS_BYTE[X]` - X is a valid byte [0, 256)
//! - `IS_HALF[X]` - X is a valid halfword [0, 2^16)
//! - `IS_B20[X]` - X is a valid 20-bit value [0, 2^20)
//!
//! ## Bitwise Operations
//! - `AND_BYTE[X, Y]` -> X & Y
//! - `OR_BYTE[X, Y]` -> X | Y
//! - `XOR_BYTE[X, Y]` -> X ^ Y
//! - `MSB8[X]` -> most significant bit of byte
//! - `MSB16[X]` -> most significant bit of halfword
//! - `ZERO[X]` -> whether X is zero
//!
//! ## Shift Helpers
//! - `HWSL[X, Z]` -> (X << Z) mod 2^16
//! - `HWSLC[X, Z]` -> X >> (16 - Z)
//!
//! ## Table Structure
//!
//! The table has 2^20 rows (256 * 256 * 16) with precomputed values.
//! Each row is indexed by (X: Byte, Y: Byte, Z: B4).
//!
//! All lookups are provided as receivers with negative multiplicity,
//! meaning other tables send to this table.

use lazy_static::lazy_static;
use math::fft::cpu::bit_reversing::in_place_bit_reverse_permute;
use math::polynomial::Polynomial;
use stark::config::{BatchedMerkleTree, Commitment};
use stark::lookup::{BusInteraction, BusValue, Multiplicity, Packing};
use stark::prover::evaluate_polynomial_on_lde_domain;
use stark::trace::{TraceTable, columns2rows};

use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read as IoRead, Write as IoWrite};
use std::path::PathBuf;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::types::{BusId, FE, GoldilocksExtension, GoldilocksField};

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
    /// Multiplicity for AND_BYTE lookups
    pub const MU_AND: usize = 11;
    /// Multiplicity for OR_BYTE lookups
    pub const MU_OR: usize = 12;
    /// Multiplicity for XOR_BYTE lookups
    pub const MU_XOR: usize = 13;
    /// Multiplicity for MSB8 lookups
    pub const MU_MSB8: usize = 14;
    /// Multiplicity for MSB16 lookups
    pub const MU_MSB16: usize = 15;
    /// Multiplicity for ZERO lookups
    pub const MU_ZERO: usize = 16;
    /// Multiplicity for IS_BYTE lookups
    pub const MU_IS_BYTE: usize = 17;
    /// Multiplicity for IS_HALF lookups
    pub const MU_IS_HALF: usize = 18;
    /// Multiplicity for IS_B20 lookups
    pub const MU_IS_B20: usize = 19;
    /// Multiplicity for HWSL lookups
    pub const MU_HWSL: usize = 20;
    /// Multiplicity for HWSLC lookups
    pub const MU_HWSLC: usize = 21;

    /// Total number of columns
    pub const NUM_COLUMNS: usize = 22;
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

    // Zero check (both X and Y must be zero)
    let is_zero = if x == 0 && y == 0 { 1 } else { 0 };

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

/// Whether this table is preprocessed (commitment is hardcoded).
///
/// Preprocessed tables have their commitment known at compile time,
/// so it's not included in proofs - both prover and verifier use the
/// hardcoded value in the Fiat-Shamir transcript.
pub const fn is_preprocessed() -> bool {
    true
}

// =========================================================================
// Preprocessed data (computed once, cached in memory)
// =========================================================================

/// Cached precomputed data for the Bitwise table.
///
/// Contains polynomial coefficients and LDE evaluations that are constant
/// regardless of the program being proven. Caching these eliminates minutes
/// of redundant computation per proof.
///
/// ## Memory Usage
/// - `polynomials`: ~88 MB (11 polynomials × 2^20 coefficients × 8 bytes)
/// - `lde_columns`: ~352 MB (11 columns × 2^22 evaluations × 8 bytes)
/// - Total: ~440 MB in memory
///
/// ## What's NOT Cached
/// - Merkle tree (~700 MB): Rebuilt from `lde_columns` during proving.
///   This can be revisited based on benchmarks.
pub struct PrecomputedBitwiseData {
    /// Configuration hash for cache validation.
    /// Hash of (LDE_BLOWUP_FACTOR, LDE_COSET_OFFSET, NUM_ROWS, code version).
    pub config_hash: [u8; 32],

    /// Merkle root commitment (32 bytes) - for verification.
    pub commitment: Commitment,

    /// Polynomial coefficients (88 MB).
    /// Shape: 11 polynomials, needed for OOD evaluation.
    pub polynomials: Vec<Polynomial<FE>>,

    /// Bit-reversed LDE evaluations of precomputed columns (352 MB).
    /// Shape: 11 columns × 2^22 evaluations.
    /// Already bit-reversed for direct use in Merkle tree construction.
    pub lde_columns: Vec<Vec<FE>>,
}

impl PrecomputedBitwiseData {
    /// Validates that the cached data matches the current configuration.
    pub fn is_valid(&self) -> bool {
        self.config_hash == compute_config_hash()
    }
}

/// Computes a hash of the configuration parameters that affect precomputed data.
/// If any of these change, cached data must be recomputed.
fn compute_config_hash() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"bitwise_precomputed_v1");
    hasher.update(LDE_BLOWUP_FACTOR.to_le_bytes());
    hasher.update(LDE_COSET_OFFSET.to_le_bytes());
    hasher.update(NUM_ROWS.to_le_bytes());
    hasher.update(NUM_PRECOMPUTED_COLS.to_le_bytes());
    hasher.finalize().into()
}

lazy_static! {
    /// Fully cached precomputed data for the Bitwise table.
    ///
    /// Contains:
    /// - Merkle root commitment (32 bytes)
    /// - Polynomial coefficients (88 MB) - needed for OOD evaluation
    /// - Bit-reversed LDE columns (352 MB) - needed for Merkle tree rebuild
    ///
    /// Tries to load from disk cache first, falls back to computing if not available.
    /// Both prover and verifier can use this data, though verifiers typically only need the commitment.
    pub static ref BITWISE_PRECOMPUTED: PrecomputedBitwiseData = load_or_compute_precomputed_data();

    /// Backward-compatible commitment accessor.
    /// Prefer using `BITWISE_PRECOMPUTED.commitment` directly.
    pub static ref BITWISE_TABLE_COMMITMENT: Commitment = BITWISE_PRECOMPUTED.commitment;
}

/// Standard blowup factor for LDE domain (matches proof options).
const LDE_BLOWUP_FACTOR: usize = 4;

/// Standard coset offset for LDE domain (matches proof options).
const LDE_COSET_OFFSET: u64 = 3;

// =========================================================================
// Disk caching
// =========================================================================

/// Magic bytes for cache file identification.
const CACHE_MAGIC: &[u8; 4] = b"BTWC";

/// Cache file version. Increment when format changes.
const CACHE_VERSION: u32 = 1;

/// Returns the path to the disk cache file.
///
/// Cache location priority:
/// 1. `LAMBDA_VM_CACHE_DIR` environment variable
/// 2. `$HOME/.lambda_vm/cache/`
fn cache_path() -> Option<PathBuf> {
    let cache_dir = if let Ok(dir) = std::env::var("LAMBDA_VM_CACHE_DIR") {
        PathBuf::from(dir)
    } else if let Some(home) = dirs::home_dir() {
        home.join(".lambda_vm").join("cache")
    } else {
        return None;
    };

    Some(cache_dir.join("bitwise_precomputed.bin"))
}

/// Loads precomputed data from disk cache, or computes it if not available.
fn load_or_compute_precomputed_data() -> PrecomputedBitwiseData {
    // Try to load from disk
    if let Some(path) = cache_path() {
        if path.exists() {
            match load_from_disk(&path) {
                Ok(data) => {
                    // Validate config hash
                    if data.config_hash == compute_config_hash() {
                        log::info!(
                            "Loaded bitwise precomputed data from cache: {}",
                            path.display()
                        );
                        return data;
                    } else {
                        log::info!(
                            "Bitwise cache config mismatch, recomputing (path: {})",
                            path.display()
                        );
                    }
                }
                Err(e) => {
                    log::debug!("Failed to load bitwise cache: {} (path: {})", e, path.display());
                }
            }
        }
    }

    // Compute from scratch
    log::info!("Computing bitwise precomputed data (this may take a few minutes on first run)...");
    let start = std::time::Instant::now();
    let data = compute_all_precomputed_data();
    log::info!("Bitwise precomputed data computed in {:?}", start.elapsed());

    // Save to disk for next time
    if let Some(path) = cache_path() {
        if let Err(e) = save_to_disk(&data, &path) {
            log::warn!("Failed to save bitwise cache: {} (path: {})", e, path.display());
        } else {
            log::info!("Saved bitwise precomputed data to cache: {}", path.display());
        }
    }

    data
}

/// Saves precomputed data to disk in binary format.
fn save_to_disk(data: &PrecomputedBitwiseData, path: &PathBuf) -> std::io::Result<()> {
    // Ensure parent directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write header
    writer.write_all(CACHE_MAGIC)?;
    writer.write_all(&CACHE_VERSION.to_le_bytes())?;
    writer.write_all(&data.config_hash)?;
    writer.write_all(&data.commitment)?;

    // Write polynomial count and sizes
    let num_polys = data.polynomials.len() as u32;
    writer.write_all(&num_polys.to_le_bytes())?;

    // Write each polynomial's coefficients
    for poly in &data.polynomials {
        let coeffs = poly.coefficients();
        let num_coeffs = coeffs.len() as u32;
        writer.write_all(&num_coeffs.to_le_bytes())?;

        for coeff in coeffs {
            let value: u64 = *coeff.value();
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    // Write LDE column count and sizes
    let num_lde_cols = data.lde_columns.len() as u32;
    writer.write_all(&num_lde_cols.to_le_bytes())?;

    // Write each LDE column
    for col in &data.lde_columns {
        let col_len = col.len() as u32;
        writer.write_all(&col_len.to_le_bytes())?;

        for elem in col {
            let value: u64 = *elem.value();
            writer.write_all(&value.to_le_bytes())?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Loads precomputed data from disk.
fn load_from_disk(path: &PathBuf) -> std::io::Result<PrecomputedBitwiseData> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read and validate magic
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != CACHE_MAGIC {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "Invalid cache magic bytes",
        ));
    }

    // Read and validate version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    let version = u32::from_le_bytes(version_bytes);
    if version != CACHE_VERSION {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Cache version mismatch: expected {}, got {}", CACHE_VERSION, version),
        ));
    }

    // Read config hash and commitment
    let mut config_hash = [0u8; 32];
    reader.read_exact(&mut config_hash)?;

    let mut commitment = [0u8; 32];
    reader.read_exact(&mut commitment)?;

    // Read polynomials
    let mut num_polys_bytes = [0u8; 4];
    reader.read_exact(&mut num_polys_bytes)?;
    let num_polys = u32::from_le_bytes(num_polys_bytes) as usize;

    let mut polynomials = Vec::with_capacity(num_polys);
    for _ in 0..num_polys {
        let mut num_coeffs_bytes = [0u8; 4];
        reader.read_exact(&mut num_coeffs_bytes)?;
        let num_coeffs = u32::from_le_bytes(num_coeffs_bytes) as usize;

        let mut coeffs = Vec::with_capacity(num_coeffs);
        for _ in 0..num_coeffs {
            let mut value_bytes = [0u8; 8];
            reader.read_exact(&mut value_bytes)?;
            let value = u64::from_le_bytes(value_bytes);
            coeffs.push(FE::from(value));
        }

        polynomials.push(Polynomial::new(&coeffs));
    }

    // Read LDE columns
    let mut num_lde_cols_bytes = [0u8; 4];
    reader.read_exact(&mut num_lde_cols_bytes)?;
    let num_lde_cols = u32::from_le_bytes(num_lde_cols_bytes) as usize;

    let mut lde_columns = Vec::with_capacity(num_lde_cols);
    for _ in 0..num_lde_cols {
        let mut col_len_bytes = [0u8; 4];
        reader.read_exact(&mut col_len_bytes)?;
        let col_len = u32::from_le_bytes(col_len_bytes) as usize;

        let mut col = Vec::with_capacity(col_len);
        for _ in 0..col_len {
            let mut value_bytes = [0u8; 8];
            reader.read_exact(&mut value_bytes)?;
            let value = u64::from_le_bytes(value_bytes);
            col.push(FE::from(value));
        }

        lde_columns.push(col);
    }

    Ok(PrecomputedBitwiseData {
        config_hash,
        commitment,
        polynomials,
        lde_columns,
    })
}

/// Computes all precomputed data for the bitwise table.
///
/// This builds:
/// 1. Polynomial coefficients from column values (for OOD evaluation)
/// 2. LDE evaluations (for Merkle tree construction)
/// 3. Merkle commitment (for verification)
///
/// The Merkle tree itself is NOT cached - it's rebuilt from LDE columns
/// during proving. This trades ~2 seconds of Merkle tree rebuild time
/// for ~700MB of memory savings.
///
/// ## Why Cache Polynomials?
/// Polynomials are needed for OOD (Out-of-Domain) evaluation in the prover.
/// The random challenge `z` is sampled AFTER commitment, so we cannot
/// precompute OOD evaluations - we need the polynomial coefficients.
///
/// ## Why Cache LDE Columns?
/// LDE columns are needed to rebuild the Merkle tree for FRI opening proofs.
/// Recomputing LDE from polynomials is expensive (FFT over 2^22 points).
fn compute_all_precomputed_data() -> PrecomputedBitwiseData {
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
    // CACHED: These polynomials are needed for OOD evaluation
    #[cfg(feature = "parallel")]
    let polynomials: Vec<Polynomial<FE>> = columns
        .par_iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for bitwise column")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let polynomials: Vec<Polynomial<FE>> = columns
        .iter()
        .map(|col| {
            Polynomial::interpolate_fft::<GoldilocksField>(col)
                .expect("FFT interpolation failed for bitwise column")
        })
        .collect();

    // Step 3: Evaluate polynomials on LDE domain (parallel)
    // CACHED: These evaluations are needed to rebuild the Merkle tree
    let coset_offset = FE::from(LDE_COSET_OFFSET);

    #[cfg(feature = "parallel")]
    let mut lde_columns: Vec<Vec<FE>> = polynomials
        .par_iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, LDE_BLOWUP_FACTOR, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    #[cfg(not(feature = "parallel"))]
    let mut lde_columns: Vec<Vec<FE>> = polynomials
        .iter()
        .map(|poly| {
            evaluate_polynomial_on_lde_domain(poly, LDE_BLOWUP_FACTOR, NUM_ROWS, &coset_offset)
                .expect("LDE evaluation failed for bitwise polynomial")
        })
        .collect();

    // Step 4: Bit-reverse permute (parallel)
    // CACHED: LDE columns are stored in bit-reversed order for direct Merkle tree use
    #[cfg(feature = "parallel")]
    lde_columns.par_iter_mut().for_each(|col| {
        in_place_bit_reverse_permute(col);
    });

    #[cfg(not(feature = "parallel"))]
    for col in lde_columns.iter_mut() {
        in_place_bit_reverse_permute(col);
    }

    // Step 5: Convert columns to rows for Merkle tree
    // NOT CACHED: This is a view transformation, cheap to redo
    let lde_rows = columns2rows(lde_columns.clone());

    // Step 6: Build Merkle tree over LDE (N * blowup leaves)
    // NOT CACHED: Tree is rebuilt from lde_columns during proving (~2 seconds)
    let tree = BatchedMerkleTree::<GoldilocksField>::build(&lde_rows)
        .expect("Failed to build Merkle tree for bitwise LDE");

    PrecomputedBitwiseData {
        config_hash: compute_config_hash(),
        commitment: tree.root,
        polynomials,
        lde_columns,
    }
}

/// Returns the preprocessed commitment for the bitwise table.
///
/// This is a convenience function that accesses the cached precomputed data.
#[inline]
pub fn preprocessed_commitment() -> Commitment {
    BITWISE_PRECOMPUTED.commitment
}

/// Returns a reference to the cached polynomial coefficients.
///
/// These polynomials are needed for OOD (Out-of-Domain) evaluation.
/// Returns a `'static` reference since the data lives in `lazy_static`.
#[inline]
pub fn precomputed_polynomials() -> &'static [Polynomial<FE>] {
    &BITWISE_PRECOMPUTED.polynomials
}

/// Returns a reference to the cached bit-reversed LDE columns.
///
/// These are needed to rebuild the Merkle tree for FRI opening proofs.
/// Returns a `'static` reference since the data lives in `lazy_static`.
#[inline]
pub fn precomputed_lde_columns() -> &'static [Vec<FE>] {
    &BITWISE_PRECOMPUTED.lde_columns
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
    let mut data = vec![FE::zero(); NUM_ROWS * cols::NUM_COLUMNS];

    for x in 0u32..256 {
        for y in 0u32..256 {
            for z in 0u32..16 {
                let row_idx = (x as usize) + (y as usize) * 256 + (z as usize) * 256 * 256;
                let base = row_idx * cols::NUM_COLUMNS;

                // Input columns
                data[base + cols::X] = FE::from(x as u64);
                data[base + cols::Y] = FE::from(y as u64);
                data[base + cols::Z] = FE::from(z as u64);

                // Bitwise operation results
                data[base + cols::AND] = FE::from((x & y) as u64);
                data[base + cols::OR] = FE::from((x | y) as u64);
                data[base + cols::XOR] = FE::from((x ^ y) as u64);

                // MSB extractions
                let msb8 = (x >> 7) & 1;
                let halfword = x + y * 256;
                let msb16 = (halfword >> 15) & 1;
                data[base + cols::MSB8] = FE::from(msb8 as u64);
                data[base + cols::MSB16] = FE::from(msb16 as u64);

                // Zero check (both X and Y must be zero)
                let is_zero = if x == 0 && y == 0 { 1u64 } else { 0u64 };
                data[base + cols::ZERO] = FE::from(is_zero);

                // Shift operations on halfword
                let sll = if z == 0 {
                    halfword
                } else {
                    (halfword << z) & 0xFFFF
                };
                let sllc = if z == 0 { 0 } else { halfword >> (16 - z) };
                data[base + cols::SLL] = FE::from(sll as u64);
                data[base + cols::SLLC] = FE::from(sllc as u64);

                // Multiplicity columns start at zero
                // They will be updated by update_multiplicities()
            }
        }
    }

    TraceTable::new_main(data, cols::NUM_COLUMNS, 1)
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
    for op in ops {
        let row = row_index(op.x, op.y, op.z);
        let mu_col = match op.lookup_type {
            BitwiseOperationType::AndByte => cols::MU_AND,
            BitwiseOperationType::OrByte => cols::MU_OR,
            BitwiseOperationType::XorByte => cols::MU_XOR,
            BitwiseOperationType::Msb8 => cols::MU_MSB8,
            BitwiseOperationType::Msb16 => cols::MU_MSB16,
            BitwiseOperationType::Zero => cols::MU_ZERO,
            BitwiseOperationType::IsByte => cols::MU_IS_BYTE,
            BitwiseOperationType::IsHalf => cols::MU_IS_HALF,
            BitwiseOperationType::IsB20 => cols::MU_IS_B20,
            BitwiseOperationType::Hwsl => cols::MU_HWSL,
            BitwiseOperationType::Hwslc => cols::MU_HWSLC,
        };

        // Increment multiplicity
        let current = trace.main_table.get_row(row)[mu_col];
        trace.set_main(row, mu_col, current + FE::one());
    }
}

/// Removes rows where all multiplicity columns are zero.
/// Returns a smaller table containing only rows with actual lookups.
///
/// # WARNING: UNSOUND FOR PRODUCTION
///
/// This function is for tests only. The reduced table is NOT a valid
/// preprocessed table because:
/// 1. Row indices no longer match the (x, y, z) encoding
/// 2. The verifier cannot verify against a preprocessed commitment
/// 3. A malicious prover could claim incorrect bitwise results
///
/// This is acceptable for tests because we're testing:
/// - Bus interaction balancing (sends = receives)
/// - Constraint satisfaction
/// - LogUp protocol correctness
#[cfg(test)]
pub(crate) fn trim_zero_rows(
    trace: TraceTable<GoldilocksField, GoldilocksExtension>,
) -> TraceTable<GoldilocksField, GoldilocksExtension> {
    use super::types::FE;

    let num_rows = trace.main_table.height;

    // Find rows with any non-zero multiplicity
    let kept_rows: Vec<usize> = (0..num_rows)
        .filter(|&row| {
            let row_data = trace.main_table.get_row(row);
            // Check all multiplicity columns (indices 11-21)
            (cols::MU_AND..=cols::MU_HWSLC).any(|col| row_data[col] != FE::zero())
        })
        .collect();

    if kept_rows.is_empty() {
        // No lookups - return minimal table with 16 rows of zeros
        let data = vec![FE::zero(); 16 * cols::NUM_COLUMNS];
        return TraceTable::new_main(data, cols::NUM_COLUMNS, 1);
    }

    // Determine new table size (next power of 2, minimum 16)
    let new_size = kept_rows.len().next_power_of_two().max(16);

    // Allocate new trace data
    let mut new_data = vec![FE::zero(); new_size * cols::NUM_COLUMNS];

    // Copy kept rows to new table
    for (new_row, &old_row) in kept_rows.iter().enumerate() {
        let old_row_data = trace.main_table.get_row(old_row);
        let base = new_row * cols::NUM_COLUMNS;
        for (col, &val) in old_row_data.iter().enumerate() {
            new_data[base + col] = val;
        }
    }

    TraceTable::new_main(new_data, cols::NUM_COLUMNS, 1)
}

/// Types of lookups the BITWISE table provides.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BitwiseOperationType {
    AndByte,
    OrByte,
    XorByte,
    Msb8,
    Msb16,
    Zero,
    IsByte,
    IsHalf,
    IsB20,
    Hwsl,
    Hwslc,
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
/// - `z`: 4-bit value (0-15), shift amount for HWSL/HWSLC
///
/// # How inputs map to operations
/// - AND/OR/XOR: `x OP y`
/// - MSB8: MSB of `x`
/// - MSB16: MSB of halfword `x + y * 256`
/// - IS_BYTE/IS_HALF: Range check on `x + y * 256`
/// - HWSL/HWSLC: Shift `x + y * 256` by `z` bits
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

    /// Create an operation for single-byte ops (MSB8, IS_BYTE).
    pub fn single_byte(lookup_type: BitwiseOperationType, x: u8) -> Self {
        Self::new(lookup_type, x, 0, 0)
    }

    /// Create an operation for halfword ops (MSB16, IS_HALF, ZERO).
    pub fn halfword(lookup_type: BitwiseOperationType, x: u8, y: u8) -> Self {
        Self::new(lookup_type, x, y, 0)
    }

    /// Create an operation for shift ops (HWSL, HWSLC).
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
        // AND_BYTE[X, Y] -> AND
        BusInteraction::receiver(
            BusId::AndByte,
            Multiplicity::Column(cols::MU_AND),
            vec![
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
        // OR_BYTE[X, Y] -> OR
        BusInteraction::receiver(
            BusId::OrByte,
            Multiplicity::Column(cols::MU_OR),
            vec![
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
        // XOR_BYTE[X, Y] -> XOR
        BusInteraction::receiver(
            BusId::XorByte,
            Multiplicity::Column(cols::MU_XOR),
            vec![
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
        // ZERO[X + 256*Y] -> ZERO
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
                ]),
                BusValue::Packed {
                    start_column: cols::ZERO,
                    packing: Packing::Direct,
                },
            ],
        ),
        // IS_BYTE[X] - range check, no output
        BusInteraction::receiver(
            BusId::IsByte,
            Multiplicity::Column(cols::MU_IS_BYTE),
            vec![BusValue::Packed {
                start_column: cols::X,
                packing: Packing::Direct,
            }],
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
        // HWSL[X + 256*Y, Z] -> SLL
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
            ],
        ),
        // HWSLC[X + 256*Y, Z] -> SLLC
        BusInteraction::receiver(
            BusId::Hwslc,
            Multiplicity::Column(cols::MU_HWSLC),
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
                    start_column: cols::SLLC,
                    packing: Packing::Direct,
                },
            ],
        ),
    ]
}
