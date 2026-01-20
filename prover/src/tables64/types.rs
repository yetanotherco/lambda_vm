//! Core types for the 64-bit VM prover tables.
//!
//! This module defines the bus IDs and helper types used across all 64-bit tables.
//!
//! ## Field Choice
//!
//! For the 64-bit VM prover, we use the Goldilocks field:
//! - Prime: p = 2^64 - 2^32 + 1
//! - Two-adicity: 32 (supports FFT up to 2^32 rows)
//! - Extension: Degree 3 (cubic extension with w³ = 2, provides 192-bit security)

use math::field::element::FieldElement;
use math::field::fields::fft_friendly::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::fields::fft_friendly::u64_goldilocks::U64GoldilocksPrimeField;

/// Base field type: Goldilocks prime field (p = 2^64 - 2^32 + 1)
pub type GoldilocksField = U64GoldilocksPrimeField;

/// Extension field type: Degree 3 extension of Goldilocks (w³ = 2)
pub type GoldilocksExtension = Degree3GoldilocksExtensionField;

/// Field element in the base Goldilocks field
pub type FE = FieldElement<GoldilocksField>;

/// Field element in the Goldilocks extension field
pub type FEE = FieldElement<GoldilocksExtension>;

/// Bus identifiers for LogUp interactions between tables.
///
/// Each bus connects senders (tables that produce values) with receivers
/// (tables that consume/verify those values). For the bus to balance,
/// the sum of sender multiplicities must equal the sum of receiver multiplicities.
#[repr(u64)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusId {
    // =========================================================================
    // Range checks (BITWISE table provides)
    // =========================================================================
    /// Range check: value is a valid byte [0, 256)
    IsByte = 0,
    /// Range check: value is a valid halfword [0, 2^16)
    IsHalfword,
    /// Range check: value is a 20-bit value [0, 2^20)
    IsB20,

    // =========================================================================
    // Bitwise operations (BITWISE table provides)
    // =========================================================================
    /// Bitwise AND of two bytes: AND_BYTE[X, Y] -> X & Y
    AndByte,
    /// Bitwise OR of two bytes: OR_BYTE[X, Y] -> X | Y
    OrByte,
    /// Bitwise XOR of two bytes: XOR_BYTE[X, Y] -> X ^ Y
    XorByte,
    /// Most significant bit of a byte: MSB8[X] -> (X >> 7) & 1
    Msb8,
    /// Most significant bit of a halfword: MSB16[X] -> (X >> 15) & 1
    Msb16,
    /// Check if value is zero: ZERO[X] -> X == 0 ? 1 : 0
    Zero,

    // =========================================================================
    // Shift helpers (BITWISE table provides)
    // =========================================================================
    /// Halfword shift left: HWSL[X, Z] -> (X << Z) & 0xFFFF
    Hwsl,
    /// Halfword shift left carry: HWSLC[X, Z] -> X >> (16 - Z)
    Hwslc,

    // =========================================================================
    // Arithmetic operations (separate tables)
    // =========================================================================
    /// Less-than comparison: LT[lhs, rhs, signed] -> lhs < rhs
    Lt,
    /// Multiplication: MUL[lhs, lhs_signed, rhs, rhs_signed, hi] -> product
    Mul,
    /// Shift operation: SHIFT[in, shift, dir, signed, word] -> out
    Shift,

    // =========================================================================
    // Memory/Control (separate tables, Phase 5)
    // =========================================================================
    /// Memory word read/write with timestamps
    Memw,
    /// Memory load with sign/zero extension
    Load,
    /// Branch target computation
    Branch,

    // =========================================================================
    // System (specs not yet defined)
    // =========================================================================
    /// Instruction decode lookup
    Decode,
    /// System call handling
    Ecall,
}

impl From<BusId> for u64 {
    fn from(id: BusId) -> u64 {
        id as u64
    }
}

// =========================================================================
// Constants for 64-bit arithmetic
// =========================================================================

/// 2^16 for halfword combining
pub const SHIFT_16: u64 = 1 << 16;

/// 2^32 for word combining
pub const SHIFT_32: u64 = 1 << 32;

/// 2^(-32) mod p for carry extraction in 64-bit addition
/// In Babybear field: p = 2^31 - 2^27 + 1
/// We need to find x such that x * 2^32 ≡ 1 (mod p)
pub const INV_2_32: u64 = {
    // 2^32 mod p = 2^32 mod (2^31 - 2^27 + 1)
    // 2^32 = 2 * 2^31 = 2 * (p + 2^27 - 1) = 2p + 2^28 - 2 ≡ 2^28 - 2 (mod p)
    // We need the inverse of (2^28 - 2) mod p
    // For now, this is a placeholder - compute at runtime or verify
    1
};

/// 2^(-16) mod p for halfword operations
pub const INV_2_16: u64 = {
    // Placeholder - compute at runtime or verify
    1
};

// =========================================================================
// Column index helpers
// =========================================================================

/// Helper to define column ranges for different 64-bit representations.
pub mod columns {
    /// A DWordWL (64-bit as 2 words) spans 2 columns
    pub const DWORD_WL_SIZE: usize = 2;

    /// A DWordHL (64-bit as 4 halves) spans 4 columns
    pub const DWORD_HL_SIZE: usize = 4;

    /// A DWordBL (64-bit as 8 bytes) spans 8 columns
    pub const DWORD_BL_SIZE: usize = 8;

    /// A DWordHHW (64-bit as Word, Half, Half) spans 3 columns
    pub const DWORD_HHW_SIZE: usize = 3;

    /// A QuadWL (128-bit as 4 words) spans 4 columns
    pub const QUAD_WL_SIZE: usize = 4;
}
