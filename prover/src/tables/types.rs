//! Core types for the 64-bit VM prover tables.
//!
//! This module defines the bus IDs, field types, and shared structures used across all 64-bit tables.
//!
//! ## Field Choice
//!
//! For the 64-bit VM prover, we use the Goldilocks field:
//! - Prime: p = 2^64 - 2^32 + 1
//! - Two-adicity: 32 (supports FFT up to 2^32 rows)
//! - Extension: Degree 3 (cubic extension with w³ = 2, provides 192-bit security)
//!
//! ## DecodeEntry
//!
//! The `DecodeEntry` struct represents decoded instruction information shared between
//! the CPU and DECODE tables. It contains all static decode-time information extracted
//! from an instruction, excluding runtime values like register contents.

use executor::vm::instruction::decoding::{ArithOp, Comparison, Instruction, LoadStoreWidth};
use math::field::element::FieldElement;
use math::field::extensions_goldilocks::Degree3GoldilocksExtensionField;
use math::field::goldilocks::GoldilocksField as GoldilocksBaseField;
use stark::table::Table;

/// Base field type: Goldilocks prime field (p = 2^64 - 2^32 + 1)
pub type GoldilocksField = GoldilocksBaseField;

/// Extension field type: Degree 3 extension of Goldilocks (w³ = 2)
pub type GoldilocksExtension = Degree3GoldilocksExtensionField;

/// Field element in the base Goldilocks field
pub type FE = FieldElement<GoldilocksField>;

/// Field element in the Goldilocks extension field
pub type FEE = FieldElement<GoldilocksExtension>;

/// Decompose a `u64` into its two little-endian 32-bit limbs as field elements:
/// `[x[0..32], x[32..64]]` (the `DWordWL` column encoding).
///
/// Lives in the prover (not the generic `Table`) because the decomposition is
/// field-size-specific: a 32-bit limb only fits because Goldilocks is ~64-bit.
#[inline]
pub fn dword_wl(x: u64) -> [FE; 2] {
    [FE::from(x & 0xFFFF_FFFF), FE::from(x >> 32)]
}

/// Allocate a zeroed `Vec<FE>` via the allocator's calloc path (demand-zeroed OS
/// pages, touched lazily on first write) instead of the element-wise clone loop
/// that `vec![FE::zero(); n]` runs. Trace fill is memory-bandwidth-bound, so
/// skipping the eager zeroing sweep of the (multi-GB) trace is a real win.
///
/// Sound because `FE` is `#[repr(transparent)]` over its `u64` base value
/// (Goldilocks has no Montgomery form), so `FE`'s canonical zero is the all-zero
/// bit pattern — identical to `0u64`. The `zeroed_fe_vec_matches_fe_zero` test
/// guards this invariant. Technique borrowed from SP1's `zeroed_f_vec`.
#[inline]
pub fn zeroed_fe_vec(len: usize) -> Vec<FE> {
    const _: () = assert!(core::mem::size_of::<FE>() == core::mem::size_of::<u64>());
    const _: () = assert!(core::mem::align_of::<FE>() == core::mem::align_of::<u64>());
    let zeros: Vec<u64> = vec![0u64; len];
    // Reinterpret the buffer as `Vec<FE>` via its raw parts rather than
    // `mem::transmute::<Vec<u64>, Vec<FE>>`. `Vec`'s field layout is unspecified
    // and may depend on its element type, so transmuting one `Vec` to another
    // relies on that unspecified layout (the std `mem::transmute` docs call this
    // out and recommend `from_raw_parts`). Rebuilding from `(ptr, len, cap)`
    // reuses the same allocation and carries no `Vec`-layout assumption.
    let mut zeros = core::mem::ManuallyDrop::new(zeros);
    // SAFETY: `FE` is `#[repr(transparent)]` over `u64` with identical size and
    // alignment (asserted above), and `0u64` is exactly `FE::zero()`'s bit
    // pattern (Goldilocks has no Montgomery form), so the zeroed `u64` buffer is
    // a valid `[FE]` of all `FE::zero()`. `len`/`capacity` are element counts and
    // the element sizes are equal, so they carry over unchanged; the eventual
    // dealloc uses the same `size * capacity` and alignment as the original
    // allocation. `ManuallyDrop` stops the source `Vec` from freeing the buffer
    // that the returned `Vec` now owns.
    unsafe { Vec::from_raw_parts(zeros.as_mut_ptr() as *mut FE, zeros.len(), zeros.capacity()) }
}

#[cfg(test)]
mod zeroed_fe_vec_tests {
    use super::*;

    /// Guards the `zeroed_fe_vec` invariant: a calloc'd all-zero buffer
    /// reinterpreted as `Vec<FE>` must equal an element-wise `FE::zero()` fill.
    #[test]
    fn zeroed_fe_vec_matches_fe_zero() {
        for len in [0usize, 1, 7, 64, 1024] {
            assert_eq!(zeroed_fe_vec(len), vec![FE::zero(); len], "len={len}");
        }
    }
}

/// Decompose a `u64` into its four little-endian 16-bit limbs as field elements:
/// `[x[0..16], x[16..32], x[32..48], x[48..64]]` (the `DWordHL` column encoding).
#[inline]
pub fn dword_hl(x: u64) -> [FE; 4] {
    [
        FE::from(x & 0xFFFF),
        FE::from((x >> 16) & 0xFFFF),
        FE::from((x >> 32) & 0xFFFF),
        FE::from((x >> 48) & 0xFFFF),
    ]
}

/// VM-specific trace writes for Goldilocks-backed tables.
///
/// These helpers live at the VM prover layer because encodings like `DWordWL`
/// assume field elements can faithfully represent the corresponding limbs.
///
/// Width names follow the VM table specs:
/// - `Byte`: 8 bits.
/// - `Half`: 16 bits.
/// - `Word`: 32 bits.
/// - `DWord`: 64 bits.
///
/// Trace columns are written in little-endian order: `start_col` always receives
/// the least-significant chunk. For homogeneous encodings like `DWordWL`,
/// `DWordHL`, and `DWordBL`, the final `L` means "little-endian limbs". Mixed
/// encodings like `DWordWHH` and `DWordHHW` keep the spec's packing name, while
/// still storing the low chunk first in the trace.
pub trait VmTable {
    /// Write an already-constructed field element into one trace cell.
    fn set_fe(&mut self, row: usize, col: usize, value: FE);

    /// Convert `value` with `FE::from` and write it into one trace cell.
    #[inline]
    fn set_u64(&mut self, row: usize, col: usize, value: u64) {
        self.set_fe(row, col, FE::from(value));
    }

    /// Write a bit column as `0` or `1`.
    #[inline]
    fn set_bool(&mut self, row: usize, col: usize, value: bool) {
        self.set_u64(row, col, u64::from(value));
    }

    /// Write an 8-bit VM `Byte` column.
    #[inline]
    fn set_byte(&mut self, row: usize, col: usize, value: u8) {
        self.set_u64(row, col, u64::from(value));
    }

    /// Write a 16-bit VM `Half` column.
    #[inline]
    fn set_half(&mut self, row: usize, col: usize, value: u16) {
        self.set_u64(row, col, u64::from(value));
    }

    /// Write a 32-bit VM `Word` column.
    #[inline]
    fn set_word(&mut self, row: usize, col: usize, value: u32) {
        self.set_u64(row, col, u64::from(value));
    }

    /// Write contiguous `Byte` columns starting at `start_col`.
    #[inline]
    fn set_bytes(&mut self, row: usize, start_col: usize, values: &[u8]) {
        for (offset, &value) in values.iter().enumerate() {
            self.set_byte(row, start_col + offset, value);
        }
    }

    /// Write contiguous `Half` columns starting at `start_col`.
    #[inline]
    fn set_halves(&mut self, row: usize, start_col: usize, values: &[u16]) {
        for (offset, &value) in values.iter().enumerate() {
            self.set_half(row, start_col + offset, value);
        }
    }

    /// Write contiguous `Word` columns starting at `start_col`.
    #[inline]
    fn set_words(&mut self, row: usize, start_col: usize, values: &[u32]) {
        for (offset, &value) in values.iter().enumerate() {
            self.set_word(row, start_col + offset, value);
        }
    }

    /// Write a `DWordBL`: eight little-endian bytes of a 64-bit value.
    ///
    /// Columns receive bits `[0..8]`, `[8..16]`, ..., `[56..64]`.
    #[inline]
    fn set_dword_bl(&mut self, row: usize, start_col: usize, value: u64) {
        self.set_bytes(row, start_col, &value.to_le_bytes());
    }

    /// Write a `DWordWL`: two little-endian 32-bit words of a 64-bit value.
    ///
    /// Columns receive bits `[0..32]` and `[32..64]`.
    #[inline]
    fn set_dword_wl(&mut self, row: usize, start_col: usize, value: u64) {
        let [lo, hi] = dword_wl(value);
        self.set_fe(row, start_col, lo);
        self.set_fe(row, start_col + 1, hi);
    }

    /// Write a `DWordHL`: four little-endian 16-bit halves of a 64-bit value.
    ///
    /// Columns receive bits `[0..16]`, `[16..32]`, `[32..48]`, and `[48..64]`.
    #[inline]
    fn set_dword_hl(&mut self, row: usize, start_col: usize, value: u64) {
        let [h0, h1, h2, h3] = dword_hl(value);
        self.set_fe(row, start_col, h0);
        self.set_fe(row, start_col + 1, h1);
        self.set_fe(row, start_col + 2, h2);
        self.set_fe(row, start_col + 3, h3);
    }

    /// Write a mixed `DWordWHH` layout.
    ///
    /// The spec name describes a 64-bit value split as a high `Word` followed by
    /// two lower `Half`s. Trace columns are still low-first, so they receive bits
    /// `[0..16]`, `[16..32]`, and `[32..64]`.
    #[inline]
    fn set_dword_whh(&mut self, row: usize, start_col: usize, value: u64) {
        let low_half = (value & 0xFFFF) as u16;
        let mid_half = ((value >> 16) & 0xFFFF) as u16;
        let high_word = ((value >> 32) & 0xFFFF_FFFF) as u32;

        self.set_half(row, start_col, low_half);
        self.set_half(row, start_col + 1, mid_half);
        self.set_word(row, start_col + 2, high_word);
    }

    /// Write a mixed `DWordHHW` layout.
    ///
    /// The spec name describes a 64-bit value split as two high `Half`s followed
    /// by a low `Word`. Trace columns are still low-first, so they receive bits
    /// `[0..32]`, `[32..48]`, and `[48..64]`.
    #[inline]
    fn set_dword_hhw(&mut self, row: usize, start_col: usize, value: u64) {
        let low_word = (value & 0xFFFF_FFFF) as u32;
        let mid_half = ((value >> 32) & 0xFFFF) as u16;
        let high_half = ((value >> 48) & 0xFFFF) as u16;

        self.set_word(row, start_col, low_word);
        self.set_half(row, start_col + 1, mid_half);
        self.set_half(row, start_col + 2, high_half);
    }
}

impl VmTable for Table<GoldilocksField> {
    #[inline]
    fn set_fe(&mut self, row: usize, col: usize, value: FE) {
        self.set(row, col, value);
    }
}

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
    /// `ARE_BYTES[X, Y]`: range check that both X and Y are valid bytes [0, 256).
    /// Single-byte checks (spec template `IS_BYTE<X>`) send the second value as 0.
    AreBytes = 0,
    /// Range check: value is a valid halfword [0, 2^16)
    IsHalfword = 1,
    /// Range check: value is a 20-bit value [0, 2^20)
    IsB20 = 2,

    // =========================================================================
    // Bitwise operations (BITWISE table provides)
    // =========================================================================
    // IDs 3, 4, and 5 are reserved for the removed legacy
    // AndByte/OrByte/XorByte buses. Byte AND/OR/XOR lookups use ByteAlu.
    /// Most significant bit of a byte: MSB8[X] -> (X >> 7) & 1
    Msb8 = 6,
    /// Most significant bit of a halfword: MSB16[X] -> (X >> 15) & 1
    Msb16 = 7,
    /// Check if value is zero: ZERO[X] -> X == 0 ? 1 : 0
    Zero = 8,

    // =========================================================================
    // Shift helpers (BITWISE table provides)
    // =========================================================================
    /// Halfword shift left: HWSL[X, Z] -> [(X << Z) & 0xFFFF, X >> (16 - Z)]
    Hwsl = 9,

    // =========================================================================
    // Arithmetic operations (separate tables)
    // =========================================================================
    // The four per-chip ALU buses (LT, MUL, DVRM, SHIFT — IDs 10/11/12/13)
    // are collapsed into [`Alu`](BusId::Alu). Their numeric IDs are reserved
    // (not removed) so the live variants below keep their discriminants stable.

    // =========================================================================
    // Memory/Control
    // =========================================================================
    /// Memory word read/write with timestamps (lookup bus from CPU)
    Memw = 14,
    // ID 15 (Load) is reserved: the load lookup is now dispatched through
    // [`MemoryOp`](BusId::MemoryOp).
    /// Internal memory consistency bus: memory[is_register, address, timestamp, value]
    /// Used for read/write pairing in MEMW table (M1-M8 in spec)
    Memory = 16,
    /// Branch target computation
    Branch = 17,

    // =========================================================================
    // System (specs not yet defined)
    // =========================================================================
    /// Instruction decode lookup
    Decode = 18,
    /// System call handling (CPU → HALT/COMMIT for all ECALLs)
    Ecall = 19,
    /// COMMIT self-referencing recursive bus (row N → row N+1)
    CommitNextByte = 20,
    /// COMMIT output bus: verifier computes the receiver contribution externally
    /// from `VmProof.public_output` using the shared LogUp challenges
    Commit = 21,
    /// Keccak core ↔ round chip: (timestamp, round, state[200 bytes])
    Keccak = 22,
    /// Keccak round ↔ RC lookup: (round, rc[8 bytes])
    KeccakRc = 23,

    // =========================================================================
    // Byte ALU (BITWISE table provides)
    // =========================================================================
    /// Unified byte-level ALU lookup: `BYTE_ALU[opsel, X, Y] -> out`, where
    /// `opsel` is an [`alu_op`] descriptor (AND=0/OR=1/XOR=2).
    ByteAlu = 24,

    // =========================================================================
    // Unified ALU + high-level memory dispatch
    // =========================================================================
    /// Unified ALU lookup: `ALU[out; in1, in2, alu_flags]`. The CPU (sender)
    /// dispatches to the ALU chips (lt/mul/dvrm/shift/eq/bytewise/cpu32) which
    /// receive on this bus, selected by the `alu_flags` byte. Replaces the
    /// per-chip `Lt`/`Mul`/`Dvrm`/`Shift` output buses.
    Alu = 25,
    /// High-level memory op: `MEMORY[out; timestamp, address, value, mem_flags]`.
    /// The CPU (sender) dispatches to `LOAD`/`STORE` based on `mem_flags`.
    /// Distinct from the low-level [`Memory`](BusId::Memory) token bus.
    MemoryOp = 26,
    /// CPU → CPU32 delegation of word (`*W`) instructions:
    /// `CPU32[timestamp, pc, instruction_length]`.
    Cpu32 = 27,

    // =========================================================================
    // EC scalar multiplication accelerator (ECSM / ECDAS)
    // =========================================================================
    /// ECDAS self-referential double/add sequence bus:
    /// (timestamp, xA, yA, xG, yG, round, op). ECSM seeds and drains it.
    Ecdas = 28,
    /// Scalar-bit bus: ECDAS sends Bit[ts, round] per step (mult = next_op);
    /// ECSM receives Bit[ts, i] for each of the 256 k bits (mult = k[i]),
    /// and sends Bit[ts, idx_k] for the MSB (mult = μ).
    Bit = 30,

    // =========================================================================
    // DMA memcpy accelerator
    // =========================================================================
    /// DMA self-referential streaming bus (COMMIT-style): each DMA table row sends
    /// `(timestamp, src_incr, dst_incr, count_decr)` to the next row and receives
    /// `(timestamp, src, dst, count)` from the previous row, chaining a variable-length
    /// copy. Only the first row receives the CPU's `Ecall`; the rest chain here.
    DmaNext = 29,

    /// DMA memset streaming bus: each DMA_SET row sends
    /// `(timestamp, dst_incr, count_decr, fill)` to the next row and receives
    /// `(timestamp, dst, count, fill)` from the previous one. Separate from
    /// [`BusId::DmaNext`] so a memcpy row can never consume a memset token.
    DmaSetNext = 32,

    // =========================================================================
    // Continuations
    // =========================================================================
    /// Cross-epoch memory bus: the local-to-global table's per-cell init/fini
    /// boundary claims, matched across epochs by the final aggregation LogUp.
    GlobalMemory = 31,

    // =========================================================================
    // Unified memmove primitive
    // =========================================================================
    /// MEMMOVE self-referential streaming bus. A row sends
    /// `(timestamp, src_incr, dst_incr, count_decr, is_set, is_commit)` to the next
    /// row and receives `(timestamp, src, dst, count, is_set, is_commit)` from the
    /// previous one. The functionality selectors travel inside the tuple, so a chain
    /// cannot change operation half way through it — the guarantee the three separate
    /// DmaNext/DmaSetNext/CommitNextByte buses used to give structurally.
    MemmoveNext = 33,
    /// COMMIT → MEMMOVE hand-off: COMMIT keeps the `sys_write` ecall number and the
    /// register-254 update, and defers its byte loop here as
    /// `(timestamp, buf_addr, start_index, count)`.
    CommitDefer = 34,
}

impl BusId {
    /// Human-readable name for debug output.
    pub fn name(&self) -> &'static str {
        match self {
            BusId::AreBytes => "AreBytes",
            BusId::IsHalfword => "IsHalfword",
            BusId::IsB20 => "IsB20",
            BusId::Msb8 => "Msb8",
            BusId::Msb16 => "Msb16",
            BusId::Zero => "Zero",
            BusId::Hwsl => "Hwsl",
            BusId::Memw => "Memw",
            BusId::Memory => "Memory",
            BusId::Branch => "Branch",
            BusId::Decode => "Decode",
            BusId::Ecall => "Ecall",
            BusId::CommitNextByte => "CommitNextByte",
            BusId::Commit => "Commit",
            BusId::MemmoveNext => "MemmoveNext",
            BusId::CommitDefer => "CommitDefer",
            BusId::Keccak => "Keccak",
            BusId::KeccakRc => "KeccakRc",
            BusId::ByteAlu => "ByteAlu",
            BusId::Alu => "Alu",
            BusId::MemoryOp => "MemoryOp",
            BusId::Cpu32 => "Cpu32",
            BusId::Ecdas => "Ecdas",
            BusId::Bit => "Bit",
            BusId::DmaNext => "DmaNext",
            BusId::DmaSetNext => "DmaSetNext",
            BusId::GlobalMemory => "GlobalMemory",
        }
    }
}

impl TryFrom<u64> for BusId {
    type Error = u64;

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(BusId::AreBytes),
            1 => Ok(BusId::IsHalfword),
            2 => Ok(BusId::IsB20),
            6 => Ok(BusId::Msb8),
            7 => Ok(BusId::Msb16),
            8 => Ok(BusId::Zero),
            9 => Ok(BusId::Hwsl),
            14 => Ok(BusId::Memw),
            16 => Ok(BusId::Memory),
            17 => Ok(BusId::Branch),
            18 => Ok(BusId::Decode),
            19 => Ok(BusId::Ecall),
            20 => Ok(BusId::CommitNextByte),
            21 => Ok(BusId::Commit),
            33 => Ok(BusId::MemmoveNext),
            34 => Ok(BusId::CommitDefer),
            22 => Ok(BusId::Keccak),
            23 => Ok(BusId::KeccakRc),
            24 => Ok(BusId::ByteAlu),
            25 => Ok(BusId::Alu),
            26 => Ok(BusId::MemoryOp),
            27 => Ok(BusId::Cpu32),
            28 => Ok(BusId::Ecdas),
            29 => Ok(BusId::DmaNext),
            32 => Ok(BusId::DmaSetNext),
            30 => Ok(BusId::Bit),
            31 => Ok(BusId::GlobalMemory),
            other => Err(other),
        }
    }
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

// Field inverses for Goldilocks prime p = 2^64 - 2^32 + 1
// Used for virtual carry computation in MUL table
//
// Only the constants actually used in mul.rs are defined here:
// - Positive: INV_2_32, INV_2_64, INV_2_96, INV_2_128 (for raw_product terms)
// - Negative: NEG_INV_2_* (for lo/hi terms which are subtracted)

/// 2^(-32) mod p
pub const INV_2_32: u64 = 18446744065119617026;
/// 2^(-64) mod p
pub const INV_2_64: u64 = 18446744065119617025;
/// 2^(-96) mod p
pub const INV_2_96: u64 = 18446744069414584320;
/// 2^(-128) mod p
pub const INV_2_128: u64 = 4294967295;

/// -(2^-16) mod p = p - 2^(-16)
pub const NEG_INV_2_16: u64 = 281474976645120;
/// -(2^-32) mod p = p - 2^(-32)
pub const NEG_INV_2_32: u64 = 4294967295;
/// -(2^-48) mod p = p - 2^(-48)
pub const NEG_INV_2_48: u64 = 281474976710656;
/// -(2^-64) mod p = p - 2^(-64)
pub const NEG_INV_2_64: u64 = 4294967296;
/// -(2^-80) mod p = p - 2^(-80)
pub const NEG_INV_2_80: u64 = 65536;
/// -(2^-96) mod p = p - 2^(-96)
pub const NEG_INV_2_96: u64 = 1;
/// -(2^-112) mod p = p - 2^(-112)
pub const NEG_INV_2_112: u64 = 18446462594437939201;
/// -(2^-128) mod p = p - 2^(-128)
pub const NEG_INV_2_128: u64 = 18446744065119617026;

// =========================================================================
// ALU operation descriptors
// =========================================================================

/// Numerical descriptors for ALU operations, per `spec/decode.typ`.
///
/// These values are the single source of truth for:
/// - the `opsel` selector of the [`BusId::ByteAlu`] lookup (AND/OR/XOR), and
/// - the low 5 bits (`alu_op`) of the packed `alu_flags` byte consumed by the
///   unified `ALU` bus and the ALU chips (shift/lt/mul/dvrm).
pub mod alu_op {
    pub const AND: u8 = 0;
    pub const OR: u8 = 1;
    pub const XOR: u8 = 2;
    pub const EQ: u8 = 3;
    pub const LT: u8 = 4;
    pub const SHIFT: u8 = 5;
    pub const SHIFTW: u8 = 6;
    pub const MUL: u8 = 7;
    pub const DIVREM: u8 = 8;
}

// =========================================================================
// packed_decode layout
// =========================================================================

/// Bit layout of the shrunk `packed_decode` field (58 bits used), per
/// `cpu.toml:184-205` and `decode_uncompressed.toml`.
///
/// This is the single source of truth shared by the DECODE-table producer and
/// the CPU's `packed_decode` reconstruction, so the DECODE bus fingerprint
/// matches on both sides.
///
pub mod packed_decode_shrunk {
    // Top-level flags + register indices.
    pub const READ_REG1: u32 = 0;
    pub const READ_REG2: u32 = 1;
    pub const WRITE_REG: u32 = 2;
    pub const WORD_INSTR: u32 = 3;
    pub const ALU: u32 = 4;
    pub const ADD: u32 = 5;
    pub const SUB: u32 = 6;
    pub const MEMORY: u32 = 7;
    pub const BRANCH: u32 = 8;
    pub const ECALL: u32 = 9;
    pub const RS1: u32 = 10;
    pub const RS2: u32 = 18;
    pub const RD: u32 = 26;
    /// `half_instruction_length`: bytes/2 (1 for C-type, 2 for regular). The
    /// half-encoding makes odd (misaligned) instruction lengths unrepresentable
    /// (`spec/src/cpu.toml`).
    pub const HALF_INSTRUCTION_LENGTH: u32 = 34;
    pub const ALU_FLAGS: u32 = 42;
    pub const MEM_FLAGS: u32 = 50;

    // `alu_flags` byte interior: bits 0-4 are the `alu_op` descriptor
    // (see [`super::alu_op`]); the high bits are flags.
    pub const ALU_FLAGS_OP_MASK: u8 = 0x1F;
    pub const ALU_FLAGS_SIGNED: u32 = 5;
    /// `signed2` (MUL) and `invert` (SHIFT/EQ/LT) are mutually exclusive and
    /// share this bit (`64·(signed2 + invert)` in `decode_uncompressed.toml`).
    pub const ALU_FLAGS_SIGNED2_OR_INVERT: u32 = 6;
    pub const ALU_FLAGS_MULDIV: u32 = 7;

    // `mem_flags` byte interior. Bit 0 aliases `JALR` (under BRANCH) and
    // `memory_op` (0=LOAD/1=STORE, under MEMORY); the two are mutually exclusive.
    pub const MEM_FLAGS_JALR_OR_OP: u32 = 0;
    pub const MEM_FLAGS_SIGNED: u32 = 1;
    pub const MEM_FLAGS_2B: u32 = 2;
    pub const MEM_FLAGS_4B: u32 = 3;
    pub const MEM_FLAGS_8B: u32 = 4;
}

/// Build the `alu_flags` byte: `alu_op + 32·signed + 64·(signed2|invert) + 128·muldiv`.
pub fn build_alu_flags(alu_op: u8, signed: bool, signed2_or_invert: bool, muldiv: bool) -> u8 {
    use packed_decode_shrunk as b;
    debug_assert!(alu_op <= b::ALU_FLAGS_OP_MASK, "alu_op must fit in 5 bits");
    alu_op
        | ((signed as u8) << b::ALU_FLAGS_SIGNED)
        | ((signed2_or_invert as u8) << b::ALU_FLAGS_SIGNED2_OR_INVERT)
        | ((muldiv as u8) << b::ALU_FLAGS_MULDIV)
}

/// Build the `mem_flags` byte: `jalr_or_op + 2·mem_signed + 4·mem_2B + 8·mem_4B + 16·mem_8B`.
pub fn build_mem_flags(
    jalr_or_memory_op: bool,
    mem_signed: bool,
    mem_2b: bool,
    mem_4b: bool,
    mem_8b: bool,
) -> u8 {
    use packed_decode_shrunk as b;
    ((jalr_or_memory_op as u8) << b::MEM_FLAGS_JALR_OR_OP)
        | ((mem_signed as u8) << b::MEM_FLAGS_SIGNED)
        | ((mem_2b as u8) << b::MEM_FLAGS_2B)
        | ((mem_4b as u8) << b::MEM_FLAGS_4B)
        | ((mem_8b as u8) << b::MEM_FLAGS_8B)
}

/// Logical (unpacked) view of the reworked `packed_decode` field. `alu_flags`
/// and `mem_flags` are stored already-packed (build them with
/// [`build_alu_flags`] / [`build_mem_flags`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct ShrunkDecode {
    pub read_register1: bool,
    pub read_register2: bool,
    pub write_register: bool,
    pub word_instr: bool,
    pub alu: bool,
    pub add: bool,
    pub sub: bool,
    pub memory: bool,
    pub branch: bool,
    pub ecall: bool,
    pub rs1: u8,
    pub rs2: u8,
    pub rd: u8,
    /// Half the byte length of the instruction (1 for C-type, 2 for regular);
    /// the real length is `2 * half_instruction_length`.
    pub half_instruction_length: u8,
    pub alu_flags: u8,
    pub mem_flags: u8,
}

impl ShrunkDecode {
    /// Pack into the 58-bit `packed_decode` field value.
    pub fn pack(&self) -> u64 {
        use packed_decode_shrunk as b;
        ((self.read_register1 as u64) << b::READ_REG1)
            | ((self.read_register2 as u64) << b::READ_REG2)
            | ((self.write_register as u64) << b::WRITE_REG)
            | ((self.word_instr as u64) << b::WORD_INSTR)
            | ((self.alu as u64) << b::ALU)
            | ((self.add as u64) << b::ADD)
            | ((self.sub as u64) << b::SUB)
            | ((self.memory as u64) << b::MEMORY)
            | ((self.branch as u64) << b::BRANCH)
            | ((self.ecall as u64) << b::ECALL)
            | ((self.rs1 as u64) << b::RS1)
            | ((self.rs2 as u64) << b::RS2)
            | ((self.rd as u64) << b::RD)
            | ((self.half_instruction_length as u64) << b::HALF_INSTRUCTION_LENGTH)
            | ((self.alu_flags as u64) << b::ALU_FLAGS)
            | ((self.mem_flags as u64) << b::MEM_FLAGS)
    }

    /// Inverse of [`pack`](Self::pack).
    pub fn unpack(packed: u64) -> Self {
        use packed_decode_shrunk as b;
        let bit = |pos: u32| (packed >> pos) & 1 == 1;
        let byte = |pos: u32| ((packed >> pos) & 0xFF) as u8;
        Self {
            read_register1: bit(b::READ_REG1),
            read_register2: bit(b::READ_REG2),
            write_register: bit(b::WRITE_REG),
            word_instr: bit(b::WORD_INSTR),
            alu: bit(b::ALU),
            add: bit(b::ADD),
            sub: bit(b::SUB),
            memory: bit(b::MEMORY),
            branch: bit(b::BRANCH),
            ecall: bit(b::ECALL),
            rs1: byte(b::RS1),
            rs2: byte(b::RS2),
            rd: byte(b::RD),
            half_instruction_length: byte(b::HALF_INSTRUCTION_LENGTH),
            alu_flags: byte(b::ALU_FLAGS),
            mem_flags: byte(b::MEM_FLAGS),
        }
    }

    /// Build the reworked packed-decode flags for an instruction, per
    /// `spec/decode.typ`. Does NOT include `pc`/`imm` (separate DECODE columns).
    ///
    /// `instruction_length` is the byte length: 2 (RV64C compressed) or 4. It is
    /// stored as `half_instruction_length = instruction_length / 2`; the real
    /// length is recovered as `2 * half_instruction_length`.
    ///
    /// Per `spec/decode.typ`: conditional branches set
    /// `BRANCH=1 ∧ ALU=1` (the EQ/LT chip computes the comparison; `BRANCH`
    /// selects `arg2 = rv2`). JAL/JALR set `BRANCH=1 ∧ JALR=1` with no ALU op —
    /// the return address `pc + instruction_length` is written to `rvd` by the
    /// CPU branch group, not the ALU.
    pub fn from_instruction(instruction: Instruction, instruction_length: u8) -> Self {
        debug_assert!(
            instruction_length.is_multiple_of(2),
            "instruction_length must be even (RISC-V instructions are 2 or 4 bytes)"
        );
        let mut d = Self {
            half_instruction_length: instruction_length / 2,
            ..Default::default()
        };
        match instruction {
            Instruction::Arith {
                dst,
                src1,
                src2,
                op,
            } => {
                d.rd = dst as u8;
                d.rs1 = src1 as u8;
                d.rs2 = src2 as u8;
                d.read_register1 = src1 != 0;
                d.read_register2 = src2 != 0;
                d.write_register = dst != 0;
                d.apply_arith_op(op, false);
            }
            Instruction::ArithImm { dst, src, op, .. } => {
                d.rd = dst as u8;
                d.rs1 = src as u8;
                d.read_register1 = src != 0;
                d.write_register = dst != 0;
                d.apply_arith_op(op, false);
            }
            Instruction::ArithW {
                dst,
                src1,
                src2,
                op,
            } => {
                d.rd = dst as u8;
                d.rs1 = src1 as u8;
                d.rs2 = src2 as u8;
                d.read_register1 = src1 != 0;
                d.read_register2 = src2 != 0;
                d.write_register = dst != 0;
                d.word_instr = true;
                d.apply_arith_op(op, true);
            }
            Instruction::ArithImmW { dst, src, op, .. } => {
                d.rd = dst as u8;
                d.rs1 = src as u8;
                d.read_register1 = src != 0;
                d.write_register = dst != 0;
                d.word_instr = true;
                d.apply_arith_op(op, true);
            }
            // JAL is represented as JALR rd, x255, imm (x255 holds pc).
            Instruction::JumpAndLink { dst, .. } => {
                d.rd = dst as u8;
                d.rs1 = 255;
                d.read_register1 = true;
                d.write_register = dst != 0;
                d.branch = true;
                d.mem_flags = build_mem_flags(true, false, false, false, false); // JALR bit
            }
            Instruction::JumpAndLinkRegister { base, dst, .. } => {
                d.rd = dst as u8;
                d.rs1 = base as u8;
                d.read_register1 = base != 0;
                d.write_register = dst != 0;
                d.branch = true;
                d.mem_flags = build_mem_flags(true, false, false, false, false); // JALR bit
            }
            Instruction::Store {
                src, base, width, ..
            } => {
                d.rs1 = base as u8;
                d.rs2 = src as u8;
                d.read_register1 = base != 0;
                d.read_register2 = src != 0;
                d.add = true; // address = rv1 + imm
                d.memory = true;
                let (m2, m4, m8) = store_width_bits(width);
                d.mem_flags = build_mem_flags(true, false, m2, m4, m8); // memory_op = store
            }
            Instruction::Load {
                dst, base, width, ..
            } => {
                d.rd = dst as u8;
                d.rs1 = base as u8;
                d.read_register1 = base != 0;
                d.write_register = dst != 0;
                d.add = true; // address = rv1 + imm
                d.memory = true;
                let (m2, m4, m8, signed) = load_width_bits(width);
                d.mem_flags = build_mem_flags(false, signed, m2, m4, m8); // memory_op = load
            }
            Instruction::Branch {
                src1, src2, cond, ..
            } => {
                d.rs1 = src1 as u8;
                d.rs2 = src2 as u8;
                d.read_register1 = src1 != 0;
                d.read_register2 = src2 != 0;
                d.branch = true;
                d.alu = true; // Q3: conditional branches go through the EQ/LT ALU chip
                let (op, signed, invert) = branch_cond_flags(cond);
                d.alu_flags = build_alu_flags(op, signed, invert, false);
            }
            // LUI is represented as ADDI rd, x0, imm.
            Instruction::LoadUpperImm { dst, .. } => {
                d.rd = dst as u8;
                d.write_register = dst != 0;
                d.add = true;
            }
            // AUIPC is represented as ADDI rd, x255, imm (x255 holds pc).
            Instruction::AddUpperImmToPc { dst, .. } => {
                d.rd = dst as u8;
                d.rs1 = 255;
                d.read_register1 = true;
                d.write_register = dst != 0;
                d.add = true;
            }
            Instruction::EcallEbreak => {
                d.rs1 = 17; // a7 holds the syscall number
                d.read_register1 = true;
                d.ecall = true;
            }
            // FENCE and CSR are treated as no-ops (ADDI x0, x0, 0).
            Instruction::Fence | Instruction::CSR { .. } => {
                d.add = true;
            }
        }
        d
    }

    /// Set the `ADD`/`SUB`/`ALU` flags and `alu_flags` byte for an `ArithOp`,
    /// per `spec/decode.typ`. `ADD`/`SUB` are fast-paths (ALU not set).
    fn apply_arith_op(&mut self, op: ArithOp, word_instr: bool) {
        let shift = if word_instr {
            alu_op::SHIFTW
        } else {
            alu_op::SHIFT
        };
        // (alu_op, signed, signed2|invert, muldiv, is_add, is_sub)
        let (alu, signed, s2_or_inv, muldiv, is_add, is_sub) = match op {
            ArithOp::Add => (0, false, false, false, true, false),
            ArithOp::Sub => (0, false, false, false, false, true),
            ArithOp::And => (alu_op::AND, false, false, false, false, false),
            ArithOp::Or => (alu_op::OR, false, false, false, false, false),
            ArithOp::Xor => (alu_op::XOR, false, false, false, false, false),
            ArithOp::ShiftLeftLogical => (shift, false, false, false, false, false),
            ArithOp::ShiftRightLogical => (shift, false, true, false, false, false), // invert = right
            ArithOp::ShiftRightArith => (shift, true, true, false, false, false),
            ArithOp::SetLessThan => (alu_op::LT, true, false, false, false, false),
            ArithOp::SetLessThanU => (alu_op::LT, false, false, false, false, false),
            ArithOp::Mul => (alu_op::MUL, true, true, false, false, false),
            ArithOp::MulHigh => (alu_op::MUL, true, true, true, false, false),
            ArithOp::MulHighSignedUnsigned => (alu_op::MUL, true, false, true, false, false),
            ArithOp::MulHighUnsigned => (alu_op::MUL, false, false, true, false, false),
            ArithOp::Div => (alu_op::DIVREM, true, false, false, false, false),
            ArithOp::DivUnsigned => (alu_op::DIVREM, false, false, false, false, false),
            ArithOp::Remainder => (alu_op::DIVREM, true, false, true, false, false),
            ArithOp::RemainderUnsigned => (alu_op::DIVREM, false, false, true, false, false),
        };
        self.add = is_add;
        self.sub = is_sub;
        self.alu = !(is_add || is_sub);
        self.alu_flags = build_alu_flags(alu, signed, s2_or_inv, muldiv);
    }

    // ---- packed `alu_flags` accessors ----

    /// The `alu_op` descriptor (bits 0-4 of `alu_flags`).
    #[inline]
    pub fn alu_op(&self) -> u8 {
        self.alu_flags & packed_decode_shrunk::ALU_FLAGS_OP_MASK
    }
    /// `signed` flag (bit 5 of `alu_flags`).
    #[inline]
    pub fn alu_signed(&self) -> bool {
        (self.alu_flags >> packed_decode_shrunk::ALU_FLAGS_SIGNED) & 1 == 1
    }
    /// Shared `signed2`/`invert` flag (bit 6 of `alu_flags`); meaning depends on
    /// `alu_op` (MUL: `signed2`; SHIFT/EQ/LT: `invert`).
    #[inline]
    pub fn alu_signed2_or_invert(&self) -> bool {
        (self.alu_flags >> packed_decode_shrunk::ALU_FLAGS_SIGNED2_OR_INVERT) & 1 == 1
    }
    /// `muldiv_selector` flag (bit 7 of `alu_flags`).
    #[inline]
    pub fn alu_muldiv(&self) -> bool {
        (self.alu_flags >> packed_decode_shrunk::ALU_FLAGS_MULDIV) & 1 == 1
    }

    // ---- packed `mem_flags` accessors (valid under `memory`/`branch`) ----

    /// Virtual `JALR` bit (bit 0 of `mem_flags`); valid under `branch`.
    #[inline]
    pub fn jalr(&self) -> bool {
        self.mem_flags & 1 == 1
    }
    /// STORE (vs LOAD) when `memory`: `memory_op` is bit 0 of `mem_flags`.
    #[inline]
    pub fn is_store(&self) -> bool {
        self.memory && (self.mem_flags & 1 == 1)
    }
    /// LOAD (vs STORE) when `memory`.
    #[inline]
    pub fn is_load(&self) -> bool {
        self.memory && (self.mem_flags & 1 == 0)
    }
    /// `mem_signed` flag (bit 1 of `mem_flags`).
    #[inline]
    pub fn mem_signed(&self) -> bool {
        (self.mem_flags >> packed_decode_shrunk::MEM_FLAGS_SIGNED) & 1 == 1
    }
    /// Memory access width in bytes (from the `mem_flags` width bits; default 1).
    #[inline]
    pub fn mem_bytes(&self) -> usize {
        use packed_decode_shrunk as b;
        if (self.mem_flags >> b::MEM_FLAGS_8B) & 1 == 1 {
            8
        } else if (self.mem_flags >> b::MEM_FLAGS_4B) & 1 == 1 {
            4
        } else if (self.mem_flags >> b::MEM_FLAGS_2B) & 1 == 1 {
            2
        } else {
            1
        }
    }

    // ---- ALU operation classifiers (valid only when `alu`) ----

    #[inline]
    pub fn is_and(&self) -> bool {
        self.alu && self.alu_op() == alu_op::AND
    }
    #[inline]
    pub fn is_or(&self) -> bool {
        self.alu && self.alu_op() == alu_op::OR
    }
    #[inline]
    pub fn is_xor(&self) -> bool {
        self.alu && self.alu_op() == alu_op::XOR
    }
    #[inline]
    pub fn is_eq(&self) -> bool {
        self.alu && self.alu_op() == alu_op::EQ
    }
    #[inline]
    pub fn is_lt(&self) -> bool {
        self.alu && self.alu_op() == alu_op::LT
    }
    #[inline]
    pub fn is_shift(&self) -> bool {
        self.alu && matches!(self.alu_op(), x if x == alu_op::SHIFT || x == alu_op::SHIFTW)
    }
    #[inline]
    pub fn is_mul(&self) -> bool {
        self.alu && self.alu_op() == alu_op::MUL
    }
    #[inline]
    pub fn is_divrem(&self) -> bool {
        self.alu && self.alu_op() == alu_op::DIVREM
    }
}

/// Memory-width bits `(mem_2B, mem_4B, mem_8B)` for STORE (1 byte = none set).
fn store_width_bits(width: LoadStoreWidth) -> (bool, bool, bool) {
    match width {
        LoadStoreWidth::Byte | LoadStoreWidth::ByteUnsigned => (false, false, false),
        LoadStoreWidth::Half | LoadStoreWidth::HalfUnsigned => (true, false, false),
        LoadStoreWidth::Word | LoadStoreWidth::WordUnsigned => (false, true, false),
        LoadStoreWidth::DoubleWord => (false, false, true),
    }
}

/// Memory-width bits `(mem_2B, mem_4B, mem_8B, mem_signed)` for LOAD.
/// `mem_signed = ¬[U]`; the full-width `LD` is not sign-extended.
fn load_width_bits(width: LoadStoreWidth) -> (bool, bool, bool, bool) {
    match width {
        LoadStoreWidth::Byte => (false, false, false, true),
        LoadStoreWidth::ByteUnsigned => (false, false, false, false),
        LoadStoreWidth::Half => (true, false, false, true),
        LoadStoreWidth::HalfUnsigned => (true, false, false, false),
        LoadStoreWidth::Word => (false, true, false, true),
        LoadStoreWidth::WordUnsigned => (false, true, false, false),
        LoadStoreWidth::DoubleWord => (false, false, true, false),
    }
}

/// `(alu_op, signed, invert)` for a branch comparison, per `spec/decode.typ`.
fn branch_cond_flags(cond: Comparison) -> (u8, bool, bool) {
    match cond {
        Comparison::Equal => (alu_op::EQ, false, false),
        Comparison::NotEqual => (alu_op::EQ, false, true),
        Comparison::LessThan => (alu_op::LT, true, false),
        Comparison::LessThanUnsigned => (alu_op::LT, false, false),
        Comparison::GreaterOrEqual => (alu_op::LT, true, true),
        Comparison::GreaterOrEqualUnsigned => (alu_op::LT, false, true),
    }
}

// =========================================================================
// DecodeEntry - Shared decode information for CPU and DECODE tables
// =========================================================================

/// A single decoded instruction entry.
///
/// This struct contains all static decode-time information extracted from an instruction.
/// It is shared between the CPU table (which uses it for execution) and the DECODE table
/// (which provides it as a lookup table).
///
/// ## Usage
///
/// - **CPU table**: `CpuOperation` contains a `DecodeEntry` plus runtime values (rv1, rv2, etc.)
/// - **DECODE table**: Stores `DecodeEntry` directly, with multiplicity tracking
///
/// The packed decode layout is defined by [`packed_decode_shrunk`] and produced
/// by [`ShrunkDecode::pack`]; consult those for the bit positions of every flag,
/// the ALU/MEM flag bytes, and the rs1/rs2/rd register indices.
#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct DecodeEntry {
    /// Program counter (64-bit).
    pub pc: u64,
    /// Fully sign-extended 64-bit immediate.
    pub imm: u64,
    /// Packed decode flags + register indices.
    pub fields: ShrunkDecode,
}

impl DecodeEntry {
    /// Creates an empty DecodeEntry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Padding row for the DECODE/CPU tables: an odd PC (never a valid fetch
    /// target, hence unprovable) with all flags zero. Replaces the old
    /// EBREAK-based padding (EBREAK has no decoding in this layout).
    pub fn padding_entry() -> Self {
        Self {
            pc: 1,
            imm: 0,
            fields: ShrunkDecode::default(),
        }
    }

    /// Packs the decode fields into the `packed_decode` field-element value.
    pub fn packed_decode(&self) -> u64 {
        self.fields.pack()
    }

    /// Decode an instruction into `(pc, imm, fields)`. `instruction_length` is
    /// 2 (RV64C compressed) or 4.
    pub fn from_instruction(pc: u64, instruction: Instruction, instruction_length: u8) -> Self {
        Self {
            pc,
            imm: imm_from_instruction(instruction),
            fields: ShrunkDecode::from_instruction(instruction, instruction_length),
        }
    }
}

/// The fully sign-extended 64-bit immediate for an instruction (0 when none).
fn imm_from_instruction(instruction: Instruction) -> u64 {
    match instruction {
        Instruction::ArithImm { imm, .. } | Instruction::ArithImmW { imm, .. } => imm as i64 as u64,
        Instruction::JumpAndLink { offset, .. }
        | Instruction::JumpAndLinkRegister { offset, .. }
        | Instruction::Store { offset, .. }
        | Instruction::Load { offset, .. }
        | Instruction::Branch { offset, .. } => offset as i64 as u64,
        Instruction::LoadUpperImm { imm, .. } | Instruction::AddUpperImmToPc { imm, .. } => {
            (imm as i32) as i64 as u64
        }
        _ => 0,
    }
}
