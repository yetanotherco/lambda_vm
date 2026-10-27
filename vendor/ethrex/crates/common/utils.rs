use crate::H256;
use ethereum_types::U256;
use ethrex_crypto::keccak::keccak_hash;
use hex::FromHexError;

pub const ZERO_U256: U256 = U256([0, 0, 0, 0]);

// RV64IM has no REV8 instruction. LLVM's generic bswap lowering is longer
// than this three-stage exchange of bytes, halfwords, and words. Other targets
// retain the compiler's native endian conversion.
#[inline(always)]
fn word_from_big_endian(value: u64) -> u64 {
    #[cfg(all(
        target_arch = "riscv64",
        target_endian = "little",
        not(target_feature = "zbb")
    ))]
    {
        let mut value = value;
        // SAFETY: this pure register-only operation uses baseline RV64I shifts,
        // masks and ORs; it accesses no memory and preserves the stack.
        unsafe {
            core::arch::asm!(
                "srli {tmp}, {value}, 8",
                "and {tmp}, {tmp}, {mask8}",
                "and {value}, {value}, {mask8}",
                "slli {value}, {value}, 8",
                "or {value}, {value}, {tmp}",
                "srli {tmp}, {value}, 16",
                "and {tmp}, {tmp}, {mask16}",
                "and {value}, {value}, {mask16}",
                "slli {value}, {value}, 16",
                "or {value}, {value}, {tmp}",
                "srli {tmp}, {value}, 32",
                "slli {value}, {value}, 32",
                "or {value}, {value}, {tmp}",
                value = inout(reg) value,
                tmp = out(reg) _,
                mask8 = in(reg) 0x00ff00ff00ff00ffu64,
                mask16 = in(reg) 0x0000ffff0000ffffu64,
                options(pure, nomem, nostack),
            );
        }
        value
    }
    #[cfg(not(all(
        target_arch = "riscv64",
        target_endian = "little",
        not(target_feature = "zbb")
    )))]
    {
        u64::from_be(value)
    }
}

/// Converts a big endian slice to a u256, faster than `u256::from_big_endian`.
pub fn u256_from_big_endian(slice: &[u8]) -> U256 {
    let mut padded = [0u8; 32];
    padded[32 - slice.len()..32].copy_from_slice(slice);

    let mut ret = [0; 4];

    let mut u64_bytes = [0u8; 8];
    for i in 0..4 {
        u64_bytes.copy_from_slice(&padded[8 * i..(8 * i + 8)]);
        ret[4 - i - 1] = word_from_big_endian(u64::from_ne_bytes(u64_bytes));
    }

    U256(ret)
}

/// Converts a constant big endian slice to a u256, faster than `u256::from_big_endian` and `u256_from_big_endian`.
///
/// Note: N should not exceed 32.
pub fn u256_from_big_endian_const<const N: usize>(slice: [u8; N]) -> U256 {
    const { assert!(N <= 32, "N must be less or equal to 32") };

    let mut padded = [0u8; 32];
    padded[32 - N..32].copy_from_slice(&slice);

    let mut ret = [0u64; 4];

    let mut u64_bytes = [0u8; 8];
    for i in 0..4 {
        u64_bytes.copy_from_slice(&padded[8 * i..(8 * i + 8)]);
        ret[4 - i - 1] = if N >= (4 - i) * 8 {
            word_from_big_endian(u64::from_ne_bytes(u64_bytes))
        } else {
            u64::from_be_bytes(u64_bytes)
        };
    }

    U256(ret)
}

/// Converts a U256 to a big endian slice.
#[inline(always)]
pub fn u256_to_big_endian(value: U256) -> [u8; 32] {
    let mut bytes = [0u8; 32];

    for i in 0..4 {
        let u64_be = word_from_big_endian(value.0[4 - i - 1]).to_ne_bytes();
        bytes[8 * i..(8 * i + 8)].copy_from_slice(&u64_be);
    }

    bytes
}

/// Decodes an EVM memory word, avoiding byte permutation for zero limbs.
#[inline(always)]
pub fn u256_from_big_endian_for_memory(value: [u8; 32]) -> U256 {
    let mut limbs = [0u64; 4];
    for (i, chunk) in value.chunks_exact(8).enumerate() {
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(chunk);
        let word = u64::from_ne_bytes(bytes);
        limbs[3 - i] = if cfg!(all(
            target_arch = "riscv64",
            target_endian = "little",
            not(target_feature = "zbb")
        )) && word == 0
        {
            0
        } else {
            word_from_big_endian(word)
        };
    }
    U256(limbs)
}

/// Converts an EVM memory word, avoiding byte permutation for zero limbs.
#[inline(always)]
pub fn u256_to_big_endian_for_memory(value: U256) -> [u8; 32] {
    let mut bytes = [0u8; 32];

    for i in 0..4 {
        let word = value.0[4 - i - 1];
        let reversed = if cfg!(all(
            target_arch = "riscv64",
            target_endian = "little",
            not(target_feature = "zbb")
        )) && word == 0
        {
            0
        } else {
            word_from_big_endian(word)
        };
        let u64_be = reversed.to_ne_bytes();
        bytes[8 * i..(8 * i + 8)].copy_from_slice(&u64_be);
    }

    bytes
}

#[inline(always)]
pub fn u256_to_h256(value: U256) -> H256 {
    H256(u256_to_big_endian(value))
}

pub fn decode_hex(hex: &str) -> Result<Vec<u8>, FromHexError> {
    let trimmed = hex.strip_prefix("0x").unwrap_or(hex);
    hex::decode(trimmed)
}

pub fn keccak(data: impl AsRef<[u8]>) -> H256 {
    H256(keccak_hash(data))
}

// Allocation-free operations on arrays.
///
/// Truncates an array of size N to size M.
/// Fails compilation if N < M.
pub fn truncate_array<const N: usize, const M: usize>(data: [u8; N]) -> [u8; M] {
    const { assert!(M <= N) };
    let mut res = [0u8; M];
    res.copy_from_slice(&data[..M]);
    res
}
