//! Tests for the keccak sponge-absorb syscall (`KECCAK_ABSORB_SYSCALL_NUMBER`).
//!
//! The multi-block cases differentially test the executor's absorb loop
//! against an independent sponge replay built on `tiny_keccak::keccakf`.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{
    ExecutionError, KECCAK_ABSORB_SYSCALL_NUMBER, KECCAK_RATE_BYTES,
};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

const STATE_ADDR: u64 = 0x1000;
const DATA_ADDR: u64 = 0x2000;

/// Deterministic SplitMix64 for reproducible "random" data.
struct SplitMix64(u64);
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Runs the absorb syscall over `n_blocks` blocks of deterministic data seeded
/// by `seed`, returning `(vm_state, reference_state)` where the reference is an
/// independent sponge replay over tiny-keccak's permutation.
fn run_absorb_differential(n_blocks: u64, seed: u64) -> ([u64; 25], [u64; 25]) {
    let mut rng = SplitMix64(seed);

    let mut state: [u64; 25] = core::array::from_fn(|i| rng.next_u64() ^ (i as u64));
    let blocks: Vec<[u64; 17]> = (0..n_blocks)
        .map(|_| core::array::from_fn(|_| rng.next_u64()))
        .collect();

    // Set up VM memory.
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();
    for (i, &lane) in state.iter().enumerate() {
        memory
            .store_doubleword(STATE_ADDR + (i as u64) * 8, lane)
            .unwrap();
    }
    for (k, block) in blocks.iter().enumerate() {
        for (j, &dw) in block.iter().enumerate() {
            memory
                .store_doubleword(
                    DATA_ADDR + (k as u64) * KECCAK_RATE_BYTES + (j as u64) * 8,
                    dw,
                )
                .unwrap();
        }
    }
    registers.write(17, KECCAK_ABSORB_SYSCALL_NUMBER).unwrap();
    registers.write(10, STATE_ADDR).unwrap();
    registers.write(11, DATA_ADDR).unwrap();
    registers.write(12, n_blocks).unwrap();

    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .expect("absorb syscall failed");

    let mut vm_state = [0u64; 25];
    for (i, lane) in vm_state.iter_mut().enumerate() {
        *lane = memory.load_doubleword(STATE_ADDR + (i as u64) * 8).unwrap();
    }

    // Independent reference: sponge replay over tiny-keccak's permutation.
    for block in &blocks {
        for (lane, &m) in state.iter_mut().zip(block.iter()) {
            *lane ^= m;
        }
        tiny_keccak::keccakf(&mut state);
    }

    (vm_state, state)
}

#[test]
fn test_absorb_single_block_matches_tiny_keccak() {
    let (vm, reference) = run_absorb_differential(1, 0xA11C_E000_0000_0001);
    assert_eq!(vm, reference, "1-block absorb diverges from tiny-keccak");
}

#[test]
fn test_absorb_two_blocks_matches_tiny_keccak() {
    let (vm, reference) = run_absorb_differential(2, 0xA11C_E000_0000_0002);
    assert_eq!(vm, reference, "2-block absorb diverges from tiny-keccak");
}

#[test]
fn test_absorb_many_blocks_matches_tiny_keccak() {
    for n in [3u64, 5, 8, 13] {
        let (vm, reference) = run_absorb_differential(n, 0xA11C_E000_0000_0100 ^ n);
        assert_eq!(vm, reference, "{n}-block absorb diverges from tiny-keccak");
    }
}

#[test]
fn test_absorb_matches_chained_permute_semantics() {
    // The absorb over n blocks must equal n manual (XOR + keccak_f1600) steps
    // with the executor's own permutation — guards the executor's loop
    // structure independently of tiny-keccak.
    use crate::vm::instruction::execution::keccak_f1600;
    let (vm, _) = run_absorb_differential(4, 0xA11C_E000_0000_0200);

    let mut rng = SplitMix64(0xA11C_E000_0000_0200);
    let mut state: [u64; 25] = core::array::from_fn(|i| rng.next_u64() ^ (i as u64));
    let blocks: Vec<[u64; 17]> = (0..4)
        .map(|_| core::array::from_fn(|_| rng.next_u64()))
        .collect();
    for block in &blocks {
        for (lane, &m) in state.iter_mut().zip(block.iter()) {
            *lane ^= m;
        }
        keccak_f1600(&mut state);
    }
    assert_eq!(vm, state);
}

/// Sets up registers for a raw absorb call without touching memory content.
fn raw_call(state_addr: u64, data_addr: u64, n_blocks: u64) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();
    registers.write(17, KECCAK_ABSORB_SYSCALL_NUMBER).unwrap();
    registers.write(10, state_addr).unwrap();
    registers.write(11, data_addr).unwrap();
    registers.write(12, n_blocks).unwrap();
    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .map(|_| ())
}

#[test]
fn test_absorb_rejects_unaligned_state_addr() {
    let err = raw_call(0x1001, DATA_ADDR, 1).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::UnalignedKeccakAbsorbStateAddress(0x1001)
    ));
}

#[test]
fn test_absorb_rejects_unaligned_data_addr() {
    let err = raw_call(STATE_ADDR, 0x2004, 1).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::UnalignedKeccakAbsorbDataAddress(0x2004)
    ));
}

#[test]
fn test_absorb_rejects_zero_blocks() {
    let err = raw_call(STATE_ADDR, DATA_ADDR, 0).unwrap_err();
    assert!(matches!(err, ExecutionError::KeccakAbsorbZeroBlocks));
}

#[test]
fn test_absorb_rejects_overflowing_state_range() {
    let state_addr = u64::MAX - 191; // 8-aligned; last byte would overflow
    let err = raw_call(state_addr, DATA_ADDR, 1).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::KeccakAbsorbStateAddressOverflow(a) if a == state_addr
    ));
}

#[test]
fn test_absorb_rejects_overflowing_data_range() {
    let data_addr = u64::MAX - 127; // 8-aligned; last byte of one 136-byte block overflows
    let err = raw_call(STATE_ADDR, data_addr, 1).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::KeccakAbsorbDataAddressOverflow(a) if a == data_addr
    ));
}

#[test]
fn test_absorb_rejects_overflowing_block_count() {
    // n_blocks × 136 overflows u64.
    let err = raw_call(STATE_ADDR, DATA_ADDR, u64::MAX / 8).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::KeccakAbsorbDataAddressOverflow(a) if a == DATA_ADDR
    ));
}

#[test]
fn test_absorb_rejects_low_limb_overflow() {
    // Data region crosses the 2^32 low-limb boundary: last byte's low limb wraps.
    let data_addr = (1u64 << 32) - 128; // 8-aligned; block's last byte is past 2^32
    let err = raw_call(STATE_ADDR, data_addr, 1).unwrap_err();
    assert!(matches!(err, ExecutionError::KeccakAbsorbAddressOverflow));
    // A block ending exactly AT the boundary (last byte 2^32 - 1) is accepted.
    raw_call(STATE_ADDR, (1u64 << 32) - 136, 1)
        .expect("block ending at the low-limb boundary must be accepted");
}

#[test]
fn test_absorb_rejects_overlapping_regions() {
    // Data starts inside the 200-byte state region.
    let err = raw_call(STATE_ADDR, STATE_ADDR + 192, 1).unwrap_err();
    assert!(matches!(err, ExecutionError::KeccakAbsorbOperandOverlap));
    // State starts inside the data region.
    let err = raw_call(DATA_ADDR + 128, DATA_ADDR, 1).unwrap_err();
    assert!(matches!(err, ExecutionError::KeccakAbsorbOperandOverlap));
    // Adjacent regions (data immediately after state) are fine.
    raw_call(STATE_ADDR, STATE_ADDR + 200, 1).expect("adjacent regions must be accepted");
}
