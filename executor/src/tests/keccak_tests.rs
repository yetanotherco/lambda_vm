//! Tests for the Keccak-f[1600] permutation and the Keccak syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ExecutionError, KECCAK_SYSCALL_NUMBER, keccak_f1600};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

#[test]
fn test_keccak_f1600_zero_input() {
    let mut state = [0u64; 25];
    keccak_f1600(&mut state);

    let expected: [u64; 25] = [
        0xF1258F7940E1DDE7,
        0x84D5CCF933C0478A,
        0xD598261EA65AA9EE,
        0xBD1547306F80494D,
        0x8B284E056253D057,
        0xFF97A42D7F8E6FD4,
        0x90FEE5A0A44647C4,
        0x8C5BDA0CD6192E76,
        0xAD30A6F71B19059C,
        0x30935AB7D08FFC64,
        0xEB5AA93F2317D635,
        0xA9A6E6260D712103,
        0x81A57C16DBCF555F,
        0x43B831CD0347C826,
        0x01F22F1A11A5569F,
        0x05E5635A21D9AE61,
        0x64BEFEF28CC970F2,
        0x613670957BC46611,
        0xB87C5A554FD00ECB,
        0x8C3EE88A1CCF32C8,
        0x940C7922AE3A2614,
        0x1841F924A2C509E4,
        0x16F53526E70465C2,
        0x75F644E97F30A13B,
        0xEAF1FF7B5CECA249,
    ];

    assert_eq!(state, expected, "keccak-f[1600] on zero input mismatch");
}

#[test]
fn test_keccak_f1600_nonzero_input() {
    let mut state = [0u64; 25];
    state[0] = 1;
    let original = state;
    keccak_f1600(&mut state);
    assert_ne!(state, original);
    assert!(state.iter().any(|&x| x != 0));
}

#[test]
fn test_keccak_syscall_rejects_unaligned_state_addr() {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    registers.write(17, KECCAK_SYSCALL_NUMBER).unwrap();
    registers.write(10, 0x1001).unwrap();

    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory, 4)
        .unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::UnalignedKeccakStateAddress(0x1001)
    ));
}

#[test]
fn test_keccak_syscall_rejects_overflowing_state_range() {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    registers.write(17, KECCAK_SYSCALL_NUMBER).unwrap();
    registers.write(10, u64::MAX - 191).unwrap();

    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory, 4)
        .unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::KeccakStateAddressOverflow(addr) if addr == u64::MAX - 191
    ));
}
