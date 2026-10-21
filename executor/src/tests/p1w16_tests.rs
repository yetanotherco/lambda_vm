//! Tests for the width-16 Poseidon1 permutation syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ExecutionError, P1W16_SYSCALL_NUMBER};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

const STATE_ADDR: u64 = 0x1000;
const P: u64 = math::field::goldilocks::GOLDILOCKS_PRIME;

fn run_p1w16(memory: &mut Memory, addr: u64) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    registers.write(17, P1W16_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr).unwrap();
    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, memory)
        .map(|_| ())
}

fn store_state(memory: &mut Memory, state: &[u64; 16]) {
    for (i, &w) in state.iter().enumerate() {
        memory
            .store_doubleword(STATE_ADDR + 8 * i as u64, w)
            .unwrap();
    }
}

fn load_state(memory: &Memory) -> [u64; 16] {
    core::array::from_fn(|i| memory.load_doubleword(STATE_ADDR + 8 * i as u64).unwrap())
}

#[test]
fn test_p1w16_syscall_permutes_in_place() {
    let input: [u64; 16] = core::array::from_fn(|i| (i as u64 + 1) * 0x0123_4567_89AB_CDEF % P);
    let mut memory = Memory::default();
    store_state(&mut memory, &input);
    run_p1w16(&mut memory, STATE_ADDR).unwrap();

    use crypto::hash::poseidon1_w16::{Fp, permute};
    let expected = permute(input.map(Fp::from)).map(|v| v.canonical());
    assert_eq!(load_state(&memory), expected);
    assert_ne!(expected, input);
}

#[test]
fn test_p1w16_syscall_accepts_the_largest_canonical_word() {
    let mut input = [0u64; 16];
    input[15] = P - 1;
    let mut memory = Memory::default();
    store_state(&mut memory, &input);
    run_p1w16(&mut memory, STATE_ADDR).unwrap();
}

#[test]
fn test_p1w16_syscall_rejects_non_canonical_input() {
    let mut input = [0u64; 16];
    input[15] = P;
    let mut memory = Memory::default();
    store_state(&mut memory, &input);
    let err = run_p1w16(&mut memory, STATE_ADDR).unwrap_err();
    assert!(matches!(err, ExecutionError::P1w16NonCanonicalInput(v) if v == P));
}

#[test]
fn test_p1w16_syscall_rejects_unaligned_state_addr() {
    let mut memory = Memory::default();
    let err = run_p1w16(&mut memory, 0x1001).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::UnalignedP1w16StateAddress(0x1001)
    ));
}

#[test]
fn test_p1w16_syscall_rejects_overflowing_state_range() {
    let mut memory = Memory::default();
    let addr = u64::MAX - 87;
    let err = run_p1w16(&mut memory, addr).unwrap_err();
    assert!(matches!(err, ExecutionError::P1w16StateAddressOverflow(a) if a == addr));
}
