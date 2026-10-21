//! Tests for the cubic-extension multiply-add syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ExecutionError, FP3MAC_SYSCALL_NUMBER, fp3mac};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

const STATE_ADDR: u64 = 0x1000;
const CELL_ADDR: u64 = 0x2000;
const P: u64 = math::field::goldilocks::GOLDILOCKS_PRIME;

fn run(memory: &mut Memory, state: u64, cell: u64) -> Result<(), ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    registers.write(17, FP3MAC_SYSCALL_NUMBER).unwrap();
    registers.write(10, state).unwrap();
    registers.write(11, cell).unwrap();
    Instruction::EcallEbreak
        .run(&mut pc, &mut registers, memory)
        .map(|_| ())
}

fn store(memory: &mut Memory, addr: u64, words: &[u64]) {
    for (i, &w) in words.iter().enumerate() {
        memory.store_doubleword(addr + 8 * i as u64, w).unwrap();
    }
}

fn load<const N: usize>(memory: &Memory, addr: u64) -> [u64; N] {
    core::array::from_fn(|i| memory.load_doubleword(addr + 8 * i as u64).unwrap())
}

#[test]
fn test_fp3mac_syscall_multiplies_and_adds_in_place() {
    let state: [u64; 6] = core::array::from_fn(|i| (i as u64 + 3) * 0x0123_4567_89AB_CDEF % P);
    let x = [P - 1, 7, 0x8000_0000_0000_0001 % P];
    let mut memory = Memory::default();
    store(&mut memory, STATE_ADDR, &state);
    store(&mut memory, CELL_ADDR, &x);
    run(&mut memory, STATE_ADDR, CELL_ADDR).unwrap();
    let out: [u64; 6] = load(&memory, STATE_ADDR);
    assert_eq!(out[..3], fp3mac(state, x));
    assert_eq!(out[3..], state[3..], "al is left alone");
    assert_eq!(load::<3>(&memory, CELL_ADDR), x, "the cell is left alone");
}

/// `t·t² = t³ = 2`: the extension is `Fp[t]/(t³ − 2)`.
#[test]
fn test_fp3mac_reduces_by_t_cubed_minus_two() {
    assert_eq!(fp3mac([0, 1, 0, 0, 0, 1], [0, 0, 0]), [2, 0, 0]);
    assert_eq!(fp3mac([0, 0, 1, 0, 0, 1], [5, 0, 0]), [5, 2, 0]);
}

#[test]
fn test_fp3mac_syscall_rejects_non_canonical_input() {
    for k in 0..9 {
        let mut words = [1u64; 9];
        words[k] = P;
        let mut memory = Memory::default();
        store(&mut memory, STATE_ADDR, &words[..6]);
        store(&mut memory, CELL_ADDR, &words[6..]);
        let err = run(&mut memory, STATE_ADDR, CELL_ADDR).unwrap_err();
        assert!(
            matches!(err, ExecutionError::Fp3MacNonCanonicalInput(v) if v == P),
            "word {k}"
        );
    }
}

#[test]
fn test_fp3mac_syscall_rejects_unaligned_addresses() {
    let mut memory = Memory::default();
    assert!(matches!(
        run(&mut memory, 0x1001, CELL_ADDR).unwrap_err(),
        ExecutionError::UnalignedFp3MacAddress(0x1001)
    ));
    assert!(matches!(
        run(&mut memory, STATE_ADDR, 0x2004).unwrap_err(),
        ExecutionError::UnalignedFp3MacAddress(0x2004)
    ));
}

#[test]
fn test_fp3mac_syscall_rejects_overflowing_ranges() {
    let mut memory = Memory::default();
    let addr = u64::MAX - 39;
    assert!(matches!(
        run(&mut memory, addr, CELL_ADDR).unwrap_err(),
        ExecutionError::Fp3MacAddressOverflow(a) if a == addr
    ));
    let cell = u64::MAX - 15;
    assert!(matches!(
        run(&mut memory, STATE_ADDR, cell).unwrap_err(),
        ExecutionError::Fp3MacAddressOverflow(a) if a == cell
    ));
}

#[test]
fn test_fp3mac_syscall_rejects_overlapping_operands() {
    let mut memory = Memory::default();
    assert!(matches!(
        run(&mut memory, STATE_ADDR, STATE_ADDR + 40).unwrap_err(),
        ExecutionError::Fp3MacOverlap(..)
    ));
    assert!(matches!(
        run(&mut memory, STATE_ADDR + 16, STATE_ADDR).unwrap_err(),
        ExecutionError::Fp3MacOverlap(..)
    ));
    run(&mut memory, STATE_ADDR, STATE_ADDR + 48).unwrap();
}
