//! Tests for the ECSM (elliptic-curve scalar multiplication) syscall.

use crate::vm::instruction::decoding::Instruction;
use crate::vm::instruction::execution::{ECSM_SYSCALL_NUMBER, ExecutionError};
use crate::vm::memory::Memory;
use crate::vm::registers::Registers;

/// secp256k1 generator x-coordinate, little-endian.
fn gx_le() -> [u8; 32] {
    let mut be = [
        0x79, 0xBE, 0x66, 0x7E, 0xF9, 0xDC, 0xBB, 0xAC, 0x55, 0xA0, 0x62, 0x95, 0xCE, 0x87, 0x0B,
        0x07, 0x02, 0x9B, 0xFC, 0xDB, 0x2D, 0xCE, 0x28, 0xD9, 0x59, 0xF2, 0x81, 0x5B, 0x16, 0xF8,
        0x17, 0x98,
    ];
    be.reverse();
    be
}

fn write_u256_le(memory: &mut Memory, addr: u64, bytes: &[u8; 32]) {
    for i in 0..4 {
        let mut dw = [0u8; 8];
        dw.copy_from_slice(&bytes[i * 8..i * 8 + 8]);
        memory
            .store_doubleword(addr + (i as u64) * 8, u64::from_le_bytes(dw))
            .unwrap();
    }
}

fn read_u256_le(memory: &Memory, addr: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    for i in 0..4 {
        let dw = memory.load_doubleword(addr + (i as u64) * 8).unwrap();
        out[i * 8..i * 8 + 8].copy_from_slice(&dw.to_le_bytes());
    }
    out
}

/// Runs the ECSM syscall with the given scalar (as little-endian bytes) and `xG`,
/// returning the `xR` written back to memory.
fn run_ecsm(k_le: &[u8; 32], xg_le: &[u8; 32]) -> Result<[u8; 32], ExecutionError> {
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();

    let addr_xr = 0x1000u64;
    let addr_xg = 0x2000u64;
    let addr_k = 0x3000u64;
    write_u256_le(&mut memory, addr_xg, xg_le);
    write_u256_le(&mut memory, addr_k, k_le);

    registers.write(17, ECSM_SYSCALL_NUMBER).unwrap();
    registers.write(10, addr_xr).unwrap();
    registers.write(11, addr_xg).unwrap();
    registers.write(12, addr_k).unwrap();

    Instruction::EcallEbreak.run(&mut pc, &mut registers, &mut memory)?;
    Ok(read_u256_le(&memory, addr_xr))
}

fn k_le(v: u64) -> [u8; 32] {
    let mut k = [0u8; 32];
    k[..8].copy_from_slice(&v.to_le_bytes());
    k
}

#[test]
fn ecsm_syscall_writes_correct_result() {
    let xg = gx_le();
    // 1·G = G
    assert_eq!(run_ecsm(&k_le(1), &xg).unwrap(), xg);
    // Matches the reference scalar multiplication for several scalars.
    for v in [2u64, 3, 5, 0xFFFF, 1_000_003] {
        assert_eq!(
            run_ecsm(&k_le(v), &xg).unwrap(),
            ecsm::scalar_mul_x(&k_le(v), &xg).unwrap(),
            "k = {v}"
        );
    }
}

#[test]
fn ecsm_syscall_rejects_zero_scalar() {
    let err = run_ecsm(&k_le(0), &gx_le()).unwrap_err();
    assert!(matches!(
        err,
        ExecutionError::Ecsm(ecsm::EcsmError::ScalarIsZero)
    ));
}

#[test]
fn ecsm_syscall_rejects_address_overflow() {
    // addr_k near the lower-limb boundary so (addr mod 2^32) + 31 overflows.
    let mut pc = 0;
    let mut registers = Registers::default();
    let mut memory = Memory::default();
    registers.write(17, ECSM_SYSCALL_NUMBER).unwrap();
    registers.write(10, 0x1000).unwrap();
    registers.write(11, 0x2000).unwrap();
    registers.write(12, 0xFFFF_FFF0).unwrap(); // (mod 2^32) + 31 ≥ 2^32
    let err = Instruction::EcallEbreak
        .run(&mut pc, &mut registers, &mut memory)
        .unwrap_err();
    assert!(matches!(err, ExecutionError::EcsmAddressOverflow));
}
